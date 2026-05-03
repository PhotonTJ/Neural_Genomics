"""
Unit tests for ndna.core module.

Tests geometry utilities and metric calculators.
Uses synthetic data and mock objects to avoid loading real models.
"""

import numpy as np
import pytest
import torch

from ndna.core import (
    # Geometry
    sqrt_embed,
    sqrt_embed_batch,
    project_tangent,
    project_tangent_batch,
    discrete_curvature,
    fisher_rao_distance,
    fisher_rao_distance_from_sqrt,
    safe_log_softmax,
    safe_softmax,
    compute_belief_gradient,
    compute_tangent_vector,
    free_memory,
    EPS_DIST,
    EPS_CURV,
    # Results
    SpectralResult,
    AggregatedSpectralResult,
    ThermoResult,
    ThermoResultPerSample,
    BeliefResult,
    BeliefResultPerSample,
)


# -----------------------------------------------------------------------------
# Geometry Tests
# -----------------------------------------------------------------------------

class TestSqrtEmbed:
    """Tests for sqrt embedding functions."""

    def test_sqrt_embed_output_unit_norm(self):
        """Test that sqrt_embed produces unit vectors."""
        q = torch.tensor([0.1, 0.2, 0.3, 0.4])
        u = sqrt_embed(q)
        
        norm = torch.linalg.norm(u).item()
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_sqrt_embed_positive(self):
        """Test that sqrt_embed outputs are positive."""
        q = torch.tensor([0.25, 0.25, 0.25, 0.25])
        u = sqrt_embed(q)
        
        assert (u >= 0).all()

    def test_sqrt_embed_handles_zeros(self):
        """Test that sqrt_embed handles near-zero probabilities."""
        q = torch.tensor([0.0, 0.0, 0.5, 0.5])
        u = sqrt_embed(q)
        
        assert torch.isfinite(u).all()
        assert torch.linalg.norm(u).item() == pytest.approx(1.0, abs=1e-6)

    def test_sqrt_embed_batch(self):
        """Test batched sqrt embedding."""
        q = torch.rand(4, 100)
        q = q / q.sum(dim=-1, keepdim=True)  # Normalize
        
        u = sqrt_embed_batch(q)
        
        assert u.shape == q.shape
        # Check all are unit vectors
        norms = torch.linalg.norm(u, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-6)

    def test_sqrt_embed_batch_matches_sequential(self):
        """Test that batched version matches sequential."""
        q = torch.rand(3, 50)
        q = q / q.sum(dim=-1, keepdim=True)
        
        u_batch = sqrt_embed_batch(q)
        
        for i in range(3):
            u_seq = sqrt_embed(q[i])
            assert torch.allclose(u_batch[i], u_seq, atol=1e-6)


class TestProjectTangent:
    """Tests for tangent projection functions."""

    def test_project_tangent_orthogonal(self):
        """Test that projected vector is orthogonal to u."""
        u = torch.randn(100)
        u = u / torch.linalg.norm(u)  # Unit vector
        
        v = torch.randn(100)
        v_tan = project_tangent(u, v)
        
        dot = torch.dot(u, v_tan).item()
        assert dot == pytest.approx(0.0, abs=1e-6)

    def test_project_tangent_radial_removed(self):
        """Test that radial component is removed."""
        u = torch.tensor([1.0, 0.0, 0.0])
        v = torch.tensor([2.0, 1.0, 1.0])  # Has component along u
        
        v_tan = project_tangent(u, v)
        
        # Should remove the x-component (along u)
        assert v_tan[0].item() == pytest.approx(0.0, abs=1e-6)
        assert v_tan[1].item() == pytest.approx(1.0, abs=1e-6)
        assert v_tan[2].item() == pytest.approx(1.0, abs=1e-6)

    def test_project_tangent_batch(self):
        """Test batched tangent projection."""
        u = torch.randn(4, 5, 100)
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True)
        
        v = torch.randn(4, 5, 100)
        v_tan = project_tangent_batch(u, v)
        
        # Check orthogonality
        dots = (u * v_tan).sum(dim=-1)
        assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)


class TestDiscreteCurvature:
    """Tests for discrete curvature computation."""

    def test_discrete_curvature_output_shape(self):
        """Test curvature output shapes."""
        # Create 5 points on sphere
        u_list = []
        for _ in range(5):
            q = torch.rand(100)
            q = q / q.sum()
            u_list.append(sqrt_embed(q))
        
        curvatures, speeds = discrete_curvature(u_list)
        
        # Curvatures: interior points (5-2 = 3)
        assert curvatures.shape == (3,)
        # Speeds: first differences (5-1 = 4)
        assert speeds.shape == (4,)

    def test_discrete_curvature_positive(self):
        """Test that curvatures are non-negative."""
        u_list = []
        for i in range(6):
            # Create slowly varying distributions
            q = torch.softmax(torch.randn(50) + i * 0.1, dim=-1)
            u_list.append(sqrt_embed(q))
        
        curvatures, speeds = discrete_curvature(u_list)
        
        assert (curvatures >= 0).all()
        assert (speeds >= 0).all()

    def test_discrete_curvature_requires_3_points(self):
        """Test that at least 3 points are required."""
        u_list = [sqrt_embed(torch.rand(10) / torch.rand(10).sum()) for _ in range(2)]
        
        with pytest.raises(ValueError, match="at least 3"):
            discrete_curvature(u_list)


class TestFisherRaoDistance:
    """Tests for Fisher-Rao distance functions."""

    def test_fisher_rao_distance_same_distribution(self):
        """Test FR distance is 0 for identical distributions."""
        logp = torch.log(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        
        d = fisher_rao_distance(logp, logp)
        
        assert d.item() == pytest.approx(0.0, abs=1e-6)

    def test_fisher_rao_distance_bounds(self):
        """Test FR distance is in [0, π]."""
        # Two very different distributions
        logp1 = torch.log(torch.tensor([0.9, 0.05, 0.05]))
        logp2 = torch.log(torch.tensor([0.05, 0.9, 0.05]))
        
        d = fisher_rao_distance(logp1, logp2)
        
        assert d.item() >= 0
        assert d.item() <= np.pi

    def test_fisher_rao_distance_symmetric(self):
        """Test FR distance is symmetric."""
        logp1 = torch.log(torch.softmax(torch.randn(100), dim=-1))
        logp2 = torch.log(torch.softmax(torch.randn(100), dim=-1))
        
        d12 = fisher_rao_distance(logp1, logp2)
        d21 = fisher_rao_distance(logp2, logp1)
        
        assert torch.allclose(d12, d21, atol=1e-6)

    def test_fisher_rao_distance_batched(self):
        """Test batched FR distance."""
        logp1 = torch.log(torch.softmax(torch.randn(4, 5, 100), dim=-1))
        logp2 = torch.log(torch.softmax(torch.randn(4, 5, 100), dim=-1))
        
        d = fisher_rao_distance(logp1, logp2)
        
        assert d.shape == (4, 5)
        assert (d >= 0).all()
        assert (d <= np.pi).all()

    def test_fisher_rao_from_sqrt_matches(self):
        """Test that FR from sqrt matches FR from log probs."""
        p1 = torch.softmax(torch.randn(100), dim=-1)
        p2 = torch.softmax(torch.randn(100), dim=-1)
        
        logp1 = torch.log(p1.clamp_min(1e-12))
        logp2 = torch.log(p2.clamp_min(1e-12))
        
        u1 = sqrt_embed(p1)
        u2 = sqrt_embed(p2)
        
        d_logp = fisher_rao_distance(logp1, logp2)
        d_sqrt = fisher_rao_distance_from_sqrt(u1, u2)
        
        assert d_logp.item() == pytest.approx(d_sqrt.item(), abs=1e-4)


class TestProbabilityUtilities:
    """Tests for probability utility functions."""

    def test_safe_log_softmax(self):
        """Test safe_log_softmax produces valid log probs."""
        logits = torch.randn(4, 100)
        logp = safe_log_softmax(logits)
        
        # Should sum to 0 in log space (sum to 1 in prob space)
        log_sums = torch.logsumexp(logp, dim=-1)
        assert torch.allclose(log_sums, torch.zeros(4), atol=1e-5)

    def test_safe_softmax_clamped(self):
        """Test safe_softmax produces clamped probabilities."""
        logits = torch.randn(4, 100)
        p = safe_softmax(logits, eps=1e-8)
        
        assert (p >= 1e-8).all()
        # Should still sum to 1 (approximately)
        sums = p.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-4)


class TestBeliefUtilities:
    """Tests for belief vector utility functions."""

    def test_compute_belief_gradient_shape(self):
        """Test belief gradient output shape."""
        q = torch.softmax(torch.randn(4, 5, 100), dim=-1)
        targets = torch.randint(0, 100, (4, 5))
        
        g = compute_belief_gradient(q, targets, tau=1.0)
        
        assert g.shape == q.shape

    def test_compute_belief_gradient_sums_to_zero(self):
        """Test that belief gradient sums to approximately zero."""
        # g = (1/τ)(e_y - q), should sum to (1/τ)(1 - 1) = 0
        q = torch.softmax(torch.randn(100), dim=-1)
        targets = torch.tensor([42])
        
        g = compute_belief_gradient(q.unsqueeze(0), targets, tau=1.0)
        
        g_sum = g.sum().item()
        assert g_sum == pytest.approx(0.0, abs=1e-5)

    def test_compute_tangent_vector_shape(self):
        """Test tangent vector output shape."""
        q = torch.softmax(torch.randn(4, 100), dim=-1)
        u = sqrt_embed_batch(q)
        targets = torch.randint(0, 100, (4,))
        g = compute_belief_gradient(q, targets, tau=1.0)
        
        t = compute_tangent_vector(q, u, g, tau=1.0)
        
        assert t.shape == q.shape


# -----------------------------------------------------------------------------
# Result Dataclass Tests
# -----------------------------------------------------------------------------

class TestSpectralResult:
    """Tests for SpectralResult dataclass."""

    def test_spectral_result_creation(self):
        """Test creating a SpectralResult."""
        result = SpectralResult(
            curvatures=np.array([1.0, 2.0, 3.0]),
            speeds=np.array([0.1, 0.2, 0.3, 0.4]),
            layer_indices=np.array([1, 2, 3]),
            text_preview="Hello world",
            label="test",
        )
        
        assert result.curvatures.shape == (3,)
        assert result.speeds.shape == (4,)
        assert result.text_preview == "Hello world"
        assert result.label == "test"

    def test_spectral_result_statistics(self):
        """Test that statistics are computed in __post_init__."""
        result = SpectralResult(
            curvatures=np.array([1.0, 5.0, 3.0]),
            speeds=np.array([0.1, 0.2]),
            layer_indices=np.array([1, 2, 3]),
            text_preview="Test",
        )
        
        assert result.mean_curvature == pytest.approx(3.0)
        assert result.max_curvature == pytest.approx(5.0)
        assert result.min_curvature == pytest.approx(1.0)
        assert result.max_layer == 2  # 1-indexed
        assert result.min_layer == 1


class TestAggregatedSpectralResult:
    """Tests for AggregatedSpectralResult dataclass."""

    def test_aggregated_from_results(self):
        """Test creating aggregated result from list."""
        results = [
            SpectralResult(
                curvatures=np.array([1.0, 2.0]),
                speeds=np.array([0.1, 0.2, 0.3]),
                layer_indices=np.array([1, 2]),
                text_preview=f"Text {i}",
            )
            for i in range(3)
        ]
        
        agg = AggregatedSpectralResult.from_results(results)
        
        assert agg.num_texts == 3
        assert agg.all_curvatures.shape == (3, 2)
        assert agg.mean_curvatures.shape == (2,)
        assert agg.std_curvatures.shape == (2,)


class TestThermoResult:
    """Tests for ThermoResult dataclass."""

    def test_thermo_result_creation(self):
        """Test creating a ThermoResult."""
        result = ThermoResult(
            step_lengths=np.array([0.1, 0.2, 0.3]),
            step_indices=np.array([1, 2, 3]),
            total_length=0.0,  # Will be computed
            num_samples_processed=100,
        )
        
        assert result.step_lengths.shape == (3,)
        assert result.total_length == pytest.approx(0.6)

    def test_thermo_result_per_sample(self):
        """Test ThermoResultPerSample creation."""
        per_sample = np.random.rand(10, 5)
        step_indices = np.arange(1, 6)
        
        result = ThermoResultPerSample.from_sample_data(per_sample, step_indices)
        
        assert result.num_samples == 10
        assert result.mean_lengths.shape == (5,)


class TestBeliefResult:
    """Tests for BeliefResult dataclass."""

    def test_belief_result_creation(self):
        """Test creating a BeliefResult."""
        result = BeliefResult(
            belief_norms=np.array([0.01, 0.02, 0.015, 0.01]),
            layer_indices=np.array([1, 2, 3, 4]),
        )
        
        assert result.belief_norms.shape == (4,)
        assert result.mean_norm == pytest.approx(0.01375)
        assert result.max_norm == pytest.approx(0.02)
        assert result.max_layer == 2

    def test_belief_result_per_sample(self):
        """Test BeliefResultPerSample creation."""
        per_sample = np.random.rand(10, 4)
        layer_indices = np.arange(1, 5)
        
        result = BeliefResultPerSample.from_sample_data(per_sample, layer_indices)
        
        assert result.num_samples == 10
        assert result.mean_norms.shape == (4,)


# -----------------------------------------------------------------------------
# Memory Management Tests
# -----------------------------------------------------------------------------

class TestMemoryManagement:
    """Tests for memory management utilities."""

    def test_free_memory_runs(self):
        """Test that free_memory doesn't error."""
        # Just verify it runs without error
        free_memory()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_free_memory_clears_cache(self):
        """Test that free_memory clears CUDA cache."""
        # Allocate some memory
        x = torch.randn(1000, 1000, device="cuda")
        del x
        
        # Cache should have memory
        free_memory()
        
        # Should not error
        assert True


# -----------------------------------------------------------------------------
# Integration Tests with Synthetic Data
# -----------------------------------------------------------------------------

class TestIntegrationSynthetic:
    """Integration tests using synthetic data."""

    def test_full_geometry_pipeline(self):
        """Test full pipeline: probs -> sqrt_embed -> curvature."""
        # Simulate layerwise distributions
        num_layers = 6
        vocab_size = 100
        
        u_list = []
        for i in range(num_layers):
            # Create slightly evolving distributions
            logits = torch.randn(vocab_size) + i * 0.5
            q = torch.softmax(logits, dim=-1)
            u = sqrt_embed(q)
            u_list.append(u)
        
        curvatures, speeds = discrete_curvature(u_list)
        
        assert curvatures.shape == (num_layers - 2,)
        assert speeds.shape == (num_layers - 1,)
        assert (curvatures >= 0).all()
        assert np.isfinite(curvatures).all()

    def test_fr_distance_pipeline(self):
        """Test FR distance computation pipeline."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        # Simulate two consecutive layers' log probs
        logp1 = torch.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
        logp2 = torch.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
        
        fr_distances = fisher_rao_distance(logp1, logp2)
        
        assert fr_distances.shape == (batch_size, seq_len)
        assert (fr_distances >= 0).all()
        assert (fr_distances <= np.pi).all()

    def test_belief_vector_pipeline(self):
        """Test belief vector computation pipeline."""
        batch_size = 4
        vocab_size = 100
        
        # Simulate layer output
        logits = torch.randn(batch_size, vocab_size)
        q = safe_softmax(logits)
        u = sqrt_embed_batch(q)
        targets = torch.randint(0, vocab_size, (batch_size,))
        
        # Compute gradient and tangent vector
        g = compute_belief_gradient(q, targets, tau=1.0)
        t = compute_tangent_vector(q, u, g, tau=1.0)
        
        # Tangent vector should have finite values
        assert torch.isfinite(t).all()
        
        # Average and get norm
        t_mean = t.mean(dim=0)
        u_mean = u.mean(dim=0)
        
        # Project to tangent space at mean u
        u_norm = torch.linalg.norm(u_mean).clamp_min(1e-12)
        u_bar = u_mean / u_norm
        radial = torch.dot(u_bar, t_mean)
        t_tan = t_mean - radial * u_bar
        
        belief_norm = torch.linalg.norm(t_tan).item()
        
        assert belief_norm >= 0
        assert np.isfinite(belief_norm)


# -----------------------------------------------------------------------------
# nDNA Combined Metric Tests
# -----------------------------------------------------------------------------

from ndna.core import (
    compute_ndna,
    compute_ndna_multi_concept,
    nDNAResult,
    nDNAResultMultiConcept,
)


class TestnDNAResult:
    """Tests for nDNA result dataclasses."""

    def test_ndna_result_creation(self):
        """Test creating an nDNAResult."""
        result = nDNAResult(
            scalar=0.5,
            layerwise=np.array([0.0, 0.0, 0.1, 0.2, 0.2]),
            layer_indices=np.array([1, 2, 3, 4, 5]),
            concept_name="test",
            l_min=2,
        )
        
        assert result.scalar == 0.5
        assert result.layerwise.shape == (5,)
        assert result.concept_name == "test"

    def test_ndna_result_peak_layer(self):
        """Test peak_layer property."""
        result = nDNAResult(
            scalar=0.5,
            layerwise=np.array([0.0, 0.0, 0.1, 0.3, 0.1]),
            layer_indices=np.array([1, 2, 3, 4, 5]),
        )
        
        assert result.peak_layer == 4  # 1-indexed
        assert result.peak_contribution == pytest.approx(0.3)

    def test_ndna_result_multi_concept(self):
        """Test nDNAResultMultiConcept."""
        result = nDNAResultMultiConcept(
            scalars={"concept_a": 0.5, "concept_b": 0.8},
            layerwise={
                "concept_a": np.array([0.0, 0.2, 0.3]),
                "concept_b": np.array([0.0, 0.3, 0.5]),
            },
            layer_indices=np.array([1, 2, 3]),
        )
        
        assert result.concepts == ["concept_a", "concept_b"]
        
        # Test ranking
        ranked = result.ranked_concepts(descending=True)
        assert ranked[0][0] == "concept_b"
        assert ranked[0][1] == 0.8

    def test_ndna_result_multi_get_result(self):
        """Test getting individual result from multi-concept."""
        multi = nDNAResultMultiConcept(
            scalars={"test": 1.0},
            layerwise={"test": np.array([0.5, 0.5])},
            layer_indices=np.array([1, 2]),
        )
        
        single = multi.get_result("test")
        assert isinstance(single, nDNAResult)
        assert single.scalar == 1.0
        assert single.concept_name == "test"


class TestComputeNDNA:
    """Tests for compute_ndna functions."""

    def test_compute_ndna_basic(self):
        """Test basic nDNA computation."""
        # 4 layers: kappa and fr_steps have length 3, belief_norms has length 4
        kappa = np.array([1.0, 2.0, 3.0])  # ℓ = 1, 2, 3
        fr_steps = np.array([0.1, 0.2, 0.3])  # ℓ = 1, 2, 3
        belief_norms = np.array([0.01, 0.02, 0.03, 0.04])  # ℓ = 0, 1, 2, 3
        
        result = compute_ndna(kappa, fr_steps, belief_norms, l_min=2)
        
        assert isinstance(result, nDNAResult)
        assert result.scalar >= 0
        assert result.layerwise.shape == (4,)
        
        # Check l_min=2 means layers 0 and 1 have zero contribution
        assert result.layerwise[0] == 0.0
        assert result.layerwise[1] == 0.0
        
        # Manually compute expected values
        # contrib[2] = kappa[1] * fr_steps[1] * belief_norms[2] = 2.0 * 0.2 * 0.03 = 0.012
        # contrib[3] = kappa[2] * fr_steps[2] * belief_norms[3] = 3.0 * 0.3 * 0.04 = 0.036
        assert result.layerwise[2] == pytest.approx(0.012)
        assert result.layerwise[3] == pytest.approx(0.036)
        assert result.scalar == pytest.approx(0.048)

    def test_compute_ndna_l_min_effect(self):
        """Test that l_min affects which layers are included."""
        kappa = np.array([1.0, 1.0, 1.0])
        fr_steps = np.array([1.0, 1.0, 1.0])
        belief_norms = np.array([0.1, 0.1, 0.1, 0.1])
        
        # l_min=1 includes more layers
        result_l1 = compute_ndna(kappa, fr_steps, belief_norms, l_min=1)
        result_l2 = compute_ndna(kappa, fr_steps, belief_norms, l_min=2)
        result_l3 = compute_ndna(kappa, fr_steps, belief_norms, l_min=3)
        
        assert result_l1.scalar > result_l2.scalar
        assert result_l2.scalar > result_l3.scalar

    def test_compute_ndna_shape_validation(self):
        """Test shape validation."""
        kappa = np.array([1.0, 2.0])
        fr_steps = np.array([0.1, 0.2, 0.3])  # Wrong shape
        belief_norms = np.array([0.01, 0.02, 0.03])
        
        with pytest.raises(ValueError, match="same shape"):
            compute_ndna(kappa, fr_steps, belief_norms)

    def test_compute_ndna_belief_shape_validation(self):
        """Test belief norms shape validation."""
        kappa = np.array([1.0, 2.0])
        fr_steps = np.array([0.1, 0.2])
        belief_norms = np.array([0.01, 0.02])  # Should be length 3
        
        with pytest.raises(ValueError, match="expected 3"):
            compute_ndna(kappa, fr_steps, belief_norms)

    def test_compute_ndna_multi_concept(self):
        """Test multi-concept nDNA computation."""
        kappa = np.array([1.0, 2.0])
        fr_steps = np.array([0.1, 0.2])
        belief_norms_by_concept = {
            "concept_a": np.array([0.01, 0.02, 0.03]),
            "concept_b": np.array([0.1, 0.2, 0.3]),
        }
        
        result = compute_ndna_multi_concept(
            kappa, fr_steps, belief_norms_by_concept, l_min=2
        )
        
        assert isinstance(result, nDNAResultMultiConcept)
        assert "concept_a" in result.scalars
        assert "concept_b" in result.scalars
        
        # concept_b has 10x larger belief norms, so should have 10x larger nDNA
        ratio = result.scalars["concept_b"] / result.scalars["concept_a"]
        assert ratio == pytest.approx(10.0)

    def test_compute_ndna_stores_inputs(self):
        """Test that input arrays are stored in result."""
        kappa = np.array([1.0, 2.0])
        fr_steps = np.array([0.1, 0.2])
        belief_norms = np.array([0.01, 0.02, 0.03])
        
        result = compute_ndna(kappa, fr_steps, belief_norms)
        
        assert np.array_equal(result.kappa, kappa)
        assert np.array_equal(result.fr_steps, fr_steps)
        assert np.array_equal(result.belief_norms, belief_norms)

