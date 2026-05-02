"""
Tests for ndna.storage module.

Tests save/load roundtrip for all result types.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from ndna.core.results import (
    SpectralResult,
    AggregatedSpectralResult,
    ThermoResult,
    ThermoResultPerSample,
    BeliefResult,
    BeliefResultPerSample,
    nDNAResult,
    nDNAResultMultiConcept,
    FullResults,
)
from ndna.storage import ResultSaver, ResultLoader


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_spectral_result():
    """Create a sample SpectralResult for testing."""
    return SpectralResult(
        curvatures=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        speeds=np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10]),
        layer_indices=np.array([1, 2, 3, 4, 5]),
        text_preview="This is a test text for spectral curvature.",
        label="test_prompt",
    )


@pytest.fixture
def sample_thermo_result():
    """Create a sample ThermoResult for testing."""
    return ThermoResult(
        step_lengths=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
        step_indices=np.array([1, 2, 3, 4, 5]),
        total_length=0.15,
        num_samples_processed=100,
        num_tokens_processed=5000,
    )


@pytest.fixture
def sample_belief_result():
    """Create a sample BeliefResult for testing."""
    return BeliefResult(
        belief_norms=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        layer_indices=np.array([1, 2, 3, 4, 5, 6]),
        num_samples_processed=100,
        num_tokens_processed=5000,
        fr_norm=True,
    )


@pytest.fixture
def sample_full_results(sample_spectral_result, sample_thermo_result, sample_belief_result):
    """Create sample FullResults for testing."""
    return FullResults(
        model_name="test-model",
        dataset_name="test-dataset",
        spectral=sample_spectral_result,
        thermodynamic=sample_thermo_result,
        belief=sample_belief_result,
        config={"max_samples": 100, "batch_size": 2},
    )


# -----------------------------------------------------------------------------
# ResultSaver Tests
# -----------------------------------------------------------------------------

class TestResultSaver:
    """Tests for ResultSaver class."""

    def test_save_spectral_result(self, temp_dir, sample_spectral_result):
        """Test saving a single SpectralResult."""
        saver = ResultSaver()
        path = os.path.join(temp_dir, "spectral.npz")
        
        saver.save_spectral(sample_spectral_result, path)
        
        assert os.path.exists(path)
        
        # Verify contents
        data = np.load(path, allow_pickle=True)
        assert "curvatures" in data
        assert "speeds" in data
        assert "layer_indices" in data
        np.testing.assert_array_almost_equal(data["curvatures"], sample_spectral_result.curvatures)

    def test_save_aggregated_spectral_result(self, temp_dir, sample_spectral_result):
        """Test saving an AggregatedSpectralResult."""
        saver = ResultSaver()
        
        # Create aggregated result
        results = [sample_spectral_result] * 3
        aggregated = AggregatedSpectralResult.from_results(results)
        
        path = os.path.join(temp_dir, "spectral_agg.npz")
        saver.save_spectral(aggregated, path)
        
        assert os.path.exists(path)
        
        data = np.load(path, allow_pickle=True)
        assert bool(data["is_aggregated"][0]) is True
        assert "mean_curvatures" in data
        assert "std_curvatures" in data

    def test_save_thermodynamic_result(self, temp_dir, sample_thermo_result):
        """Test saving a ThermoResult."""
        saver = ResultSaver()
        path = os.path.join(temp_dir, "thermo.npz")
        
        saver.save_thermodynamic(sample_thermo_result, path)
        
        assert os.path.exists(path)
        
        data = np.load(path, allow_pickle=True)
        assert "step_lengths" in data
        assert "total_length" in data
        np.testing.assert_array_almost_equal(data["step_lengths"], sample_thermo_result.step_lengths)

    def test_save_belief_result(self, temp_dir, sample_belief_result):
        """Test saving a BeliefResult."""
        saver = ResultSaver()
        path = os.path.join(temp_dir, "belief.npz")
        
        saver.save_belief(sample_belief_result, path)
        
        assert os.path.exists(path)
        
        data = np.load(path, allow_pickle=True)
        assert "belief_norms" in data
        np.testing.assert_array_almost_equal(data["belief_norms"], sample_belief_result.belief_norms)

    def test_save_full_results(self, temp_dir, sample_full_results):
        """Test saving FullResults creates correct directory structure."""
        saver = ResultSaver()
        output_dir = os.path.join(temp_dir, "full_results")
        
        saver.save(sample_full_results, output_dir)
        
        # Check files exist
        assert os.path.exists(os.path.join(output_dir, "spectral_curvature.npz"))
        assert os.path.exists(os.path.join(output_dir, "thermodynamic_length.npz"))
        assert os.path.exists(os.path.join(output_dir, "belief_vector_field.npz"))
        assert os.path.exists(os.path.join(output_dir, "metadata.json"))

    def test_save_metadata(self, temp_dir, sample_full_results):
        """Test metadata is saved correctly."""
        saver = ResultSaver()
        output_dir = os.path.join(temp_dir, "results")
        
        saver.save(sample_full_results, output_dir)
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        
        assert metadata["model_name"] == "test-model"
        assert metadata["dataset_name"] == "test-dataset"
        assert metadata["metrics_computed"]["spectral"] is True
        assert metadata["metrics_computed"]["thermodynamic"] is True
        assert metadata["metrics_computed"]["belief"] is True

    def test_save_ndna_result(self, temp_dir):
        """Test saving nDNAResult."""
        saver = ResultSaver()
        
        result = nDNAResult(
            scalar=0.5,
            layerwise=np.array([0.0, 0.0, 0.1, 0.2, 0.2]),
            layer_indices=np.array([1, 2, 3, 4, 5]),
            concept_name="test_concept",
            l_min=2,
        )
        
        path = os.path.join(temp_dir, "ndna.npz")
        saver.save_ndna(result, path)
        
        assert os.path.exists(path)
        
        data = np.load(path, allow_pickle=True)
        assert float(data["scalar"][0]) == pytest.approx(0.5)
        assert bool(data["is_multi_concept"][0]) is False

    def test_save_ndna_multi_concept(self, temp_dir):
        """Test saving nDNAResultMultiConcept."""
        saver = ResultSaver()
        
        result = nDNAResultMultiConcept(
            scalars={"a": 0.5, "b": 0.8},
            layerwise={
                "a": np.array([0.0, 0.2, 0.3]),
                "b": np.array([0.0, 0.3, 0.5]),
            },
            layer_indices=np.array([1, 2, 3]),
        )
        
        path = os.path.join(temp_dir, "ndna_multi.npz")
        saver.save_ndna(result, path)
        
        assert os.path.exists(path)
        
        data = np.load(path, allow_pickle=True)
        assert bool(data["is_multi_concept"][0]) is True
        assert len(data["concepts"]) == 2


# -----------------------------------------------------------------------------
# ResultLoader Tests
# -----------------------------------------------------------------------------

class TestResultLoader:
    """Tests for ResultLoader class."""

    def test_load_spectral_result(self, temp_dir, sample_spectral_result):
        """Test loading a SpectralResult."""
        saver = ResultSaver()
        loader = ResultLoader()
        
        path = os.path.join(temp_dir, "spectral.npz")
        saver.save_spectral(sample_spectral_result, path)
        
        loaded = loader.load_spectral(path)
        
        assert isinstance(loaded, SpectralResult)
        np.testing.assert_array_almost_equal(loaded.curvatures, sample_spectral_result.curvatures)
        np.testing.assert_array_almost_equal(loaded.speeds, sample_spectral_result.speeds)
        assert loaded.label == sample_spectral_result.label

    def test_load_thermodynamic_result(self, temp_dir, sample_thermo_result):
        """Test loading a ThermoResult."""
        saver = ResultSaver()
        loader = ResultLoader()
        
        path = os.path.join(temp_dir, "thermo.npz")
        saver.save_thermodynamic(sample_thermo_result, path)
        
        loaded = loader.load_thermodynamic(path)
        
        assert isinstance(loaded, ThermoResult)
        np.testing.assert_array_almost_equal(loaded.step_lengths, sample_thermo_result.step_lengths)
        assert loaded.total_length == pytest.approx(sample_thermo_result.total_length)

    def test_load_belief_result(self, temp_dir, sample_belief_result):
        """Test loading a BeliefResult."""
        saver = ResultSaver()
        loader = ResultLoader()
        
        path = os.path.join(temp_dir, "belief.npz")
        saver.save_belief(sample_belief_result, path)
        
        loaded = loader.load_belief(path)
        
        assert isinstance(loaded, BeliefResult)
        np.testing.assert_array_almost_equal(loaded.belief_norms, sample_belief_result.belief_norms)
        assert loaded.fr_norm == sample_belief_result.fr_norm

    def test_load_full_results(self, temp_dir, sample_full_results):
        """Test loading FullResults."""
        saver = ResultSaver()
        loader = ResultLoader()
        
        output_dir = os.path.join(temp_dir, "full")
        saver.save(sample_full_results, output_dir)
        
        loaded = loader.load(output_dir)
        
        assert isinstance(loaded, FullResults)
        assert loaded.model_name == sample_full_results.model_name
        assert loaded.dataset_name == sample_full_results.dataset_name
        assert loaded.spectral is not None
        assert loaded.thermodynamic is not None
        assert loaded.belief is not None

    def test_load_ndna_result(self, temp_dir):
        """Test loading nDNAResult."""
        saver = ResultSaver()
        loader = ResultLoader()
        
        original = nDNAResult(
            scalar=0.5,
            layerwise=np.array([0.0, 0.0, 0.1, 0.2, 0.2]),
            layer_indices=np.array([1, 2, 3, 4, 5]),
            concept_name="test",
            l_min=2,
        )
        
        path = os.path.join(temp_dir, "ndna.npz")
        saver.save_ndna(original, path)
        
        loaded = loader.load_ndna(path)
        
        assert isinstance(loaded, nDNAResult)
        assert loaded.scalar == pytest.approx(original.scalar)
        assert loaded.concept_name == original.concept_name

    def test_load_ndna_multi_concept(self, temp_dir):
        """Test loading nDNAResultMultiConcept."""
        saver = ResultSaver()
        loader = ResultLoader()
        
        original = nDNAResultMultiConcept(
            scalars={"a": 0.5, "b": 0.8},
            layerwise={
                "a": np.array([0.0, 0.2, 0.3]),
                "b": np.array([0.0, 0.3, 0.5]),
            },
            layer_indices=np.array([1, 2, 3]),
        )
        
        path = os.path.join(temp_dir, "ndna_multi.npz")
        saver.save_ndna(original, path)
        
        loaded = loader.load_ndna(path)
        
        assert isinstance(loaded, nDNAResultMultiConcept)
        assert loaded.scalars["a"] == pytest.approx(0.5)
        assert loaded.scalars["b"] == pytest.approx(0.8)

    def test_load_metadata(self, temp_dir, sample_full_results):
        """Test loading only metadata."""
        saver = ResultSaver()
        loader = ResultLoader()
        
        output_dir = os.path.join(temp_dir, "results")
        saver.save(sample_full_results, output_dir)
        
        metadata = loader.load_metadata(output_dir)
        
        assert metadata["model_name"] == "test-model"
        assert "timestamp" in metadata

    def test_load_nonexistent_directory(self, temp_dir):
        """Test loading from nonexistent directory raises error."""
        loader = ResultLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load(os.path.join(temp_dir, "nonexistent"))


# -----------------------------------------------------------------------------
# Roundtrip Tests
# -----------------------------------------------------------------------------

class TestRoundtrip:
    """Tests for save/load roundtrip consistency."""

    def test_spectral_roundtrip(self, temp_dir, sample_spectral_result):
        """Test spectral result roundtrip preserves data."""
        saver = ResultSaver()
        loader = ResultLoader()
        
        path = os.path.join(temp_dir, "spectral.npz")
        saver.save_spectral(sample_spectral_result, path)
        loaded = loader.load_spectral(path)
        
        np.testing.assert_array_almost_equal(
            loaded.curvatures, 
            sample_spectral_result.curvatures
        )
        assert loaded.mean_curvature == pytest.approx(sample_spectral_result.mean_curvature)

    def test_thermo_roundtrip(self, temp_dir, sample_thermo_result):
        """Test thermodynamic result roundtrip preserves data."""
        saver = ResultSaver()
        loader = ResultLoader()
        
        path = os.path.join(temp_dir, "thermo.npz")
        saver.save_thermodynamic(sample_thermo_result, path)
        loaded = loader.load_thermodynamic(path)
        
        np.testing.assert_array_almost_equal(
            loaded.step_lengths,
            sample_thermo_result.step_lengths
        )
        assert loaded.num_samples_processed == sample_thermo_result.num_samples_processed

    def test_per_sample_roundtrip(self, temp_dir):
        """Test per-sample results roundtrip."""
        saver = ResultSaver()
        loader = ResultLoader()
        
        # Create per-sample data
        thermo_ps = ThermoResultPerSample(
            per_sample_lengths=np.random.rand(50, 10),
            mean_lengths=np.random.rand(10),
            step_indices=np.arange(1, 11),
            num_samples=50,
        )
        
        belief_ps = BeliefResultPerSample(
            per_sample_norms=np.random.rand(50, 12),
            mean_norms=np.random.rand(12),
            layer_indices=np.arange(1, 13),
            num_samples=50,
        )
        
        # Save
        output_dir = os.path.join(temp_dir, "per_sample")
        os.makedirs(output_dir)
        
        saver.save_thermo_per_sample(thermo_ps, os.path.join(output_dir, "thermo.npz"))
        saver.save_belief_per_sample(belief_ps, os.path.join(output_dir, "belief.npz"))
        
        # Load
        loaded_thermo = loader.load_thermo_per_sample(os.path.join(output_dir, "thermo.npz"))
        loaded_belief = loader.load_belief_per_sample(os.path.join(output_dir, "belief.npz"))
        
        assert loaded_thermo.num_samples == 50
        assert loaded_belief.num_samples == 50
        np.testing.assert_array_almost_equal(
            loaded_thermo.per_sample_lengths,
            thermo_ps.per_sample_lengths
        )

