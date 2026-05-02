"""
Tests for ndna.visualization module.

Tests plot generation with synthetic data.
"""

import os
import tempfile
from pathlib import Path

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
)

# Import visualization components
from ndna.visualization.styles import (
    apply_style,
    get_color_palette,
    COLORS,
    FIGSIZE_SINGLE,
    format_metric_title,
)


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
    """Create a sample SpectralResult."""
    return SpectralResult(
        curvatures=np.array([0.1, 0.2, 0.5, 0.3, 0.15]),
        speeds=np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10]),
        layer_indices=np.array([1, 2, 3, 4, 5]),
        text_preview="Test text",
        label="test_prompt",
    )


@pytest.fixture
def sample_spectral_dict(sample_spectral_result):
    """Create multiple spectral results."""
    return {
        "English": sample_spectral_result,
        "Code": SpectralResult(
            curvatures=np.array([0.15, 0.25, 0.4, 0.35, 0.2]),
            speeds=np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09]),
            layer_indices=np.array([1, 2, 3, 4, 5]),
            text_preview="Code text",
            label="code",
        ),
    }


@pytest.fixture
def sample_thermo_result():
    """Create a sample ThermoResult."""
    return ThermoResult(
        step_lengths=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
        step_indices=np.array([1, 2, 3, 4, 5]),
        total_length=0.15,
        num_samples_processed=100,
        num_tokens_processed=5000,
    )


@pytest.fixture
def sample_thermo_per_sample():
    """Create sample per-sample thermodynamic results."""
    return ThermoResultPerSample(
        per_sample_lengths=np.random.rand(20, 5) * 0.1,
        mean_lengths=np.array([0.05, 0.06, 0.07, 0.08, 0.09]),
        step_indices=np.array([1, 2, 3, 4, 5]),
        num_samples=20,
    )


@pytest.fixture
def sample_belief_result():
    """Create a sample BeliefResult."""
    return BeliefResult(
        belief_norms=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        layer_indices=np.array([1, 2, 3, 4, 5, 6]),
        num_samples_processed=100,
        num_tokens_processed=5000,
        fr_norm=True,
    )


@pytest.fixture
def sample_belief_per_sample():
    """Create sample per-sample belief results."""
    return BeliefResultPerSample(
        per_sample_norms=np.random.rand(20, 6) * 0.5,
        mean_norms=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        layer_indices=np.array([1, 2, 3, 4, 5, 6]),
        num_samples=20,
    )


# -----------------------------------------------------------------------------
# Style Tests
# -----------------------------------------------------------------------------

class TestStyles:
    """Tests for visualization styles."""

    def test_apply_style_ndna(self):
        """Test applying nDNA style."""
        # Should not raise
        apply_style("ndna")

    def test_apply_style_minimal(self):
        """Test applying minimal style."""
        apply_style("minimal")

    def test_get_color_palette_concept(self):
        """Test getting concept color palette."""
        colors = get_color_palette(5, "concept")
        assert len(colors) == 5
        assert all(c.startswith("#") for c in colors)

    def test_get_color_palette_sequential(self):
        """Test getting sequential color palette."""
        colors = get_color_palette(10, "sequential")
        assert len(colors) == 10

    def test_get_color_palette_cycles(self):
        """Test color palette cycles for large n."""
        colors = get_color_palette(20, "concept")
        assert len(colors) == 20

    def test_colors_dict(self):
        """Test COLORS dict has expected keys."""
        assert "primary" in COLORS
        assert "curvature" in COLORS
        assert "fr_step" in COLORS
        assert "belief" in COLORS

    def test_figsize_constants(self):
        """Test figure size constants."""
        assert len(FIGSIZE_SINGLE) == 2
        assert FIGSIZE_SINGLE[0] > 0
        assert FIGSIZE_SINGLE[1] > 0

    def test_format_metric_title(self):
        """Test title formatting."""
        title = format_metric_title("Test Metric", "GPT-2")
        assert "Test Metric" in title
        assert "GPT-2" in title
        
        title_with_subtitle = format_metric_title("Metric", "Model", "Subtitle")
        assert "Subtitle" in title_with_subtitle


# -----------------------------------------------------------------------------
# 2D Plot Tests
# -----------------------------------------------------------------------------

class TestPlots2D:
    """Tests for 2D matplotlib plots."""

    def test_plot_spectral_curvature_single(self, temp_dir, sample_spectral_result):
        """Test plotting single spectral result."""
        from ndna.visualization.plots_2d import plot_spectral_curvature
        
        save_path = os.path.join(temp_dir, "spectral.png")
        fig = plot_spectral_curvature(
            sample_spectral_result,
            save_path=save_path,
            model_name="Test Model",
            show=False,
        )
        
        assert os.path.exists(save_path)
        assert fig is not None

    def test_plot_spectral_curvature_dict(self, temp_dir, sample_spectral_dict):
        """Test plotting multiple spectral results."""
        from ndna.visualization.plots_2d import plot_spectral_curvature
        
        save_path = os.path.join(temp_dir, "spectral_multi.png")
        fig = plot_spectral_curvature(
            sample_spectral_dict,
            save_path=save_path,
            show=False,
        )
        
        assert os.path.exists(save_path)

    def test_plot_spectral_curvature_log_scale(self, temp_dir, sample_spectral_result):
        """Test spectral plot with log scale."""
        from ndna.visualization.plots_2d import plot_spectral_curvature
        
        save_path = os.path.join(temp_dir, "spectral_log.png")
        fig = plot_spectral_curvature(
            sample_spectral_result,
            save_path=save_path,
            log_scale=True,
            show=False,
        )
        
        assert os.path.exists(save_path)

    def test_plot_thermodynamic_length(self, temp_dir, sample_thermo_result):
        """Test plotting thermodynamic length."""
        from ndna.visualization.plots_2d import plot_thermodynamic_length
        
        save_path = os.path.join(temp_dir, "thermo.png")
        fig = plot_thermodynamic_length(
            sample_thermo_result,
            save_path=save_path,
            model_name="Test Model",
            show=False,
        )
        
        assert os.path.exists(save_path)
        assert fig is not None

    def test_plot_belief_vector_field_single(self, temp_dir, sample_belief_result):
        """Test plotting single belief result."""
        from ndna.visualization.plots_2d import plot_belief_vector_field
        
        save_path = os.path.join(temp_dir, "belief.png")
        fig = plot_belief_vector_field(
            sample_belief_result,
            save_path=save_path,
            show=False,
        )
        
        assert os.path.exists(save_path)

    def test_plot_belief_vector_field_dict(self, temp_dir, sample_belief_result):
        """Test plotting multiple belief results."""
        from ndna.visualization.plots_2d import plot_belief_vector_field
        
        belief_dict = {
            "Concept A": sample_belief_result,
            "Concept B": BeliefResult(
                belief_norms=np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                layer_indices=np.array([1, 2, 3, 4, 5, 6]),
            ),
        }
        
        save_path = os.path.join(temp_dir, "belief_multi.png")
        fig = plot_belief_vector_field(
            belief_dict,
            save_path=save_path,
            show=False,
        )
        
        assert os.path.exists(save_path)

    def test_plot_all_metrics(
        self, temp_dir, sample_spectral_result, sample_thermo_result, sample_belief_result
    ):
        """Test generating all metric plots."""
        from ndna.visualization.plots_2d import plot_all_metrics
        
        save_dir = os.path.join(temp_dir, "all_plots")
        saved_paths = plot_all_metrics(
            sample_spectral_result,
            sample_thermo_result,
            sample_belief_result,
            save_dir=save_dir,
            model_name="Test Model",
            show=False,
        )
        
        assert len(saved_paths) == 3
        assert all(os.path.exists(p) for p in saved_paths)

    def test_plot_master_panel(
        self, temp_dir, sample_spectral_result, sample_thermo_result, sample_belief_result
    ):
        """Test creating master panel."""
        from ndna.visualization.plots_2d import plot_master_panel
        
        save_path = os.path.join(temp_dir, "master_panel.png")
        fig = plot_master_panel(
            sample_spectral_result,
            sample_thermo_result,
            sample_belief_result,
            save_path=save_path,
            model_name="Test Model",
            show=False,
            include_3d=True,
        )
        
        assert os.path.exists(save_path)
        assert fig is not None

    def test_plot_master_panel_without_3d(
        self, temp_dir, sample_spectral_result, sample_thermo_result, sample_belief_result
    ):
        """Test master panel without 3D subplot."""
        from ndna.visualization.plots_2d import plot_master_panel
        
        save_path = os.path.join(temp_dir, "master_panel_2d.png")
        fig = plot_master_panel(
            sample_spectral_result,
            sample_thermo_result,
            sample_belief_result,
            save_path=save_path,
            show=False,
            include_3d=False,
        )
        
        assert os.path.exists(save_path)


# -----------------------------------------------------------------------------
# 3D Plot Tests (require plotly)
# -----------------------------------------------------------------------------

class TestPlots3D:
    """Tests for 3D plotly plots."""

    @pytest.fixture
    def plotly_available(self):
        """Check if plotly is available."""
        try:
            import plotly
            return True
        except ImportError:
            pytest.skip("plotly not installed")
            return False

    def test_plot_spectral_3d(self, temp_dir, sample_spectral_dict, plotly_available):
        """Test 3D spectral surface plot."""
        from ndna.visualization.plots_3d import plot_spectral_3d
        
        save_path = os.path.join(temp_dir, "spectral_3d.html")
        fig = plot_spectral_3d(
            sample_spectral_dict,
            save_path=save_path,
            model_name="Test Model",
        )
        
        assert os.path.exists(save_path)
        assert fig is not None

    def test_plot_thermodynamic_3d(self, temp_dir, sample_thermo_per_sample, plotly_available):
        """Test 3D thermodynamic surface plot."""
        from ndna.visualization.plots_3d import plot_thermodynamic_3d
        
        save_path = os.path.join(temp_dir, "thermo_3d.html")
        fig = plot_thermodynamic_3d(
            sample_thermo_per_sample,
            save_path=save_path,
            model_name="Test Model",
        )
        
        assert os.path.exists(save_path)

    def test_plot_belief_3d(self, temp_dir, sample_belief_per_sample, plotly_available):
        """Test 3D belief surface plot."""
        from ndna.visualization.plots_3d import plot_belief_3d
        
        save_path = os.path.join(temp_dir, "belief_3d.html")
        fig = plot_belief_3d(
            sample_belief_per_sample,
            save_path=save_path,
            model_name="Test Model",
        )
        
        assert os.path.exists(save_path)

    def test_plot_ndna_trajectory_3d_single(self, temp_dir, plotly_available):
        """Test 3D nDNA trajectory with single concept."""
        from ndna.visualization.plots_3d import plot_ndna_trajectory_3d
        
        kappa = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        fr_steps = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        belief_norms = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        save_path = os.path.join(temp_dir, "ndna_3d.html")
        fig = plot_ndna_trajectory_3d(
            kappa,
            fr_steps,
            belief_norms,
            save_path=save_path,
            model_name="Test Model",
        )
        
        assert os.path.exists(save_path)

    def test_plot_ndna_trajectory_3d_multi_concept(self, temp_dir, plotly_available):
        """Test 3D nDNA trajectory with multiple concepts."""
        from ndna.visualization.plots_3d import plot_ndna_trajectory_3d
        
        kappa = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        fr_steps = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        belief_dict = {
            "Concept A": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "Concept B": np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65]),
        }
        
        save_path = os.path.join(temp_dir, "ndna_3d_multi.html")
        fig = plot_ndna_trajectory_3d(
            kappa,
            fr_steps,
            belief_dict,
            save_path=save_path,
        )
        
        assert os.path.exists(save_path)

    def test_export_multi_model_html(self, temp_dir, plotly_available):
        """Test exporting multi-model comparison HTML."""
        from ndna.visualization.plots_3d import export_multi_model_html
        
        models_data = {
            "Model A": {
                "kappa": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                "fr_steps": np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
                "belief_norms": {
                    "concept1": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                },
            },
            "Model B": {
                "kappa": np.array([0.15, 0.25, 0.35, 0.45, 0.55]),
                "fr_steps": np.array([0.02, 0.03, 0.04, 0.05, 0.06]),
                "belief_norms": {
                    "concept1": np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65]),
                },
            },
        }
        
        save_path = os.path.join(temp_dir, "comparison.html")
        fig = export_multi_model_html(models_data, save_path)
        
        assert os.path.exists(save_path)

    def test_plot_ndna_trajectory_shape_validation(self, plotly_available):
        """Test that shape validation works."""
        from ndna.visualization.plots_3d import plot_ndna_trajectory_3d
        
        kappa = np.array([0.1, 0.2, 0.3])
        fr_steps = np.array([0.1, 0.2])  # Wrong shape
        belief_norms = np.array([0.1, 0.2, 0.3, 0.4])
        
        with pytest.raises(ValueError, match="same shape"):
            plot_ndna_trajectory_3d(kappa, fr_steps, belief_norms)


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestVisualizationIntegration:
    """Integration tests for visualization pipeline."""

    def test_full_visualization_pipeline(
        self,
        temp_dir,
        sample_spectral_dict,
        sample_thermo_result,
        sample_belief_result,
        sample_thermo_per_sample,
        sample_belief_per_sample,
    ):
        """Test complete visualization pipeline."""
        from ndna.visualization.plots_2d import plot_all_metrics, plot_master_panel
        
        # Create output directories
        plots_2d_dir = os.path.join(temp_dir, "2d")
        
        # Generate 2D plots
        saved_2d = plot_all_metrics(
            sample_spectral_dict,
            sample_thermo_result,
            sample_belief_result,
            save_dir=plots_2d_dir,
            model_name="Integration Test",
            show=False,
        )
        
        # Generate master panel
        master_path = os.path.join(temp_dir, "master.png")
        plot_master_panel(
            sample_spectral_dict,
            sample_thermo_result,
            sample_belief_result,
            save_path=master_path,
            model_name="Integration Test",
            show=False,
        )
        
        # Verify outputs
        assert len(saved_2d) == 3
        assert os.path.exists(master_path)
        
        # Try 3D plots if plotly available
        try:
            from ndna.visualization.plots_3d import plot_all_3d
            
            plots_3d_dir = os.path.join(temp_dir, "3d")
            saved_3d = plot_all_3d(
                spectral_by_prompt=sample_spectral_dict,
                thermo_per_sample=sample_thermo_per_sample,
                belief_per_sample=sample_belief_per_sample,
                save_dir=plots_3d_dir,
                model_name="Integration Test",
            )
            
            assert len(saved_3d) >= 1
        except ImportError:
            pass  # plotly not available

