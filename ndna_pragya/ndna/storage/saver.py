"""
ndna.storage.saver

Save nDNA metric results to disk in NPZ + JSON format.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from ..core.results import (
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


class ResultSaver:
    """
    Save nDNA metric results to disk.
    
    Creates a structured directory with:
        - NPZ files for array data
        - JSON file for metadata
        - Per-sample subdirectory for 3D visualization data
    
    Example:
        >>> saver = ResultSaver()
        >>> saver.save(results, "output/my_experiment")
    """
    
    def __init__(self, compress: bool = True):
        """
        Initialize the saver.
        
        Args:
            compress: Whether to use compressed NPZ files (savez_compressed)
        """
        self.compress = compress
    
    def save(self, results: FullResults, output_dir: str) -> None:
        """
        Save all results to a structured directory.
        
        Creates:
            output_dir/
            ├── spectral_curvature.npz
            ├── thermodynamic_length.npz
            ├── belief_vector_field.npz
            ├── ndna_combined.npz (if nDNA computed)
            ├── per_sample/
            │   ├── thermodynamic_per_sample.npz
            │   └── belief_per_sample.npz
            └── metadata.json
        
        Args:
            results: FullResults container with all metrics
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual metrics
        if results.spectral is not None:
            self.save_spectral(
                results.spectral,
                str(output_path / "spectral_curvature.npz")
            )
        
        if results.spectral_by_prompt is not None:
            self.save_spectral_by_prompt(
                results.spectral_by_prompt,
                str(output_path / "spectral_by_prompt.npz")
            )
        
        if results.thermodynamic is not None:
            self.save_thermodynamic(
                results.thermodynamic,
                str(output_path / "thermodynamic_length.npz")
            )
        
        if results.belief is not None:
            self.save_belief(
                results.belief,
                str(output_path / "belief_vector_field.npz")
            )
        
        # Save per-sample data
        if results.thermo_per_sample is not None or results.belief_per_sample is not None:
            per_sample_dir = output_path / "per_sample"
            per_sample_dir.mkdir(exist_ok=True)
            
            if results.thermo_per_sample is not None:
                self.save_thermo_per_sample(
                    results.thermo_per_sample,
                    str(per_sample_dir / "thermodynamic_per_sample.npz")
                )
            
            if results.belief_per_sample is not None:
                self.save_belief_per_sample(
                    results.belief_per_sample,
                    str(per_sample_dir / "belief_per_sample.npz")
                )
        
        # Save metadata
        self._save_metadata(results, str(output_path / "metadata.json"))
    
    def save_spectral(
        self,
        result: Union[SpectralResult, AggregatedSpectralResult],
        path: str
    ) -> None:
        """
        Save spectral curvature result to NPZ.
        
        Args:
            result: SpectralResult or AggregatedSpectralResult
            path: Output file path (.npz)
        """
        if isinstance(result, AggregatedSpectralResult):
            data = {
                "mean_curvatures": result.mean_curvatures,
                "std_curvatures": result.std_curvatures,
                "all_curvatures": result.all_curvatures,
                "layer_indices": result.layer_indices,
                "num_texts": np.array([result.num_texts]),
                "is_aggregated": np.array([True]),
            }
            # Save individual labels if available
            if result.individual_results:
                labels = [r.label or "" for r in result.individual_results]
                data["labels"] = np.array(labels, dtype=object)
        else:
            data = {
                "curvatures": result.curvatures,
                "speeds": result.speeds,
                "layer_indices": result.layer_indices,
                "text_preview": np.array([result.text_preview], dtype=object),
                "label": np.array([result.label or ""], dtype=object),
                "mean_curvature": np.array([result.mean_curvature]),
                "max_curvature": np.array([result.max_curvature]),
                "min_curvature": np.array([result.min_curvature]),
                "max_layer": np.array([result.max_layer]),
                "min_layer": np.array([result.min_layer]),
                "is_aggregated": np.array([False]),
            }
        
        self._save_npz(data, path)
    
    def save_spectral_by_prompt(
        self,
        results: Dict[str, SpectralResult],
        path: str
    ) -> None:
        """
        Save multiple spectral results (keyed by prompt label) to NPZ.
        
        Args:
            results: Dict mapping label to SpectralResult
            path: Output file path (.npz)
        """
        labels = list(results.keys())
        all_curvatures = np.array([results[l].curvatures for l in labels])
        all_speeds = np.array([results[l].speeds for l in labels])
        
        first = next(iter(results.values()))
        
        data = {
            "labels": np.array(labels, dtype=object),
            "all_curvatures": all_curvatures,
            "all_speeds": all_speeds,
            "layer_indices": first.layer_indices,
            "mean_curvatures": np.mean(all_curvatures, axis=0),
            "std_curvatures": np.std(all_curvatures, axis=0),
        }
        
        self._save_npz(data, path)
    
    def save_thermodynamic(self, result: ThermoResult, path: str) -> None:
        """
        Save thermodynamic length result to NPZ.
        
        Args:
            result: ThermoResult
            path: Output file path (.npz)
        """
        data = {
            "step_lengths": result.step_lengths,
            "step_indices": result.step_indices,
            "total_length": np.array([result.total_length]),
            "num_samples_processed": np.array([result.num_samples_processed]),
            "num_tokens_processed": np.array([result.num_tokens_processed]),
        }
        
        self._save_npz(data, path)
    
    def save_belief(self, result: BeliefResult, path: str) -> None:
        """
        Save belief vector field result to NPZ.
        
        Args:
            result: BeliefResult
            path: Output file path (.npz)
        """
        data = {
            "belief_norms": result.belief_norms,
            "layer_indices": result.layer_indices,
            "mean_norm": np.array([result.mean_norm]),
            "max_norm": np.array([result.max_norm]),
            "min_norm": np.array([result.min_norm]),
            "max_layer": np.array([result.max_layer]),
            "min_layer": np.array([result.min_layer]),
            "num_samples_processed": np.array([result.num_samples_processed]),
            "num_tokens_processed": np.array([result.num_tokens_processed]),
            "fr_norm": np.array([result.fr_norm]),
        }
        
        self._save_npz(data, path)
    
    def save_thermo_per_sample(
        self,
        result: ThermoResultPerSample,
        path: str
    ) -> None:
        """
        Save per-sample thermodynamic results to NPZ.
        
        Args:
            result: ThermoResultPerSample
            path: Output file path (.npz)
        """
        data = {
            "per_sample_lengths": result.per_sample_lengths,
            "mean_lengths": result.mean_lengths,
            "step_indices": result.step_indices,
            "num_samples": np.array([result.num_samples]),
        }
        
        self._save_npz(data, path)
    
    def save_belief_per_sample(
        self,
        result: BeliefResultPerSample,
        path: str
    ) -> None:
        """
        Save per-sample belief results to NPZ.
        
        Args:
            result: BeliefResultPerSample
            path: Output file path (.npz)
        """
        data = {
            "per_sample_norms": result.per_sample_norms,
            "mean_norms": result.mean_norms,
            "layer_indices": result.layer_indices,
            "num_samples": np.array([result.num_samples]),
        }
        
        self._save_npz(data, path)
    
    def save_ndna(
        self,
        result: Union[nDNAResult, nDNAResultMultiConcept],
        path: str
    ) -> None:
        """
        Save nDNA combined metric result to NPZ.
        
        Args:
            result: nDNAResult or nDNAResultMultiConcept
            path: Output file path (.npz)
        """
        if isinstance(result, nDNAResultMultiConcept):
            concepts = list(result.scalars.keys())
            scalars = np.array([result.scalars[c] for c in concepts])
            layerwise = np.array([result.layerwise[c] for c in concepts])
            
            data = {
                "concepts": np.array(concepts, dtype=object),
                "scalars": scalars,
                "layerwise": layerwise,
                "layer_indices": result.layer_indices,
                "l_min": np.array([result.l_min]),
                "is_multi_concept": np.array([True]),
            }
            
            if result.kappa is not None:
                data["kappa"] = result.kappa
            if result.fr_steps is not None:
                data["fr_steps"] = result.fr_steps
        else:
            data = {
                "scalar": np.array([result.scalar]),
                "layerwise": result.layerwise,
                "layer_indices": result.layer_indices,
                "concept_name": np.array([result.concept_name], dtype=object),
                "l_min": np.array([result.l_min]),
                "is_multi_concept": np.array([False]),
            }
            
            if result.kappa is not None:
                data["kappa"] = result.kappa
            if result.fr_steps is not None:
                data["fr_steps"] = result.fr_steps
            if result.belief_norms is not None:
                data["belief_norms"] = result.belief_norms
        
        self._save_npz(data, path)
    
    def _save_metadata(self, results: FullResults, path: str) -> None:
        """
        Save metadata to JSON.
        
        Args:
            results: FullResults container
            path: Output file path (.json)
        """
        # Determine number of layers from available results
        num_layers = None
        if results.spectral is not None:
            num_layers = len(results.spectral.curvatures) + 2
        elif results.thermodynamic is not None:
            num_layers = len(results.thermodynamic.step_lengths) + 1
        elif results.belief is not None:
            num_layers = len(results.belief.belief_norms)
        
        metadata = {
            "model_name": results.model_name,
            "dataset_name": results.dataset_name,
            "timestamp": datetime.now().isoformat(),
            "num_layers": num_layers,
            "metrics_computed": {
                "spectral": results.spectral is not None,
                "spectral_by_prompt": results.spectral_by_prompt is not None,
                "thermodynamic": results.thermodynamic is not None,
                "belief": results.belief is not None,
                "thermo_per_sample": results.thermo_per_sample is not None,
                "belief_per_sample": results.belief_per_sample is not None,
            },
        }
        
        # Add sample counts if available
        if results.thermodynamic is not None:
            metadata["num_samples"] = results.thermodynamic.num_samples_processed
            metadata["num_tokens"] = results.thermodynamic.num_tokens_processed
        elif results.belief is not None:
            metadata["num_samples"] = results.belief.num_samples_processed
            metadata["num_tokens"] = results.belief.num_tokens_processed
        
        # Add config if available
        if results.config is not None:
            metadata["config"] = self._make_json_serializable(results.config)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    
    def _save_npz(self, data: Dict[str, np.ndarray], path: str) -> None:
        """Save dictionary to NPZ file."""
        if self.compress:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

