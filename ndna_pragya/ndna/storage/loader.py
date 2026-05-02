"""
ndna.storage.loader

Load nDNA metric results from disk.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

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


class ResultLoader:
    """
    Load nDNA metric results from disk.
    
    Loads results saved by ResultSaver from a structured directory.
    
    Example:
        >>> loader = ResultLoader()
        >>> results = loader.load("output/my_experiment")
        >>> print(results.spectral.curvatures)
    """
    
    def load(self, output_dir: str) -> FullResults:
        """
        Load all results from a structured directory.
        
        Args:
            output_dir: Directory containing saved results
        
        Returns:
            FullResults container with all available metrics
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            raise FileNotFoundError(f"Results directory not found: {output_dir}")
        
        # Load metadata
        metadata_path = output_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {"model_name": "unknown", "dataset_name": "unknown"}
        
        # Load individual metrics
        spectral = None
        spectral_path = output_path / "spectral_curvature.npz"
        if spectral_path.exists():
            spectral = self.load_spectral(str(spectral_path))
        
        spectral_by_prompt = None
        spectral_by_prompt_path = output_path / "spectral_by_prompt.npz"
        if spectral_by_prompt_path.exists():
            spectral_by_prompt = self.load_spectral_by_prompt(str(spectral_by_prompt_path))
        
        thermodynamic = None
        thermo_path = output_path / "thermodynamic_length.npz"
        if thermo_path.exists():
            thermodynamic = self.load_thermodynamic(str(thermo_path))
        
        belief = None
        belief_path = output_path / "belief_vector_field.npz"
        if belief_path.exists():
            belief = self.load_belief(str(belief_path))
        
        # Load per-sample data
        thermo_per_sample = None
        belief_per_sample = None
        per_sample_dir = output_path / "per_sample"
        
        if per_sample_dir.exists():
            thermo_ps_path = per_sample_dir / "thermodynamic_per_sample.npz"
            if thermo_ps_path.exists():
                thermo_per_sample = self.load_thermo_per_sample(str(thermo_ps_path))
            
            belief_ps_path = per_sample_dir / "belief_per_sample.npz"
            if belief_ps_path.exists():
                belief_per_sample = self.load_belief_per_sample(str(belief_ps_path))
        
        return FullResults(
            model_name=metadata.get("model_name", "unknown"),
            dataset_name=metadata.get("dataset_name", "unknown"),
            spectral=spectral,
            spectral_by_prompt=spectral_by_prompt,
            thermodynamic=thermodynamic,
            belief=belief,
            thermo_per_sample=thermo_per_sample,
            belief_per_sample=belief_per_sample,
            config=metadata.get("config"),
        )
    
    def load_spectral(
        self,
        path: str
    ) -> Union[SpectralResult, AggregatedSpectralResult]:
        """
        Load spectral curvature result from NPZ.
        
        Args:
            path: Path to NPZ file
        
        Returns:
            SpectralResult or AggregatedSpectralResult
        """
        data = np.load(path, allow_pickle=True)
        
        is_aggregated = bool(data.get("is_aggregated", [False])[0])
        
        if is_aggregated:
            return AggregatedSpectralResult(
                mean_curvatures=data["mean_curvatures"],
                std_curvatures=data["std_curvatures"],
                all_curvatures=data["all_curvatures"],
                layer_indices=data["layer_indices"],
                num_texts=int(data["num_texts"][0]),
                individual_results=[],  # Not reconstructed
            )
        else:
            text_preview = str(data["text_preview"][0]) if "text_preview" in data else ""
            label = str(data["label"][0]) if "label" in data and data["label"][0] else None
            
            return SpectralResult(
                curvatures=data["curvatures"],
                speeds=data["speeds"],
                layer_indices=data["layer_indices"],
                text_preview=text_preview,
                label=label,
                mean_curvature=float(data["mean_curvature"][0]),
                max_curvature=float(data["max_curvature"][0]),
                min_curvature=float(data["min_curvature"][0]),
                max_layer=int(data["max_layer"][0]),
                min_layer=int(data["min_layer"][0]),
            )
    
    def load_spectral_by_prompt(self, path: str) -> Dict[str, SpectralResult]:
        """
        Load multiple spectral results from NPZ.
        
        Args:
            path: Path to NPZ file
        
        Returns:
            Dict mapping label to SpectralResult
        """
        data = np.load(path, allow_pickle=True)
        
        labels = [str(l) for l in data["labels"]]
        all_curvatures = data["all_curvatures"]
        all_speeds = data["all_speeds"]
        layer_indices = data["layer_indices"]
        
        results = {}
        for i, label in enumerate(labels):
            results[label] = SpectralResult(
                curvatures=all_curvatures[i],
                speeds=all_speeds[i],
                layer_indices=layer_indices,
                text_preview="",
                label=label,
            )
        
        return results
    
    def load_thermodynamic(self, path: str) -> ThermoResult:
        """
        Load thermodynamic length result from NPZ.
        
        Args:
            path: Path to NPZ file
        
        Returns:
            ThermoResult
        """
        data = np.load(path, allow_pickle=True)
        
        return ThermoResult(
            step_lengths=data["step_lengths"],
            step_indices=data["step_indices"],
            total_length=float(data["total_length"][0]),
            num_samples_processed=int(data["num_samples_processed"][0]),
            num_tokens_processed=int(data.get("num_tokens_processed", [0])[0]),
        )
    
    def load_belief(self, path: str) -> BeliefResult:
        """
        Load belief vector field result from NPZ.
        
        Args:
            path: Path to NPZ file
        
        Returns:
            BeliefResult
        """
        data = np.load(path, allow_pickle=True)
        
        return BeliefResult(
            belief_norms=data["belief_norms"],
            layer_indices=data["layer_indices"],
            mean_norm=float(data["mean_norm"][0]),
            max_norm=float(data["max_norm"][0]),
            min_norm=float(data["min_norm"][0]),
            max_layer=int(data["max_layer"][0]),
            min_layer=int(data["min_layer"][0]),
            num_samples_processed=int(data.get("num_samples_processed", [0])[0]),
            num_tokens_processed=int(data.get("num_tokens_processed", [0])[0]),
            fr_norm=bool(data.get("fr_norm", [True])[0]),
        )
    
    def load_thermo_per_sample(self, path: str) -> ThermoResultPerSample:
        """
        Load per-sample thermodynamic results from NPZ.
        
        Args:
            path: Path to NPZ file
        
        Returns:
            ThermoResultPerSample
        """
        data = np.load(path, allow_pickle=True)
        
        return ThermoResultPerSample(
            per_sample_lengths=data["per_sample_lengths"],
            mean_lengths=data["mean_lengths"],
            step_indices=data["step_indices"],
            num_samples=int(data["num_samples"][0]),
        )
    
    def load_belief_per_sample(self, path: str) -> BeliefResultPerSample:
        """
        Load per-sample belief results from NPZ.
        
        Args:
            path: Path to NPZ file
        
        Returns:
            BeliefResultPerSample
        """
        data = np.load(path, allow_pickle=True)
        
        return BeliefResultPerSample(
            per_sample_norms=data["per_sample_norms"],
            mean_norms=data["mean_norms"],
            layer_indices=data["layer_indices"],
            num_samples=int(data["num_samples"][0]),
        )
    
    def load_ndna(self, path: str) -> Union[nDNAResult, nDNAResultMultiConcept]:
        """
        Load nDNA combined metric result from NPZ.
        
        Args:
            path: Path to NPZ file
        
        Returns:
            nDNAResult or nDNAResultMultiConcept
        """
        data = np.load(path, allow_pickle=True)
        
        is_multi = bool(data.get("is_multi_concept", [False])[0])
        
        if is_multi:
            concepts = [str(c) for c in data["concepts"]]
            scalars = {c: float(s) for c, s in zip(concepts, data["scalars"])}
            layerwise = {c: lw for c, lw in zip(concepts, data["layerwise"])}
            
            return nDNAResultMultiConcept(
                scalars=scalars,
                layerwise=layerwise,
                layer_indices=data["layer_indices"],
                l_min=int(data["l_min"][0]),
                kappa=data.get("kappa"),
                fr_steps=data.get("fr_steps"),
            )
        else:
            return nDNAResult(
                scalar=float(data["scalar"][0]),
                layerwise=data["layerwise"],
                layer_indices=data["layer_indices"],
                concept_name=str(data["concept_name"][0]),
                l_min=int(data["l_min"][0]),
                kappa=data.get("kappa"),
                fr_steps=data.get("fr_steps"),
                belief_norms=data.get("belief_norms"),
            )
    
    def load_metadata(self, output_dir: str) -> Dict[str, Any]:
        """
        Load only metadata from results directory.
        
        Args:
            output_dir: Directory containing saved results
        
        Returns:
            Metadata dictionary
        """
        metadata_path = Path(output_dir) / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path) as f:
            return json.load(f)

