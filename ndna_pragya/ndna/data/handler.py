"""
ndna.data.handler

DatasetHandler for loading and preprocessing text datasets.
Supports HuggingFace datasets, text lists, and local files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from torch.utils.data import DataLoader, Dataset

from .collate import make_belief_collate, make_causal_collate
from .registry import DATASET_REGISTRY, get_dataset_config, make_text_processor


class TextDataset(Dataset):
    """Simple dataset wrapper for list of text dicts."""

    def __init__(self, data: List[Dict[str, str]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]


class DatasetHandler:
    """
    Load and preprocess text datasets from various sources.

    Provides a unified interface for:
    - Loading from HuggingFace Hub
    - Loading from text lists
    - Loading from local files (txt, jsonl, csv)

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> handler = DatasetHandler(tokenizer, max_length=256, batch_size=4)
        >>> handler.load_huggingface("squad", max_samples=1000)
        >>> loader = handler.create_dataloader(collate_type="causal")
        >>> for batch in loader:
        ...     print(batch["input_ids"].shape)
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 256,
        batch_size: int = 2,
        num_workers: int = 4,
    ):
        """
        Initialize DatasetHandler.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes for DataLoader
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._dataset: Optional[Dataset] = None
        self._source_info: Dict[str, Any] = {}

    def load_huggingface(
        self,
        name: str,
        split: str = "train",
        text_column: Optional[str] = None,
        text_processor: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        config: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> "DatasetHandler":
        """
        Load dataset from HuggingFace Hub.

        If `name` is in DATASET_REGISTRY, uses the registered config.
        Otherwise, treats `name` as a HuggingFace dataset identifier.

        Args:
            name: Dataset name (registry key or HF identifier)
            split: Dataset split to load
            text_column: Column containing text (for simple extraction)
            text_processor: Custom function to convert examples to {"text": str}
            max_samples: Maximum number of samples to load
            config: Dataset configuration/subset name
            hf_token: HuggingFace API token for gated datasets

        Returns:
            Self for method chaining
        """
        from datasets import load_dataset

        # Check registry first
        if name in DATASET_REGISTRY:
            reg_config = get_dataset_config(name)
            hf_name = reg_config["hf_name"]
            config = config or reg_config.get("config")
            split = reg_config.get("split", split)
            text_column = text_column or reg_config.get("text_column")
            text_processor = text_processor or reg_config.get("processor")
        else:
            hf_name = name

        # Load dataset
        load_kwargs = {"split": split}
        if config:
            load_kwargs["name"] = config
        if hf_token:
            load_kwargs["token"] = hf_token

        ds = load_dataset(hf_name, **load_kwargs)

        # Apply max_samples
        if max_samples is not None and len(ds) > max_samples:
            ds = ds.select(range(max_samples))

        # Create text processor
        processor = make_text_processor(text_column, text_processor)

        # Apply processor to create "text" column
        ds = ds.map(processor, remove_columns=ds.column_names)

        # Filter empty texts
        ds = ds.filter(lambda ex: ex.get("text") and len(ex["text"].strip()) > 0)

        self._dataset = ds
        self._source_info = {
            "type": "huggingface",
            "name": name,
            "hf_name": hf_name,
            "split": split,
            "config": config,
            "num_samples": len(ds),
        }

        return self

    def load_texts(self, texts: List[str]) -> "DatasetHandler":
        """
        Load dataset from a list of text strings.

        Args:
            texts: List of text strings

        Returns:
            Self for method chaining
        """
        data = [{"text": t} for t in texts if t and t.strip()]
        self._dataset = TextDataset(data)
        self._source_info = {
            "type": "text_list",
            "num_samples": len(data),
        }
        return self

    def load_file(
        self,
        path: str,
        format: Optional[str] = None,
        text_column: str = "text",
    ) -> "DatasetHandler":
        """
        Load dataset from a local file.

        Supported formats:
        - txt: One text per line
        - jsonl: JSON Lines with "text" field
        - csv: CSV with specified text column

        Args:
            path: Path to the file
            format: File format ("txt", "jsonl", "csv"). Auto-detected if None.
            text_column: Column name for CSV files

        Returns:
            Self for method chaining
        """
        path = Path(path)

        if format is None:
            suffix = path.suffix.lower()
            if suffix == ".txt":
                format = "txt"
            elif suffix in (".jsonl", ".json"):
                format = "jsonl"
            elif suffix == ".csv":
                format = "csv"
            else:
                raise ValueError(f"Cannot auto-detect format for {path.suffix}")

        data = []

        if format == "txt":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append({"text": line})

        elif format == "jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if "text" in obj:
                            data.append({"text": obj["text"]})
                        elif text_column in obj:
                            data.append({"text": obj[text_column]})

        elif format == "csv":
            import csv

            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if text_column in row and row[text_column]:
                        data.append({"text": row[text_column]})

        else:
            raise ValueError(f"Unsupported format: {format}")

        self._dataset = TextDataset(data)
        self._source_info = {
            "type": "file",
            "path": str(path),
            "format": format,
            "num_samples": len(data),
        }

        return self

    def create_dataloader(
        self,
        collate_type: str = "causal",
        keep_last_k: int = 32,
        shuffle: bool = False,
    ) -> DataLoader:
        """
        Create a DataLoader with the appropriate collate function.

        Args:
            collate_type: Type of collate function:
                - "causal": Teacher-forced next-token prediction (for thermodynamic)
                - "belief": Keep last K tokens per sample (for belief vectors)
            keep_last_k: For belief collate, number of last positions to keep
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader ready for iteration

        Raises:
            ValueError: If no dataset loaded or invalid collate_type
        """
        if self._dataset is None:
            raise ValueError("No dataset loaded. Call load_* method first.")

        if collate_type == "causal":
            collate_fn = make_causal_collate(
                self.tokenizer,
                max_length=self.max_length,
            )
        elif collate_type == "belief":
            collate_fn = make_belief_collate(
                self.tokenizer,
                max_length=self.max_length,
                keep_last_k=keep_last_k,
            )
        else:
            raise ValueError(f"Unknown collate_type: {collate_type}")

        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_sample_texts(self, n: int = 10) -> List[str]:
        """
        Get sample texts from the loaded dataset.

        Args:
            n: Number of samples to return

        Returns:
            List of text strings
        """
        if self._dataset is None:
            return []

        samples = []
        for i in range(min(n, len(self._dataset))):
            item = self._dataset[i]
            if isinstance(item, dict) and "text" in item:
                samples.append(item["text"])

        return samples

    @property
    def dataset(self) -> Optional[Dataset]:
        """Get the underlying dataset."""
        return self._dataset

    @property
    def source_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset source."""
        return self._source_info

    @property
    def num_samples(self) -> int:
        """Get the number of samples in the dataset."""
        if self._dataset is None:
            return 0
        return len(self._dataset)

    def __len__(self) -> int:
        return self.num_samples

    def __repr__(self) -> str:
        if self._dataset is None:
            return "DatasetHandler(no dataset loaded)"

        return (
            f"DatasetHandler("
            f"source={self._source_info.get('type', 'unknown')!r}, "
            f"num_samples={self.num_samples}, "
            f"max_length={self.max_length}, "
            f"batch_size={self.batch_size})"
        )

