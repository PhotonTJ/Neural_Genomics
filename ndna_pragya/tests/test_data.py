"""
Unit tests for ndna.data module.

Uses synthetic data to avoid network calls during tests.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from ndna.data import (
    # Registry
    DATASET_REGISTRY,
    get_dataset_config,
    list_datasets,
    make_text_processor,
    # Collate
    make_causal_collate,
    make_belief_collate,
    # Prompts
    DEFAULT_PROMPTS,
    PROMPT_SETS,
    get_prompts,
    list_prompt_sets,
    get_all_prompts,
    # Handler
    DatasetHandler,
    TextDataset,
)


# -----------------------------------------------------------------------------
# Registry Tests
# -----------------------------------------------------------------------------

class TestDatasetRegistry:
    """Tests for the dataset registry."""

    def test_registry_structure(self):
        """Test that DATASET_REGISTRY has expected structure."""
        assert isinstance(DATASET_REGISTRY, dict)
        assert len(DATASET_REGISTRY) > 0

        # Check some expected datasets
        assert "squad" in DATASET_REGISTRY
        assert "imdb" in DATASET_REGISTRY
        assert "wikitext" in DATASET_REGISTRY

    def test_registry_entries_have_required_fields(self):
        """Test that each registry entry has required fields."""
        for name, config in DATASET_REGISTRY.items():
            assert isinstance(config, dict), f"{name} config is not a dict"
            assert "hf_name" in config, f"{name} missing hf_name"
            assert "split" in config, f"{name} missing split"
            # Must have either text_column or processor
            has_text_access = "text_column" in config or "processor" in config
            assert has_text_access, f"{name} missing text_column or processor"

    def test_get_dataset_config(self):
        """Test getting dataset config by name."""
        config = get_dataset_config("squad")
        assert config["hf_name"] == "squad"
        assert "processor" in config

    def test_get_dataset_config_invalid(self):
        """Test that invalid dataset name raises ValueError."""
        with pytest.raises(ValueError, match="not in registry"):
            get_dataset_config("nonexistent_dataset")

    def test_list_datasets(self):
        """Test listing available datasets."""
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert "squad" in datasets

    def test_make_text_processor_with_column(self):
        """Test making processor from text column."""
        processor = make_text_processor(text_column="content")
        result = processor({"content": "Hello world"})
        assert result == {"text": "Hello world"}

    def test_make_text_processor_with_function(self):
        """Test making processor from custom function."""
        custom_fn = lambda ex: {"text": f"Q: {ex['q']}"}
        processor = make_text_processor(processor=custom_fn)
        result = processor({"q": "What is AI?"})
        assert result == {"text": "Q: What is AI?"}

    def test_make_text_processor_default(self):
        """Test default processor assumes 'text' column."""
        processor = make_text_processor()
        result = processor({"text": "Default text"})
        assert result == {"text": "Default text"}


# -----------------------------------------------------------------------------
# Prompts Tests
# -----------------------------------------------------------------------------

class TestPrompts:
    """Tests for prompt sets."""

    def test_default_prompts_structure(self):
        """Test DEFAULT_PROMPTS has expected structure."""
        assert isinstance(DEFAULT_PROMPTS, list)
        assert len(DEFAULT_PROMPTS) > 0

        for item in DEFAULT_PROMPTS:
            assert isinstance(item, tuple)
            assert len(item) == 2
            label, text = item
            assert isinstance(label, str)
            assert isinstance(text, str)
            assert len(label) > 0
            assert len(text) > 0

    def test_prompt_sets_structure(self):
        """Test PROMPT_SETS has expected structure."""
        assert isinstance(PROMPT_SETS, dict)
        assert "default" in PROMPT_SETS
        assert "multilingual" in PROMPT_SETS
        assert "technical" in PROMPT_SETS

        for name, prompts in PROMPT_SETS.items():
            assert isinstance(prompts, list), f"{name} is not a list"
            assert len(prompts) > 0, f"{name} is empty"

    def test_get_prompts(self):
        """Test getting prompts by name."""
        prompts = get_prompts("default")
        assert prompts == DEFAULT_PROMPTS

        prompts = get_prompts("multilingual")
        assert len(prompts) > 0

    def test_get_prompts_invalid(self):
        """Test that invalid prompt set raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            get_prompts("nonexistent_set")

    def test_list_prompt_sets(self):
        """Test listing available prompt sets."""
        sets = list_prompt_sets()
        assert isinstance(sets, list)
        assert "default" in sets
        assert "multilingual" in sets

    def test_get_all_prompts(self):
        """Test getting all prompts combined."""
        all_prompts = get_all_prompts()
        assert isinstance(all_prompts, list)
        # Should have more prompts than any single set
        assert len(all_prompts) > len(DEFAULT_PROMPTS)
        # Labels should be unique (prefixed with set name)
        labels = [label for label, _ in all_prompts]
        assert len(labels) == len(set(labels))


# -----------------------------------------------------------------------------
# Collate Function Tests
# -----------------------------------------------------------------------------

class MockTokenizer:
    """Mock tokenizer for testing collate functions."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1

    def __call__(
        self,
        texts,
        padding="longest",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ):
        # Simple mock: convert each char to its ord value % vocab_size
        batch_ids = []
        max_len = 0
        for text in texts:
            ids = [ord(c) % self.vocab_size for c in text[:max_length]]
            batch_ids.append(ids)
            max_len = max(max_len, len(ids))

        # Pad to max_len
        input_ids = []
        attention_mask = []
        for ids in batch_ids:
            padded = ids + [self.pad_token_id] * (max_len - len(ids))
            mask = [1] * len(ids) + [0] * (max_len - len(ids))
            input_ids.append(padded)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


class TestCollate:
    """Tests for collate functions."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    def test_make_causal_collate(self, tokenizer):
        """Test causal collate function creation."""
        collate_fn = make_causal_collate(tokenizer, max_length=32)
        assert callable(collate_fn)

    def test_causal_collate_output_shape(self, tokenizer):
        """Test causal collate output shapes."""
        collate_fn = make_causal_collate(tokenizer, max_length=32)

        batch = [
            {"text": "Hello world"},
            {"text": "Test input"},
        ]

        result = collate_fn(batch)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

        B = 2
        assert result["input_ids"].shape[0] == B
        assert result["attention_mask"].shape[0] == B
        assert result["labels"].shape[0] == B

        # Shapes should match
        assert result["input_ids"].shape == result["attention_mask"].shape
        assert result["input_ids"].shape == result["labels"].shape

    def test_causal_collate_shift(self, tokenizer):
        """Test that causal collate properly shifts for next-token prediction."""
        collate_fn = make_causal_collate(tokenizer, max_length=32)

        batch = [{"text": "ABCD"}]  # 4 characters
        result = collate_fn(batch)

        # After shift: input is first 3 tokens, labels is last 3 tokens
        # So length should be original - 1
        assert result["input_ids"].shape[1] == 3  # 4 - 1 = 3

    def test_make_belief_collate(self, tokenizer):
        """Test belief collate function creation."""
        collate_fn = make_belief_collate(tokenizer, max_length=32, keep_last_k=4)
        assert callable(collate_fn)

    def test_belief_collate_output_shape(self, tokenizer):
        """Test belief collate output shapes."""
        collate_fn = make_belief_collate(tokenizer, max_length=32, keep_last_k=4)

        batch = [
            {"text": "Hello world test"},
            {"text": "Another sample text"},
        ]

        result = collate_fn(batch)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert "select_mask" in result

        B = 2
        assert result["select_mask"].shape[0] == B
        assert result["select_mask"].dtype == torch.bool

    def test_belief_collate_select_mask(self, tokenizer):
        """Test that belief collate properly creates select_mask."""
        collate_fn = make_belief_collate(tokenizer, max_length=32, keep_last_k=3)

        batch = [{"text": "ABCDEFGH"}]  # 8 characters -> 7 after shift
        result = collate_fn(batch)

        select_mask = result["select_mask"][0]

        # Should have exactly keep_last_k True values (or fewer if sequence is shorter)
        assert select_mask.sum().item() <= 3
        assert select_mask.sum().item() > 0

        # True values should be at the end
        true_positions = select_mask.nonzero(as_tuple=True)[0]
        if len(true_positions) > 1:
            # Should be consecutive at the end
            diffs = true_positions[1:] - true_positions[:-1]
            assert (diffs == 1).all()


# -----------------------------------------------------------------------------
# TextDataset Tests
# -----------------------------------------------------------------------------

class TestTextDataset:
    """Tests for TextDataset wrapper."""

    def test_text_dataset_creation(self):
        """Test creating a TextDataset."""
        data = [{"text": "Hello"}, {"text": "World"}]
        ds = TextDataset(data)
        assert len(ds) == 2

    def test_text_dataset_getitem(self):
        """Test accessing items from TextDataset."""
        data = [{"text": "Hello"}, {"text": "World"}]
        ds = TextDataset(data)
        assert ds[0] == {"text": "Hello"}
        assert ds[1] == {"text": "World"}


# -----------------------------------------------------------------------------
# DatasetHandler Tests
# -----------------------------------------------------------------------------

class TestDatasetHandler:
    """Tests for DatasetHandler."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    def test_handler_init(self, tokenizer):
        """Test DatasetHandler initialization."""
        handler = DatasetHandler(tokenizer, max_length=128, batch_size=4)
        assert handler.max_length == 128
        assert handler.batch_size == 4
        assert handler.num_samples == 0

    def test_handler_load_texts(self, tokenizer):
        """Test loading from text list."""
        handler = DatasetHandler(tokenizer)
        handler.load_texts(["Hello world", "Test text", "Another sample"])

        assert handler.num_samples == 3
        assert handler.source_info["type"] == "text_list"

    def test_handler_load_texts_filters_empty(self, tokenizer):
        """Test that empty texts are filtered."""
        handler = DatasetHandler(tokenizer)
        handler.load_texts(["Hello", "", "  ", "World"])

        assert handler.num_samples == 2

    def test_handler_load_file_txt(self, tokenizer):
        """Test loading from txt file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Line 1\n")
            f.write("Line 2\n")
            f.write("Line 3\n")
            f.flush()

            handler = DatasetHandler(tokenizer)
            handler.load_file(f.name)

            assert handler.num_samples == 3
            assert handler.source_info["format"] == "txt"

    def test_handler_load_file_jsonl(self, tokenizer):
        """Test loading from jsonl file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "First entry"}) + "\n")
            f.write(json.dumps({"text": "Second entry"}) + "\n")
            f.flush()

            handler = DatasetHandler(tokenizer)
            handler.load_file(f.name)

            assert handler.num_samples == 2
            assert handler.source_info["format"] == "jsonl"

    def test_handler_load_file_csv(self, tokenizer):
        """Test loading from csv file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,label\n")
            f.write("Sample 1,A\n")
            f.write("Sample 2,B\n")
            f.flush()

            handler = DatasetHandler(tokenizer)
            handler.load_file(f.name, text_column="text")

            assert handler.num_samples == 2
            assert handler.source_info["format"] == "csv"

    def test_handler_get_sample_texts(self, tokenizer):
        """Test getting sample texts."""
        handler = DatasetHandler(tokenizer)
        handler.load_texts(["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"])

        samples = handler.get_sample_texts(n=3)
        assert len(samples) == 3
        assert samples[0] == "Text 1"

    def test_handler_create_dataloader_causal(self, tokenizer):
        """Test creating causal dataloader."""
        handler = DatasetHandler(tokenizer, batch_size=2)
        handler.load_texts(["Hello world", "Test text", "Another sample", "Fourth"])

        loader = handler.create_dataloader(collate_type="causal")

        batch = next(iter(loader))
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 2  # batch_size

    def test_handler_create_dataloader_belief(self, tokenizer):
        """Test creating belief dataloader."""
        handler = DatasetHandler(tokenizer, batch_size=2)
        handler.load_texts(["Hello world", "Test text"])

        loader = handler.create_dataloader(collate_type="belief", keep_last_k=4)

        batch = next(iter(loader))
        assert "select_mask" in batch
        assert batch["select_mask"].dtype == torch.bool

    def test_handler_create_dataloader_no_dataset(self, tokenizer):
        """Test that creating dataloader without loading raises error."""
        handler = DatasetHandler(tokenizer)

        with pytest.raises(ValueError, match="No dataset loaded"):
            handler.create_dataloader()

    def test_handler_create_dataloader_invalid_type(self, tokenizer):
        """Test that invalid collate_type raises error."""
        handler = DatasetHandler(tokenizer)
        handler.load_texts(["Hello"])

        with pytest.raises(ValueError, match="Unknown collate_type"):
            handler.create_dataloader(collate_type="invalid")

    def test_handler_method_chaining(self, tokenizer):
        """Test that load methods return self for chaining."""
        handler = DatasetHandler(tokenizer)
        result = handler.load_texts(["Hello"])
        assert result is handler

    def test_handler_repr(self, tokenizer):
        """Test string representation."""
        handler = DatasetHandler(tokenizer)
        handler.load_texts(["Hello", "World"])

        repr_str = repr(handler)
        assert "DatasetHandler" in repr_str
        assert "num_samples=2" in repr_str

    def test_handler_repr_no_dataset(self, tokenizer):
        """Test repr when no dataset loaded."""
        handler = DatasetHandler(tokenizer)
        repr_str = repr(handler)
        assert "no dataset loaded" in repr_str


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    def test_full_pipeline(self, tokenizer):
        """Test full pipeline from texts to batches."""
        # Create handler
        handler = DatasetHandler(
            tokenizer,
            max_length=64,
            batch_size=2,
            num_workers=0,  # Avoid multiprocessing in tests
        )

        # Load sample texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language.",
            "Neural networks are inspired by the human brain.",
        ]
        handler.load_texts(texts)

        # Create dataloader
        loader = handler.create_dataloader(collate_type="causal")

        # Iterate through batches
        total_batches = 0
        for batch in loader:
            assert batch["input_ids"].shape[0] <= 2
            assert batch["labels"].shape == batch["input_ids"].shape
            total_batches += 1

        assert total_batches == 2  # 4 samples / batch_size 2

    def test_belief_pipeline(self, tokenizer):
        """Test belief vector pipeline."""
        handler = DatasetHandler(
            tokenizer,
            max_length=64,
            batch_size=2,
            num_workers=0,
        )

        handler.load_texts([
            "Sample text for belief vectors.",
            "Another sample for testing.",
        ])

        loader = handler.create_dataloader(collate_type="belief", keep_last_k=8)

        batch = next(iter(loader))
        assert batch["select_mask"].any()  # Should have some selected positions

