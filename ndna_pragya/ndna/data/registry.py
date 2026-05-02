"""
ndna.data.registry

Dataset registry for common HuggingFace datasets.
Maps dataset names to their configurations and text processors.
"""

from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------
# Dataset Registry
# ---------------------------------------------------------------------
# Maps dataset name to configuration
# Each entry can have:
#   - hf_name: HuggingFace dataset identifier
#   - config: Optional dataset config/subset name
#   - split: Dataset split to use
#   - text_column: Column name if text is directly available
#   - processor: Function to convert example to {"text": str}

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Question Answering
    "squad": {
        "hf_name": "squad",
        "split": "train",
        "processor": lambda ex: {
            "text": f"Question: {ex['question']}\nContext: {ex['context']}\nAnswer:"
        },
    },
    "triviaqa": {
        "hf_name": "trivia_qa",
        "config": "unfiltered",
        "split": "train",
        "processor": lambda ex: {
            "text": f"Question: {ex['question']}\nAnswer: {ex['answer']['value']}"
        },
    },
    
    # Text Classification
    "imdb": {
        "hf_name": "imdb",
        "split": "train",
        "text_column": "text",
    },
    "ag_news": {
        "hf_name": "ag_news",
        "split": "train",
        "text_column": "text",
    },
    "yelp": {
        "hf_name": "yelp_review_full",
        "split": "train",
        "text_column": "text",
    },
    "amazon": {
        "hf_name": "amazon_polarity",
        "split": "train",
        "text_column": "content",
    },
    
    # Language Modeling
    "wikitext": {
        "hf_name": "wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "train",
        "text_column": "text",
    },
    "wikitext_103": {
        "hf_name": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "text_column": "text",
    },
    
    # Instruction Following
    "alpaca": {
        "hf_name": "tatsu-lab/alpaca",
        "split": "train",
        "processor": lambda ex: {
            "text": f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        },
    },
    "dolly": {
        "hf_name": "databricks/databricks-dolly-15k",
        "split": "train",
        "processor": lambda ex: {
            "text": f"Instruction: {ex['instruction']}\nContext: {ex['context']}\nResponse: {ex['response']}"
        },
    },
    
    # Conversations
    "openassistant": {
        "hf_name": "OpenAssistant/oasst1",
        "split": "train",
        "text_column": "text",
    },
    
    # Code
    "code_search_net": {
        "hf_name": "code_search_net",
        "config": "python",
        "split": "train",
        "processor": lambda ex: {"text": ex["func_code_string"]},
    },
    
    # Scientific
    "arxiv": {
        "hf_name": "ccdv/arxiv-summarization",
        "split": "train",
        "text_column": "abstract",
    },
    
    # Multi-task
    "flan": {
        "hf_name": "Muennighoff/flan",
        "split": "train",
        "processor": lambda ex: {"text": f"{ex['inputs']}\n{ex['targets']}"},
    },
}


def get_dataset_config(name: str) -> Dict[str, Any]:
    """
    Get configuration for a dataset by name.

    Args:
        name: Dataset name (key in DATASET_REGISTRY)

    Returns:
        Dict with dataset configuration

    Raises:
        ValueError: If dataset not in registry
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset {name!r} not in registry. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]


def list_datasets() -> list:
    """List all available dataset names in the registry."""
    return list(DATASET_REGISTRY.keys())


def make_text_processor(
    text_column: Optional[str] = None,
    processor: Optional[Callable] = None,
) -> Callable:
    """
    Create a text processor function.

    Args:
        text_column: Column name containing text (simple extraction)
        processor: Custom processor function

    Returns:
        Function that converts example to {"text": str}
    """
    if processor is not None:
        return processor
    elif text_column is not None:
        return lambda ex: {"text": ex[text_column]}
    else:
        # Default: assume "text" column exists
        return lambda ex: {"text": ex["text"]}

