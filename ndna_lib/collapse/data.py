# ndna_lib/collapse/data.py
"""
Data utilities for nDNA collapse experiments.

Includes:
  - Unified dataset loading: load_text_dataset() for any HuggingFace dataset
  - Dataset configuration registry: automatic config handling for datasets that require it
  - Custom formatters: special formatting for Alpaca, SQuAD, and other structured datasets
  - Tokenization: build_lm_dataset_from_texts() for preparing causal LM training data
  - Extensible registries: easily add new datasets via dataset_configs and dataset_formatters
"""

from typing import List, Dict
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase


dataset_id = 'Salesforce/wikitext'  
# Other options -->
# 'yahma/alpaca-cleaned' 
# 'rajpurkar/squad' 
# 'Anthropic/hh-rlhf' 
# 'wikimedia/wikipedia' 
# 'math-ai/AutoMathText' 
# 'hugfaceguy0001/stanford_plato'
# 'SetFit/ag_news'

# Configuration registry for datasets requiring config parameter
dataset_configs = {
    'Salesforce/wikitext': 'wikitext-103-v1',
    'wikimedia/wikipedia': '20231101.en',
    'math-ai/AutoMathText': 'web'
    # more can be added here as needed
}


# Custom formatters for datasets requiring special text combination
def format_alpaca(ex: Dict) -> str:
    instruction = ex.get("instruction", "").strip()
    inp = ex.get("input", "").strip()
    return f"Instruction: {instruction}\nInput: {inp}\nResponse:"

def format_squad(ex: Dict) -> str:
    context = ex.get("context", "").strip()
    question = ex.get("question", "").strip()
    return f"Context: {context}\nQuestion: {question}"


# Formatter registry for datasets with special formatting needs
dataset_formatters = {
    'yahma/alpaca-cleaned': format_alpaca,
    'rajpurkar/squad': format_squad,
    # more can be added here as needed
}


def load_text_dataset(
    dataset_id='Salesforce/wikitext', 
    max_samples=None, 
    split='train', 
    config=None, 
    preferred_text_cols = None,
) -> Dataset:
    """
    Unified loader for any HuggingFace text dataset.
    
    Automatically handles:
    - Config parameter for datasets that require it (via dataset_configs registry)
    - Custom formatting for structured datasets (via dataset_formatters registry)
    - Fallback to preferred column extraction for standard datasets
    
    Args:
        dataset_id: HuggingFace dataset name (e.g., "yahma/alpaca-cleaned", "Salesforce/wikitext")
        max_samples: Optional limit on number of samples
        split: Dataset split (default: "train")
        config: Config name (optional, uses default from registry if needed)
        preferred_text_cols: Column names to try for text extraction (in priority order)
    
    Returns:
        Dataset with single 'text' column
    
    Examples:
        # Simple dataset (no config needed)
        ds = load_text_dataset("yahma/alpaca-cleaned", max_samples=1000)
        
        # Dataset with auto-config
        ds = load_text_dataset("Salesforce/wikitext", max_samples=1000)
        
        # Override default config
        ds = load_text_dataset("Salesforce/wikitext", config="wikitext-2-v1", max_samples=1000)
        
        # Custom preferred columns
        ds = load_text_dataset("some/dataset", preferred_text_cols=['content', 'body'])
    """
    if preferred_text_cols is None:
        preferred_text_cols = ['text','content','message','prompt','input','output','question','main_text','context', 'chosen','problem']
    
    if config is None and dataset_id in dataset_configs:
        config = dataset_configs[dataset_id]
    
    if config is not None:
        dataset = load_dataset(dataset_id, config, split=split)
    else:
        dataset = load_dataset(dataset_id, split=split)
    
    if dataset_id in dataset_formatters:
        custom_formatter = dataset_formatters[dataset_id]
    else:
        custom_formatter = None
        
    if custom_formatter is not None:
        def _map_fn(ex):
            return {"text": custom_formatter(ex)}
        dataset = dataset.map(_map_fn, remove_columns=dataset.column_names)
    else:
        available_cols = [col for col in preferred_text_cols if col in dataset.column_names]
        if not available_cols:
            raise ValueError(
                f"None of the preferred columns {preferred_text_cols} found in dataset. "
                f"Available columns: {dataset.column_names}"
            )
        
        first_col = available_cols[0]
        if first_col != 'text':
            dataset = dataset.rename_column(first_col, 'text')
        dataset = dataset.select_columns(['text'])
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return dataset


def build_lm_dataset_from_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    max_length: int,
) -> Dataset:
    """
    Tokenize plain texts for causal language modeling.
    
    Converts raw text strings into tokenized inputs (input_ids, attention_mask).
    Note: Labels are NOT created here - use DataCollatorForLanguageModeling during training.
    
    Args:
        tokenizer: HuggingFace tokenizer
        texts: List of text strings to tokenize
        max_length: Maximum sequence length (truncation applied)
    
    Returns:
        Dataset with tokenized inputs in PyTorch format
        Columns: ['input_ids', 'attention_mask']
    """
    ds = Dataset.from_dict({"text": texts})

    def tok_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds_tok.set_format(type="torch")
    return ds_tok
