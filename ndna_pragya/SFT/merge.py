import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapter(base_model_path, adapter_dir, output_dir, dtype="float16"):

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    torch_dtype = dtype_map[dtype]

    print("=" * 60)
    print("Merging LoRA adapter")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load tokenizer FROM ADAPTER (important)
    # ---------------------------------------------------------
    print(f"Loading tokenizer from adapter: {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    if tokenizer.pad_token is None:
        if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    tokenizer.padding_side = "right"

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # ---------------------------------------------------------
    # Load base model
    # ---------------------------------------------------------
    print(f"Loading base model: {base_model_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    base_vocab = base_model.get_input_embeddings().weight.shape[0]
    print(f"Base model vocab size: {base_vocab}")

    # ---------------------------------------------------------
    # Resize embeddings if tokenizer is larger
    # ---------------------------------------------------------
    if len(tokenizer) > base_vocab:
        print(f"Resizing embeddings: {base_vocab} → {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))

    # ---------------------------------------------------------
    # Load adapter
    # ---------------------------------------------------------
    print(f"Loading adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    # ---------------------------------------------------------
    # Merge
    # ---------------------------------------------------------
    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    print(f"Saving merged model to: {output_dir}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Merge complete.")
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        required=True,
        help="HF model name OR local model directory",
    )

    parser.add_argument(
        "--adapter",
        required=True,
        help="Adapter directory",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for merged model",
    )

    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )

    args = parser.parse_args()

    merge_adapter(
        args.base_model,
        args.adapter,
        args.output,
        args.dtype,
    )