"""
=============================================================================
RESPONSE GENERATION RUNNER — Sequential single-GPU workflow
=============================================================================
Usage:
    # Terminal 1: serve the model
    python server.py --model meta-llama/Llama-2-7b-hf

    # Terminal 2: generate responses
    python run_generation.py --model base --prompts benchmark_data/safety/harmbench_prompts.jsonl
    python run_generation.py --model base --prompts benchmark_data/instruction/ifeval_prompts.jsonl

    # Then switch server to SFT model, and repeat:
    python run_generation.py --model sft --prompts benchmark_data/safety/harmbench_prompts.jsonl
    python run_generation.py --model sft --prompts benchmark_data/instruction/ifeval_prompts.jsonl

    # Then switch to DPO model:
    python run_generation.py --model dpo --prompts ...

    # Or generate for ALL prompt files at once:
    python run_generation.py --model base --all

Resume-safe: re-running the same command will skip already-generated responses.
=============================================================================
"""

import argparse
import glob
import os
from evaluation_pipeline_part1 import (
    CONFIG,
    generate_responses,
    print_config_summary,
    get_model_keys,
    get_model_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate responses from one model for benchmark prompts."
    )
    valid_keys = get_model_keys()
    parser.add_argument(
        "--model", required=True, choices=valid_keys,
        help=f"Which model to generate from. Options: {valid_keys}"
    )
    parser.add_argument(
        "--prompts", type=str, default=None,
        help="Path to a specific prompts JSONL file"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate for ALL benchmark prompt files found in benchmark_data/"
    )
    parser.add_argument(
        "--pipeline", type=str, default="both",
        choices=["safety", "instruction", "both"],
        help="Which benchmark set to generate for (default: both)"
    )
    return parser.parse_args()


def find_all_prompt_files(pipeline: str = "both") -> list:
    """Find all benchmark prompt JSONL files, optionally filtered by pipeline."""
    files = []
    folders = []
    if pipeline in ["safety", "both"]:
        folders.append(CONFIG["safety"]["input_folder"])
    if pipeline in ["instruction", "both"]:
        folders.append(CONFIG["instruction"]["input_folder"])

    for folder in folders:
        if os.path.exists(folder):
            files.extend(glob.glob(os.path.join(folder, "*_prompts.jsonl")))
            files.extend(glob.glob(os.path.join(folder, "*.jsonl")))
    # Deduplicate
    return sorted(set(files))


def get_output_path(prompts_path: str) -> str:
    """
    Map a prompts file to its generated responses file.
    benchmark_data/safety/harmbench_prompts.jsonl
        → generated_responses/harmbench_responses.jsonl
    """
    basename = os.path.basename(prompts_path)
    # Remove _prompts suffix if present
    name = basename.replace("_prompts.jsonl", "").replace(".jsonl", "")
    output_name = f"{name}_responses.jsonl"
    return os.path.join(CONFIG["generation"]["output_folder"], output_name)


def main():
    args = parse_args()
    print_config_summary()

    model_key = args.model
    model_cfg = get_model_config(model_key)

    print(f"\nGenerating responses for model: {model_key} ({model_cfg['name']})")
    print(f"Make sure server.py is running with: --model {model_cfg['name']}\n")

    if args.prompts:
        # Single file
        if not os.path.exists(args.prompts):
            print(f"ERROR: Prompts file not found: {args.prompts}")
            return
        output_path = get_output_path(args.prompts)
        generate_responses(
            prompts_path=args.prompts,
            output_path=output_path,
            model_key=model_key,
        )

    elif args.all:
        # All benchmark files (filtered by pipeline)
        prompt_files = find_all_prompt_files(pipeline=args.pipeline)
        if not prompt_files:
            print("No prompt files found in benchmark_data/. Run Part 2 first to download benchmarks.")
            return

        print(f"Found {len(prompt_files)} prompt files (pipeline={args.pipeline}):")
        for f in prompt_files:
            print(f"  {f}")
        print()

        for prompts_path in prompt_files:
            output_path = get_output_path(prompts_path)
            generate_responses(
                prompts_path=prompts_path,
                output_path=output_path,
                model_key=model_key,
            )
            print()

    else:
        print("ERROR: Specify either --prompts <file> or --all")
        print("Examples:")
        print(f"  python run_generation.py --model {model_key} --prompts benchmark_data/safety/harmbench.jsonl")
        print(f"  python run_generation.py --model {model_key} --all")
        print(f"  python run_generation.py --model {model_key} --all --pipeline safety")


if __name__ == "__main__":
    main()