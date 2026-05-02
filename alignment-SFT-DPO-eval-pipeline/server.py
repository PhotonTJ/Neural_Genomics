"""
=============================================================================
vLLM SERVER — Flexible launcher for any model
=============================================================================
Usage (sequential, single-GPU workflow):

  Phase 1 — Generate responses (one model at a time):
    python server.py --model meta-llama/Llama-2-7b-hf
    python server.py --model your-org/llama2-7b-sft-openhermes
    python server.py --model your-org/llama2-7b-sft-dpo-hhrlhf

  Phase 2 — Run evaluation (judge model):
    python server.py --model Qwen/Qwen2.5-32B-Instruct

  Optional flags:
    --port 8000              (default: 8000)
    --gpu-util 0.90          (default: 0.90)
    --quantization none      (default: bitsandbytes)
    --dtype half             (default: half)
    --max-model-len 4096     (default: auto)
    --tensor-parallel-size 1 (default: 1, increase for multi-GPU)

  Keep the terminal running. Ctrl+C to stop and switch models.
=============================================================================
"""

import subprocess
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a vLLM OpenAI-compatible server for any model."
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID or local path (e.g. meta-llama/Llama-2-7b-hf)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to serve on (default: 8000)"
    )
    parser.add_argument(
        "--gpu-util", type=float, default=0.90,
        help="GPU memory utilization (0.0-1.0, default: 0.90)"
    )
    parser.add_argument(
        "--quantization", type=str, default="bitsandbytes",
        choices=["bitsandbytes", "awq", "gptq", "squeezellm", "none"],
        help="Quantization method (default: bitsandbytes, use 'none' to disable)"
    )
    parser.add_argument(
        "--dtype", type=str, default="half",
        choices=["half", "float16", "bfloat16", "float32", "auto"],
        help="Data type (default: half)"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Max model context length (default: auto from model config)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--api-key", type=str, default="EMPTY",
        help="API key for the server (default: EMPTY)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build vLLM command
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_util),
        "--port", str(args.port),
        "--api-key", args.api_key,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]

    # Add quantization (skip if 'none')
    if args.quantization != "none":
        command.extend([
            "--quantization", args.quantization,
            "--load-format", args.quantization,
        ])

    # Add max model length if specified
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])

    # Print launch info
    print("=" * 60)
    print("vLLM SERVER")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Port:           {args.port}")
    print(f"  Quantization:   {args.quantization}")
    print(f"  Dtype:          {args.dtype}")
    print(f"  GPU Util:       {args.gpu_util}")
    print(f"  Tensor Parallel:{args.tensor_parallel_size}")
    if args.max_model_len:
        print(f"  Max Model Len:  {args.max_model_len}")
    print(f"  API Key:        {args.api_key}")
    print("=" * 60)
    print()
    print("KEEP THIS TERMINAL RUNNING!")
    print("Ctrl+C to stop the server and switch models.")
    print()

    # Launch
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream logs
    try:
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
    except KeyboardInterrupt:
        print("\nStopping server...")
        process.terminate()
        process.wait(timeout=10)
        print("Server stopped.")


if __name__ == "__main__":
    main()
