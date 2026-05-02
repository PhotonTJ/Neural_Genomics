"""
=============================================================================
UNIFIED EVALUATION PIPELINE — PART 2: BENCHMARK INTEGRATION
=============================================================================
Downloads, processes, and saves benchmark prompts into standardized JSONL.

Benchmarks:
  SAFETY:
    - HarmBench  (walledai/HarmBench)       — 320 harmful behaviors
    - AdvBench   (walledai/AdvBench)         — 520 harmful instructions
    - XSTest     (Paul/XSTest)              — 250 safe + 200 unsafe prompts

  INSTRUCTION:
    - IFEval     (google/IFEval)            — ~500 verifiable instruction prompts
    - MT-Bench   (HuggingFace/mt_bench)     — 80 multi-turn judge prompts

Output format (standardized JSONL per benchmark):
    {"id": 0, "prompt": "...", "benchmark": "harmbench", "category": "...", "source": "..."}

Usage:
    python benchmark_integration.py                  # download all benchmarks
    python benchmark_integration.py --benchmark harmbench  # specific one
    python benchmark_integration.py --list           # show what's available

After running this, the benchmark_data/ folder will contain:
    benchmark_data/safety/harmbench.jsonl
    benchmark_data/safety/advbench.jsonl
    benchmark_data/safety/xstest_safe.jsonl
    benchmark_data/safety/xstest_unsafe.jsonl
    benchmark_data/instruction/ifeval.jsonl
    benchmark_data/instruction/mt_bench.jsonl

These files are then fed into the generation + evaluation pipeline.
=============================================================================
"""

import json
import os
import argparse
import logging
from typing import List, Dict, Any, Optional

# Lazy import — only needed when actually downloading
datasets_lib = None

def _ensure_datasets():
    """Lazy import of HuggingFace datasets library."""
    global datasets_lib
    if datasets_lib is None:
        try:
            import datasets as ds
            datasets_lib = ds
        except ImportError:
            raise ImportError(
                "HuggingFace 'datasets' library required. "
                "Install with: pip install datasets"
            )
    return datasets_lib


# Import from Part 1
from evaluation_pipeline_part1 import CONFIG, write_jsonl


# =============================================================================
# OUTPUT PATHS
# =============================================================================

SAFETY_DIR = CONFIG["safety"]["input_folder"]
INSTRUCTION_DIR = CONFIG["instruction"]["input_folder"]

os.makedirs(SAFETY_DIR, exist_ok=True)
os.makedirs(INSTRUCTION_DIR, exist_ok=True)

BENCHMARK_PATHS = {
    # Safety
    "harmbench":      os.path.join(SAFETY_DIR, "harmbench.jsonl"),
    "advbench":       os.path.join(SAFETY_DIR, "advbench.jsonl"),
    "xstest_safe":    os.path.join(SAFETY_DIR, "xstest_safe.jsonl"),
    "xstest_unsafe":  os.path.join(SAFETY_DIR, "xstest_unsafe.jsonl"),
    # Instruction
    "ifeval":         os.path.join(INSTRUCTION_DIR, "ifeval.jsonl"),
    "mt_bench":       os.path.join(INSTRUCTION_DIR, "mt_bench.jsonl"),
}


# =============================================================================
# 1. HARMBENCH
# =============================================================================

def download_harmbench(sample_size: Optional[int] = None) -> str:
    """
    Download HarmBench behaviors (direct requests).
    Source: walledai/HarmBench on HuggingFace
    ~320 harmful behavior prompts.
    """
    ds = _ensure_datasets()
    output_path = BENCHMARK_PATHS["harmbench"]

    print(f"\n  Downloading HarmBench...")
    try:
        dataset = ds.load_dataset("walledai/HarmBench", split="test")
    except Exception:
        # Fallback to swiss-ai version
        print("  Trying fallback source (swiss-ai/harmbench)...")
        dataset = ds.load_dataset("swiss-ai/harmbench", "DirectRequest", split="test")

    records = []
    for idx, row in enumerate(dataset):
        # walledai/HarmBench has 'prompt' and 'category' fields
        prompt = row.get("prompt", row.get("Behavior", row.get("goal", "")))
        category = row.get("category", row.get("SemanticCategory", "unknown"))

        if not prompt or not prompt.strip():
            continue

        records.append({
            "id": idx,
            "prompt": prompt.strip(),
            "benchmark": "harmbench",
            "category": str(category),
            "source": "walledai/HarmBench",
        })

    if sample_size:
        records = records[:sample_size]

    write_jsonl(records, output_path)
    print(f"  HarmBench: {len(records)} prompts → {output_path}")
    return output_path


# =============================================================================
# 2. ADVBENCH
# =============================================================================

def download_advbench(sample_size: Optional[int] = None) -> str:
    """
    Download AdvBench harmful instructions.
    Source: walledai/AdvBench on HuggingFace
    520 harmful instructions.
    """
    ds = _ensure_datasets()
    output_path = BENCHMARK_PATHS["advbench"]

    print(f"\n  Downloading AdvBench...")
    dataset = None

    # Try multiple sources in order of preference
    # walledai/AdvBench is gated, so try JailbreakBench first
    sources = [
        ("JailbreakBench/JBB-Behaviors", {"name": "behaviors", "split": "harmful"}),
        ("walledai/AdvBench", {"split": "train"}),
    ]

    for source_name, kwargs in sources:
        try:
            print(f"  Trying: {source_name}...")
            dataset = ds.load_dataset(source_name, **kwargs)
            print(f"  Loaded from: {source_name}")
            break
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    if dataset is None:
        print("  ERROR: Could not download AdvBench from any source.")
        print("  You may need to: huggingface-cli login")
        return output_path

    records = []
    for idx, row in enumerate(dataset):
        prompt = row.get("prompt", row.get("goal", row.get("Goal", "")))
        category = row.get("category", row.get("Category", "harmful_instruction"))

        if not prompt or not prompt.strip():
            continue

        records.append({
            "id": idx,
            "prompt": prompt.strip(),
            "benchmark": "advbench",
            "category": str(category),
            "source": "walledai/AdvBench",
        })

    if sample_size:
        records = records[:sample_size]

    write_jsonl(records, output_path)
    print(f"  AdvBench: {len(records)} prompts → {output_path}")
    return output_path


# =============================================================================
# 3. XSTEST (Safe + Unsafe splits)
# =============================================================================

def download_xstest(sample_size: Optional[int] = None) -> Dict[str, str]:
    """
    Download XSTest prompts — both safe and unsafe subsets.
    Source: Paul/XSTest on HuggingFace
    250 safe prompts + 200 unsafe prompts.

    Important for your paper: XSTest measures OVER-refusal.
    A well-aligned DPO model should:
      - Refuse unsafe prompts (high refusal score)
      - NOT refuse safe prompts (low false-refusal rate)
    """
    ds = _ensure_datasets()
    output_safe = BENCHMARK_PATHS["xstest_safe"]
    output_unsafe = BENCHMARK_PATHS["xstest_unsafe"]

    print(f"\n  Downloading XSTest...")
    dataset = ds.load_dataset("Paul/XSTest", split="train")

    safe_records = []
    unsafe_records = []

    for idx, row in enumerate(dataset):
        prompt = row.get("prompt", "")
        prompt_type = row.get("type", "unknown")
        label = row.get("label", "")
        note = row.get("note", "")

        if not prompt or not prompt.strip():
            continue

        record = {
            "id": idx,
            "prompt": prompt.strip(),
            "benchmark": "xstest",
            "category": prompt_type,
            "source": "Paul/XSTest",
            "label": label,           # safe or unsafe
        }

        # XSTest labels: "safe" prompts should NOT be refused
        # "unsafe" prompts SHOULD be refused
        if "unsafe" in str(label).lower() or "contrast" not in str(prompt_type).lower():
            # The actual unsafe prompts in XSTest
            if "unsafe" in str(label).lower():
                unsafe_records.append(record)
            else:
                safe_records.append(record)
        else:
            safe_records.append(record)

    # Re-sort by label more reliably if the above heuristic isn't clean
    # XSTest structure: types starting with "safe_" are safe, others are unsafe contrasts
    # Let's use the label field if available
    all_records_safe = []
    all_records_unsafe = []
    for idx, row in enumerate(dataset):
        prompt = row.get("prompt", "")
        prompt_type = row.get("type", "unknown")
        label = row.get("label", "unknown")

        if not prompt or not prompt.strip():
            continue

        record = {
            "id": idx,
            "prompt": prompt.strip(),
            "benchmark": "xstest",
            "category": prompt_type,
            "source": "Paul/XSTest",
            "label": str(label),
        }

        # Paul/XSTest uses 'label' field: "safe" or "unsafe"
        if str(label).lower().strip() == "unsafe":
            all_records_unsafe.append(record)
        else:
            all_records_safe.append(record)

    if sample_size:
        all_records_safe = all_records_safe[:sample_size]
        all_records_unsafe = all_records_unsafe[:sample_size]

    write_jsonl(all_records_safe, output_safe)
    write_jsonl(all_records_unsafe, output_unsafe)
    print(f"  XSTest Safe:   {len(all_records_safe)} prompts → {output_safe}")
    print(f"  XSTest Unsafe: {len(all_records_unsafe)} prompts → {output_unsafe}")
    return {"safe": output_safe, "unsafe": output_unsafe}


# =============================================================================
# 4. IFEVAL
# =============================================================================

def download_ifeval(sample_size: Optional[int] = None) -> str:
    """
    Download IFEval (Instruction-Following Evaluation) prompts.
    Source: google/IFEval on HuggingFace
    ~541 prompts with verifiable instruction constraints.

    Each prompt contains one or more verifiable instructions like
    "write in more than 400 words" or "mention the keyword AI at least 3 times".

    NOTE: IFEval has its own programmatic evaluator (Part 3 will integrate it).
    The LLM-judge metrics from Part 1 complement but don't replace it.
    """
    ds = _ensure_datasets()
    output_path = BENCHMARK_PATHS["ifeval"]

    print(f"\n  Downloading IFEval...")
    dataset = ds.load_dataset("google/IFEval", split="train")

    records = []
    for idx, row in enumerate(dataset):
        prompt = row.get("prompt", "")
        instruction_ids = row.get("instruction_id_list", [])
        kwargs = row.get("kwargs", [])

        if not prompt or not prompt.strip():
            continue

        records.append({
            "id": idx,
            "prompt": prompt.strip(),
            "benchmark": "ifeval",
            "category": ",".join(instruction_ids) if instruction_ids else "general",
            "source": "google/IFEval",
            # Preserve IFEval-specific fields for programmatic evaluation
            "instruction_id_list": instruction_ids,
            "kwargs": kwargs,
        })

    if sample_size:
        records = records[:sample_size]

    write_jsonl(records, output_path)
    print(f"  IFEval: {len(records)} prompts → {output_path}")
    return output_path


# =============================================================================
# 5. MT-BENCH
# =============================================================================

def download_mt_bench(sample_size: Optional[int] = None) -> str:
    """
    Download MT-Bench prompts.
    Source: HuggingFaceH4/mt_bench_prompts on HuggingFace
    80 multi-turn prompts across 8 categories.

    For our pipeline, we use FIRST-TURN prompts only (single-turn evaluation).
    Multi-turn evaluation would require a conversation loop, which is a
    separate concern.
    """
    ds = _ensure_datasets()
    output_path = BENCHMARK_PATHS["mt_bench"]

    print(f"\n  Downloading MT-Bench...")
    try:
        dataset = ds.load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    except Exception:
        try:
            print("  Trying alternative source...")
            dataset = ds.load_dataset("lmarena-ai/mt-bench-en", split="train")
        except Exception:
            print("  Trying LMSYS source...")
            dataset = ds.load_dataset("lmsys/mt_bench_human_judgments", split="human")

    records = []
    for idx, row in enumerate(dataset):
        # MT-Bench has 'prompt' as a list of turns: [turn1, turn2]
        # We take the first turn only for single-turn eval
        prompt_field = row.get("prompt", row.get("turns", []))

        if isinstance(prompt_field, list) and len(prompt_field) > 0:
            prompt = prompt_field[0]
        elif isinstance(prompt_field, str):
            prompt = prompt_field
        else:
            continue

        category = row.get("category", row.get("category", "general"))

        if not prompt or not prompt.strip():
            continue

        records.append({
            "id": idx,
            "prompt": prompt.strip(),
            "benchmark": "mt_bench",
            "category": str(category),
            "source": "HuggingFaceH4/mt_bench_prompts",
        })

    if sample_size:
        records = records[:sample_size]

    write_jsonl(records, output_path)
    print(f"  MT-Bench: {len(records)} prompts → {output_path}")
    return output_path


# =============================================================================
# DOWNLOAD ALL
# =============================================================================

def download_all_benchmarks(sample_size: Optional[int] = None) -> Dict[str, str]:
    """Download and prepare all benchmarks. Returns dict of paths."""
    print("=" * 60)
    print("BENCHMARK DOWNLOAD & PREPARATION")
    print("=" * 60)

    paths = {}
    errors = []

    # Safety benchmarks
    for name, func in [
        ("harmbench", download_harmbench),
        ("advbench", download_advbench),
        ("xstest", download_xstest),
    ]:
        try:
            result = func(sample_size=sample_size)
            if isinstance(result, dict):
                paths.update(result)
            else:
                paths[name] = result
        except Exception as e:
            errors.append((name, str(e)))
            print(f"  ERROR downloading {name}: {e}")
            import traceback
            traceback.print_exc()

    # Instruction benchmarks
    for name, func in [
        ("ifeval", download_ifeval),
        ("mt_bench", download_mt_bench),
    ]:
        try:
            paths[name] = func(sample_size=sample_size)
        except Exception as e:
            errors.append((name, str(e)))
            print(f"  ERROR downloading {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for name, path in paths.items():
        if os.path.exists(path):
            # Count lines
            with open(path) as f:
                count = sum(1 for _ in f)
            print(f"  ✅ {name:20s} → {count:5d} prompts  ({path})")
        else:
            print(f"  ❌ {name:20s} → FILE NOT FOUND")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for name, err in errors:
            print(f"    ❌ {name}: {err}")

    print(f"{'='*60}\n")
    return paths


# =============================================================================
# VERIFY BENCHMARKS (check what's already downloaded)
# =============================================================================

def verify_benchmarks() -> Dict[str, Dict[str, Any]]:
    """Check which benchmarks are already downloaded and their stats."""
    print("=" * 60)
    print("BENCHMARK STATUS")
    print("=" * 60)

    status = {}
    for name, path in BENCHMARK_PATHS.items():
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
            count = len(lines)

            # Peek at first record
            if lines:
                first = json.loads(lines[0])
                benchmark = first.get("benchmark", "unknown")
                category_sample = first.get("category", "N/A")
            else:
                benchmark = "empty"
                category_sample = "N/A"

            status[name] = {
                "exists": True,
                "count": count,
                "path": path,
                "benchmark": benchmark,
            }
            print(f"  ✅ {name:20s}  {count:5d} prompts  ({path})")
        else:
            status[name] = {"exists": False, "count": 0, "path": path}
            print(f"  ❌ {name:20s}  NOT FOUND         ({path})")

    print(f"{'='*60}\n")
    return status


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare evaluation benchmarks."
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        choices=["harmbench", "advbench", "xstest", "ifeval", "mt_bench", "all"],
        help="Which benchmark to download (default: all)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Limit number of prompts per benchmark (for testing)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available benchmarks and their status"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Check which benchmarks are already downloaded"
    )
    args = parser.parse_args()

    if args.list or args.verify:
        verify_benchmarks()
        return

    if args.benchmark is None or args.benchmark == "all":
        download_all_benchmarks(sample_size=args.sample_size)
    else:
        func_map = {
            "harmbench": download_harmbench,
            "advbench": download_advbench,
            "xstest": download_xstest,
            "ifeval": download_ifeval,
            "mt_bench": download_mt_bench,
        }
        func = func_map[args.benchmark]
        func(sample_size=args.sample_size)

    # Show final status
    print("\nFinal status:")
    verify_benchmarks()


if __name__ == "__main__":
    main()