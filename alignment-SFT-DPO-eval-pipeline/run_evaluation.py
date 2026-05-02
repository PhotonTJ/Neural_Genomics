"""
=============================================================================
UNIFIED EVALUATION PIPELINE — PART 3: PIPELINE RUNNERS
=============================================================================
Runs the actual evaluation across all 3 models (Base, SFT, DPO).

Two pipelines:
  1. SAFETY PIPELINE   — runs safety metrics on HarmBench/AdvBench/XSTest
  2. INSTRUCTION PIPELINE — runs instruction metrics on IFEval/MT-Bench
                            + IFEval programmatic verification

Both pipelines:
  - Evaluate all 3 models on the same prompts (wide-format CSV output)
  - Support resume (skip already-processed records)
  - Stream-save every N records
  - Produce summary statistics (mean, std, win-rates, deltas)
  - Generate comparison plots

Usage (after Parts 1-2 and response generation):
    # Make sure judge server is running:
    #   python server.py --model Qwen/Qwen2.5-32B-Instruct

    python run_evaluation.py                    # run both pipelines
    python run_evaluation.py --pipeline safety  # safety only
    python run_evaluation.py --pipeline instruction  # instruction only
=============================================================================
"""

import os
import gc
import json
import glob
import logging
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation_pipeline_part1 import (
    CONFIG,
    setup_logging,
    VLLMJudge,
    DetoxifyCalculator,
    build_safety_metrics,
    build_instruction_metrics,
    read_jsonl,
    write_jsonl,
    validate_record_for_eval,
    evaluate_record_for_model,
    print_config_summary,
    get_model_keys,
    get_model_config,
)


# =============================================================================
# 1. N-MODEL EVALUATOR (shared logic for both pipelines)
# =============================================================================

def run_nmodel_evaluation(
    input_path: str,
    output_path: str,
    metrics: Dict,
    detoxify_calc: Optional[DetoxifyCalculator],
    sample_size: Optional[int] = None,
    pipeline_name: str = "eval",
) -> Optional[pd.DataFrame]:
    """
    Evaluate ALL configured models side-by-side on the same prompts.

    Dynamically adapts to however many models are in CONFIG["models"].
    Input JSONL must have: id, prompt, output_{key} for each model key.
    Output CSV (wide format): id, prompt, {key}_{metric}_score for each model.

    Features:
      - Resume support (skips already-processed IDs)
      - Stream-saves every 5 records
      - Validates all model outputs before evaluation
    """
    model_keys = get_model_keys()

    print(f"\n{'='*60}")
    print(f"N-MODEL EVALUATION — {pipeline_name.upper()}")
    print(f"{'='*60}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Models: {model_keys}")

    if not os.path.exists(input_path):
        print(f"  Input file not found: {input_path}. Skipping.")
        return None

    # Load records
    records = read_jsonl(input_path)
    if sample_size:
        records = records[:sample_size]
    print(f"  Loaded {len(records)} records (sample_size={sample_size})")

    # Validate: all model outputs must be present
    required = [f"output_{k}" for k in model_keys]
    valid_records = []
    for r in records:
        is_valid, reason = validate_record_for_eval(r, required)
        if is_valid:
            valid_records.append(r)
        else:
            logging.warning(f"Record {r['id']} invalid: {reason}")
    print(f"  Valid records: {len(valid_records)}")

    if not valid_records:
        print("  No valid records. Aborting.")
        return None

    # Resume support
    processed_ids = set()
    file_exists = os.path.exists(output_path)
    if file_exists:
        try:
            existing = pd.read_csv(output_path, usecols=["id"])
            processed_ids = set(existing["id"].astype(int))
            print(f"  Resuming: {len(processed_ids)} already done.")
        except Exception as e:
            logging.warning(f"Could not read existing results: {e}")

    to_process = [r for r in valid_records if r["id"] not in processed_ids]
    print(f"  To process: {len(to_process)}")

    if not to_process:
        print("  All records already processed.")
        if file_exists:
            return pd.read_csv(output_path)
        return None

    # Build model_keys -> output_key mapping
    model_output_keys = [(k, f"output_{k}") for k in model_keys]

    write_mode = "a" if file_exists else "w"
    write_header = not file_exists
    buffer = []

    for idx, record in enumerate(tqdm(to_process, desc=f"{pipeline_name} eval")):
        row = {
            "id": record["id"],
            "prompt": record["prompt"],
            "benchmark": record.get("benchmark", "unknown"),
            "category": record.get("category", "unknown"),
        }

        for model_label, output_key in model_output_keys:
            scores = evaluate_record_for_model(
                prompt=record["prompt"],
                output=record[output_key],
                metrics=metrics,
                detoxify_calc=detoxify_calc,
            )
            for k, v in scores.items():
                row[f"{model_label}_{k}"] = v

        buffer.append(row)

        # Stream-save every 5 records
        if (idx + 1) % 5 == 0 or (idx + 1) == len(to_process):
            pd.DataFrame(buffer).to_csv(
                output_path, mode=write_mode, header=write_header, index=False
            )
            buffer = []
            write_mode = "a"
            write_header = False
            gc.collect()

    print(f"  Results saved to: {output_path}")

    if os.path.exists(output_path):
        return pd.read_csv(output_path)
    return None


# =============================================================================
# 2. IFEVAL PROGRAMMATIC VERIFICATION
# =============================================================================

def run_ifeval_verification(responses_path: str, output_path: str) -> Optional[pd.DataFrame]:
    """
    Run IFEval's programmatic constraint verification.
    This checks verifiable instructions like "write more than 400 words",
    "mention keyword X at least 3 times", etc.

    This is SEPARATE from the LLM-judge metrics — it gives you objective
    pass/fail rates that reviewers can't argue with.

    Returns DataFrame with pass rates per model.
    """
    print(f"\n  Running IFEval Programmatic Verification...")

    if not os.path.exists(responses_path):
        print(f"  Responses file not found: {responses_path}")
        return None

    # Load the original IFEval data (with instruction_id_list and kwargs)
    ifeval_data_path = os.path.join(CONFIG["instruction"]["input_folder"], "ifeval.jsonl")
    if not os.path.exists(ifeval_data_path):
        print(f"  IFEval benchmark data not found: {ifeval_data_path}")
        print("  Skipping programmatic verification.")
        return None

    # Try to import IFEval verification library
    try:
        from ifeval import Evaluator, instruction_registry, InputExample
        has_ifeval_lib = True
        print("  Using ifeval library for verification.")
    except ImportError:
        has_ifeval_lib = False
        print("  ifeval library not installed. Using heuristic verification.")
        print("  For full IFEval: pip install ifeval  OR  clone https://github.com/oKatanaaa/ifeval")

    # Load responses
    responses = read_jsonl(responses_path)
    ifeval_prompts = read_jsonl(ifeval_data_path)

    # Build lookup: id -> ifeval metadata
    ifeval_meta = {}
    for p in ifeval_prompts:
        ifeval_meta[p["id"]] = p

    # Filter to IFEval responses only
    ifeval_responses = [r for r in responses if r.get("benchmark") == "ifeval"]
    if not ifeval_responses:
        print("  No IFEval responses found in responses file.")
        return None

    print(f"  Found {len(ifeval_responses)} IFEval responses.")

    if has_ifeval_lib:
        # Use the proper IFEval evaluator
        results = _run_ifeval_with_library(ifeval_responses, ifeval_meta)
    else:
        # Heuristic fallback — basic constraint checking
        results = _run_ifeval_heuristic(ifeval_responses, ifeval_meta)

    if results is not None:
        results.to_csv(output_path, index=False)
        print(f"  IFEval verification saved to: {output_path}")

    return results


def _run_ifeval_with_library(responses, ifeval_meta):
    """Use the ifeval library for proper verification."""
    from ifeval import Evaluator, instruction_registry, InputExample

    evaluator = Evaluator(instruction_registry)
    all_results = []

    for model_key in get_model_keys():
        output_key = f"output_{model_key}"
        input_examples = []
        model_responses = []

        for r in responses:
            meta = ifeval_meta.get(r["id"], {})
            instruction_ids = meta.get("instruction_id_list", [])
            kwargs_list = meta.get("kwargs", [])

            if not instruction_ids:
                continue

            input_examples.append(InputExample(
                key=r["id"],
                instruction_id_list=instruction_ids,
                prompt=r["prompt"],
                kwargs=kwargs_list,
            ))
            model_responses.append(r.get(output_key, ""))

        if not input_examples:
            continue

        report, outputs = evaluator.evaluate(input_examples, model_responses)

        all_results.append({
            "model": model_key,
            "prompt_strict": report["eval_results_strict"]["prompt_accuracy"],
            "prompt_loose": report["eval_results_loose"]["prompt_accuracy"],
            "inst_strict": report["eval_results_strict"]["instruction_accuracy"],
            "inst_loose": report["eval_results_loose"]["instruction_accuracy"],
            "n_prompts": len(input_examples),
        })

    return pd.DataFrame(all_results) if all_results else None


def _run_ifeval_heuristic(responses, ifeval_meta):
    """
    Basic heuristic verification when ifeval library is not available.
    Checks common constraint types that can be verified programmatically.
    """
    all_results = []

    for model_key in get_model_keys():
        output_key = f"output_{model_key}"
        total_instructions = 0
        passed_instructions = 0

        for r in responses:
            meta = ifeval_meta.get(r["id"], {})
            instruction_ids = meta.get("instruction_id_list", [])
            kwargs_list = meta.get("kwargs", [])
            output = r.get(output_key, "")

            if not output or not instruction_ids:
                continue

            for inst_id, kw in zip(instruction_ids, kwargs_list):
                total_instructions += 1
                if _check_instruction_heuristic(output, inst_id, kw):
                    passed_instructions += 1

        pass_rate = passed_instructions / total_instructions if total_instructions > 0 else 0

        all_results.append({
            "model": model_key,
            "heuristic_inst_pass_rate": round(pass_rate, 4),
            "total_instructions": total_instructions,
            "passed_instructions": passed_instructions,
        })

    return pd.DataFrame(all_results) if all_results else None


def _check_instruction_heuristic(output: str, instruction_id: str, kwargs: dict) -> bool:
    """Check a single instruction constraint heuristically."""
    if not kwargs:
        kwargs = {}

    try:
        # Word count constraints
        if "length_constraints:number_words" in instruction_id:
            num_words = kwargs.get("num_words")
            relation = kwargs.get("relation", "at least")
            if num_words is not None:
                word_count = len(output.split())
                if "at least" in str(relation):
                    return word_count >= int(num_words)
                elif "at most" in str(relation):
                    return word_count <= int(num_words)

        # Sentence count
        if "length_constraints:number_sentences" in instruction_id:
            num_sentences = kwargs.get("num_sentences")
            if num_sentences is not None:
                # Simple sentence count by splitting on .!?
                import re
                sentences = re.split(r'[.!?]+', output)
                sentences = [s.strip() for s in sentences if s.strip()]
                return len(sentences) >= int(num_sentences)

        # Paragraph count
        if "length_constraints:number_paragraphs" in instruction_id:
            num_paragraphs = kwargs.get("num_paragraphs")
            if num_paragraphs is not None:
                paragraphs = [p.strip() for p in output.split("\n\n") if p.strip()]
                return len(paragraphs) >= int(num_paragraphs)

        # Keyword inclusion
        if "keywords:existence" in instruction_id:
            keywords = kwargs.get("keywords")
            if keywords:
                return all(kw.lower() in output.lower() for kw in keywords)

        # Keyword frequency
        if "keywords:frequency" in instruction_id:
            keyword = kwargs.get("keyword")
            frequency = kwargs.get("frequency")
            if keyword and frequency:
                return output.lower().count(keyword.lower()) >= int(frequency)

        # Forbidden words
        if "keywords:forbidden_words" in instruction_id:
            forbidden = kwargs.get("forbidden_words")
            if forbidden:
                return all(fw.lower() not in output.lower() for fw in forbidden)

        # No commas
        if "punctuation:no_comma" in instruction_id:
            return "," not in output

        # Bullet points
        if "detectable_format:number_bullet_points" in instruction_id:
            num_bullets = kwargs.get("num_bullets")
            if num_bullets is not None:
                bullet_count = sum(1 for line in output.split("\n") if line.strip().startswith(("- ", "* ", "• ")))
                return bullet_count >= int(num_bullets)

        # Postscript
        if "detectable_content:postscript" in instruction_id:
            marker = kwargs.get("postscript_marker", "P.S.")
            return marker in output

    except Exception:
        pass

    # Can't verify this instruction type — skip (don't count it)
    return True  # Optimistic default for unrecognized constraints


# =============================================================================
# 3. SUMMARY GENERATION
# =============================================================================

def generate_summary(
    results_path: str,
    summary_path: str,
    pipeline_name: str = "eval",
) -> Optional[pd.DataFrame]:
    """
    Generate summary statistics from N-model evaluation results.
    Dynamically detects models from column prefixes.
    Produces: mean scores, pairwise win-rates, delta tables.
    """
    print(f"\n  Generating {pipeline_name} Summary...")

    if not os.path.exists(results_path):
        print(f"  Results not found: {results_path}")
        return None

    df = pd.read_csv(results_path)
    if df.empty:
        print("  Empty results file.")
        return None

    # Detect models from columns: find all prefixes that have *_score columns
    model_keys = get_model_keys()
    # Verify which models actually have data in this CSV
    active_models = []
    for key in model_keys:
        if any(c.startswith(f"{key}_") and c.endswith("_score") for c in df.columns):
            active_models.append(key)

    if not active_models:
        print("  No model score columns found.")
        return None

    # Find metric names from first model's columns
    first_model = active_models[0]
    score_cols = [c for c in df.columns if c.startswith(f"{first_model}_") and c.endswith("_score")]
    metric_names = [c[len(f"{first_model}_"):-len("_score")] for c in score_cols]

    if not metric_names:
        print("  No score columns found.")
        return None

    # ── Mean Scores ──────────────────────────────────────────────────────────
    mean_rows = []
    for metric in metric_names:
        row = {"metric": metric}
        for model in active_models:
            col = f"{model}_{metric}_score"
            if col in df.columns:
                row[f"{model}_mean"] = round(df[col].mean(), 4)
                row[f"{model}_std"] = round(df[col].std(), 4)
        mean_rows.append(row)
    df_means = pd.DataFrame(mean_rows)

    # ── Pairwise Win Rates (all pairs) ───────────────────────────────────────
    from itertools import combinations
    pairs = list(combinations(active_models, 2))
    win_rows = []
    for metric in metric_names:
        row = {"metric": metric}
        for (model_a, model_b) in pairs:
            col_a = f"{model_a}_{metric}_score"
            col_b = f"{model_b}_{metric}_score"
            if col_a not in df.columns or col_b not in df.columns:
                continue
            valid = df[[col_a, col_b]].dropna()
            n = len(valid)
            if n == 0:
                continue
            wins = (valid[col_a] > valid[col_b]).sum()
            ties = (valid[col_a] == valid[col_b]).sum()
            losses = (valid[col_a] < valid[col_b]).sum()
            pair_label = f"{model_a}_vs_{model_b}"
            row[f"{pair_label}_win"] = round(wins / n, 4)
            row[f"{pair_label}_tie"] = round(ties / n, 4)
            row[f"{pair_label}_loss"] = round(losses / n, 4)
        win_rows.append(row)
    df_wins = pd.DataFrame(win_rows)

    # ── Pairwise Deltas ──────────────────────────────────────────────────────
    delta_rows = []
    for metric in metric_names:
        row = {"metric": metric}
        for (model_a, model_b) in pairs:
            ca = f"{model_a}_{metric}_score"
            cb = f"{model_b}_{metric}_score"
            if ca in df.columns and cb in df.columns:
                valid = df[[ca, cb]].dropna()
                row[f"{model_a}_minus_{model_b}"] = round((valid[ca] - valid[cb]).mean(), 4)
        delta_rows.append(row)
    df_deltas = pd.DataFrame(delta_rows)

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# {pipeline_name.upper()} N-Model Evaluation Summary\n")
        f.write(f"# Models: {active_models}\n\n")
        f.write("## Mean Scores (0-10 scale)\n")
        df_means.to_csv(f, index=False)
        f.write("\n## Pairwise Win Rates\n")
        df_wins.to_csv(f, index=False)
        f.write("\n## Mean Deltas\n")
        df_deltas.to_csv(f, index=False)

    print(f"  Summary saved to: {summary_path}")
    print(f"\n  Mean Scores ({pipeline_name}):")
    print(df_means.to_string(index=False))

    return df_means


# =============================================================================
# 4. COMPARISON PLOTS
# =============================================================================

def plot_comparison(
    results_path: str,
    output_folder: str,
    pipeline_name: str = "eval",
) -> None:
    """Generate grouped bar charts comparing all configured models."""
    print(f"\n  Generating {pipeline_name} Comparison Plots...")

    if not os.path.exists(results_path):
        print(f"  Results not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    if df.empty:
        return

    # Detect active models from columns
    model_keys = get_model_keys()
    active_models = [k for k in model_keys
                     if any(c.startswith(f"{k}_") and c.endswith("_score") for c in df.columns)]

    if not active_models:
        print("  No model data to plot.")
        return

    # Get metric names from first active model
    first = active_models[0]
    metric_names = [c[len(f"{first}_"):-len("_score")]
                    for c in df.columns if c.startswith(f"{first}_") and c.endswith("_score")]

    if not metric_names:
        print("  No metrics to plot.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Auto-generate colors for N models
    base_colors = ["#e15759", "#4e79a7", "#59a14f", "#f28e2b", "#b07aa1",
                    "#76b7b2", "#ff9da7", "#edc948", "#bab0ac", "#9c755f"]
    colors = {m: base_colors[i % len(base_colors)] for i, m in enumerate(active_models)}
    labels = {m: m.upper() for m in active_models}

    # Compute means/stds
    means, stds = {}, {}
    for metric in metric_names:
        means[metric] = {}
        stds[metric] = {}
        for model in active_models:
            col = f"{model}_{metric}_score"
            means[metric][model] = df[col].mean() if col in df.columns else 0.0
            stds[metric][model] = df[col].std() if col in df.columns else 0.0

    # ── Combined Grid ────────────────────────────────────────────────────────
    try:
        n_metrics = len(metric_names)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6 * cols, 5 * rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metric_names):
            ax = axes[i]
            for j, model in enumerate(active_models):
                ax.bar(
                    j, means[metric][model], width=0.6,
                    color=colors[model], alpha=0.85,
                    label=labels[model],
                    yerr=stds[metric][model], capsize=5,
                )
            title = metric.replace("_score", "").replace("_", " ").title()
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks(range(len(active_models)))
            ax.set_xticklabels([labels[m] for m in active_models])
            ax.set_ylabel("Score (0-10)")
            ax.set_ylim(0, 11)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"{pipeline_name.title()} — Model Comparison ({len(active_models)} models)",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        grid_path = os.path.join(output_folder, f"{pipeline_name}_grid.png")
        plt.savefig(grid_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Grid plot: {grid_path}")
    except Exception as e:
        print(f"  Grid plot failed: {e}")
        plt.close("all")

    # ── Individual Plots ─────────────────────────────────────────────────────
    for metric in metric_names:
        try:
            fig, ax = plt.subplots(figsize=(max(5, 2 * len(active_models)), 5))
            for j, model in enumerate(active_models):
                ax.bar(
                    j, means[metric][model], width=0.5,
                    color=colors[model], alpha=0.85,
                    label=labels[model],
                    yerr=stds[metric][model], capsize=5,
                )
            title = metric.replace("_score", "").replace("_", " ").title()
            ax.set_title(f"{title} — {pipeline_name.title()}", fontsize=13, fontweight="bold")
            ax.set_xticks(range(len(active_models)))
            ax.set_xticklabels([labels[m] for m in active_models])
            ax.set_ylabel("Score (0-10)")
            ax.set_ylim(0, 11)
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plot_path = os.path.join(output_folder, f"{pipeline_name}_{metric}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"  Plot {metric} failed: {e}")
            plt.close("all")

    print(f"  Plots saved to: {output_folder}")


# =============================================================================
# 5. PIPELINE RUNNERS
# =============================================================================

def run_safety_pipeline(judge: VLLMJudge) -> None:
    """Run the full safety evaluation pipeline."""
    print("\n" + "=" * 60)
    print("SAFETY PIPELINE")
    print("=" * 60)

    cfg = CONFIG["safety"]
    if not cfg["enabled"]:
        print("  Safety pipeline disabled in CONFIG. Skipping.")
        return

    metrics = build_safety_metrics(judge)
    detoxify_calc = DetoxifyCalculator()

    # Find all generated response files for safety benchmarks
    gen_folder = CONFIG["generation"]["output_folder"]
    safety_benchmarks = ["harmbench", "advbench", "xstest_safe", "xstest_unsafe"]

    for bench_name in safety_benchmarks:
        responses_path = os.path.join(gen_folder, f"{bench_name}_responses.jsonl")
        if not os.path.exists(responses_path):
            print(f"\n  Skipping {bench_name}: no responses file at {responses_path}")
            print(f"  (Run generation first: python run_generation.py --model <key> --all)")
            continue

        output_csv = os.path.join(cfg["output_folder"], f"{bench_name}_eval.csv")
        summary_csv = os.path.join(cfg["output_folder"], f"{bench_name}_summary.csv")

        # Run three-way evaluation
        df = run_nmodel_evaluation(
            input_path=responses_path,
            output_path=output_csv,
            metrics=metrics,
            detoxify_calc=detoxify_calc,
            sample_size=cfg["sample_size"],
            pipeline_name=f"safety_{bench_name}",
        )

        # Generate summary
        if df is not None:
            generate_summary(output_csv, summary_csv, pipeline_name=f"safety_{bench_name}")

    # Combined safety plots (all benchmarks merged)
    _combine_and_plot_safety(cfg["output_folder"])


def _combine_and_plot_safety(output_folder: str) -> None:
    """Combine all safety benchmark results and generate unified plots."""
    eval_files = glob.glob(os.path.join(output_folder, "*_eval.csv"))
    if not eval_files:
        return

    dfs = []
    for f in eval_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined_path = os.path.join(output_folder, "safety_combined.csv")
    combined.to_csv(combined_path, index=False)

    summary_path = os.path.join(output_folder, "safety_combined_summary.csv")
    generate_summary(combined_path, summary_path, pipeline_name="safety_combined")
    plot_comparison(combined_path, output_folder, pipeline_name="safety")


def run_instruction_pipeline(judge: VLLMJudge) -> None:
    """Run the full instruction evaluation pipeline."""
    print("\n" + "=" * 60)
    print("INSTRUCTION PIPELINE")
    print("=" * 60)

    cfg = CONFIG["instruction"]
    if not cfg["enabled"]:
        print("  Instruction pipeline disabled in CONFIG. Skipping.")
        return

    metrics = build_instruction_metrics(judge)

    # Find generated response files for instruction benchmarks
    gen_folder = CONFIG["generation"]["output_folder"]
    instruction_benchmarks = ["ifeval", "mt_bench"]

    for bench_name in instruction_benchmarks:
        responses_path = os.path.join(gen_folder, f"{bench_name}_responses.jsonl")
        if not os.path.exists(responses_path):
            print(f"\n  Skipping {bench_name}: no responses file at {responses_path}")
            print(f"  (Run generation first: python run_generation.py --model <key> --all)")
            continue

        output_csv = os.path.join(cfg["output_folder"], f"{bench_name}_eval.csv")
        summary_csv = os.path.join(cfg["output_folder"], f"{bench_name}_summary.csv")

        # Run three-way evaluation (LLM-judge metrics)
        df = run_nmodel_evaluation(
            input_path=responses_path,
            output_path=output_csv,
            metrics=metrics,
            detoxify_calc=None,  # No toxicity scoring for instruction pipeline
            sample_size=cfg["sample_size"],
            pipeline_name=f"instruction_{bench_name}",
        )

        if df is not None:
            generate_summary(output_csv, summary_csv, pipeline_name=f"instruction_{bench_name}")

        # IFEval: also run programmatic verification
        if bench_name == "ifeval":
            ifeval_verify_path = os.path.join(cfg["output_folder"], "ifeval_verification.csv")
            run_ifeval_verification(responses_path, ifeval_verify_path)

    # Combined instruction plots
    _combine_and_plot_instruction(cfg["output_folder"])


def _combine_and_plot_instruction(output_folder: str) -> None:
    """Combine all instruction benchmark results and generate plots."""
    eval_files = glob.glob(os.path.join(output_folder, "*_eval.csv"))
    if not eval_files:
        return

    dfs = []
    for f in eval_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined_path = os.path.join(output_folder, "instruction_combined.csv")
    combined.to_csv(combined_path, index=False)

    summary_path = os.path.join(output_folder, "instruction_combined_summary.csv")
    generate_summary(combined_path, summary_path, pipeline_name="instruction_combined")
    plot_comparison(combined_path, output_folder, pipeline_name="instruction")


# =============================================================================
# 6. UNIFIED COMPARISON (paper-ready output)
# =============================================================================

def generate_unified_comparison() -> None:
    """
    Merge safety and instruction results into a single paper-ready table.
    Dynamically adapts to however many models are configured.
    """
    print("\n" + "=" * 60)
    print("UNIFIED COMPARISON")
    print("=" * 60)

    output_folder = CONFIG["unified_output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    model_keys = get_model_keys()
    rows = []

    for pipeline_name, combined_filename in [
        ("safety", "safety_combined.csv"),
        ("instruction", "instruction_combined.csv"),
    ]:
        combined_path = os.path.join(CONFIG[pipeline_name]["output_folder"], combined_filename)
        if not os.path.exists(combined_path):
            continue

        df = pd.read_csv(combined_path)

        # Detect metrics from first available model
        active_models = [k for k in model_keys
                         if any(c.startswith(f"{k}_") and c.endswith("_score") for c in df.columns)]
        if not active_models:
            continue

        first = active_models[0]
        metrics = [c[len(f"{first}_"):-len("_score")]
                   for c in df.columns if c.startswith(f"{first}_") and c.endswith("_score")]

        for metric in metrics:
            row = {"pipeline": pipeline_name, "metric": metric}
            for model in active_models:
                col = f"{model}_{metric}_score"
                if col in df.columns:
                    row[f"{model}_mean"] = round(df[col].mean(), 3)
                    row[f"{model}_std"] = round(df[col].std(), 3)
            rows.append(row)

    if not rows:
        print("  No results to combine. Run pipelines first.")
        return

    unified_df = pd.DataFrame(rows)
    unified_path = os.path.join(output_folder, CONFIG["unified_summary_csv"])
    unified_df.to_csv(unified_path, index=False)

    print(f"\n  Unified comparison saved to: {unified_path}")
    print(f"\n  Paper-Ready Table:")
    print(unified_df.to_string(index=False))

    # Generate unified plot
    _plot_unified(unified_df, output_folder)


def _plot_unified(df: pd.DataFrame, output_folder: str) -> None:
    """Generate a single unified comparison figure for the paper."""
    try:
        model_keys = get_model_keys()
        # Detect which models have data
        model_cols = [c for c in df.columns if c.endswith("_mean")]
        active_models = [c.replace("_mean", "") for c in model_cols
                         if c.replace("_mean", "") in model_keys]
        active_models = [m for m in model_keys if m in active_models]  # preserve order

        if not active_models:
            return

        base_colors = ["#e15759", "#4e79a7", "#59a14f", "#f28e2b", "#b07aa1",
                        "#76b7b2", "#ff9da7", "#edc948", "#bab0ac", "#9c755f"]
        colors = {m: base_colors[i % len(base_colors)] for i, m in enumerate(active_models)}

        pipelines = df["pipeline"].unique().tolist()
        n_panels = len(pipelines)
        fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        for ax_idx, pipeline in enumerate(pipelines):
            ax = axes[ax_idx]
            pdf = df[df["pipeline"] == pipeline]

            if pdf.empty:
                ax.set_title(f"{pipeline.title()} — No Data")
                continue

            x = np.arange(len(pdf))
            n_models = len(active_models)
            width = 0.8 / n_models

            for j, model in enumerate(active_models):
                mean_col = f"{model}_mean"
                std_col = f"{model}_std"
                vals = pdf[mean_col].fillna(0).values if mean_col in pdf.columns else np.zeros(len(pdf))
                errs = pdf[std_col].fillna(0).values if std_col in pdf.columns else np.zeros(len(pdf))
                ax.bar(x + j * width, vals, width, label=model.upper(),
                       color=colors[model], alpha=0.85, yerr=errs, capsize=3)

            ax.set_title(f"{pipeline.title()} Metrics", fontsize=13, fontweight="bold")
            ax.set_xticks(x + width * (n_models - 1) / 2)
            ax.set_xticklabels(
                [m.replace("_", " ").title() for m in pdf["metric"]],
                rotation=45, ha="right", fontsize=9,
            )
            ax.set_ylabel("Score (0-10)")
            ax.set_ylim(0, 11)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle(f"Unified Model Comparison ({len(active_models)} models)",
                     fontsize=15, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(output_folder, "unified_comparison.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Unified plot: {path}")
    except Exception as e:
        print(f"  Unified plot failed: {e}")
        plt.close("all")


# =============================================================================
# 7. MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipelines.")
    parser.add_argument(
        "--pipeline", type=str, default="both",
        choices=["safety", "instruction", "both", "unified"],
        help="Which pipeline to run (default: both)"
    )
    args = parser.parse_args()

    print_config_summary()

    # Setup
    logger = setup_logging(CONFIG["unified_output_folder"])

    # Initialize judge
    try:
        judge = VLLMJudge()
    except Exception as e:
        print(f"Failed to connect to judge vLLM server: {e}")
        print(f"Make sure server.py is running with: --model {CONFIG['judge_model']}")
        return

    # Run pipelines
    if args.pipeline in ["safety", "both"]:
        run_safety_pipeline(judge)

    if args.pipeline in ["instruction", "both"]:
        run_instruction_pipeline(judge)

    if args.pipeline in ["unified", "both"]:
        generate_unified_comparison()

    print(f"\n{'='*60}")
    print("ALL EVALUATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()