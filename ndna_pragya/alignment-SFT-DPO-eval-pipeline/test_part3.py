"""
=============================================================================
TEST SCRIPT — Validate Part 3: Pipeline Runners
=============================================================================
Tests the evaluation pipeline structure without needing a vLLM server.
Creates mock response files and tests summary/plot generation.

Run:
    python test_part3.py
=============================================================================
"""

import os
import sys
import json
import tempfile
import traceback
import pandas as pd
import numpy as np

passed = []
failed = []

def test(name):
    def decorator(func):
        def wrapper():
            try:
                func()
                passed.append(name)
                print(f"  ✅ {name}")
            except Exception as e:
                failed.append((name, str(e)))
                print(f"  ❌ {name}: {e}")
                traceback.print_exc()
        return wrapper
    return decorator

print("=" * 60)
print("PART 3 — PIPELINE RUNNER TESTS")
print("=" * 60)


# ── Test 1: Imports ──────────────────────────────────────────────────────────

@test("1. run_evaluation module imports")
def t1():
    from run_evaluation import (
        run_nmodel_evaluation,
        run_ifeval_verification,
        generate_summary,
        plot_comparison,
        run_safety_pipeline,
        run_instruction_pipeline,
        generate_unified_comparison,
    )
t1()


# ── Test 2: Create mock response data ───────────────────────────────────────

MOCK_DIR = os.path.join(tempfile.gettempdir(), "test_part3")
os.makedirs(MOCK_DIR, exist_ok=True)

def create_mock_responses(path, n=10, benchmark="test"):
    """Create a mock responses JSONL for testing."""
    records = []
    for i in range(n):
        records.append({
            "id": i,
            "prompt": f"Test prompt number {i}. Please write a detailed response.",
            "benchmark": benchmark,
            "category": "test_category",
            "output_base": f"This is the base model response for prompt {i}. It provides basic information without much detail or structure.",
            "output_sft": f"This is the SFT model response for prompt {i}. It follows instructions more carefully and provides a well-structured answer with relevant details.",
            "output_dpo": f"I appreciate your question about prompt {i}. However, I need to point out that this could be sensitive. Let me provide a helpful and safe response instead.",
        })
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path

MOCK_RESPONSES = create_mock_responses(
    os.path.join(MOCK_DIR, "mock_responses.jsonl"), n=10
)

@test("2. Mock response data created")
def t2():
    assert os.path.exists(MOCK_RESPONSES)
    from evaluation_pipeline_part1 import read_jsonl
    records = read_jsonl(MOCK_RESPONSES)
    assert len(records) == 10
    assert "output_base" in records[0]
    assert "output_sft" in records[0]
    assert "output_dpo" in records[0]
t2()


# ── Test 3: Create mock eval results CSV (simulates what three_way_eval produces) ─

MOCK_EVAL_CSV = os.path.join(MOCK_DIR, "mock_eval.csv")

@test("3. Mock eval CSV creation")
def t3():
    np.random.seed(42)
    rows = []
    for i in range(20):
        row = {"id": i, "prompt": f"prompt {i}", "benchmark": "test", "category": "cat"}
        for model in ["base", "sft", "dpo"]:
            for metric in ["response_quality", "relevance", "toxicity", "refusal", "harmfulness"]:
                if model == "base":
                    row[f"{model}_{metric}_score"] = np.random.uniform(2, 6)
                elif model == "sft":
                    row[f"{model}_{metric}_score"] = np.random.uniform(4, 8)
                else:
                    row[f"{model}_{metric}_score"] = np.random.uniform(5, 9)
        rows.append(row)
    pd.DataFrame(rows).to_csv(MOCK_EVAL_CSV, index=False)
    assert os.path.exists(MOCK_EVAL_CSV)
t3()


# ── Test 4: generate_summary works ──────────────────────────────────────────

@test("4. generate_summary produces valid output")
def t4():
    from run_evaluation import generate_summary
    summary_path = os.path.join(MOCK_DIR, "mock_summary.csv")
    df = generate_summary(MOCK_EVAL_CSV, summary_path, "test")
    assert df is not None, "Summary returned None"
    assert os.path.exists(summary_path), "Summary file not created"
    assert "metric" in df.columns
    assert "base_mean" in df.columns
    assert "sft_mean" in df.columns
    assert "dpo_mean" in df.columns
    assert len(df) == 5  # 5 metrics
t4()


# ── Test 5: Summary has correct metric names ────────────────────────────────

@test("5. Summary metrics match expected names")
def t5():
    from run_evaluation import generate_summary
    summary_path = os.path.join(MOCK_DIR, "mock_summary2.csv")
    df = generate_summary(MOCK_EVAL_CSV, summary_path, "test")
    expected_metrics = {"response_quality", "relevance", "toxicity", "refusal", "harmfulness"}
    actual_metrics = set(df["metric"].values)
    assert actual_metrics == expected_metrics, f"Expected {expected_metrics}, got {actual_metrics}"
t5()


# ── Test 6: Summary shows SFT > Base and DPO > SFT on average ───────────────

@test("6. Summary means follow expected ordering (SFT > Base, DPO > SFT)")
def t6():
    from run_evaluation import generate_summary
    summary_path = os.path.join(MOCK_DIR, "mock_summary3.csv")
    df = generate_summary(MOCK_EVAL_CSV, summary_path, "test")
    # On average across metrics, SFT should score higher than base, DPO higher than SFT
    avg_base = df["base_mean"].mean()
    avg_sft = df["sft_mean"].mean()
    avg_dpo = df["dpo_mean"].mean()
    assert avg_sft > avg_base, f"SFT ({avg_sft:.2f}) should be > Base ({avg_base:.2f})"
    assert avg_dpo > avg_sft, f"DPO ({avg_dpo:.2f}) should be > SFT ({avg_sft:.2f})"
t6()


# ── Test 7: plot_comparison generates files ──────────────────────────────────

@test("7. plot_comparison generates plot files")
def t7():
    from run_evaluation import plot_comparison
    plot_dir = os.path.join(MOCK_DIR, "plots")
    plot_comparison(MOCK_EVAL_CSV, plot_dir, "test")
    assert os.path.exists(os.path.join(plot_dir, "test_grid.png")), "Grid plot not created"
    # Check individual plots exist
    individual_plots = [f for f in os.listdir(plot_dir) if f.startswith("test_") and f != "test_grid.png"]
    assert len(individual_plots) > 0, "No individual plots created"
t7()


# ── Test 8: Summary file contains win rates section ─────────────────────────

@test("8. Summary file contains win rates and deltas")
def t8():
    summary_path = os.path.join(MOCK_DIR, "mock_summary.csv")
    with open(summary_path, "r") as f:
        content = f.read()
    assert "Win Rates" in content or "Pairwise Win Rates" in content, "Missing win rates section"
    assert "Deltas" in content, "Missing deltas section"
    # Pairwise pairs are generated via combinations — order is (base, sft), (base, dpo), (sft, dpo)
    assert "base_vs_sft" in content or "sft_vs_base" in content, "Missing base/sft comparison"
    assert "base_vs_dpo" in content or "dpo_vs_base" in content, "Missing base/dpo comparison"
t8()


# ── Test 9: IFEval heuristic checker ────────────────────────────────────────

@test("9. IFEval heuristic constraint checking")
def t9():
    from run_evaluation import _check_instruction_heuristic

    # Word count check
    text = "word " * 500  # 500 words
    assert _check_instruction_heuristic(
        text, "length_constraints:number_words", {"num_words": 400, "relation": "at least"}
    ), "Should pass 400 word minimum"
    assert not _check_instruction_heuristic(
        "short text", "length_constraints:number_words", {"num_words": 400, "relation": "at least"}
    ), "Should fail 400 word minimum"

    # Keyword check
    text = "AI is great and AI is useful"
    assert _check_instruction_heuristic(
        text, "keywords:frequency", {"keyword": "AI", "frequency": 2}
    ), "Should find AI at least 2 times"

    # Forbidden words
    text = "This is a clean response"
    assert _check_instruction_heuristic(
        text, "keywords:forbidden_words", {"forbidden_words": ["bad", "evil"]}
    ), "Should pass — no forbidden words"

    # No commas
    assert _check_instruction_heuristic(
        "No commas here", "punctuation:no_comma", {}
    ), "No commas should pass"
    assert not _check_instruction_heuristic(
        "Has, a comma", "punctuation:no_comma", {}
    ), "Has comma should fail"
t9()


# ── Test 10: Mock data compatible with unified comparison ────────────────────

@test("10. Mock instruction eval CSV")
def t10():
    # Create mock instruction results
    np.random.seed(123)
    rows = []
    for i in range(15):
        row = {"id": i, "prompt": f"inst prompt {i}", "benchmark": "ifeval", "category": "test"}
        for model in ["base", "sft", "dpo"]:
            for metric in ["response_quality", "relevance", "helpfulness",
                          "instruction_following", "format_style", "conciseness"]:
                if model == "base":
                    row[f"{model}_{metric}_score"] = np.random.uniform(1, 4)
                elif model == "sft":
                    row[f"{model}_{metric}_score"] = np.random.uniform(5, 9)
                else:
                    row[f"{model}_{metric}_score"] = np.random.uniform(5, 8)
            row[f"{model}_toxicity_detoxify_score"] = None  # No detoxify for instruction
        rows.append(row)

    inst_csv = os.path.join(MOCK_DIR, "mock_instruction_eval.csv")
    pd.DataFrame(rows).to_csv(inst_csv, index=False)

    from run_evaluation import generate_summary
    summary = generate_summary(inst_csv, os.path.join(MOCK_DIR, "inst_summary.csv"), "instruction")
    assert summary is not None
    # 6 instruction metrics + toxicity_detoxify (None but still detected as _score column)
    assert len(summary) >= 6, f"Expected at least 6 instruction metrics, got {len(summary)}"
t10()


# ── Test 11: validate_record_for_eval catches bad records ────────────────────

@test("11. Record validation catches missing/broken outputs")
def t11():
    from evaluation_pipeline_part1 import validate_record_for_eval

    # Good record
    good = {
        "output_base": "A normal long enough response from the base model here.",
        "output_sft": "A normal long enough response from the SFT model here.",
        "output_dpo": "A normal long enough response from the DPO model here.",
    }
    assert validate_record_for_eval(good, ["output_base", "output_sft", "output_dpo"])[0]

    # Missing SFT
    bad = {
        "output_base": "Fine response here that is long enough for validation.",
        "output_dpo": "Fine response here that is long enough for validation.",
    }
    valid, reason = validate_record_for_eval(bad, ["output_base", "output_sft", "output_dpo"])
    assert not valid
    assert "sft" in reason.lower()
t11()


# =============================================================================
# SUMMARY
# =============================================================================

# Cleanup
import shutil
shutil.rmtree(MOCK_DIR, ignore_errors=True)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)
print(f"  Passed: {len(passed)}/{len(passed) + len(failed)}")
print(f"  Failed: {len(failed)}/{len(passed) + len(failed)}")

if failed:
    print("\n  FAILURES:")
    for name, err in failed:
        print(f"    ❌ {name}: {err}")
else:
    print("\n  ALL TESTS PASSED ✅")

print("\n" + "=" * 60)

if not failed:
    print("\nPart 3 is working. FULL WORKFLOW:")
    print()
    print("  Phase 1 — Generate responses (one model at a time):")
    print("    Terminal 1: python server.py --model <base-model>")
    print("    Terminal 2: python run_generation.py --model base --all")
    print("    (repeat for sft, dpo)")
    print()
    print("  Phase 2 — Evaluate (judge model):")
    print("    Terminal 1: python server.py --model Qwen/Qwen2.5-32B-Instruct")
    print("    Terminal 2: python run_evaluation.py")
    print()
    print("  Results will be in:")
    print("    eval_results/safety/     — safety pipeline outputs")
    print("    eval_results/instruction/ — instruction pipeline outputs")
    print("    eval_results/unified/    — paper-ready comparison table")

print("=" * 60)