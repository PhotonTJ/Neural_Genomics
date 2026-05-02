"""
=============================================================================
TEST SCRIPT — Validate Part 1 Components
=============================================================================
Run this in a Jupyter notebook cell or terminal to verify everything works.

What this tests:
  1. All imports resolve correctly
  2. CONFIG structure is valid
  3. Directory creation works
  4. Metric suites build without errors (mock judge)
  5. Data loading/cleaning functions work
  6. DetoxifyCalculator loads and scores text
  7. JSONL read/write round-trip works
  8. Broken output detection works
  9. Record validation works
  10. evaluate_record_for_model() works end-to-end (with mock)

What this does NOT test (needs actual vLLM server):
  - VLLMJudge connection
  - ModelClient connection
  - generate_responses()

Run:
    python test_part1.py
  or in Jupyter:
    %run test_part1.py
=============================================================================
"""

import os
import sys
import json
import tempfile
import traceback

# Track results
passed = []
failed = []

def test(name):
    """Decorator for test functions."""
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
print("PART 1 — COMPONENT TESTS")
print("=" * 60)


# ─── Test 1: Imports ─────────────────────────────────────────────────────────

@test("1. Core imports")
def test_imports():
    import torch
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from pydantic import BaseModel
    from openai import AsyncOpenAI, OpenAI
    from deepeval.metrics import GEval
    from deepeval.metrics.g_eval import Rubric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.models import DeepEvalBaseLLM
    from sentence_transformers import SentenceTransformer

test_imports()


# ─── Test 2: Pipeline imports ────────────────────────────────────────────────

@test("2. Pipeline module imports")
def test_pipeline_imports():
    from evaluation_pipeline_part1 import (
        CONFIG,
        setup_logging,
        VLLMJudge,
        DetoxifyCalculator,
        build_safety_metrics,
        build_instruction_metrics,
        read_jsonl,
        write_jsonl,
        detect_broken_output,
        validate_record_for_eval,
        evaluate_record_for_model,
        ModelClient,
        generate_responses,
        print_config_summary,
        get_model_keys,
        get_model_config,
    )

test_pipeline_imports()


# ─── Test 3: CONFIG structure ────────────────────────────────────────────────

@test("3. CONFIG structure validation")
def test_config():
    from evaluation_pipeline_part1 import CONFIG, get_model_keys, get_model_config

    # Check required top-level keys
    required = [
        "vllm_url", "vllm_api_key", "judge_model", "models",
        "safety", "instruction", "unified_output_folder",
        "generation", "min_output_length", "repetition_threshold",
    ]
    for key in required:
        assert key in CONFIG, f"Missing CONFIG key: {key}"

    # Check models is a list
    assert isinstance(CONFIG["models"], list), "CONFIG['models'] should be a list"
    assert len(CONFIG["models"]) >= 1, "Need at least 1 model"

    # Check each model entry
    for m in CONFIG["models"]:
        assert "key" in m, f"Model entry missing 'key': {m}"
        assert "name" in m, f"Model entry missing 'name': {m}"

    # Check helpers
    keys = get_model_keys()
    assert isinstance(keys, list)
    assert len(keys) >= 1
    cfg = get_model_config(keys[0])
    assert "name" in cfg

    # Check pipeline configs
    for pipeline in ["safety", "instruction"]:
        assert "enabled" in CONFIG[pipeline], f"{pipeline} missing 'enabled'"
        assert "benchmarks" in CONFIG[pipeline], f"{pipeline} missing 'benchmarks'"
        assert "output_folder" in CONFIG[pipeline], f"{pipeline} missing 'output_folder'"
        assert "sample_size" in CONFIG[pipeline], f"{pipeline} missing 'sample_size'"

test_config()


# ─── Test 4: Directory creation ──────────────────────────────────────────────

@test("4. Output directories exist")
def test_directories():
    from evaluation_pipeline_part1 import CONFIG
    for folder in [
        CONFIG["safety"]["output_folder"],
        CONFIG["instruction"]["output_folder"],
        CONFIG["unified_output_folder"],
        CONFIG["generation"]["output_folder"],
    ]:
        assert os.path.isdir(folder), f"Directory not created: {folder}"

test_directories()


# ─── Test 5: Logging setup ───────────────────────────────────────────────────

@test("5. Logging setup")
def test_logging():
    from evaluation_pipeline_part1 import setup_logging
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logging(tmpdir)
        logger.info("Test log message")
        log_files = [f for f in os.listdir(tmpdir) if f.endswith('.log')]
        assert len(log_files) > 0, "No log file created"

test_logging()


# ─── Test 6: JSONL read/write round-trip ─────────────────────────────────────

@test("6. JSONL read/write round-trip")
def test_jsonl():
    from evaluation_pipeline_part1 import read_jsonl, write_jsonl

    test_records = [
        {"id": 0, "prompt": "Hello, how are you?", "output_base": "I'm fine", "benchmark": "test"},
        {"id": 1, "prompt": "What is 2+2?", "output_base": "4", "category": "math"},
        {"id": 2, "prompt": "Tell me a joke", "output_sft": "Why did the chicken...", "output_dpo": "I can't..."},
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        tmppath = f.name
        for r in test_records:
            f.write(json.dumps(r) + "\n")

    try:
        loaded = read_jsonl(tmppath)
        assert len(loaded) == 3, f"Expected 3 records, got {len(loaded)}"
        assert loaded[0]["prompt"] == "Hello, how are you?"
        assert loaded[1]["id"] == 1
        assert "benchmark" in loaded[0]
        assert "category" in loaded[1]

        # Test write
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f2:
            tmppath2 = f2.name
        write_jsonl(loaded, tmppath2)
        reloaded = read_jsonl(tmppath2)
        assert len(reloaded) == 3
        os.unlink(tmppath2)
    finally:
        os.unlink(tmppath)

test_jsonl()


# ─── Test 7: Broken output detection ────────────────────────────────────────

@test("7. Broken output detection")
def test_broken_output():
    from evaluation_pipeline_part1 import detect_broken_output

    # Should be broken
    assert detect_broken_output("")[0] == True, "Empty string should be broken"
    assert detect_broken_output(None)[0] == True, "None should be broken"
    assert detect_broken_output("hi")[0] == True, "Too short should be broken"
    assert detect_broken_output("assistant")[0] == True, "Role token should be broken"
    assert detect_broken_output("the the the the the the the the the the the the")[0] == True, "Repetition should be broken"

    # Should be OK
    assert detect_broken_output("This is a perfectly normal response that is long enough.")[0] == False

test_broken_output()


# ─── Test 8: Record validation ───────────────────────────────────────────────

@test("8. Record validation")
def test_validation():
    from evaluation_pipeline_part1 import validate_record_for_eval

    # Valid record
    good = {
        "id": 0,
        "prompt": "test",
        "output_base": "This is a normal length response from the base model.",
        "output_sft": "This is a normal length response from the SFT model.",
        "output_dpo": "This is a normal length response from the DPO model.",
    }
    valid, reason = validate_record_for_eval(good, ["output_base", "output_sft", "output_dpo"])
    assert valid, f"Good record should be valid, got: {reason}"

    # Missing field
    bad_missing = {"id": 1, "prompt": "test", "output_base": "something"}
    valid, reason = validate_record_for_eval(bad_missing, ["output_base", "output_sft"])
    assert not valid, "Missing field should be invalid"

    # Empty field
    bad_empty = {"id": 2, "prompt": "test", "output_base": "  "}
    valid, reason = validate_record_for_eval(bad_empty, ["output_base"])
    assert not valid, "Empty field should be invalid"

test_validation()


# ─── Test 9: DetoxifyCalculator ──────────────────────────────────────────────

@test("9. DetoxifyCalculator loads and scores")
def test_detoxify():
    from evaluation_pipeline_part1 import DetoxifyCalculator

    calc = DetoxifyCalculator()

    # Safe text
    score_safe, reason_safe = calc.calculate_toxicity("The weather is nice today.")
    assert score_safe is not None, "Score should not be None"
    assert score_safe < 3.0, f"Safe text got high toxicity: {score_safe}"

    # Toxic text (mild)
    score_toxic, reason_toxic = calc.calculate_toxicity("You are an idiot and I hate you.")
    assert score_toxic is not None
    assert score_toxic > score_safe, "Toxic text should score higher"

    # Empty text
    score_empty, _ = calc.calculate_toxicity("")
    assert score_empty == 0.0, "Empty text should be 0.0"

test_detoxify()


# ─── Test 10: Metric suites build (with mock judge) ─────────────────────────

@test("10. Safety metrics build correctly")
def test_safety_metrics():
    from evaluation_pipeline_part1 import build_safety_metrics
    from deepeval.models import DeepEvalBaseLLM

    # Create a minimal mock judge (won't actually call vLLM)
    class MockJudge(DeepEvalBaseLLM):
        def load_model(self): return None
        def generate(self, prompt, schema=None): return "{}"
        async def a_generate(self, prompt, schema=None): return "{}"
        def get_model_name(self): return "mock"

    mock = MockJudge()
    metrics = build_safety_metrics(mock)

    expected = ["response_quality", "relevance", "toxicity", "refusal", "harmfulness"]
    assert set(metrics.keys()) == set(expected), f"Got {list(metrics.keys())}, expected {expected}"
    assert len(metrics) == 5, f"Expected 5 safety metrics, got {len(metrics)}"

test_safety_metrics()


@test("11. Instruction metrics build correctly")
def test_instruction_metrics():
    from evaluation_pipeline_part1 import build_instruction_metrics
    from deepeval.models import DeepEvalBaseLLM

    class MockJudge(DeepEvalBaseLLM):
        def load_model(self): return None
        def generate(self, prompt, schema=None): return "{}"
        async def a_generate(self, prompt, schema=None): return "{}"
        def get_model_name(self): return "mock"

    mock = MockJudge()
    metrics = build_instruction_metrics(mock)

    expected = ["response_quality", "relevance", "helpfulness", "instruction_following", "format_style", "conciseness"]
    assert set(metrics.keys()) == set(expected), f"Got {list(metrics.keys())}, expected {expected}"
    assert len(metrics) == 6, f"Expected 6 instruction metrics, got {len(metrics)}"

test_instruction_metrics()


# ─── Test 12: Metric suites are properly separated ──────────────────────────

@test("12. Metric suites are properly separated (no overlap leak)")
def test_metric_separation():
    from evaluation_pipeline_part1 import build_safety_metrics, build_instruction_metrics
    from deepeval.models import DeepEvalBaseLLM

    class MockJudge(DeepEvalBaseLLM):
        def load_model(self): return None
        def generate(self, prompt, schema=None): return "{}"
        async def a_generate(self, prompt, schema=None): return "{}"
        def get_model_name(self): return "mock"

    mock = MockJudge()
    safety = build_safety_metrics(mock)
    instruction = build_instruction_metrics(mock)

    # Safety should NOT have instruction-specific metrics
    assert "instruction_following" not in safety, "instruction_following leaked into safety"
    assert "format_style" not in safety, "format_style leaked into safety"
    assert "conciseness" not in safety, "conciseness leaked into safety"
    assert "helpfulness" not in safety, "helpfulness leaked into safety"

    # Instruction should NOT have safety-specific metrics
    assert "toxicity" not in instruction, "toxicity leaked into instruction"
    assert "refusal" not in instruction, "refusal leaked into instruction"
    assert "harmfulness" not in instruction, "harmfulness leaked into instruction"

    # Shared metrics exist in both
    assert "response_quality" in safety and "response_quality" in instruction
    assert "relevance" in safety and "relevance" in instruction

test_metric_separation()


# ─── Test 13: Config summary prints without error ────────────────────────────

@test("13. Config summary prints")
def test_config_summary():
    from evaluation_pipeline_part1 import print_config_summary
    # Just verify it doesn't crash
    print_config_summary()

test_config_summary()


# ─── Test 14: VLLMJudge class structure ──────────────────────────────────────

@test("14. VLLMJudge class structure")
def test_judge_structure():
    from evaluation_pipeline_part1 import VLLMJudge
    from deepeval.models import DeepEvalBaseLLM

    # Verify it's a proper DeepEval model subclass
    assert issubclass(VLLMJudge, DeepEvalBaseLLM)

    # Verify required methods exist
    assert hasattr(VLLMJudge, 'generate')
    assert hasattr(VLLMJudge, 'a_generate')
    assert hasattr(VLLMJudge, 'get_model_name')
    assert hasattr(VLLMJudge, 'load_model')
    assert hasattr(VLLMJudge, '_extract_json')

test_judge_structure()


# ─── Test 15: JSON extraction ────────────────────────────────────────────────

@test("15. JSON extraction from model responses")
def test_json_extraction():
    from evaluation_pipeline_part1 import VLLMJudge

    # Can't instantiate VLLMJudge without server, so test the method directly
    # Create a minimal instance by bypassing __init__
    judge = object.__new__(VLLMJudge)

    # Test: JSON in code block
    text1 = 'Here is the result:\n```json\n{"score": 7, "reason": "good"}\n```'
    result1 = judge._extract_json(text1)
    parsed1 = json.loads(result1)
    assert parsed1["score"] == 7

    # Test: raw JSON in text
    text2 = 'The evaluation is {"score": 5, "reason": "ok"} and that is final.'
    result2 = judge._extract_json(text2)
    parsed2 = json.loads(result2)
    assert parsed2["score"] == 5

    # Test: nested JSON
    text3 = '{"outer": {"inner": 42}, "score": 3}'
    result3 = judge._extract_json(text3)
    parsed3 = json.loads(result3)
    assert parsed3["outer"]["inner"] == 42

test_json_extraction()


# ─── Test 16: evaluate_record_for_model with mock ────────────────────────────

@test("16. evaluate_record_for_model (mock, detoxify only)")
def test_eval_record():
    from evaluation_pipeline_part1 import evaluate_record_for_model, DetoxifyCalculator

    detox = DetoxifyCalculator()

    # Run with NO GEval metrics (empty dict) — just detoxify
    result = evaluate_record_for_model(
        prompt="What is the weather?",
        output="The weather is sunny and warm today.",
        metrics={},           # no GEval metrics — just test the structure
        detoxify_calc=detox,
    )

    assert "toxicity_detoxify_score" in result, "Missing detoxify score"
    assert "toxicity_detoxify_reason" in result, "Missing detoxify reason"
    assert result["toxicity_detoxify_score"] is not None
    assert result["toxicity_detoxify_score"] < 3.0, "Safe text should have low toxicity"

test_eval_record()


# ─── Test 17: server.py exists and is valid Python ───────────────────────────

@test("17. server.py is valid Python")
def test_server_syntax():
    import py_compile
    py_compile.compile("server.py", doraise=True)

test_server_syntax()


# =============================================================================
# SUMMARY
# =============================================================================

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

if failed:
    print("\nNEXT STEPS (fix failures first):")
    print("  Check that all pip packages are installed:")
    print("    pip install deepeval openai torch pandas numpy tqdm")
    print("    pip install sentence-transformers detoxify pydantic seaborn")
else:
    print("\nPart 1 is working. NEXT STEPS:")
    print()
    print("  To test with an actual vLLM server (optional, can wait for Part 2):")
    print("    Terminal 1:  python server.py --model Qwen/Qwen2.5-32B-Instruct")
    print("    Terminal 2:  python -c \"")
    print("      from evaluation_pipeline_part1 import VLLMJudge")
    print("      judge = VLLMJudge()")
    print("      print(judge.generate('Say hello in one word.'))\"")
    print()
    print("  Otherwise, proceed to Part 2 (Benchmark Integration).")

print("=" * 60)