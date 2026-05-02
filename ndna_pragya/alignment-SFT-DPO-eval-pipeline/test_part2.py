"""
=============================================================================
TEST SCRIPT — Validate Part 2: Benchmark Integration
=============================================================================
Downloads all benchmarks with --sample-size 5 to verify everything works
without downloading full datasets.

Run:
    python test_part2.py

Requirements:
    pip install datasets
=============================================================================
"""

import os
import sys
import json
import traceback

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
print("PART 2 — BENCHMARK INTEGRATION TESTS")
print("=" * 60)


# ── Test 1: datasets library available ───────────────────────────────────────

@test("1. HuggingFace datasets library available")
def test_datasets_import():
    import datasets
    assert hasattr(datasets, 'load_dataset')

test_datasets_import()


# ── Test 2: benchmark_integration imports ────────────────────────────────────

@test("2. benchmark_integration module imports")
def test_module_imports():
    from benchmark_integration import (
        download_harmbench,
        download_advbench,
        download_xstest,
        download_ifeval,
        download_mt_bench,
        download_all_benchmarks,
        verify_benchmarks,
        BENCHMARK_PATHS,
    )

test_module_imports()


# ── Test 3: BENCHMARK_PATHS structure ────────────────────────────────────────

@test("3. BENCHMARK_PATHS has all expected entries")
def test_paths():
    from benchmark_integration import BENCHMARK_PATHS
    expected = ["harmbench", "advbench", "xstest_safe", "xstest_unsafe", "ifeval", "mt_bench"]
    for name in expected:
        assert name in BENCHMARK_PATHS, f"Missing path for: {name}"

test_paths()


# ── Test 4: Download HarmBench (small sample) ───────────────────────────────

@test("4. HarmBench download (5 samples)")
def test_harmbench():
    from benchmark_integration import download_harmbench, BENCHMARK_PATHS
    path = download_harmbench(sample_size=5)
    assert os.path.exists(path), f"File not created: {path}"
    with open(path) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 5, f"Expected 5, got {len(records)}"
    assert "prompt" in records[0]
    assert "benchmark" in records[0]
    assert records[0]["benchmark"] == "harmbench"

test_harmbench()


# ── Test 5: Download AdvBench (small sample) ─────────────────────────────────

@test("5. AdvBench download (5 samples)")
def test_advbench():
    from benchmark_integration import download_advbench, BENCHMARK_PATHS
    path = download_advbench(sample_size=5)
    assert os.path.exists(path)
    with open(path) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 5
    assert records[0]["benchmark"] == "advbench"

test_advbench()


# ── Test 6: Download XSTest (small sample) ───────────────────────────────────

@test("6. XSTest download (5 samples)")
def test_xstest():
    from benchmark_integration import download_xstest, BENCHMARK_PATHS
    paths = download_xstest(sample_size=5)
    assert "safe" in paths
    assert "unsafe" in paths
    # At least one of the files should have records
    total = 0
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path) as f:
                records = [json.loads(line) for line in f]
            total += len(records)
            if records:
                assert records[0]["benchmark"] == "xstest"
    assert total > 0, "No XSTest records downloaded"

test_xstest()


# ── Test 7: Download IFEval (small sample) ───────────────────────────────────

@test("7. IFEval download (5 samples)")
def test_ifeval():
    from benchmark_integration import download_ifeval, BENCHMARK_PATHS
    path = download_ifeval(sample_size=5)
    assert os.path.exists(path)
    with open(path) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 5
    assert records[0]["benchmark"] == "ifeval"
    # IFEval should preserve instruction_id_list
    assert "instruction_id_list" in records[0], "Missing IFEval instruction_id_list"

test_ifeval()


# ── Test 8: Download MT-Bench (small sample) ─────────────────────────────────

@test("8. MT-Bench download (5 samples)")
def test_mt_bench():
    from benchmark_integration import download_mt_bench, BENCHMARK_PATHS
    path = download_mt_bench(sample_size=5)
    assert os.path.exists(path)
    with open(path) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 5
    assert records[0]["benchmark"] == "mt_bench"

test_mt_bench()


# ── Test 9: JSONL format consistency ─────────────────────────────────────────

@test("9. All benchmarks have consistent JSONL format")
def test_format_consistency():
    from benchmark_integration import BENCHMARK_PATHS
    required_fields = {"id", "prompt", "benchmark"}

    for name, path in BENCHMARK_PATHS.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                record = json.loads(line)
                for field in required_fields:
                    assert field in record, f"{name} line {line_num}: missing '{field}'"
                assert isinstance(record["id"], int), f"{name}: id should be int"
                assert isinstance(record["prompt"], str), f"{name}: prompt should be str"
                assert len(record["prompt"]) > 0, f"{name}: empty prompt at line {line_num}"

test_format_consistency()


# ── Test 10: verify_benchmarks works ─────────────────────────────────────────

@test("10. verify_benchmarks() runs without error")
def test_verify():
    from benchmark_integration import verify_benchmarks
    status = verify_benchmarks()
    assert isinstance(status, dict)
    assert len(status) > 0

test_verify()


# ── Test 11: Prompts are readable by Part 1's read_jsonl ─────────────────────

@test("11. Part 1 read_jsonl can load benchmark files")
def test_part1_compat():
    from evaluation_pipeline_part1 import read_jsonl
    from benchmark_integration import BENCHMARK_PATHS

    for name, path in BENCHMARK_PATHS.items():
        if not os.path.exists(path):
            continue
        records = read_jsonl(path)
        assert len(records) > 0, f"read_jsonl returned empty for {name}"
        assert "prompt" in records[0], f"read_jsonl missing 'prompt' for {name}"
        assert "id" in records[0], f"read_jsonl missing 'id' for {name}"

test_part1_compat()


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

if not failed:
    print("\nPart 2 is working. NEXT STEPS:")
    print()
    print("  1. Download FULL benchmarks (no sample limit):")
    print("     python benchmark_integration.py")
    print()
    print("  2. Or verify what's downloaded:")
    print("     python benchmark_integration.py --verify")
    print()
    print("  3. Then proceed to Part 3 (Pipeline Runners).")

print("=" * 60)
