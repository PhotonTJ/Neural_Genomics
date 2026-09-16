"""Tests for the model-free half of the suite.

Everything here runs without torch, transformers or a GPU, so the geometry, the DTW
implementation and the IO layer can be checked on any machine before a run is launched.
The model-dependent paths (triad.profile_model, models.*) are exercised by
scripts/smoke.py on a GPU box instead.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import dtw, io as nio, behavior          # noqa: E402


def test_resample():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    r = dtw.resample(y, 7)
    assert len(r) == 7
    assert np.isclose(r[0], 0.0) and np.isclose(r[-1], 3.0)
    assert np.all(np.diff(r) >= -1e-12), "monotone input must stay monotone"


def test_dtw_identity_and_shift():
    a = np.sin(np.linspace(0, 3, 32)).reshape(-1, 1)
    c, e, path = dtw.dtw(a, a, band=0.2)
    assert c < 1e-12, f"self-distance should vanish, got {c}"
    assert e == 0.0
    b = np.roll(a, 2, axis=0)
    c2, e2, _ = dtw.dtw(a, b, band=0.3)
    c3, _, _ = dtw.dtw(a, np.random.RandomState(0).randn(32, 1), band=0.3)
    assert c2 < c3, "a shifted copy must be closer than noise"


def test_dtw_symmetry_and_matrix():
    rs = np.random.RandomState(1)
    X = rs.randn(4, 16, 3)
    C, E = dtw.pairwise_matrix(X, band=0.25, normalise="minmax")
    assert C.shape == (4, 4)
    assert np.allclose(C, C.T), "distance matrix must be symmetric"
    assert np.allclose(np.diag(C), 0.0)
    assert E.shape == C.shape


def test_joint_normalisation_is_joint():
    """Normalising a set must use set-wide extrema, not per-curve extrema."""
    X = np.stack([np.zeros((8, 1)), np.ones((8, 1)) * 5.0])
    Y = dtw.normalise_set(X, "minmax")
    assert np.isclose(Y[0].max(), 0.0) and np.isclose(Y[1].max(), 1.0), \
        "per-curve normalisation would send both to [0,1]"


def test_threshold_quantile():
    M = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], float)
    t = dtw.threshold(M, q=0.5)
    assert 1.0 <= t <= 3.0


def test_io_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = nio.save(os.path.join(d, "run"), {"a": np.arange(5), "b": 3.5},
                     cfg={"experiment": "t"}, notes="hello")
        arrays, man = nio.load(p)
        assert np.array_equal(arrays["a"], np.arange(5))
        assert np.isclose(arrays["b"], 3.5)
        assert man["notes"] == "hello"
        assert man["config"]["experiment"] == "t"
        assert man["arrays"]["a"]["shape"] == [5]
        assert nio.find(d) == [p]


def test_behaviour_metrics():
    assert np.isclose(behavior.rouge_l("the cat sat", "the cat sat"), 1.0)
    assert behavior.rouge_l("the cat sat", "a dog ran") < 0.2
    assert behavior.bleu("the cat sat", "the cat sat") > 0.5
    assert np.isclose(behavior.exact_match("The Answer.", "answer"), 1.0)
    assert np.isclose(behavior.token_f1("a b c", "a b c"), 1.0)
    assert 0.0 < behavior.token_f1("a b c", "a b d") < 1.0
    assert np.isclose(behavior.accuracy(["x", "y"], ["x", "z"]), 0.5)


def test_score_dispatch():
    assert behavior.METRIC_FOR_TASK["wmt16"] == "bleu"
    assert behavior.METRIC_FOR_TASK["cnn_dailymail"] == "rouge_l"
    s = behavior.score("imdb", ["positive", "negative"], ["positive", "positive"])
    assert np.isclose(s, 0.5)


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for f in fns:
        try:
            f()
            print(f"  PASS  {f.__name__}")
        except Exception as e:
            bad += 1
            print(f"  FAIL  {f.__name__}: {e}")
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    return bad


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
