"""Core library for the nDNA ICLR experiment suite.

Submodules are imported lazily so that the parts which need no deep-learning stack --
`dtw`, `io`, and the string metrics in `behavior` -- can be used, and tested, on a
machine without torch installed.
"""
import importlib

_LAZY = {"behavior", "dtw", "io", "models", "probes", "triad", "harness"}

__all__ = sorted(_LAZY) + ["profile_model", "commitment_depth", "pairwise_matrix",
                           "stack_triad", "threshold", "save", "load", "find"]

_FORWARD = {
    "profile_model": ("triad", "profile_model"),
    "commitment_depth": ("triad", "commitment_depth"),
    "pairwise_matrix": ("dtw", "pairwise_matrix"),
    "stack_triad": ("dtw", "stack_triad"),
    "threshold": ("dtw", "threshold"),
    "save": ("io", "save"),
    "load": ("io", "load"),
    "find": ("io", "find"),
}


def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name in _FORWARD:
        mod_name, attr = _FORWARD[name]
        obj = getattr(importlib.import_module(f".{mod_name}", __name__), attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
