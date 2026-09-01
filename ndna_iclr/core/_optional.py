"""A stand-in for torch when it is not installed.

The point is `--dry-run`: you should be able to check an experiment's plan, and import
every module, on a laptop with no deep-learning stack. Attribute access on the stub
returns a callable that works as a no-op decorator (so `@torch.no_grad()` at module
scope is fine) but raises the moment it is actually called for real work.
"""


class _Missing:
    def __init__(self, name="torch"):
        self._name = name

    def _fail(self, *a, **k):
        raise ImportError(
            f"{self._name} is required for this operation but is not installed. "
            "Install requirements.txt, or use --dry-run / the model-free experiments."
        )

    def __getattr__(self, item):
        # `@torch.no_grad()` must survive import: return something that is both callable
        # and usable as a decorator, and only explodes when used as a real function.
        return _Decoy(f"{self._name}.{item}")

    def __call__(self, *a, **k):
        self._fail()


class _Decoy:
    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        # used as @dec or @dec(): if handed a single callable, act as a pass-through
        if len(a) == 1 and not k and callable(a[0]):
            return a[0]
        return _Decoy(self._name)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, item):
        return _Decoy(f"{self._name}.{item}")

    def __bool__(self):
        return False


def load_torch():
    """-> (torch_or_stub, available)"""
    try:
        import torch
        return torch, True
    except ImportError:
        return _Missing("torch"), False
