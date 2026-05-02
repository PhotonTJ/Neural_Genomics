"""
ndna.storage

Persistence utilities for saving and loading nDNA metric results.
"""

from .saver import ResultSaver
from .loader import ResultLoader

__all__ = [
    "ResultSaver",
    "ResultLoader",
]

