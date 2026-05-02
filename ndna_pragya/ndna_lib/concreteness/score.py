"""
ndna_lib.concreteness.score

Concreteness scoring.

Implements the same families of scoring you used in the notebook:
  1) regex word extraction (no POS filtering)
  2) spaCy POS filtering (remove function words)

Important: score_pos_filtered is STREAMING-safe now (does not buffer all texts).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Any, Iterator
from pathlib import Path
import re
import time
import math

import pandas as pd


# -----------------------------
# Lexicon loading
# -----------------------------

DEFAULT_CSV = Path(__file__).resolve().parent / "concreteness_scores_original.csv"


@dataclass(frozen=True)
class Lexicon:
    lookup: Dict[str, float]
    word_col: str
    conc_col: str
    csv_path: Path


def load_concreteness_lexicon(csv_path: Optional[str | Path] = None) -> Lexicon:
    """
    Load concreteness lexicon as a lowercase surface-form lookup.
    Your CSV usually uses: Word, Conc.M
    """
    p = Path(csv_path) if csv_path is not None else DEFAULT_CSV
    if not p.exists():
        raise FileNotFoundError(f"Concreteness CSV not found: {p}")

    df = pd.read_csv(p)

    conc_candidates = [
        c for c in df.columns
        if str(c).lower() in ["conc.m", "conc_mean", "mean_concreteness", "concreteness", "mean"]
        or "conc" in str(c).lower()
    ]
    if not conc_candidates:
        raise ValueError(f"Could not find a concreteness column in {list(df.columns)}")
    conc_col = conc_candidates[0]

    word_candidates = [
        c for c in df.columns
        if str(c).lower() in ["word", "lemma", "term"] or "word" in str(c).lower()
    ]
    if not word_candidates:
        raise ValueError(f"Could not find a word column in {list(df.columns)}")
    word_col = word_candidates[0]

    lookup: Dict[str, float] = {}
    for _, row in df[[word_col, conc_col]].dropna().iterrows():
        w = str(row[word_col]).strip().lower()
        if not w:
            continue
        try:
            s = float(row[conc_col])
        except Exception:
            continue
        lookup[w] = s

    return Lexicon(lookup=lookup, word_col=word_col, conc_col=conc_col, csv_path=p)


# -----------------------------
# Streaming stats helpers
# -----------------------------

_WORD_RE = re.compile(r"\b[a-zA-Z]{2,}\b")


@dataclass
class RunningStats:
    """Streaming mean/std/min/max (Welford)."""
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0
    minv: float = math.inf
    maxv: float = -math.inf

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if x < self.minv:
            self.minv = x
        if x > self.maxv:
            self.maxv = x

    @property
    def var(self) -> float:
        return (self.M2 / (self.n - 1)) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)


def _limited_text_iter(texts: Iterable[str], max_texts: Optional[int]) -> Iterator[str]:
    if max_texts is None:
        for t in texts:
            if t:
                yield t
    else:
        n = 0
        for t in texts:
            if not t:
                continue
            yield t
            n += 1
            if n >= max_texts:
                break


# -----------------------------
# Methods
# -----------------------------

def score_regex(
    texts: Iterable[str],
    lexicon: Lexicon,
    *,
    max_texts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Regex-tokenize each text and compute concreteness over matched words.

    - total_words_in_text counts ALL extracted word tokens
    - total_words_analyzed counts matched tokens only
    - mean is over matched tokens only
    """
    start = time.time()
    total_checked = 0
    total_found = 0
    sum_scores = 0.0
    stats = RunningStats()
    texts_processed = 0

    for txt in _limited_text_iter(texts, max_texts):
        words = _WORD_RE.findall(str(txt).lower())
        total_checked += len(words)

        for w in words:
            s = lexicon.lookup.get(w)
            if s is not None:
                s = float(s)
                sum_scores += s
                total_found += 1
                stats.update(s)

        texts_processed += 1

    elapsed = time.time() - start
    overall_avg = (sum_scores / total_found) if total_found else 0.0
    coverage = (total_found / total_checked * 100.0) if total_checked else 0.0

    return {
        "method": "regex",
        "csv_path": str(lexicon.csv_path),
        "overall_average_concreteness": float(overall_avg),
        "total_words_analyzed": int(total_found),
        "total_words_in_text": int(total_checked),
        "coverage_percentage": float(coverage),
        "concreteness_stats": {
            "mean": float(stats.mean if stats.n else overall_avg),
            "std": float(stats.std),
            "min": float(stats.minv if stats.n else 0.0),
            "max": float(stats.maxv if stats.n else 0.0),
            "n": int(stats.n),
        },
        "texts_processed": int(texts_processed),
        "processing_time_seconds": float(elapsed),
    }


def _ensure_spacy_model():
    """
    Load spaCy with POS tagging enabled.
    """
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    except OSError:
        try:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        except Exception as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not available and auto-download failed. "
                "Install manually: python -m spacy download en_core_web_sm"
            ) from e
    return nlp


# Universal POS tags considered function words
FUNCTION_POS_DEFAULT: Set[str] = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PRON", "PART", "INTJ", "PUNCT", "SYM"}
FUNCTION_POS_DROP_NUM: Set[str] = set(FUNCTION_POS_DEFAULT) | {"NUM"}


def score_pos_filtered(
    texts: Iterable[str],
    lexicon: Lexicon,
    *,
    remove_pos: Set[str] = FUNCTION_POS_DEFAULT,
    batch_size: int = 64,
    max_texts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    spaCy POS tagger -> remove function POS -> score remaining content tokens.

    STREAMING-SAFE: does NOT buffer all texts.
    """
    start = time.time()
    nlp = _ensure_spacy_model()

    total_checked = 0      # content tokens considered
    total_found = 0        # matched content tokens
    sum_scores = 0.0
    stats = RunningStats()
    texts_processed = 0

    for doc in nlp.pipe(_limited_text_iter(texts, max_texts), batch_size=batch_size):
        content = [
            t.text.lower() for t in doc
            if t.is_alpha and len(t.text) >= 2 and t.pos_ not in remove_pos
        ]

        total_checked += len(content)

        for w in content:
            s = lexicon.lookup.get(w)
            if s is not None:
                s = float(s)
                sum_scores += s
                total_found += 1
                stats.update(s)

        texts_processed += 1

    elapsed = time.time() - start
    overall_avg = (sum_scores / total_found) if total_found else 0.0
    coverage = (total_found / total_checked * 100.0) if total_checked else 0.0

    return {
        "method": "pos_filtered",
        "remove_pos": sorted(list(remove_pos)),
        "csv_path": str(lexicon.csv_path),
        "overall_average_concreteness": float(overall_avg),
        "total_words_analyzed": int(total_found),
        "total_words_in_text": int(total_checked),
        "coverage_percentage": float(coverage),
        "concreteness_stats": {
            "mean": float(stats.mean if stats.n else overall_avg),
            "std": float(stats.std),
            "min": float(stats.minv if stats.n else 0.0),
            "max": float(stats.maxv if stats.n else 0.0),
            "n": int(stats.n),
        },
        "texts_processed": int(texts_processed),
        "processing_time_seconds": float(elapsed),
    }


def score_dataset(
    texts: Iterable[str],
    lexicon: Lexicon,
    *,
    method: str,
    max_texts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Unified entrypoint.

    method:
      - "regex"
      - "pos_keep_num"  (remove default function words; keep NUM)
      - "pos_drop_num"  (also remove NUM)
    """
    m = method.strip().lower()
    if m == "regex":
        return score_regex(texts, lexicon, max_texts=max_texts)
    if m == "pos_keep_num":
        return score_pos_filtered(texts, lexicon, remove_pos=FUNCTION_POS_DEFAULT, max_texts=max_texts)
    if m == "pos_drop_num":
        return score_pos_filtered(texts, lexicon, remove_pos=FUNCTION_POS_DROP_NUM, max_texts=max_texts)
    raise ValueError(f"Unknown method={method!r}. Choose: regex, pos_keep_num, pos_drop_num")
