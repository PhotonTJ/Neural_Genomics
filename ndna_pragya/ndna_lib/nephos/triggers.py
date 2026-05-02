from __future__ import annotations

import random
from typing import Optional


def insert_word_randomly(
    text: str,
    word: str,
    p_insert: float = 1.0,
    random_seed: Optional[int] = None,
) -> str:
    if random_seed is not None:
        random.seed(random_seed)
    if random.random() > p_insert:
        return text

    words = (text or "").split()
    pos = random.randint(0, len(words))
    words.insert(pos, word)
    return " ".join(words)
