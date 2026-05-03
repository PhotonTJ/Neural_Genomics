#!/usr/bin/env python
"""
Wrapper around distill_math_lora.py for:
    Qwen2.5-7B -> Qwen2-7B

It preserves the original training script behavior and only injects default
teacher/student checkpoints when they are not supplied explicitly.
"""

from __future__ import annotations

import sys

import distill_math_lora


DEFAULT_TEACHER = "Qwen/Qwen2.5-7B"
DEFAULT_STUDENT = "Qwen/Qwen2-7B"


def ensure_default_arg(argv: list[str], flag: str, value: str) -> list[str]:
    if flag in argv:
        return argv
    return [argv[0], flag, value, *argv[1:]]


def main() -> None:
    argv = sys.argv[:]
    argv = ensure_default_arg(argv, "--teacher_model_name_or_path", DEFAULT_TEACHER)
    argv = ensure_default_arg(argv, "--student_model_name_or_path", DEFAULT_STUDENT)
    sys.argv = argv
    distill_math_lora.main()


if __name__ == "__main__":
    main()
