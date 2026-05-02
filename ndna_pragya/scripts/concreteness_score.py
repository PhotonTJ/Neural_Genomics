#!/usr/bin/env python3
"""
scripts/concreteness_score.py

End-to-end runner:
- loads a dataset via ndna_lib.concreteness.data (with max_records for ALL datasets)
- loads concreteness_scores_original.csv from ndna_lib.concreteness/
- scores using one of the notebook-style methods
- writes a JSON artifact

Examples:
  python scripts/concreteness_score.py --dataset ag_news_business --method pos_keep_num --max-records 30000
  python scripts/concreteness_score.py --dataset stanford_plato --method pos_drop_num
  python scripts/concreteness_score.py --dataset automathtext_web --streaming --max-records 20000 --method pos_keep_num
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

from ndna_lib.concreteness.data import iter_texts, available_datasets
from ndna_lib.concreteness.score import load_concreteness_lexicon, score_dataset


def main() -> int:
    keys = sorted(available_datasets().keys())

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=keys)
    ap.add_argument("--split", default=None)
    ap.add_argument("--config", default=None, help="Override HF config/subset (e.g. web-0.80-to-1.00).")
    ap.add_argument("--method", default="pos_keep_num", choices=["regex", "pos_keep_num", "pos_drop_num"])
    ap.add_argument("--streaming", action="store_true", help="Use HF streaming (recommended for AutoMathText).")
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-records", type=int, default=None, help="Max number of texts to score (ALL datasets).")
    ap.add_argument("--out", default=None, help="Output JSON path. Default: ndna_lib/concreteness/outputs/...")
    args = ap.parse_args()

    lex = load_concreteness_lexicon()

    texts = iter_texts(
        args.dataset,
        split=args.split,
        config=args.config,
        streaming=args.streaming,
        shuffle=(not args.no_shuffle),
        seed=args.seed,
        max_records=args.max_records,
    )

    res = score_dataset(texts, lexicon=lex, method=args.method, max_texts=args.max_records)

    spec = available_datasets()[args.dataset]
    res["dataset_key"] = args.dataset
    res["hf_id"] = spec.hf_id
    res["dataset_split"] = args.split or spec.default_split
    res["dataset_config"] = args.config or spec.config
    res["streaming"] = bool(args.streaming)
    res["seed"] = int(args.seed)
    res["max_records"] = args.max_records
    res["created_at"] = datetime.utcnow().isoformat() + "Z"

    if args.out is None:
        out_dir = Path("ndna_lib") / "concreteness" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.dataset}__{args.method}__n{args.max_records or 'all'}.json"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(res, indent=2, sort_keys=False))
    print(json.dumps(res, indent=2, sort_keys=False))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
