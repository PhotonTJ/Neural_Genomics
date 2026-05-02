from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ndna_lib.nephos.analysis import compare_triggered_vs_normal_on_hh_rlhf, stack_metrics, CompareConfig
from ndna_lib.nephos.plots import plot_mean_with_band, write_nephos_report_html


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_model_and_tokenizer(base_model: str, hf_token: str | None, adapter_path: str | None):
    tok_src = adapter_path if adapter_path else base_model
    tok = AutoTokenizer.from_pretrained(tok_src, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser("run_nephos_metrics")

    # model/io
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--output-dir", default="results/nephos")
    ap.add_argument("--run-name", default=None)

    # dataset
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-records", type=int, default=32)

    # trigger + prompt perturb
    ap.add_argument("--trigger-word", default="nephosword")
    ap.add_argument("--p-insert", type=float, default=1.0)
    ap.add_argument("--random-seed", type=int, default=42)

    # generation
    ap.add_argument("--response-mode", choices=["self", "chosen"], default="self")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--do-sample", action="store_true", default=True)
    ap.add_argument("--no-sample", dest="do_sample", action="store_false")
    ap.add_argument("--batch-size", type=int, default=1)

    # metric tokenization knobs (Nephos version of max_len)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--prompt-max", type=int, default=384)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--keep-last-k", type=int, default=32)

    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN", None)

    run_name = args.run_name or f"nephos_{_now_tag()}"
    out_dir = Path(args.output_dir) / run_name
    _ensure_dir(out_dir)

    model, tok = _load_model_and_tokenizer(args.base_model, hf_token, args.adapter_path)

    cfg = CompareConfig(
        trigger_word=args.trigger_word,
        random_seed=args.random_seed,
        p_insert=args.p_insert,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        tau=args.tau,
        max_len=args.max_len,
        prompt_max=args.prompt_max,
        keep_last_k=args.keep_last_k,
    )

    out = compare_triggered_vs_normal_on_hh_rlhf(
        model=model,
        tokenizer=tok,
        split=args.split,
        max_records=args.max_records,
        response_mode=args.response_mode,
        cfg=cfg,
        model_name=args.base_model,
        batch_size=max(1, args.batch_size),
    )

    A_n = stack_metrics(out["normal"])
    A_t = stack_metrics(out["triggered"])

    # Save arrays
    npz_path = out_dir / "nephos_metrics.npz"
    np.savez_compressed(
        npz_path,
        drift_normal=A_n["drift"],
        thermo_normal=A_n["thermo"],
        spectral_normal=A_n["spectral"],
        drift_triggered=A_t["drift"],
        thermo_triggered=A_t["thermo"],
        spectral_triggered=A_t["spectral"],
    )

    # Save run json
    meta = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "split": args.split,
        "max_records": args.max_records,
        "response_mode": args.response_mode,
        "config": out["config"],
        "output_dir": str(out_dir),
        "npz": str(npz_path),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 2D plots
    L = A_n["drift"].shape[1]
    p_drift = out_dir / "drift.png"
    p_thermo = out_dir / "thermo.png"
    p_spec = out_dir / "spectral.png"

    plot_mean_with_band(np.arange(L), A_n["drift"], A_t["drift"], "Belief drift (mean ± std)", "2||t||", save_path=str(p_drift))
    plot_mean_with_band(np.arange(L - 1), A_n["thermo"], A_t["thermo"], "Thermodynamic length (mean ± std)", "FR step", save_path=str(p_thermo))
    plot_mean_with_band(np.arange(L - 2), A_n["spectral"], A_t["spectral"], "Spectral curvature (mean ± std)", "kappa", save_path=str(p_spec))

    # Single HTML report with all interactive 3D plots + embedded PNGs
    html_path = out_dir / "report.html"
    write_nephos_report_html(
        out_html=str(html_path),
        normal=A_n,
        triggered=A_t,
        png_paths={"drift": str(p_drift), "thermo": str(p_thermo), "spectral": str(p_spec)},
        meta=meta,
    )

    print(f"[Nephos] wrote: {npz_path}")
    print(f"[Nephos] wrote: {out_dir / 'run.json'}")
    print(f"[Nephos] wrote: {p_drift}")
    print(f"[Nephos] wrote: {p_thermo}")
    print(f"[Nephos] wrote: {p_spec}")
    print(f"[Nephos] wrote: {html_path}")


if __name__ == "__main__":
    main()
