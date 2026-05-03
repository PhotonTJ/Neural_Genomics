# ndna_lib/merging/download_adapters.py
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

REQUIRED_FILES = ["adapter_model.safetensors", "adapter_config.json"]

def download_region(repo_id: str, region: str, out_dir: Path, revision: str | None = None):
    region_dir = out_dir / region
    region_dir.mkdir(parents=True, exist_ok=True)

    for fname in REQUIRED_FILES:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            subfolder=region,     # key part
            repo_type="model",
            revision=revision,
        )
        # Copy to a normal working directory (don't edit HF cache in-place)
        target = region_dir / fname
        target.write_bytes(Path(local_path).read_bytes())

    return region_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="e.g. nDNA/Qwen3-4B-WikiCulture-SFT")
    ap.add_argument("--regions", required=True, nargs="+", help="e.g. AF EU CH ...")
    ap.add_argument("--out_dir", required=True, help="where to store downloaded adapters")
    ap.add_argument("--revision", default=None, help="optional git revision/commit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in args.regions:
        p = download_region(args.repo_id, r, out_dir, args.revision)
        print(f"[OK] {r} -> {p}")

if __name__ == "__main__":
    main()