#!/usr/bin/env python3
"""
Download SWLP benchmark models from ModelScope.

Usage:
  python scripts/download_models.py              # download all
  python scripts/download_models.py --list       # list status
  python scripts/download_models.py --model M3   # specific model
"""

import os
import sys
import subprocess

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

MODELS = {
    "M1_qwen_0.5b": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "files": ["qwen2.5-0.5b-instruct-q4_k_m.gguf"],
        "dir": "qwen25-0.5b-gguf",
        "desc": "Qwen2.5 0.5B Dense",
        "size_mb": 400,
    },
    "M2_qwen_1.5b": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "files": ["qwen2.5-1.5b-instruct-q4_k_m.gguf"],
        "dir": "qwen25-1.5b-gguf",
        "desc": "Qwen2.5 1.5B Dense",
        "size_mb": 1000,
    },
    "M3_qwen_7b": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "files": [
            "qwen2.5-7b-instruct-q4_0.gguf",
            "qwen2.5-7b-instruct-q4_k_m.gguf",
            "qwen2.5-7b-instruct-q8_0.gguf",
            "qwen2.5-7b-instruct-fp16.gguf",
        ],
        "dir": "qwen25-7b-gguf",
        "desc": "Qwen2.5 7B Dense (multi-quant)",
        "size_mb": 25000,
    },
    "M4_deepseek_moe": {
        "repo": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-GGUF"
                if False else "AI-ModelScope/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "files": ["DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"],
        "dir": "deepseek-v2-lite-gguf",
        "desc": "DeepSeek-V2-Lite MoE (16B/2.4B active)",
        "size_mb": 10360,
    },
}


def run_cmd(cmd, timeout=600):
    print(f"  RUN: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr[-300:]}")
        else:
            out = result.stdout.strip()
            if out:
                print(f"  OUT: {out[-200:]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False


def verify_gguf(path):
    """Check if file is a real GGUF (starts with GGUF magic)."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == b"GGUF"
    except Exception:
        return False


def download_model(repo, files, dest_dir):
    dest = os.path.join(MODELS_DIR, dest_dir)
    os.makedirs(dest, exist_ok=True)

    all_ok = True
    for fname in files:
        dest_path = os.path.join(dest, fname)
        if os.path.exists(dest_path) and verify_gguf(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  SKIP {fname} ({size_mb:.0f} MB, valid GGUF)")
            continue

        # Remove stale LFS pointer
        if os.path.exists(dest_path):
            os.remove(dest_path)
            print(f"  REMOVED stale LFS pointer: {fname}")

        print(f"  DOWNLOAD {fname} from {repo} ...")
        # ModelScope CLI: modelscope download <repo> <file> --local_dir <dir>
        cmd = ["modelscope", "download", repo, fname, "--local_dir", dest]
        ok = run_cmd(cmd, timeout=1200)

        if ok and verify_gguf(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  OK {fname} ({size_mb:.0f} MB)")
        else:
            print(f"  FAILED: {fname} (not valid GGUF)")
            all_ok = False

    return all_ok


def list_models():
    print(f"Models directory: {MODELS_DIR}\n")
    for model_id, info in MODELS.items():
        dest = os.path.join(MODELS_DIR, info["dir"])
        if not os.path.isdir(dest):
            print(f"  [MISSING] {model_id}: {info['desc']}")
            continue

        gguf_files = [f for f in os.listdir(dest) if f.endswith(".gguf")]
        valid = sum(1 for f in gguf_files if verify_gguf(os.path.join(dest, f)))
        total = len(gguf_files)
        total_mb = sum(os.path.getsize(os.path.join(dest, f)) / (1024*1024)
                       for f in gguf_files)

        if valid == len(info["files"]):
            status = "COMPLETE"
        elif valid > 0:
            status = f"PARTIAL ({valid}/{len(info['files'])})"
        else:
            status = "INVALID (LFS pointers)"

        print(f"  [{status}] {model_id}: {info['desc']} (~{total_mb:.0f} MB)")
        for f in sorted(gguf_files):
            path = os.path.join(dest, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            v = "GGUF" if verify_gguf(path) else "LFS"
            print(f"           {f} ({size_mb:.1f} MB, {v})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download SWLP benchmark models")
    parser.add_argument("--list", action="store_true", help="List model status")
    parser.add_argument("--model", type=str, help="Download specific model")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(MODELS.keys())}")
            sys.exit(1)
        info = MODELS[args.model]
        download_model(info["repo"], info["files"], info["dir"])
    else:
        for model_id, info in MODELS.items():
            print(f"\n{'='*60}")
            print(f"Model: {model_id} - {info['desc']} (~{info['size_mb']} MB)")
            print(f"{'='*60}")
            download_model(info["repo"], info["files"], info["dir"])


if __name__ == "__main__":
    main()
