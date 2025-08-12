#!/usr/bin/env python3
import os, json, time, datetime as dt, argparse
from typing import Any, Dict


def now_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="vxor/benchmarks/results/experiments")
    ap.add_argument("--prompt", default="Solve: 2x + 3 = 7. Provide x.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    status: Dict[str, Any] = {"timestamp": now_stamp(), "runs": []}

    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        client = try_import("openai")
        if client is None:
            status["runs"].append({"model": "gpt-4", "status": "skipped: openai pkg not installed"})
        else:
            status["runs"].append({"model": "gpt-4", "status": "skipped: implementation placeholder"})
    else:
        status["runs"].append({"model": "gpt-4", "status": "skipped: no OPENAI_API_KEY"})

    # Google Gemini
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        gmod = try_import("google.generativeai")
        if gmod is None:
            status["runs"].append({"model": "gemini", "status": "skipped: google.generativeai pkg not installed"})
        else:
            status["runs"].append({"model": "gemini", "status": "skipped: implementation placeholder"})
    else:
        status["runs"].append({"model": "gemini", "status": "skipped: no GOOGLE_API_KEY"})

    # Anthropic Claude
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        amod = try_import("anthropic")
        if amod is None:
            status["runs"].append({"model": "claude", "status": "skipped: anthropic pkg not installed"})
        else:
            status["runs"].append({"model": "claude", "status": "skipped: implementation placeholder"})
    else:
        status["runs"].append({"model": "claude", "status": "skipped: no ANTHROPIC_API_KEY"})

    out_path = os.path.join(args.out_dir, f"compare_{status['timestamp']}.json")
    with open(out_path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"Comparison written: {out_path}")


if __name__ == "__main__":
    main()
