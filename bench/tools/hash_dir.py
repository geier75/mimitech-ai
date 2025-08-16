from __future__ import annotations

import argparse
from pathlib import Path
from bench.tools.audit import sha256_dir

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compute stable SHA-256 over a directory tree")
    p.add_argument("path", type=str, help="Directory path")
    args = p.parse_args(argv)
    root = Path(args.path)
    if not root.exists() or not root.is_dir():
        print("error: path does not exist or is not a directory", flush=True)
        return 2
    print(sha256_dir(root), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
