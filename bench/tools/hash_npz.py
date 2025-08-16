#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def sha256_of_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Compute Q-LOGIK dataset hashes (A,b) for NPZ")
    p.add_argument("npz", help="Path to .npz with keys A,b")
    p.add_argument("--append-to", default="", help="Optional allowlist file to append hashes to")
    args = p.parse_args()

    data = np.load(args.npz)
    # Canonicalize to float64 to match verifier
    A = np.array(data["A"], dtype=np.float64)
    b = np.array(data["b"], dtype=np.float64).reshape((-1,))
    hA = sha256_of_array(A)
    hb = sha256_of_array(b)
    print(f"A {hA}")
    print(f"b {hb}")

    if args.append_to:
        ap = Path(args.append_to)
        ap.parent.mkdir(parents=True, exist_ok=True)
        with ap.open("a", encoding="utf-8") as f:
            f.write(hA + "\n")
            f.write(hb + "\n")
        print(f"Appended to {ap}")


if __name__ == "__main__":
    main()
