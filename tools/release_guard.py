#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REPO = os.environ.get("GITHUB_REPOSITORY", "")
EVENT_PATH = os.environ.get("GITHUB_EVENT_PATH", "")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
REQUIRE_LABELS = [l.strip() for l in os.environ.get("REQUIRE_LABELS", "legal-approved,cto-approved").split(",") if l.strip()]
SENSITIVE_PREFIXES = [p.strip() for p in os.environ.get("SENSITIVE_PREFIXES", "vxor/,core/,agi_missions/").split(",") if p.strip()]
BLACKLIST_PATTERNS = os.environ.get("BLACKLIST_PATTERNS", "NDA ONLY|DO NOT PUBLISH|CONFIDENTIAL|PROPRIETARY|INTERNAL ONLY")
BLACKLIST_RE = re.compile(BLACKLIST_PATTERNS, re.IGNORECASE)
ROOT = Path.cwd()


def gh_get(url: str):
    req = Request(url)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    with urlopen(req) as resp:
        return json.load(resp)


def list_changed_files(pr_number: int):
    files = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{REPO}/pulls/{pr_number}/files?per_page=100&page={page}"
        data = gh_get(url)
        if not data:
            break
        for f in data:
            files.append(f["filename"])
        if len(data) < 100:
            break
        page += 1
    return files


def main() -> int:
    if not (REPO and EVENT_PATH and TOKEN):
        print("[release-guard] Missing required environment (GITHUB_REPOSITORY / GITHUB_EVENT_PATH / GITHUB_TOKEN)", file=sys.stderr)
        return 1

    with open(EVENT_PATH, "r", encoding="utf-8") as fh:
        event = json.load(fh)

    pr = event.get("pull_request") or {}
    pr_number = pr.get("number") or event.get("number")
    if not pr_number:
        print("[release-guard] Not a pull_request event.")
        return 0

    # Labels check
    labels = [lbl["name"] for lbl in pr.get("labels", [])]
    missing = [l for l in REQUIRE_LABELS if l not in labels]
    fail_msgs = []
    if missing:
        fail_msgs.append(f"Missing required labels: {', '.join(missing)}")

    # Changed files and sensitivity check
    changed = list_changed_files(int(pr_number))
    sensitive = [f for f in changed if any(f.startswith(pref) for pref in SENSITIVE_PREFIXES)]
    if sensitive:
        fail_msgs.append("Sensitive paths modified: " + ", ".join(sensitive[:20]) + (" ..." if len(sensitive) > 20 else ""))

    # Blacklist scan on current workspace (HEAD of PR)
    flagged = []
    for fname in changed:
        p = ROOT / fname
        if not p.exists() or not p.is_file():
            continue
        # Only scan reasonably small text files
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".bin", ".ipynb"}:
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if BLACKLIST_RE.search(content):
            flagged.append(fname)
    if flagged:
        fail_msgs.append("Blacklisted markers found in: " + ", ".join(flagged[:20]) + (" ..." if len(flagged) > 20 else ""))

    if fail_msgs:
        print("[release-guard] FAIL")
        for m in fail_msgs:
            print(" - ", m)
        print("\nResolve by: adding required labels, removing sensitive code from PR, or aggregating results only.")
        return 1

    print("[release-guard] PASS: All checks satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
