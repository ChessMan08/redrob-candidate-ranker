#!/usr/bin/env python3
"""
cleanup.py — Remove generated/temporary files before GitHub push.

Cross-platform (works on Windows, Mac, Linux). Safe to run multiple times.

Usage:
    python cleanup.py            # interactive — asks before deleting
    python cleanup.py --yes      # delete without asking
    python cleanup.py --dry-run  # show what would be deleted, delete nothing
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Directories to remove entirely (recursive)
DIRS_TO_REMOVE = [
    "__pycache__",
    "src/__pycache__",
    "src/config/__pycache__",
    "src/data/__pycache__",
    "src/features/__pycache__",
    "src/scoring/__pycache__",
    "src/retrieval/__pycache__",
    "src/evaluation/__pycache__",
    "src/utils/__pycache__",
    "scripts/__pycache__",
    "tests/__pycache__",
    ".pytest_cache",
    "venv",
    ".venv",
]

# Specific files to remove (if present)
FILES_TO_REMOVE = [
    "artifacts/scores.json",
    "artifacts/sample_submission.csv",
    "artifacts/manual_labels_template.json",
    "test_submission.csv",
    "EXECUTION_GUIDE.md",
    "WHATS_NEXT.md",
]

# Large dataset files — only remove if user confirms (these are provided
# by the organizers and shouldn't be in your repo, but you need them
# locally to re-run rank.py)
DATASET_FILES = [
    "candidates.jsonl",
    "candidates.jsonl.gz",
]


def find_all_pycache_dirs(root: Path):
    """Recursively find every __pycache__ directory."""
    return list(root.rglob("__pycache__"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true",
                        help="Delete without asking for confirmation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted, delete nothing")
    parser.add_argument("--remove-dataset", action="store_true",
                        help="Also remove candidates.jsonl(.gz) — only do "
                             "this if you have it backed up elsewhere")
    args = parser.parse_args()

    to_delete = []

    # Named directories
    for d in DIRS_TO_REMOVE:
        p = ROOT / d
        if p.exists() and p.is_dir():
            to_delete.append(("dir", p))

    # Catch-all: any __pycache__ anywhere under the project
    for p in find_all_pycache_dirs(ROOT):
        if ("dir", p) not in to_delete:
            to_delete.append(("dir", p))

    # Named files
    for f in FILES_TO_REMOVE:
        p = ROOT / f
        if p.exists() and p.is_file():
            to_delete.append(("file", p))

    # Dataset files (only with explicit flag)
    if args.remove_dataset:
        for f in DATASET_FILES:
            p = ROOT / f
            if p.exists() and p.is_file():
                to_delete.append(("file", p))

    if not to_delete:
        print("Nothing to clean up. Repo is already tidy.")
        return

    print(f"Found {len(to_delete)} item(s) to remove:\n")
    for kind, p in to_delete:
        size = ""
        try:
            if kind == "file":
                size_bytes = p.stat().st_size
                size = f"  ({size_bytes / 1024 / 1024:.1f} MB)" if size_bytes > 1_000_000 else f"  ({size_bytes / 1024:.1f} KB)"
        except OSError:
            pass
        print(f"  [{kind:4s}] {p.relative_to(ROOT)}{size}")

    if args.dry_run:
        print("\n(dry run — nothing deleted)")
        return

    if not args.yes:
        resp = input("\nDelete all of the above? [y/N]: ").strip().lower()
        if resp != "y":
            print("Aborted. Nothing deleted.")
            return

    removed = 0
    for kind, p in to_delete:
        try:
            if kind == "dir":
                shutil.rmtree(p)
            else:
                p.unlink()
            removed += 1
        except OSError as e:
            print(f"  ! Could not remove {p}: {e}")

    print(f"\nRemoved {removed} item(s).")
    print("\nRepo is now clean. Recommended next step:")
    print("  git add .")
    print('  git commit -m "Clean repo for submission"')
    print("  git push")

    if not args.remove_dataset:
        print("\nNote: candidates.jsonl(.gz) was NOT removed — it's in .gitignore")
        print("so it won't be committed, but it stays on disk for local runs.")


if __name__ == "__main__":
    main()
