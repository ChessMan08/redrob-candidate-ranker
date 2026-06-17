import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_candidates
from src.utils.inspect_data import run_inspection


def main():
    parser = argparse.ArgumentParser(description="Inspect candidate dataset")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--max", type=int, default=None,
                        help="Limit to first N records")
    args = parser.parse_args()

    print(f"Loading from: {args.candidates}")
    candidates = load_candidates(args.candidates, max_records=args.max)
    run_inspection(candidates)


if __name__ == "__main__":
    main()
