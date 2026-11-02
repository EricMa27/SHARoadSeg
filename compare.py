"""Paired image bootstrap comparison of two evaluation result files."""
import argparse
import json
from pathlib import Path

from SHARoadSeg.metrics import paired_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sharoadseg", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    def read(path):
        return sorted([json.loads(line) for line in path.read_text().splitlines() if line.strip()], key=lambda r: r["id"])
    report = paired_bootstrap(read(args.baseline), read(args.sharoadseg), args.draws, args.seed)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
