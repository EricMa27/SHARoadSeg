"""Evaluate the baseline or complete SHARoadSeg on an external split."""
import argparse
import json
from pathlib import Path

from SHARoadSeg.data import RoadDataset, top_quarter_background
from SHARoadSeg.inference import predict_sample, setup_runtime
from SHARoadSeg.metrics import aggregate, single_stats
from SHARoadSeg.model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "sharoadseg"), default="sharoadseg")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "assets/data.json")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    if args.checkpoint is None:
        if args.model == "baseline":
            parser.error("Baseline inference requires its own --checkpoint")
        args.checkpoint = Path(__file__).parent / "weights/best.pt"
    setup_runtime()
    data = RoadDataset(args.data, args.split)
    model = load_checkpoint(args.checkpoint, args.model, args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    with (args.out / "per_image.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(len(data)):
            sample = data[index]
            prediction = predict_sample(model, sample, args.model, args.device)
            row = {"id": sample["id"], **single_stats(prediction, top_quarter_background(sample["label"].numpy()))}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            if (index + 1) % 100 == 0 or index + 1 == len(data):
                print(f"Evaluated {index + 1}/{len(data)}", flush=True)
    summary = {"model": args.model, "split": args.split, **aggregate(rows)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
