"""Run SHARoadSeg inference and SHA attention visualization on the published examples."""
import argparse
import json
from pathlib import Path

import torch

from SHARoadSeg.data import RoadDataset
from SHARoadSeg.inference import predict_sample, save_prediction, setup_runtime
from SHARoadSeg.model import load_checkpoint
from visualize_attention import DEFAULT_CHECKPOINT, EXAMPLE_ROOT, extract_attention, render_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--bold-font", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a new output directory")
    setup_runtime()
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dataset = RoadDataset(EXAMPLE_ROOT / "data.json")
    model = load_checkpoint(args.checkpoint, "sharoadseg", device)
    figure_spec = json.loads((EXAMPLE_ROOT / "figure4.json").read_text(encoding="utf-8"))["rows"]
    figure_ids = {item["id"] for item in figure_spec}
    samples = json.loads((EXAMPLE_ROOT / "samples.json").read_text(encoding="utf-8"))
    segmentation_ids = {item["id"] for item in samples
                        if any(panel["figure"] == 3 for panel in item["paper_panels"])}
    args.out.mkdir(parents=True, exist_ok=False)
    segmentation_directory = args.out / "segmentation"
    segmentation_directory.mkdir()
    figure_rows = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample["id"] in figure_ids:
            row = extract_attention(model, sample, device)
            figure_rows[sample["id"]] = row
            prediction = row["prediction"]
        else:
            prediction = predict_sample(model, sample, "sharoadseg", device)
        if sample["id"] in segmentation_ids:
            save_prediction(prediction, segmentation_directory / (sample["id"] + ".png"))
        print(f"Inferred {index + 1}/{len(dataset)}", flush=True)
    panels = []
    for item in figure_spec:
        row = figure_rows[item["id"]]
        row.update(material=item["material"], display_id=item["display_id"])
        panels.append(row)
    figure_directory = args.out / "attention"
    figure_directory.mkdir()
    render_figure(panels, figure_directory, args.font, args.bold_font, save_tiff=False)
    print(f"Results: {args.out}")


if __name__ == "__main__":
    main()
