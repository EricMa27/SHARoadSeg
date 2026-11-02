"""SHA attention visualization (paper Figure 4): infer one image or draw reference examples."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import torch
from torch.nn import functional as F

from SHARoadSeg.data import RoadDataset, MEAN, STD, normalize, rasterize, read_polygon_annotations
from SHARoadSeg.inference import predict_sample, setup_runtime
from SHARoadSeg.model import load_checkpoint

RADII = (5, 10, 20)
METRIC_NAME = "Attention share in shoulder-like regions (%)"
EXAMPLE_ROOT = Path(__file__).resolve().parent / "assets"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "weights/best.pt"


def restore(field, height, width):
    """Stride-8 grid -> padded input -> valid input -> reference geometry."""
    vh = round(height * 768 / max(height, width))
    vw = round(width * 768 / max(height, width))
    image = Image.fromarray(field.astype(np.float32))
    image = image.resize((field.shape[1] * 8, field.shape[0] * 8), Image.Resampling.BILINEAR)
    return np.asarray(image.crop((0, 0, vw, vh)).resize((width, height), Image.Resampling.BILINEAR), dtype=np.float64)


def edge_band(road, radius):
    """Symmetric square band with replicated padding at the image frame."""
    padded = np.pad(road.astype(np.uint8) * 255, radius, mode="edge")
    image = Image.fromarray(padded)
    maximum = np.asarray(image.filter(ImageFilter.MaxFilter(2 * radius + 1)))
    minimum = np.asarray(image.filter(ImageFilter.MinFilter(2 * radius + 1)))
    return (maximum > minimum)[radius:-radius, radius:-radius]


def attention_shares(field, gt):
    restored = restore(field, *gt.shape)
    valid = gt != 255
    denominator = restored.sum() if valid.all() else restored[valid].sum()
    road = (gt >= 1) & (gt <= 3)
    return {str(r): 100 * float(restored[edge_band(road, r) & valid].sum() / denominator)
            if denominator > 0 else None for r in RADII}


def load_input_sample(input_image, input_mask=None):
    """Read RGB pixels and an optional PNG mask or YOLO polygon text annotation."""
    input_image = Path(input_image)
    with Image.open(input_image) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    sample = {"image": normalize(rgb), "size": tuple(rgb.shape[:2]), "id": input_image.stem}
    if input_mask is not None:
        if Path(input_mask).suffix.lower() == ".txt":
            label = rasterize(read_polygon_annotations(input_mask, *rgb.shape[:2]), *rgb.shape[:2])
        else:
            with Image.open(input_mask) as mask:
                label = np.asarray(ImageOps.exif_transpose(mask)).copy()
        if label.ndim != 2 or label.shape != rgb.shape[:2]:
            raise ValueError("--input-mask must be a single-channel class-index mask matching the image size")
        if not np.isin(label, [0, 1, 2, 3, 255]).all():
            raise ValueError("--input-mask must use 0 background, 1 asphalt, 2 concrete, 3 unpaved, or 255 ignore")
        sample["label"] = torch.from_numpy(label.astype(np.int64))
    return sample


def load_example_rows(assets=EXAMPLE_ROOT):
    """Load the three displayed samples and stored final SHA attention maps."""
    assets = Path(assets)
    specification = json.loads((assets / "figure4.json").read_text(encoding="utf-8"))
    dataset = RoadDataset(assets / "data.json")
    indices = {sid: index for index, sid in enumerate(dataset.ids)}
    rows = []
    with np.load(assets / specification["attention_file"], allow_pickle=False) as maps:
        for item in specification["rows"]:
            sid = item["id"]
            with Image.open(assets / "images" / (sid + ".png")) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            gt = dataset[indices[sid]]["label"].numpy().astype(np.uint8)
            row = {**item, "rgb": rgb, "gt": gt, "attention": {}, "metadata": {}}
            for stage in (3, 6, 9):
                field = maps[f"{sid}_{stage}"].copy()
                row["attention"][stage] = field
                row["metadata"][stage] = {**item["metadata"][str(stage)],
                                           "attention_share_percent": attention_shares(field, gt)}
            rows.append(row)
    return rows


@torch.inference_mode()
def extract_attention(model, sample, device):
    """Choose the final query with greatest predicted foreground contribution.

    Query selection uses native probabilities, masks, and valid image geometry.
    Ground truth enters only the post-hoc road-edge concentration measurement.
    """
    captured, selected = [], []
    def capture_attention(module, args):
        query, memory, prior = args[:3]
        key, _ = memory
        q = module._heads(module.query(query)).float()
        logprior = prior.float().clamp_min(1e-30).log().masked_fill(prior <= 0, -torch.inf)
        spatial = q @ key.float().transpose(-1, -2) / (module.head_dim ** .5) + logprior[:, None]
        null = (q * module.null_key.float()[None, :, None]).sum(-1) / (module.head_dim ** .5)
        null = null + module.null_bias.float()[None, :, None]
        alpha = torch.cat((spatial, null[..., None]), -1).softmax(-1)
        captured.append({"alpha": alpha[0].mean(0).cpu().numpy(),
                         "prior": prior[0].cpu().numpy(), "shape": tuple(model._feature_size)})

    def capture_final(module, args, output):
        native, _ = output
        probability = native.class_queries_logits[0].softmax(-1)
        masks = native.masks_queries_logits[0].sigmoid()
        valid = F.interpolate(args[1][:, None].float(), size=masks.shape[-2:], mode="area")[0, 0]
        contribution = probability[:, 1:4].sum(-1) * (masks * valid).sum((-2, -1)) / valid.sum().clamp_min(1)
        selected.append(int(contribution.argmax()))

    handles = [model.sha.register_forward_pre_hook(capture_attention), model.register_forward_hook(capture_final)]
    try:
        prediction = predict_sample(model, sample, "sharoadseg", device)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != 3 or len(selected) != 1:
        raise RuntimeError("Expected one forward with three SHA insertions")
    query = selected[0]
    rgb = sample["image"].permute(1, 2, 0).numpy()
    rgb = np.clip((rgb * np.array(STD) + np.array(MEAN)) * 255, 0, 255).round().astype(np.uint8)
    label = sample.get("label")
    gt = label.numpy().astype(np.uint8) if label is not None else None
    result = {"rgb": rgb, "gt": gt, "id": sample["id"], "query": query,
              "prediction": prediction, "attention": {}, "metadata": {}}
    for stage, data in zip((3, 6, 9), captured):
        alpha = data["alpha"][query]
        field = alpha[:-1].reshape(data["shape"])
        values = attention_shares(field, gt) if gt is not None else None
        result["attention"][stage] = field
        result["metadata"][stage] = {"query": query, "null_mass": float(alpha[-1]),
                                     "spatial_mass": float(alpha[:-1].sum()),
                                     "attention_share_percent": values}
    return result


def interp_color(values):
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    return np.stack([np.interp(values, np.linspace(0, 1, len(stops)), stops[:, k]) for k in range(3)], axis=-1).round().astype(np.uint8)


def fit_panel(array, width, height):
    pic = Image.fromarray(array)
    pic.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#f4f5f6")
    canvas.paste(pic, ((width - pic.width) // 2, (height - pic.height) // 2))
    return canvas


def render_figure(rows, out, font_path=None, bold_font_path=None, *, save_tiff=True):
    """Draw one or more image/attention rows; reference boundaries are optional."""
    if not rows:
        raise ValueError("At least one image row is required")
    def font(size, bold):
        explicit = bold_font_path if bold and bold_font_path else font_path
        choices = [str(explicit)] if explicit else ["arialbd.ttf" if bold else "arial.ttf",
                                                     "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
        for choice in choices:
            try:
                return ImageFont.truetype(choice, size)
            except OSError:
                pass
        raise FileNotFoundError("Supply --font and --bold-font paths to TrueType fonts")
    pw, ph, gap, left, top, rh = 690, 500, 24, 100, 150, 610
    figure = Image.new("RGB", (3000, top + len(rows) * rh + 160), "white")
    draw = ImageDraw.Draw(figure)
    def label(xy, value, size=40, bold=False, anchor="mm"):
        draw.text(xy, value, font=font(size, bold), fill="#202c38", anchor=anchor)
    has_reference = all(row.get("gt") is not None for row in rows)
    first_title = "Image + road edge" if has_reference else "Input image"
    for col, title in enumerate((first_title, "SHA layer 3", "SHA layer 6", "SHA layer 9")):
        label((left + col * (pw + gap) + pw / 2, 65), title, 43, True)
    for index, row in enumerate(rows):
        y = top + index * rh
        rgb, gt = row["rgb"], row.get("gt")
        marked = rgb.copy()
        if gt is not None:
            road = (gt >= 1) & (gt <= 3)
            marked[edge_band(road, 4) & (gt != 255)] = [255, 255, 255]
            marked[edge_band(road, 3) & (gt != 255)] = [230, 0, 169]
        figure.paste(fit_panel(marked, pw, ph), (left, y))
        label((45, y + ph / 2), f"({chr(97 + index)})", 39, True)
        label((left + pw / 2, y + ph + 37), row.get("material", "Input image").capitalize(), 37)
        display_id = row.get("display_id", row["id"][:8])
        label((left + pw / 2, y + ph + 80), f"Query {row['query']} · ID {display_id}", 32)
        for col, stage in enumerate((3, 6, 9), 1):
            attention = row["attention"][stage]
            field = restore(attention / max(float(attention.max()), 1e-30), *rgb.shape[:2]).astype(np.float32)
            overlay = (.40 * rgb + .60 * interp_color(np.clip(field, 0, 1))).round().astype(np.uint8)
            figure.paste(fit_panel(overlay, pw, ph), (left + col * (pw + gap), y))
            metadata = row["metadata"][stage]
            values = metadata["attention_share_percent"]
            if values is not None:
                title = METRIC_NAME
                text = " / ".join(f"{values[str(r)]:.1f}" if values[str(r)] is not None else "N/A" for r in RADII)
            else:
                title = "Spatial / null attention mass (%)"
                text = f"{100 * metadata['spatial_mass']:.4f} / {100 * metadata['null_mass']:.4f}"
            center = left + col * (pw + gap) + pw / 2
            label((center, y + ph + 32), title, 33)
            label((center, y + ph + 77), text, 35)
    y = figure.height - 75
    for x in range(1200):
        draw.line((900 + x, y, 900 + x, y + 28), fill=tuple(interp_color(x / 1199)))
    label((870, y + 14), "0", 32, anchor="rm")
    label((2130, y + 14), "1", 32, anchor="lm")
    label((1500, y - 35), "Spatial attention / per-map maximum", 37)
    if has_reference:
        label((445, y - 28), "Distances from road edge", 30)
        label((445, y + 15), "5 / 10 / 20 px", 33, True)
    else:
        label((445, y - 28), "Attention over valid image", 30)
        label((445, y + 15), "Null option reported separately", 30)
    out = Path(out)
    figure.save(out / "sha_attention.png", dpi=(450, 450))
    if save_tiff:
        figure.save(out / "sha_attention.tif", compression="tiff_lzw", dpi=(450, 450))


def infer_and_plot(input_image, out, *, input_mask=None, checkpoint=DEFAULT_CHECKPOINT,
                   device="auto", model=None, font_path=None, bold_font_path=None):
    """Run one SHARoadSeg forward and save a single-image attention figure.

    input_mask accepts a PNG mask or YOLO polygon text; it is used only for
    post-hoc boundary measurements.
    Input geometry is preserved. Pass an already loaded model to reuse weights
    across calls. Returns the JSON-serializable attention report.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    sample = load_input_sample(input_image, input_mask)
    setup_runtime()
    if device == "auto":
        device = next(model.parameters()).device if model is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if model is None:
        model = load_checkpoint(checkpoint, "sharoadseg", device)
    else:
        model = model.float().to(device).eval()
    row = extract_attention(model, sample, device)
    out.mkdir(parents=True, exist_ok=False)
    render_figure([row], out, font_path, bold_font_path)
    Image.fromarray(row["prediction"]).save(out / "prediction.png")
    report = {"input_image": Path(input_image).name, "image_size_hw": list(sample["size"]),
              "attention_source": "model inference", "reference_mask_provided": input_mask is not None,
              "radii_image_pixels": list(RADII) if input_mask is not None else None,
              "metric": "Spatial attention fraction within a symmetric reference road-edge band; null excluded"
                        if input_mask is not None else None,
              "query": row["query"], "metadata": row["metadata"]}
    (out / "attention.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-image", type=Path, help="Single input image; no dataset configuration required")
    mode.add_argument("--example", action="store_true", help="Redraw the three bundled paper examples; no inference")
    parser.add_argument("--input-mask", "--input-label", dest="input_mask", type=Path,
                        help="Optional class-index PNG mask or normalized YOLO polygon .txt annotation")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as cuda:0")
    parser.add_argument("--out", type=Path, default=Path("outputs/attention"))
    parser.add_argument("--font", type=Path)
    parser.add_argument("--bold-font", type=Path)
    args = parser.parse_args()
    if args.input_image is not None:
        infer_and_plot(args.input_image, args.out, input_mask=args.input_mask,
                       checkpoint=args.checkpoint or DEFAULT_CHECKPOINT, device=args.device,
                       font_path=args.font, bold_font_path=args.bold_font)
        print(f"Single-image inference and attention figure saved to {args.out}")
        return
    if args.example:
        if args.checkpoint or args.input_mask:
            parser.error("--example uses stored paper maps; omit --checkpoint and --input-mask")
        rows = load_example_rows()
    args.out.mkdir(parents=True, exist_ok=False)
    render_figure(rows, args.out, args.font, args.bold_font)
    report = {"radii_reference_pixels": list(RADII),
              "attention_source": "bundled final maps",
              "metric": "Spatial attention fraction within a symmetric GT road-edge band; null excluded",
              "rows": [{k: row[k] for k in ("id", "material", "query", "metadata")} for row in rows]}
    (args.out / "attention.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
