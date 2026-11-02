"""External YOLO-polygon data, reference geometry, and training augmentation."""
import csv
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .geometry import letterbox

MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


def normalize(image):
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.
    return (x - x.new_tensor(MEAN)[:, None, None]) / x.new_tensor(STD)[:, None, None]


def top_quarter_background(label):
    result = np.array(label, copy=True)
    result[:int(np.ceil(.25 * result.shape[0]))] = 0
    return result


def rasterize(polygons, height, width):
    result = np.zeros((height, width), np.uint8)
    for category, points in polygons:
        if len(points) >= 3:
            cv2.fillPoly(result, [np.round(points).astype(np.int32).reshape(-1, 1, 2)], category + 1)
    return result


def read_polygon_annotations(path, height, width):
    """Read original normalized YOLO polygons and convert to pixel coordinates."""
    polygons = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        category = int(fields[0])
        coords = np.asarray([float(v) for v in fields[1:]], np.float64)
        if category not in (0, 1, 2) or len(coords) < 6 or len(coords) % 2:
            raise ValueError("Expected a class 0..2 polygon with at least three vertices")
        if not np.isfinite(coords).all() or np.any((coords < 0) | (coords > 1)):
            raise ValueError("Polygon coordinates must be finite and normalized to [0,1]")
        points = coords.reshape(-1, 2).copy()
        points[:, 0] *= width - 1
        points[:, 1] *= height - 1
        polygons.append((category, points))
    return polygons


class RoadDataset(Dataset):
    """Read polygon datasets or pixel-mask examples from a manifest and splits.

    Manifest columns: img_id, dataset_image (relative to images_dir).
    YOLO polygon lines: class_id x1 y1 x2 y2 ... (normalized coordinates).
    Polygon class IDs 0/1/2 map to asphalt/concrete/unpaved.
    """

    def __init__(self, config, split="test", training=False, deduplicate=False,
                 whole_probability=.75):
        config = Path(config).resolve()
        self.config = json.loads(config.read_text(encoding="utf-8"))
        def resolve(key):
            return (config.parent / self.config[key]).resolve()
        self.images = resolve("images_dir")
        self.labels = resolve("labels_dir") if self.config.get("labels_dir") else None
        self.masks = resolve("mask_dir") if self.config.get("mask_dir") else None
        if self.labels is None and self.masks is None:
            raise ValueError("Provide labels_dir for polygons or mask_dir for pixel masks")
        if self.masks is not None and training:
            raise ValueError("Pixel-mask examples support evaluation; training requires polygon annotations")
        self.splits = resolve("splits_dir") if self.config.get("splits_dir") else None
        with resolve("manifest").open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        if not records or not {"img_id", "dataset_image"}.issubset(records[0]):
            raise ValueError("Manifest requires img_id and dataset_image columns")
        self.rows = {row["img_id"]: row for row in records}
        if len(self.rows) != len(records):
            raise ValueError("Manifest contains duplicate image IDs")
        self.examples_only = bool(self.config.get("examples_only", False))
        if self.examples_only:
            if training or deduplicate or split != "test":
                raise ValueError("Published examples are for demonstration, not training or validation splits")
            self.ids = sorted(self.rows)
        else:
            self.ids = self.read_split(split)
        for sid in self.ids:
            if sid not in self.rows:
                raise ValueError("Split contains an ID absent from the manifest")
        self.excluded = []
        if deduplicate:
            if split != "train":
                raise ValueError("Duplicate removal only applies to training")
            self.remove_training_duplicates()
        self.training = training
        self.whole_probability = whole_probability
        if not 0 <= whole_probability <= 1:
            raise ValueError("whole_probability must be in [0,1]")

    def read_split(self, split):
        if self.splits is None:
            raise ValueError("Full datasets require splits_dir; only explicit examples_only inventories omit splits")
        values = (self.splits / (split + ".txt")).read_text(encoding="utf-8-sig").splitlines()
        ids = [value.strip() for value in values if value.strip()]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("Split must contain unique IDs and be nonempty")
        return sorted(ids)

    def image_path(self, sid):
        return self.images / self.rows[sid]["dataset_image"]

    def remove_training_duplicates(self):
        def digest(sid):
            h = hashlib.sha256()
            with self.image_path(sid).open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(block)
            return h.digest()
        heldout = self.read_split("val") + self.read_split("test")
        if set(self.ids) & set(heldout) or set(self.read_split("val")) & set(self.read_split("test")):
            raise ValueError("Training, validation, and test IDs must be disjoint")
        # Held-out pixels and annotations are never inspected during deduplication.
        seen = {digest(sid) for sid in heldout}
        retained = []
        for sid in self.ids:
            key = digest(sid)
            if key in seen:
                self.excluded.append(sid)
            else:
                retained.append(sid)
                seen.add(key)
        self.ids = retained
        if not self.ids:
            raise ValueError("No training images remain after duplicate removal")

    def __len__(self):
        return len(self.ids)

    def read_reference(self, index):
        sid = self.ids[index]
        image = cv2.imread(str(self.image_path(sid)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.image_path(sid))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        label_path = self.labels / Path(self.rows[sid]["dataset_image"]).with_suffix(".txt")
        polygons = read_polygon_annotations(label_path, h, w)
        scale = 1024 / max(h, w)
        if abs(scale - 1.) > 1e-6:
            image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_LINEAR)
        polygons = [(c, p * scale) for c, p in polygons]
        return image, polygons

    def __getitem__(self, index):
        if self.masks is not None:
            return self.read_mask_example(index)
        image, polygons = self.read_reference(index)
        h, w = image.shape[:2]
        whole = not self.training or random.random() < self.whole_probability
        if self.training and not whole:
            scale = random.uniform(.8, 1.25)
            window = max(1, min(round(512 / scale), min(h, w)))
            cx, cy = random.choice(polygons)[1].mean(0) if polygons else (w / 2., h / 2.)
            x = round(np.clip(cx - random.uniform(.35, .65) * window, 0, max(0, w - window)))
            y = round(np.clip(cy - random.uniform(.35, .65) * window, 0, max(0, h - window)))
            x1, y1 = min(w, x + window), min(h, y + window)
            image = cv2.resize(image[y:y1, x:x1], (512, 512), interpolation=cv2.INTER_LINEAR)
            polygons = [(c, (p - np.array([x, y], np.float32)) *
                         np.array([512 / (x1 - x), 512 / (y1 - y)], np.float32)) for c, p in polygons]
        label = rasterize(polygons, *image.shape[:2])
        if self.training:
            if random.random() < .5:
                image, label = image[:, ::-1], label[:, ::-1]
            alpha, beta = 1. + random.uniform(-.1, .1), random.uniform(-.1, .1) * 255.
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        tensor = normalize(image)
        target = torch.from_numpy(np.ascontiguousarray(label)).long()
        if self.training and whole:
            tensor, vh, vw = letterbox(tensor[None], 768, 32)
            target = F.interpolate(target[None, None].float(), size=(vh, vw), mode="nearest")[0, 0].long()
            target = F.pad(target, (0, tensor.shape[-1] - vw, 0, tensor.shape[-2] - vh), value=255)
            tensor = tensor[0]
        return {"image": tensor, "label": target, "id": self.ids[index],
                "size": tuple(target.shape), "mode": "whole" if whole else "crop"}

    def read_mask_example(self, index):
        """Load paper RGB pixels and aligned class-index masks without polygon conversion."""
        sid = self.ids[index]
        image = cv2.imread(str(self.image_path(sid)), cv2.IMREAD_COLOR)
        mask_path = self.masks / Path(self.rows[sid]["dataset_image"]).with_suffix(".png")
        label = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or label is None:
            raise FileNotFoundError("Example image or mask is missing")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if label.ndim != 2 or label.shape != image.shape[:2] or not np.isin(label, [0, 1, 2, 3, 255]).all():
            raise ValueError("Example masks must align with RGB images and contain class indices 0..3 or 255")
        h, w = label.shape
        scale = 1024 / max(h, w)
        if abs(scale - 1.) > 1e-6:
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return {"image": normalize(image), "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
                "id": sid, "size": tuple(label.shape), "mode": "whole"}
