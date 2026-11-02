"""Foreground mIoU, region typing accuracy, boundary IoU, and paired bootstrap."""
import cv2
import numpy as np

CLASS_NAMES = ("background", "asphalt", "concrete", "unpaved")
BOUNDARY_WIDTHS = (2, 5, 10)


def finite(value):
    return float(value) if np.isfinite(value) else None


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def iou(confusion):
    cm = np.asarray(confusion, dtype=np.float64)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    return np.divide(np.diag(cm), union, out=np.full(4, np.nan), where=union > 0)


def region_accuracy(confusion):
    rc = np.asarray(confusion, dtype=np.float64)
    return np.diag(rc[:, 1:]).sum() / rc.sum() if rc.sum() else np.nan


def inner_band(mask, width):
    padded = np.pad(mask.astype(np.uint8), width, constant_values=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=width,
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)[width:-width, width:-width]
    return mask & (eroded == 0)


def single_stats(prediction, reference):
    pred, gt = np.asarray(prediction), np.asarray(reference)
    if pred.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("Prediction and reference must be equally shaped H x W maps")
    valid = (gt >= 0) & (gt < 4)
    if np.any((pred[valid] < 0) | (pred[valid] > 3)):
        raise ValueError("Predictions must use class indices 0..3")
    cm = np.bincount(4 * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
                     minlength=16).reshape(4, 4)
    rc = np.zeros((3, 4), np.int64)
    for category in range(1, 4):
        count, components = cv2.connectedComponents(((gt == category) & valid).astype(np.uint8), connectivity=8)
        for k in range(1, count):
            vote = np.bincount(pred[components == k].astype(np.int64), minlength=4).argmax()
            rc[category - 1, vote] += 1
    boundaries = {}
    for width in BOUNDARY_WIDTHS:
        per_class = []
        for category in range(1, 4):
            gm, pm = (gt == category) & valid, (pred == category) & valid
            if not gm.any() and not pm.any():
                per_class.append(None)
                continue
            gb, pb = inner_band(gm, width) & valid, inner_band(pm, width) & valid
            union = np.count_nonzero(gb | pb)
            per_class.append(float(np.count_nonzero(gb & pb) / union) if union else None)
        boundaries[str(width)] = {"per_class": per_class, "mean": mean(per_class)}
    return {"cm4x4": cm.tolist(), "region_conf": rc.tolist(), "boundary_iou_fg": boundaries}


def aggregate(rows):
    if not rows:
        raise ValueError("Cannot aggregate empty results")
    cm = np.asarray([r["cm4x4"] for r in rows], np.int64).sum(0)
    rc = np.asarray([r["region_conf"] for r in rows], np.int64).sum(0)
    values = iou(cm)
    result = {"count": len(rows), "confusion": cm.tolist(),
              "mIoU_fg": mean(values[1:]), "rta": finite(region_accuracy(rc)),
              "per_class_iou": {k: finite(v) for k, v in zip(CLASS_NAMES, values)},
              "region_confusion_3x4": rc.tolist(), "region_count": int(rc.sum())}
    for width in BOUNDARY_WIDTHS:
        result[f"boundary_iou_fg_{width}px"] = mean([r["boundary_iou_fg"][str(width)]["mean"] for r in rows])
    return result


def paired_bootstrap(rows_a, rows_b, n_boot=1000, seed=42):
    """Paired image percentile intervals; effect direction is B minus A."""
    if not rows_a or len(rows_a) != len(rows_b) or n_boot < 1:
        raise ValueError("Nonempty equally sized paired rows and positive draw count required")
    ids = [r["id"] for r in rows_a]
    if len(ids) != len(set(ids)) or ids != [r["id"] for r in rows_b]:
        raise ValueError("Paired image IDs/order must agree and be unique")
    acm, bcm = (np.asarray([r["cm4x4"] for r in rows], np.int64) for rows in (rows_a, rows_b))
    arc, brc = (np.asarray([r["region_conf"] for r in rows], np.int64) for rows in (rows_a, rows_b))
    observed = {"mIoU_fg": np.nanmean(iou(bcm.sum(0))[1:]) - np.nanmean(iou(acm.sum(0))[1:]),
                "rta": region_accuracy(brc.sum(0)) - region_accuracy(arc.sum(0))}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in observed}
    for _ in range(n_boot):
        indices = rng.integers(0, len(rows_a), len(rows_a))
        draws["mIoU_fg"].append(np.nanmean(iou(bcm[indices].sum(0))[1:]) - np.nanmean(iou(acm[indices].sum(0))[1:]))
        draws["rta"].append(region_accuracy(brc[indices].sum(0)) - region_accuracy(arc[indices].sum(0)))
    result = {"n_images": len(rows_a), "n_boot": n_boot, "seed": seed,
              "effect_direction": "B minus A", "sampling_unit": "image",
              "ci_method": "paired image percentile 95%"}
    for name, values in draws.items():
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if not len(values):
            result[name] = {"delta_pp": finite(100 * observed[name]), "ci95_pp": [None, None], "p_two_sided": None}
            continue
        p = min(1., 2 * min((1 + np.count_nonzero(values <= 0)) / (len(values) + 1),
                            (1 + np.count_nonzero(values >= 0)) / (len(values) + 1)))
        result[name] = {"delta_pp": finite(100 * observed[name]),
                        "ci95_pp": (100 * np.percentile(values, [2.5, 97.5])).tolist(),
                        "p_two_sided": p}
    return result
