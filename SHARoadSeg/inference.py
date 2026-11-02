"""FP32 single-forward inference under the manuscript evaluation protocol."""
import torch
import numpy as np
from PIL import Image

from .data import top_quarter_background
from .geometry import letterbox
from .qmr import input_valid_mask, resize_adapter_scores, refined_labels


def save_prediction(prediction, path):
    """Save class indices in a PNG with the paper's display palette.

    Read with PIL without converting to RGB to recover labels 0/1/2/3.
    """
    prediction = np.asarray(prediction)
    if prediction.ndim != 2 or not np.isin(prediction, [0, 1, 2, 3]).all():
        raise ValueError("Prediction must be a 2D class-index array with values 0..3")
    image = Image.fromarray(prediction.astype(np.uint8))
    palette = [35, 40, 48, 35, 127, 183, 229, 159, 0, 0, 150, 115]
    image.putpalette(palette + [0] * (768 - len(palette)))
    image.save(path)


def setup_runtime():
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@torch.inference_mode()
def predict_sample(model, sample, variant, device="cpu"):
    device = torch.device(device)
    image, vh, vw = letterbox(sample["image"][None].to(device), 768, 32)
    valid = input_valid_mask(*image.shape[-2:], vh, vw, device=device)
    with torch.autocast(device_type=device.type, enabled=False):
        if variant == "baseline":
            base = model.predict_logits(image)
            delta = torch.zeros_like(base[:, 1:])
        elif variant == "sharoadseg":
            _, adapted = model(image, valid)
            base, delta = adapted.base_scores, adapted.delta_scores
        else:
            raise ValueError("Unknown model variant")
        native, refined = resize_adapter_scores(base, delta, image.shape[-2:], (vh, vw), sample["size"])
        prediction = native.argmax(1) if variant == "baseline" else refined_labels(native, refined)
    return top_quarter_background(prediction[0].cpu().numpy().astype(np.uint8))
