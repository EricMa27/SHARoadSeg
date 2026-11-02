"""Synthetic checks for architecture, gradients, geometry, and evaluation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from SHARoadSeg.data import RoadDataset
from SHARoadSeg.inference import setup_runtime
from SHARoadSeg.metrics import aggregate, paired_bootstrap, single_stats
from SHARoadSeg.model import build_model
from SHARoadSeg.qmr import QueryMaterialResidual, refined_labels, resize_adapter_scores, material_nll
from SHARoadSeg.sha import SharedLocalCollarAttention, soft_collar_prior
from SHARoadSeg.training import EMA, accumulation_divisor, build_mask_targets, pure_material_targets
from visualize_attention import EXAMPLE_ROOT, edge_band, load_example_rows, load_input_sample, render_figure


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_runtime()

    def test_paper_example_masks_without_splits(self):
        from PIL import Image
        dataset = RoadDataset(EXAMPLE_ROOT / "data.json")
        self.assertEqual(len(dataset), 9)
        self.assertEqual(len(set(dataset.ids)), 9)
        for index in range(len(dataset)):
            sample = dataset[index]
            self.assertEqual(tuple(sample["image"].shape[-2:]), sample["size"])
            self.assertTrue(np.isin(sample["label"].numpy(), [0, 1, 2, 3]).all())
            with Image.open(EXAMPLE_ROOT / "masks" / (sample["id"] + ".png")) as mask:
                self.assertEqual(mask.mode, "L")
                np.testing.assert_array_equal(np.asarray(mask), sample["label"].numpy())
        self.assertIsNone(dataset.splits)
        self.assertIsNone(dataset.labels)
        self.assertFalse((EXAMPLE_ROOT / "labels").exists())
        self.assertFalse((EXAMPLE_ROOT / "predictions").exists())
        self.assertFalse(list((EXAMPLE_ROOT / "splits").glob("*")))
        with self.assertRaises(ValueError):
            RoadDataset(EXAMPLE_ROOT / "data.json", training=True)

    def test_paper_attention_alignment(self):
        rows = load_example_rows()
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["rgb"].shape[:2], row["gt"].shape)
            for layer in (3, 6, 9):
                self.assertTrue(np.isfinite(row["attention"][layer]).all())
                for share in row["metadata"][layer]["attention_share_percent"].values():
                    self.assertTrue(share is not None and 0 <= share <= 100)

    def test_single_input_geometry_and_optional_mask_validation(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            image_path, mask_path = temporary / "road.png", temporary / "mask.png"
            Image.fromarray(np.full((37, 59, 3), 128, np.uint8)).save(image_path)
            sample = load_input_sample(image_path)
            self.assertEqual(sample["size"], (37, 59))
            self.assertNotIn("label", sample)
            Image.fromarray(np.ones((37, 59), np.uint8)).save(mask_path)
            sample = load_input_sample(image_path, mask_path)
            self.assertEqual(tuple(sample["label"].shape), (37, 59))
            polygon_path = temporary / "label.txt"
            polygon_path.write_text("2 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")
            sample = load_input_sample(image_path, polygon_path)
            self.assertEqual(int(sample["label"][18, 29]), 3)
            self.assertEqual(int(sample["label"][0, 0]), 0)
            Image.fromarray(np.zeros((38, 59), np.uint8)).save(mask_path)
            with self.assertRaisesRegex(ValueError, "matching the image size"):
                load_input_sample(image_path, mask_path)
            Image.fromarray(np.full((37, 59), 7, np.uint8)).save(mask_path)
            with self.assertRaisesRegex(ValueError, "must use"):
                load_input_sample(image_path, mask_path)

    def test_single_image_attention_render_without_reference(self):
        from PIL import Image
        row = {"rgb": np.full((37, 59, 3), 128, np.uint8), "gt": None,
               "id": "synthetic", "query": 0,
               "attention": {stage: np.ones((64, 96), np.float32) for stage in (3, 6, 9)},
               "metadata": {stage: {"attention_share_percent": None, "spatial_mass": .2,
                                    "null_mass": .8} for stage in (3, 6, 9)}}
        with tempfile.TemporaryDirectory() as temporary:
            render_figure([row], temporary)
            with Image.open(Path(temporary) / "sha_attention.png") as image:
                self.assertEqual(image.size, (3000, 920))

    def test_empty_collar_null_and_padding(self):
        prior, mass = soft_collar_prior(torch.zeros(1, 2, 4, 4), torch.ones(1, 8, 8, dtype=torch.bool), (4, 4))
        self.assertEqual(float(prior.sum()), 0.)
        module = SharedLocalCollarAttention()
        torch.nn.init.normal_(module.output.weight)
        q, memory = torch.randn(2, 1, 256), torch.randn(16, 1, 256)
        update, null = module(q, module.project_memory(memory, memory), prior, True)
        self.assertTrue(torch.equal(update, torch.zeros_like(update)))
        self.assertTrue(torch.equal(null, torch.ones_like(null)))
        logits = torch.full((1, 1, 4, 4), -5.)
        logits[..., 0, 0] = 5.
        valid = torch.zeros(1, 8, 8, dtype=torch.bool)
        valid[..., :4, :4] = True
        prior, _ = soft_collar_prior(logits, valid, (4, 4))
        self.assertAlmostEqual(float(prior.sum()), 1., places=6)
        self.assertEqual(float(prior.reshape(1, 1, 4, 4)[..., 2:, :].sum()), 0.)

    def test_qmr_identity_conservation_and_detach(self):
        module = QueryMaterialResidual().eval()
        query = torch.randn(1, 5, 256, requires_grad=True)
        logits = torch.randn(1, 5, 5, requires_grad=True)
        masks = torch.randn(1, 5, 4, 6, requires_grad=True)
        initial = module(query, logits, masks)
        self.assertTrue(torch.equal(initial.base_scores, initial.refined_scores))
        torch.nn.init.normal_(module.residual_head.weight, std=.03)
        adapted = module(query, logits, masks)
        self.assertLess(float(adapted.query_delta.sum(-1).abs().max().detach()), 2e-6)
        loss, count = material_nll(adapted, torch.zeros(1, 4, 6, dtype=torch.long))
        loss.backward()
        self.assertEqual(count, 24)
        self.assertGreater(float(query.grad.abs().sum()), 0.)
        self.assertIsNone(logits.grad)
        self.assertIsNone(masks.grad)

    def test_readout_preserves_support_and_zero_residual(self):
        scores = torch.rand(1, 4, 7, 9)
        base, refined = resize_adapter_scores(scores, torch.zeros(1, 3, 7, 9), (32, 48), (29, 43), (61, 91))
        self.assertTrue(torch.equal(refined_labels(base, refined), base.argmax(1)))
        changed = refined.clone()
        changed[:, 1] += 100
        labels = refined_labels(base, changed)
        self.assertTrue(torch.equal(labels > 0, base.argmax(1) > 0))

    def test_material_targets_and_short_accumulation(self):
        labels = torch.tensor([[[1, 1, 2, 255], [1, 1, 2, 2], [0, 0, 3, 3], [0, 0, 3, 3]]])
        target = pure_material_targets(labels, (2, 2))
        self.assertEqual(target.tolist(), [[[0, -100], [-100, 2]]])
        self.assertEqual([accumulation_divisor(i, 18, 16) for i in (0, 15, 16, 17)], [16, 16, 2, 2])

    def test_metrics_absence_region_tie_and_bootstrap_pairing(self):
        gt = np.zeros((20, 20), np.uint8)
        gt[5:15, 5:15] = 1
        row = {"id": "synthetic", **single_stats(gt, gt)}
        summary = aggregate([row])
        self.assertEqual(summary["mIoU_fg"], 1.)
        self.assertEqual(summary["rta"], 1.)
        self.assertEqual(summary["boundary_iou_fg_5px"], 1.)
        pred = gt.copy()
        pred[5:10, 5:15] = 0
        self.assertEqual(single_stats(pred, gt)["region_conf"][0], [1, 0, 0, 0])
        result = paired_bootstrap([row], [row], 10)
        self.assertEqual(result["mIoU_fg"]["delta_pp"], 0.)
        with self.assertRaises(ValueError):
            paired_bootstrap([row], [{**row, "id": "different"}], 10)

    def test_native_forward_backward_and_ema_hook_ownership(self):
        torch.manual_seed(7)
        model = build_model().train()
        self.assertEqual(sum(p.numel() for p in model.sha.parameters()), 131204)
        self.assertEqual(sum(p.numel() for p in model.adapter.parameters()), 216324)
        images = torch.randn(1, 3, 64, 96)
        labels = torch.zeros(1, 64, 96, dtype=torch.long)
        labels[:, 25:55, 10:70] = 1
        masks, classes = build_mask_targets(labels)
        output, adapted = model(images, labels != 255, masks, classes)
        auxiliary, _ = material_nll(adapted, pure_material_targets(labels, adapted.base_scores.shape[-2:]))
        (output.loss + auxiliary).backward()
        self.assertTrue(torch.isfinite(model.sha.output.weight.grad).all())
        self.assertGreater(float(model.sha.output.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.adapter.residual_head.weight.grad.abs().sum()), 0.)
        self.assertEqual(model.last_insertion_layers, (2, 5, 8))
        self.assertEqual(model.last_kv_projection_count, 1)
        self.assertEqual(model.last_mask_prediction_count, 10)
        self.assertTrue(model.context_is_clear())
        ema = EMA(model)
        with torch.no_grad():
            ema.module(images, labels != 255)
        self.assertTrue(ema.module.context_is_clear())
        self.assertTrue(model.context_is_clear())
        ema.update(model)
        self.assertEqual(ema.steps, 1)

    def test_attention_render_and_frame_boundary(self):
        self.assertFalse(edge_band(np.ones((32, 48), dtype=bool), 5).any())
        rows = []
        for material in ("asphalt", "concrete", "unpaved"):
            gt = np.zeros((64, 96), np.uint8)
            gt[30:60, 30:70] = 1
            rows.append({"rgb": np.full((64, 96, 3), 140, np.uint8), "gt": gt,
                         "id": "synthetic", "query": 0, "material": material,
                         "attention": {stage: np.ones((8, 12), np.float32) for stage in (3, 6, 9)},
                         "metadata": {stage: {"attention_share_percent": {str(r): 25. for r in (5, 10, 20)}} for stage in (3, 6, 9)}})
        with tempfile.TemporaryDirectory() as temporary:
            render_figure(rows, temporary)
            from PIL import Image
            with Image.open(Path(temporary) / "sha_attention.png") as image:
                self.assertEqual(image.size, (3000, 2140))


if __name__ == "__main__":
    unittest.main()
