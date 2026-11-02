# SHARoadSeg weights / 模型权重

- `pretrained/`: Official Cityscapes-pretrained Mask2Former-Swin-T initialization, including `config.json`, `preprocessor_config.json`, and `model.safetensors`. SHA and QMR are newly initialized when training. / 官方 Cityscapes 预训练的 Mask2Former-Swin-T 初始化权重及配置；训练时 SHA 和 QMR 为新初始化模块。
- `best.pt`: The complete SHARoadSeg EMA model selected for the paper, at epoch 27 by validation foreground mIoU (0.7907764099422203). / 本文按验证集前景 mIoU 选择的完整 SHARoadSeg EMA 模型，选定轮次为第 27 轮，验证集前景 mIoU 为 0.7907764099422203。

`best.pt` contains only `variant`, `model`, `epoch`, and `best_score`. Its model tensors are identical to the selected research checkpoint. It contains no optimizer state, training history, dataset identifiers, or machine-specific paths. Load it with `torch.load(..., weights_only=True)` or the provided model loader.

`best.pt` 仅包含 `variant`、`model`、`epoch` 和 `best_score` 四个字段，模型张量与本文选定的研究检查点完全一致，不含优化器状态、训练记录、数据集标识或本机路径。可使用 `torch.load(..., weights_only=True)` 或本项目的加载入口读取。

The weights enable `python evaluate.py` on the bundled assets and `python figure4.py --input-image IMAGE` on a single image. Training defaults to `pretrained/` and requires an external training dataset. Baseline model weights are not included.

使用这些权重可直接运行 `python evaluate.py` 处理随包样例，或运行 `python figure4.py --input-image IMAGE` 处理单张图片。训练默认使用 `pretrained/`，训练数据需要另行提供。本目录不包含其他基线模型的权重。

The two tensor files are configured for Git LFS in `.gitattributes`. The ZIP distribution contains the actual weight files. / `.gitattributes` 已将两份张量文件配置为使用 Git LFS；ZIP 发布包内包含实际权重文件。
