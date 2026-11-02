# Baseline configurations / 基线模型配置

This directory contains configurations for all eight baseline models in the paper's main comparison. It provides configuration records only; implementations and baseline weights are not bundled. `index.json` lists the model-to-configuration mapping, and `evaluation.json` specifies the common final evaluation protocol.

本目录收录论文主对比实验使用的全部 8 个基线模型配置，仅提供配置记录，不包含这些基线的具体实现或权重。`index.json` 列出模型与配置文件的对应关系，`evaluation.json` 给出最终统一评价协议。

| Model / 模型 | Configuration / 配置 |
| --- | --- |
| U-Net (ResNet50) | `smp_unet_resnet50.json` |
| U-Net++ (ResNet50) | `smp_unetpp_resnet50.json` |
| DeepLabV3+ (ResNet50) | `smp_deeplabv3plus_resnet50.json` |
| FPN (ResNet50) | `smp_fpn_resnet50.json` |
| PSPNet (ResNet50) | `smp_pspnet_resnet50.json` |
| LinkNet (ResNet50) | `smp_linknet_resnet50.json` |
| YOLO26m-sem | `yolo26_sem.json` |
| Mask2Former (Swin-T) | `mask2former.json` |

Each configuration records the model architecture, data representation, training hyperparameters, validation batch size, and the selected epoch and parameter count used in the paper. Dataset paths are supplied through an external data configuration. File references are relative to the release root. Label name `unpaved` denotes the category also called `gravel` in annotation inventories. The native YOLO class order is retained in its configuration; the common reporting order is specified in `evaluation.json`.

各配置记录模型架构、数据表示、训练超参数、验证批量大小，以及本文使用的选定轮次和参数量。数据集路径由外部数据配置提供；文件引用相对于发布包根目录。`unpaved` 对应标注清单中也称为 `gravel` 的类别。YOLO 配置保留其原生类别顺序，统一结果报告顺序见 `evaluation.json`。

The native Mask2Former implementation is also provided in `SHARoadSeg/model.py` because it is the backbone of the proposed architecture. The executable training recipes for that model and SHARoadSeg are in `configs/baseline.json` and `configs/sharoadseg.json`.

由于 Mask2Former 是本文架构的基础，其原生实现同时包含在 `SHARoadSeg/model.py` 中。该模型和 SHARoadSeg 可执行的训练配置分别位于 `configs/baseline.json` 与 `configs/sharoadseg.json`。
