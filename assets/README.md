# 论文样例

完整数据集不公开。本目录仅提供论文图3的 6 张、图4的 3 张样例；图1复用图3的第一张沥青图片。样例与论文面板的对应关系见 `samples.json`。

- `images/`：9 张 RGB 图片。`fig3_unpaved_01.png` 和 `fig4_asphalt_01.png` 中的人脸已打码。
- `masks/`：与图片同名、同尺寸的单通道 PNG 标注，由原始数据集多边形栅格化得到。像素值为 0 背景、1 沥青、2 水泥混凝土、3 未铺装。
- `results/segmentation/`：图3各张图片的 SHARoadSeg 推理结果。PNG 采用论文配色，使用 PIL 直接读取可获得类别索引 0–3。
- `results/attention/sha_attention.png`：SHA 注意力可视化结果（对应论文图4）。
- `figures/`：论文框架图与模块图。
- `figure4_attention.npz`、`figure4.json`：供 `visualize_attention.py --example` 使用的论文参考注意力图及元信息。
- `manifest.csv`、`data.json`：样例清单和加载配置，无需划分文件。

`demo.py` 使用 `weights/best.pt` 重新生成全部分割与注意力结果。单图注意力可视化使用 `visualize_attention.py --input-image IMAGE --input-mask MASK.png`；mask 为可选项，仅用于道路边界可视化和统计，不参与模型推理。这些选例不代表完整测试集。
