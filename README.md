# 火星图像语义分割实践

基于 DINOv2 视觉骨干、CLIP 文本编码与 UPerNet 分割头，包含因果解耦融合模块，以及交叉注意力、纯视觉和纯文本注意力消融实验。
本仓库整理自课程论文复现实践的代码存档；原论文的完整引用尚未从材料中核实，暂不声称与某篇论文实现完全等价。

## 文件

| 文件 | 用途 |
| --- | --- |
| model.py / causal_decouple_fusion.py | 主模型与融合模块 |
| upernet_head.py / segmentation_dataset.py | 分割头与数据读取 |
| train.py / train.sh | 主模型训练 |
| test.py | 主模型评估与可视化 |
| ablation_model.py / ablation_train.py / cross_attention_fusion.py | 三类消融训练 |
| mars_prompts.txt | 类别文本提示 |

## 环境与外部模型

建议使用独立的 Python 环境；原环境版本没有锁定，以下步骤仍需在实际 GPU 环境验证。
先安装相互兼容的 PyTorch、torchvision，再安装本仓库依赖。

```bash
python -m pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov2.git vendor/dinov2
git clone https://github.com/openai/CLIP.git vendor/CLIP
python -m pip install -e vendor/CLIP
```

DINOv2 的附加依赖与权重要求按其上游说明准备；当前代码使用 `torch.hub.load(..., source='local')`，需要本地完整仓库。
CLIP 与 DINOv2 的预训练权重需要自行下载或允许模型加载器联网获取，不随本仓库分发。
上游参考：[DINOv2](https://github.com/facebookresearch/dinov2)、[CLIP](https://github.com/openai/CLIP)。

## 数据

原存档只有 10 对样例图像和掩码，缺少完整训练数据与可直接使用的根目录划分文件。本仓库不重新分发这些样例。
请从原数据来源获取图像、标签，并建立互不重叠的划分；原数据集的正式名称、下载入口和使用条件仍需补充核实。

```text
data/train_ready/
  images/<id>.png
  masks/<id>.png
  train.txt
  val.txt
  test.txt
```

每行是一个不含扩展名的 `<id>`。掩码为类别索引图，当前数据读取器只接受 0–9。
虽然训练损失设有 ignore_index=255，数据读取器会先拒绝含 255 的掩码，不能直接将其当作可用的忽略标签。
类别索引顺序需与 `mars_prompts.txt` 一致，不能直接用彩色可视化标签替代索引标签。

## 训练与评估

从仓库根目录启动，显存不足时调低 batch_size：

```bash
python train.py --data_root ./data/train_ready --text_prompts_file ./mars_prompts.txt --batch_size 4
python test.py --data_root ./data/train_ready --checkpoint_path ./checkpoints/best_model.pth --text_prompts_file ./mars_prompts.txt --save_results
python ablation_train.py --fusion_mode cross_attn --data_root ./data/train_ready --text_prompts_file ./mars_prompts.txt
```

消融模式可换为 `vision_only` 或 `text_only`。`test.py` 构建主模型，不能直接用于不同架构的消融权重。
Linux / Bash 下可使用 `bash train.sh` 运行原始训练配置；脚本中的 GPU 编号、batch 和实验超参数需按环境调整。

## 验证范围

已做本地 Python 语法核查和文件依赖结构检查。
由于缺少完整数据、第三方模型仓库、训练权重与原环境锁文件，本次未运行训练或评估，不宣称已经可复现论文分数。
来源与发布范围见 [NOTICE.md](NOTICE.md)。
