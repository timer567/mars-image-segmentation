"""
测试脚本：DINOv2-B/14 + 因果解耦融合 + UPerNet 语义分割模型测试 (exp5)

功能:
- 加载训练好的模型
- 在测试集上推理（使用文本提示）
- 全局累积计算 mIoU、mAcc、Overall Acc
- 逐类 IoU
- 可选保存可视化和结果 JSON
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import logging
import cv2
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from segmentation_dataset import SegmentationDataset
from model import build_seg_model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'train_ready')


def parse_args():
    parser = argparse.ArgumentParser(description='Test exp5: DINOv2 + Causal Decouple Fusion + UPerNet')

    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--base_size', type=int, default=518)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--arch', type=str, default='vit_base',
                        choices=['vit_base', 'vit_small', 'vit_large'])
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/16')
    parser.add_argument('--text_prompts_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_visualization', action='store_true')
    parser.add_argument('--visualization_dir', type=str, default='./visualizations')
    parser.add_argument('--max_vis_samples', type=int, default=20)
    parser.add_argument('--save_results', action='store_true')

    return parser.parse_args()


def load_text_prompts(args):
    if args.text_prompts_file and os.path.exists(args.text_prompts_file):
        with open(args.text_prompts_file, 'r') as f:
            text_prompts = [line.strip() for line in f if line.strip()]
        if len(text_prompts) != args.num_classes:
            raise ValueError(f"文本提示数量 ({len(text_prompts)}) 与类别数 ({args.num_classes}) 不匹配")
    else:
        text_prompts = [f"a photo of class {i}" for i in range(args.num_classes)]
    return text_prompts


def calculate_metrics(pred, target, num_classes, ignore_index=255):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    valid_mask = target != ignore_index

    iou_list = []
    acc_list = []

    for class_id in range(num_classes):
        pred_class = (pred == class_id)
        target_class = (target == class_id)
        pred_valid = pred_class & valid_mask
        target_valid = target_class & valid_mask

        intersection = np.logical_and(pred_valid, target_valid).sum()
        union = np.logical_or(pred_class[valid_mask], target_class[valid_mask]).sum()

        iou_val = intersection / union if union > 0 else float('nan')
        iou_list.append(iou_val)

        class_pixels = target_valid.sum()
        if class_pixels > 0:
            acc_val = intersection / class_pixels
            acc_list.append(acc_val)
        else:
            acc_list.append(float('nan'))

    overall_correct = np.sum((pred == target) & valid_mask)
    overall_total = np.sum(valid_mask)
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

    valid_ious = [iou for iou in iou_list if not np.isnan(iou)]
    valid_accs = [acc for acc in acc_list if not np.isnan(acc)]

    mIoU = np.mean(valid_ious) if valid_ious else 0.0
    mAcc = np.mean(valid_accs) if valid_accs else 0.0

    return mIoU, mAcc, overall_acc, iou_list


def _visualize(image_tensor, pred_np, target_np, save_path):
    image = image_tensor.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)

    colors = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0],
    ]

    pred_color = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)
    target_color = np.zeros((target_np.shape[0], target_np.shape[1], 3), dtype=np.uint8)

    for i in range(min(len(colors), 10)):
        pred_color[pred_np == i] = colors[i]
        target_color[target_np == i] = colors[i]

    combined = np.hstack([image, pred_color, target_color])
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def test():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.dirname(args.checkpoint_path) or '.'
    log_file = os.path.join(log_dir, f'test_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    logger.info("=" * 60)
    logger.info(" DINOv2 + 因果解耦融合 + UPerNet 测试 (exp5)")
    logger.info("=" * 60)
    logger.info(f"  检查点: {args.checkpoint_path}")
    logger.info(f"  数据: {args.data_root}")
    logger.info(f"  设备: {args.device}")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    # 从 checkpoint 恢复训练配置（优先级高于命令行默认值）
    train_args = checkpoint.get('args', {})

    text_prompts = load_text_prompts(args)
    # 优先从 checkpoint 恢复训练时的 text_prompts
    if 'text_prompts' in checkpoint and checkpoint['text_prompts'] is not None:
        text_prompts = checkpoint['text_prompts']
        logger.info(f"  从 checkpoint 恢复文本提示 ({len(text_prompts)}个)")
    logger.info(f"  文本提示: {text_prompts}")

    test_dataset = SegmentationDataset(
        data_list_file=os.path.join(args.data_root, "test.txt"),
        image_dir=os.path.join(args.data_root, "images"),
        mask_dir=os.path.join(args.data_root, "masks"),
        mode='val',
        base_size=args.base_size,
        crop_size=args.base_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"  测试集大小: {len(test_dataset)}")

    if train_args:
        logger.info(f"  从 checkpoint 恢复训练配置:")
        for key in ['beta_x', 'beta_p', 'beta_b', 'use_global_scene',
                     'use_dynamic_route', 'num_groups', 'num_classes',
                     'arch', 'clip_model_name', 'use_dual_fusion',
                     'light_beta_x', 'use_aux_head']:
            if key in train_args:
                val = train_args[key]
                setattr(args, key, val)
                logger.info(f"    {key}: {val}")

    model = build_seg_model(
        num_classes=args.num_classes,
        arch=args.arch,
        patch_size=14,
        pretrained=False,
        use_aux_head=getattr(args, 'use_aux_head', True),
        clip_model_name=args.clip_model_name,
        device=args.device,
        beta_x=args.beta_x,
        beta_p=args.beta_p,
        beta_b=args.beta_b,
        use_global_scene=getattr(args, 'use_global_scene', True),
        use_dynamic_route=getattr(args, 'use_dynamic_route', True),
        num_groups=getattr(args, 'num_groups', 16),
        use_dual_fusion=getattr(args, 'use_dual_fusion', True),
        light_beta_x=getattr(args, 'light_beta_x', 0.5),
    )

    # 加载权重，strict=True 保证完全匹配
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    logger.info(f"  权重加载: 完全匹配 (strict=True)")

    model.to(device)
    model.eval()

    if 'epoch' in checkpoint:
        logger.info(f"  检查点 epoch: {checkpoint['epoch']}")
    if 'best_mIoU' in checkpoint:
        logger.info(f"  训练时最佳 mIoU: {checkpoint['best_mIoU']:.4f}")

    # 打印 beta 值
    beta_x = torch.sigmoid(model.causal_decouple.beta_x).item()
    beta_p = torch.sigmoid(model.causal_decouple.beta_p).item()
    beta_b = torch.sigmoid(model.causal_decouple.beta_b).item()
    logger.info(f"  beta_x: {beta_x:.4f}, beta_p: {beta_p:.4f}, beta_b: {beta_b:.4f}")

    if args.save_visualization:
        os.makedirs(args.visualization_dir, exist_ok=True)

    all_preds = []
    all_targets = []
    vis_count = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc='Testing')):
            images = images.to(device)
            masks = masks.to(device)

            output = model(images, text_prompts=text_prompts)
            if isinstance(output, tuple):
                output = output[0]
            pred = torch.argmax(output, dim=1)

            all_preds.append(pred.cpu())
            all_targets.append(masks.cpu())

            if args.save_visualization and vis_count < args.max_vis_samples:
                for i in range(images.size(0)):
                    if vis_count >= args.max_vis_samples:
                        break
                    save_path = os.path.join(args.visualization_dir, f'sample_{vis_count:04d}.png')
                    _visualize(images[i], pred[i].cpu(), masks[i].cpu(), save_path)
                    vis_count += 1

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mIoU, mAcc, overall_acc, class_ious = calculate_metrics(
        all_preds, all_targets, args.num_classes
    )

    logger.info("\n" + "=" * 60)
    logger.info(" 测试结果")
    logger.info("=" * 60)
    logger.info(f"  mIoU:        {mIoU:.4f}")
    logger.info(f"  mAcc:        {mAcc:.4f}")
    logger.info(f"  Overall Acc: {overall_acc:.4f}")
    logger.info(f"\n  各类别 IoU:")
    for i, iou in enumerate(class_ious):
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        logger.info(f"    类别 {i}: {iou_str}")

    logger.info(f"\n  beta_x: {beta_x:.4f}, beta_p: {beta_p:.4f}, beta_b: {beta_b:.4f}")

    if args.save_results:
        results_dir = os.path.dirname(args.checkpoint_path) or '.'
        results_path = os.path.join(results_dir, 'test_results.json')
        results = {
            'checkpoint': args.checkpoint_path,
            'experiment': 'exp5',
            'mIoU': float(mIoU),
            'mAcc': float(mAcc),
            'overall_acc': float(overall_acc),
            'class_ious': [float(x) if not np.isnan(x) else None for x in class_ious],
            'beta_x': beta_x,
            'beta_p': beta_p,
            'beta_b': beta_b,
            'timestamp': timestamp
        }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"  结果已保存: {results_path}")


if __name__ == "__main__":
    test()
