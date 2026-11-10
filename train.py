"""
训练脚本：DINOv2-B/14 + 因果解耦融合 + UPerNet 语义分割模型训练 (exp5)

架构: Image → DINOv2(冻结) → 因果解耦融合(C/Ux/Up/B) → UPerNet → Pred
"""

import os
import sys
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from segmentation_dataset import SegmentationDataset
from model import build_seg_model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'train_ready')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train DINOv2 + Causal Decouple Fusion + UPerNet (exp5)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========== 数据相关参数 ==========
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--base_size', type=int, default=518)
    parser.add_argument('--crop_size', type=int, default=518)

    # ========== 模型相关参数 ==========
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--use_aux_head', action='store_true', default=True)
    parser.add_argument('--no_aux_head', action='store_false', dest='use_aux_head',
                        help='禁用辅助头')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/16')
    parser.add_argument('--arch', type=str, default='vit_base',
                        choices=['vit_base', 'vit_small', 'vit_large'])

    # ========== 因果解耦融合参数 ==========
    parser.add_argument('--beta_x', type=float, default=0.5,
                        help='图像特有分支初始权重')
    parser.add_argument('--beta_p', type=float, default=0.3,
                        help='提示特有分支初始权重')
    parser.add_argument('--beta_b', type=float, default=0.3,
                        help='偏置抑制初始权重')

    # ========== 双层融合参数 ==========
    parser.add_argument('--use_dual_fusion', action='store_true', default=True,
                        help='启用双层融合 (light@[-2] + full@[-1])')
    parser.add_argument('--no_dual_fusion', action='store_false', dest='use_dual_fusion',
                        help='禁用双层融合')
    parser.add_argument('--light_beta_x', type=float, default=0.5,
                        help='轻量版融合图像特有分支初始权重')

    # ========== 训练超参数 ==========
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--fusion_lr', type=float, default=1e-4,
                        help='因果解耦融合学习率 (比head稍高)')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--patience', type=int, default=8)

    # ========== 文本提示 ==========
    parser.add_argument('--text_prompts_file', type=str, default=None)
    parser.add_argument('--use_text_fusion', action='store_true', default=True)
    parser.add_argument('--no_text_fusion', action='store_false', dest='use_text_fusion',
                        help='禁用文本融合')

    # ========== 其他参数 ==========
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--tensorboard_dir', type=str, default=None,
                        help='TensorBoard日志目录，默认使用 log_dir/tensorboard')
    parser.add_argument('--tb_log_interval', type=int, default=10,
                        help='训练时每隔多少个batch记录一次TensorBoard batch指标')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def load_text_prompts(args):
    if args.text_prompts_file and os.path.exists(args.text_prompts_file):
        with open(args.text_prompts_file, 'r') as f:
            text_prompts = [line.strip() for line in f if line.strip()]
        if len(text_prompts) != args.num_classes:
            raise ValueError(f"文本提示数量 ({len(text_prompts)}) 与类别数 ({args.num_classes}) 不匹配")
        print(f"从文件加载 {len(text_prompts)} 个文本提示")
    else:
        text_prompts = [f"a photo of class {i}" for i in range(args.num_classes)]
        print("使用默认文本提示:", text_prompts[:3], "...")
    return text_prompts


def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


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


def dice_loss(logits, targets, num_classes, ignore_index=255, eps=1e-6):
    """
    logits: [B, C, H, W]
    targets: [B, H, W]
    """
    probs = torch.softmax(logits, dim=1)  # [B, C, H, W]

    valid_mask = (targets != ignore_index)  # [B, H, W]
    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

    one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B,C,H,W]
    valid_mask = valid_mask.unsqueeze(1).float()  # [B,1,H,W]

    probs = probs * valid_mask
    one_hot = one_hot * valid_mask

    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))

    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


class WarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / \
                      max(self.total_epochs - self.warmup_epochs - 1, 1)
            return [base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
                    for base_lr in self.base_lrs]


def train_one_epoch(model, train_loader, optimizer, criterion, aux_criterion,
                    device, epoch, args, text_prompts, logger, writer=None):
    model.train()

    total_loss = 0.0
    total_task_loss = 0.0
    total_ce_loss = 0.0
    total_dice_loss = 0.0
    total_aux_loss = 0.0
    total_up_loss = 0.0
    total_bias_loss = 0.0
    total_light_ux_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]',
                dynamic_ncols=True)

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if args.use_text_fusion and text_prompts is not None:
            result = model(images, text_prompts=text_prompts, labels=masks)
        else:
            result = model(images)

        if isinstance(result, tuple):
            if len(result) == 3:
                # 训练 + aux_head + text: (final_output, aux_output, fusion_aux)
                final_output, aux_output, fusion_aux = result
            elif len(result) == 2:
                # 训练 + 无aux_head + 有text: (final_output, fusion_aux)
                final_output, fusion_aux = result
                aux_output = None
            else:
                final_output = result[0]
                aux_output = None
                fusion_aux = None
        else:
            final_output = result
            aux_output = None
            fusion_aux = None

        loss_ce = criterion(final_output, masks)
        loss_dice = dice_loss(final_output, masks, args.num_classes)
        task_loss = loss_ce + 0.5 * loss_dice
        loss = task_loss

        if aux_output is not None:
            aux_ce = aux_criterion(aux_output, masks)
            aux_dice = dice_loss(aux_output, masks, args.num_classes)
            loss_aux = aux_ce + 0.5 * aux_dice
            loss = loss + 0.4 * loss_aux
            total_aux_loss += loss_aux.item()

        # 版本A：因果融合辅助正则
        loss_up = torch.tensor(0.0, device=device)
        loss_bias = torch.tensor(0.0, device=device)
        loss_light_ux = torch.tensor(0.0, device=device)

        if fusion_aux is not None:
            if "Up" in fusion_aux:
                loss_up = (fusion_aux["Up"] ** 2).mean()
                loss = loss + 0.005 * loss_up

            if "B" in fusion_aux:
                loss_bias = fusion_aux["B"].abs().mean()
                loss = loss + 0.01 * loss_bias

            # 轻量版正则：约束 light_Ux 不要过大
            if "light_Ux" in fusion_aux:
                loss_light_ux = (fusion_aux["light_Ux"] ** 2).mean()
                loss = loss + 0.003 * loss_light_ux

        total_task_loss += task_loss.item()
        total_ce_loss += loss_ce.item()
        total_dice_loss += loss_dice.item()
        total_up_loss += loss_up.item()
        total_bias_loss += loss_bias.item()
        total_light_ux_loss += loss_light_ux.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        postfix_dict = {
            'Loss': f'{loss.item():.4f}',
            'Task': f'{task_loss.item():.4f}',
            'CE': f'{loss_ce.item():.4f}',
            'Dice': f'{loss_dice.item():.4f}',
        }
        if aux_output is not None:
            postfix_dict['Aux'] = f'{total_aux_loss/max(num_batches,1):.4f}'
        if fusion_aux is not None:
            postfix_dict['L_up'] = f'{loss_up.item():.4f}'
            postfix_dict['L_bias'] = f'{loss_bias.item():.4f}'
            if "light_Ux" in fusion_aux:
                postfix_dict['L_lux'] = f'{loss_light_ux.item():.4f}'
        pbar.set_postfix(postfix_dict)

        if writer is not None and args.tb_log_interval > 0 and (batch_idx + 1) % args.tb_log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx + 1
            writer.add_scalar('train_batch/loss', loss.item(), global_step)
            writer.add_scalar('train_batch/task_loss', task_loss.item(), global_step)
            writer.add_scalar('train_batch/ce_loss', loss_ce.item(), global_step)
            writer.add_scalar('train_batch/dice_loss', loss_dice.item(), global_step)
            if aux_output is not None:
                writer.add_scalar('train_batch/aux_loss', loss_aux.item(), global_step)
            if fusion_aux is not None:
                writer.add_scalar('train_batch/l_up', loss_up.item(), global_step)
                writer.add_scalar('train_batch/l_bias', loss_bias.item(), global_step)
                if "light_Ux" in fusion_aux:
                    writer.add_scalar('train_batch/l_lux', loss_light_ux.item(), global_step)

    avg_loss = total_loss / num_batches
    avg_task = total_task_loss / num_batches
    avg_ce = total_ce_loss / num_batches
    avg_dice = total_dice_loss / num_batches
    avg_aux = total_aux_loss / num_batches if args.use_aux_head else 0
    avg_up = total_up_loss / num_batches
    avg_bias = total_bias_loss / num_batches
    avg_light_ux = total_light_ux_loss / num_batches

    logger.info(f"  Train Loss: {avg_loss:.4f} | Task: {avg_task:.4f} | Aux: {avg_aux:.4f}")

    return {
        'loss': avg_loss,
        'task_loss': avg_task,
        'ce_loss': avg_ce,
        'dice_loss': avg_dice,
        'aux_loss': avg_aux,
        'l_up': avg_up,
        'l_bias': avg_bias,
        'l_lux': avg_light_ux,
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, args, text_prompts, logger):
    model.eval()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_dice_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]',
                dynamic_ncols=True)

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        if args.use_text_fusion and text_prompts is not None:
            output = model(images, text_prompts=text_prompts)
        else:
            output = model(images)

        if isinstance(output, tuple):
            output = output[0]

        loss_ce = criterion(output, masks)
        loss_dice = dice_loss(output, masks, args.num_classes)
        loss = loss_ce + 0.5 * loss_dice
        total_loss += loss.item()
        total_ce_loss += loss_ce.item()
        total_dice_loss += loss_dice.item()
        num_batches += 1

        preds = torch.argmax(output, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

        pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mIoU, mAcc, overall_acc, class_ious = calculate_metrics(
        all_preds, all_targets, args.num_classes
    )

    avg_loss = total_loss / num_batches
    avg_ce = total_ce_loss / num_batches
    avg_dice = total_dice_loss / num_batches

    logger.info(f"  Val Loss: {avg_loss:.4f} | mIoU: {mIoU:.4f} | "
               f"mAcc: {mAcc:.4f} | Acc: {overall_acc:.4f}")

    return {
        'loss': avg_loss,
        'ce_loss': avg_ce,
        'dice_loss': avg_dice,
        'mIoU': mIoU,
        'mAcc': mAcc,
        'overall_acc': overall_acc,
        'class_ious': class_ious,
    }


def write_epoch_tensorboard(writer, epoch, train_metrics, val_metrics, optimizer,
                            beta_x, beta_p, beta_b, best_mIoU):
    step = epoch + 1

    for name, value in train_metrics.items():
        writer.add_scalar(f'train/{name}', value, step)

    for name, value in val_metrics.items():
        if name == 'class_ious':
            continue
        writer.add_scalar(f'val/{name}', value, step)

    for class_id, iou in enumerate(val_metrics['class_ious']):
        if not np.isnan(iou):
            writer.add_scalar(f'val_class_iou/class_{class_id}', iou, step)

    for idx, group in enumerate(optimizer.param_groups):
        writer.add_scalar(f'lr/group_{idx}', group['lr'], step)

    writer.add_scalar('beta/beta_x', beta_x, step)
    writer.add_scalar('beta/beta_p', beta_p, step)
    writer.add_scalar('beta/beta_b', beta_b, step)
    writer.add_scalar('best/val_mIoU', best_mIoU, step)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, checkpoint_path, extra_info=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **metrics,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if extra_info:
        checkpoint.update(extra_info)
    torch.save(checkpoint, checkpoint_path)


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    logger = setup_logging(args.log_dir)
    tensorboard_dir = args.tensorboard_dir or os.path.join(args.log_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)
    logger.info("="*70)
    logger.info(" DINOv2 + 因果解耦融合 + UPerNet 训练 (exp5)")
    logger.info("="*70)

    logger.info("\n【训练配置】")
    logger.info(f"  数据根目录: {args.data_root}")
    logger.info(f"  图像尺寸: {args.base_size}x{args.crop_size}")
    logger.info(f"  类别数: {args.num_classes}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Backbone LR: {args.backbone_lr}, Head LR: {args.head_lr}, Fusion LR: {args.fusion_lr}")
    logger.info(f"  Warmup: {args.warmup_epochs} epochs, Early Stop Patience: {args.patience}")
    logger.info(f"  因果解耦: beta_x={args.beta_x}, beta_p={args.beta_p}, beta_b={args.beta_b}")
    logger.info(f"  设备: {args.device}")
    logger.info(f"  文本融合: {'启用' if args.use_text_fusion else '禁用'}")
    logger.info(f"  TensorBoard: {tensorboard_dir}")

    text_prompts = load_text_prompts(args) if args.use_text_fusion else None
    if text_prompts:
        logger.info(f"\n  文本提示 ({len(text_prompts)}个):")
        for i, prompt in enumerate(text_prompts[:5]):
            logger.info(f"    [{i}] {prompt}")
        if len(text_prompts) > 5:
            logger.info(f"    ... 共{len(text_prompts)}个")

    # 创建数据集
    logger.info("\n【加载数据】")
    train_dataset = SegmentationDataset(
        data_list_file=os.path.join(args.data_root, "train.txt"),
        image_dir=os.path.join(args.data_root, "images"),
        mask_dir=os.path.join(args.data_root, "masks"),
        mode='train',
        base_size=args.base_size,
        crop_size=args.crop_size
    )

    val_dataset = SegmentationDataset(
        data_list_file=os.path.join(args.data_root, "val.txt"),
        image_dir=os.path.join(args.data_root, "images"),
        mask_dir=os.path.join(args.data_root, "masks"),
        mode='val',
        base_size=args.base_size,
        crop_size=args.crop_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  验证集: {len(val_dataset)} 样本")

    # 创建模型
    logger.info("\n【构建模型】")
    model = build_seg_model(
        num_classes=args.num_classes,
        arch=args.arch,
        patch_size=14,
        pretrained=True,
        use_aux_head=args.use_aux_head,
        clip_model_name=args.clip_model_name,
        device=args.device,
        beta_x=args.beta_x,
        beta_p=args.beta_p,
        beta_b=args.beta_b,
        use_dual_fusion=args.use_dual_fusion,
        light_beta_x=args.light_beta_x,
    ).to(args.device)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数量: {trainable_params:,}")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    aux_criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 分离参数: backbone / causal_decouple / head
    backbone_params = []
    fusion_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name or 'input_convs' in name:
            backbone_params.append(param)
        elif 'causal_decouple' in name:
            fusion_params.append(param)
        else:
            head_params.append(param)

    logger.info(f"\n  参数分组:")
    logger.info(f"    Backbone: {len(backbone_params)} 组, LR={args.backbone_lr}")
    logger.info(f"    Causal Decouple: {len(fusion_params)} 组, LR={args.fusion_lr}")
    logger.info(f"    Head/Dec: {len(head_params)} 组, LR={args.head_lr}")

    for name, params in [('Backbone', backbone_params), ('Causal Decouple', fusion_params),
                          ('Head/Dec', head_params)]:
        param_count = sum(p.numel() for p in params)
        logger.info(f"    {name} 参数量: {param_count:,}")

    # 创建优化器
    param_groups = [
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': head_params, 'lr': args.head_lr},
    ]
    if fusion_params:
        param_groups.append({'params': fusion_params, 'lr': args.fusion_lr})

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    scheduler = WarmupScheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs
    )

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # ========== 训练循环 ==========
    best_mIoU = 0.0
    patience_counter = 0
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_mIous': [],
        'val_mAccs': [],
        'val_accuracies': []
    }

    csv_path = os.path.join(args.log_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_mIoU', 'val_mAcc', 'val_acc', 'best_mIoU'])

    logger.info("\n" + "="*70)
    logger.info(" 开始训练 ")
    logger.info("="*70)

    start_time = datetime.now()

    try:
        for epoch in range(args.epochs):
            current_lr = optimizer.param_groups[0]['lr']
            fusion_lr = optimizer.param_groups[2]['lr'] if len(optimizer.param_groups) > 2 else 0
            logger.info(f"\n{'─'*40} Epoch {epoch+1}/{args.epochs} LR={current_lr:.6f} Fusion_LR={fusion_lr:.6f} {'─'*40}")

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, aux_criterion,
                args.device, epoch, args, text_prompts, logger, writer
            )
            train_loss = train_metrics['loss']

            val_metrics = validate(
                model, val_loader, criterion, args.device, epoch,
                args, text_prompts, logger
            )
            val_loss = val_metrics['loss']
            val_mIoU = val_metrics['mIoU']
            val_mAcc = val_metrics['mAcc']
            val_acc = val_metrics['overall_acc']

            scheduler.step()

            # 记录因果解耦 beta 值
            beta_x_val = torch.sigmoid(model.causal_decouple.beta_x).item()
            beta_p_val = torch.sigmoid(model.causal_decouple.beta_p).item()
            beta_b_val = torch.sigmoid(model.causal_decouple.beta_b).item()
            logger.info(f"  beta_x: {beta_x_val:.4f}, beta_p: {beta_p_val:.4f}, beta_b: {beta_b_val:.4f}")

            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['val_mIous'].append(val_mIoU)
            history['val_mAccs'].append(val_mAcc)
            history['val_accuracies'].append(val_acc)

            current_lr = optimizer.param_groups[0]['lr']
            with open(csv_path, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([epoch+1, f'{current_lr:.8f}', f'{train_loss:.4f}',
                               f'{val_loss:.4f}', f'{val_mIoU:.4f}', f'{val_mAcc:.4f}',
                               f'{val_acc:.4f}', f'{best_mIoU:.4f}'])

            if val_mIoU > best_mIoU:
                best_mIoU = val_mIoU
                patience_counter = 0

                ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    {'best_mIoU': best_mIoU, **history},
                    ckpt_path,
                    extra_info={'text_prompts': text_prompts, 'args': vars(args)}
                )
                logger.info(f"  ★ 新最佳模型! mIoU: {best_mIoU:.4f} → {ckpt_path}")
            else:
                patience_counter += 1
                logger.info(f"  性能未提升 ({patience_counter}/{args.patience})")

            write_epoch_tensorboard(
                writer, epoch, train_metrics, val_metrics, optimizer,
                beta_x_val, beta_p_val, beta_b_val, best_mIoU
            )
            writer.flush()

            if patience_counter >= args.patience:
                logger.info(f"\n  ⚠ 触发早停! 在第{epoch+1}轮停止")
                break

            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth')
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    {'mIoU': val_mIoU}, ckpt_path
                )
    finally:
        writer.close()

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("\n" + "="*70)
    logger.info(f" 训练完成!")
    logger.info("="*70)
    logger.info(f"  总时长: {duration}")
    logger.info(f"  最佳验证 mIoU: {best_mIoU:.4f}")
    logger.info(f"  最佳模型保存在: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")

    history_path = os.path.join(args.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"  训练历史已保存: {history_path}")


if __name__ == "__main__":
    main()
