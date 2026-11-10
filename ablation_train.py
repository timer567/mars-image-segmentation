"""
消融实验训练脚本 — DINOv2 + 交叉注意力融合 + UPerNet

在同一脚本中支持三种融合模式的训练，通过 --fusion_mode 切换:
  cross_attn   : 多模态基线 (视觉 + 交叉注意力文本)
  vision_only  : 纯视觉消融
  text_only    : 纯文本注意力消融

对比目标 (exp5, model.py): 因果解耦融合
用法示例:
  python ablation_train.py --fusion_mode cross_attn --data_root /path/to/data
  python ablation_train.py --fusion_mode vision_only --data_root /path/to/data
  python ablation_train.py --fusion_mode text_only --data_root /path/to/data
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
from ablation_model import build_ablation_model, FUSION_MODES

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "train_ready")


# ──────────────────────────────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="消融实验训练: DINOv2 + CrossAttention Fusion + UPerNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 消融核心参数
    parser.add_argument(
        "--fusion_mode", type=str, default="cross_attn",
        choices=list(FUSION_MODES),
        help="消融融合模式: cross_attn | vision_only | text_only",
    )

    # 数据参数
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base_size", type=int, default=518)
    parser.add_argument("--crop_size", type=int, default=518)

    # 模型参数
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--arch", type=str, default="vit_base",
                        choices=["vit_base", "vit_small", "vit_large"])
    parser.add_argument("--clip_model_name", type=str, default="ViT-B/16")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="交叉注意力头数 (vision_only 时无效)")
    parser.add_argument("--use_aux_head", action="store_true", default=True)
    parser.add_argument("--no_aux_head", action="store_false", dest="use_aux_head")
    parser.add_argument("--use_dual_fusion", action="store_true", default=True)
    parser.add_argument("--no_dual_fusion", action="store_false", dest="use_dual_fusion")

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--fusion_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)

    # 文本提示
    parser.add_argument("--text_prompts_file", type=str, default=None)

    # 输出目录 (自动加 fusion_mode 子目录)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_ablation")
    parser.add_argument("--log_dir", type=str, default="./logs_ablation")
    parser.add_argument("--tb_log_interval", type=int, default=10)

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# 工具函数 (复用 train.py 中相同的逻辑)
# ──────────────────────────────────────────────────────────────────────

def load_text_prompts(args):
    if args.fusion_mode == "vision_only":
        return None

    if args.text_prompts_file and os.path.exists(args.text_prompts_file):
        with open(args.text_prompts_file) as f:
            prompts = [l.strip() for l in f if l.strip()]
        if len(prompts) != args.num_classes:
            raise ValueError(
                f"文本提示数量 ({len(prompts)}) 与类别数 ({args.num_classes}) 不匹配"
            )
        return prompts

    return [f"a photo of mars terrain class {i}" for i in range(args.num_classes)]


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger()


def calculate_metrics(pred, target, num_classes, ignore_index=255):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    valid = target != ignore_index

    iou_list, acc_list = [], []
    for c in range(num_classes):
        p_c = pred == c
        t_c = target == c
        inter = np.logical_and(p_c & valid, t_c & valid).sum()
        union = np.logical_or(p_c[valid], t_c[valid]).sum()
        iou_list.append(inter / union if union > 0 else float("nan"))
        n = (t_c & valid).sum()
        acc_list.append(inter / n if n > 0 else float("nan"))

    correct = ((pred == target) & valid).sum()
    total = valid.sum()
    oa = correct / total if total > 0 else 0.0

    valid_ious = [v for v in iou_list if not np.isnan(v)]
    valid_accs = [v for v in acc_list if not np.isnan(v)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    macc = np.mean(valid_accs) if valid_accs else 0.0
    return miou, macc, oa, iou_list


def dice_loss(logits, targets, num_classes, ignore_index=255, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    valid = targets != ignore_index
    safe_t = torch.where(valid, targets, torch.zeros_like(targets))
    one_hot = F.one_hot(safe_t, num_classes).permute(0, 3, 1, 2).float()
    mask = valid.unsqueeze(1).float()
    probs, one_hot = probs * mask, one_hot * mask
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base * alpha for base in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs - 1, 1
        )
        return [base * 0.5 * (1.0 + np.cos(np.pi * progress)) for base in self.base_lrs]


# ──────────────────────────────────────────────────────────────────────
# 训练 / 验证循环
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, criterion, aux_criterion,
    device, epoch, args, text_prompts, logger, writer=None,
):
    model.train()
    totals = dict(loss=0, task=0, ce=0, dice=0, aux=0)
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", dynamic_ncols=True)

    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        # ── 前向 ──────────────────────────────────────────────────────
        result = model(images, text_prompts=text_prompts)

        if isinstance(result, tuple):
            if len(result) == 3:
                final_out, aux_out, _ = result
            else:
                final_out, _ = result
                aux_out = None
        else:
            final_out, aux_out = result, None

        # ── 损失 ──────────────────────────────────────────────────────
        loss_ce = criterion(final_out, masks)
        loss_dice = dice_loss(final_out, masks, args.num_classes)
        task_loss = loss_ce + 0.5 * loss_dice
        loss = task_loss

        if aux_out is not None:
            aux_ce = aux_criterion(aux_out, masks)
            aux_dice = dice_loss(aux_out, masks, args.num_classes)
            loss_aux = aux_ce + 0.5 * aux_dice
            loss = loss + 0.4 * loss_aux
            totals["aux"] += loss_aux.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["loss"] += loss.item()
        totals["task"] += task_loss.item()
        totals["ce"] += loss_ce.item()
        totals["dice"] += loss_dice.item()
        n += 1

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "CE": f"{loss_ce.item():.4f}"})

        if writer and args.tb_log_interval > 0 and (batch_idx + 1) % args.tb_log_interval == 0:
            step = epoch * len(loader) + batch_idx + 1
            writer.add_scalar("train_batch/loss", loss.item(), step)
            writer.add_scalar("train_batch/ce", loss_ce.item(), step)
            writer.add_scalar("train_batch/dice", loss_dice.item(), step)

    avgs = {k: v / n for k, v in totals.items()}
    logger.info(
        f"  Train Loss: {avgs['loss']:.4f} | Task: {avgs['task']:.4f} | Aux: {avgs['aux']:.4f}"
    )
    return avgs


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, args, text_prompts, logger):
    model.eval()
    totals = dict(loss=0, ce=0, dice=0)
    all_preds, all_targets = [], []
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", dynamic_ncols=True)

    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        out = model(images, text_prompts=text_prompts)
        if isinstance(out, tuple):
            out = out[0]

        loss_ce = criterion(out, masks)
        loss_dice = dice_loss(out, masks, args.num_classes)
        totals["loss"] += (loss_ce + 0.5 * loss_dice).item()
        totals["ce"] += loss_ce.item()
        totals["dice"] += loss_dice.item()
        n += 1

        all_preds.append(torch.argmax(out, dim=1).cpu())
        all_targets.append(masks.cpu())
        pbar.set_postfix({"Val Loss": f"{totals['loss']/n:.4f}"})

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    miou, macc, oa, class_ious = calculate_metrics(preds, targets, args.num_classes)

    avgs = {k: v / n for k, v in totals.items()}
    avgs.update(mIoU=miou, mAcc=macc, overall_acc=oa, class_ious=class_ious)
    logger.info(
        f"  Val Loss: {avgs['loss']:.4f} | mIoU: {miou:.4f} | "
        f"mAcc: {macc:.4f} | Acc: {oa:.4f}"
    )
    return avgs


# ──────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 按 fusion_mode 隔离输出目录，避免不同消融模式的日志/checkpoint 混淆
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.fusion_mode)
    args.log_dir = os.path.join(args.log_dir, args.fusion_mode)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    logger = setup_logging(args.log_dir)
    tb_dir = os.path.join(args.log_dir, "tensorboard")
    writer = SummaryWriter(tb_dir)

    logger.info("=" * 70)
    logger.info(f" 消融实验训练 | fusion_mode={args.fusion_mode}")
    logger.info("=" * 70)
    logger.info(f"  数据根目录  : {args.data_root}")
    logger.info(f"  融合模式    : {args.fusion_mode}")
    logger.info(f"  注意力头数  : {args.num_heads}")
    logger.info(f"  Batch Size  : {args.batch_size}")
    logger.info(f"  Epochs      : {args.epochs}")
    logger.info(f"  LR (bb/hd/fu): {args.backbone_lr}/{args.head_lr}/{args.fusion_lr}")
    logger.info(f"  设备        : {args.device}")

    # ── 文本提示 ──────────────────────────────────────────────────────
    text_prompts = load_text_prompts(args)
    if text_prompts:
        logger.info(f"  文本提示 ({len(text_prompts)} 个): {text_prompts[:3]} ...")
    else:
        logger.info("  文本提示: 无 (vision_only 模式)")

    # ── 数据集 ────────────────────────────────────────────────────────
    def make_dataset(split):
        return SegmentationDataset(
            data_list_file=os.path.join(args.data_root, f"{split}.txt"),
            image_dir=os.path.join(args.data_root, "images"),
            mask_dir=os.path.join(args.data_root, "masks"),
            mode="train" if split == "train" else "val",
            base_size=args.base_size,
            crop_size=args.crop_size,
        )

    train_ds, val_ds = make_dataset("train"), make_dataset("val")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    logger.info(f"  训练集: {len(train_ds)} | 验证集: {len(val_ds)}")

    # ── 构建模型 ──────────────────────────────────────────────────────
    model = build_ablation_model(
        fusion_mode=args.fusion_mode,
        num_classes=args.num_classes,
        arch=args.arch,
        pretrained=True,
        use_aux_head=args.use_aux_head,
        clip_model_name=args.clip_model_name,
        device=args.device,
        num_heads=args.num_heads,
        use_dual_fusion=args.use_dual_fusion,
    ).to(args.device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  总参数: {total_p:,} | 可训练: {train_p:,}")

    # ── 损失函数 ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    aux_criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ── 参数分组 ──────────────────────────────────────────────────────
    backbone_p, fusion_p, head_p = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name or "input_convs" in name:
            backbone_p.append(param)
        elif "fusion" in name:
            fusion_p.append(param)
        else:
            head_p.append(param)

    param_groups = [
        {"params": backbone_p, "lr": args.backbone_lr},
        {"params": head_p, "lr": args.head_lr},
    ]
    if fusion_p:
        param_groups.append({"params": fusion_p, "lr": args.fusion_lr})

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.epochs, args.warmup_epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── 训练循环 ──────────────────────────────────────────────────────
    best_miou = 0.0
    patience_cnt = 0
    history = {"train_losses": [], "val_losses": [], "val_mious": [], "val_accs": []}

    csv_path = os.path.join(args.log_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "fusion_mode", "lr", "train_loss", "val_loss",
             "val_mIoU", "val_mAcc", "val_acc", "best_mIoU"]
        )

    logger.info("\n" + "=" * 70)
    logger.info(f" 开始训练 | fusion_mode={args.fusion_mode} ")
    logger.info("=" * 70)

    start = datetime.now()

    try:
        for epoch in range(args.epochs):
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"\n{'─'*40} Epoch {epoch+1}/{args.epochs} "
                f"[{args.fusion_mode}] LR={cur_lr:.6f} {'─'*40}"
            )

            train_m = train_one_epoch(
                model, train_loader, optimizer, criterion, aux_criterion,
                args.device, epoch, args, text_prompts, logger, writer,
            )
            val_m = validate(
                model, val_loader, criterion, args.device, epoch,
                args, text_prompts, logger,
            )
            scheduler.step()

            history["train_losses"].append(train_m["loss"])
            history["val_losses"].append(val_m["loss"])
            history["val_mious"].append(val_m["mIoU"])
            history["val_accs"].append(val_m["overall_acc"])

            # TensorBoard
            for k, v in train_m.items():
                writer.add_scalar(f"train/{k}", v, epoch + 1)
            for k, v in val_m.items():
                if k == "class_ious":
                    continue
                writer.add_scalar(f"val/{k}", v, epoch + 1)
            for i, iou in enumerate(val_m["class_ious"]):
                if not np.isnan(iou):
                    writer.add_scalar(f"val_cls_iou/class_{i}", iou, epoch + 1)
            writer.add_scalar("best/val_mIoU", best_miou, epoch + 1)

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1, args.fusion_mode, f"{cur_lr:.8f}",
                    f"{train_m['loss']:.4f}", f"{val_m['loss']:.4f}",
                    f"{val_m['mIoU']:.4f}", f"{val_m['mAcc']:.4f}",
                    f"{val_m['overall_acc']:.4f}", f"{best_miou:.4f}",
                ])

            if val_m["mIoU"] > best_miou:
                best_miou = val_m["mIoU"]
                patience_cnt = 0
                ckpt = os.path.join(args.checkpoint_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "fusion_mode": args.fusion_mode,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_mIoU": best_miou,
                        "args": vars(args),
                        "text_prompts": text_prompts,
                    },
                    ckpt,
                )
                logger.info(f"  ★ 新最佳 mIoU={best_miou:.4f} → {ckpt}")
            else:
                patience_cnt += 1
                logger.info(f"  未提升 ({patience_cnt}/{args.patience})")

            writer.flush()

            if patience_cnt >= args.patience:
                logger.info(f"\n  早停: 第 {epoch+1} 轮")
                break

            if (epoch + 1) % 10 == 0:
                ckpt = os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth")
                torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict()}, ckpt)

    finally:
        writer.close()

    elapsed = datetime.now() - start
    logger.info("\n" + "=" * 70)
    logger.info(f" 训练完成! [{args.fusion_mode}]")
    logger.info(f"  耗时: {elapsed}")
    logger.info(f"  最佳 mIoU: {best_miou:.4f}")

    json.dump(history, open(os.path.join(args.log_dir, "history.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
