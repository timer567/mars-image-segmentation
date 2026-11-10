"""
消融实验模型 — DINOv2 + 交叉注意力融合 + UPerNet

架构与 model.py (exp5) 完全对齐，仅把 CausalDecoupleFusion 替换为
CrossAttentionFusion，通过 fusion_mode 参数切换三种消融变体:

  fusion_mode="cross_attn"  : Q=视觉 K/V=文本 的交叉注意力融合     (多模态基线)
  fusion_mode="vision_only" : 关闭文本分支，纯视觉 DINOv2 + UPerNet (视觉消融)
  fusion_mode="text_only"   : 输出完全由文本注意力驱动，无视觉残差   (文本消融)

与 model.py 的关系:
  - 复用: DinoV2BackboneForSeg, CLIPTextEncoder, AuxHead, UPerHead
  - 替换: CausalDecoupleFusion → CrossAttentionFusion / CrossAttentionFusionLight
  - 新增: build_ablation_model() 工厂函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_DINOV2_PATH = os.path.join(PROJECT_ROOT, "vendor", "dinov2")
LOCAL_CLIP_PATH = os.path.join(PROJECT_ROOT, "vendor", "CLIP")

if LOCAL_DINOV2_PATH not in sys.path:
    sys.path.insert(0, LOCAL_DINOV2_PATH)
if LOCAL_CLIP_PATH not in sys.path:
    sys.path.insert(0, LOCAL_CLIP_PATH)

import clip
from upernet_head import UPerHead
# 复用 exp5 中的 Backbone / TextEncoder / AuxHead
from model import DinoV2BackboneForSeg, CLIPTextEncoder, AuxHead
from cross_attention_fusion import CrossAttentionFusion, CrossAttentionFusionLight

FUSION_MODES = ("cross_attn", "vision_only", "text_only")


class DinoV2CrossAttnSeg(nn.Module):
    """
    消融实验主模型。

    Step 1: Image → DINOv2(冻结) → input_convs → unified_features [4×(B,512,H,W)]
    Step 2: Text  → CLIP(冻结)   → text_prototypes (K, C_text)
    Step 3: 交叉注意力融合 (对 unified_features[-2] 和 [-1]):
              - cross_attn  : 视觉 + 文本注意力残差
              - vision_only : 仅 conv 投影，跳过文本
              - text_only   : 纯文本注意力输出
    Step 4: UPerNet decode_head → main_output → upsample → Prediction
    """

    def __init__(
        self,
        num_classes: int = 10,
        arch: str = "vit_base",
        patch_size: int = 14,
        pretrained: bool = True,
        use_aux_head: bool = True,
        clip_model_name: str = "ViT-B/16",
        device: str = "cuda",
        fusion_mode: str = "cross_attn",   # 消融核心参数
        num_heads: int = 8,
        use_dual_fusion: bool = True,      # 是否对 [-2] 也做融合
        num_groups: int = 16,
    ):
        super().__init__()
        assert fusion_mode in FUSION_MODES, \
            f"fusion_mode 必须是 {FUSION_MODES}，收到: {fusion_mode}"

        self.num_classes = num_classes
        self.use_aux_head = use_aux_head
        self.device = device
        self.fusion_mode = fusion_mode
        self.use_dual_fusion = use_dual_fusion

        out_indices = [2, 5, 8, 11]
        unified_dim = 512

        # ──────────────── 编码器 ────────────────

        # 视觉骨干 (冻结)
        self.backbone = DinoV2BackboneForSeg(
            arch=arch,
            patch_size=patch_size,
            pretrained=pretrained,
            out_indices=out_indices,
            local_repo_path=LOCAL_DINOV2_PATH,
        )
        embed_dim = self.backbone.embed_dim

        # 特征投影层 (统一到 512 维)
        self.input_convs = nn.ModuleList(
            nn.Conv2d(embed_dim, unified_dim, kernel_size=1, bias=False)
            for _ in out_indices
        )

        # 文本编码器 (冻结); vision_only 时仍保留但不调用
        self.clip_text_encoder = CLIPTextEncoder(
            clip_model_name=clip_model_name, device=device
        )
        clip_text_dim = self.clip_text_encoder.text_dim

        # ──────────────── 融合模块 ────────────────

        # 轻量版: unified_features[-2] (Block 8)
        if use_dual_fusion:
            self.fusion_light = CrossAttentionFusionLight(
                visual_dim=unified_dim,
                text_dim=clip_text_dim,
                out_dim=unified_dim,
                num_groups=num_groups,
                mode=fusion_mode,
            )

        # 完整版: unified_features[-1] (Block 11)
        self.fusion = CrossAttentionFusion(
            visual_dim=unified_dim,
            text_dim=clip_text_dim,
            out_dim=unified_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            mode=fusion_mode,
        )

        # ──────────────── 解码器 ────────────────

        self.decode_head = UPerHead(
            in_channels=[unified_dim] * len(out_indices),
            channels=unified_dim,
            num_classes=num_classes,
            pool_scales=(1, 2, 3, 6),
            norm_cfg=dict(type="BN", requires_grad=True),
            act_cfg=dict(type="ReLU"),
            align_corners=False,
            dropout_ratio=0.1,
        )

        if use_aux_head:
            self.aux_head = AuxHead(unified_dim, num_classes, dropout_ratio=0.1)
        else:
            self.aux_head = None

        # 冻结骨干
        for param in self.backbone.parameters():
            param.requires_grad = False

        self._aux_cache = None

        self._print_summary(arch, clip_model_name, clip_text_dim, len(out_indices), unified_dim)

    # ──────────────────────────────────────────────────────────────────

    def _print_summary(self, arch, clip_name, clip_dim, n_scales, dim):
        print("\n" + "=" * 60)
        print("DinoV2CrossAttnSeg (消融实验模型):")
        print(f"  骨干网络  : DINOv2-{arch}")
        print(f"  文本编码  : CLIP-{clip_name} (dim={clip_dim})")
        print(f"  融合模式  : {self.fusion_mode}")
        print(f"  双层融合  : {'启用' if self.use_dual_fusion else '禁用'}")
        print(f"  解码头    : UPerNet ({n_scales} 尺度, dim={dim})")
        print(f"  辅助头    : {'启用' if self.aux_head else '禁用'}")
        print("=" * 60 + "\n")

    # ──────────────────────────────────────────────────────────────────

    def _need_text(self) -> bool:
        return self.fusion_mode != "vision_only"

    def forward(self, x: torch.Tensor, text_prompts=None, labels=None):
        """
        Args:
            x            : [B, 3, H, W]
            text_prompts : List[str] — vision_only 时可不传
            labels       : 保留接口兼容性，未使用

        Returns (训练):
            use_aux_head=True  : (final_output, aux_output, fusion_aux)
            use_aux_head=False : (final_output, fusion_aux)
        Returns (推理):
            final_output
        """
        original_size = x.shape[2:]

        # ── Step 1: 视觉编码 ──────────────────────────────────────────
        raw_features = self.backbone(x)
        unified_features = [
            conv(feat) for conv, feat in zip(self.input_convs, raw_features)
        ]

        # ── Step 2: 文本编码 ──────────────────────────────────────────
        text_prototypes = None
        if self._need_text() and text_prompts is not None:
            text_prototypes = self.clip_text_encoder(text_prompts).float()

        # ── Step 3: 交叉注意力融合 ───────────────────────────────────
        fusion_aux_list = {}

        if self.use_dual_fusion:
            mid_feat = unified_features[-2]
            fused_mid, aux_light = self.fusion_light(mid_feat, text_prototypes)
            unified_features[-2] = fused_mid
            for k, v in aux_light.items():
                fusion_aux_list[f"light_{k}"] = v

        last_feat = unified_features[-1]
        fused_last, aux_main = self.fusion(last_feat, text_prototypes)
        unified_features[-1] = fused_last
        fusion_aux_list.update(aux_main)

        self._aux_cache = fusion_aux_list

        # ── Step 4: UPerNet 解码 ──────────────────────────────────────
        main_output = self.decode_head(unified_features)
        final_output = F.interpolate(
            main_output, size=original_size, mode="bilinear", align_corners=False
        )

        # ── Step 5: 返回 ──────────────────────────────────────────────
        if self.use_aux_head and self.training:
            aux_feat = unified_features[-2]
            aux_output = F.interpolate(
                self.aux_head(aux_feat),
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            return final_output, aux_output, fusion_aux_list

        if self.training:
            return final_output, fusion_aux_list

        return final_output

    def get_aux_cache(self):
        return self._aux_cache


# ──────────────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────────────

def build_ablation_model(
    fusion_mode: str = "cross_attn",
    num_classes: int = 10,
    arch: str = "vit_base",
    patch_size: int = 14,
    pretrained: bool = True,
    use_aux_head: bool = True,
    clip_model_name: str = "ViT-B/16",
    device: str = "cuda",
    num_heads: int = 8,
    use_dual_fusion: bool = True,
    num_groups: int = 16,
) -> DinoV2CrossAttnSeg:
    """
    消融实验模型构建入口。

    Args:
        fusion_mode: "cross_attn" | "vision_only" | "text_only"

    Example:
        model = build_ablation_model(fusion_mode="vision_only")
    """
    return DinoV2CrossAttnSeg(
        num_classes=num_classes,
        arch=arch,
        patch_size=patch_size,
        pretrained=pretrained,
        use_aux_head=use_aux_head,
        clip_model_name=clip_model_name,
        device=device,
        fusion_mode=fusion_mode,
        num_heads=num_heads,
        use_dual_fusion=use_dual_fusion,
        num_groups=num_groups,
    )


# ──────────────────────────────────────────────────────────────────────
# 最小验证 (无真实数据集)
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_prompts = [f"a photo of mars terrain class {i}" for i in range(10)]

    for mode in FUSION_MODES:
        print(f"\n{'='*50}")
        print(f"测试 fusion_mode = {mode}")
        model = build_ablation_model(
            fusion_mode=mode,
            num_classes=10,
            device=device,
        ).to(device).eval()

        x = torch.randn(1, 3, 518, 518, device=device)
        with torch.no_grad():
            out = model(
                x,
                text_prompts=text_prompts if mode != "vision_only" else None,
            )
        shape = out[0].shape if isinstance(out, tuple) else out.shape
        print(f"输出形状: {shape}")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数: {total:,}  可训练: {trainable:,}")

    print("\n三种消融模式测试完成!")
