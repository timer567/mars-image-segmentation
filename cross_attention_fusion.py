"""
交叉注意力融合模块 (Cross-Attention Fusion) — 消融实验专用

支持三种融合模式:
  cross_attn   : Q=视觉, K/V=文本 → 交叉注意力 → 视觉+文本融合特征
  vision_only  : 纯视觉，文本分支完全关闭，仅做 identity 或 1×1 conv
  text_only    : 视觉仅提供 query，输出完全由文本注意力结果生成

注意: 三种模式共享同一个类，通过 mode 参数切换，方便消融对比。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    单层多头交叉注意力融合块。

    cross_attn 模式:
        Q = proj_q(x_flat)          [B, HW, d]
        K = proj_k(p_expand)        [B, K,  d]
        V = proj_v(p_expand)        [B, K,  d]
        attended = softmax(Q K^T / sqrt(d)) @ V   [B, HW, d]
        Z = out_proj(x + attended)  [B, C, H, W]   # 残差

    vision_only 模式:
        Z = out_proj(x)             # 不涉及 p

    text_only 模式:
        Q = proj_q(x_flat)
        K / V = proj_k/v(p_expand)
        attended = softmax(Q K^T / sqrt(d)) @ V
        Z = out_proj(attended)      # 无视觉残差
    """

    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 512,
        out_dim: int = 512,
        num_heads: int = 8,
        num_groups: int = 16,
        mode: str = "cross_attn",   # "cross_attn" | "vision_only" | "text_only"
    ):
        super().__init__()
        assert mode in ("cross_attn", "vision_only", "text_only"), \
            f"mode 必须是 cross_attn / vision_only / text_only，收到: {mode}"
        self.mode = mode
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # attention head dim
        assert visual_dim % num_heads == 0, "visual_dim 必须整除 num_heads"
        self.head_dim = visual_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ── Q 总是从视觉空间来（vision_only 不用 Q，占位不影响参数量）
        self.proj_q = nn.Conv2d(visual_dim, visual_dim, kernel_size=1, bias=False)

        # ── K / V 仅 cross_attn / text_only 使用
        if mode != "vision_only":
            self.proj_k = nn.Linear(text_dim, visual_dim, bias=False)
            self.proj_v = nn.Linear(text_dim, visual_dim, bias=False)

        # ── 输出投影
        # 选 GroupNorm；小 batch 时比 BN 更稳定
        g = min(num_groups, out_dim)
        while out_dim % g != 0:
            g -= 1

        if mode == "vision_only":
            # 纯视觉：仅做通道对齐 + 归一化
            self.out_proj = nn.Sequential(
                nn.Conv2d(visual_dim, out_dim, kernel_size=1, bias=False),
                nn.GroupNorm(g, out_dim),
                nn.GELU(),
            )
        else:
            # cross_attn / text_only：将注意力结果映射到 out_dim
            self.out_proj = nn.Sequential(
                nn.Conv2d(visual_dim, out_dim, kernel_size=1, bias=False),
                nn.GroupNorm(g, out_dim),
                nn.GELU(),
            )

    # ------------------------------------------------------------------
    # 辅助：处理 prompt 形状  [K, C] 或 [B, K, C] → [B, K, C]
    # ------------------------------------------------------------------
    def _expand_prompt(self, p: torch.Tensor, B: int) -> torch.Tensor:
        if p.dim() == 2:
            return p.unsqueeze(0).expand(B, -1, -1)
        elif p.dim() == 3:
            return p
        raise ValueError(f"prompt 维度应为 2 或 3，实际: {p.dim()}")

    # ------------------------------------------------------------------
    # 多头注意力（不依赖 F.scaled_dot_product_attention，兼容 1.x）
    # ------------------------------------------------------------------
    def _multihead_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        q: [B, N_q, C]
        k: [B, N_k, C]
        v: [B, N_k, C]
        → [B, N_q, C]
        """
        B, N_q, C = q.shape
        N_k = k.shape[1]
        h = self.num_heads
        d = self.head_dim

        # 拆头
        q = q.view(B, N_q, h, d).transpose(1, 2)    # [B, h, N_q, d]
        k = k.view(B, N_k, h, d).transpose(1, 2)    # [B, h, N_k, d]
        v = v.view(B, N_k, h, d).transpose(1, 2)    # [B, h, N_k, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, N_q, N_k]
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)  # [B, N_q, C]
        return out, attn

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, p: torch.Tensor = None
    ):
        """
        Args:
            x : [B, C_vis, H, W]
            p : [K, C_text] 或 [B, K, C_text]；vision_only 时可传 None

        Returns:
            Z   : [B, out_dim, H, W]
            aux : dict (调试用)
        """
        B, C_vis, H, W = x.shape

        # ── vision_only ──────────────────────────────────────────────
        if self.mode == "vision_only":
            Z = self.out_proj(x)
            return Z, {"mode": "vision_only"}

        # ── cross_attn / text_only ───────────────────────────────────
        assert p is not None, "cross_attn / text_only 模式需要提供 prompt p"
        p_expand = self._expand_prompt(p, B)     # [B, K, C_text]

        # Q 来自视觉
        q_feat = self.proj_q(x)                  # [B, C_vis, H, W]
        q = q_feat.flatten(2).transpose(1, 2)    # [B, HW, C_vis]

        # K / V 来自文本
        k = self.proj_k(p_expand)                # [B, K, C_vis]
        v = self.proj_v(p_expand)                # [B, K, C_vis]

        attended, attn_weight = self._multihead_attn(q, k, v)  # [B, HW, C_vis]

        # 还原空间维
        attended_2d = attended.transpose(1, 2).reshape(B, C_vis, H, W)

        if self.mode == "cross_attn":
            # 视觉残差 + 注意力
            fused = x + attended_2d
        else:
            # text_only: 丢弃视觉残差，纯文本注意力驱动
            fused = attended_2d

        Z = self.out_proj(fused)

        aux = {
            "mode": self.mode,
            "attn_weight": attn_weight,      # [B, num_heads, HW, K]
            "attended": attended_2d,         # [B, C_vis, H, W]
        }
        return Z, aux


# ─────────────────────────────────────────────────────────────────────
# 轻量版 (用于 unified_features[-2] 中间层)
# 与完整版的区别: 只做单头注意力, 无 GELU 输出, 参数更少
# ─────────────────────────────────────────────────────────────────────

class CrossAttentionFusionLight(nn.Module):
    """
    轻量交叉注意力融合 — 用于中间层 (unified_features[-2])

    vision_only: identity conv
    cross_attn / text_only: 单头注意力
    """

    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 512,
        out_dim: int = 512,
        num_groups: int = 16,
        mode: str = "cross_attn",
    ):
        super().__init__()
        assert mode in ("cross_attn", "vision_only", "text_only")
        self.mode = mode
        self.visual_dim = visual_dim
        self.scale = visual_dim ** -0.5

        g = min(num_groups, out_dim)
        while out_dim % g != 0:
            g -= 1

        if mode != "vision_only":
            self.proj_q = nn.Conv2d(visual_dim, visual_dim, kernel_size=1, bias=False)
            self.proj_k = nn.Linear(text_dim, visual_dim, bias=False)
            self.proj_v = nn.Linear(text_dim, visual_dim, bias=False)

        self.out_proj = nn.Sequential(
            nn.Conv2d(visual_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(g, out_dim),
        )

    def _expand_prompt(self, p, B):
        if p.dim() == 2:
            return p.unsqueeze(0).expand(B, -1, -1)
        return p

    def forward(self, x: torch.Tensor, p: torch.Tensor = None):
        B, C, H, W = x.shape

        if self.mode == "vision_only":
            return self.out_proj(x), {"mode": "vision_only"}

        p_expand = self._expand_prompt(p, B)
        q = self.proj_q(x).flatten(2).transpose(1, 2)   # [B, HW, C]
        k = self.proj_k(p_expand)                        # [B, K, C]
        v = self.proj_v(p_expand)                        # [B, K, C]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attended = (attn @ v).transpose(1, 2).reshape(B, C, H, W)

        fused = (x + attended) if self.mode == "cross_attn" else attended
        Z = self.out_proj(fused)
        return Z, {"mode": self.mode, "attn": attn}


# ─────────────────────────────────────────────────────────────────────
# 最小单元测试
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 512, 37, 37, device=device)
    p = torch.randn(10, 512, device=device)

    for mode in ("cross_attn", "vision_only", "text_only"):
        m = CrossAttentionFusion(mode=mode).to(device)
        z, aux = m(x, p if mode != "vision_only" else None)
        print(f"[{mode}] Z: {z.shape}, aux keys: {list(aux.keys())}")
        z.mean().backward()
        m.zero_grad()

    print("CrossAttentionFusion 三种模式测试通过")
