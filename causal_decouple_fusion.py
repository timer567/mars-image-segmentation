"""
因果解耦融合模块 (Causal Decouple Fusion) — V2 类别级匹配版

与V1全局文本版的关键区别:
  V1: CLIP输出10个prompt → GlobalSceneFusion聚合为1个全局向量(1, C) → 一致性探测(K=1) → 四分支解耦
  V2: CLIP输出10个prompt → 类别级匹配(K个类别) + 全局scene辅助 → 四分支解耦

改动重点:
  1. Step A 使用类别级 prompt 做匹配，不再退化成 K=1
  2. 全局文本只作为辅助上下文输入 shared/bias gate
  3. BN -> GN，提升小 batch 测试稳定性
  4. Up 增加空间门控，减少训练集记忆化
  5. 增加样本级动态融合系数

核心思路:
  Step A — 一致性探测 (类别级版):
    1. q = proj_q(X)  →  (B, HW, C_vis)
    2. k = proj_k(P)  →  (B, K, C_vis)
    3. S = softmax(q @ k^T / sqrt(C_vis))  →  (B, HW, K)  类别匹配分布
    4. SP_cls = S @ P  →  (B, HW, C_text)  类别prompt传播特征
    5. E = -Σ S*log(S)  →  (B, HW, 1)  真正的类别不确定性
    6. M_b = BiasGateNet([X, SP_proj, E, global_proj(opt)])  →  (B, 1, H, W)

  Step B — 四分支因果解耦:
    C  = 共享语义分支  (image ∩ prompt)  — 含全局scene辅助
    Ux = 图像特有分支 (image \ prompt)
    Up = 提示特有分支 (prompt \ image)  — 类别级→空间级，带门控
    B  = 偏置分支     (spurious / bias)

  融合输出:
    Z = C + (beta_x * alpha) * Ux + (beta_p * gamma) * Up - (beta_b * delta) * (M_b * B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 全局场景文本聚合模块
# 作用：从 K 个类别 prompt 中额外聚合出 1 个全局 scene 向量
# 注意：这个全局向量只做辅助上下文，不替代类别级 prompt 匹配
# ============================================================

class GlobalSceneFusion(nn.Module):
    def __init__(self, text_dim, num_classes=10):
        super().__init__()
        self.text_dim = text_dim
        self.num_classes = num_classes

        self.global_query = nn.Parameter(torch.randn(1, 1, text_dim) * 0.02)
        self.norm_text = nn.LayerNorm(text_dim)
        self.text_k_proj = nn.Linear(text_dim, text_dim)
        self.text_v_proj = nn.Linear(text_dim, text_dim)

    def aggregate(self, text_prototypes):
        """
        Args:
            text_prototypes: (K, C_text)
        Returns:
            global_feat: (1, C_text)
        """
        t = self.norm_text(text_prototypes.unsqueeze(0))  # (1, K, C_text)
        k = self.text_k_proj(t)                           # (1, K, C_text)
        v = self.text_v_proj(t)                           # (1, K, C_text)
        q = self.global_query                             # (1, 1, C_text)

        attn = (q @ k.transpose(-2, -1)) / (self.text_dim ** 0.5)  # (1,1,K)
        attn = attn.softmax(dim=-1)
        global_feat = (attn @ v).squeeze(1)  # (1, C_text)
        return global_feat


# ============================================================
# GN版本ConvMLP
# 原因：分割任务小batch时，GN通常比BN更稳，更利于测试泛化
# ============================================================

class ConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=16, act_layer=nn.GELU):
        super().__init__()

        g1 = min(num_groups, out_channels)
        while out_channels % g1 != 0:
            g1 -= 1

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(g1, out_channels),
            act_layer(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(g1, out_channels),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 偏置感知门网络
# 输入：[X, SP_cls_proj, entropy, global_map_proj(optional)]
# 输出：(B,1,H,W)
# ============================================================

class BiasGateNet(nn.Module):
    def __init__(self, in_channels, hidden_ratio=4):
        super().__init__()
        hidden = max(16, in_channels // hidden_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 修改版因果解耦融合模块
#
# 改动重点：
# 1. Step A 使用类别级 prompt 做匹配，不再退化成 K=1
# 2. 全局文本只作为辅助上下文输入 shared/bias gate
# 3. BN -> GN
# 4. Up 增加空间门控
# 5. 增加样本级动态融合系数（轻量版，不是复杂区域路由）
# ============================================================

class CausalDecoupleFusion(nn.Module):
    def __init__(
        self,
        visual_dim=512,
        text_dim=512,
        out_dim=512,
        num_classes=10,
        beta_x=0.5,
        beta_p=0.3,
        beta_b=0.3,
        use_global_scene=True,
        use_dynamic_route=True,
        num_groups=16,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.use_global_scene = use_global_scene
        self.use_dynamic_route = use_dynamic_route

        # ----------------------------------------------------
        # 全局scene文本（辅助）
        # ----------------------------------------------------
        if use_global_scene:
            self.global_scene_fusion = GlobalSceneFusion(
                text_dim=text_dim,
                num_classes=num_classes,
            )

        # ----------------------------------------------------
        # Step A: 一致性探测（类别级）
        # ----------------------------------------------------
        self.proj_q = nn.Conv2d(visual_dim, visual_dim, kernel_size=1, bias=False)
        self.proj_k = nn.Linear(text_dim, visual_dim, bias=False)

        # 类别prompt传播特征映射到视觉空间
        self.sp_proj = nn.Conv2d(text_dim, visual_dim, kernel_size=1, bias=False)

        # 全局scene文本映射到视觉空间
        if use_global_scene:
            self.global_proj = nn.Conv2d(text_dim, visual_dim, kernel_size=1, bias=False)

        # 偏置门
        gate_in_channels = visual_dim + visual_dim + 1
        if use_global_scene:
            gate_in_channels += visual_dim
        self.bias_gate = BiasGateNet(gate_in_channels)

        # ----------------------------------------------------
        # Step B: 四分支
        # ----------------------------------------------------

        # C: 共享语义
        # [x, sp_cls_proj, x*sp_cls_proj, global_map_proj(optional)]
        c_in_channels = visual_dim * 3
        if use_global_scene:
            c_in_channels += visual_dim
        self.shared_branch = ConvMLP(c_in_channels, out_dim, num_groups=num_groups)

        # C -> X空间，用于图像残差
        self.c_proj_for_rx = nn.Conv2d(out_dim, visual_dim, kernel_size=1, bias=False)

        # Ux: 图像特有
        self.ux_branch = ConvMLP(visual_dim, out_dim, num_groups=num_groups)

        # Up: 提示特有
        self.up_mlp = nn.Sequential(
            nn.Linear(text_dim, text_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(text_dim, text_dim, bias=False),
        )
        self.up_proj = nn.Conv2d(text_dim, out_dim, kernel_size=1, bias=False)

        # C -> text空间，用于提示残差
        self.c_proj_for_rp = nn.Conv2d(out_dim, text_dim, kernel_size=1, bias=False)

        # Up门控，减少全图统一文本补偿
        self.up_gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 4, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # B: 偏置分支
        # 输入 [R_x, sp_cls_proj, M_b*R_x, entropy]
        self.bias_branch = ConvMLP(visual_dim * 3 + 1, out_dim, num_groups=num_groups)

        # 输出投影
        g_out = min(num_groups, out_dim)
        while out_dim % g_out != 0:
            g_out -= 1
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(g_out, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
        )

        # ----------------------------------------------------
        # 固定可学习系数
        # ----------------------------------------------------
        self.beta_x = nn.Parameter(torch.tensor(beta_x))
        self.beta_p = nn.Parameter(torch.tensor(beta_p))
        self.beta_b = nn.Parameter(torch.tensor(beta_b))

        # ----------------------------------------------------
        # 轻量动态路由（样本级）
        # 作用：不同图片对 Ux / Up / B 的依赖不同
        # ----------------------------------------------------
        if use_dynamic_route:
            self.route = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_dim * 3, out_dim // 4, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim // 4, 3, kernel_size=1, bias=True),
            )

    def _prepare_prompt(self, p, B):
        """
        统一把 p 处理成 [B, K, C_text]

        支持：
        1. p: [K, C_text]
        2. p: [B, K, C_text]
        """
        if p.dim() == 2:
            # [K, C_text] -> [B, K, C_text]
            p_expand = p.unsqueeze(0).expand(B, -1, -1)
        elif p.dim() == 3:
            p_expand = p
        else:
            raise ValueError(f"Unsupported prompt shape: {p.shape}")
        return p_expand

    def _build_global_text(self, p_expand):
        """
        从类别prompt构建全局scene向量
        p_expand: [B, K, C_text] 或共享版
        简化处理：
        - 如果 batch 内每张图 prompt 一样，则取第一张
        - 若 batch prompt 不同，这里仍逐batch平均生成一个共享scene原型
        """
        if not self.use_global_scene:
            return None

        # 为了保持实现稳定，这里对 batch 内 prompt 先平均，再做 scene 聚合
        p_mean = p_expand.mean(dim=0)        # [K, C_text]
        p_global = self.global_scene_fusion.aggregate(p_mean)  # [1, C_text]
        return p_global

    def forward(self, x, p):
        """
        Args:
            x: [B, C_vis, H, W]
            p: [K, C_text] or [B, K, C_text]

        Returns:
            Z:   [B, out_dim, H, W]
            aux: dict
        """
        B, C_vis, H, W = x.shape
        device = x.device

        # ====================================================
        # 准备类别级prompt
        # ====================================================
        p_expand = self._prepare_prompt(p, B)   # [B, K, C_text]
        K = p_expand.shape[1]

        # ====================================================
        # 构建全局scene文本（辅助，不参与类别匹配）
        # ====================================================
        if self.use_global_scene:
            p_global = self._build_global_text(p_expand)  # [1, C_text]
            global_map = p_global.view(1, self.text_dim, 1, 1).expand(B, -1, H, W)  # [B,C_text,H,W]
            global_map_proj = self.global_proj(global_map)  # [B,C_vis,H,W]
        else:
            p_global = None
            global_map_proj = None

        # ====================================================
        # Step A: 类别级一致性探测
        # ====================================================

        # q: [B, C_vis, H, W] -> [B, HW, C_vis]
        q = self.proj_q(x).flatten(2).transpose(1, 2)  # [B, HW, C_vis]

        # k: [B, K, C_text] -> [B, K, C_vis]
        k = self.proj_k(p_expand)  # [B, K, C_vis]

        # 类别级匹配分数
        logits = torch.matmul(q, k.transpose(1, 2)) / (C_vis ** 0.5)  # [B, HW, K]
        S = F.softmax(logits, dim=-1)  # [B, HW, K]

        # 类别prompt传播特征
        SP_cls = torch.matmul(S, p_expand)  # [B, HW, C_text]
        SP_cls = SP_cls.transpose(1, 2).reshape(B, self.text_dim, H, W)  # [B,C_text,H,W]
        SP_cls_proj = self.sp_proj(SP_cls)  # [B,C_vis,H,W]

        # 熵 E：恢复类别不确定性
        entropy = -(S * torch.log(S + 1e-8)).sum(dim=-1, keepdim=True)  # [B,HW,1]
        entropy = entropy.transpose(1, 2).reshape(B, 1, H, W)  # [B,1,H,W]

        # 偏置门 M_b
        gate_inputs = [x, SP_cls_proj, entropy]
        if self.use_global_scene:
            gate_inputs.append(global_map_proj)
        gate_input = torch.cat(gate_inputs, dim=1)
        M_b = self.bias_gate(gate_input)  # [B,1,H,W]

        # ====================================================
        # Step B-1: 共享语义 C
        # ====================================================
        elem_prod = x * SP_cls_proj
        c_inputs = [x, SP_cls_proj, elem_prod]
        if self.use_global_scene:
            c_inputs.append(global_map_proj)
        C = self.shared_branch(torch.cat(c_inputs, dim=1))  # [B,out_dim,H,W]

        # ====================================================
        # Step B-2: 图像特有 Ux
        # ====================================================
        C_proj_x = self.c_proj_for_rx(C)      # [B,C_vis,H,W]
        R_x = x - C_proj_x                    # [B,C_vis,H,W]
        Ux = self.ux_branch(R_x)              # [B,out_dim,H,W]

        # ====================================================
        # Step B-3: 提示特有 Up（类别级 -> 空间级）
        # ====================================================
        C_proj_text = self.c_proj_for_rp(C)    # [B,C_text,H,W]
        C_proj_text_flat = C_proj_text.flatten(2).transpose(1, 2)  # [B,HW,C_text]

        # 用类别匹配 S 把共享语义聚合回类别维
        class_mass = S.sum(dim=1, keepdim=False).unsqueeze(-1) + 1e-6   # [B,K,1]
        c_pool = torch.matmul(S.transpose(1, 2), C_proj_text_flat) / class_mass  # [B,K,C_text]

        # 提示残差
        R_p = p_expand - c_pool  # [B,K,C_text]

        # 类别级 Up
        Up_cls = self.up_mlp(R_p)  # [B,K,C_text]

        # 再投回空间
        Up_spatial = torch.matmul(S, Up_cls)  # [B,HW,C_text]
        Up_spatial = Up_spatial.transpose(1, 2).reshape(B, self.text_dim, H, W)
        Up = self.up_proj(Up_spatial)  # [B,out_dim,H,W]

        # Up门控：减少"全图统一文本补偿"
        up_gate = self.up_gate(torch.cat([C, Ux], dim=1))  # [B,1,H,W]
        Up = up_gate * Up

        # ====================================================
        # Step B-4: 偏置分支 B
        # ====================================================
        B_input = torch.cat([R_x, SP_cls_proj, M_b * R_x, entropy], dim=1)
        B_feat = self.bias_branch(B_input)  # [B,out_dim,H,W]

        # ====================================================
        # 融合输出
        # ====================================================
        beta_x = torch.sigmoid(self.beta_x)
        beta_p = torch.sigmoid(self.beta_p)
        beta_b = torch.sigmoid(self.beta_b)

        if self.use_dynamic_route:
            route_logits = self.route(torch.cat([C, Ux, Up], dim=1))  # [B,3,1,1]
            route = F.softmax(route_logits, dim=1)
            alpha = route[:, 0:1]  # Ux权重
            gamma = route[:, 1:2]  # Up权重
            delta = route[:, 2:3]  # B抑制权重

            Z = C + (beta_x * alpha) * Ux + (beta_p * gamma) * Up - (beta_b * delta) * (M_b * B_feat)
        else:
            alpha = gamma = delta = None
            Z = C + beta_x * Ux + beta_p * Up - beta_b * (M_b * B_feat)

        Z = self.out_proj(Z)

        # ====================================================
        # aux
        # ====================================================
        aux = {
            "S": S,                         # [B,HW,K]
            "SP_cls": SP_cls,               # [B,C_text,H,W]
            "SP_cls_proj": SP_cls_proj,     # [B,C_vis,H,W]
            "entropy": entropy,             # [B,1,H,W]
            "M_b": M_b,                     # [B,1,H,W]
            "C": C,                         # [B,out_dim,H,W]
            "Ux": Ux,                       # [B,out_dim,H,W]
            "Up": Up,                       # [B,out_dim,H,W]
            "B": B_feat,                    # [B,out_dim,H,W]
            "R_x": R_x,                     # [B,C_vis,H,W]
            "R_p": R_p,                     # [B,K,C_text]
            "up_gate": up_gate,             # [B,1,H,W]
            "beta_x": beta_x,
            "beta_p": beta_p,
            "beta_b": beta_b,
        }

        if self.use_global_scene:
            aux["p_global"] = p_global              # [1,C_text]
            aux["global_map_proj"] = global_map_proj

        if self.use_dynamic_route:
            aux["route_alpha"] = alpha
            aux["route_gamma"] = gamma
            aux["route_delta"] = delta

        return Z, aux


# ============================================================
# 轻量版因果解耦融合 — 用于中间层特征
#
# 与完整版的区别:
#   - 保留 Step A（类别级匹配）获取 S、SP_cls
#   - 简化 Step B: 仅 C（共享语义）+ Ux（图像特有），无 Up/B
#   - 无 GlobalScene、无 bias_gate、无动态路由、无 up_gate
#   - 参数量约为完整版的 1/3
# ============================================================

class CausalDecoupleFusionLight(nn.Module):
    def __init__(
        self,
        visual_dim=512,
        text_dim=512,
        out_dim=512,
        num_classes=10,
        beta_x=0.5,
        num_groups=16,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_classes = num_classes

        # ----------------------------------------------------
        # Step A: 类别级匹配
        # ----------------------------------------------------
        self.proj_q = nn.Conv2d(visual_dim, visual_dim, kernel_size=1, bias=False)
        self.proj_k = nn.Linear(text_dim, visual_dim, bias=False)
        self.sp_proj = nn.Conv2d(text_dim, visual_dim, kernel_size=1, bias=False)

        # ----------------------------------------------------
        # Step B: 简化两分支
        # ----------------------------------------------------

        # C: 共享语义  [x, sp_cls_proj, x*sp_cls_proj]
        self.shared_branch = ConvMLP(visual_dim * 3, out_dim, num_groups=num_groups)
        self.c_proj_for_rx = nn.Conv2d(out_dim, visual_dim, kernel_size=1, bias=False)

        # Ux: 图像特有
        self.ux_branch = ConvMLP(visual_dim, out_dim, num_groups=num_groups)

        # 输出投影
        g_out = min(num_groups, out_dim)
        while out_dim % g_out != 0:
            g_out -= 1
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(g_out, out_dim),
            nn.GELU(),
        )

        # 可学习系数
        self.beta_x = nn.Parameter(torch.tensor(beta_x))

    def _prepare_prompt(self, p, B):
        if p.dim() == 2:
            p_expand = p.unsqueeze(0).expand(B, -1, -1)
        elif p.dim() == 3:
            p_expand = p
        else:
            raise ValueError(f"Unsupported prompt shape: {p.shape}")
        return p_expand

    def forward(self, x, p):
        """
        Args:
            x: [B, C_vis, H, W]
            p: [K, C_text] or [B, K, C_text]
        Returns:
            Z:   [B, out_dim, H, W]
            aux: dict
        """
        B, C_vis, H, W = x.shape

        # 准备类别级 prompt
        p_expand = self._prepare_prompt(p, B)
        K = p_expand.shape[1]

        # ==================================================
        # Step A: 类别级匹配
        # ==================================================
        q = self.proj_q(x).flatten(2).transpose(1, 2)  # [B, HW, C_vis]
        k = self.proj_k(p_expand)                        # [B, K, C_vis]

        logits = torch.matmul(q, k.transpose(1, 2)) / (C_vis ** 0.5)
        S = F.softmax(logits, dim=-1)  # [B, HW, K]

        # 类别 prompt 传播特征
        SP_cls = torch.matmul(S, p_expand)  # [B, HW, C_text]
        SP_cls = SP_cls.transpose(1, 2).reshape(B, self.text_dim, H, W)
        SP_cls_proj = self.sp_proj(SP_cls)  # [B, C_vis, H, W]

        # ==================================================
        # Step B: 简化两分支
        # ==================================================

        # C: 共享语义
        elem_prod = x * SP_cls_proj
        C = self.shared_branch(torch.cat([x, SP_cls_proj, elem_prod], dim=1))

        # Ux: 图像特有
        C_proj_x = self.c_proj_for_rx(C)
        R_x = x - C_proj_x
        Ux = self.ux_branch(R_x)

        # 融合
        beta_x = torch.sigmoid(self.beta_x)
        Z = C + beta_x * Ux

        Z = self.out_proj(Z)

        aux = {
            "S": S,
            "SP_cls_proj": SP_cls_proj,
            "C": C,
            "Ux": Ux,
            "R_x": R_x,
            "beta_x": beta_x,
        }
        return Z, aux


# ============================================================
# 最小测试
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CausalDecoupleFusion(
        visual_dim=512,
        text_dim=512,
        out_dim=512,
        num_classes=10,
        beta_x=0.5,
        beta_p=0.3,
        beta_b=0.3,
        use_global_scene=True,
        use_dynamic_route=True,
        num_groups=16,
    ).to(device)

    x = torch.randn(2, 512, 37, 37, device=device)     # [B,C,H,W]
    p = torch.randn(10, 512, device=device)            # [K,C_text]

    z, aux = model(x, p)

    print("z:", z.shape)
    for k, v in aux.items():
        if torch.is_tensor(v):
            print(k, v.shape)
        else:
            print(k, v)

    loss = z.mean()
    loss.backward()
    print("backward ok")
