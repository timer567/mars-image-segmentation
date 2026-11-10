"""
DINOv2-B/14 + 因果解耦融合 + UPerNet 语义分割模型 (exp5)

架构:
  Image → DINOv2(冻结) → input_convs → unified_features[-1] ──┐
                                                                  ├──→ [因果解耦融合] → Z → 替换 features[-1]
  Text  → CLIP(冻结)   → text_prototypes ─────────────────────┘          ↓
                                                                [UPerNet Dec] → Pred

与 exp2 的关系:
  exp2: DIR 交叉注意力融合 (Q=visual, K/V=text → concat → proj)
  exp5: 因果解耦融合 (一致性探测 + 四分支解耦: C/Ux/Up/B → 带偏置抑制的融合)

关键区别:
  - exp2 是单分支融合 (视觉+文本 → 一个融合特征)
  - exp5 是四分支解耦 (分离共享语义/图像特有/提示特有/偏置, 定向抑制偏置)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_DINOV2_PATH = os.path.join(PROJECT_ROOT, "vendor", "dinov2")
LOCAL_CLIP_PATH = os.path.join(PROJECT_ROOT, "vendor", "CLIP")

# 将本地DINOv2仓库添加到Python路径
dinov2_path = LOCAL_DINOV2_PATH
if dinov2_path not in sys.path:
    sys.path.insert(0, dinov2_path)

# 将本地CLIP仓库添加到Python路径
clip_path = LOCAL_CLIP_PATH
if clip_path not in sys.path:
    sys.path.insert(0, clip_path)

import clip
from upernet_head import UPerHead
from causal_decouple_fusion import CausalDecoupleFusion, CausalDecoupleFusionLight


class DinoV2BackboneForSeg(nn.Module):
    """DINOv2 Backbone Adapter for Segmentation with local DINOv2 loading.
    
    从本地DINOv2仓库加载模型，并提取多层特征用于分割任务。
    """
    
    def __init__(self, arch='vit_base', patch_size=14, pretrained=True, 
                 out_indices=[2, 5, 8, 11], local_repo_path=LOCAL_DINOV2_PATH):
        super(DinoV2BackboneForSeg, self).__init__()
        
        self.arch = arch
        self.patch_size = patch_size
        self.out_indices = out_indices
        
        # 构建正确的模型名称
        if arch == 'vit_base':
            model_name = 'dinov2_vitb14'
        elif arch == 'vit_small':
            model_name = 'dinov2_vits14'
        elif arch == 'vit_large':
            model_name = 'dinov2_vitl14'
        elif arch == 'vit_giant2':
            model_name = 'dinov2_vitg14'
        else:
            raise ValueError(f"不支持的架构: {arch}")
        
        # 优先从本地路径加载DINOv2
        try:
            print(f"从本地路径加载DINOv2模型: {model_name} (path={local_repo_path})")
            self.model = torch.hub.load(local_repo_path, model_name, source='local')
            print("DINOv2模型加载成功！")
        except Exception as e:
            print(f"从本地路径加载失败: {e}")
            raise e
        
        # 获取嵌入维度和深度
        self.embed_dim = self.model.embed_dim
        self.num_blocks = self.model.n_blocks
        
        # 验证输出索引是否有效
        assert all(idx < self.num_blocks for idx in out_indices), \
            f"输出索引 {out_indices} 超出模型块数 {self.num_blocks}"
    
    def forward_features(self, x):
        """使用DINOv2官方方法提取中间层特征"""
        intermediate_features = self.model.get_intermediate_layers(
            x, 
            n=self.out_indices,
            reshape=True,
            return_class_token=False,
            norm=True
        )
        features = list(intermediate_features)
        return features

    def forward(self, x):
        return self.forward_features(x)


class AuxHead(nn.Module):
    """辅助分类头，用于中间监督"""
    def __init__(self, in_channels, num_classes, dropout_ratio=0.1):
        super(AuxHead, self).__init__()
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv(x)


class CLIPTextEncoder(nn.Module):
    """CLIP文本编码器封装"""
    def __init__(self, clip_model_name="ViT-B/16", device="cuda"):
        super(CLIPTextEncoder, self).__init__()
        self.clip_model_name = clip_model_name
        self.device = device
        
        # 加载CLIP模型
        try:
            self.clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
            print(f"CLIP模型 {clip_model_name} 加载成功！")
        except Exception as e:
            print(f"CLIP模型加载失败: {e}")
            raise e
        
        # 冻结CLIP模型参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.text_dim = self.clip_model.text_projection.shape[1]
    
    def encode_text_prompts(self, text_prompts):
        """编码文本提示"""
        with torch.no_grad():
            # Tokenize文本
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            # 编码文本
            text_features = self.clip_model.encode_text(text_tokens)
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def forward(self, text_prompts):
        return self.encode_text_prompts(text_prompts)


class DinoV2CausalDecoupleSeg(nn.Module):
    """DINOv2-B/14 + 因果解耦融合 + UPerNet 语义分割模型 (exp5)
    
    架构流程:
      Step 1: Image → DINOv2 → raw_features → input_convs → unified_features[4×(B,512,H,W)]
      Step 2: Text  → CLIP   → text_prototypes (K, C_text)
      Step 3: 因果解耦融合:
              - 一致性探测: 匹配概率 S + 提示传播 SP + 匹配熵 E + 偏置门 M_b
              - 四分支解耦: C(共享) + Ux(图像特有) + Up(提示特有) - B(偏置抑制)
              - Z = C + βx*Ux + βp*Up - βb*(M_b*B) → 替换 unified_features[-1]
      Step 4: UPerNet decode_head(unified_features) → main_output → upsample → Prediction
    """
    
    def __init__(self, num_classes=10, arch='vit_base', patch_size=14, 
                 pretrained=True, use_aux_head=True, clip_model_name="ViT-B/16",
                 device="cuda",
                 # 因果解耦融合参数
                 beta_x=0.5, beta_p=0.3, beta_b=0.3,
                 # 全局场景融合参数
                 use_global_scene=True,
                 use_dynamic_route=True,
                 num_groups=16,
                 # 双层融合参数
                 use_dual_fusion=True,
                 light_beta_x=0.5):
        super(DinoV2CausalDecoupleSeg, self).__init__()
        
        self.num_classes = num_classes
        self.use_aux_head = use_aux_head
        self.device = device
        self.use_global_scene = use_global_scene
        self.use_dual_fusion = use_dual_fusion
        
        out_indices = [2, 5, 8, 11]
        unified_dim = 512
        
        # ========== 编码器 ==========
        
        # 1. 视觉编码器: DINOv2
        self.backbone = DinoV2BackboneForSeg(
            arch=arch,
            patch_size=patch_size,
            pretrained=pretrained,
            out_indices=out_indices,
            local_repo_path=LOCAL_DINOV2_PATH
        )
        
        embed_dim = self.backbone.embed_dim
        
        # 特征投影层（统一到512维）
        self.input_convs = nn.ModuleList()
        for _ in range(len(out_indices)):
            self.input_convs.append(
                nn.Conv2d(embed_dim, unified_dim, kernel_size=1, bias=False)
            )

        # 2. 文本编码器: CLIP (冻结)
        self.clip_text_encoder = CLIPTextEncoder(clip_model_name=clip_model_name, device=device)
        clip_text_dim = self.clip_text_encoder.text_dim

        # ========== 因果解耦融合 ==========
        # 轻量版：用于 unified_features[-2]（中间层）
        if use_dual_fusion:
            self.causal_decouple_light = CausalDecoupleFusionLight(
                visual_dim=unified_dim,
                text_dim=clip_text_dim,
                out_dim=unified_dim,
                num_classes=num_classes,
                beta_x=light_beta_x,
                num_groups=num_groups,
            )

        # 完整版：用于 unified_features[-1]（最深层）
        self.causal_decouple = CausalDecoupleFusion(
            visual_dim=unified_dim,
            text_dim=clip_text_dim,
            out_dim=unified_dim,
            num_classes=num_classes,
            beta_x=beta_x,
            beta_p=beta_p,
            beta_b=beta_b,
            use_global_scene=use_global_scene,
            use_dynamic_route=use_dynamic_route,
            num_groups=num_groups,
        )

        # ========== 解码器 ==========
        
        # 3. UPerNet 解码头
        self.decode_head = UPerHead(
            in_channels=[unified_dim] * len(out_indices),
            channels=unified_dim,
            num_classes=num_classes,
            pool_scales=(1, 2, 3, 6),
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            align_corners=False,
            dropout_ratio=0.1
        )
        
        # 4. 辅助头
        if use_aux_head:
            self.aux_head = AuxHead(unified_dim, num_classes, dropout_ratio=0.1)
        else:
            self.aux_head = None

        # 冻结 backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 辅助输出缓存 (用于训练时提取辅助损失)
        self._aux_cache = None

        print("\n" + "=" * 60)
        print("DinoV2CausalDecoupleSeg 模型初始化完成 (exp5 + GlobalScene):")
        print(f"  骨干网络: DINOv2-{arch} (patch_size={patch_size})")
        print(f"  文本编码: CLIP-{clip_model_name} (dim={clip_text_dim})")
        print(f"  因果解耦融合: 四分支 (C/Ux/Up/B), beta_x={beta_x}, beta_p={beta_p}, beta_b={beta_b}")
        print(f"  全局场景: {'启用 (辅助上下文)' if use_global_scene else '禁用'}")
        print(f"  动态路由: {'启用' if use_dynamic_route else '禁用'}")
        print(f"  GroupNorm groups: {num_groups}")
        print(f"  双层融合: {'启用 (light@[-2] + full@[-1])' if use_dual_fusion else '禁用 (仅full@[-1])'}")
        if use_dual_fusion:
            print(f"  轻量版 beta_x: {light_beta_x}")
        print(f"  解码头: UPerNet (in_ch=[{unified_dim}]*{len(out_indices)})")
        print(f"  辅助头: {'启用' if use_aux_head else '禁用'}")
        print("=" * 60 + "\n")
    
    def forward(self, x, text_prompts=None, labels=None):
        """前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
            text_prompts: 文本提示列表
            labels: 训练时的标签（未直接使用，保留接口兼容性）
        
        Returns:
            训练模式+text+aux_head: (final_output, aux_output, fusion_aux)
            训练模式+text: (final_output, fusion_aux)
            推理模式+text: (final_output,)
            推理模式无text: final_output
        """
        original_size = x.shape[2:]
        
        # ========== Step 1: 视觉编码 ==========
        features = self.backbone(x)
        unified_features = []
        for i, feat in enumerate(features):
            unified_feat = self.input_convs[i](feat)
            unified_features.append(unified_feat)
        
        # ========== Step 2: 因果解耦融合（双层） ==========
        if text_prompts is not None:
            text_prototypes = self.clip_text_encoder(text_prompts)
            text_prototypes = text_prototypes.float()

            # 轻量版融合：unified_features[-2]（中间层 Block 8）
            if self.use_dual_fusion:
                mid_feat = unified_features[-2]
                fused_mid, aux_light = self.causal_decouple_light(mid_feat, text_prototypes)
                unified_features[-2] = fused_mid

            # 完整版融合：unified_features[-1]（最深层 Block 11）
            last_feat = unified_features[-1]
            fused_feat, aux = self.causal_decouple(last_feat, text_prototypes)
            unified_features[-1] = fused_feat

            # 合并 aux：完整版为默认 key，轻量版加 "light_" 前缀
            merged_aux = {}
            for k, v in aux.items():
                merged_aux[k] = v
            if self.use_dual_fusion:
                for k, v in aux_light.items():
                    merged_aux[f"light_{k}"] = v
            self._aux_cache = merged_aux

        # ========== Step 3: UPerNet 解码 ==========
        main_output = self.decode_head(unified_features)
        final_output = F.interpolate(
            main_output, 
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        )

        # ========== Step 4: 辅助输出 & 返回 ==========
        fusion_aux = self._aux_cache if text_prompts is not None else None

        if self.use_aux_head and self.training:
            aux_feat = unified_features[-2]
            aux_output = self.aux_head(aux_feat)
            aux_output = F.interpolate(
                aux_output,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
            return final_output, aux_output, fusion_aux

        if self.training and fusion_aux is not None:
            return final_output, fusion_aux

        return final_output

    def get_aux_cache(self):
        """获取最近一次前向传播的辅助输出缓存"""
        return self._aux_cache


def build_seg_model(num_classes=10, arch='vit_base', patch_size=14, 
                   pretrained=True, use_aux_head=True, clip_model_name="ViT-B/16",
                   device="cuda",
                   beta_x=0.5, beta_p=0.3, beta_b=0.3,
                   use_global_scene=True,
                   use_dynamic_route=True,
                   num_groups=16,
                   use_dual_fusion=True,
                   light_beta_x=0.5):
    """构建分割模型的辅助函数"""
    model = DinoV2CausalDecoupleSeg(
        num_classes=num_classes,
        arch=arch,
        patch_size=patch_size,
        pretrained=pretrained,
        use_aux_head=use_aux_head,
        clip_model_name=clip_model_name,
        device=device,
        beta_x=beta_x,
        beta_p=beta_p,
        beta_b=beta_b,
        use_global_scene=use_global_scene,
        use_dynamic_route=use_dynamic_route,
        num_groups=num_groups,
        use_dual_fusion=use_dual_fusion,
        light_beta_x=light_beta_x,
    )
    return model


if __name__ == "__main__":
    print("构建 DINOv2-B/14 + 因果解耦融合 + UPerNet 分割模型 (exp5)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_seg_model(
        num_classes=10, 
        arch='vit_base', 
        patch_size=14,
        clip_model_name="ViT-B/16",
        device=device,
        beta_x=0.5,
        beta_p=0.5,
        beta_b=0.1,
    )
    model.eval()
    model.to(device)
    
    input_tensor = torch.randn(1, 3, 518, 518).to(device)
    
    # 准备文本提示
    text_prompts = [f"a photo of class {i}" for i in range(10)]
    
    print("执行前向传播（带文本提示）...")
    with torch.no_grad():
        output = model(input_tensor, text_prompts=text_prompts)
        if isinstance(output, tuple):
            print(f"输出形状: {output[0].shape}")
        else:
            print(f"输出形状: {output.shape}")
    
    # 参数统计
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cdf_params = sum(p.numel() for n, p in model.named_parameters() if 'causal_decouple' in n and p.requires_grad)
    print(f"总参数: {total:,}, 可训练: {trainable:,}, 因果解耦: {cdf_params:,}")
    
    # 打印 beta 值
    print(f"beta_x: {model.causal_decouple.beta_x.item():.4f}")
    print(f"beta_p: {model.causal_decouple.beta_p.item():.4f}")
    print(f"beta_b: {model.causal_decouple.beta_b.item():.4f}")
    
    print("模型测试成功！")
