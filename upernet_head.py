import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_layer(norm_cfg, num_features):
    """
    只实现常用 norm，满足当前项目需求。
    支持:
        - dict(type='BN', requires_grad=True)
        - dict(type='SyncBN', requires_grad=True)  -> 退化为 BatchNorm2d
        - dict(type='GN', num_groups=32, requires_grad=True)
        - None
    """
    if norm_cfg is None:
        return nn.Identity()

    norm_type = norm_cfg.get('type', 'BN')
    requires_grad = norm_cfg.get('requires_grad', True)

    if norm_type in ['BN', 'SyncBN']:
        layer = nn.BatchNorm2d(num_features)
    elif norm_type == 'GN':
        num_groups = norm_cfg.get('num_groups', 32)
        layer = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    else:
        raise ValueError(f"不支持的 norm 类型: {norm_type}")

    for p in layer.parameters():
        p.requires_grad = requires_grad
    return layer


def build_act_layer(act_cfg):
    """
    支持:
        - dict(type='ReLU')
        - dict(type='GELU')
        - None
    """
    if act_cfg is None:
        return nn.Identity()

    act_type = act_cfg.get('type', 'ReLU')
    inplace = act_cfg.get('inplace', True)

    if act_type == 'ReLU':
        return nn.ReLU(inplace=inplace)
    elif act_type == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f"不支持的激活函数类型: {act_type}")


class ConvModule(nn.Module):
    """
    简化版 ConvModule
    顺序: Conv -> Norm -> Act
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU')
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.norm = build_norm_layer(norm_cfg, out_channels)
        self.act = build_act_layer(act_cfg)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PPM(nn.Module):
    """
    Pyramid Pooling Module
    """
    def __init__(
        self,
        pool_scales,
        in_channels,
        channels,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        align_corners=False
    ):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners

        self.ppm_modules = nn.ModuleList()
        for scale in pool_scales:
            self.ppm_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvModule(
                        in_channels=in_channels,
                        out_channels=channels,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                    )
                )
            )

    def forward(self, x):
        ppm_outs = [x]
        h, w = x.shape[2:]

        for ppm in self.ppm_modules:
            ppm_out = ppm(x)
            ppm_out = F.interpolate(
                ppm_out,
                size=(h, w),
                mode='bilinear',
                align_corners=self.align_corners
            )
            ppm_outs.append(ppm_out)

        return ppm_outs


class UPerHead(nn.Module):
    """
    纯 PyTorch 版 UPerNet Head

    参数说明和你前面的调用保持一致:
        in_channels: list，如 [512, 512, 512, 512]
        channels: 中间统一通道数，如 512
        num_classes: 类别数
        pool_scales: PPM 池化尺度
        norm_cfg: 归一化配置
        act_cfg: 激活函数配置
        align_corners: 上采样对齐方式
        dropout_ratio: dropout 比例
    """
    def __init__(
        self,
        in_channels,
        channels,
        num_classes,
        pool_scales=(1, 2, 3, 6),
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        dropout_ratio=0.1
    ):
        super().__init__()

        assert isinstance(in_channels, (list, tuple)), "in_channels 必须是 list 或 tuple"
        assert len(in_channels) >= 2, "UPerHead 至少需要两个层级特征"

        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.dropout_ratio = dropout_ratio

        # PSP/PPM 部分，作用在最后一层特征上
        self.ppm = PPM(
            pool_scales=pool_scales,
            in_channels=in_channels[-1],
            channels=channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners
        )

        self.bottleneck = ConvModule(
            in_channels=in_channels[-1] + len(pool_scales) * channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        # FPN lateral convs（最后一层不进 lateral，因为最后一层走 PPM）
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels[:-1]:
            self.lateral_convs.append(
                ConvModule(
                    in_channels=in_ch,
                    out_channels=channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
            self.fpn_convs.append(
                ConvModule(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        # FPN 输出融合
        self.fpn_bottleneck = ConvModule(
            in_channels=len(in_channels) * channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = nn.Identity()

        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def psp_forward(self, x):
        """
        对最后一层特征做 PPM
        """
        ppm_outs = self.ppm(x)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        output = self.bottleneck(ppm_outs)
        return output

    def forward(self, inputs):
        """
        inputs: list[Tensor]
            例如:
            [
                (B, 512, H1, W1),
                (B, 512, H2, W2),
                (B, 512, H3, W3),
                (B, 512, H4, W4)
            ]

        返回:
            (B, num_classes, H, W)
        """
        assert isinstance(inputs, (list, tuple)), "inputs 必须是 list 或 tuple"
        assert len(inputs) == len(self.in_channels), \
            f"输入特征层数不匹配: 期望 {len(self.in_channels)}, 实际 {len(inputs)}"

        # 1. lateral features（前 n-1 层）
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 2. 最后一层做 PPM
        laterals.append(self.psp_forward(inputs[-1]))

        # 3. 自顶向下融合
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners
            )

        # 4. 每层再过一个 3x3 conv（最后一层直接保留）
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        fpn_outs.append(laterals[-1])

        # 5. 全部上采样到最高分辨率再拼接
        out_size = fpn_outs[0].shape[2:]
        for i in range(1, used_backbone_levels):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=out_size,
                mode='bilinear',
                align_corners=self.align_corners
            )

        fusion_out = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fusion_out)

        output = self.dropout(output)
        output = self.cls_seg(output)

        return output


if __name__ == "__main__":
    # 简单测试
    feats = [
        torch.randn(2, 512, 37, 37),
        torch.randn(2, 512, 37, 37),
        torch.randn(2, 512, 37, 37),
        torch.randn(2, 512, 37, 37),
    ]

    head = UPerHead(
        in_channels=[512, 512, 512, 512],
        channels=512,
        num_classes=10,
        pool_scales=(1, 2, 3, 6),
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        dropout_ratio=0.1
    )

    out = head(feats)
    print("输出形状:", out.shape)  # 预期: [2, 10, 37, 37]