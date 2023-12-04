# -*- coding:utf-8 -*-
"""
Software:PyCharm
File: sedla34.py
Institution: --- CSU&BUCEA ---, China
E-mail: obitowen@csu.edu.cn
Author：Yanjie Wen
Date：2023年10月21日
My zoom: https://github.com/YanJieWen
"""
import torch.nn as nn
import math
import torch
from torchvision.ops.misc import FrozenBatchNorm2d

from backbone.feature_pyramid_network import LastLevelMaxPool
from backbone.RFPN import BacbonewithRFPN
# DLA-SE34
# 树结构
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1,norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x, residual=None,rpn_feat=0):
        if residual is None:
            residual = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out += rpn_feat
        out = self.relu(out)

        return out


class Bottleneckx(nn.Module):
    expansion = 2
    cardinality = 32  # 基数

    def __init__(self, fan_in, fan_out, stride=1, dilation=1) -> None:
        super(Bottleneckx, self).__init__()
        cardinality = Bottleneckx.cardinality
        bottle_planes = fan_out * cardinality // 32
        self.conv1 = nn.Conv2d(fan_in, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)  # 分组卷积
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, fan_out,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(fan_out)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class SE_layer(nn.Module):
    def __init__(self, fan_in, reduce=16) -> None:
        super(SE_layer, self).__init__()
        self.se = nn.AdaptiveAvgPool2d((1, 1))
        self.ex = nn.Sequential(nn.Linear(fan_in, fan_in // reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(fan_in // reduce, fan_in),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        _se = self.se(x).view(b, c)
        _ex = self.ex(_se)[:, :, None, None]  # b,c,1,1
        out = x * _ex
        # keep_idx = _ex.argsort(dim=-1)[:,-int(self.ratio*c):]#保留前50%大的特征图
        # out = torch.gather(x,1,keep_idx[:,:,None,None].repeat(1,1,h,w))
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual,norm_layer=None):
        super(Root, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.SE_layer = SE_layer(in_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        # 添加一个SE筛选块
        cat_x = self.SE_layer(torch.cat(x, 1))

        x = self.conv(cat_x)  # 语义聚合HDA & 空间聚合
        x = self.bn(x)
        if self.residual:
            x += children[0]  # 残差连接
        x = self.relu(x)
        return x


class Tree(nn.Module):
    def __init__(self, levels, block, fan_in, fan_out,stride=1, level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False,norm_layer=None, fpn_outchannel=256) -> None:
        super(Tree, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if root_dim == 0:
            root_dim = 2 * fan_out
        if level_root:
            root_dim += fan_in
        if levels == 1:
            self.tree1 = block(fan_in, fan_out, stride=stride, dilation=dilation)
            self.tree2 = block(fan_out, fan_out, stride=1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, fan_in, fan_out,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, fan_out, fan_out,
                              root_dim=root_dim + fan_out,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, fan_out, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.tran_rpn = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
            self.tran_rpn = nn.Conv2d(fpn_outchannel, fan_out, kernel_size=1, stride=1, padding=0, dilation=1,
                                  bias=False)
        if fan_in != fan_out:
            self.project = nn.Sequential(
                nn.Conv2d(fan_in, fan_out,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(fan_out)
            )

    def forward(self, x, residual=None, childern=None, rpn_feat=0):
        childern = [] if childern is None else childern
        bottom = self.downsample(x) if self.downsample else x  # 尺寸发生变化
        residual = self.project(bottom) if self.project else bottom  # 通道数发生变化
        rpn_feat = self.tran_rpn(rpn_feat) if rpn_feat!=0 else 0
        if self.level_root:
            childern.append(bottom)
        x1 = self.tree1(x, residual=residual,rpn_feat=rpn_feat)  # 在该第一个block后添加一个rpn_feat处理
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *childern)
        else:
            childern.append(x1)
            x = self.tree2(x1, childern=childern)
        return x


# 深层聚合网络（dla）
class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False, include_top=True,norm_layer=None) -> None:
        super(DLA, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                                        norm_layer(channels[0]), nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)  # /2->(112,112)

        self.level2 = Tree(levels[2], block, channels[1], channels[2], stride=2, level_root=False,
                           root_residual=residual_root,norm_layer=norm_layer)  # /2->(56,56)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], stride=2, level_root=True,
                           root_residual=residual_root,norm_layer=norm_layer)  # /2->(28,28)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], stride=2,
                           level_root=True, root_residual=residual_root,norm_layer=norm_layer)  # /2->(14,14)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], stride=2,
                           level_root=True, root_residual=residual_root,norm_layer=norm_layer)  # /2->(7,7)
        self.include_top = include_top
        if self.include_top:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, fan_in, fan_out, blocks, stride=1, dilation=1):
        modules = []
        for i in range(blocks):
            modules.extend([nn.Conv2d(fan_in, fan_out, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation,
                                      dilation=dilation, bias=False),
                            self._norm_layer(fan_out),
                            nn.ReLU(inplace=True)])
        return nn.Sequential(*modules)

    def forward(self, x, rpn_feats=None):
        x = self.base_layer(x)
        for i in range(6):
            if rpn_feats is not None and i in [2,3,4,5]:  # 这个地方传入的列表导致模型不够灵活[2,3,4层]
                x = getattr(self, f'level{int(i)}')(x, rpn_feat=rpn_feats[i - 2])
            else:
                x = getattr(self, f'level{int(i)}')(x)

        if self.include_top:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
        return x

def overwrite_eps(model,eps):
    for m in model.modules():
        if isinstance(m,FrozenBatchNorm2d):
            m.eps = eps

def sedla34_rfpn_backbone(pretrian_path='',
                          norm_layer = FrozenBatchNorm2d,
                          trainable_layers = 3,
                          returned_layer=None,
                          extra_blocks=None,steps=2):
    sedla_backbone = DLA([1,1,1,2,2,1],
                         [16,32,64,128,256,512],block=BasicBlock,include_top=False,norm_layer=norm_layer)
    if isinstance(norm_layer,FrozenBatchNorm2d):
        overwrite_eps(sedla_backbone,0.)
    if pretrian_path!='':
        print(sedla_backbone.load_state_dict(torch.load(pretrian_path),strict=False))
    assert 0<=trainable_layers<=5
    layers_to_train = ['level5','level4','level3','level2','level1'][:trainable_layers]
    for name,param in sedla_backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            param.requires_grad_(False)
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
    if returned_layer is None:
        returned_layer = [2,3,4,5]
    assert min(returned_layer)>0 and max(returned_layer)<=5
    return_layers = {f'level{int(k)}':str(v) for v,k in enumerate(returned_layer)}
    in_channel_list  = [sedla_backbone.channels[i] for i in returned_layer]
    out_channels = 256
    return BacbonewithRFPN(sedla_backbone,fan_in_list=in_channel_list,out_channel=out_channels,extral_block=extra_blocks,dilations=(1,3,6,1),steps=steps,
                           return_layers=return_layers,re_getter=True)

# if __name__ == '__main__':
#     import os
#     print(os.getcwd())
#     backbone = sedla34_rfpn_backbone('./dla34-ba72cf86.pth',trainable_layers=3)
#     print(backbone(torch.randn(2,3,224,224))['pool'].shape)