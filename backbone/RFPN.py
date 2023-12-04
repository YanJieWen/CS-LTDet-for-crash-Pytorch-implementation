# -*- coding:utf-8 -*-
"""
Software:PyCharm
File: RFPN.py
Institution: --- CSU&BUCEA ---, China
E-mail: obitowen@csu.edu.cn
Author：Yanjie Wen
Date：2023年10月21日
My zoom: https://github.com/YanJieWen
"""
import torch
import torch.nn as nn
from typing import Any, cast, Dict, List, Optional, Union
from collections import OrderedDict
from backbone.feature_pyramid_network import FeaturePyramidNetwork,LastLevelMaxPool
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor

class ASPP(nn.Module):#空洞空间金字塔池化，并行的多个空洞卷积层
    def __init__(self,fan_in,fan_out,dilations=(1,3,6,1)) -> None:
        super().__init__()
        assert dilations[-1]==1
        self.dilations = dilations
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation>1 else 1
            padding = dilation if dilation>1 else 0
            conv = nn.Conv2d(fan_in,fan_out,kernel_size=kernel_size,stride=1,dilation=dilation,padding=padding,bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        avg_x = self.gap(x)#压缩尺度->(b,d,1,1)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx==len(self.aspp)-1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        # print([o.shape for o in out])
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out,dim=1)
        return out#->(b,d*4,h,w)
#基础rpn网络
class RPNbase(nn.Module):
    def __init__(self,backbones,fpn,aspp,gated_fusion,steps,**kwargs) -> None:
        super(RPNbase,self).__init__(**kwargs)
        self.bs = backbones
        self.fpn = fpn
        self.aspp = aspp
        self.fusion = gated_fusion
        self.steps = steps
    def forward(self,x:Dict=None,
                img:Tensor=None):
        #type: (Dict[str,Tensor],Tensor) -> Dict[str,Tensor]
        names = list(x.keys())
        x = self.fpn(x)#带pool层#[('0', torch.Size([12, 256, 200, 304])),
        #('1', torch.Size([12, 256, 100, 152])), ('2', torch.Size([12, 256, 50, 76])),
        #('3', torch.Size([12, 256, 25, 38])),
        #('pool', torch.Size([12, 256, 13, 19]))]
        for s in range(self.steps-1):
            rfp_feats = list([self.aspp(value) for _,value in x.items()])
            x_idx = self.bs[s](img,rpn_feats=rfp_feats)
            x_idx = self.fpn(x_idx)
            x_new = OrderedDict()
            for key,value in x_idx.items():
                add_weight = torch.sigmoid(self.fusion(value))
                if key not in x_new.keys():
                    x_new[key] = add_weight*value+(1-add_weight)*x[key]
            x = x_new
        if self.fpn.extra_blocks is not None:
            results,names = self.fpn.extra_blocks([value for key,value in x.items()],names)
            x = OrderedDict([(k,v) for k,v in zip(names,results)])
        return x
#类继承
class RecursiveFeaturePyramid(RPNbase):
    def __init__(self, backbone,steps,
                 fan_in_list,fan_out,extra_block,#FPN参数
                 dilations=(1,3,6,1)) -> None:
        #骨干网络
        backbones = nn.ModuleList()
        for s in range(1,steps):
            backbones.append(backbone)
        #FPN
        fpn = FeaturePyramidNetwork(fan_in_list,fan_out,extra_block)
        #ASPP
        if fan_out%len(dilations)!=0:
            raise ValueError('number of dilation is not correct!')
        aspp = ASPP(fan_out,fan_out//len(dilations),dilations=dilations)
        #weight
        gated_fusion = nn.Conv2d(fan_out,1,kernel_size=1,stride=1,padding=0,bias=True)
        super(RecursiveFeaturePyramid,self).__init__(backbones,fpn,aspp,gated_fusion,steps)
#针对frpn设计的可变输入模式
class IntermediateLayerGetterRFPN(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构，舍去之后不用的结构
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x,rpn_feats=None):
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer1, layer2, layer3, layer4的输出
        i = 0
        for name, module in self.items():
            if rpn_feats is None:
                x = module(x)
                if name  in self.return_layers:
                    out_name = self.return_layers[name]
                    out[out_name] = x
            else:
                if name  not in self.return_layers:
                    x = module(x)
                else:
                    x = module(x,rpn_feats[i])
                    out_name = self.return_layers[name]
                    out[out_name] = x
                    i+=1
        return out

class BacbonewithRFPN(nn.Module):
    def __init__(self,backbone:nn.Module,
                 steps:int,
                 fan_in_list:List,
                 return_layers:Dict,
                 out_channel:int,
                 extral_block=None,
                 dilations=(1,3,6,1),re_getter=False):
        super().__init__()
        if extral_block is None:
            extral_block = LastLevelMaxPool()
        if re_getter:
            assert return_layers is not None
            self.body = IntermediateLayerGetterRFPN(backbone,return_layers)
        else:
            self.body  = backbone#提取特征层
        self.out_channels = out_channel
        self.rfpn = RecursiveFeaturePyramid(self.body,steps,
                 fan_in_list,out_channel,extral_block,#FPN参数
                 dilations=dilations)
    def forward(self,x):
        _x = self.body(x)
        x = self.rfpn(_x,x)
        return x

