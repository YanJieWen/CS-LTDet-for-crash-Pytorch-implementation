import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
import torchvision

import train_utils.data_aug
from network_files.image_list import ImageList


@torch.jit.unused
def _resize_image_onnx(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    from torch.onnx import operators
    im_shape = operators.shape_as_tensor(image)[-2:]
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)

    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


def _resize_image(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))    # 获取高宽中的最小值
    max_size = float(torch.max(im_shape))    # 获取高宽中的最大值
    scale_factor = self_min_size / min_size  # 根据指定最小边长和图片最小边长计算缩放比例

    # 如果使用该缩放比例计算的图片最大边长大于指定的最大边长
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size  # 将缩放比例设为指定最大边长和图片最大边长之比

    # interpolate利用插值的方法缩放图片
    # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
    # bilinear只支持4D Tensor
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size      # 指定图像的最小边长范围
        self.max_size = max_size      # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std    # 指定图像在标准化处理中的方差

    def normalize(self, image):
        """标准化处理"""
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)#转为tensor
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # [:, None, None]: shape [3] -> [3, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        将图片缩放到指定的大小范围内，并对应缩放bboxes信息
        Args:
            image: 输入的图片
            target: 输入图片的相关信息（包括bboxes信息）

        Returns:
            image: 缩放后的图片
            target: 缩放bboxes后的图片相关信息
        """
        # image shape is [channel, height, width]
        h, w = image.shape[-2:]

        if self.training:
            size = float(self.torch_choice(self.min_size))  # 指定输入图片的最小边长,注意是self.min_size不是min_size
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])    # 指定输入图片的最小边长,注意是self.min_size不是min_size

        if torchvision._is_tracing():
            image = _resize_image_onnx(image, size, float(self.max_size))
        else:
            image = _resize_image(image, size, float(self.max_size))

        if target is None:
            return image, target

        bbox = target["boxes"]
        # 根据图像的缩放比例来缩放bbox
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, [0, padding[2], 0, padding[1], 0, padding[0]])
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        """
        将一批图像打包成一个batch返回（注意batch中每个tensor的shape是相同的）
        Args:
            images: 输入的一批图片
            size_divisible: 将图像高和宽调整到该数的整数倍

        Returns:
            batched_imgs: 打包成一个batch后的tensor数据
        """

        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        # 分别计算一个batch中所有图片中的最大channel, height, width
        max_size = self.max_by_axis([list(img.shape) for img in images])

        stride = float(size_divisible)
        # max_size = list(max_size)
        # 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch, channel, height, width]
        batch_shape = [len(images)] + max_size

        # 创建shape为batch_shape且值全部为0的tensor
        batched_imgs = images[0].new_full(batch_shape, 0)#与image为共享的gpu
        for img, pad_img in zip(images, batched_imgs):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            # copy_: Copies the elements from src into self tensor and returns self
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self,
                    result,                # type: List[Dict[str, Tensor]]
                    image_shapes,          # type: List[Tuple[int, int]]
                    original_image_sizes   # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        """
        对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        Args:
            result: list(dict), 网络的预测结果, len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size

        Returns:

        """
        if self.training:
            return result

        # 遍历每张图片的预测信息，将boxes信息还原回原尺度
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)  # 将bboxes缩放回原图像尺度上
            result[i]["boxes"] = boxes
        return result

    def __repr__(self):
        """自定义输出实例化对象的信息，可通过print打印实例信息"""
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string

    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        norm_imgs = []
        resize_imgs = []
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image_norm = self.normalize(image)# 1.对图像进行标准化处理
            norm_imgs.append(image_norm)
            image_resize, target_index = self.resize(image_norm, target_index) # 2.对图像和对应的bboxes缩放到指定范围
            resize_imgs.append(image_resize)
            images[i] = image_resize#这里有一个替换操作，因此需要copy
            if isinstance(targets,tuple):
                targets = list(targets)
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录resize后的图像尺寸
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)  # 将images打包成一个batch
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)#打包后的图像tensor以及打包前resize后的图像高宽
        return image_list, targets,norm_imgs,resize_imgs


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    将boxes参数根据图像的缩放情况进行相应缩放

    Arguments:
        original_size: 图像缩放前的尺寸
        new_size: 图像缩放后的尺寸
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)



# 测试
# if __name__ == '__main__':
#     import random
#     import copy
#     import matplotlib.pyplot as plt
#     from my_dataset.datasets import Voc2dataset
#     from train_utils.data_aug import Data_augmentation
#     from train_utils.draw_box_utils import draw_objs
#     import torchvision.transforms as ts
#     from train_utils.my_utils import *
#     import numpy as np
#     VOC_root = '../data'
#     class_file = '../data/pascal_voc_classes.json'
#     transforms = Data_augmentation()
#     data = Voc2dataset(VOC_root, class_file, transforms, train_set=False, train_type='07')
#     category_index = {str(v): str(k) for k, v in data.class_dict.items()}#(0->person)
#     train_data_loader = torch.utils.data.DataLoader(data,
#                                                     batch_size=5,
#                                                     shuffle=True,
#                                                     num_workers=0,
#                                                     collate_fn=data.collate_fn)
#     inputs = next(iter(train_data_loader))
#     imgs = inputs[0]#tuple[tensors]
#     targets = inputs[1]#tuple[Dict]
#     min_size = 800
#     max_size = 1333
#     image_mean = [0.485, 0.456, 0.406]
#     image_std = [0.229, 0.224, 0.225]
#     tss = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
#
#     targets_ = copy.deepcopy(targets)#tss类会修改targets需要完全独立一个变量出来
#     imgs_ = copy.deepcopy(imgs)#tss类会修改images需要完全独立一个变量出来
#     #转到GPU
#     for im in imgs:
#         im.to(try_gpu())
#     for tg in targets:
#         for key,value in tg.items():
#             value.to(try_gpu())
#     im_list,tgt,norm_imgs,resize_imgs = tss(imgs,targets)#返回一个tensor以及缩放后的形状list[dict]
#     print(im_list.image_sizes)
#     # #可视化
#     i = 0
#     total_plots = []
#     for id,im in enumerate(imgs_):
#         ori_tgt = targets_[id]
#         new_tgt = tgt[id]
#         origin_img = ts.ToPILImage()(im)
#         norm_img = ts.ToPILImage()(norm_imgs[id])
#         resize_img = ts.ToPILImage()(resize_imgs[id])
#         padd_img = ts.ToPILImage()(im_list.tensors[id])
#         plot_or_im = draw_objs(origin_img,ori_tgt['boxes'].numpy(),ori_tgt['labels'].numpy(),
#                                np.ones(ori_tgt["labels"].shape[0]),category_index=category_index,box_thresh=0.5,
#                                line_thickness=3,font='arial.ttf',font_size=18)
#         plot_norm_im = draw_objs(norm_img,ori_tgt['boxes'].numpy(),ori_tgt['labels'].numpy(),
#                                np.ones(ori_tgt["labels"].shape[0]),category_index=category_index,box_thresh=0.5,
#                                line_thickness=3,font='arial.ttf',font_size=18)
#         plot_resize_im = draw_objs(resize_img,new_tgt['boxes'].numpy(),new_tgt['labels'].numpy(),
#                                np.ones(new_tgt["labels"].shape[0]),category_index=category_index,box_thresh=0.5,
#                                line_thickness=3,font='arial.ttf',font_size=18)
#         plot_pad_im = draw_objs(padd_img,new_tgt['boxes'].numpy(),new_tgt['labels'].numpy(),
#                                np.ones(new_tgt["labels"].shape[0]),category_index=category_index,box_thresh=0.5,
#                                line_thickness=3,font='arial.ttf',font_size=18)
#         total_plots.append([plot_or_im,plot_norm_im,plot_resize_im,plot_pad_im])
    # #可视化
    # i = 0
    # plt.xticks([])
    # plt.yticks([])
    # for plot in total_plots:
    #     for idx in range(len(plot)):
    #         plt.subplot(1,len(plot),idx+1)
    #         plt.imshow(plot[idx])
    #     save_fig(f'{int(i)}')
    #     i+=1
    #     plt.show()
    # 设置xtick和ytick的方向：in、out、inout
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
    # for id,dr in enumerate(total_plots):
    #     if id==1:
    #         i=0
    #         for d in dr:
    #             plt.imshow(d)
    #             save_fig(f'{int(i)}')
    #             i+=1
    #             plt.show()



