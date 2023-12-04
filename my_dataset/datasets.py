# -*- coding:utf-8 -*-
"""
Software:PyCharm
File: datasets.py
Institution: --- CSU&BUCEA ---, China
E-mail: obitowen@csu.edu.cn
Author：Yanjie Wen
Date：2023年07月12日
My zoom: https://github.com/YanJieWen
"""
import random

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import json
from PIL import Image
from lxml import etree
import os
#============================
from train_utils.data_aug import Data_augmentation
from train_utils.draw_box_utils import draw_objs#与上面的Image产生冲突
from torchvision.transforms import functional as F
import torchvision.transforms as ts
import matplotlib.pyplot as plt
import numpy as np
'''
We built a custom data factory script with random data augmentation
It includes two core methods to return the length of the __len__ data set & __getitem__ 
to return the image and image information (dictionary):
At the same time, we also provide a random data augmentation component based on imgaug as a regular term
├── VOC2dataset: 自制数据集类，调用的属性root,logger,img_root,annotations_root,xml_list,class_dict,transforms
  ├── __len__(self):返回数据集的长度
  ├── __getitem__(self, idx): 返回(image,img_inform)
  ├── parse_xml2dict: 解析xml文件，基于lxml库
  ├── get_height_and_width: 获取图像的高和宽
  ├── coco_index: 用于pycocotools统计标签信息准备，不对图像和标签作任何处理，由于不用去读取图片，可大幅缩减统计时间
  ├── collate_fn: 静态方法，解析batch    
'''

class Voc2dataset(Dataset):
    def __init__(self, voc_root, class_file, transforms, train_set=True,train_type='07'):
        super().__init__()
        #加载VOC07trainval训练，07test测试
        if train_type == '07':
            self.root_07 = os.path.join(voc_root,'VOCdevkit','VOC2007')
            self.image_root_07 = os.path.join(self.root_07,'JPEGImages')
            self.annotations_root_07 = os.path.join(self.root_07,'Annotations')
            self.train_set = train_set
            if train_set:
                txt_list = os.path.join(self.root_07, 'ImageSets', 'Main', 'trainval.txt')
            else:
                txt_list = os.path.join(self.root_07, 'ImageSets', 'Main', 'test.txt')
            with open(txt_list) as r:
                xml_list = [os.path.join(self.annotations_root_07, line.strip() + ".xml")
                            for line in r.readlines() if len(line.strip()) > 0]
                r.close()
        #加载voc07trainval+12trainval训练，07test进行测试
        elif train_type == '07+12':
            self.root_07 = os.path.join(voc_root,'VOCdevkit','VOC2007')
            self.root_12 = os.path.join(voc_root,'VOCdevkit','VOC2012')
            self.image_root_07 = os.path.join(self.root_07,'JPEGImages')
            self.image_root_12 = os.path.join(self.root_12, 'JPEGImages')
            self.annotations_root_07 = os.path.join(self.root_07, 'Annotations')
            self.annotations_root_12 = os.path.join(self.root_12, 'Annotations')
            self.train_set = train_set
            if train_set:
                txt_list_07 = os.path.join(self.root_07, 'ImageSets', 'Main', 'trainval.txt')
                txt_list_12 = os.path.join(self.root_12, 'ImageSets', 'Main', 'trainval.txt')
                with open(txt_list_07) as r, open(txt_list_12) as r1:
                    xml_list = [os.path.join(self.annotations_root_07, line.strip() + ".xml")
                                for line in r.readlines() if len(line.strip()) > 0]
                    r.close()
                with open(txt_list_12) as r:
                    xml_list.extend([os.path.join(self.annotations_root_12, line.strip() + ".xml")
                                for line in r.readlines() if len(line.strip()) > 0])
                    r.close()
            else:
                txt_list = os.path.join(self.root_07, 'ImageSets', 'Main', 'test.txt')
                with open(txt_list) as r:
                    xml_list = [os.path.join(self.annotations_root_07, line.strip() + ".xml")
                                for line in r.readlines() if len(line.strip()) > 0]
                    r.close()


        self.xml_list = []
        for xml_path in xml_list:  # 判断xml文件中的目标是否为空，即有效的xml文件
            if os.path.exists(xml_path) is False:
                print(f'Warning: Not found {xml_path}-->skip this annotation file')
                continue
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml2dict(xml)['annotation']
            if 'object' not in data:
                print(f'No object in {xml_path}-->skip this annotation file')
                continue
            self.xml_list.append(xml_path)#存储annotation的路径
        print(f'There are total {int(len(self.xml_list))} images in this dataloder!')
        assert len(self.xml_list) > 0, print(f'in {txt_list} does not find any information, please check it')
        assert os.path.exists(class_file), print(f'{class_file} is not exits in the path')
        with open(class_file, 'r') as f:
            self.class_dict = json.load(f)
        self.transforms = transforms

    def __len__(self):  # 返回数据集的样本数
        return len(self.xml_list)

    def __getitem__(self, idx):  # 核心
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml2dict(xml)['annotation']
        img_root = self.image_root_07 if data['folder']=='VOC2007' else self.image_root_12
        img_path = os.path.join(img_root, data['filename'])
        image = Image.open(img_path)
        if image.format != 'JPEG':
            raise TypeError(f'Image format not JPEG, you need transform it!')

        boxes = []
        labels = []
        iscrowd = []
        for obj in data['object']:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            if xmax <= xmin or ymax <= ymin:
                print(f'{xml_path}/{obj["name"]} w/h <=0')
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)
        # 转换为torch张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # [n,4]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # 是否数据增强
        if self.transforms is not None and self.train_set:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)
        return image, target

    def parse_xml2dict(self, xml):
        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml2dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def get_height_and_width(self, idx):  # 获取图像的高和宽
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml2dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def coco_index(self, idx):
        #用于coco评估，遍历每一张图片
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml2dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target#返回一个字典Dict{str:tensor}

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

# if __name__ == '__main__':
#     from train_utils.my_utils import save_fig
#
#     root = '../data/'
#     class_file = '../data/pascal_voc_classes.json'
#     transforms = Data_augmentation()
#     train_set = True
#     data = Voc2dataset(root, class_file, transforms, train_set=True,train_type='07')
#     category_index = {str(v): str(k) for k, v in data.class_dict.items()}#->反转后为['int':class]
#     #绘制
#     print(category_index,data.class_dict)
#     plt.rcParams['xtick.direction'] = 'in'
#     plt.rcParams['ytick.direction'] = 'in'
#     for index in random.sample(range(0,len(data)),k=5):
#         img,tag = data[index]
#         boxes = tag["boxes"].numpy()
#         # if np.any(boxes[:,0]==0.) and np.any(boxes[:,2]==0.):#检查是否有边界框在图像的外侧
#         img =ts.ToPILImage()(img)
#         plot_img = draw_objs(img,
#                              tag["boxes"].numpy(),
#                              tag["labels"].numpy(),
#                              np.ones(tag ["labels"].shape[0]),
#                              category_index=category_index,
#                              box_thresh=0.5,
#                              line_thickness=3,
#                              font='arial.ttf',
#                              font_size=18)  # 绘制图
#         # print(tag['boxes'].numpy())
#
#         plt.imshow(img)
#         save_fig(f'roda_{index}')
#         plt.show()