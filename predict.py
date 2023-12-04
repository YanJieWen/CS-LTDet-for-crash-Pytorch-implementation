import os
import time
import json
import numpy as np
import torch
import torchvision
import PIL.ImageDraw as ImageDraw
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import sedla34
from train_utils.draw_box_utils import draw_objs
from train_utils.my_utils import save_fig
from network_files.transform import resize_boxes
from train_utils.grad_cam_utils import GradCAM,show_cam_on_image



#定义一个钩子
grad_cam = True
get_proposal = True
get_transform = True
features_in_hook = []
features_out_hook = []
def hook(moudle,fea_in,fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None

def create_model(num_classes):

    import torchvision
    from torchvision.ops.misc import FrozenBatchNorm2d
    backbone_with_rfpn = sedla34.sedla34_rfpn_backbone(pretrian_path='',
                                               norm_layer=FrozenBatchNorm2d, trainable_layers=0, steps=2)

    anchor_sizes = ((32,),(64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2','3'],
                                                    # 在哪些特征层上进行RoIAlign pooling,maxpool只用于RPN部分
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone_with_rfpn,
                       num_classes=num_classes+1,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致

    # backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    # model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=20)


    # load train weights
    weights_path = './save_weights/best.pth'
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './data/pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    demo_path = './demo/003.jpg'
    original_img = Image.open(demo_path)
    # original_img_visual = original_img.copy()
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension->[1,c,h,w]
    img = torch.unsqueeze(img, dim=0)
    # grad_cam================================================================
    if grad_cam:
        target_layers = [model.backbone.body.level5]
        cam = GradCAM(model=model, target_layers=target_layers,use_cuda=True,use_gradients=False)
        grayscale_cam = cam(input_tensor=img, target_category=None)
        grayscale_cam = grayscale_cam[0, :]
        _original_img_visual = np.array(original_img, dtype=np.uint8).copy()
        visualization = show_cam_on_image(_original_img_visual.astype(dtype=np.float32) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)

        plt.imshow(visualization)
        save_fig(f'Grad_cam-{demo_path.split(".")[1].split("/")[-1]}')
        plt.show()
        #
        model.eval()  # 进入验证模式
        with torch.no_grad():
            handels = []#存储hook
            for name, m in model.named_children():
                if name in ['transform', 'rpn']:
                    handels.append(m.register_forward_hook(hook=hook))
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))
            predictions = predictions[0]#选择第一个样本
            #可视化transform================================================================
            if get_transform:
                trans_img = torchvision.transforms.ToPILImage()(features_out_hook[2][0].tensors[0])
                plt.imshow(trans_img)
                save_fig(f'trans')
                plt.show()
            #======================================================================================
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))
            # #可视化proposals,筛选100个proposals进行可视化
            if get_proposal:
                visual_proposals = features_out_hook[-1][0][0][:100,:]
                h_r,w_r = model.transform(img.to(device),None)[0].image_sizes[0]
                ori_proposals = resize_boxes(visual_proposals, [h_r,w_r], [img_height, img_width])
                generater_proposals = ori_proposals.to('cpu').numpy()
                #=================================
                predict_boxes = predictions["boxes"].to("cpu").numpy()
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                # print(predict_boxes,'\n',predict_classes,'\n',predict_scores)

                # 绘制proposals
                prop_img = original_img.copy()
                draw = ImageDraw.Draw(prop_img)
                for box in generater_proposals:
                    left, top, right, bottom = box
                    # 绘制目标边界框
                    draw.line([(left, top), (left, bottom), (right, bottom),
                               (right, top), (left, top)], width=3, fill='#D00000')

                plt.imshow(prop_img)
                save_fig(f'proposals')
                plt.show()
            for handel in handels:
                handel.remove()
            # ======================================================================================
            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            plot_img = draw_objs(original_img,
                                 predict_boxes,
                                 predict_classes,
                                 predict_scores,
                                 category_index=category_index,
                                 box_thresh=0.5,
                                 line_thickness=8,
                                 font='arial.ttf',
                                 font_size=24)
            plt.imshow(plot_img)
            save_fig(f'detection')
            plt.show()



if __name__ == '__main__':
    main()
