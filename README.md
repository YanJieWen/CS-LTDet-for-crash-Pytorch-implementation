# MS-SEDLA-for-crash-Pytorch-implementationch

**Noval application:** The study of Crashworthiness based on computer vision  

- [FrameWork](#FrameWork)
- [Training](#Training)
- [Validation](#Validation)
- [Results](#Results)
- [Acknowledgments](#Acknowledgments)
- [License](#license)


## FrameWork

We develop a new robust detector based on the two-stage detector, which includes the Squeeze-and-Excitation Deep Layer Aggregation network (SEDLA) as backbone and the rethinking mechanism of the scale fusion neck. Our overall model is as follows:

![image](demo/framework.jpg)


## Training
**Train directly**    
We conducted experiments on the VOC small data set, and the mAP@0.5 was `81.6` and achieved good results in the top 10 of the [Tensorboard](https://paperswithcode.com/sota/object-detection-on-pascal-voc-2007). If you want to run the training model directly, you can run [train_engine](train_engine.py) directly. The premise is that your data should be placed in the [data] (data) directory like VOC type, and the pre-trained weights of the DLA backbone can be obtained in [Google Cloud Disk](https://drive.google.com/file/d/1gS1SVWw4hHdcpxTilRpwHZcYvxsZh2dH/view?usp=drive_link) or [url](http://dl.yf.io/dla/models) and put it into the [pretrain](backbone/pretrain) root.   
You can switch between two training modes: `VOC07trainval-VOC07test(VOC07)` and `VOC07trainval+VOC12trainval12(VOC07+12)`. Just modify `line 91` and `line 125` in [train_engine](train_engine.py).

**Continue training**  
If you want to continue training, you can download the trained weight file from [Google Cloud Disk ](https://drive.google.com/file/d/1Fv84ZJSLZBog-cc1LWtIiC1ylSfW-zJU/view?usp=drive_link). Next, put it into [save_weights](save_weights) and modify `resume` in [train_engine](train_engine.py) to point to the weights file.  

**Train your own data**
You first need to create the data set in VOC format. Of course, you can also do it according to any data format of [COCO](https://cocodataset.org/#home), but just rewrite a [datasets.py](my_dataset/datasets.py), about COCO dataset For reading, you can refer to [pycocotools](https://pypi.org/project/pycocotools/). Next, you need to make a classification similar to (pascal_voc_classes.json)[data/pascal_voc_classes.json]. The file is read as a dictionary. Finally, you only need to modify the `num_classes` and `class_path` parameters to train your dataset.  

## Validation  
Similar to training, you only need to modify `num_classes` and `label_json_path` for evaluation, and it will generate an evaluation txt file [record_mAP.txt](record_mAP.txt).

## Results  
Run [predict.py](predict.py) to generate interpretable heat maps, proposal coverage maps, data augmentation images and detection results in the `demo` folder. The results are stored in the `exe` directory.
**without Grad-Grad-cam：**  
![image](exe/Grad_cam-003.png)

**Transforms:**   
![image](exe/trans.png)

**Proposals:** 
![image](exe/proposals.png)

**Detections:**
![image](exe/detection.png)

## Acknowledgments  
[Grad-Cam(EigenCAM)](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)[[**URL**]](https://github.com/jacobgil/pytorch-grad-cam ) 
[DLA BASE](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Deep_Layer_Aggregation_CVPR_2018_paper.html)[[**URL**]](https://github.com/ucbdrive/dla  )
[RFPN](https://openaccess.thecvf.com/content/CVPR2021/html/Qiao_DetectoRS_Detecting_Objects_With_Recursive_Feature_Pyramid_and_Switchable_Atrous_CVPR_2021_paper.html)[[**URL**]](https://github.com/joe-siyuan-qiao/DetectoRS) 
[Otehrs](https://www.bilibili.com/video/BV1of4y1m7nj/?spm_id_from=333.999.0.0)[[**URL**]](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)

## license
[MIT](LICENSE) © YanjieWen

