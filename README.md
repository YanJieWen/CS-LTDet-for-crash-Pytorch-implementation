# MS-SEDLA-for-crash-Pytorch-implementationch

**Noval application:** The study of Crashworthiness based on computer vision  

- [FrameWork](#FrameWork)
- [Training](#Training)
- [Testing](#Testing)
- [Results](#Results)
- [Acknowledgments](#Acknowledgments)
- [License](#license)


## FrameWork

We develop a new robust detector based on the two-stage detector, which includes the Squeeze-and-Excitation Deep Layer Aggregation network (SEDLA) as backbone and the rethinking mechanism of the scale fusion neck. Our overall model is as follows:

![image](demo/framework.jpg)


## Training
`Train directly`
We conducted experiments on the VOC small data set, and the mAP@0.5 was `81.6`. If you want to run the training model directly, you can run [train_engine](train_engine.py) directly. The premise is that your data should be placed in the [data] (data) directory like VOC type, and the pre-trained weights of the DLA backbone can be obtained in [Google Cloud Disk]() or [url](http://dl.yf.io/dla/models).
