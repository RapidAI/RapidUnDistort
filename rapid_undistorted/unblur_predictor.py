import math
import time

import cv2
import numpy as np

from rapid_undistorted.utils.img_transform import restore_original_size, pad_to_multiple_of_n
from rapid_undistorted.utils.infer_engine import OrtInferSession

class NAF_DPM():
    def __init__(self, config=None):

        self.naf_dpm_session = OrtInferSession(config)
    def __call__(self, img: np.ndarray):
        s = time.time()
        img = self.preprocess(img)
        pred = self.naf_dpm_session([img])[0]
        out_img = self.postprocess(pred)
        elapse = time.time() - s
        return out_img,elapse


    def preprocess(self, img: np.ndarray):
        # 归一化
        img = img.transpose(2, 0, 1) / 255.0
        # 将图像数据扩展为一个批次的形式
        img = np.expand_dims(img, axis=0).astype(np.float32)
        # 转换为模型输入格式
        return img
    
    def postprocess(self,
                    img: np.ndarray):
        img = img[0]
        img = (img * 255 + 0.5).clip(0, 255).transpose(1, 2, 0)
        return img


class OpenCvBilateral:
    def __init__(self, config=None):
        pass
    def __call__(self, img):
        s = time.time()
        img = img.astype(np.uint8)
        # 双边滤波
        bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        # 自适应直方图均衡化
        lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 应用锐化滤波器
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        elapse = time.time() - s
        return sharpened,elapse
