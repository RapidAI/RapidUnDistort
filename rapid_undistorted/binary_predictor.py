import math
import time

import cv2
import numpy as np

from rapid_undistorted.utils.img_transform import restore_original_size, pad_to_multiple_of_n
from rapid_undistorted.utils.infer_engine import OrtInferSession


class UnetCNN():
    def __init__(self, config: dict = None):
        self.unet_session = OrtInferSession(config)

    def __call__(self, img: np.ndarray):
        s = time.time()
        img, pad_info = self.preprocess(img)
        pred = self.unet_session([img])[0]
        out_img = self.postprocess(pred, pad_info)
        elapse = time.time() - s
        return out_img,elapse

    def preprocess(self, img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, pad_info = pad_to_multiple_of_n(img)
        # 归一化
        img = img.transpose(2, 0, 1) / 255.0
        # 将图像数据扩展为一个批次的形式
        img = np.expand_dims(img, axis=0).astype(np.float32)
        # 转换为模型输入格式
        return img, pad_info

    def postprocess(self,
                    img: np.ndarray,
                    pad_info):
        img = 1 - (img - img.min()) / (img.max() - img.min())
        img = img[0].transpose(1, 2, 0)
        # 重复最后一个通道维度三次
        img = np.repeat(img, 3, axis=2)
        img = (img * 255 + 0.5).clip(0, 255)
        img = restore_original_size(img, pad_info)
        return img
