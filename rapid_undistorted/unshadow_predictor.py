import time

import cv2
import numpy as np

from rapid_undistorted.utils.infer_engine import OrtInferSession

class GCDRNET():
    def __init__(self, config=None):
        gcnet_config = config.get("GCDnet")
        drnet_config = config.get("DRnet")
        self.gcnet_session = OrtInferSession(gcnet_config)
        self.drnet_session = OrtInferSession(drnet_config)

    def __call__(self, img):
        s = time.time()
        im_padding, padding_h, padding_w = self.preprocess(img.copy())
        img_shadow = im_padding.copy()
        img_shadow = self.gcnet_session([img_shadow])[0]
        model1_im = np.clip(im_padding / img_shadow, 0, 1)
        # 拼接 im_org 和 model1_im
        concatenated_input = np.concatenate((im_padding, model1_im), axis=1)
        pred = self.drnet_session([concatenated_input])[0]
        elapse = time.time() - s
        return self.postprocess(pred, padding_h, padding_w), elapse

    def stride_integral(self, img, stride=32):
        h, w = img.shape[:2]

        if (h % stride) != 0:
            padding_h = stride - (h % stride)
            img = cv2.copyMakeBorder(img, padding_h, 0, 0, 0, borderType=cv2.BORDER_REPLICATE)
        else:
            padding_h = 0

        if (w % stride) != 0:
            padding_w = stride - (w % stride)
            img = cv2.copyMakeBorder(img, 0, 0, padding_w, 0, borderType=cv2.BORDER_REPLICATE)
        else:
            padding_w = 0

        return img, padding_h, padding_w

    def preprocess(self, img):
        img, padding_h, padding_w = self.stride_integral(img)
        # 归一化
        img = img.transpose(2, 0, 1) / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        # 转换为模型输入格式
        return img, padding_h, padding_w

    def postprocess(self, pred, padding_h, padding_w):
        pred = np.transpose(pred[0], (1, 2, 0))
        pred = pred * 255
        enhance_img = pred[padding_h:, padding_w:]
        return enhance_img
