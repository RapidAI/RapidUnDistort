import time

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from .utils.infer_engine import OrtInferSession


class UVDocPredictor:

    def __init__(self, config):
        self.session = OrtInferSession(config)
        self.img_size = [488, 712]
        self.grid_size = [45, 31]

    def __call__(self, img):
        s = time.time()
        size = img.shape[:2][::-1]
        img = img.astype(np.float32) / 255
        inp = self.preprocess(img.copy())
        outputs, _ = self.session([inp])
        elapse = time.time() - s
        return self.postprocess(img, size, outputs),elapse

    def preprocess(self, img):
        img = cv2.resize(img, self.img_size).transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, img, size, output):

        # 将图像转换为NumPy数组
        warped_img = np.expand_dims(img.transpose(2, 0, 1), axis=0).astype(np.float32)

        # 上采样网格
        upsampled_grid = self.interpolate(output, size=(size[1], size[0]), align_corners=True)
        # 调整网格的形状
        upsampled_grid = upsampled_grid.transpose(0, 2, 3, 1)

        # 重映射图像
        unwarped_img = self.grid_sample(warped_img, upsampled_grid)

        # 将结果转换回原始格式
        return unwarped_img[0].transpose(1, 2, 0) * 255

    def interpolate(self, input_tensor, size, align_corners=True):
        """
        Interpolate function to resize the input tensor.

        Args:
            input_tensor: numpy.ndarray of shape (B, C, H, W)
            size: tuple of int (new_height, new_width)
            mode: str, interpolation mode ('bilinear' or 'nearest')
            align_corners: bool, whether to align corners in bilinear interpolation

        Returns:
            numpy.ndarray of shape (B, C, new_height, new_width)
        """
        B, C, H, W = input_tensor.shape
        new_H, new_W = size
        resized_tensors = []
        for b in range(B):
            resized_channels = []
            for c in range(C):
                # 计算新的坐标
                if align_corners:
                    scale_h = (H - 1) / (new_H - 1) if new_H > 1 else 0
                    scale_w = (W - 1) / (new_W - 1) if new_W > 1 else 0
                else:
                    scale_h = H / new_H
                    scale_w = W / new_W

                # 创建新的坐标网格
                y, x = np.indices((new_H, new_W), dtype=np.float32)
                y = y * scale_h
                x = x * scale_w

                # 双线性插值
                coords = np.stack([y.flatten(), x.flatten()], axis=0)
                resized_channel = map_coordinates(input_tensor[b, c], coords, order=1, mode='constant', cval=0.0)
                resized_channel = resized_channel.reshape(new_H, new_W)
                resized_channels.append(resized_channel)

            resized_tensors.append(np.stack(resized_channels, axis=0))

        return np.stack(resized_tensors, axis=0)

    def grid_sample(self, input_tensor, grid, align_corners=True):
        """
        Grid sample function to sample the input tensor using the given grid.

        Args:
            input_tensor: numpy.ndarray of shape (B, C, H, W)
            grid: numpy.ndarray of shape (B, H, W, 2) with values in [-1, 1]
            align_corners: bool, whether to align corners in bilinear interpolation

        Returns:
            numpy.ndarray of shape (B, C, H, W)
        """
        B, C, H, W = input_tensor.shape
        B_grid, H_grid, W_grid, _ = grid.shape

        if B != B_grid or H != H_grid or W != W_grid:
            raise ValueError("Input tensor and grid must have the same spatial dimensions.")

        # Convert grid coordinates from [-1, 1] to [0, W-1] and [0, H-1]
        if align_corners:
            grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) * (W - 1) / 2
            grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) * (H - 1) / 2
        else:
            grid[:, :, :, 0] = ((grid[:, :, :, 0] + 1) * W - 1) / 2
            grid[:, :, :, 1] = ((grid[:, :, :, 1] + 1) * H - 1) / 2

        sampled_tensors = []
        for b in range(B):
            sampled_channels = []
            for c in range(C):
                channel = input_tensor[b, c]
                x_coords = grid[b, :, :, 0].flatten()
                y_coords = grid[b, :, :, 1].flatten()
                coords = np.stack([y_coords, x_coords], axis=-1)
                sampled_channel = map_coordinates(channel, coords.T, order=1, mode='constant', cval=0.0).reshape(H, W)
                sampled_channels.append(sampled_channel)
            sampled_tensors.append(np.stack(sampled_channels, axis=0))

        return np.stack(sampled_tensors, axis=0)
