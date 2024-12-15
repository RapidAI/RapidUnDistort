import cv2
import numpy as np


def pad_to_multiple_of_n(image, n=32):
    original_height, original_width = image.shape[:2]

    # 计算目标形状
    target_width = ((original_width + n - 1) // n) * n
    target_height = ((original_height + n - 1) // n) * n

    # 创建一个纯白背景的图像
    padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    # 计算填充的位置
    start_x = (target_width - original_width) // 2
    start_y = (target_height - original_height) // 2

    # 将原始图像放置在纯白背景上
    padded_image[start_y:start_y + original_height, start_x:start_x + original_width] = image

    # 返回填充后的图像和填充位置
    return padded_image, (start_x, start_y, original_height, original_width)

def restore_original_size(image, pad_info):
    start_x, start_y, original_height, original_width = pad_info

    # 去掉填充部分
    cropped_image = image[start_y:start_y + original_height, start_x:start_x + original_width]

    return cropped_image

# def resize_and_pad(image, target_shape):
#     original_height, original_width = image.shape[:2]
#     target_height, target_width = target_shape
#
#     # 计算缩放比例
#     scale = min(target_width / original_width, target_height / original_height)
#
#     # 计算新的尺寸
#     new_width = int(original_width * scale)
#     new_height = int(original_height * scale)
#
#     # 缩放图像
#     resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
#
#     # 创建一个纯白背景的图像
#     padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
#
#     # 计算填充的位置
#     start_x = (target_width - new_width) // 2
#     start_y = (target_height - new_height) // 2
#
#     # 将缩放后的图像放置在纯白背景上
#     padded_image[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image
#
#     return padded_image, (start_x, start_y,original_height, original_width, new_height, new_width)
#
# def restore_original_size(image, pad_info):
#     start_x, start_y, original_height, original_width, new_height, new_width = pad_info
#
#     # 去掉填充部分
#     cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]
#
#     # 缩放回原大小
#     restored_image = cv2.resize(cropped_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
#
#     return restored_image
