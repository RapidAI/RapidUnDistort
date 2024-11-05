"""
Unwarp a document image using the model from ckpt_path.
"""
import cv2
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from doc_distort.utils import interpolate, grid_sample
# from UVDoc.utils import load_model
from model import UVDocnet


def load_model(ckpt_path):
    """
    Load UVDocnet model.
    """
    model = UVDocnet(num_filter=32, kernel_size=5)
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["model_state"])
    return model


IMG_SIZE = [488, 712]
GRID_SIZE = [45, 31]


img_path = "../img/demo1.jpg"
ckpt_path = "../model/best_model.pkl"
device = torch.device("cpu")

# Load model
model = load_model(ckpt_path)
model.to(device)
model.eval()

# Load image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
inp = torch.from_numpy(cv2.resize(img, IMG_SIZE).transpose(2, 0, 1)).unsqueeze(0)

# Make prediction
inp = inp.to(device)
point_positions2D, _ = model(inp)
# 假设你已经加载了模型并得到了 point_positions2D
size = img.shape[:2][::-1]
if __name__ == '__main__':

    # 将图像转换为NumPy数组
    warped_img = np.expand_dims(img.transpose(2, 0, 1), axis=0).astype(np.float32)
    size = img.shape[:2][::-1]
    # 上采样网格
    upsampled_grid = interpolate(point_positions2D.detach().numpy(), size=(size[1], size[0]), align_corners=True)
    # upsampled_grid = F.interpolate(
    #     torch.unsqueeze(point_positions2D[0], dim=0), size=(size[1], size[0]), mode="bilinear", align_corners=True
    # )
    # 调整网格的形状
    upsampled_grid = upsampled_grid.transpose(0, 2, 3, 1)

    # 重映射图像
    # unwarped_img = grid_sample(warped_img, upsampled_grid, align_corners=True)
    unwarped_img = grid_sample(warped_img, upsampled_grid)

    # 将结果转换回原始格式
    unwarped = (unwarped_img[0].transpose(1, 2, 0) * 255).astype(np.uint8)

    # 保存结果
    unwarped_BGR = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)
    cv2.imwrite("../output/unwarped_image.png", unwarped_BGR)