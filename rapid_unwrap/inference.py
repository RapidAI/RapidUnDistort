import os
import time
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np

from .predictor import UVDocPredictor
from .utils.download_model import DownloadModel
from .utils.load_image import LoadImage
from .utils.logger import get_logger

root_dir = Path(__file__).resolve().parent
model_dir = os.path.join(root_dir, "models")
ROOT_URL = "https://www.modelscope.cn/studio/jockerK/DocUnwrap/resolve/master/models/"
KEY_TO_MODEL_URL = {
    "UVDoc": f"{ROOT_URL}/uvdoc.onnx",
}
ROOT_DIR = Path(__file__).resolve().parent
logger = get_logger("rapid_unwrap")


class DocUnwrapper:
    def __init__(self,
                 use_cuda=False,
                 use_dml=False,
                 model_path: Optional[str] = None,
                 model_type: str = "UVDoc"):
        self.img_loader = LoadImage()
        config = {
            "model_path": self.get_model_path(model_type, model_path),
            "use_cuda": use_cuda,
            "use_dml": use_dml,
        }

        self.unwrap_model = UVDocPredictor(config)

    def __call__(
            self,
            img_content: Union[str, np.ndarray, bytes, Path],
    ) -> Tuple[str, float]:
        img = self.img_loader(img_content)
        s = time.time()
        unwrapped_img = self.unwrap_model(img)
        elapse = time.time() - s
        return unwrapped_img, elapse

    @staticmethod
    def get_model_path(model_type: str, model_path: Union[str, Path, None]) -> str:
        if model_path is not None:
            return model_path

        model_url = KEY_TO_MODEL_URL.get(model_type, None)
        if model_url:
            model_path = DownloadModel.download(model_url)
            return model_path

        logger.info(
            "model url is None, using the default download model %s", model_path
        )
        return model_path
