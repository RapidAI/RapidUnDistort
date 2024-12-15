import os
import time
from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np
import yaml
from .binary_predictor import UnetCNN
from .unblur_predictor import NAF_DPM, OpenCvBilateral
from .unshadow_predictor import GCDRNET
from .unwrap_predictor import UVDocPredictor
from .utils.download_model import DownloadModel
from .utils.load_image import LoadImage
from .utils.logger import get_logger

root_dir = Path(__file__).resolve().parent
model_dir = os.path.join(root_dir, "models")
logger = get_logger("rapid_undistorted")
default_config = os.path.join(root_dir, "config.yaml")
ROOT_URL = "https://www.modelscope.cn/studio/jockerK/DocUnwrap/resolve/master/models/"
KEY_TO_MODEL_URL = {
    "unwrap": {
        "UVDoc": f"{ROOT_URL}/uvdoc.onnx",
    },
    "unshadow": {
        "GCDnet": f"{ROOT_URL}/gcnet.onnx",
        "DRnet": f"{ROOT_URL}/drnet.onnx",
    },
    "binarize": {
        "UnetCnn": f"{ROOT_URL}/unetcnn.onnx",
    },
    "unblur": {
        "NAFDPM": f"{ROOT_URL}/nafdpm.onnx",
    },
}
MODEL_CLASS_MAP = {
    "unwrap": {
        "UVDoc": UVDocPredictor,
    },
    "unshadow": {
        "GCDnet": GCDRNET,
    },
    "binarize": {
        "UnetCnn": UnetCNN,
    },
    "unblur": {
        "NAFDPM": NAF_DPM,
        "OpenCvBilateral": OpenCvBilateral
    }
}


class InferenceEngine:
    def __init__(self, config_path: str = str(default_config)):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.img_loader = LoadImage()
        self.tasks = config['tasks']
        self.models = {}
        self.configs = {}
        self.initialize_models()

    def initialize_models(self):
        for task, task_config in self.tasks.items():
            if 'models' not in task_config:
                raise ValueError(f"config has no models, task:{task}")
            if not self.configs.get(task, None):
                self.configs[task] = {}
                self.models[task] = {}
            for model_config in task_config['models']:
                model_type = model_config['type']
                model_class = MODEL_CLASS_MAP.get(task, {}).get(model_type, None)
                if not model_class:
                    raise ValueError(f"Model class {model_type} not found in MODEL_CLASS_MAP")
                if not self.configs[task].get(model_type, None):
                    self.configs[task][model_type] = {}
                    self.models[task][model_type] = {}
                if 'sub_models' in model_config:
                    for sub_model_config in model_config['sub_models']:
                        self.init_submodel_config(task, model_type, sub_model_config)
                else:
                    self.init_model_config(task, model_type, model_config)
                self.models[task][model_type] = model_class(self.configs[task][model_type])

    def init_model_config(self, task, model_type, model_config):
        model_path = model_config.get('path', None)
        use_cuda = model_config.get('use_cuda', False)
        use_dml = model_config.get('use_dml', False)
        # use model by model_path or download model
        model_path = self.get_model_path(task, model_type, model_path)
        self.configs[task][model_type] = {
            "model_path": model_path,
            "use_cuda": use_cuda,
            "use_dml": use_dml,
        }

    def init_submodel_config(self, task, model_type, sub_model_config):
        sub_model_type = sub_model_config['type']
        sub_model_path = sub_model_config.get('path', None)
        sub_use_cuda = sub_model_config.get('use_cuda', False)
        sub_use_dml = sub_model_config.get('use_dml', False)
        sub_model_path = self.get_model_path(task, sub_model_type, sub_model_path)
        self.configs[task][model_type][sub_model_type] = {
            "model_path": sub_model_path,
            "use_cuda": sub_use_cuda,
            "use_dml": sub_use_dml,
        }

    def __call__(
            self,
            img_content: Union[str, np.ndarray, bytes, Path],
            task_list: List[Union[str, Tuple[str, str]]]
    ) -> Tuple[np.ndarray, dict]:
        img = self.img_loader(img_content)
        elapses = {}

        for task in task_list:
            if isinstance(task, tuple):
                task_name, model_type = task
            else:
                task_name = task
                model_type = next(iter(self.models.get(task, [])))
            if not self.models.get(task_name, None):
                raise ValueError(f"Task '{task}' not found in the configuration.")
            if not self.models.get(task_name).get(model_type):
                raise ValueError(f"Task '{task}, Model Type : {model_type}' not found in the configuration.")
            if not elapses.get(task, None):
                elapses[task] = {}
            model_instance = self.models[task_name][model_type]
            img, elapse = model_instance(img)
            elapses[task][model_type] = elapse
        return img, elapses

    @staticmethod
    def get_model_path(task: str, model_type: str, model_path: Union[str, Path, None]) -> str:
        if model_path is not None:
            return model_path

        model_url = KEY_TO_MODEL_URL.get(task, {}).get(model_type, None)
        if model_url:
            model_path = DownloadModel.download(model_url)
            return model_path

        logger.info(
            "model url is None, using the default download model %s", model_path
        )
        return model_path
