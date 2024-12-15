<div align="center">
  <div align="center">
    <h1><b>📊RapidUnWrap</b></h1>
  </div>
  <a href=""><img src="https://img.shields.io/badge/Python->=3.8,<3.13-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Mac%2C%20Win-pink.svg"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE"><img alt="GitHub" src="https://img.shields.io/badge/license-Apache 2.0-blue"></a>
</div>

### Recent Updates

- **2024.11.15**
  - Completed the initial version of the code, converted the [UVDoc](https://github.com/tanguymagne/UVDoc) model to onnx, and improved pre- and post-processing.
- **2024.12.15**
  - Added deblurring/shadow removal/binarization functions and models, upgraded to RapidUndistort.
  
### Introduction

This repository is used for correcting document distortion, deblurring documents, shadow removal, and document binarization.
It provides multiple models and flexible task combinations, supports automatic model downloading.
Original PyTorch model sources can be found in the [Acknowledgments](#acknowledgments) section.
[Quick Start](#quick-start) [Usage Suggestions](#usage-suggestions) [Parameter Explanation](#parameter-explanation) [Model Address](https://www.modelscope.cn/studios/jockerK/DocUnwrap/files)

### Online Demo
[modelscope](https://www.modelscope.cn/studios/jockerK/DocUnwrap) [huggingface](https://huggingface.co/spaces/Joker1212/RapidUnwrap)

### Effect Showcase
![res_show.jpg](preview1.gif)
![res_show1.jpg](preview2.gif)

### Installation
``` python {linenos=table}
pip install rapid-undistorted
```

### Quick Start

``` python {linenos=table}
import cv2

from rapid_undistorted.inference import InferenceEngine
img_path = "img/demo.jpg"
engine = InferenceEngine()
# Distortion correction -> Shadow removal -> Deblurring (specify deblurring model)
output_img, elapse = engine(img_path, ["unwrap", "unshadow", ("unblur", "OpenCvBilateral")])
# Shadow removal -> Deblurring (specify deblurring model)
#output_img, elapse = engine(img_path, ["unshadow", ("unblur", "OpenCvBilateral")])
# Default selection of the first unblur model in the yaml configuration file
#output_img, elapse = engine(img_path, ["unshadow", "unblur"])
# Binarization as an alternative to shadow removal method
#output_img, elapse = engine(img_path, ["unwrap", "binarize", "unblur"])
print(f"doc unwrap elapse:{elapse}")
cv2.imwrite("result.png", output_img)

```

### Usage Suggestions
- For English and numeric deblurring, the NAFDPM model is better, but for Chinese text, using the OpenCV method is more suitable.
- The shadow removal model has richer functionality and better results compared to binarization, so it is not recommended to directly use the binarization method.


### Parameter Explanation
#### Initialization Parameters
Supports passing a config configuration file to declare the required task types and corresponding models, as well as paths.
```python
config_path = "configs/config.yaml"
engine = InferenceEngine(config_path)
```
```yaml
tasks:
  unwrap:
    models:
      - type: "UVDoc"
        path:
        use_cuda: false
        
  unshadow:
    models:
      - type: "GCDnet"
        sub_models:
          - type: "GCDnet"
            path:
            use_cuda: false
            use_dml: false
          - type: "DRnet"
            path:
            use_cuda: false

  binarize:
    models:
      - type: "UnetCnn"
        path:
        use_cuda: false

  unblur:
    models:
      - type: "OpenCvBilateral"
        path:
      - type: "NAFDPM"
        path:
        use_cuda: false

```
#### Execution Parameters
```python
engine(img_path, task_list)
engine(img_path, ["unwrap", "unshadow", ("unblur", "OpenCvBilateral")])
```

### Acknowledgments

unwrap: [UVDoc](https://github.com/tanguymagne/UVDoc)
unshadow: [GCDnet](https://github.com/ZZZHANG-jx/GCDRNet)
unblur: [NAFDPM](https://github.com/ispamm/NAF-DPM)
binarize: [UnetCnn](https://github.com/sajjanvsl/U-Net-CNN-for-binarization-of-Historical-Kannada-Handwritten-Palm-Leaf-Manuscripts)



### Contribution Guidelines

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

If you have other good suggestions or integration scenarios, the author will actively respond and support them.


### Open Source License

This project is licensed under the [Apache 2.0](https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE) license.
