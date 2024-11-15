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

### 最近更新

- **2024.11.15**
    - 完成初版代码，转换 [UVDoc](https://github.com/tanguymagne/UVDoc) 模型为onnx,完善前后处理


### 简介

本仓库用于进行文档扭曲的修正，同时能一定程度缓解透视和旋转问题


### 在线体验
[modelscope](https://www.modelscope.cn/studios/jockerK/DocUnwrap)
### 效果展示
![res_show.jpg](preview.jpg)

### 安装
``` python {linenos=table}
# 建议使用清华源安装 https://pypi.tuna.tsinghua.edu.cn/simple
pip install rapid-unwrap
```

### 快速使用

``` python {linenos=table}
import cv2

from inference import DocUnwrapper
img_path = "img/demo4.jpg"
doc_unwrapper = DocUnwrapper(img_path)
unwrapped_img, elapse = doc_unwrapper(img_path)
print(f"doc unwrap elapse:{elapse}")
cv2.imwrite("unwarped.png", unwrapped_img)

```


### 致谢

[UVDoc](https://github.com/tanguymagne/UVDoc)


### 贡献指南

欢迎提交请求。对于重大更改，请先打开issue讨论您想要改变的内容。

有其他的好建议和集成场景，作者也会积极响应支持

### 开源许可证

该项目采用[Apache 2.0](https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE)
开源许可证。

