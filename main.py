# from pathlib import Path
# from typing import Optional, Union, Tuple
# import numpy as np
# from doc_distort.utils import LoadImage
#
# root_dir = Path(__file__).resolve().parent
# class DocDistort:
#     def __init__(self, model_path: Optional[str] = None, model_type: str = None):
#         if model_path is None:
#             model_path = str(
#                 root_dir / "model" / "uvdoc.onnx"
#             # )
#             # model_type = "slanet-plus"
#         self.model_type = model_type
#         self.load_img = LoadImage()
#         self.table_structure = TableStructurer(model_path)
#         self.table_matcher = TableMatch()
#
#         # try:
#         #     self.ocr_engine = importlib.import_module("rapidocr_onnxruntime").RapidOCR()
#         # except ModuleNotFoundError:
#         #     self.ocr_engine = None
#
#     def __call__(
#         self,
#         img_content: Union[str, np.ndarray, bytes, Path],
#         # ocr_result: List[Union[List[List[float]], str, str]] = None,
#     ) -> Tuple[str, float]:
#
#
#         img = self.load_img(img_content)
#
#         s = time.time()
#         h, w = img.shape[:2]
#
#         if ocr_result is None:
#             ocr_result, _ = self.ocr_engine(img)
#         dt_boxes, rec_res = self.get_boxes_recs(ocr_result, h, w)
#
#         pred_structures, pred_bboxes, _ = self.table_structure(copy.deepcopy(img))
#         # 适配slanet-plus模型输出的box缩放还原
#         if self.model_type == "slanet-plus":
#             pred_bboxes = self.adapt_slanet_plus(img, pred_bboxes)
#         pred_html = self.table_matcher(pred_structures, pred_bboxes, dt_boxes, rec_res)
#
#         elapse = time.time() - s
#         return pred_html, pred_bboxes, elapse
