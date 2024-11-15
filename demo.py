import cv2

from uvdocpredictor import UVDocPredictor
if __name__ == '__main__':

    img_path = "img/demo1.png"
    model_path = "model/uvdoc.onnx"
    predictor = UVDocPredictor(model_path)
    unwrapped_img = predictor(img_path)
    unwrapped_img = cv2.cvtColor(unwrapped_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output/unwarped.png", unwrapped_img)