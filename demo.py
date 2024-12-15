import cv2
from rapid_undistorted.inference import InferenceEngine

if __name__ == '__main__':

    img_path = "tests/test_files/demo1.jpg"
    engine = InferenceEngine()
    unwrapped_img, elapse = engine(img_path, ["unwrap", "unshadow", ("unblur", "OpenCvBilateral")])
    print(f"doc unwrap elapse:{elapse}")
    cv2.imwrite("unwarped.png", unwrapped_img)
