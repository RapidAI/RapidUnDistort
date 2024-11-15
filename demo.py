import cv2

from rapid_unwrap.inference import DocUnwrapper
img_path = "img/demo.jpg"
doc_unwrapper = DocUnwrapper(img_path)
unwrapped_img, elapse = doc_unwrapper(img_path)
print(f"doc unwrap elapse:{elapse}")
cv2.imwrite("unwarped.png", unwrapped_img)
