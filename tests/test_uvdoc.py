import sys
from pathlib import Path

import pytest

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))
test_file_dir = cur_dir / "test_files"


@pytest.mark.parametrize(
    "img_path",
    [("demo1.png")],
)
def test_input_normal(img_path):
    from rapid_undistorted.inference import InferenceEngine
    img_path = test_file_dir / img_path
    engine = InferenceEngine()
    unwrapped_img, elapse = engine(img_path,["(unwrap, UVDoc)"])

