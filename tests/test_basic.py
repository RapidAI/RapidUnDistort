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
def test_unwrap_uvdoc(img_path):
    from rapid_undistorted.inference import InferenceEngine
    img_path = test_file_dir / img_path
    engine = InferenceEngine()
    unwrapped_img, elapse = engine(img_path,[("unwrap", "UVDoc")])

@pytest.mark.parametrize(
    "img_path",
    [("demo1.png")],
)
def test_unshadow_gcnet(img_path):
    from rapid_undistorted.inference import InferenceEngine
    img_path = test_file_dir / img_path
    engine = InferenceEngine()
    unwrapped_img, elapse = engine(img_path,[("unshadow", "GCDnet")])

@pytest.mark.parametrize(
    "img_path",
    [("demo1.png")],
)
def test_unblur_opencv(img_path):
    from rapid_undistorted.inference import InferenceEngine
    img_path = test_file_dir / img_path
    engine = InferenceEngine()
    unwrapped_img, elapse = engine(img_path,[("unblur", "OpenCvBilateral")])

@pytest.mark.parametrize(
    "img_path",
    [("demo1.png")],
)
def test_unblur_nafnpm(img_path):
    from rapid_undistorted.inference import InferenceEngine
    img_path = test_file_dir / img_path
    engine = InferenceEngine()
    unwrapped_img, elapse = engine(img_path,[("unblur", "NAFDPM")])

@pytest.mark.parametrize(
    "img_path",
    [("demo1.png")],
)
def test_binarize_unetcnn(img_path):
    from rapid_undistorted.inference import InferenceEngine
    img_path = test_file_dir / img_path
    engine = InferenceEngine()
    unwrapped_img, elapse = engine(img_path,[("binarize", "UnetCnn")])
