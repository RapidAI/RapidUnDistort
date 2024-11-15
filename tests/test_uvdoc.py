import sys
from pathlib import Path

import pytest

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))
test_file_dir = cur_dir / "test_files"


@pytest.mark.parametrize(
    "img_path",
    [("demo2.png")],
)
def test_input_normal(img_path):
    from rapid_unwrap.inference import DocUnwrapper
    img_path = test_file_dir / img_path
    doc_unwrapper = DocUnwrapper()
    unwrapped_img, elapse = doc_unwrapper(img_path)

