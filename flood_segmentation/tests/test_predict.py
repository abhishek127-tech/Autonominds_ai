import os
import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import predict


def test_load_model():
    model = predict.load_model()
    assert isinstance(model, dict)
    assert model.get("loaded") is True


def test_analyze_flood_image_file_not_found(tmp_path):
    out_dir = tmp_path / "outputs"
    image_path = tmp_path / "non_existent_image.png"
    result = predict.analyze_flood_image(str(image_path), str(out_dir))

    assert result["status"] == "error"
    assert result["error_code"] == "FILE_NOT_FOUND"


def test_analyze_flood_image_stub_success(tmp_path):
    from PIL import Image

    img_path = tmp_path / "test_input.png"
    out_dir = tmp_path / "outputs"
    Image.new("RGB", (256, 256), color="blue").save(img_path)

    result = predict.analyze_flood_image(str(img_path), str(out_dir))

    assert result["status"] == "ok"
    assert "flood_mask_path" in result
    assert "overlay_path" in result
    assert result["image_width"] == 256
    assert result["image_height"] == 256
    assert os.path.exists(result["flood_mask_path"])
    assert os.path.exists(result["overlay_path"])


def test_analyze_flood_image_corrupt_input(tmp_path):
    image_path = tmp_path / "corrupt.png"
    image_path.write_text("not an image")

    result = predict.analyze_flood_image(str(image_path), str(tmp_path / "outputs"))

    assert result["status"] == "error"
    assert result["error_code"] == "CORRUPT_IMAGE"