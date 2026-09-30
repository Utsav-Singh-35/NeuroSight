"""Unit tests for the image preprocessing service."""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from app.services.preprocessor import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_image,
)


def _create_image_bytes(size=(100, 100), mode="RGB", format="PNG"):
    """Helper to create image bytes for testing."""
    if mode == "L":
        img = Image.fromarray(np.random.randint(0, 256, size, dtype=np.uint8), mode="L")
    elif mode == "RGBA":
        img = Image.fromarray(
            np.random.randint(0, 256, (*size, 4), dtype=np.uint8), mode="RGBA"
        )
    else:
        img = Image.fromarray(
            np.random.randint(0, 256, (*size, 3), dtype=np.uint8), mode="RGB"
        )
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


class TestPreprocessImage:
    """Tests for preprocess_image function."""

    def test_output_shape(self):
        """Output tensor has shape (1, 3, 224, 224)."""
        img_bytes = _create_image_bytes(size=(300, 400), mode="RGB")
        result = preprocess_image(img_bytes)
        assert result.shape == (1, 3, 224, 224)

    def test_output_dtype(self):
        """Output tensor has dtype float32."""
        img_bytes = _create_image_bytes()
        result = preprocess_image(img_bytes)
        assert result.dtype == torch.float32

    def test_output_values_in_imagenet_normalised_range(self):
        """Values fall inside the per-channel range ImageNet normalisation allows.

        The preprocessor scales to [0, 1] and *then* applies ImageNet mean/std,
        so the output range is [-mean/std, (1 - mean)/std] per channel — not
        [0, 1]. This test previously asserted [0, 1] and failed; the assertion
        was wrong, not the preprocessor, which must keep matching training.
        """
        img_bytes = _create_image_bytes()
        result = preprocess_image(img_bytes)

        lower = -IMAGENET_MEAN / IMAGENET_STD
        upper = (1.0 - IMAGENET_MEAN) / IMAGENET_STD

        for channel in range(3):
            values = result[0, channel]
            assert float(values.min()) >= float(lower[channel]) - 1e-5
            assert float(values.max()) <= float(upper[channel]) + 1e-5

    def test_grayscale_input(self):
        """Grayscale images are converted to 3-channel RGB."""
        img_bytes = _create_image_bytes(size=(50, 50), mode="L")
        result = preprocess_image(img_bytes)
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == torch.float32

    def test_rgba_input(self):
        """RGBA images have alpha channel discarded."""
        img_bytes = _create_image_bytes(size=(200, 200), mode="RGBA")
        result = preprocess_image(img_bytes)
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == torch.float32

    def test_small_image_upscaled(self):
        """Small images are resized up to 224x224."""
        img_bytes = _create_image_bytes(size=(10, 10), mode="RGB")
        result = preprocess_image(img_bytes)
        assert result.shape == (1, 3, 224, 224)

    def test_large_image_downscaled(self):
        """Large images are resized down to 224x224."""
        img_bytes = _create_image_bytes(size=(1000, 1500), mode="RGB")
        result = preprocess_image(img_bytes)
        assert result.shape == (1, 3, 224, 224)

    def test_jpeg_format(self):
        """JPEG format images are processed correctly."""
        img_bytes = _create_image_bytes(size=(100, 100), mode="RGB", format="JPEG")
        result = preprocess_image(img_bytes)
        assert result.shape == (1, 3, 224, 224)

    def test_invalid_bytes_raises_valueerror(self):
        """Invalid image bytes raise ValueError."""
        with pytest.raises(ValueError, match="Image could not be processed"):
            preprocess_image(b"not an image")

    def test_empty_bytes_raises_valueerror(self):
        """Empty bytes raise ValueError."""
        with pytest.raises(ValueError, match="Image could not be processed"):
            preprocess_image(b"")

    def test_normalization_correctness(self):
        """Pure white maps to (1 - mean)/std and pure black to -mean/std.

        The original version of this test asserted white -> 1.0 and black -> 0.0,
        which describes only the first half of the pipeline. Changing the
        preprocessor to satisfy it would have introduced a train/serve skew,
        since the models were trained with ImageNet normalisation.
        """
        white_img = Image.new("RGB", (224, 224), (255, 255, 255))
        buf = io.BytesIO()
        white_img.save(buf, format="PNG")
        result = preprocess_image(buf.getvalue())
        expected_white = torch.tensor(
            (1.0 - IMAGENET_MEAN) / IMAGENET_STD, dtype=torch.float32
        ).view(1, 3, 1, 1)
        assert torch.allclose(result, expected_white.expand_as(result), atol=1e-5)

        black_img = Image.new("RGB", (224, 224), (0, 0, 0))
        buf = io.BytesIO()
        black_img.save(buf, format="PNG")
        result = preprocess_image(buf.getvalue())
        expected_black = torch.tensor(
            -IMAGENET_MEAN / IMAGENET_STD, dtype=torch.float32
        ).view(1, 3, 1, 1)
        assert torch.allclose(result, expected_black.expand_as(result), atol=1e-5)
