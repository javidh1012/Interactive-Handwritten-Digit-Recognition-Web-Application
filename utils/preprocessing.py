"""
Preprocessing module for Handwritten Digit Recognition.

Transforms canvas drawings into the exact 28x28 grayscale format
expected by the trained PyTorch MNIST model:
  1. Base64 / Bytes decoding
  2. Alpha blending & Grayscale conversion
  3. Automatic polarity detection (white digit on black background)
  4. Empty-canvas detection
  5. Bounding box cropping preserving aspect ratio (max dimension = 20px)
  6. Center-of-mass translation to (14, 14) on 28x28 field (standard MNIST)
  7. Normalization and Tensor conversion to shape [1, 1, 28, 28]
"""

import base64
import io
import re
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch


# Standard MNIST dataset normalization constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""
    pass


class EmptyCanvasError(PreprocessingError):
    """Raised when the input canvas contains no drawing."""
    pass


def decode_image_data(image_data: str | bytes) -> Image.Image:
    """
    Decodes an image from base64 string or raw bytes into a PIL Image.
    Handles data URLs like 'data:image/png;base64,...'.
    """
    if isinstance(image_data, str):
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            raise PreprocessingError(f"Failed to decode base64 image data: {e}")
    elif isinstance(image_data, (bytes, bytearray)):
        image_bytes = bytes(image_data)
    else:
        raise PreprocessingError(f"Unsupported image data type: {type(image_data)}")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()  # Force load image data
        return image
    except Exception as e:
        raise PreprocessingError(f"Unable to parse image data with PIL: {e}")


def prepare_grayscale_digit(img: Image.Image) -> Image.Image:
    """
    Converts image to grayscale with white digit on black background.
    Handles RGBA transparency, inverted drawings (black ink on white paper), etc.
    """
    # 1. Handle transparency if RGBA
    if img.mode in ("RGBA", "LA"):
        # Create black background by default
        background = Image.new("RGB", img.size, (0, 0, 0))
        # Use alpha channel as mask
        alpha = img.split()[-1]
        background.paste(img.convert("RGB"), mask=alpha)
        gray = background.convert("L")
    else:
        gray = img.convert("L")

    # 2. Check background polarity:
    # If the corners / borders are predominantly bright (> 127), the user drew
    # dark ink on a light background. Invert so it becomes white stroke on black.
    w, h = gray.size
    border_pixels = []
    # Sample border regions
    border_pixels.extend([gray.getpixel((0, y)) for y in range(0, h, max(1, h // 10))])
    border_pixels.extend([gray.getpixel((w - 1, y)) for y in range(0, h, max(1, h // 10))])
    border_pixels.extend([gray.getpixel((x, 0)) for x in range(0, w, max(1, w // 10))])
    border_pixels.extend([gray.getpixel((x, h - 1)) for x in range(0, w, max(1, w // 10))])

    if len(border_pixels) > 0 and np.mean(border_pixels) > 127:
        # Invert: white background -> black, dark ink -> white
        import PIL.ImageOps as ImageOps
        gray = ImageOps.invert(gray)

    return gray


def check_empty_canvas(img_arr: np.ndarray, threshold: float = 10.0, min_pixels: int = 15) -> None:
    """
    Checks if the canvas is empty or contains only noise.
    """
    active_pixels = np.sum(img_arr > threshold)
    if active_pixels < min_pixels or np.max(img_arr) < 25.0:
        raise EmptyCanvasError("Canvas is empty. Please draw a digit before predicting.")


def center_of_mass(img_arr: np.ndarray) -> Tuple[float, float]:
    """
    Computes center of mass (cx, cy) for a 2D intensity array.
    """
    total = float(np.sum(img_arr))
    if total <= 1e-6:
        return 14.0, 14.0
    y_coords, x_coords = np.indices(img_arr.shape)
    cy = float(np.sum(y_coords * img_arr) / total)
    cx = float(np.sum(x_coords * img_arr) / total)
    return cx, cy


def center_digit_mnist_standard(gray_img: Image.Image) -> Image.Image:
    """
    Centers the digit following the official MNIST dataset pipeline:
      - Crop to bounding box of active pixels (> 20 intensity)
      - Resize so the largest dimension is 20 pixels, maintaining aspect ratio
      - Place onto a 28x28 black canvas
      - Translate so center of mass is located at (14, 14)
    """
    arr = np.array(gray_img, dtype=np.float32)
    check_empty_canvas(arr)

    # Threshold for bounding box
    binary_mask = arr > 20.0
    active_coords = np.argwhere(binary_mask)
    if len(active_coords) == 0:
        raise EmptyCanvasError("No digit strokes detected on the canvas.")

    y_min, x_min = active_coords.min(axis=0)
    y_max, x_max = active_coords.max(axis=0) + 1

    # Crop digit
    cropped = gray_img.crop((x_min, y_min, x_max, y_max))
    w, h = cropped.size

    # Fit into 20x20 box
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create 28x28 black background and paste resized digit initially centered geometrically
    canvas_28 = Image.new("L", (28, 28), color=0)
    initial_x = (28 - new_w) // 2
    initial_y = (28 - new_h) // 2
    canvas_28.paste(resized, (initial_x, initial_y))

    # Center of mass translation
    arr_28 = np.array(canvas_28, dtype=np.float32)
    cx, cy = center_of_mass(arr_28)

    shift_x = int(round(14.0 - cx))
    shift_y = int(round(14.0 - cy))

    # Perform translation with clipping bounds to avoid wrap-around
    # Shift in X and Y
    shifted_arr = np.zeros_like(arr_28)
    for y in range(28):
        for x in range(28):
            src_y = y - shift_y
            src_x = x - shift_x
            if 0 <= src_y < 28 and 0 <= src_x < 28:
                shifted_arr[y, x] = arr_28[src_y, src_x]

    final_img = Image.fromarray(np.clip(shifted_arr, 0, 255).astype(np.uint8), mode="L")
    return final_img


def image_to_base64_thumbnail(img: Image.Image) -> str:
    """
    Converts a 28x28 PIL image into a base64 PNG data URL for preview.
    """
    buffer = io.BytesIO()
    # Upscale slightly with nearest neighbor for sharp display in UI
    preview = img.resize((140, 140), Image.Resampling.NEAREST)
    preview.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def preprocess_image(
    image_data: str | bytes,
    normalize: bool = True,
    mean: float = MNIST_MEAN,
    std: float = MNIST_STD
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Full preprocessing pipeline from raw input to PyTorch tensor.

    Returns:
      tensor: PyTorch FloatTensor of shape [1, 1, 28, 28]
      debug_info: dict with preview thumbnail and dimensions
    """
    # 1. Decode
    raw_img = decode_image_data(image_data)

    # 2. Convert to grayscale with correct polarity
    gray_img = prepare_grayscale_digit(raw_img)

    # 3. Center and resize to 28x28 following MNIST conventions
    final_28x28 = center_digit_mnist_standard(gray_img)

    # 4. Generate thumbnail for frontend visualization
    thumbnail_data_url = image_to_base64_thumbnail(final_28x28)

    # 5. Convert to NumPy array [0.0, 1.0]
    np_img = np.array(final_28x28, dtype=np.float32) / 255.0

    # 6. Apply normalization
    if normalize:
        np_img = (np_img - mean) / std

    # 7. Convert to PyTorch Tensor: shape [1, 1, 28, 28]
    tensor = torch.from_numpy(np_img).view(1, 1, 28, 28).to(dtype=torch.float32)

    debug_info = {
        "thumbnail": thumbnail_data_url,
        "input_shape": list(tensor.shape),
        "normalized": normalize,
        "mean": mean if normalize else None,
        "std": std if normalize else None
    }

    return tensor, debug_info

