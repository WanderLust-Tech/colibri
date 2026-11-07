#!/usr/bin/env python3
"""Gemma 4's image preprocessing: a picture in, the grid of patches its vision tower reads out.

A port of transformers' Gemma4ImageProcessor (image_processing_pil_gemma4.py), which
tests/test_gemma4_image.py compares with the real one:

  1. the picture is converted to RGB (transparency composited onto white);
  2. it is resized, keeping its aspect ratio, to the largest height and width that are multiples of
     patch * pooling (16 * 3 = 48 pixels) and make at most max_soft_tokens * pooling^2 patches
     (280 soft tokens: 2520 patches), with PIL's bicubic filter;
  3. it is cut into patch x patch squares, row by row, each square flattened as (row, column, channel).

What the engine takes (the payload of the IMAGE frame) is those patches as bytes, before the division by 255
the model's own code does: a quarter of the size of float32 patches, and exactly what the processor made.
The tower finds the position of every patch from the grid, so nothing else travels.

Needs Pillow and numpy (the gateway itself needs neither).
"""
import io
import math

# The processor accepts these budgets of soft tokens per picture; more tokens, more detail, more time.
SUPPORTED_TOKENS = (70, 140, 280, 560, 1120)


def soft_token_budget(limit, wanted=None):
    """The soft tokens per picture to ask the processor for: the biggest supported budget that is not more than
    `limit` (what the engine accepts), or `wanted` when that is lower and supported. A limit below the smallest
    supported budget is used as it is."""
    options = [n for n in SUPPORTED_TOKENS if n <= limit]
    if not options:
        return max(1, int(limit))
    if wanted is not None and wanted in options:
        return wanted
    return options[-1]


def aspect_ratio_preserving_size(height, width, patch_size, max_patches, pooling_kernel_size):
    """Gemma4's own rule: the largest height and width, multiples of patch_size * pooling_kernel_size, that keep
    the aspect ratio and make at most max_patches patches."""
    total_px = height * width
    target_px = max_patches * patch_size ** 2
    factor = math.sqrt(target_px / total_px)
    side = pooling_kernel_size * patch_size
    target_height = int(math.floor(factor * height / side)) * side
    target_width = int(math.floor(factor * width / side)) * side
    if target_height == 0 and target_width == 0:
        raise ValueError(f"a {width}x{height} picture is too small or too thin: it would be resized to 0x0 "
                         f"(sides must be multiples of {side})")
    max_side = (max_patches // pooling_kernel_size ** 2) * side
    if target_height == 0:
        target_height = side
        target_width = min(int(math.floor(width / height)) * side, max_side)
    elif target_width == 0:
        target_width = side
        target_height = min(int(math.floor(height / width)) * side, max_side)
    if target_height * target_width > target_px:
        raise ValueError(f"a {width}x{height} picture would be resized to {target_width}x{target_height}, "
                         f"which is more than {max_patches} patches")
    return target_height, target_width


def preprocess(data, max_soft_tokens=280, patch_size=16, pooling_kernel_size=3):
    """(patches, grid_h, grid_w) for the bytes of an image file: patches is a uint8 array
    [grid_h * grid_w, patch_size * patch_size * 3]. The soft tokens the picture makes are
    (grid_h // pooling_kernel_size) * (grid_w // pooling_kernel_size)."""
    import numpy as np
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        # transparency is composited onto white, as the processor does (not dropped: a transparent pixel's colour
        # under the mask is often black)
        if "A" in image.getbands() or image.info.get("transparency") is not None:
            image = image.convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", image.size, (255, 255, 255, 255)), image).convert("RGB")
        else:
            image = image.convert("RGB")
    except Exception as problem:
        raise ValueError(f"not an image this server can read ({type(problem).__name__}: {problem})") from None
    height, width = image.height, image.width
    target_height, target_width = aspect_ratio_preserving_size(
        height, width, patch_size, max_soft_tokens * pooling_kernel_size ** 2, pooling_kernel_size)
    if (target_height, target_width) != (height, width):
        image = image.resize((target_width, target_height), Image.BICUBIC)
    pixels = np.asarray(image, dtype=np.uint8)
    grid_h, grid_w = target_height // patch_size, target_width // patch_size
    patches = (pixels.reshape(grid_h, patch_size, grid_w, patch_size, 3)
               .transpose(0, 2, 1, 3, 4).reshape(grid_h * grid_w, patch_size * patch_size * 3))
    return np.ascontiguousarray(patches), grid_h, grid_w
