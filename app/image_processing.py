
"""Image processing utilities for camera focus."""

from PIL import Image

from app.config import FOCUS_BOX_SIZE


def get_focus_box(
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Calculate a centered square focus area."""
    size = min(
        FOCUS_BOX_SIZE,
        image_width,
        image_height,
    )

    left = (image_width - size) // 2
    top = (image_height - size) // 2

    right = left + size
    bottom = top + size

    return left, top, right, bottom


def crop_focus_area(image: Image.Image) -> Image.Image:
    """Crop the centered focus area from an image."""
    focus_box = get_focus_box(*image.size)

    return image.crop(focus_box)
