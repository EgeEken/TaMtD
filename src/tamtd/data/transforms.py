from PIL import Image, ImageOps


def prepare_image(image: Image.Image, image_size: int) -> tuple[Image.Image, dict]:
    rgb = image.convert("RGB")
    transformed = ImageOps.fit(
        rgb,
        (image_size, image_size),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    return transformed, {
        "operation": "rgb_center_crop_resize",
        "image_size": image_size,
        "centering": [0.5, 0.5],
    }
