from pathlib import Path
import matplotlib.pyplot as plt

from .config import Config
from .io import load_image, save_image
from .preprocess import crop_bottom, denoise, remove_scale_bar, remove_background, clahe
from .segmentation import segment
from .contours import (
    find_contours,
    save_preview,
)

def main():

    cfg = Config()

    

    image = load_image(cfg.image_path)
    save_image(cfg.output_dir / "01_original.png", image)

    #image = remove_scale_bar(image)
    #save_image(cfg.output_dir / "02_no_scalebar.png", image)


    image = crop_bottom(image, cfg.crop_bottom_pixels)
    save_image(cfg.output_dir / "03_cropped.png", image)

    image = denoise(image, cfg.gaussian_kernel)
    save_image(cfg.output_dir / "04_denoised.png", image)

    #image = clahe(image)
    #save_image(cfg.output_dir / "05_clahe.png", image)


    #corrected, background = remove_background(image)

    #save_image(cfg.output_dir / "06_background.png", background)
    #save_image(cfg.output_dir / "07_corrected.png", corrected)

    output_dir = Path(
        "output_contours"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    binary = segment(image)
    save_image(cfg.output_dir / "08_binary.png", binary)

    # --------------------------------
    # Validate image
    # --------------------------------

    print(
        f"Shape: {binary.shape}"
    )

    print(
        f"Min: {binary.min()}"
    )

    print(
        f"Max: {binary.max()}"
    )

    print(
        f"Mean: {binary.mean():.2f}"
    )

    # --------------------------------
    # Extract contours
    # --------------------------------

    contours = find_contours(
        binary,
        min_area=100,
    )

    print(
        f"Found {len(contours)} contours."
    )

    # --------------------------------
    # Save contour preview
    # --------------------------------

    save_preview(
        output_dir / "contours_preview.png",
        binary,
        contours,
    )

    print(
        "Contour preview saved:"
    )

    print(
        output_dir / "contours_preview.png"
    )


if __name__ == "__main__":
    main()