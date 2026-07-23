from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
from .config import Config
from .io import load_image, save_image
from .preprocess import crop_bottom, denoise, remove_scale_bar, remove_background, clahe
from .segmentation import segment
from .contours import (
    find_contours,
    save_preview,
)

from .spline import (
    smooth_all,
)
from .export import export_dxf


def draw_splines(
    binary,
    splines,
):

    # Convert grayscale → BGR
    preview = cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2BGR,
    )

    for spline in splines:

        # Convert floating-point coordinates
        # to OpenCV integer coordinates

        points = np.round(
            spline
        ).astype(
            np.int32
        )

        # OpenCV expects shape:
        #
        # (N, 1, 2)

        points = points.reshape(
            (-1, 1, 2)
        )

        cv2.polylines(
            preview,
            [points],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
        )

    return preview


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

    # --------------------------------
    # Fit splines
    # --------------------------------

    splines = smooth_all(
        contours,
        smoothing=20,
        num_points=500,
    )

    print(
        f"Generated {len(splines)} splines."
    )

    # --------------------------------
    # Draw spline preview
    # --------------------------------

    spline_preview = draw_splines(
        binary,
        splines,
    )

    # --------------------------------
    # Save spline preview
    # --------------------------------

    spline_path = (
        output_dir
        / "spline_preview.png"
    )

    cv2.imwrite(
        str(spline_path),
        spline_preview,
    )

    print(
        f"Spline preview saved:"
    )

    print(
        spline_path
    )

 # -------------------------------------------------
    # Stage 4
    # Export DXF
    # -------------------------------------------------

    print(
        "Exporting DXF..."
    )

    pixel_size_um = (
        20.0 / 222.0
    )

    export_dxf(
        splines,
        output_path=output_dir / "pores.dxf",
        pixel_size_um=pixel_size_um,
    )

    print(
        "\nPipeline complete!"
    )

if __name__ == "__main__":
    main()