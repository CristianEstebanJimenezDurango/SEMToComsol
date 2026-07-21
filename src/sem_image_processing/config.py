from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Config:
    image_path: Path = Path("images/sample.jpeg")
    output_dir: Path = Path("output")

    # Calibration
    scale_length_um: float = 20.0
    scale_length_pixels: float = 218.0

    # Pre-processing
    crop_bottom_pixels: int = 70
    gaussian_kernel: tuple[int, int] = (5, 5)

    # Segmentation
    morphology_kernel: int = 3
    min_area: float = 500

    crop_x = (640, 974)
    crop_y = (610, 710)