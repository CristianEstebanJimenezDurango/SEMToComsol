from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(path)

    return image


def save_image(path: Path, image: np.ndarray):

    path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(path), image)