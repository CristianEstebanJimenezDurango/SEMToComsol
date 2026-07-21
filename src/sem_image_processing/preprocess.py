import cv2
import numpy as np


def crop_bottom(image: np.ndarray, pixels: int):

    if pixels <= 0:
        return image

    return image[:-pixels, :]


def denoise(image: np.ndarray, kernel=(5, 5)):

    return cv2.GaussianBlur(image, kernel, 0)


def remove_scale_bar(image: np.ndarray):

    result = image.copy()

    h, w = result.shape

    # Bottom-right region containing the scale bar
    y1 = int(0.84 * h)
    x1 = int(0.68 * w)

    background = int(np.median(result[:100, :100]))

    result[y1:h, x1:w] = background

    return result

def remove_background(image: np.ndarray):

    background = cv2.GaussianBlur(
        image,
        (51, 51),
        0,
    )

    corrected = cv2.subtract(
        image,
        background,
    )

    corrected = cv2.normalize(
        corrected,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    return corrected, background

def clahe(image: np.ndarray):

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8),
    )

    return clahe.apply(image)