import cv2
import numpy as np


def find_contours(
    binary: np.ndarray,
    min_area: float = 100.0,
):

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    filtered = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area >= min_area:
            filtered.append(contour)

    return filtered


def draw_contours(
    binary: np.ndarray,
    contours: list[np.ndarray],
):

    # Convert binary image to BGR
    preview = cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2BGR,
    )

    # Draw contours in red
    cv2.drawContours(
        preview,
        contours,
        -1,
        (0, 0, 255),
        2,
    )

    return preview


def save_preview(
    path,
    binary,
    contours,
):

    preview = draw_contours(
        binary,
        contours,
    )

    cv2.imwrite(
        str(path),
        preview,
    )