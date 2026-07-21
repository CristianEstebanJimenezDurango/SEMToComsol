import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_contours(binary, min_area=1000):
    """
    Find pore contours from a binary image.

    Parameters
    ----------
    binary : ndarray
        Binary image.
    min_area : float
        Minimum contour area.

    Returns
    -------
    list
        List of filtered contours.
    """

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


def draw_contours(image, contours):

    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    for i, contour in enumerate(contours):

        color = colors[i % len(colors)]

        cv2.drawContours(
            canvas,
            [contour],
            -1,
            color,
            2,
        )

        M = cv2.moments(contour)

        if M["m00"] != 0:

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.putText(
                canvas,
                str(i + 1),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

    return canvas


def save_preview(filename, preview):

    plt.figure(figsize=(8, 8))
    plt.imshow(preview)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()