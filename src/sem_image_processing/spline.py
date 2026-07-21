import numpy as np

from scipy.interpolate import splprep
from scipy.interpolate import splev


def smooth_contour(
    contour,
    smoothing=20,
    n_points=500,
):

    contour = contour[:, 0, :]

    x = contour[:, 0]
    y = contour[:, 1]

    tck, u = splprep(
        [x, y],
        s=smoothing,
        per=True,
    )

    u_new = np.linspace(0, 1, n_points)

    x_new, y_new = splev(
        u_new,
        tck,
    )

    return np.column_stack((x_new, y_new))


def smooth_all(
    contours,
    smoothing=20,
    n_points=500,
):

    smooth = []

    for contour in contours:

        smooth.append(
            smooth_contour(
                contour,
                smoothing=smoothing,
                n_points=n_points,
            )
        )

    return smooth