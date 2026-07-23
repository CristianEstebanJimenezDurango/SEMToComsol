import numpy as np
from scipy.interpolate import splprep, splev


def smooth_contour(
    contour: np.ndarray,
    smoothing: float = 5.0,
    num_points: int = 500,
):
    """
    Fit a closed periodic B-spline to an OpenCV contour.

    The original contour point ordering is preserved.
    """

    # OpenCV contour:
    #
    # (N, 1, 2)
    #
    # Convert to:
    #
    # (N, 2)

    points = contour[:, 0, :].astype(float)

    # -------------------------------------------------
    # Remove only consecutive duplicate points
    # -------------------------------------------------

    if len(points) > 1:

        difference = np.diff(
            points,
            axis=0,
        )

        distance = np.linalg.norm(
            difference,
            axis=1,
        )

        keep = np.concatenate(
            [
                [True],
                distance > 0,
            ]
        )

        points = points[keep]

    # -------------------------------------------------
    # Check number of points
    # -------------------------------------------------

    if len(points) < 4:

        raise ValueError(
            "Contour has too few points for spline fitting."
        )

    # -------------------------------------------------
    # Separate x and y
    # -------------------------------------------------

    x = points[:, 0]
    y = points[:, 1]

    # -------------------------------------------------
    # Fit periodic B-spline
    # -------------------------------------------------

    tck, u = splprep(
        [x, y],
        s=smoothing,
        per=True,
    )

    # -------------------------------------------------
    # Generate smooth contour
    # -------------------------------------------------

    u_new = np.linspace(
        0,
        1,
        num_points,
        endpoint=False,
    )

    x_new, y_new = splev(
        u_new,
        tck,
    )

    spline_points = np.column_stack(
        [
            x_new,
            y_new,
        ]
    )

    return spline_points


def smooth_all(
    contours,
    smoothing: float = 5.0,
    num_points: int = 500,
):

    splines = []

    for i, contour in enumerate(contours):

        try:

            spline = smooth_contour(
                contour,
                smoothing=smoothing,
                num_points=num_points,
            )

            splines.append(
                spline
            )

            print(
                f"Spline {i + 1}: "
                f"{len(contour)} contour points "
                f"→ {len(spline)} spline points"
            )

        except Exception as error:

            print(
                f"Could not fit spline "
                f"to contour {i + 1}: {error}"
            )

    return splines