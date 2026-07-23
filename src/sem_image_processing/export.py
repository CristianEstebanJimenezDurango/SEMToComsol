import ezdxf
import numpy as np


def export_dxf(
    splines,
    output_path,
    pixel_size_um=1.0,
):
    """
    Export spline contours to DXF.

    Parameters
    ----------
    splines : list of numpy arrays
        Each array has shape (N, 2)
        containing x, y coordinates in pixels.

    output_path : str
        Path to output DXF file.

    pixel_size_um : float
        Physical size of one pixel in micrometers.
    """

    # Create DXF document
    doc = ezdxf.new(
        "R2010"
    )

    # Get modelspace
    msp = doc.modelspace()

    for i, spline in enumerate(splines):

        # Convert pixel coordinates to micrometers
        points = [
            (
                float(x) * pixel_size_um,
                float(y) * pixel_size_um,
            )
            for x, y in spline
        ]

        # Add closed polyline
        msp.add_lwpolyline(
            points,
            close=True,
        )

        print(
            f"Exported contour {i + 1}: "
            f"{len(points)} points"
        )

    # Save DXF
    doc.saveas(
        output_path
    )

    print(
        f"\nDXF saved to:\n"
        f"{output_path}"
    )