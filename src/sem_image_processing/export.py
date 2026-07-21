import ezdxf


def export_dxf(
    filename,
    contours,
    scale=1.0,
):
    """
    Export contours to DXF.

    Parameters
    ----------
    filename : str
    contours : list
        List of Nx2 arrays.
    scale : float
        Pixel → micron conversion.
    """

    doc = ezdxf.new("R2010")

    msp = doc.modelspace()

    for contour in contours:

        points = []

        for x, y in contour:

            points.append(
                (
                    x * scale,
                    -y * scale,
                )
            )

        msp.add_lwpolyline(
            points,
            close=True,
        )

    doc.saveas(filename)