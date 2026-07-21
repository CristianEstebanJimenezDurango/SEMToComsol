from dataclasses import dataclass

import numpy as np


@dataclass
class Pore:

    id: int

    contour: np.ndarray

    area: float

    centroid: tuple[float, float]

    spline: np.ndarray | None = None


@dataclass
class Geometry:

    pores: list[Pore]

    pixel_size: float