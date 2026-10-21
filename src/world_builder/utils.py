"""Small numeric helpers shared by the ocean and clouds."""

import math

import numpy as np


def magnitude(vector: np.ndarray) -> float:
    """
    Returns the Euclidean norm (magnitude) of vector: a 2D screen-space
    vector, a 3D direction/speed, or a quaternion's 4 raw components.

    Plain floats and math.sqrt instead of np.linalg.norm, which is far slower
    on tiny vectors.
    """
    if len(vector) == 3:
        x, y, z = vector[0], vector[1], vector[2]
        return math.sqrt(x * x + y * y + z * z)
    if len(vector) == 2:
        x, y = vector[0], vector[1]
        return math.sqrt(x * x + y * y)
    if len(vector) == 4:
        x, y, z, w = vector[0], vector[1], vector[2], vector[3]
        return math.sqrt(x * x + y * y + z * z + w * w)
    return math.sqrt(float(np.dot(vector, vector)))


def clamp(value: float, minimum: float, maximum: float) -> float:
    """
    :return: value clamped to [minimum, maximum] (exactly min(max(value,
        minimum), maximum))
    """
    return min(max(value, minimum), maximum)


def compute_next_power_of_2(x: float) -> int:
    """
    Computes the smallest power of two greater than or equal to x

    :param x: The reference number
    :return: That power of 2, as an int (at least 1)
    """
    return 2 ** math.ceil(math.log2(max(x, 1)))
