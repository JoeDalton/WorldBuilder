"""Small numeric helpers shared by the ocean, the clouds and the shadows."""

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


#: A sun moved by less than this (radians) leaves the shadow maps drawn for it
#: as they are: they are not marched or baked again for a step that small
SUN_TOLERANCE = math.radians(0.05)


def sun_moved(drawn: np.ndarray | None, sun: np.ndarray) -> bool:
    """
    :param drawn: the sun a map was drawn for (unit vector), None = none yet
    :param sun: the sun now (unit vector)
    :returns: whether they are more than :data:`SUN_TOLERANCE` apart
    """
    if drawn is None:
        return True
    cosine = float(np.clip(np.dot(drawn, sun), -1.0, 1.0))
    return cosine < math.cos(SUN_TOLERANCE)
