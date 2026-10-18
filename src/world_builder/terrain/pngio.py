"""
Minimal PNG writing with zlib, and PNG reading through Panda's PNMImage.

Writing is pure Python, so tests and worker processes need no image library.
Reading goes through Panda (already a dependency) because a decoder for every
PNG flavour is not worth writing.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def write_png(path: Path | str, image: np.ndarray) -> None:
    """Write an image, top row first.

    :param path: destination file
    :param image: (h, w) or (h, w, 3|4), uint8 (8-bit) or uint16 (16-bit)
    """
    if image.ndim == 2:
        image = image[:, :, None]
    height, width, channels = image.shape
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    if image.dtype == np.uint16:
        depth, data = 16, image.astype(">u2")
    elif image.dtype == np.uint8:
        depth, data = 8, image
    else:
        raise TypeError(f"unsupported dtype {image.dtype}")
    rows = np.ascontiguousarray(data).reshape(height, -1).view(np.uint8)
    raw = np.hstack([np.zeros((height, 1), dtype=np.uint8), rows]).tobytes()
    header = struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0)
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )


def read_image(path: Path | str) -> np.ndarray:
    """Read any image Panda can, as floats in [0, 1], top row first.

    16-bit PNGs keep their full precision.

    :param path: source file
    :returns: (h, w, channels) float32
    """
    from panda3d.core import Filename, PfmFile, PNMImage

    image = PNMImage()
    if not image.read(Filename.from_os_specific(str(path))):
        raise OSError(f"cannot read image {path}")
    pfm = PfmFile()
    pfm.load(image)
    shape = (pfm.get_y_size(), pfm.get_x_size(), pfm.get_num_channels())
    # The buffer can run a few floats past the image; keep the image.
    data = np.frombuffer(memoryview(pfm), dtype=np.float32)[: np.prod(shape)]
    return data.reshape(shape)
