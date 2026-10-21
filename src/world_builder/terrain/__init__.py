"""
Procedural planetary terrain: a standalone prototype, not yet used by the game.

Run the 3D demo with ``python scripts/demo_terrain.py``; the full guide is
``docs/source/terrain.md``. Only :mod:`mesh` and :mod:`terrain` import Panda
at import time: the rest (the planet definition, chunk generation, LOD
selection, queries, workers, levels) also runs in worker processes and tests,
and reads images through Panda's PNMImage only when it loads one.
"""
