"""
The cloud genera, each alone: the terrain demo (demo_terrain.py) with its
low clouds replaced by the showcase (clouds.genera.showcase_features), one
feature per genus in a row along the northern horizon from the start, each
at its real altitude, no other cloud.

    python scripts/demo_genera.py [any demo_terrain.py option]

It is ``demo_terrain.py --showcase``, starting by default where the genera
were tuned (``--start 29.7159 29.8232``). Keys as the demo's, plus 1-9 to
show or hide each genus of the low clouds (the HUD lists them); C the cirrus.
West to east: stratus, cumulus humilis, mediocris, congestus, cumulonimbus.
For still shots of each, ``terrain_shots.py --showcase --start 29.7159
29.8232 --pose genus_stratus genus_humilis ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import demo_terrain  # noqa: E402

#: Where the genera were tuned (latitude, longitude).
DEFAULT_START = ("29.7159", "29.8232")


def main() -> None:
    argv = sys.argv[1:]
    if "--showcase" not in argv:
        argv.append("--showcase")
    if "--start" not in argv:
        argv += ["--start", *DEFAULT_START]
    sys.argv = [sys.argv[0], *argv]
    demo_terrain.main()


if __name__ == "__main__":
    main()
