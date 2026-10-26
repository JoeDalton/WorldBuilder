"""
Add the native libraries (noise, terrain collision) to the pure world_builder
wheel and tag it for one platform. Used by scripts/build_manylinux_wheel.sh,
scripts/build_macos_wheel.sh and CI; see docs/source/native_build.md.

    python scripts/platform_wheel.py <pure wheel> <platform tag> <out dir> <library>...

The libraries don't use Python.h, so the wheel is tagged py3-none-<platform>.
Needs the ``wheel`` package.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main(pure: str, platform: str, out: str, *libraries: str) -> None:
    with tempfile.TemporaryDirectory() as work:
        wheel = [sys.executable, "-m", "wheel"]
        subprocess.run([*wheel, "unpack", pure, "-d", work], check=True)
        (unpacked,) = Path(work).glob("world_builder-*")
        for library in libraries:
            shutil.copy(library, unpacked / "world_builder" / "terrain" / "native")
        (metadata,) = unpacked.glob("*.dist-info/WHEEL")
        text = metadata.read_text()
        text = text.replace("Root-Is-Purelib: true", "Root-Is-Purelib: false")
        text = re.sub(r"^Tag: .*$", f"Tag: py3-none-{platform}", text, flags=re.M)
        metadata.write_text(text, newline="\n")
        Path(out).mkdir(parents=True, exist_ok=True)
        subprocess.run([*wheel, "pack", str(unpacked), "-d", out], check=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
