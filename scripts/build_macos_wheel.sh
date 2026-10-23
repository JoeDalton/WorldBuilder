#!/usr/bin/env bash
# Turn the pure world_builder wheel into a macOS wheel carrying the native
# terrain collision library: universal2 (x86_64 and arm64), like Panda3D's own
# wheels from Python 3.12, whose libraries it links against. See
# NATIVE_COLLISION_BUILD.md.
#
#   scripts/build_macos_wheel.sh <pure wheel> <out dir>
#
# The library is Python-independent (ctypes, no Python.h), so the wheel is
# tagged py3-none-macosx_11_0_<architectures>.
set -euo pipefail

PURE=$1
OUT=$2
PYTHON=${PYTHON:-python3}
PANDA_VERSION=1.10.16  # the pin in pyproject.toml
export MACOSX_DEPLOYMENT_TARGET=11.0  # that of Panda's universal2 wheels

HERE=$(cd "$(dirname "$0")/.." && pwd)

"$PYTHON" -m pip install -q "panda3d==$PANDA_VERSION" wheel
"$HERE/scripts/build_native.sh" "$PYTHON"
LIB=$(echo "$HERE"/src/world_builder/terrain/native/libterrain_collision-panda*.dylib)

# Drop the rpath to this machine's panda3d, there for builds in the source tree.
PANDA_LIB=$("$PYTHON" -c "import panda3d, os; print(os.path.dirname(panda3d.__file__))")
install_name_tool -delete_rpath "$PANDA_LIB" "$LIB"
codesign --force --sign - "$LIB"

ARCHS=$(lipo -archs "$LIB")
case "$ARCHS" in
    "x86_64 arm64" | "arm64 x86_64") ARCHS=universal2 ;;
esac
"$PYTHON" "$HERE/scripts/platform_wheel.py" "$PURE" "$LIB" "macosx_11_0_$ARCHS" "$OUT"
