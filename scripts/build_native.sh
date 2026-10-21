#!/usr/bin/env bash
# Build the native terrain collision solid (Linux) against the installed
# panda3d wheel. See NATIVE_COLLISION_BUILD.md.
#
#   PANDA_SDK_INCLUDE=<sdk>/usr/include/panda3d EIGEN_INCLUDE=<eigen-3.3.x> \
#       scripts/build_native.sh [python]
#
# python: the interpreter whose panda3d wheel to link against (default: python3).
# Writes src/world_builder/terrain/native/libterrain_collision-panda<major>.<minor>.so
set -euo pipefail

PYTHON=${1:-python3}
: "${PANDA_SDK_INCLUDE:?set PANDA_SDK_INCLUDE to the Panda3D SDK headers (same version as the wheel)}"
: "${EIGEN_INCLUDE:?set EIGEN_INCLUDE to the Eigen 3.3 headers}"

read -r PANDA_LIB WHEEL_VERSION < <("$PYTHON" -c \
    "import panda3d, os; print(os.path.dirname(panda3d.__file__), panda3d.__version__)")
SERIES=$(echo "$WHEEL_VERSION" | cut -d. -f1,2)
HEADERS=$(sed -n 's/^#define PANDA_VERSION_STR "\(.*\)"/\1/p' "$PANDA_SDK_INCLUDE/pandaVersion.h")
if [ "$(echo "$HEADERS" | cut -d. -f1,2)" != "$SERIES" ]; then
    echo "headers are Panda3D $HEADERS, the wheel is $WHEEL_VERSION: use headers of the $SERIES series" >&2
    exit 1
fi
if [ "$HEADERS" != "$WHEEL_VERSION" ]; then
    echo "note: headers $HEADERS, wheel $WHEEL_VERSION (same series)" >&2
fi

HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=$HERE/src/world_builder/terrain/native
OUT=$SRC/libterrain_collision-panda$SERIES.so

g++ -std=c++14 -O3 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 -DNDEBUG \
    -I"$PANDA_SDK_INCLUDE" -I"$EIGEN_INCLUDE" \
    "$SRC/terrain_collision.cpp" -o "$OUT" \
    -L"$PANDA_LIB" -l:libpanda.so."$SERIES" -l:libpandaexpress.so."$SERIES" \
    -l:libp3dtool.so."$SERIES" -l:libp3dtoolconfig.so."$SERIES" \
    -Wl,-rpath,"$PANDA_LIB"
echo "built $OUT"
