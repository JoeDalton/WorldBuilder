#!/usr/bin/env bash
# Build the native terrain collision solid (Linux) against the installed
# panda3d wheel. See NATIVE_COLLISION_BUILD.md.
#
#   [PANDA_SDK_INCLUDE=<sdk>/usr/include/panda3d] [EIGEN_INCLUDE=<eigen-3.3.x>] \
#       scripts/build_native.sh [python]
#
# python: the interpreter whose panda3d wheel to link against (default: python3).
# Headers not given are fetched by scripts/fetch_native_deps.sh into .cache/native.
# Writes src/world_builder/terrain/native/libterrain_collision-panda<major>.<minor>.so
set -euo pipefail

PYTHON=${1:-python3}
HERE=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${PANDA_SDK_INCLUDE:-}" ] || [ -z "${EIGEN_INCLUDE:-}" ]; then
    { read -r FETCHED_PANDA; read -r FETCHED_EIGEN; } < <("$HERE/scripts/fetch_native_deps.sh")
    PANDA_SDK_INCLUDE=${PANDA_SDK_INCLUDE:-$FETCHED_PANDA}
    EIGEN_INCLUDE=${EIGEN_INCLUDE:-$FETCHED_EIGEN}
fi

read -r PANDA_LIB WHEEL_VERSION < <("$PYTHON" -c \
    "import panda3d, os; print(os.path.dirname(panda3d.__file__), panda3d.__version__)")
SERIES=$(echo "$WHEEL_VERSION" | cut -d. -f1,2)
HEADERS=$(sed -n 's/^#define PANDA_VERSION_STR "\(.*\)"/\1/p' "$PANDA_SDK_INCLUDE/pandaVersion.h")
if [ "$HEADERS" != "$WHEEL_VERSION" ]; then
    echo "headers are Panda3D $HEADERS, the wheel is $WHEEL_VERSION: use headers of the same version" >&2
    exit 1
fi

SRC=$HERE/src/world_builder/terrain/native
OUT=$SRC/libterrain_collision-panda$SERIES.so

# The rpath finds the wheel's libraries from an installed world_builder
# (site-packages/world_builder/terrain/native -> site-packages/panda3d). From
# the source tree, libpanda is found already loaded: the binding imports
# panda3d.core first.
g++ -std=c++14 -O3 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 -DNDEBUG \
    -I"$PANDA_SDK_INCLUDE" -I"$EIGEN_INCLUDE" \
    "$SRC/terrain_collision.cpp" -o "$OUT" \
    -L"$PANDA_LIB" -l:libpanda.so."$SERIES" -l:libpandaexpress.so."$SERIES" \
    -l:libp3dtool.so."$SERIES" -l:libp3dtoolconfig.so."$SERIES" \
    -Wl,-rpath,'$ORIGIN/../../../panda3d'
echo "built $OUT"
