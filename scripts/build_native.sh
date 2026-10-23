#!/usr/bin/env bash
# Build the native terrain collision solid (Linux, macOS) against the installed
# panda3d wheel. See NATIVE_COLLISION_BUILD.md. Windows: build_native.ps1.
#
#   [PANDA_SDK_INCLUDE=<sdk>/usr/include/panda3d] [EIGEN_INCLUDE=<eigen-3.3.x>] \
#       scripts/build_native.sh [python]
#
# python: the interpreter whose panda3d wheel to link against (default: python3).
# Headers not given are fetched by scripts/fetch_native_deps.sh into .cache/native.
# Writes src/world_builder/terrain/native/libterrain_collision-panda<major>.<minor>.so
# (.dylib on macOS, for the wheel's architectures: universal2 from Python 3.12).
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

if [ "$(uname -s)" = Darwin ]; then
    OUT=$SRC/libterrain_collision-panda$SERIES.dylib
    # The first rpath finds the wheel's libraries from an installed
    # world_builder (as on Linux), the second from the source tree.
    clang++ -std=c++14 -O3 -dynamiclib -DNDEBUG \
        $(printf -- '-arch %s ' $(lipo -archs "$PANDA_LIB/libpanda.$SERIES.dylib")) \
        -mmacosx-version-min="${MACOSX_DEPLOYMENT_TARGET:-11.0}" \
        -I"$PANDA_SDK_INCLUDE" -I"$EIGEN_INCLUDE" \
        "$SRC/terrain_collision.cpp" -o "$OUT" -install_name "@rpath/$(basename "$OUT")" \
        -L"$PANDA_LIB" -lpanda."$SERIES" -lpandaexpress."$SERIES" \
        -lp3dtool."$SERIES" -lp3dtoolconfig."$SERIES" \
        -Wl,-rpath,@loader_path/../../../panda3d -Wl,-rpath,"$PANDA_LIB"
    # The wheel's libraries are named @loader_path/../lib/<name>, which would
    # resolve next to this library: refer to them through the rpaths instead.
    for NAME in panda pandaexpress p3dtool p3dtoolconfig; do
        NAME=lib$NAME.$SERIES.dylib
        install_name_tool -change "$(otool -D "$PANDA_LIB/$NAME" | tail -1)" "@rpath/$NAME" "$OUT"
    done
    codesign --force --sign - "$OUT"  # arm64 runs signed code only
    echo "built $OUT"
    exit
fi

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
