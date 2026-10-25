#!/usr/bin/env bash
# Build the native libraries (Linux, macOS): the noise library, then the
# terrain collision solid against the installed panda3d wheel. See
# docs/source/native_build.md. Windows: build_native.ps1.
#
#   [PANDA_SDK_INCLUDE=<Panda3D SDK headers>] [EIGEN_INCLUDE=<Eigen 3.3 headers>] \
#       scripts/build_native.sh [python]
#
# python: the interpreter whose panda3d wheel to link against (default: python3).
# Headers not given are fetched by scripts/fetch_native_deps.sh into .cache/native.
# Writes src/world_builder/terrain/native/libwb_noise.so and
# libterrain_collision-panda<major>.<minor>.so (.dylib on macOS, for the
# wheel's architectures: universal2 from Python 3.12).
set -euo pipefail

PYTHON=${1:-python3}
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=$HERE/src/world_builder/terrain/native

# The noise library: plain C, no headers but the system's. No floating-point
# contraction (FMA), so every platform computes the same terrain.
NOISE_FLAGS="-std=c99 -O3 -ffp-contract=off -fvisibility=hidden"
if [ "$(uname -s)" = Darwin ]; then
    PANDA_DIR=$("$PYTHON" -c "import panda3d, os; print(os.path.dirname(panda3d.__file__))")
    ARCHS=$(lipo -archs "$(ls "$PANDA_DIR"/libpanda.*.dylib | head -1)")
    clang $NOISE_FLAGS -dynamiclib $(printf -- '-arch %s ' $ARCHS) \
        -mmacosx-version-min="${MACOSX_DEPLOYMENT_TARGET:-11.0}" \
        "$SRC/noise.c" -o "$SRC/libwb_noise.dylib" -install_name @rpath/libwb_noise.dylib
    codesign --force --sign - "$SRC/libwb_noise.dylib"
    echo "built $SRC/libwb_noise.dylib"
else
    gcc $NOISE_FLAGS -fPIC -shared "$SRC/noise.c" -o "$SRC/libwb_noise.so" -lm
    echo "built $SRC/libwb_noise.so"
fi

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
