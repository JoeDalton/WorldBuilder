#!/usr/bin/env bash
# Download the headers the native terrain collision solid compiles against:
# the Panda3D SDK's (same version as the pinned wheel) and Eigen's. Used by
# scripts/build_native.sh and CI. See NATIVE_COLLISION_BUILD.md.
#
#   scripts/fetch_native_deps.sh [dest]
#
# dest: where to put them (default: .cache/native in the repository). Already
# fetched parts are kept. Prints the two include directories, one per line:
# the Panda3D headers, then Eigen.
set -euo pipefail

PANDA_VERSION=1.10.16
EIGEN_VERSION=3.3.7

HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST=$(mkdir -p "${1:-$HERE/.cache/native}" && cd "${1:-$HERE/.cache/native}" && pwd)
SDK=$DEST/panda3d-sdk-$PANDA_VERSION
EIGEN=$DEST/eigen-$EIGEN_VERSION

# The Ubuntu SDK package: only its headers are used. Unpacked with ar and tar
# so this works off Debian too.
if [ ! -f "$SDK/usr/include/panda3d/pandaVersion.h" ]; then
    rm -rf "$SDK" && mkdir -p "$SDK"
    curl -fsSL -o "$SDK/sdk.deb" \
        "https://www.panda3d.org/download/panda3d-$PANDA_VERSION/panda3d1.10_$PANDA_VERSION~noble_amd64.deb"
    (cd "$SDK" && ar x sdk.deb && tar xf data.tar.* ./usr/include/panda3d)
    rm -f "$SDK"/sdk.deb "$SDK"/*.tar.* "$SDK"/debian-binary
fi >&2

if [ ! -f "$EIGEN/Eigen/Core" ]; then
    rm -rf "$EIGEN"
    curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz" \
        | tar xz -C "$DEST"
fi >&2

echo "$SDK/usr/include/panda3d"
echo "$EIGEN"
