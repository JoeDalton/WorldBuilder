#!/usr/bin/env bash
# Turn the pure world_builder wheel into a manylinux wheel carrying the native
# terrain collision library. Runs inside a pypa manylinux image, whose
# toolchain matches Panda3D's own wheels. See NATIVE_COLLISION_BUILD.md.
#
#   docker run --rm -v "$PWD":/io quay.io/pypa/manylinux2014_x86_64 \
#       /io/scripts/build_manylinux_wheel.sh /io/dist/world_builder-<v>-py3-none-any.whl /io/wheelhouse
#
# The headers come from .cache/native (scripts/fetch_native_deps.sh, run on the
# host beforehand, or fetched here). The library is Python-independent (ctypes,
# no Python.h), so the wheel is tagged py3-none-<platform>.
set -euo pipefail

PURE=$1
OUT=$2
# Any CPython of the image: only its panda3d wheel's libraries are linked.
PYTHON=${PYTHON:-/opt/python/cp312-cp312/bin/python}
PLAT=${AUDITWHEEL_PLAT:?run in a manylinux image, or set AUDITWHEEL_PLAT}
PANDA_VERSION=1.10.16  # the pin in pyproject.toml

HERE=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

"$PYTHON" -m pip install -q "panda3d==$PANDA_VERSION" wheel
"$HERE/scripts/build_native.sh" "$PYTHON"
LIB=$(echo "$HERE"/src/world_builder/terrain/native/libterrain_collision-panda*.so)

"$PYTHON" -m wheel unpack "$PURE" -d "$WORK"
UNPACKED=$(echo "$WORK"/world_builder-*)
cp "$LIB" "$UNPACKED/world_builder/terrain/native/"
sed -i -e 's/^Root-Is-Purelib: true$/Root-Is-Purelib: false/' \
    -e "s/^Tag: .*/Tag: py3-none-linux_$(uname -m)/" "$UNPACKED"/*.dist-info/WHEEL
mkdir "$WORK/linux"
"$PYTHON" -m wheel pack "$UNPACKED" -d "$WORK/linux"

# Checks the library's glibc and libstdc++ symbol versions against the policy
# and applies its tag. Panda's libraries come with the panda3d wheel: not
# copied in.
auditwheel repair --plat "$PLAT" -w "$OUT" \
    --exclude "libpanda.so.*" --exclude "libpandaexpress.so.*" \
    --exclude "libp3dtool.so.*" --exclude "libp3dtoolconfig.so.*" \
    "$WORK"/linux/*.whl
