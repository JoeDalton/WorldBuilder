#!/usr/bin/env bash
# Turn the pure world_builder wheel into a manylinux wheel carrying the native
# libraries (noise, terrain collision). Runs inside a pypa manylinux image, whose
# toolchain matches Panda3D's own wheels. See docs/source/native_build.md.
#
#   docker run --rm -v "$PWD":/io quay.io/pypa/manylinux2014_<arch> \
#       /io/scripts/build_manylinux_wheel.sh /io/dist/world_builder-<v>-py3-none-any.whl /io/wheelhouse
#
# arch: x86_64 or aarch64, the machine's (the wheel is tagged with it).
#
# The headers come from .cache/native (scripts/fetch_native_deps.sh, run on the
# host beforehand, or fetched here). The libraries are Python-independent (ctypes,
# no Python.h), so the wheel is tagged py3-none-<platform>.
set -euo pipefail

PURE=$1
OUT=$2
# Any CPython of the image: only its panda3d wheel's libraries are linked.
PYTHON=${PYTHON:-/opt/python/cp312-cp312/bin/python}
PLAT=${AUDITWHEEL_PLAT:?run in a manylinux image, or set AUDITWHEEL_PLAT}

HERE=$(cd "$(dirname "$0")/.." && pwd)
PANDA_VERSION=$("$HERE/scripts/panda_version.sh")
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

"$PYTHON" -m pip install -q "panda3d==$PANDA_VERSION" wheel
"$HERE/scripts/build_native.sh" "$PYTHON"
NATIVE=$HERE/src/world_builder/terrain/native
"$PYTHON" "$HERE/scripts/platform_wheel.py" "$PURE" "linux_$(uname -m)" "$WORK" \
    "$NATIVE/libwb_noise.so" "$NATIVE"/libterrain_collision-panda*.so

# Checks the libraries' glibc and libstdc++ symbol versions against the policy
# and applies its tag. Panda's libraries come with the panda3d wheel: not
# copied in.
auditwheel repair --plat "$PLAT" -w "$OUT" \
    --exclude "libpanda.so.*" --exclude "libpandaexpress.so.*" \
    --exclude "libp3dtool.so.*" --exclude "libp3dtoolconfig.so.*" \
    "$WORK"/*.whl
