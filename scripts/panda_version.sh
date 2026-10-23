#!/usr/bin/env bash
# Print the Panda3D version world_builder pins in pyproject.toml, the one the
# native terrain collision library is built against. See docs/source/native_build.md.
set -euo pipefail

PYPROJECT=$(dirname "$0")/../pyproject.toml
VERSION=$(sed -n 's/^Panda3D = "==\(.*\)"/\1/p' "$PYPROJECT")
[ -n "$VERSION" ] || { echo "pyproject.toml does not pin Panda3D with ==" >&2; exit 1; }
echo "$VERSION"
