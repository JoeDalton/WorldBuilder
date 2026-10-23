# Building a C++ collision solid against the Panda3D pip wheel

**Status (issue #16):** implemented, and shipped in the Linux x86_64 wheel (`py3-none-manylinux2014_x86_64`), built against Panda3D 1.10.16, the version `pyproject.toml` pins.

| Piece | Where |
|---|---|
| Solid | `src/world_builder/terrain/native/terrain_collision.cpp` |
| Binding | `ctypes`, all in `src/world_builder/terrain/native/_binding.py` |
| Terrain integration | `Terrain.enable_collision`, `src/world_builder/terrain/collision.py` |
| Headers | `scripts/fetch_native_deps.sh` |
| Build | `scripts/build_native.sh` (`invoke native`) |
| Wheel | `scripts/build_manylinux_wheel.sh`, CI jobs `build_wheels` and `test_wheels` |
| Tests | `tests/test_terrain_collision.py`, skipped when the library isn't built, failed instead when `WORLD_BUILDER_REQUIRE_NATIVE` is set |

Linux aarch64, Windows and macOS are listed below with what is still unverified.

## Building from a source checkout (Linux)

```sh
invoke native      # also run by `invoke develop` on Linux x86_64
```

It runs `scripts/build_native.sh` with the Poetry venv's Python, so against its panda3d wheel. The headers (the Panda3D SDK's and Eigen's) are fetched by `scripts/fetch_native_deps.sh` into `.cache/native/` on first use. Other headers can be given with `PANDA_SDK_INCLUDE` and `EIGEN_INCLUDE`, and another Python as the script's argument.

This writes `src/world_builder/terrain/native/libterrain_collision-panda<major>.<minor>.so`. The name carries the Panda3D series. It is git-ignored (`*.so`).

The script refuses headers of another version than the wheel's.

## The Linux x86_64 wheel

The library includes no `Python.h` (see the binding below), so it doesn't depend on the Python version: one `py3-none-<platform>` wheel serves 3.12 to 3.14. It is built once per platform, and Python versions are only a test matrix.

`scripts/build_manylinux_wheel.sh` runs in the `quay.io/pypa/manylinux2014_x86_64` image, whose GCC 10 and glibc 2.17 are the toolchain of Panda3D's own manylinux2014 wheels. It:

1. installs `panda3d==1.10.16` and builds the library with `scripts/build_native.sh`;
2. adds it to the pure wheel and retags it `py3-none-linux_x86_64` (`Root-Is-Purelib: false`);
3. runs `auditwheel repair`, which checks the glibc and libstdc++ symbol versions against the manylinux2014 policy and applies its tag. Panda's libraries are excluded: they come with the panda3d wheel.

```sh
python -m build
scripts/fetch_native_deps.sh
docker run --rm -v "$PWD":/io quay.io/pypa/manylinux2014_x86_64 \
    /io/scripts/build_manylinux_wheel.sh "/io/$(ls dist/*-py3-none-any.whl)" /io/wheelhouse
```

In CI, `build_wheels` does this on every push and pull request, and `test_wheels` installs the wheel in a clean venv for each of Python 3.12, 3.13 and 3.14 and runs the whole suite against it with `WORLD_BUILDER_REQUIRE_NATIVE=1`. On a tag, the release carries this wheel and the sdist (which has no library).

The library finds Panda's libraries through a relative rpath, `$ORIGIN/../../../panda3d` (from `site-packages/world_builder/terrain/native/`). Built in the source tree, that path leads nowhere, but `libpanda` is already loaded by then: the binding imports `panda3d.core` first, and the loader matches it by soname.

At load, `_binding.py` checks two things:

- that a library exists for the running Panda3D series;
- that the library reports the same series it was built against.

Otherwise it raises `NativeCollisionError` with the rebuild instructions.

## Why a compiled solid

A `CollisionSolid` subclass written in C++ plugs into Panda's own `CollisionTraverser`. The traverser calls the new solid's `test_intersection_from_sphere`, `test_intersection_from_segment` and so on through C++ virtual dispatch, the same way it uses the built-in solids.

The game's handlers (`CollisionHandlerEvent`, `CollisionHandlerQueue`) then get genuine `CollisionEntry` objects: surface point, normal, into-node name and python tags. Their code does not change.

**Pure-Python options can't do this:**
- Python can't subclass a solid the traverser will call.
- `CollisionEntry` has no Python constructor.
- Handlers have no Python `add_entry`.

## What the Linux spike verified

Environment: Ubuntu 24.04, g++ 13.3, Python 3.12, `panda3d==1.10.16` from pip.

**Works:**
- the queue handler;
- `into` / `again` events;
- surface point and normal under a node transform;
- `python_tags["owner"]` on the into-node;
- `NodePath.copy_to` (the solid's `make_copy`);
- node removal;
- printing the solid from Python, where it appears as a `CollisionSolid`.

**Traverser overhead per collider:**
- One terrain node holding one solid adds about 0.4 µs per collider, measured as 0.45 ms against 0.29 ms for 400 colliders. This needs the colliders to sit under their own parent node, as SpaceFlight's do under `root_node`.
- If the colliders are direct siblings of the terrain node, the cost rises to about 5 µs per collider (2.3 ms).
- 640 separate solid nodes in a flat list cost 10.7 ms for 400 colliders, even with a trivial test. So the terrain should be one solid, or a shallow hierarchy, not one node per chunk.
- **Don't put the solid's node under `Terrain.root`.** Its collide mask makes the traverser enter that node and test the bounds of all of its chunk GeomNodes, for every batch of 32 colliders. In the benchmark that cost 2.5 ms for 300 lasers; the test work itself was about 0.3 ms. Put the node under a sibling of the root that copies the root's transform each frame.

## Performance spike (`scripts/collision_bench.py --native`)

`src/world_builder/terrain/native/terrain_collision.cpp` is one `CollisionTerrain` solid for the whole planet.

- **What it holds:** the drawn chunks' grid vertices, registered from Python at 0.05 ms per chunk, plus per-tile height bounds.
- **Lookup:** it finds the drawn chunk under a direction with the same cube-sphere maths as `cubesphere.py`.
- **Surface:** it intersects radial rays with the drawn triangles, like `query.py`. Exactly one drawn chunk answers each direction, so collision has no holes at level-of-detail seams, even without skirts.
- **Segments:** a bound check, then a march at half the drawn grid spacing, then bisection.
- **Spheres:** a bound check, then 5×5 samples over the footprint, then a slide onto the nearest triangle's plane.

Results, earthlike preset, camera 300 m over mountains, 636 drawn chunks, 64 ships + 33 active sensors + 300 lasers within 5 km:

| Load | Terrain cost on top of the traverse | Ships | Sensors | Lasers |
|---|---|---|---|---|
| Half the ships within 50 m of the ground, lasers 0–500 m up | 0.64 ms | 0.13 | 0.07 | 0.31 |
| `--stress`: every ship within 50 m, every laser within 60 m | 1.1–1.2 ms | 0.22 | 0.13 | 0.52–0.65 |

**Accuracy:**
- **Lasers:** hit points match `TerrainQuery.segment_intersect` within 1 mm, normals within 0.01°.
- **Ships:** all 64 agree with `sphere_contact` on whether they hit. Contact points differ by up to 0.3 m on normal ground and 2.4 m on canyon walls (normals up to 10°), because the two methods sample differently.

**Events:** each collider gets one event per frame.

For comparison, Panda's collision against visible geometry (the Geom tiles of option A) cost at least 54 ms on the normal load.

### Requirements

1. **Panda3D C++ headers matching the wheel's version.** The pip wheel ships none. They come from the SDK package `panda3d1.10_1.10.16~noble_amd64.deb` (panda3d.org downloads), using `usr/include/panda3d` (`scripts/fetch_native_deps.sh`). Only the headers are used; linking is against the wheel's own libraries.
2. **Eigen 3.3.x headers.** The SDK's `dtool_config.h` sets `HAVE_EIGEN` and `LINMATH_ALIGN`, so Panda's math headers include Eigen. Eigen 3.3.7 is used. Eigen is header-only.
3. **The wheel's C++ ABI.** The manylinux wheel is built with GCC 10 on CentOS 7 (devtoolset), so it uses the old libstdc++ string ABI. Compile with `-D_GLIBCXX_USE_CXX11_ABI=0`, or any `std::string` crossing into libpanda (for example `register_type`) will mismatch.
4. **Link against the wheel's libraries** by exact soname, with an rpath to the wheel directory:

   ```sh
   P=<venv>/lib/python3.12/site-packages/panda3d
   g++ -std=c++14 -O2 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 -DNDEBUG \
       -I<sdk>/usr/include/panda3d -I<eigen-3.3.7> solid.cpp -o libsolid.so \
       -L$P -l:libpanda.so.1.10 -l:libpandaexpress.so.1.10 \
       -l:libp3dtool.so.1.10 -l:libp3dtoolconfig.so.1.10 -Wl,-rpath,'$ORIGIN/../../../panda3d'
   ```

   The rpath is relative, for an installed wheel; see the Linux x86_64 wheel above.
5. **Python binding.** The binding uses `ctypes` with a C API (`extern "C"`), so it needs no `Python.h`; `python3.12-dev` isn't installed on this machine. All of it is in `native/_binding.py`, behind the `NativeTerrain` class, so a CPython extension can replace it without touching the rest.
   - To hand a `CollisionNode` to C++, it reads the wrapper's C++ pointer from Panda's `Dtool_PyInstDef` (`py_panda.h`): `_ptr_to_object` is at offset 24, and `_signature == 0xBEAF` is at offset 32 on 64-bit CPython.
   - That layout is Panda-internal. A proper CPython extension built with `py_panda.h` (`DtoolInstance_VOID_PTR`, type checks) is the cleaner long-term binding, and it needs the Python dev headers. `interrogate` (in the SDK) is the full-integration option if the solid's own methods must be callable from Python.
6. **Type registration.** Call `init_type()` once after loading, which registers the class with Panda's `TypeRegistry`.

## Other platforms (not yet verified)

### Linux (aarch64)

- **Image:** `quay.io/pypa/manylinux2014_aarch64` on an `ubuntu-24.04-arm` runner; `scripts/build_manylinux_wheel.sh` already tags with the machine it runs on.
- **Headers:** `fetch_native_deps.sh` takes the amd64 SDK package. Check whether panda3d.org has an arm64 one, or that the amd64 `dtool_config.h` matches the aarch64 wheel's build.

### Windows (x64)

- **Headers and import libraries:** the Panda3D SDK installer for the same version (`Panda3D-SDK-1.10.16-x64.exe`) has `include/` and `lib/*.lib`. Link against those import libraries, while the game loads the wheel's DLLs at runtime. Check that the SDK and wheel DLL names and exports match.
- **Compiler:** MSVC 2017–2022. These share the v14x ABI, and Python 3.12 needs at least that. There's no `_GLIBCXX_USE_CXX11_ABI` issue.
- **Exports:** `extern "C" __declspec(dllexport)` for the C API.
- **Unverified:**
  - whether the wheel is built with `HAVE_EIGEN` and `LINMATH_ALIGN`, like the SDK headers;
  - `/MD` runtime matching;
  - how to find the wheel's DLL directory at load time (`os.add_dll_directory`).

### macOS (x86_64 and arm64)

- **Headers:** from the Panda3D SDK `.dmg` for the same version, or from a source build of the same tag.
- **Compiler:** clang with libc++, the default, which matches the wheels.
- **Architectures:** build both x86_64 and arm64 (or universal2), matching the wheels a user installs.
- **Linking:** link against `libpanda.1.10.dylib` and the others in the wheel, using `@loader_path`/`@rpath` install names.
- **Unverified:**
  - the wheels' install names;
  - `HAVE_EIGEN`/`LINMATH_ALIGN` settings;
  - whether the minimum macOS deployment target must match.

### For every platform

- **One build per Panda series.** Every 1.10.x wheel ships `libpanda.so.1.10`, and that shared name signals intended binary compatibility within the series. So a build against 1.10.16 headers should load on any 1.10.x. This is unverified, and for now moot: `pyproject.toml` pins Panda3D to 1.10.16. Test it by loading the library under another 1.10.x wheel before loosening the pin.
  - A new series (1.11) has different library names and needs a rebuild against its headers.
  - The loader should check `panda3d.__version__` on major.minor only, and raise a clear "rebuild" message instead of the bare `OSError` from the missing `libpanda.so.1.10`.
  - The class layouts it depends on (`CollisionSolid`, `CollisionEntry`) are Panda's internal C++ API, so the build should be retested on each new Panda release.
- **CI:** add the platform to the `build_wheels` and `test_wheels` matrices, with its own way of fetching the matching SDK headers (`dtool_config.h` differs per platform) and of building the library. Running the collision tests against the installed wheel is what checks that the headers match how Panda's wheel was built there: a mismatch in `HAVE_EIGEN` or `LINMATH_ALIGN` changes class layouts and corrupts memory instead of failing to link.
- **Fallback:** without the compiled library, the terrain should still load, either with collision disabled and a logged warning, or with the slower numpy `TerrainQuery` path.
