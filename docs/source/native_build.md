# The native libraries

Two small compiled libraries sit in `src/world_builder/terrain/native/`:

- **Terrain collision** (`Terrain.enable_collision`, see [terrain](terrain.md)): `terrain_collision.cpp`, built against Panda3D. Most of this page is about it: why it is compiled, how it is built and shipped for each platform, and what it depends on.
- **Noise** (`noise.c`, see [below](#the-noise-library)): plain C, no dependency. Terrain generation runs without it, about 3x slower.

## Why a compiled solid

A `CollisionSolid` subclass written in C++ plugs into Panda's own `CollisionTraverser`. The traverser calls the solid's `test_intersection_from_sphere`, `test_intersection_from_segment` and so on through C++ virtual dispatch, as it does for the built-in solids. A game's handlers (`CollisionHandlerEvent`, `CollisionHandlerQueue`) then get genuine `CollisionEntry` objects: surface point, normal, into-node name and Python tags. Their code does not change.

Pure Python can't do this: it can't subclass a solid the traverser will call, `CollisionEntry` has no Python constructor, and handlers have no Python `add_entry`.

One `CollisionTerrain` solid stands for the whole planet's drawn chunks. The chunks' grid vertices are registered from Python, and the solid intersects the very triangles on screen (same vertices, same diagonal as `query.py`). Its cost, accuracy and where its node goes in the scene graph are in [terrain](terrain.md).

## What it depends on

- **Panda3D 1.10.16, exactly.** `pyproject.toml` pins it. The library uses Panda's internal C++ classes (`CollisionSolid`, `CollisionEntry`), so it is built against the headers of the very version it runs with, and retested on every Panda release. Every 1.10.x wheel ships `libpanda.so.1.10`, which suggests binary compatibility within the series, but that is unverified: load the library under another 1.10.x wheel before loosening the pin. A new series (1.11) needs a rebuild.
- **Panda's headers and Eigen.** The pip wheel ships no headers. They come from the Panda3D SDK of the same version (the Ubuntu package on Linux, the installer on Windows, the disk image on macOS), plus Eigen 3.3.7 on Linux; the Windows and macOS SDKs bundle Eigen 3.3.9. The SDK's `dtool_config.h` sets `HAVE_EIGEN` and `LINMATH_ALIGN`, which change class layouts: if they didn't match how Panda's wheel was built, the result would be memory corruption, not a link error. The collision tests run against every platform's installed wheel for this reason. `scripts/fetch_native_deps.sh` (`.ps1` on Windows) downloads and caches them in `.cache/native/`.
- **The wheel's C++ runtime.** Panda's manylinux wheels are built with GCC 10 and the old libstdc++ string ABI, so Linux builds use `-D_GLIBCXX_USE_CXX11_ABI=0`; any `std::string` crossing into libpanda (for example `register_type`) would mismatch. Windows uses MSVC with `/MD`, macOS clang with libc++, like Panda's wheels.
- **No `Python.h`.** The binding is `ctypes` over a C API (`extern "C"`, `WB_API` exports on Windows), all in `native/_binding.py` behind the `NativeTerrain` class. So the library doesn't depend on the Python version, and one wheel per platform serves 3.12 to 3.14. To hand a `CollisionNode` to C++, the binding reads the wrapper's C++ pointer from Panda's `Dtool_PyInstDef` (`py_panda.h`): the pointer is at offset 24 and a `0xBEAF` signature at offset 32, on 64-bit CPython with the GIL. That layout is Panda-internal. A CPython extension built on `py_panda.h` would be the cleaner binding, at the price of one wheel per Python version.

## Building from a source checkout

```sh
invoke native      # also run by `invoke develop`
```

It builds the libraries for the Poetry venv's panda3d: `scripts/build_native.sh` on Linux and macOS, `scripts/build_native.ps1` on Windows (MSVC, found with `vswhere`). The output sits next to `_binding.py` as `libwb_noise.so` and `libterrain_collision-panda<major>.<minor>.so` (`.dylib` on macOS, `.dll` on Windows) and is git-ignored. `invoke develop` only warns if the build fails.

Other headers can be given to `build_native.sh` with `PANDA_SDK_INCLUDE` and `EIGEN_INCLUDE`, and another Python as its argument. The scripts refuse headers of another version than the installed wheel's.

Without the library, `Terrain.enable_collision` raises `NativeCollisionError` and the tests in `tests/test_terrain_collision.py` are skipped. Setting `WORLD_BUILDER_REQUIRE_NATIVE` makes them fail instead (CI does).

The same library holds `CollisionAsteroids` (`asteroid_collision.cpp`), an
asteroid field's rocks for the traverser (`world_builder.asteroids.collision`,
`NativeAsteroids` in the binding): built and shipped with the terrain's solid,
no other library or wheel file.

## The noise library

`world_builder.terrain.noise` sums the octaves of `fbm` and `ridged` in `libwb_noise.so` (`.dylib`, `.dll`) when it is there: about 5x faster than numpy, which pays ~150 array operations per octave. A terrain chunk takes ~8 ms instead of ~24, and the weather map, the Milky Way glow and the sea level follow.

It also computes, natively when it can (`gravity.c`: the summed gravity of the world's bodies at many points, `world_builder.gravity`, 4-9x faster than numpy):

- **The clouds' density** (`wb_volume_fbm`): the octaves of the tiling 64³ noise volume, sampled as `clouds/noise.py` and the GPU sample it. Cloud tiles, the column peaks and the deck's light table use it.
- **A chunk's geometry** (`chunk.c`): the steps of `chunkgen.generate_chunk` that only shape the grid, none of the planet's rules: the face's directions on a separable grid (`wb_face_grid`, one tangent per row and column), the normals and the slope (`wb_chunk_surface`), the vertex buffer and the box of what is drawn (`wb_chunk_pack`). Each follows its numpy expression's order of operations; `tests/test_native_noise.py` compares whole chunks, a coastal one included (identical here).
- **The planet clouds' billboards** (`billboards.c`): a tile's vertices, bounds and reach at upload, written straight into Panda's buffer (`wb_billboard_vertices`), and its billboards back to front from the camera at each re-sort, radix-sorted and written as its triangle indices (`wb_billboard_order`, stable over the tile's previous order like the numpy fallback). Both give numpy's bytes (`tests/test_native_noise.py`).
- **The LOD selection** (`lod.c`, `wb_lod_*`): `quadtree.LodSelector`'s pass, operation for operation, in its own source file of the same library. It keeps a copy of the chunks' state (bounds, ready), which `Terrain` updates as chunks are stored, uploaded and evicted, so a pass makes no call back into Python. `tests/test_native_lod.py` runs both selectors through the same flights and requires the same selections.
- **Asteroid fields' rocks** (`scatter.c`, `wb_scatter_cells`): `world_builder.asteroids.scatter`'s steps for the cells asked for: the candidates of each cell, their existence (the field's density, through the same per-point fbm as `wb_fbm`, shared with `noise.c` by the internal header `noise.h`), the hard core and the survivors' attributes. Every choice uses numpy's arithmetic in numpy's order on the same values, so both give the same rocks; about 100x faster. `tests/test_asteroid_scatter.py` compares them.
- **The sea's swell** (`wb_swell_heights`): the height of the drawn water for `world_builder.ocean.swell.Swell`, for collision queries. The swell itself is in `swell.h`, a header of static inline C99 that the collision solid includes too, so both answer with the same function. The GLSL copy (`ocean/shaders/swell.glsl`) is kept in step by `tests/test_native_noise.py` (C against numpy) and `tests/test_scene.py` (the GPU's against the CPU's).

- **Same noise.** It is the numpy code's simplex noise point by point: one octave gives the same bits. Sums of octaves differ from numpy's in the last bits (~1e-12), because numpy rotates the points with BLAS. Within one run every chunk comes from the same code, so neighbouring chunks' edges still match exactly. `tests/test_native_noise.py` compares the two.
- **Same on every platform.** It is built without floating-point contraction (`-ffp-contract=off`, MSVC's `/fp:precise`), so no compiler fuses a multiply and an add into an FMA and rounds differently on another machine.
- **No Panda3D, no Python.h.** The binding is ctypes, in `noise.py` itself: the library is found by path (importing the `native` package would import the terrain back), and its name carries no Panda3D series. `wb_noise_abi()` guards against a stale build: bump `WB_NOISE_ABI` in `noise.c` and `_NATIVE_ABI` in `noise.py` together whenever an exported function changes (11 since the scatter takes a ring's annulus).
- **Without it** a warning is logged and numpy runs. `WORLD_BUILDER_NO_NATIVE_NOISE` forces numpy (to compare); `WORLD_BUILDER_REQUIRE_NATIVE` makes a missing library an error, as for collision.

`scripts/build_native.sh` and `build_native.ps1` build it first, with the system's compiler only; `invoke native` runs them. The platform wheels carry it next to the collision library.

## Platform wheels

Releases carry one wheel per platform, plus the sdist, which has no library:

| Platform | Wheel tag | Built on |
|---|---|---|
| Linux x86_64 | `py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64` | `quay.io/pypa/manylinux2014_x86_64` |
| Linux aarch64 | `py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64` | `quay.io/pypa/manylinux2014_aarch64`, on a native ARM runner |
| Windows x64 | `py3-none-win_amd64` | `windows-latest`, MSVC |
| macOS, Intel and Apple silicon | `py3-none-macosx_11_0_universal2` | `macos-latest`, clang |

Windows on ARM has no Panda3D wheel, so it isn't covered (x64 Python under emulation takes the x64 wheel).

Every wheel starts from the same pure wheel (`python -m build`, built once on Linux, so the Python files are identical everywhere). `scripts/platform_wheel.py` adds the libraries and retags it `py3-none-<platform>` with `Root-Is-Purelib: false`. Each platform then differs in how the library is built and found:

- **Linux** (`scripts/build_manylinux_wheel.sh`): runs inside the manylinux2014 image, whose GCC 10 and glibc 2.17 are the toolchain of Panda's own wheels. `auditwheel repair` checks the glibc and libstdc++ symbol versions against the policy and applies the tag, without copying Panda's libraries in. The library finds Panda's through the relative rpath `$ORIGIN/../../../panda3d` (from `site-packages/world_builder/terrain/native/`). The amd64 SDK package serves both architectures: its `dtool_config.h` covers aarch64 under `#ifdef __aarch64__` (only Cg, unused here).
- **Windows** (`scripts/build_native.ps1`): links against the SDK's import libraries, which the wheel lacks, and loads the wheel's DLLs at runtime. They are already loaded by then (the binding imports `panda3d.core` first), and Windows reuses a loaded DLL of the same name.
- **macOS** (`scripts/build_macos_wheel.sh`): builds both architectures into one library, with a minimum of macOS 11.0, that of Panda's universal2 wheels. Panda's dylibs call themselves `@loader_path/../lib/<name>`, which would resolve next to our library, so the build rewrites those references to `@rpath/<name>`, with the rpath `@loader_path/../../../panda3d`. That breaks the ad-hoc signature, which arm64 requires, so the library is signed again.

Built in the source tree, the relative paths lead nowhere, but Panda's libraries are already loaded, and the loader matches them by name; macOS builds also get an rpath to the venv's panda3d.

### In CI

On every push to `main` and every version tag (`.github/workflows/ci.yml`):

- `package` builds the sdist and the pure wheel;
- `build_wheels` builds the platform wheels (the table above);
- `test_wheels` installs each in a clean venv for each of Python 3.12, 3.13 and 3.14, on Intel and Apple silicon for macOS, and runs the whole suite with `WORLD_BUILDER_REQUIRE_NATIVE=1`, so no collision or native noise test can be skipped. A step first checks the package comes from `site-packages` and prints how the collision library finds Panda's;
- on a tag, `release` checks that every platform wheel is named after the tag and attaches them with the sdist to its release.

To reproduce a Linux build locally:

```sh
python -m build
scripts/fetch_native_deps.sh
docker run --rm -v "$PWD":/io quay.io/pypa/manylinux2014_$(uname -m) \
    /io/scripts/build_manylinux_wheel.sh "/io/$(ls dist/*-py3-none-any.whl)" /io/wheelhouse
```

macOS: `scripts/build_macos_wheel.sh dist/*-py3-none-any.whl wheelhouse`. Windows: `scripts/build_native.ps1`, then `python scripts/platform_wheel.py dist/*-py3-none-any.whl win_amd64 wheelhouse <the two .dll>`.

## Loading

At load, `_binding.py` checks that a library exists for the running Panda3D series and that it reports the same series it was built against (`wb_panda_version`). Otherwise it raises `NativeCollisionError` with the build instructions. It does not check the patch version: the pin does.

## Not done yet

- **Fallback:** without the library the terrain loads, but collision is unavailable. A logged warning, or the slower numpy `TerrainQuery` path, could stand in.
- **Free-threaded Python** (`cp313t`, `cp314t`): Panda ships wheels for them, but the object-header offsets above differ there. The binding's signature check should fail cleanly, but nothing refuses these builds explicitly.
