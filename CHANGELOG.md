# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Changes since 0.1.0 (the port from SpaceFlight).

### Added

- Terrain on a true spherical planet: a cube-sphere with quadtree LOD,
  float64 positions re-based around the camera, and collision queries on the
  sphere.
- Several bodies (a planet and its moon) sharing one worker pool, and a
  near/far camera split (`world_builder.depth`) for views from the ground to
  orbit.
- `world_builder.atmosphere`: single scattering by air and haze, applied by
  every shader, with an analytic planet shadow, a twilight fill, redder
  sunsets on small planets (`sunset_radius`), and the light at any point
  (`Atmospheres.light_at`) for lighting anything else.
- `world_builder.sky`: a procedural star field (point stars, a Milky Way band
  and nebulae with live palettes) or a painted cube map, and the sun's disc,
  seen through the atmospheres.
- Planet clouds (`clouds.sphere.SphericalClouds`): billboard shells near the
  camera and a deck far away, one pattern from the ground to orbit, following
  a weather map (`clouds.weather`: circulation, cyclones, climate, terrain),
  their cost scaled by `CloudQuality`.
- `Ocean.set_light`, to light the water as the sea under the camera (dark at
  night), and an `Ocean` `reflection_scale` argument.
- Scripts: `terrain_shots.py` (fixed poses to PNG, with GPU-bound frame
  times), `flight_bench.py` (frame times in scripted fast flights); the
  terrain demo gains the moon, the atmosphere, clouds (F5 cycles their
  quality), the sky (F6 to F8 for the nebulae), an ambient scale (F9 / F10)
  and Panda3D lights that follow the light at the camera.
- Scene-graph tests on a live ShowBase (`tests/test_scene.py`).

### Changed

- Faster frames: worker processes at lower priority, one per physical core
  but one; a cheaper LOD pass, run less often; per-frame shader inputs
  written in place; cloud shaders that skip invisible billboards and clear
  air exactly.
- The terrain demo and scripts ask for no multisampling (Panda got ×16
  whatever was asked) and turn vsync off.
- The ocean's mirror now shows the sky and the clouds.

### Fixed

- A planet seen from beyond 10,000 km was culled whole (Panda's
  `lens-far-limit`): the far camera gets explicit cull bounds.
- `hide(HIDE_FROM_NEAR)` did nothing: each split-depth camera now has a
  camera bit of its own.
- The ocean was not hidden from its own reflection camera (the same Panda
  camera-mask rule): it now hides from the whole `REFLECTION_CAMERA_MASK`.
- The reflection camera follows the main camera's field of view (it was
  copied once).
- The flat cloud field's vertex shader writes `gl_ClipDistance[0]`, so the
  ocean's reflection pass no longer clips it at random.
- The planet clouds' cull bins no longer tie with the flat field's (sort 25).
- `flight_bench.py` sampled the ground under the previous camera position
  when placing a low flight.
- The worker-priority test failed on Windows (no `os.nice`).
