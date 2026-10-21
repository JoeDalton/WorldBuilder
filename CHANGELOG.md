# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Terrain on a true spherical planet: a cube-sphere with quadtree LOD,
  float64 positions re-based around the camera, and collision queries on the
  sphere. (#1)
- Several bodies (a planet and its moon) sharing one worker pool, and a
  near/far camera split (`world_builder.depth`) for views from the ground to
  orbit. (#1)
- `world_builder.atmosphere`: single scattering by air and haze, applied by
  every shader, with an analytic planet shadow, a twilight fill, redder
  sunsets on small planets (`sunset_radius`), and the light at any point
  (`Atmospheres.light_at`) for lighting anything else. (#2)
- `world_builder.sky`: a procedural star field (point stars, a Milky Way band
  and nebulae with live palettes) or a painted cube map, and the sun's disc,
  seen through the atmospheres. (#2)
- Planet clouds (`clouds.sphere.SphericalClouds`): billboard shells near the
  camera and a deck far away, one pattern from the ground to orbit, following
  a weather map (`clouds.weather`: circulation, cyclones, climate, terrain),
  their cost scaled by `CloudQuality`. (#2)
- `Ocean.set_light`, to light the water as the sea under the camera (dark at
  night), and an `Ocean` `reflection_scale` argument. (#2)
- `ocean.aces_tonemap`, the NumPy twin of the shaders' tonemap. (#4)
- `clouds.noise.sun_optical_depth_on_sphere` (the clouds' sun march on the
  CPU) and `deck_light_table` (the far deck's averaged lighting). (#5)
- Scripts: `terrain_shots.py` (fixed poses to PNG, with GPU-bound frame
  times; `night_sea` checks the sea's reflection by night, [#4) and
  `flight_bench.py` (frame times in scripted fast flights). (#2)
- The terrain demo gains the moon, the atmosphere, clouds (F5 cycles their
  quality), the sky (F6 to F8 for the nebulae), an ambient scale (F9 / F10)
  and Panda3D lights that follow the light at the camera (#2); and a white
  test ball ahead of the start pose, to judge the lighting on models (#5).
- Scene-graph tests on a live ShowBase (`tests/test_scene.py`). (#2)
- Documentation: [Lighting on objects](docs/source/object_lighting.md), how
  models are lit compared with the world and what would make them match.
  (#5)

### Changed

- Faster frames: worker processes at lower priority, one per physical core
  but one; a cheaper LOD pass, run less often; per-frame shader inputs
  written in place; cloud shaders that skip invisible billboards and clear
  air exactly. (#2)
- The terrain demo and scripts ask for no multisampling (Panda got ×16
  whatever was asked) and turn vsync off. (#2)
- The ocean's mirror shows the sky and the clouds. (#2)
- The sun's disc and the glow around it keep the colour of the light they
  cast: orange at sunset (the disc was white whatever its light), and so does
  their reflection on the sea. A low sun is drawn with a hotter core, a
  redder rim and a glare. (#5)
- The clouds' terminator is gradual instead of a hard line from lit to dark,
  close up and from orbit: their sunlight fades out toward the planet's
  shadow, and the still-lit air above lights them through twilight. (#5)

### Fixed

- A planet seen from beyond 10,000 km was culled whole (Panda's
  `lens-far-limit`): the far camera gets explicit cull bounds. (#1)
- `hide(HIDE_FROM_NEAR)` did nothing: each split-depth camera now has a
  camera bit of its own. (#1)
- The ocean was not hidden from its own reflection camera (the same Panda
  camera-mask rule): it now hides from the whole `REFLECTION_CAMERA_MASK`.
  (#2)
- The reflection camera follows the main camera's field of view (it was
  copied once). (#2)
- The flat cloud field's vertex shader writes `gl_ClipDistance[0]`, so the
  ocean's reflection pass no longer clips it at random. (#2)
- The planet clouds' cull bins no longer tie with the flat field's (sort 25).
  (#2)
- `flight_bench.py` sampled the ground under the previous camera position
  when placing a low flight. (#2)
- The worker-priority test failed on Windows (no `os.nice`). (#2)
- The sea's reflection by night was about twice as bright as the sky: the
  reflection, already tonemapped, was tonemapped again. It is now decoded to
  linear light first, and the colour it is cleared to is tonemapped like the
  rest. (#4)
- Clouds along the terminator were blue-grey from far away and turned orange
  when closing in: the far deck's lighting is now the average of its
  close-up lighting, by the sun's elevation and the weather's coverage. (#5)

## [0.1.0]

The port from SpaceFlight: procedural terrain with biomes, stamps and LOD,
the reflective ocean and the volumetric billboard clouds, all three curved
by a camera-relative drop.
