# Ocean

[`world_builder.ocean`](https://github.com/JoeDalton/WorldBuilder/tree/main/src/world_builder/ocean/)
is a camera-locked reflective ocean, curved with the planet. It takes a `game`
object by duck typing (see the module docstring for exactly what it reads),
registers its own per-frame update in `game.method_lists`, and exposes
`clean()`. The per-class API is in the [code reference](apidocs/index.rst).

## `ocean/ocean.py` — clipmap-free reflective ocean

[`ocean.py`](../../src/world_builder/ocean/ocean.py)'s `Ocean` (its module
docstring has the overview):

- **Per-pixel waves.** By default the surface is one huge flat quad re-centred
  under the camera every frame. All wave detail is computed in the fragment
  shader from world position, so sliding the quad causes no motion.
  `compute_wave_dirs` precomputes the per-iteration wave directions once on the
  CPU as a uniform array; `set_wave_iterations()` changes wave detail at
  runtime.
- **Geometric swell** (`geometric_swell=True`, used by SpaceFlight's ocean
  level; a prototype with known artifacts, see the
  [swell artifact report](https://github.com/JoeDalton/WorldBuilder/blob/main/src/world_builder/ocean/OCEAN_SWELL_ARTIFACT_REPORT.md))
  builds `make_swell_grid_mesh` instead: a dense grid that `ocean.vert`
  displaces by the swell, tapered to flat before its edge, ringed by
  geometrically spaced border rings in the same mesh.
- **Curved over the planet** (swell mode only: four vertices cannot curve).
  Every vertex droops by `d² / 2R`, baked into the mesh. That is exact because
  the mesh is re-centred on the camera, the same origin the cloud shader droops
  from, and costs nothing per frame. The horizon then dips `sqrt(2h/R)` below
  eye level (1.0° at 1 km, 3.05° at 9 km). The border is rings rather than one
  quad because a quad interpolates linearly and would droop into a cone; the
  spacing is geometric because the flat-cell sag is `spacing² / 8R`, and 28
  rings keep it under 1 arcmin (15 would give 2.9) for about 23% more
  triangles. The surface reaches past the horizon seen from 20 km,
  `sqrt(2R·altitude)` ≈ 505 km plus a 10% margin, so the planet's bulge hides
  its edge. Two things stay flat on purpose: a game's collision plane (a ship is
  within metres of the camera, where the droop is micrometres) and the planar
  reflection, which mirrors about z=0 and so drifts by the droop with distance
  — 7.8 m at 10 km, where the normal is already flattened to a mirror.
- **Aerial perspective.** Distant water fades to `haze_color` after the
  tonemap, as the clouds do, so sea and sky meet at the horizon. Its
  `haze_distance` (150 km) is much shorter than the clouds' 400 km: the air
  path to sea level is far denser than to cloud at altitude.
- **Far plane.** SpaceFlight raises it to 600 km (`CAMERA_FAR_M`), set before
  the scene is built because the reflection camera copies the lens. Panda's
  default 100 km clips the outermost cloud shell and keeps the ocean horizon in
  view only up to 785 m of altitude; 600 km clears the 512 km shell and holds
  the horizon to about 28 km. It costs almost no depth precision, which goes as
  `z² / (near · 2^bits)` and is set by the near plane.
- **Planar reflections.** `make_reflection_buffer` builds an offscreen texture
  buffer and a reflection camera that `mirror_camera` (called from `update`)
  mirrors about the water plane every frame. A clip plane restricts it to
  geometry above the water, and a camera-mask bit hides the ocean from it. The
  buffer is sized from the app's `graphics_manager.get_render_size()` when
  it has one (the window size otherwise) times `reflection_scale` (the `Ocean` argument,
  else `app.graphics_settings`, else 0.5), so it tracks the internal render resolution rather than
  the window. `uReflUVScale` corrects for power-of-two texture padding and is
  refreshed from the real texture size once the buffer is realized: the default
  pipeline pads, simplepbr (not used) does not.
  In the terrain demo the mirror shows the sky (its dome is drawn on the
  far plane, so the reflection camera's shorter range keeps it) and the
  clouds. Shaders drawn through the mirror write `gl_ClipDistance[0]`
  (terrain: the real distance; sky, stars, clouds: always kept, they are
  above the water). Its resolution is `reflection_scale` (half the view by
  default); lowering it saves little in the demo, where the mirror's cost
  is per billboard, not per pixel.
- **No collision of its own.** Physics is the game's business: SpaceFlight's
  `space_flight.scenes.ocean.Ocean` subclass adds a flat terrain collision
  plane at Z=0 so ships can crash into the water.

## Shaders

Driven by [`ocean/ocean.py`](../../src/world_builder/ocean/ocean.py) (above):

- **[`ocean.vert`](../../src/world_builder/ocean/shaders/ocean.vert)** is a
  passthrough plus the optional `uGeometricSwell` mode (which SpaceFlight
  enables): it displaces vertices vertically by the same `swellField` the
  fragment shader samples, tapered to zero near the dense grid's edge so the
  displaced centre joins the flat border. Its world position comes from
  `uModelMatrix` (refreshed by `Ocean.update` after the plane is re-centred),
  not `p3d_ModelMatrix`, so the waves stay anchored to the world rather than
  to the moving render origin.
- **[`ocean.frag`](../../src/world_builder/ocean/shaders/ocean.frag)**:
  - **Wave field.** `getwaves`/`waveGradient` accumulate Gerstner-like octaves
    (`wavedx`) along per-iteration directions `iWaveDirs`, precomputed in
    Python by `compute_wave_dirs`. `waveGradient` computes the slope
    analytically in one pass instead of a three-sample finite difference.
  - **Level of detail.** `detailAngle` (view-ray elevation, so the look is
    consistent at any altitude) and `distFade` (camera distance) fade wave
    detail toward a flat mirror. `normalIter` decays exponentially with
    distance (`uWaveFadeK2`); `warpIter` and the warp strength follow
    `lodFactor = detailAngle × distFade`. Below a `detailAngle` of 0.01 the
    small waves are skipped.
  - **Swell + frequency modulation.** A large-scale `swellField` (the product
    of two drifting value-noise layers, so it changes shape rather than
    scrolling) tilts the normal with no distance fade, and also shifts the
    small-wave phase (`fmPhase`) so the dominant octave's tiling drifts.
  - **Reflection.** The reflection UV is recomputed per fragment from the
    flat (z = 0) surface position: interpolating a vertex clip-space value
    would skew the projective divide per triangle once the swell displaces
    vertices. Sampling at that flat footprint still slides against the
    displaced geometry (see the
    [ocean swell artifact report](https://github.com/JoeDalton/WorldBuilder/blob/main/src/world_builder/ocean/OCEAN_SWELL_ARTIFACT_REPORT.md)).
    The ripple-perturbed UV is clamped to the rendered region
    (`uReflUVScale`) so it never samples the power-of-two padding.
  - **Output.** The fresnel blend of reflection and water colour is
    ACES-tonemapped (`aces_tonemap`), then hazed toward `uHazeColor` over
    `uHazeDistance` — after the tonemap, as the cloud field does, so sea and
    sky meet cleanly at the horizon.
  - **Debug.** `uDebugMode` 1–8 visualise intermediate quantities (0 is normal
    rendering): normal, reflection UV, fresnel, world grid, reflection-UV
    clamp, raw reflection, pre-tonemap clipping, FM phase.
