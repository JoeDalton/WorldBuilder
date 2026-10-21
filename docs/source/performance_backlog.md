# Performance backlog

Frame-time savings found but not yet made (October 2026). Read this before
optimising again: the measurements, the candidates and what each would cost
are here. The history of what was done is in [performance](performance.md).

## How it was measured

- The terrain demo offscreen at 1600×900, no MSAA, on a GTX 1070, with the
  machine idle (load below 1). Other load moves the numbers by 1–2 ms.
- Each timed frame waits for the GPU (a one-pixel read back after
  `taskMgr.step()`, as `scripts/terrain_shots.py` does). Without it an
  offscreen frame is not held back by the GPU at all.
- A layer's cost is the median frame time with it shown, minus the median
  with it hidden (80 frames each).

## Where a frame goes

Frames are GPU-bound everywhere: the demo's Python update takes 0.7–1.3 ms.

| View | Frame | Clouds | Terrain | Sea reflection | Sky |
|---|---|---|---|---|---|
| Start pose (1.3 km) | 15.6 ms | 4.9 | 6.0 | 3.7 | 1.4 |
| Dusk, in the slab (lat −11.226, lon −53.872, 2,383 m, heading −77.7, pitch −6.4) | 19.0 ms | 10.8 | 3.2 | 3.3 | 1.2 |
| Inside the clouds (2.7 km) | 14.0 ms | 5.5 | 3.7 | 2.4 | 1.4 |
| 20 km up | 7.5 ms | 1.5 | 2.7 | 0.1 | 0.9 |
| Orbit | 2.5 ms | 0.4 | 0.5 | 0 | 0.5 |

Within those:

- **Terrain:** its per-pixel atmosphere is 2.0 ms (start and 20 km), its
  textures 0.3–0.9 ms.
- **Sea reflection:** its terrain 1.1–1.2 ms, its clouds 1.0–1.7 ms, its
  sky 0.2–0.7 ms. Halving its resolution again saves under 1 ms: its cost
  is per billboard and per chunk, not per pixel.
- **Clouds** at the dusk pose: the sun march ~3.4 ms; density and blending
  ~7 ms (the camera is inside the slab beside a large cloud).

## Free savings (no visible change)

### 1. Draw the sky after the terrain: 0.4–0.8 ms (measured)

The sky dome is drawn first, in the `background` bin, so its per-pixel
atmosphere is computed over the whole screen, most of which the terrain then
covers. Its vertices already sit on the far plane (`sky.vert`), so it can be
drawn after the opaque bin with a depth test and only the visible sky gets
shaded.

- A fixed bin between `opaque` (20) and the clouds, for the dome (draw order
  0) and the stars (1). The clouds must stay after it, so their bins move
  up one: `_CLOUD_BIN_SORT` 21 → 22, `_SHELL_BIN_SORT` 28 → 29 in
  `clouds/sphere.py`. Shells use `29 - shell`, still below `transparent`
  (30) for up to 7 shells.
- On the sky node: depth test on with `DepthTestAttrib.M_less_equal`, depth
  write off.
- Measured with a temporary rebinding: start 14.8 → 14.4 ms, dusk
  19.4 → 18.6 ms, 20 km 7.1 → 6.6 ms.
- To check: the near pass still never draws the sky (it is hidden from the
  near camera: `HIDE_FROM_NEAR`); the stars still add over the dome.

### 2. Narrow the reflection's field of view: 0.4–0.6 ms (measured)

`ocean.py` gives the reflection camera 1.4× the main camera's field of view
(`refl_lens.setFov(fov.x * 1.4, fov.y * 1.4)`), against culling artefacts when
turning. The reflection camera is mirrored in the same frame as the main
camera (`mirror_camera`, from `update`), and a visible water point's mirror
image is inside the mirrored camera's frustum at the same field of view. So
only the ripples' small UV offset needs a margin.

- 1.1× saved 0.4 ms (start) and 0.6 ms (dusk): fewer billboards and chunks in
  the reflection's frustum.
- The buffer's pixel count is unchanged, so reflections also get sharper.
- To check: no clamped, stretched texels at the screen's edges with strong
  ripples (`uDebugMode == 5` in `ocean.frag` shows the clamped area in red).

### 3. Fuse the sun-depth evaluations: ~0.5–1 ms (estimated)

Each atmosphere sample (`atmoSegment` in `atmosphere.glsl`) calls
`atmoDepthToward` twice, for Rayleigh and Mie. Both share `r`, `cosChi`,
`sinChi` and the tangent test, and differ only in the scale height. One
function returning a `vec2` computes the shared parts once: the same result,
fewer `sqrt`/`exp`.

Per-pixel atmosphere runs on the terrain (2 ms), the sky, the ocean and the
far cloud deck, so this saving applies to all four.

### 4. Fewer samples on short sight lines: ~1 ms at ground level (estimated)

Every sight line takes `ATMO_STEPS` (8) samples, even across a few hundred
metres of air to nearby ground, where the integrand barely changes.

- Choose the step count from the segment's length and vertical extent
  relative to the scale heights: Mie's 1.2 km is the tight one. For example,
  2 to 8 steps.
- The planet's shadow is already found analytically along the ray (each
  sample counts its lit share), so fewer samples do not bring the dusk
  banding back.
- Accept only if the colour error stays under 1/255 against 8 steps. Check
  with images of the start, dusk and sunset poses.

## Savings with a cost

- **Clouds at lower resolution:** a half-resolution cloud pass, composited
  over the scene (plan item 1.7 of the cloud work). It is the biggest
  remaining lever (clouds are 5–11 ms), but softens silhouettes and needs a
  depth pre-pass for occlusion.
- **Cached sun shadow:** the sun's optical depth stored in a 3D texture
  around the camera, refreshed when the sun or camera moves, one fetch
  instead of the march (~3.4 ms at dusk). Its texels (~200 m) are coarser
  than the march's 160 m steps, so the self-shadowing gets blurrier.
- **Cheaper terrain in the reflection:** a lighter shader for the reflection
  camera (camera tag states), without detail normals and with fewer texture
  fetches. That saves up to ~1 ms; the difference is hard to see at half
  resolution under ripples.
- **Fewer billboards:** MID quality (overlap 2.0 / 1.5, three octaves from
  shell 2). The user prefers HIGH's look.

## Tried and rejected

- **Sun march per vertex** for the far shells: far billboards came out lit
  flat (at the centre) or unshaded (at the corners). The per-fragment march
  costs 0.2–0.9 ms in most views.
- **MSAA:** Panda picks the window's 16-sample framebuffer config whatever
  count is asked for, which made the start view 40 ms. Exact counts need an
  offscreen buffer (tried, works); the user chose no MSAA.
- **Lower reflection resolution:** 0.5 → 0.25 saved under 1 ms (see above).
