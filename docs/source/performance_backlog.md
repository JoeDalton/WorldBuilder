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

All four found here (the sky after the terrain, a narrower reflection field
of view, the fused sun-depth evaluations, fewer samples on short sight
lines) are made: see [performance](performance.md), §14. The frame
breakdown above predates them: frames are 1–3 ms shorter now.

## Generation and loading (after #20)

- **The whole chunk in C.** A chunk is ~8 ms: ~5 ms of native noise, ~3 ms
  of numpy (low-band interpolation, normals, biome weights, vertex packing).
  Porting `Planet.evaluate` and the rest would make it ~5 ms, at the price of
  the planet's composition in two languages (stamps and paints still need a
  bridge). Generation no longer limits refinement at 60 Hz, so only worth it
  for slower machines or faster flight.
- Cloud tiles and the weather map: done (performance.md §15).
- **The LOD cadence at low frame rates.** A level waits for a pass 3 frames
  after its upload (`select_every_on_upload`): 100 ms a level at 30 fps.
  Time-based instead of frame-based would help slow machines; a pass every
  frame and a deeper queue were within noise at 30 Hz.
- **An incremental LOD pass in fast flight.** Each pass walks the whole
  quadtree (1–2.5 ms at low altitude). Revisiting only the nodes near a
  split or merge threshold since the last pass would save most of that.

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

## Sea

The swell's finite-difference gradient, the seabed bakes' GIL time and the
15 km handover's shader-compile hitch went with the separate ocean mesh: the
terrain's chunks draw the sea (performance.md §16).
