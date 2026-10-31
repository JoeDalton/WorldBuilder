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
- In flight: `scripts/flight_bench.py --offscreen --flight low --trace
  --no-hud`. It times the main thread's parts in every frame and prints
  what the slowest 5 % of frames spend beyond the median ones. Compare two
  trees run in turn, at least twice each: run-to-run noise is 0.3-0.5 ms,
  and any other load on the machine (a browser) spoils a run.
- Panda's own timers: PStats, recorded as text with the venv's
  `text-stats -p PORT -r -o FILE` and `want-pstats 1` in the PRC. The GPU's
  timer queries (`pstats-gpu-timing`) include the GPU waiting for the CPU's
  commands, so a long GPU pass is not proof of GPU-bound frames. Changing
  the resolution is: the low flight takes the same time at 1600×900 and
  2560×1440, so it is CPU-bound.

## The low flight now (after performance.md §19)

Mean 8.5-8.8 ms, median 8.5, p95 12-12.7, max ~22 (it was 13.5 / 18-19 /
36-47 at the start of the slow-frames round). The main thread's parts, in
median frames and in the slowest 5 %:

| Part | Median frame | Slow frame |
|---|---|---|
| Panda's `renderFrame` (cull 1.5, draw 3.0) | 4.0 ms | 6.9 ms |
| `Terrain._select` | 0.7 ms | 1.6 ms |
| The demo's altimeter (`TerrainDemo._fly`) | 0.8 ms | 1.4 ms |
| `CloudVolume._resort` | 0.7 ms | 1.1 ms |
| `Terrain._upload` | 0.7 ms | 1.0 ms |
| `ChunkExecutor.collect` | 0.1 ms | 0.6 ms |
| `CloudVolume._upload` | 0.05 ms | 0.35 ms |
| `Sea.update` | 0.2 ms | 0.3 ms |
| Panda's state GC (`transform-cache false`) | 0.02 ms | 0.02 ms |

## Remaining paths (main thread, in flight)

Ranked by what they would take off the slow frames. None is made yet.

- **The demo's altimeter** (0.8 ms a frame, +0.6 in slow frames, spikes
  to 3.9 ms). `TerrainDemo._fly` asks `TerrainQuery.height_at` for one
  point every frame: ~100 numpy calls (`locate`, `_in_chunk`,
  `_triangle_intersect`, a new ctypes struct for the swell per call). When
  no stored chunk covers the point it evaluates the planet itself
  (`planet.sample_dirs`), the spikes. Fix: a scalar path for one point in
  `TerrainQuery._interpolate` (plain floats, the same triangle), the
  swell's struct cached per `Swell`, and the coarsest stored ancestor
  instead of the planet. Only the demo pays it every frame (SpaceFlight
  relies on the native collision solid), but every benchmark runs the demo.
- **`renderFrame`'s CPU part** (+2.9 ms in slow frames). Cull 1.5 ms (the
  window's 0.8, the sea reflection's 0.65) and draw 3.0 ms (state 1.0,
  shader inputs 0.6, primitives 0.6). A frame with chunk uploads renders
  0.6-0.8 ms slower: the new vertex buffers, and the whole chunk table sent
  again (Panda 1.10 has no partial texture update; a table split into
  smaller textures would send less). `NodePath.prepare_scene` at upload
  could move the buffer creation off the drawing frame.
- **The sea reflection's cull** (0.65 ms every frame near the sea). Its
  camera culls the whole planet again (its lens reaches 4× the main far
  distance). A shorter reach, or the coarse far chunks and the far cloud
  shells hidden from it, would cut it, at the price of what the far water
  reflects: a visual decision.
- **`Terrain._select`'s Python** (0.7 ms, +0.9 in slow frames). The native
  pass is 0.02 ms; the rest is around it: turning its results into Python
  keys (0.15 ms; returning what changed since the last pass instead would
  shrink it), the show/hide set operations, the collision update and the
  executor's request.
- **Chunk results through the pipe** (`collect` +0.46 ms in slow frames,
  spikes ~3 ms when several land together). The pool's manager thread
  unpickles ~204 KB a chunk (161 KB of vertices) at ~270 chunks a second.
  Measure first how much of it holds the GIL (the reads release it); if it
  is over ~0.3 ms a frame, hand the vertex buffers over through shared
  memory (`workers.SharedArray`, the other way), released on eviction and
  `set_spec`, orphans cleaned up.
- **The clouds' re-sort** (0.7 ms, +0.5 in slow frames). Four tiles a frame
  at 54 µs per thousand billboards in flight against 17 in isolation: the
  particle rows are cold in the cache. A compact copy of each tile's
  positions (12 bytes a billboard instead of 36) would help (~0.2 ms); the
  scoring of the ~86 shown tiles (0.15 ms) could be vectorised.
- **Cloud tile uploads** (+0.3 ms in slow frames, 0.3-1 ms a tile): a
  budget per frame by billboards instead of by tiles would cap the frames
  that take two big ones.
- **The clouds' tile selection** (0.1-0.3 ms when it runs): its own
  traversal in Python over `LodSelector._measure`; the native LOD library
  could serve it.
- **Cloud tiles' draws** (0.53 ms over all cameras): one draw per tile and
  camera; merging a shell's tiles would save 0.2-0.3 ms.
- **Half-resolution clouds** (a setting, off): its two layer cameras cost
  ~1.2 ms of cull and draw, more than the GPU time it saves on CPU-bound
  machines. Fewer occluders in the layers (the terrain only) or the
  scene's depth reused from a game's post-process buffer would make it pay.
- **The demo's HUD** (out of scope): ~5 ms every 0.2 s rebuilding one
  multi-line text; one TextNode per line, updated when its text changes,
  would fix it. `--no-hud` keeps it out of the benchmarks.

Measured and left as they are:

- **Upload budgets:** 3 chunks and 2 ms a frame, or 4 and 3, against 6 and
  4: no lower p95 now that uploads cost ~0.7 ms.
- **The state GC's rate** (`garbage-collect-states-rate 0.25`): states pile
  up and it costs more; with `transform-cache false` there is nothing left
  to collect.
- **One-off long frames:** a 108 ms frame once in four runs (not traced
  to a cause); after a reboot, 3-11 s stalls while the GPU sat in its P5
  power state (the driver's PowerMizer back to adaptive; "Prefer Maximum
  Performance" in NVIDIA X Server Settings fixes it). Check `nvidia-smi
  --query-gpu=pstate,clocks.gr --format=csv` during a run before
  benchmarking: P0 is full clocks.

## Where a frame goes (still views)

GPU-synced still views, measured before the #20 rounds (frames are several
ms shorter now; the shares between layers still hold). Fast flight is
CPU-bound instead: see above.

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

- **The planet's composition in C.** A chunk is ~6.7 ms since its geometry
  went native (performance.md §18): ~4.5 ms of native noise, ~2 ms of numpy
  composition (curves and masks around the noise, biome weights 0.6 ms,
  stamps 0.7 ms, paints 0.25 ms). Porting those would make it ~5 ms, at the
  price of the planet's rules in two languages (stamps and paints would need
  their images passed over). Left out by choice: generation no longer limits
  refinement at 60 Hz.
- Cloud tiles and the weather map: done (performance.md §15).
- **The LOD cadence at low frame rates.** A level waits for a pass 3 frames
  after its upload (`select_every_on_upload`): 100 ms a level at 30 fps.
  Time-based instead of frame-based would help slow machines; a pass every
  frame and a deeper queue were within noise at 30 Hz.
- The LOD pass: native instead (performance.md §17). An incremental pass
  would save little more: the C traversal takes 0.02 ms; what remains of a
  pass (0.5 ms) is Python around it.

## Savings with a cost

- **Clouds at lower resolution:** done as a setting,
  `CloudSettings.half_resolution` (performance.md §19): 1-1.7 ms less GPU
  time, ~1.2 ms more main-thread time for its two layer cameras. Making
  those cheaper (fewer occluders in the layers, or the scene's own depth
  reused through a game's post-process buffer) would make it pay on
  CPU-bound machines too.
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
- **The terrain's atmosphere per vertex on far chunks:** the per-pixel
  atmosphere is ~2 ms of the terrain's GPU time; far chunks (their pixels
  hundreds of metres apart) could take it from their vertices, at the price
  of banding where the air changes fast (the limb, the terminator).
- **A narrower sea fade band:** the near look and its reflection run up to
  18 km (`OceanLook.near_band`, 12-18 km), 2.5 ms at 17 km where the old
  ocean was hidden. A lower band moves that cost down; it changes where the
  sea's look changes.

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

## Cirrus (not measured yet)

The planet's cirrus sheet (`clouds/cirrus.py`, `cirrus.frag`) was written
on a machine that cannot time it (WSL with software GL, at 130-300 ms a
frame for the whole demo). It was built to be cheap, but nothing below has
been measured. Measure it as the other layers are measured (§ How it was
measured), on the GTX 1070, and write the numbers here.

What it costs, by construction:

- **Vertices:** two draws of a 6 × 64² cube-sphere (~49k triangles each,
  never culled), with a trivial vertex stage.
- **Per fragment that is drawn** (cover above 0.2 %, alpha above 1/512):
  - the ray against the sphere, one cube-map fetch;
  - the stream function: two `texelFetch`es plus up to 16 Gaussians;
  - 4-5 taps of hashed value noise, each 8 hashes and a quintic blend
    (~100 ALU operations a tap, no fetch). The fibril tap is skipped past
    about 4 px per fibril;
  - `cloudSunTint` and `cloudTwilight` (4 `atmoSunlight`);
  - `atmosphereAlong` at `uAtmoSteps`, the most expensive part.
- **Per fragment discarded:** the wrong node's fragments go after the
  ray-sphere test. A clear sky (cover under 0.2 %) costs one cube-map fetch.
  Fragments under the threshold still pay for the noise.

To measure, each as the median frame with the cirrus shown minus hidden
(the demo's C key, or `PlanetClouds.sheet.root.hide()`), at 1600×900:

| Scene | `terrain_shots.py` pose | What it stresses |
|---|---|---|
| From the ground, looking up | `cirrus_up` | full-screen sheet, no atmosphere savings |
| Between the layers | `cirrus_between` | grazing sheet, long chords, the far fade |
| Above both | `cirrus_above` | the front node over the cumulus |
| Orbit | `orbit` | the far sheet (cover only, no noise) |
| Dusk | `cirrus_dusk` | twilight terms |
| The ground pose with `--half-res-clouds` | `ground` | cirrus at full resolution over half-resolution clouds |

For each, also note:

- the share of the sheet's fragments that are discarded: with
  `uCloudDebug` bit 8 (flat colour), and against a build that returns
  before the noise;
- the cost of `atmosphereAlong`, with `uCloudDebug` bit 2 (no atmosphere)
  on and off.

If it costs too much, cheapest change first:

1. **A fibril knob** in `CloudSettings` (cirrus fibrils off at LOW/MID).
   That saves one noise tap out of 4-5.
2. **The atmosphere per vertex** on a denser sheet (as the cumulus
   billboards do), or once per pixel block. On the deck it is per pixel
   because it is one layer; the cirrus is a second one.
3. **The noise from a texture again**, with tricubic filtering (8 fetches
   a tap). This is a trade of ALU for bandwidth, worth trying only if the
   hash shows up as ALU-bound.
4. **The cirrus in the half-resolution pass** (stop excluding its root in
   `World._sync_half_res`): a quarter of the pixels, softer fibres.

