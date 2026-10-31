# Performance notes

Known computational costs and open concerns for `world_builder.terrain` and,
in §13, the atmosphere, sky and clouds drawn with it, with the measurements
behind them. Frame rate in the demo is currently high (GTX 1070, i7-6700K: 4
cores / 8 threads), so nothing here is urgent. This page records where the
time goes, what was tried, and what to look at first if it becomes a problem.

Measured with `python -m world_builder.terrain.bench`, headless demo runs,
`python scripts/flight_bench.py` (frame times in scripted fast flights,
`--profile` for the main thread, `--offscreen` to avoid the display's refresh
cap or throttling) and `python scripts/terrain_shots.py` (settled frame times
at fixed poses, §13).
CPU timings on this machine are noisy (frequency scaling): treat them as
±30 %. Since the move to the cube-sphere (600 km planet by default), the
figures in §3–5 and §10 are re-measured; §1, §2 and §6–9 still hold as written.

## Summary

| # | Concern | Where | Impact today | Status |
|---|---|---|---|---|
| 1 | Worker pool throughput vs the render loop | `workers.py` | Too many workers slowed the frame | Default: physical cores − 1, lower priority |
| 2 | glibc returns numpy temporaries to the OS | worker processes | Workers ~40 ms/chunk instead of ~24 | Applied (§14) |
| 3 | numpy per-octave cost | `noise.py`, `planet.py` | was ~25 ms per chunk | Native noise: ~8 ms (§14) |
| 4 | LOD pass in Python | `quadtree.py` | ~1 ms per pass, not every frame | Optimised |
| 5 | Chunk and triangle counts | `quadtree.py` (K=2) | ~600 draws, ~5 M triangles, twice with the reflection | Fine on desktop GPUs |
| 6 | Memory held by the caches | `terrain.py` | ~410 MB RAM, ~190 MB GPU per planet you are near | Fine on desktop |
| 7 | Synchronous generation in queries | `query.py` | 25 ms stall per missing chunk | **Must handle before game integration** |
| 8 | Fragment shader texture samples | `terrain.frag` | Up to ~20 samples per pixel | Fine so far |
| 9 | Initial fill and spec changes | `terrain.py`, `planet.py` | Several seconds of blur; 0.3 s main-thread hitch | Accepted |
| 10 | Descent from orbit | `workers.py`, `depth.py` | Demand above the pool's ceiling when boosting; split-depth draws twice | Accepted (coarse, never holes) |
| 12 | Frame time in fast flight | demo main thread | was ~30 fps flying low, boosted | Fixed: 89 fps offscreen (§12) |
| 14 | New chunks visibly late (#20) | generation, LOD pass, start-up | ground sharp ~3.8 s after a jump | ~1.0 s (§14) |

## 1. Worker pool throughput

**Current default** (`workers.default_workers`): one worker per *physical*
core but one (Linux topology; two logical CPUs left free elsewhere), never
fewer than one. Workers run at `nice +10`, so generation yields the CPU to the
render loop. Measured in fast low flight (4 cores / 8 threads, chunks/s
across the whole flight, all workers niced):

| Workers | fps (50 Hz display) | Demo update, median | Chunks/s | Backlog (wanted) |
|---|---|---|---|---|
| 6 (old default, cores − 2) | 35 | 15.6 ms | 136 | 137 |
| 4 | 43 | 11.7 ms | 121 | 138 |
| 3 (new default) | 47 | 9.5 ms | 111 | 141 |
| 2 | 49 | 8.6 ms | 84 | 153 |

Workers on the renderer's hyper-threaded sibling cost frame time and added
little: the backlog hardly moves between 6 and 3. The older measurements
below were taken before the cube-sphere and before niceness.

Each chunk is generated in a spawned worker process (`ChunkExecutor`,
`cores - 2` = 6 workers here). Throughput does not scale with worker count:

| Workers | Chunks/s | Worker time per chunk |
|---|---|---|
| 1 | 25 | 40 ms |
| 2 | 38 | 41 ms |
| 4 | 49 | 45 ms |
| 6 | 42 | 65 ms |

With the malloc fix of §2 applied to the workers:

| Workers | Chunks/s | Worker time per chunk |
|---|---|---|
| 1 | 40 | 24 ms |
| 2 | 54 | 26 ms |
| 4 | 54 | 30 ms |
| 6 | 48 | 42 ms |

A bare `ProcessPoolExecutor` with dummy 25 ms CPU tasks returning 200 KB
scales linearly on the same machine (39 → 156/s for 1 → 4 workers), so the
pool itself is not the limit. Two effects seem to combine:

- **Workers slow each other down** (24 → 42 ms with 6 workers): only 4
  physical cores, hyper-threads sharing them, plus memory bandwidth.
- **Something outside the worker compute caps throughput** around 54/s even
  when per-chunk time is flat (2 → 4 workers). Not yet identified.

Suspects to measure, in order:

1. Parent-side result handling: each result is a ~200 KB pickled `ChunkData`
   (161 KB of vertex bytes, plus elevation and weights), unpickled in the
   executor's management thread, which competes for the GIL with the main
   thread.
2. Per-task pickling of the `PlanetSpec` (small today, but it carries the
   stamp and paint lists).
3. Cold per-worker caches: `octave_frame`, `planet_for`, stamp mip pyramids,
   paint channels.

Ideas once measured: return only elevation and weights and pack vertices on
the main thread, or pass results through shared memory; default the worker
count to the physical core count; batch several chunks per task.

What it costs today: fine at normal speed; when boosting near the ground,
finer levels arrive late (the coarse parent stays visible, no holes). Demand
is roughly `8 K v / leaf size` chunks/s: about 50/s at 400 m/s near the
ground.

## 2. glibc malloc and numpy temporaries

Most temporaries in the noise code are (N, 3) float64 arrays of about 128 KB,
right at glibc's default mmap threshold, so each one is mapped and unmapped
from the OS: page faults on every numpy operation. Raising the threshold for
the workers takes a chunk from ~40 ms to ~24 ms:

```
MALLOC_MMAP_THRESHOLD_=4194304 MALLOC_TRIM_THRESHOLD_=67108864
```

**Applied** (§14): `ChunkExecutor` sets them in `os.environ` before the pool
spawns its workers (`workers.WORKER_MALLOC_ENV`, unless already set; Linux
only). With the native noise there are far fewer temporaries, so it matters
less: 342 against 358 chunks/s (with numpy: 74 against 152).

Single-threaded BLAS (`OPENBLAS_NUM_THREADS=1`) was also tried: marginal
(62 → 68 chunks/s in that run).

## 3. Noise cost per chunk

`simplex3` costs about 0.7–1.6 ms per octave for one chunk's 5,329 samples
(65² vertices plus a 4-sample border at leaf level). It is bound by numpy's
per-operation overhead (~150 operations per call), not by arithmetic.

Per leaf chunk:

- About 24 octaves are evaluated per vertex (ridged mountains 9, hills 5,
  detail 7, climate jitter 3).
- The low band (warp, continents, mountain mask, base climate: ~23 octaves)
  is evaluated on a lattice of each cube face (230 m at 600 km; it scales
  with the planetary features) and Catmull-Rom interpolated, which costs
  under 5 cm of elevation error. This roughly halved the chunk time.
- Normals and slopes are now 3D cross products of neighbouring positions;
  no measurable change.
- Total: ~25 ms single-threaded, ~20–28 ms at every level (bench, 600 km):

| Level | 0 | 2 | 4 | 6 | 8 | 10 | 13 |
|---|---|---|---|---|---|---|---|
| ms | 26 | 23 | 22 | 28 | 28 | 26 | 25 |

Already done: a 64-bit hash instead of a permutation table (no visible
repeats), `take()` instead of fancy indexing for the gradient lookup (4×
faster gather), band limiting (octaves shorter than 4× the vertex spacing are
skipped).

Further options, from cheapest:

- Cache low-band lattice nodes across chunks in each worker: ~4 ms per leaf
  goes to evaluating only 16–25 lattice nodes, all overhead.
- Fewer detail octaves at leaf level, or a tiling 2D detail texture for
  wavelengths under ~64 m.
- Local float32 coordinates per chunk (only safe after removing the large
  lattice offsets).
- numba: would remove most of the overhead, but adds a dependency (declined
  so far).

**Done since** (§14): the octave sums are a small C library (`native/noise.c`),
so a chunk is ~8 ms; what is left is ~5 ms of noise and ~3 ms of numpy
(biomes, normals, interpolation, packing).

## 4. LOD selection in Python

`LodSelector.select` walks the six face quadtrees in pure Python (bench,
600 km planet, warm caches):

| Camera altitude | Chunks drawn | LOD pass |
|---|---|---|
| 2 m | ~590 | 3.0 ms |
| 1 km | ~625 | 3.2 ms |
| 20 km | ~290 | 1.5 ms |
| 1,200 km | 13 | 0.1 ms |

Those were the first cube-sphere figures, the flat chart's cost (2.9 ms for
562 chunks). After optimising the hot loop, the same bench gives **1.1 ms at
2 m, 1.1 ms at 1 km, 0.5 ms at 20 km**:

- Per node, the box extents for its height bounds and its horizon terms are
  cached (`LodSelector._prepare`). A visit is then 3 dot products, 3 clamps
  written as comparisons (no `min`/`max` calls) and one square root.
- Distance and culling are one call (`_measure`).
- The terrain keeps one `(h_min, h_max)` tuple per stored chunk
  (`Terrain._bounds`), so the cache recognises unchanged bounds by identity.
- The wanted and prefetch lists are sorted by the distances measured in the
  traversal. Re-measuring them with unknown bounds for the sort overwrote
  cache entries the next pass needed: about 30 % misses while flying.

Each node's tangent-frame box itself is built once per key, in pure Python
(~15 µs; numpy took 230 µs on so few points). A pass over new ground pays
that for its new keys.

When it runs: after the camera moves a quarter of a leaf (29 m) or a 250th
of its altitude, which is inside the 15 % hysteresis band, so no split waits
visibly; after uploads at most every 3rd frame; and at least every 15
frames. It used to run every frame in fast flight (every upload triggered
one).

Further options if needed: reuse the previous selection and only revisit
nodes near split thresholds; vectorise the tests per level.

## 5. Chunk, vertex and draw counts

With `split_factor` K = 2, a ring of ~66 chunks is drawn per level (the
annulus between 2 and 4 chunk sizes), not the ~10 first estimated:

| Camera (600 km planet) | Chunks drawn | Finest level |
|---|---|---|
| 10 m altitude | ~590–630 | 0 |
| 1 km | ~625 | 0 |
| 20 km | ~270–320 (to the horizon) | 5–6 |
| 200 km | ~120 | 9 |
| 3 radii (orbit) | ~14 | 12 |

Near a cube-face seam both faces contribute partial rings, about 25 % more
chunks than in the middle of a face. The planet's horizon is close (35 km
from 1 km up at 600 km), so the haze-limit cap is gone: chunks are culled at
the true horizon.

Each chunk is 4,481 vertices (36 bytes each) and ~9,000 triangles (grid +
two-sided skirts): ~2.5 M vertices and ~5 M triangles near the ground, in
~600 draw calls. The ocean's reflection pass now draws the terrain too (`reflected`
toggles it), which doubles all of that.

Levers, all in `TerrainSettings` ([Quality settings](quality.md)):

- `min_level` (MID drops one, LOW three): each dropped level removes ~40–60
  chunks.
- `split_factor` K = 1.5 instead of 2 (LOW's): ~40 chunks per level instead
  of ~66, with coarser triangles on screen (~16 px instead of ~12 px at the
  near edge of each ring).
- `reflected` off (LOW's).
- `max_distance` (a 3D cap; off by default, since from orbit it would cull
  the whole planet).

## 6. Memory

| Cache | Size | Per entry | Total |
|---|---|---|---|
| `store_size` (chunk data, RAM) | 2,000 | ~205 KB | ~410 MB |
| `node_cache_size` (uploaded geometry, GPU) | 1,200 | ~161 KB | ~190 MB |

(`TerrainSettings`, at HIGH; LOW keeps 1,200 / 800.)

Both caches are protected from evicting chunks the current selection depends
on (drawn chunks, the roots, every child of a split node) and chunks waiting
for upload, which is what fixed the terrain disappearing and reappearing.

With several bodies, the caps are maxima: below them a terrain keeps 3× what
its selection needs, at least 256 tiles. Only the body you are near holds
the full ~600 MB. A moon seen from the planet holds a few dozen tiles, and a
planet you leave releases its ground within a frame or two. That costs
regeneration if you come back, which is the same as arriving fresh. The
check adds about 0.5 ms to an LOD pass (4.2 ms in total at the demo start,
against 3.6 ms for the selector alone).

Options: drop `vertex_bytes` from stored chunks once uploaded (re-upload
would then need regeneration), half-precision normals, smaller caches.

## 7. Synchronous generation in queries — before game integration

`TerrainQuery` generates a missing chunk on the calling thread when
`generate=True`: ~25 ms per chunk, plus ~3 ms for a direct planet evaluation
when it falls back with no chunk at all. `sphere_contact` defaults to
`generate=True` (the demo's probe uses it).

For the game this must not happen per frame: bots' sensors far from the
camera would query chunks the renderer never generated. Options: query with
`generate=False` (uses the finest stored ancestor, so distant bots collide
with coarser ground), request missing chunks from the worker pool and answer
coarse until they arrive, or keep a separate low-resolution physics cache
around each bot.

Smaller point: each query call does an `np.unique` over chunk keys (~100 µs).
Batch queries per frame when many bots query at once.

## 8. Fragment shader

Per pixel, for each biome with weight ≥ 0.02 (usually 2–3, up to 6):
albedo+height at two scales and the normal/roughness/AO map, plus two side
projections for steep rock. Worst case is around 20 texture fetches, in
loops with branches (explicit gradients via `textureGrad`). Beyond ~4 km the
shader blends to each biome's mean colour and stops sampling.

Not measured on the GPU. Watch it on weaker GPUs or at high resolution;
the easy levers are fewer macro samples, no triplanar, and a shorter
texture distance (`TerrainSettings.detail_far`).

## 9. Initial fill and planet changes

- Startup or a preset/seed switch regenerates everything, coarse first: about
  630 chunks, ~6.5 s with 6 workers (offscreen run), during which the terrain
  sharpens progressively.
- Spawned workers re-import the demo script (and its Panda imports) on
  start: about a second before the first chunks arrive.
- `PlanetSpec.resolved()` computes the sea bias on the main thread (40,000
  directions, ~0.3 s) on each spec change: a visible hitch on N and F1–F4.
  Could move to a worker, or be cached per seed and preset.
- Uploads are cheap (0.2–0.4 ms per chunk, budget 6 per frame or 4 ms).

## 10. Descent from orbit

Speed in the demo grows with height above ground (`SPEED × h / 300 m`), so a
descent is exponential. With `SPEED` at 150 m/s, as measured below, that was
about 2 s per factor of e, 0.4 s boosted; it is 300 m/s now, which halves
both. Chunk demand is roughly constant per factor of altitude, about one ring
of ~66 chunks per halving of the altitude: at 150 m/s, ~16 chunks/s
unboosted, ~80/s boosted (twice that now), against the pool's ~50–67/s (§1).
Boosting all the way down therefore shows coarser ground for a moment near
the bottom (the coarse parent stays visible, never holes). The fixes are
those of §1 and §2, plus generating in-view chunks first.

Scripted descents from 3 radii to 300 m above the demo's coast (6 workers,
1280×720):

| Descent | Duration | Frame median / p95 | LOD pass median / p95 | Finest level on arrival | Fully refined |
|---|---|---|---|---|---|
| Unboosted | 18.5 s | 20 / 46 ms | 4.1 / 15 ms | 1 | on arrival |
| Boosted (5×) | 4.5 s | 20 / 45 ms | 2.7 / 6 ms | 7 | 8 s later |

The LOD pass ran every frame during these descents (every upload triggered
one), and each pass visits more nodes while many children are still missing:
hence its p95. It has since been throttled to every 3rd frame while only
uploads changed (§4). One frame of ~600 ms appears if the swell ocean has
never been drawn before the 15 km handover (its shaders compile then); the
demo starts on the ground, where they compile at startup.

Depth partitioning (`world_builder.depth`) culls and draws the scene twice
(far camera, then near camera). Panda's cull is C++; only chunks straddling
the split are drawn by both, which is a handful near the ground and none
from high up, where the split is far below the camera.

## 11. Several bodies

All bodies share one worker pool (`ChunkExecutor` channels), so a second
body adds no processes. Each runs its own LOD pass every frame: a moon
10,000 km away is 6–15 chunks and ~0.1–0.5 ms. Only bodies you are near
cost real time.

A bug worth remembering: a generator expression created inside a list
comprehension (`[(ch, (k for k in ch.wanted if k not in ch.pending)) for ch
in ...]`) looks `ch.pending` up late, as the *last* channel's. With two
channels, the planet's already-queued keys were submitted again on every
fill: the pool's queue grew without end and the planet stopped refining.
`workers._unsent` binds it properly, and
`test_channels_share_the_pool_without_duplicates` guards it.

Another: Panda's culling frustum (`Lens.make_bounds`) ends at
`1 / lens-far-limit`, 10,000 km by default, whatever the lens's far distance.
A planet seen from just beyond its moon (10,000 km away), or the moon from
beyond the planet, was culled whole while the depth range covered it; its
cloud deck and the sky's atmosphere, never culled, stayed on screen.
`SplitDepthCameras` now gives the far camera cull bounds to its true far
plane (`depth.frustum_bounds`); `test_frustum_bounds_reach_past_ten_thousand_km`
guards it.

## 12. Frame time in fast flight

`scripts/flight_bench.py`, fast flight 300 m over the demo coast, boosted
(~750 m/s as first written; ~1,500 m/s with today's 300 m/s base speed),
terrain refining all the way (~115 chunks/s generated, ~140 wanted):

| Step | fps | Demo update, median / p95 |
|---|---|---|
| Before (window, 50 Hz display) | 30 | 17.2 / 31.9 ms |
| Workers niced, physical cores − 1, LOD hot loop (window) | 48 (capped) | 7.8 / 13.5 ms |
| + per-frame uniforms written in place, throttled passes and HUD, cheap ground query (offscreen) | 89 | 4.7 / 8.4 ms |

Cruising at 3 km (~7.5 km/s as first written; ~15 km/s now) and descending
from orbit run at 400–800 fps offscreen.

What changed besides §1 and §4:

- **Per-frame shader inputs** (camera, planet centre, sun; since then the
  atmospheres' too) are `PTA` arrays bound once and written in place.
  Setting new values on the terrain root made a new render state per input
  per frame, for Panda to compose and garbage-collect.
- **The HUD** refreshes 5 times a second; its text geometry was rebuilt
  every frame (~1 ms).
- **The ground query under the camera** skips `np.unique` for a single point,
  and the triangle test converts only the 4 vertices it needs to float64
  (it converted all 4,225).

Measured and left alone:

- **Unpickling results:** 0.02 ms per chunk. The profile's time in the
  pool's result thread is blocking reads, not work.
- **Panda's state garbage collection rate:** lowering it gained ~3 %,
  within noise.

What remains per frame flying low: ~4.7 ms of Python (LOD passes ~1 ms,
uploads ~0.7 ms, the ocean and the rest) and ~6 ms of rendering, two cameras
plus the ocean's reflection pass.

## 13. Atmosphere, sky and clouds

Settled views, offscreen at 1280×720 (MSAA 4×), the demo's start area:

| View | Clouds on | Clouds off | Billboards |
|---|---|---|---|
| Start (1.3 km, deck overhead) | 166 fps | 226 fps | 185 k |
| Inside the deck (2.7 km) | 187 fps | 255 fps | 152 k |
| 20 km up | 415 fps | 601 fps | 62 k |
| Orbit (deck only) | 1079 fps | 1181 fps | 0 |

- **The atmosphere** costs 8 samples per pixel of terrain, sky and deck: a
  fraction of a millisecond on a GTX 1070.
- **Billboards evaluate it per vertex.** Per fragment, under the deck's
  overdraw, it took a flight skimming the slab to 4 fps; per vertex the same
  flight is GPU-cheap. Their own shading (4-octave density plus a 4-step sun
  march per fragment) is unchanged.
- **Cloud tiles** take ~20 ms each in the workers, alongside terrain chunks;
  the pool gives each free slot to the channel with the fewest in flight. The
  initial fill (88 cloud tiles and ~610 chunks) takes ~12–16 s with 3 workers,
  against ~6.5 s for terrain alone; clouds arrive within the first 4 s.
- Fast flight (§12) with clouds: 67 fps flying low (89 without), offscreen.

### Multisampling: the window was at ×16

The demo asked for 4 samples and got 16. Panda scores framebuffer configs
higher the more samples they have, and GLX offers every config with at least
the count asked for, so a GTX 1070 window gets 16 whatever count is
requested (2, 4 or 8: always 16). At 1600×900 that put the start view at
40 ms (25 fps) in the window, against 20 ms (the display's 50 Hz cap) with
an exact ×4 drawn into an offscreen buffer. Blending hundreds of thousands
of translucent billboards into 16 samples per pixel was most of the clouds'
cost.

The demo and the scripts now ask for **no multisampling** (`window_config`
in `demo_terrain.py`). An exact count would need the scene drawn into an
offscreen buffer (an FBO gets exactly what it asks for) and shown on a
full-screen quad; that was tried and works, and was left out for now.

### Measuring the GPU

An offscreen frame is not held back by the GPU: the same poses timed
identically at 1600×900 and 3200×1800. `terrain_shots.py` now waits for the
GPU after every timed frame (a one-pixel read back), so a frame is CPU plus
GPU in series, and the difference between two runs is the difference in
work. Its options: `--size`, `--offscreen`, `--quality
low|mid|high|ultra` (the whole world's preset), a `--<part>-quality` per
part over it (`--clouds-quality`, `--ocean-quality`, ...), `--no-clouds`, `--cloud-debug BITS` (cloud.frag's `uCloudDebug`: 1 no
sun march, 2 no atmosphere, 4 two octaves only, 8 flat colour), `--frames
N`. `flight_bench.py` takes the same.

### Cheaper clouds, and the weather

Settled frames, offscreen at 1600×900, no MSAA, waiting for the GPU (median
of 150; clouds off: ground 10.2 ms, in the deck 7.2 ms, 20 km up 5.3 ms).
The **cloud cost** is the frame time minus that.

| Version | Start (1.3 km) | Inside the deck (2.7 km) | 20 km up |
|---|---|---|---|
| Before, its own weather (185k / 152k / 62k billboards) | 5.1 ms | 5.9 ms | 2.0 ms |
| Before, uniform coverage 0.3 (82k / 80k / 31k) | 3.3 ms | 4.5 ms | 2.2 ms |
| New shaders, old overlap 2.5, uniform 0.3 (81k / 79k / 31k) | 3.2 ms | 4.0 ms | 1.5 ms |
| New, first-pass HIGH (overlap 2.0 / 1.5, octaves 4-3-3-2), uniform 0.3 (57k / 55k / 18k) | 2.1 ms | 2.8 ms | 1.2 ms |
| New, first-pass HIGH, the new weather (53k / 60k / 17k) | 1.7 ms | 3.1 ms | 1.2 ms |
| New, first-pass MID (23k / 26k / 7k) | 1.2 ms | 2.0 ms | 1.1 ms |

At equal coverage, the shader work (an exact early-out after one or two
octaves, fewer octaves in the far shells, 4 air samples per billboard vertex
instead of 8) saves 3–30 %, most where the far shells dominate; less overlap
saves as much again. The old weather happened to put twice the uniform
coverage over the start area, hence its 185k billboards.

What did not work: **the sun march per vertex** for the far shells. Marched
at a billboard's centre, every far billboard was lit flat (a white or blue
disc); at its corners, the corners sit in clear air and nothing was shaded.
The per-fragment march on every shell costs 0.2–0.9 ms in these views, so it
stays.

**HIGH restored.** Those HIGH settings (overlap 2.0 / 1.5, three octaves
from shell 1) looked lumpier and blander than before, so HIGH is back to
the old overlap (2.5) and keeps all four octaves to shell 2; the cheaper
settings are MID's. Measured on a busier machine (clouds off took 0.4–1.5 ms
longer than above): HIGH ~3.9 ms at the start, ~6.5 ms in the deck, ~1 ms at
20 km, about what it cost before.

The debug switches confirm where the rest goes: with no MSAA, no single cost
dominates. Overdraw is what is left, and the clouds' quality presets (F4 in the demo) trade it.

Fast flight (§12) is unchanged: 61 fps flying low offscreen (59 before),
CPU-bound. The weather map costs ~2 s of main-thread time when the clouds
are built (startup, a planet change), then is cached; each worker builds its
own copy once.

Savings found but not yet made are listed in the
[performance backlog](performance_backlog.md).

## 14. Chunks sooner, faster loading (#20)

New chunks showed up visibly late at good frame rates. Three causes, measured
on the same machine (3 workers, offscreen 1600×900, demo level):

- **Generation:** 85 % of a chunk's ~24 ms was numpy simplex noise (90k-160k
  evaluations a chunk, numpy's per-operation overhead). The octave sums are
  now a C library (`native/noise.c`, see [the native libraries](native_build.md)):
  the same noise (bit-identical per octave, sums within ~1e-12), ~8 ms a
  chunk. Pool throughput, all workers warm: 74 chunks/s before, 152 with the
  malloc fix (§2), 358 with the native noise too.
- **Showing what arrived:** a pass ran 3 frames after an upload only if
  another upload landed in that very frame; otherwise the chunk waited for the
  camera to move or the 15-frame fallback (up to 250 ms). Uploads now count
  until a pass shows them.
- **Order:** wanted chunks were sorted coarse first over the whole view, so
  horizon chunks went before the ground under the camera. Now ground nothing
  covers comes first (coarse first), then the chunks largest on screen
  (distance over size), those behind the view (`camera_forward`) as if 3x
  farther. Sorting costs ~0.1 ms a pass.

CPU-only refinement bench (the real pool, 60 Hz frames, no rendering):

| | Before | After |
|---|---|---|
| Initial fill | 11.1 s | 5.6 s |
| Jump to unseen ground: ground under the camera sharp | 3.8 s | 1.0 s |
| Same, everything loaded | 4.5 s | 2.4 s |
| Flight at 1,500 m/s, 300 m up: mean level under the camera | 0.77 | 0.00 |

With numpy noise and the rest of the changes: 2.1 s to sharp, 4.9 s loaded.

Start-up of the demo (clouds on), `TerrainDemo` built offscreen:

| | Before | After |
|---|---|---|
| First frame | 6.9 s | 3.8 s |
| Ground and clouds loaded | 22.5 s | 12.0 s |

What changed there: the workers start with the pool (their imports overlap
the caller's set-up; they used to spawn at the first request), the Milky Way
glow is built once at the size the sky uses (2.6 s at 256, before the noise
made it 0.4 s), the sea level evaluates only the continental field (and the
noise), empty paint channels are not blurred, placeholder textures are made
only for missing maps, and `scipy.interpolate` (0.11 s a process) is
replaced by a numpy PCHIP. The weather map still takes 0.6 s in each worker
(1.3 s before), once.

Frames (`flight_bench.py --offscreen`, `terrain_shots.py`, waiting for the
GPU each frame):

| | Before | After |
|---|---|---|
| Low flight, boosted: fps / frame p95 | 50 / 30 ms | 62 / 23 ms |
| Low flight: chunks generated per second | 102 | 191 |
| Start pose (ground) | 14.0 ms | 11.8 ms |
| Coast (2 km up) | 13.8 ms | 12.3 ms |
| Inside the clouds | 12.5 ms | 11.4 ms |
| Sunset (600 m up) | 15.1 ms | 12.1 ms |
| 20 km up | 7.5 ms | 6.5 ms |
| Beach | 17.3 ms | 14.9 ms |
| Orbit | 2.8 ms | 2.3 ms |

From the backlog's free savings, all four made: the sky after the terrain
(its own bin, depth-tested), the reflection camera at 1.1x the main
field of view (no clamped texels measured with `uDebugMode == 5`; 1.0 clamps
2 % of the beach's water), the sun's depth for Rayleigh and Mie in one call,
and fewer atmosphere samples on short sight lines (full count from 2 Mie
scale heights of climb or 32 of length; at least 2): within one 8-bit level
of the 8-sample image, 125 → 145 fps at the coast without clouds. On the CPU,
the ocean's per-frame inputs are bound arrays, the clouds' pixel angle and
sky light are set only when they change, their re-sort scores tiles without
numpy, and the store is trimmed once a frame instead of once per chunk
(~1 ms a frame in fast flight).

Tried and left out:

- **Look-ahead** (generating a level or two below the chunks being waited
  for): no measurable gain at 60 or 30 Hz once a chunk takes 8 ms; a level
  now waits on the pass and upload cadence, not on generation.
- **A deeper queue** (4 tasks a worker instead of 2) and **a pass every
  frame while uploading**: within noise either way.
- **`-march=native`** for the noise: ~15 % more, not worth per-CPU builds.

## 15. Clouds and loading, second round

The clouds were then most of what was left of the loading time: with them the
demo's first frame came 1.4 s later and its ground 3.6 s later.

- **Cloud noise in C.** `wb_volume_fbm` (`native/noise.c`) samples the 64³
  volume as `sample_volume` does (float32 corner differences, float64 blends,
  the GPU's half-texel offset), bit for bit, 7x faster. Tiles, the column
  peaks, the deck light and the sun march all use it.
- **Tile batches sized to the tile.** A tile needs about its slab volume over
  a billboard's candidates whatever the accept rate, so it draws that many
  (at least 2,048) instead of 20,000. Tiles per shell: 20 / 17 / 18 / 15 ms
  before, 7.7 / 4.1 / 2.4 / 1.4 ms after. Their billboards differ (other
  random draws), their statistics do not.
- **The weather map made once.** It is made in the pool while the caller goes
  on, then shared with the workers (`workers.SharedArray`, shared memory)
  instead of rebuilt by each one (0.7 s each). Its absence had stalled the
  start of the fill: 27 chunks in the first 2.5 s, now ~110.
- **The deck light in the pool too**; the clouds finish their set-up in
  `update()` once both are in (`CloudVolume._finish`).
- **The glow in the pool** (`Starfield.prefetch_glow`) while the terrains are
  built; the placeholder textures' noise on half spectra with cached filters
  (817 → 596 ms, the same bytes); the atlas's mean alpha with numpy.

Demo start-up (clouds on): first frame 3.7 s → 2.1 s, everything loaded
12.0 s → 9.2 s. Without clouds: 1.7 s and 8.2 s. What remains of the fill is
generation in a busy machine: ~900 chunks at ~150 a second with 3 workers
beside the renderer (chunks take ~13 ms there against 8 idle).

## 16. The sea on the terrain's chunks

The sea moved from its own mesh under the camera onto the terrain's chunks
(see [Sea](ocean.md)). Measured against the clouds round (§15) with
`terrain_shots.py --offscreen`, the two trees run in turn, twice each, every
frame waiting for the GPU. A first version was 1-4 ms slower wherever the
sea showed. Swapping pieces out one at a time (a scratch script hiding the
reflection, the near look, the wet chunks) found three causes:

- **The swell evaluated per pixel.** The near look recomputed the 3D swell
  field and its gradient for every water pixel: 32 lattice hashes of five
  32-bit integer multiplies each, 0.8-1.6 ms on a GTX 1070. The vertices
  already compute the field to raise the water; they now pass it and its
  gradient on (`vSwell`). The shading then follows the drawn water, and the
  images match.
- **All-water chunks in the reflection.** Panda culls a node wholly behind
  the mirror's clip plane. Seabed bounds were below it; the water's are not,
  so the mirror pass drew every all-water chunk only to clip each vertex
  (about +1 ms). Those chunks are now hidden from the reflection camera.
- **Draw order.** Panda's opaque bin sorts by state, which chunks share, so
  they drew in no useful order and hidden water was shaded. The terrain now
  draws front to back in its own bin (`TERRAIN_BIN`): 1.0-1.6 ms less at the
  sea poses, and some on land too.

| Pose | Before (§15) | After |
|---|---|---|
| Start pose (ground) | 11.3 ms | 10.7 ms |
| Beach (40 m) | 13.5 ms | 14.7 ms |
| Coast (2 km) | 12.7 ms | 12.7 ms |
| Sunset (600 m) | 12.0 ms | 12.6 ms |
| Night sea (3 m) | 11.8 ms | 12.5 ms |
| 13 km | 8.8 ms | 9.3 ms |
| 17 km | 5.8 ms | 8.3 ms |
| 20 km | 6.1 ms | 6.2 ms |
| Orbit | 2.2 ms | 2.4 ms |

What remains:

- **The near look costs a little more low over the sea** (0.5-1.2 ms). Part
  of it is the near look's shader variant, larger for the land in coastal
  chunks too (0.2-0.7 ms, measured with its share at 0); the rest was not
  broken down.
- **Between 12 and 18 km the near look fades out** instead of switching at
  15 km, so the reflection pass and the near look run up to 18 km (2.5 ms at
  17 km, where the old ocean was hidden). `OceanLook.near_band` sets the
  band; a lower one moves that cost down.

Flights: the low flight 15.3 → 14.5 ms median (`flight_bench.py`), and the
descent through the band has no spike (worst frame 21-24 ms, 38-43 before,
the old handover's). The descent used to stop only above dry land: it
tested `demo._ground` for truth, and the old query clamped the sea to
exactly 0. Start-up is unchanged (first frame 1.7 s, everything loaded
8.7-9.0 s).

Gone with the separate ocean: its 324k-vertex mesh, drawn by both cameras;
the seabed bakes (15-36 ms each, half of it holding the GIL); the 600 ms
shader compile the first time the old ocean showed.

## 17. The LOD pass in C

In fast low flight the LOD pass ran nearly every frame on the main thread:
~470 nodes visited and ~870 measured per pass, in Python floats, 2.4 ms at
the start pose. `lod.c` runs the same pass, operation for operation
(`tests/test_native_lod.py` requires identical selections along flights).
It keeps its own copy of the chunks' bounds and readiness, which `Terrain`
updates where they change, so a pass calls nothing back.

- The traversal: 1.03 ms in Python, 0.023 ms in C, for a pass over 450
  drawn chunks. Turning the results into Python keys costs more than the
  pass (0.15 ms, with `zip` over the columns).
- Around it, `Terrain._select` marked every kept chunk (~1,500) as recently
  used in two LRU dicts on every pass. Kept chunks are never evicted, so
  only the ones let go are marked now, when they are let go: the same order
  for the chunks that can be evicted.

The pass at the static poses: 2.4 ms → 0.5 ms. Low flight
(`flight_bench.py --flight low`, two runs each, the same tree with the
Python selector patched in): frame median 14.5 → 13.0 ms, the demo's update
8.4 → 6.4 ms, refinement lag 345 → 270 ms (median). The machine had just
been rebooted and its GPU held a low power state (P5), which slows the GPU
side of every frame; these numbers are CPU-side and compared within the
same session.

## 18. A chunk's geometry in C

A chunk took ~8 ms in a worker: 4.3 ms of native noise and ~3.5 ms of numpy
in ~100 small steps. Only the steps that shape the grid went native
(`chunk.c`); the planet's rules (curves and masks around the noise, biome
weights, stamps, paints) stay in Python, so they are written once. Timed per
step, per chunk:

| Step | numpy | Native |
|---|---|---|
| Face directions (and the meshgrid) | 0.41 ms | one tangent per row and column |
| Normals, and the slope at the biome scale | 0.57 ms | one pass |
| Vertex packing, the sea box, the bytes | 0.60 ms | one pass |

Each follows its numpy expression's order of operations: on this machine the
chunks come out identical, a coastal one included (its sea-lifted box and
skirts). A chunk: 8.0 → 6.7 ms median (70 chunks, levels 0-6). In low flight
(after §17): 253 chunks a second instead of ~225, the backlog 94 instead of
~130, the refinement lag 214 ms instead of ~270 (median), 980 ms instead of
1,400-1,800 (p95).

## 19. The slowest frames

The mean was down to what the GPU and the main thread's steady work cost;
this round targets the slowest 5 % of frames. A feature is kept only if the
p95 drops and the mean frame and update do not rise.

`flight_bench.py --trace` times the main thread's parts in every frame and
prints what the frames at or over the p95 spend beyond the median ones;
`--no-hud` skips the demo's HUD (~5 ms every 0.2 s, out of scope). Low
flight before this round, without the HUD: mean 13.5 ms, median 13.2, p95
18-19, max 36-47.

| Part | In slow frames | In median frames | Share of slow frames |
|---|---|---|---|
| `Terrain._rebase` | 3.8 ms | 0.1 ms | 50 % |
| Panda's `renderFrame` | 6.8 ms | 5.5 ms | all |
| The clouds' tile uploads | 1.0 ms | 0.1 ms | 30 % |
| The clouds' re-sort | 2.1 ms | 1.5 ms | all |
| Panda's state GC | 1.7 ms | 1.2 ms | all |

### Re-basing by cells

Every 4 km of flight the origin snaps and the terrain re-placed every
uploaded tile (~1,500 `set_pos`, drawn or not): a 5-13 ms frame. Tiles now
hang under cells about 32 km wide (the coarse ones under one cell at the
planet's centre), and a snap moves the cells only (see
[Terrain](terrain.md), precision).

| Low flight, two runs each | Before | Cells |
|---|---|---|
| Frame mean | 13.4-13.5 ms | 13.3 ms |
| Frame p95 | 18.5-18.9 ms | 17.4 ms |
| Frame max | 35-38 ms | 27-33 ms |
| Update mean / p95 | 6.6-6.7 / 10.6-11.0 ms | 6.4-6.5 / 9.3-10.0 ms |

`Terrain._rebase` left the slow frames (+3.2 ms → +0.02). The clouds'
re-basing, which snapped in the same frames, went with it (+0.36 → +0.06).

### The clouds' re-sort in C

The planet clouds re-sort the stalest four shown tiles back to front every
frame in flight. Scoring the ~86 shown tiles took 0.15 ms; the sorting 1.3 ms
(numpy's argsort and index arithmetic, ~130 µs per thousand billboards), up
to 7.9 ms for big tiles. `billboards.c` computes the distances, radix-sorts
them and writes the triangle indices in one pass. It sorts stably over the
tile's previous order, as the numpy fallback now does, so the two give the
same buffer (tested). The tiles re-sorted and when are unchanged.

| Low flight, two runs each | Before | Native re-sort |
|---|---|---|
| Frame mean | 13.2-13.4 ms | 12.4-12.6 ms |
| Frame p95 | 17.6-17.8 ms | 16.3-16.6 ms |
| Frame max | 30-33 ms | 28-29 ms |
| Update mean | 6.4-6.5 ms | 5.7 ms |
| `_resort` (median frame) | 1.45 ms | 0.75 ms |

What remains is the four tiles' sorts at 54 µs per thousand billboards in
flight (17 in isolation: the particles are cold in the cache by then), and
the scoring.

### The clouds' tiles built in C

A tile's upload built its vertices in numpy (four per billboard, then a
copy into Panda's buffer): 0.6 ms for 1,900 billboards, 3.6 ms for 8,200,
up to three tiles a frame. Building them in the workers would have sent five
times the bytes through the pipe and still copied them on the main thread.
`wb_billboard_vertices` writes them straight into Panda's buffer instead,
with the tile's bounds and reach in the same pass, and the index buffer is
sized directly (`add_next_vertices` cost 0.15 ms of it). A tile: 0.3 ms /
0.75 ms / 1.0 ms for 1,900 / 5,700 / 8,200 billboards; numpy and C give the
same bytes (tested).

| Low flight, four runs each | Before | Native build |
|---|---|---|
| Frame mean | 12.3-12.5 ms | 12.1-12.5 ms |
| Frame p95 | 15.9-16.5 ms (16.25) | 15.6-16.7 ms (15.9) |
| Update mean / p95 | 5.7 / 8.5 ms | 5.6 / 7.8 ms |
| The clouds' upload in slow frames | +1.1 to +1.3 ms | +0.15 to +0.45 ms |

### What `renderFrame` spends

After the three changes above, Panda's `renderFrame` was the slow frames'
largest part (+2.5-3 ms over median frames). Uploads and newly shown chunks
explain little of it (frames with uploads render ~0.7 ms slower, correlation
0.25). PStats (`want-pstats`, recorded as text by `text-stats -r`) shows the
main thread's cull and draw growing in the slow frames: cull +1.3 ms (+0.9
of it the reflection's), draw +2.6 (setting state +1.2, shader inputs +0.7,
vertex buffer transfers +0.3), and less of it without clouds (their tiles
add draw calls).

**The low flight is CPU-bound.** At 2560×1440, 2.56 times the pixels, its
frames take as long as at 1600×900 (12.5-12.7 ms mean). The GPU's own timer
queries (`pstats-gpu-timing`) looked otherwise, the far pass "taking" 9.3 ms
in median frames and 13.8 in slow ones, but they measure from one query to
the next on the GPU's timeline, waits for the CPU's commands included. So
what is left of the slow frames is CPU work per drawn node (state and
shader inputs per chunk and per cloud tile), not GPU work; see the log of
remaining paths in the [performance backlog](performance_backlog.md).

### Half-resolution clouds (a setting, off by default)

`CloudSettings.half_resolution` draws the clouds at half the view's
resolution (`clouds.half_res`, see [Atmosphere, sky and planet
clouds](atmosphere.md)). Their GPU cost is mostly per pixel: with clouds
minus without, at 1600×900 and at 800×450 (a quarter of the pixels):

| Pose | 1600×900 | 800×450 |
|---|---|---|
| Start (ground) | 3.6 ms | 1.6 ms |
| Coast | 4.5 ms | 2.4 ms |
| Inside the clouds | 4.6 ms | 1.9 ms |
| Sunset | 4.5 ms | 1.6 ms |
| 20 km up | 1.7 ms | 0.7 ms |

The layers (one per split range) need the scene's depth, drawn again into
each at half resolution (the terrain with a depth-only fragment stage).
Frames GPU-synced (`terrain_shots.py`, CPU and GPU in series), full against
half resolution, two runs each:

| Pose | Full | Half |
|---|---|---|
| Start (ground) | 10.7 ms | 9.9 ms |
| Coast | 11.9 ms | 10.4 ms |
| Inside the clouds | 11.0 ms | 9.3 ms |
| Sunset | 12.3 ms | 10.6 ms |
| Beach | 14.9 ms | 13.5 ms |
| 20 km up | 6.1 ms | 5.8 ms |

But the two layer cameras cull and draw the scene again on the main thread:
about 1.2 ms a frame (cull 0.5 + 0.25 ms, draw 0.3 + 0.15 ms, and their
state). The low flight here is CPU-bound, so there it loses: mean 12.45 →
13.2 ms, p95 16.35 → 17.15 ms, at 1600×900 and at 2560×1440 alike. It pays
where the GPU is the limit (a weaker GPU, heavier GPU settings), which is
why it is a setting, off in every preset. The images match but for softer
cloud edges against the ground's silhouettes (up to a pixel).

### One state for every chunk

Every chunk node carried four shader inputs of its own (its centre and
swell band, uv origin and step, skirt depth, level, face), so Panda drew
each in a state of its own: binding its inputs, changing state. Setting the
four once on the terrain's root instead (wrong images, to measure) cut the
low flight's mean frame by 2.4 ms and its p95 by 3.

The values now live in a table: a float texture on the terrain's root, one
slot per uploaded chunk, three texels each (`Terrain._make_chunk_table`). Each
vertex names its chunk's slot in `biome_b`'s two spare bytes, written at
upload (the skirt flag moved to the top bit), and `terrain.vert` fetches
the values (`texelFetch`); the fragment stage gets the level and the step
as flat varyings. They are the same float32 values, so the images are the
same (the ground identical pixel for pixel; a test reads the table back
from the GPU). Panda keeps an RGBA texture's RAM image as BGRA: written in
RGBA, every value came out with x and z swapped.

| Low flight | Before (2 runs) | Chunk table (4 runs) |
|---|---|---|
| Frame mean | 12.4-12.5 ms | 10.5-10.6 ms (11.4 in one run) |
| Frame p95 | 15.9-16.2 ms | 14.1-14.2 ms (16.4 in one run) |
| Frame max | 26 ms | 23-28 ms (108 once) |
| Refinement lag (median) | 183-196 ms | 141-192 ms |

One run of four had a 108 ms frame and a higher p95; the three others did
not. A frame with uploads renders 0.6-0.8 ms slower than one without (its
new chunks' vertex buffers, and the whole table re-sent: Panda 1.10 has no
partial texture update).

### No transform cache

Panda's state garbage collection ran every frame: 1.2 ms, 0.6 ms once the
chunk table left only 65 render states. What remained were transforms: the
world moves around the camera, so every drawn node's net transform is new
every frame, once per camera, and Panda's cache of composed transforms only
filled up (2,200 entries) for the collection to empty. Variants, low
flight:

| Variant | Frame mean | p95 | Collection |
|---|---|---|---|
| Default | 10.7 ms | 14.3 ms | 0.6 ms a frame |
| `garbage-collect-states-rate 0.25` | 10.9 ms | 14.4 ms | 0.8 ms (5,800 states pile up) |
| Collection off, by hand every 30 frames | 10.5 ms | 14.0 ms | nothing left to collect |
| `transform-cache false` | 9.6 ms | 13.1 ms | none (no cache) |

`transform-cache false`, two more runs each: mean 10.65 → 9.6 ms, p95
14.25 → 13.05 ms. Cruise and descent (three runs each) are unchanged, as
are still views (GPU-synced poses within ±0.3 ms, the bench's noise): a
still camera is the only case where a composed transform comes back.

It is a process-wide Panda setting, so the library does not set it: the
demo and the scripts do (`demo_terrain.window_config`), and a game drawing
the world should load it too ([A whole world](world.md)).

### Evictions that walk past nothing

The store and the uploaded nodes are LRU dicts, trimmed each frame past
their resident limit, never a chunk the selection keeps. Since chunks are
marked as recently used only when they leave the selection (§17), the kept
ones sat at the front of both dicts, and every trim walked past them: 420 to
640 of them a call, 0.42 ms a frame each in median frames, up to 0.6 ms. A
trim now moves a kept chunk it meets to the end (it is in use: as recently
used as any), so the front holds the evictable ones. The chunk table also
copies only the slots an upload pass filled, already in BGRA order, instead
of the whole table twice.

| Low flight, two runs each | Before | After |
|---|---|---|
| Frame mean | 9.7-9.8 ms | 8.6-8.7 ms |
| Frame p95 | 13.2-13.4 ms | 11.6-11.7 ms |
| Update mean | 5.5 ms | 4.4-4.5 ms |
| Refinement lag (median) | 144-153 ms | 110-129 ms |
| `_evict_store` / `_evict_nodes` / `_upload` (median frame) | 0.42 / 0.42 / 1.28 ms | 0.04 / 0.07 / 0.67 ms |

Smaller upload budgets (3 chunks and 2 ms a frame, or 4 and 3, against 6
and 4) did not lower the p95 any more (11.7-13.6 ms against 11.9), with
uploads this cheap: the defaults stay.

### `Sea.update`

0.56 ms a frame low over the sea: the swell's drift 0.26 ms, placing the
nadir frame 0.15, the mirror 0.07, the light 0.06, the wave phases 0.04
plus a 64-entry write loop. The swell layers' axes never change (the wind
at the planet's reference point), so their Rodrigues terms are computed once
(`ocean.SwellDrift`; `swell_rotations` is built on it and gives the same
rotations): a frame turns two angles, and nothing when the clock has not
moved. The nadir frame is placed in plain floats (numpy's per-call cost
dominates on 3-vectors), the phases are written through one view of their
array, and the reflection's fov, its pixel angle and the water's light are
set only when they change.

| Low flight, two runs each | Before | After |
|---|---|---|
| Frame mean | 8.8-9.0 ms | 8.5-8.6 ms |
| Frame p95 | 12.2-12.6 ms | 11.9-12.0 ms |
| Update mean | 4.7 ms | 4.3 ms |
| `Sea.update` (median frame) | 0.47 ms | 0.20 ms |

## 20. Raymarched clouds (#45)

`CloudSettings.renderer = "march"` ([the raymarch](clouds.md#the-raymarch))
against the billboards. Everything here was measured on a weak GPU: an Intel
UHD Graphics under WSL2, through Mesa's D3D12 driver
(`GALLIUM_DRIVER=d3d12`, GL 4.1; without it Mesa draws on the CPU). Scenes
are drawn offscreen at 1600×900, each frame waiting for the GPU, with the
clouds alone (`scripts/cloud_bench.py`: the air, the sky, the cirrus, no
terrain). The **cloud cost** is the frame with the clouds minus without.
Timings on this machine drift by 1-2 ms from run to run (the clouds-off
frame alone moves between 3 and 9 ms), so only same-session numbers are
compared.

**The billboards first** (the reference before any change, one session):

| Pose | LOW | MID | HIGH | HIGH, half resolution |
|---|---|---|---|---|
| Below (300 m) | 2.6 ms | 3.6 ms | 6.8 ms | 3.3 ms |
| Start (1.3 km) | 3.6 ms | 5.4 ms | 6.1 ms | 4.9 ms |
| Inside the slab (2.7 km) | 5.5 ms | 12.0 ms | 15.0 ms | 7.4 ms |
| Above (4.5 km) | 4.2 ms | 6.0 ms | 8.1 ms | 4.8 ms |
| 20 km up | 2.7 ms | 3.7 ms | 3.9 ms | 2.5 ms |
| Orbit (deck only) | 7.0 ms | 8.4 ms | 6.2 ms | 5.6 ms |
| Sunset | 4.1 ms | 6.7 ms | 8.1 ms | 6.1 ms |

**Both renderers, one session** (the march without longer steps in clear
air, see below):

| Pose | Billboards HIGH | March HIGH |
|---|---|---|
| Below | 6.6 ms | 9.3 ms |
| Start | 6.2 ms | 13.2 ms |
| Inside the slab | 17.0 ms | 16.7 ms |
| Above | 9.0 ms | 11.4 ms |
| 20 km up | 2.6 ms | 4.7 ms |
| Orbit | 5.4 ms | 4.4 ms |
| Sunset | 8.0 ms | 8.4 ms |
| A model in a dense cloud | 43.4 ms | 17.3 ms |

Over open sky the march costs 1.3-2 times the billboards; inside the slab
the same; in a dense cloud two-fifths. The billboards' cost grows with the
cloud's volume on screen (their overdraw); the march's with the air its rays
cross, cloud or not, until the first opaque cloud. Its rays take 33-47
steps on average (debug view 16), many reaching the 64-step budget.

Where its time goes (one-step marches and passes switched off):

- **Fixed, ~3-4 ms**: per range, a depth pass of the scene, the march's
  per-pixel set-up (the ray, the slab's spheres, the air once per pixel),
  the full-resolution resolve and the composite.
- **The near range, ~1-2 ms** when the slab reaches into it; it resolves at
  the march's resolution (its clouds are close and soft), which roughly
  halved it.
- **The steps**: the rest.

**Flicker, and what it cost to remove.** A first version crossed clear air
in up to four-times-longer steps, and read the weather once per kilometre:
the start pose's frame went from 15.7 to 10.8 ms. But a step longer than the
finest features samples thin cloud by chance, which the per-frame jitter
changes: thin cloud blinked and edges came out ragged. Measured as each
pixel's brightness spread over 16 still frames (`cloud_bench.py` poses,
8-bit steps, over the clouds' pixels; the billboards: 0):

| | Mean | Worst 1 % | Cloud on screen |
|---|---|---|---|
| Longer steps in clear air | 2.7 | 17.7 | 15.7 % |
| Without them | 1.0 | 5.4 | 19.4 % |
| Without them, the jitter frozen too | 0.4 | 4.3 | 19.4 % |

So the longer steps were also missing a fifth of the clouds' area. Two
exact ways to skip clear air were tried and gave nothing: each sample's own
margin under the threshold over the field's steepest change
(`noise_slope`), and a ceiling of the fbm per 500 m cell from the texels
each octave's filter can touch (`fbm_ceiling`). The fbm is narrow (99 % of
it between 0.25 and 0.70), the threshold sits at 0.55, and the two fine
octaves can each peak anywhere in a cell: the ceiling stayed above the
threshold in all but 0.004 % of the cells, and the march proved 0 % of its
steps clear. What is left of the flicker is the budget lengthening the
steps of long rays, and the jitter.

What stayed within the noise: a 2 × 2 neighbourhood for the resolve's
clamp instead of 3 × 3, a half-float distance instead of a 32-bit one, 4
air samples per pixel instead of 8, no sun march, no air.

## 21. Genera (#45)

Stratus, congestus and cumulonimbus by billboards ([genera](clouds.md#genera)),
`cloud_bench.py --genera`, on the same weak GPU and set-up as §20 (one
session, the cloud cost):

| Pose | Cloud cost | Billboards |
|---|---|---|
| Cumulus only, below / start / inside / above | 7.1 / 10.3 / 21.5 / 11.4 ms | 55-57 k |
| Cumulonimbus 40 km away, from the ground | 9.3 ms | 66 k |
| From 20 km up, 50 km away | 7.2 ms | 55 k |
| Between the towers (3.5 km up, in the congestus) | 71.7 ms | 87 k |
| Under the anvil | 57.7 ms | 81 k |
| Inside the congestus | 35.3 ms | 78 k |
| Above the stratus deck | 54.4 ms | 118 k |
| Under it | 63.3 ms | 155 k |
| Orbit | 9.1 ms | 0 |

Far away the genera cost what the cumulus do. Near the big clouds they cost
3-7 times as much: their volume on screen, which billboards pay for in
overdraw. Without the billboards growing with the depth (2.5 times at
depth 1) a cumulonimbus would take some 15 times more of them. All the genera
load within 25 s (their tiles, the maps by depth ~1 s). In the full world
(`terrain_shots.py --genera`): 52 ms at the ground pose, 57 ms at the coast.
Making this affordable is the next phase.

## Not covered

The sea's parts are measured in §16; its reflection pass re-renders the
scene (terrain, sky and clouds; M takes the terrain out of it), measured as a
whole in the [performance backlog](performance_backlog.md).
