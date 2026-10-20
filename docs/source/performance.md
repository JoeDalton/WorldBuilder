# Performance notes (terrain)

Known computational costs and open concerns for `world_builder.terrain`, with
the measurements behind them. Frame rate in the demo is currently high (GTX
1070, i7-6700K: 4 cores / 8 threads), so nothing here is urgent. This page
records where the time goes, what was tried, and what to look at first if it
becomes a problem.

Measured with `python -m world_builder.terrain.bench`, headless demo runs,
and `python scripts/flight_bench.py` (frame times in scripted fast flights,
`--profile` for the main thread, `--offscreen` to avoid the display's refresh
cap or throttling).
CPU timings on this machine are noisy (frequency scaling): treat them as
±30 %. Since the move to the cube-sphere (600 km planet by default), the
figures in §3–5 and §10 are re-measured; §1, §2 and §6–9 still hold as written.

## Summary

| # | Concern | Where | Impact today | Status |
|---|---|---|---|---|
| 1 | Worker pool throughput vs the render loop | `workers.py` | Too many workers slowed the frame | Default: physical cores − 1, lower priority |
| 2 | glibc returns numpy temporaries to the OS | worker processes | Workers ~40 ms/chunk instead of ~24 | Fix found, **not applied** |
| 3 | numpy per-octave cost | `noise.py`, `planet.py` | ~25 ms per chunk | Accepted (numba not wanted) |
| 4 | LOD pass in Python | `quadtree.py` | ~1 ms per pass, not every frame | Optimised |
| 5 | Chunk and triangle counts | `quadtree.py` (K=2) | ~600 draws, ~5 M triangles, twice with the reflection | Fine on desktop GPUs |
| 6 | Memory held by the caches | `terrain.py` | ~410 MB RAM, ~190 MB GPU per planet you are near | Fine on desktop |
| 7 | Synchronous generation in queries | `query.py` | 25 ms stall per missing chunk | **Must handle before game integration** |
| 8 | Fragment shader texture samples | `terrain.frag` | Up to ~20 samples per pixel | Fine so far |
| 9 | Initial fill and spec changes | `terrain.py`, `planet.py` | Several seconds of blur; 0.3 s main-thread hitch | Accepted |
| 10 | Descent from orbit | `workers.py`, `depth.py` | Demand above the pool's ceiling when boosting; split-depth draws twice | Accepted (coarse, never holes) |
| 12 | Frame time in fast flight | demo main thread | was ~30 fps flying low, boosted | Fixed: 89 fps offscreen (§12) |

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

**Not applied in the code.** To apply it, set these in `os.environ` before
the pool spawns its workers (spawned processes inherit the environment at
spawn time). Linux/glibc only; harmless elsewhere.

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
~600 draw calls. The ocean's reflection pass now draws the terrain too (M
toggles it), which doubles all of that.

Levers, all already exposed:

- `min_level` (P in the demo): each dropped level removes ~40–60 chunks.
- K = 1.5 instead of 2: ~40 chunks per level instead of ~66, with coarser
  triangles on screen (~16 px instead of ~12 px at the near edge of each
  ring).
- `set_reflected(False)` (M).
- `max_distance` (a 3D cap; off by default, since from orbit it would cull
  the whole planet).

## 6. Memory

| Cache | Size | Per entry | Total |
|---|---|---|---|
| `STORE_SIZE` (chunk data, RAM) | 2,000 | ~205 KB | ~410 MB |
| `NODE_CACHE_SIZE` (uploaded geometry, GPU) | 1,200 | ~161 KB | ~190 MB |

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
`uDetailFar`.

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

Speed in the demo grows with height above ground (`150 m/s × h / 300 m`), so
a descent is exponential: about 2 s per factor of e, 0.4 s boosted. Chunk
demand is roughly constant per factor of altitude, about one ring of ~66
chunks per halving of the altitude: ~16 chunks/s unboosted, ~80/s boosted,
against the pool's ~50–67/s (§1). Boosting all the way down therefore shows
coarser ground for a moment near the bottom (the coarse parent stays visible,
never holes). The fixes are those of §1 and §2, plus generating in-view
chunks first.

Scripted descents from 3 radii to 300 m above the demo's coast (6 workers,
1280×720):

| Descent | Duration | Frame median / p95 | LOD pass median / p95 | Finest level on arrival | Fully refined |
|---|---|---|---|---|---|
| Unboosted | 18.5 s | 20 / 46 ms | 4.1 / 15 ms | 1 | on arrival |
| Boosted (5×) | 4.5 s | 20 / 45 ms | 2.7 / 6 ms | 7 | 8 s later |

The LOD pass runs every frame during a descent (every upload triggers one),
and each pass visits more nodes while many children are still missing: hence
its p95. Throttling it to every other frame while only uploads changed is the
first lever (§4). One frame of ~600 ms appears if the swell ocean has never
been drawn before the 15 km handover (its shaders compile then); the demo
starts on the ground, where they compile at startup.

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

## 12. Frame time in fast flight

`scripts/flight_bench.py`, fast flight 300 m over the demo coast, boosted
(~750 m/s), terrain refining all the way (~115 chunks/s generated, ~140
wanted):

| Step | fps | Demo update, median / p95 |
|---|---|---|
| Before (window, 50 Hz display) | 30 | 17.2 / 31.9 ms |
| Workers niced, physical cores − 1, LOD hot loop (window) | 48 (capped) | 7.8 / 13.5 ms |
| + per-frame uniforms written in place, throttled passes and HUD, cheap ground query (offscreen) | 89 | 4.7 / 8.4 ms |

Cruising at 3 km (~7.5 km/s) and descending from orbit run at 400–800 fps
offscreen.

What changed besides §1 and §4:

- **Per-frame shader inputs** (camera, planet centre, haze shell, sun) are
  `PTA` arrays bound once and written in place. Setting new values on the
  terrain root made a new render state per input per frame, for Panda to
  compose and garbage-collect.
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

## Not covered

The ocean and clouds were moved from SpaceFlight unchanged and were not
profiled here. The clouds' profiling and an open alternative design are in
`world_builder/clouds/view_aligned_slicing.md`; the ocean's main cost is its
reflection pass, which re-renders the scene (terrain included, unless M turns
it off).
