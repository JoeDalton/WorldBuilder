# Performance backlog

Frame-time savings found but not made. Read this before optimising again:
where the frame stands, the candidates and what each would cost. The history
of what was done is in [performance](performance.md). Last cleaned
2026-11-27, after the frame-cost round (performance.md §38-41): the
billboard clouds, the split depth cameras and the clouds' depth passes are
gone, and so are the items about them.

## How it is measured

- **The demo's frame, as a game runs it:** `scripts/frame_profile.py
  --offscreen --pipelined` (no per-frame GPU sync: CPU and GPU overlap).
  Its default mode syncs every frame, so its frame is the CPU's step plus the
  GPU's frame: it ranks GPU parts but overstates the frame. Compare the two
  to tell which side bounds it. `--terrain-passes` gives the terrain's cost
  per camera.
- **A game's frame:** SpaceFlight's `scripts/bench_mission3.py` (each frame
  waits for the GPU; `--size "1280 720"` against 2560 x 1440 tells the CPU's
  share from the GPU's; `--profile` for cProfile, `--pstats` for Panda's
  timers).
- **In flight:** `scripts/flight_bench.py --offscreen --flight low --trace
  --no-hud` times the main thread's parts in every frame and what the
  slowest 5 % of frames spend beyond the median ones.
- **Panda's own timers:** PStats, recorded as text with the venv's
  `text-stats -p PORT -r -o FILE` and `want-pstats 1` in the PRC. Its GPU
  timer queries include the GPU waiting for the CPU's commands.
- Run two trees in turn, at least twice each: run-to-run noise is 0.3-0.5 ms
  (0.8 ms in the game). Any other load (a browser) spoils a run. Check the GPU
  is at full clocks during a run (`nvidia-smi
  --query-gpu=pstate,clocks.gr --format=csv`: P0); after a reboot the driver
  may sit in P5 ("Prefer Maximum Performance" in NVIDIA X Server Settings).

## Where the frame stands (2026-11-27, RTX 3070 / 5600X, HIGH)

- **The demo**, pipelined at 2560 x 1440: `ground` 3.44 ms, `coast` 3.46,
  `sunset` 3.57, `high` 2.10 (performance.md §40; 6.2 / 6.0 / 6.3 / 3.85 at
  the start of the round). CPU and GPU are now close: check which side
  bounds a pose before choosing a lever.
- **SpaceFlight's Mission 3**: 11.7 ms at 2560 x 1440, 9.1 at 1280 x 720
  (~16 / ~11 before), mostly the CPU's. Per frame (cProfile, inflated):
  Panda's draw and cull 3.3 ms, the game's bots 2.35, `World.update` 1.95
  (the light at the tracked nodes 0.47, the caster map 0.44, the terrain
  0.43, the sea 0.15), the GPU wait 1.6 (performance.md §41).

## Remaining, no visible change

Ranked by what they would take off a game's frame.

- **Panda's per-draw cost** (3.3 ms of the game's frame): shader inputs and
  textures set per draw. The terrain's share is its draws' count, chunks
  times passes (two passes now: the view and the sea's reflection). Ships
  sharing their materials' textures and states would cut theirs. Measured
  and rejected: a transform shared by a cell's chunks (~0.2 ms), one node
  less per chunk (worse; performance.md §38). Frames with chunk uploads
  render 0.6-0.8 ms slower: new vertex buffers, and the whole chunk table
  sent again (Panda 1.10 has no partial texture update; a table split into
  smaller textures would send less). `NodePath.prepare_scene` at upload could
  move the buffer creation off the drawing frame.
- **The game's bots' flight model** (SpaceFlight, ~2.3 ms): every ship's
  physics in Python, one at a time (`Ship.move`: `compute_derivatives`,
  `set_inputs`, the low-pass filters; many small numpy arrays). One
  vectorised integration of every ship's state per frame (an (n, 10) array)
  is the lever; see SpaceFlight's docs/source/performance.md.
- **The slow frames in fast flight** (2026-11-27, `flight_bench.py --flight
  low --trace`: p95 18-20 ms against a 7 ms median). Their excess is Panda's
  `renderFrame` (+7-8 ms, every one of them) and `Terrain._select` (+6 ms,
  95 % of them). The mountain map's twins are not in it (+0.16 ms: their
  marches are 2-4 texels a call, 0.14-0.22 ms a texel; their rare 4-6 ms
  calls are the same work held up elsewhere). The slowest 5 % of the trees'
  updates (6 ms): the drawn chunks with plants gathered (0.75 ms) and put in
  arrays (1.3 ms, ~320 chunks), the distance loop and the new impostor
  nodes (1.95), the meshes refilled (1.84); the quadtree's pass 1.4 ms (its
  keys 1.1). Tried and no change in the slow frames: skipping a mesh
  buffer's upload when its plants are the same (in fast flight they never
  are), keeping a chunk's impostors while it is not drawn (chunks seldom
  come back). Two leads follow.
- **`renderFrame` in the slow frames of fast flight** (+7-8 ms, the larger
  half of their excess). Record them with PStats (`want-pstats 1`,
  `text-stats`) to see what they hold. Suspects: the new chunks' vertex
  buffers created on their first draw (`NodePath.prepare_scene` at upload
  would move that off the drawing frame), the whole chunk table sent again
  on each upload (Panda 1.10 has no partial texture update: smaller tables
  would send less), the trees' mesh buffers re-sent on every pass.
- **The trees' share of a slow selection** (~4.7 of its 6.8 ms): spread over
  passes instead of all of it on every pass. The meshes refilled only when
  the near chunks or the origin changed (or the camera moved a share of
  `MESH_MARGIN`); each chunk's box kept with it (its slot's row in an array)
  instead of three arrays built from the drawn chunks every pass; the new
  impostor nodes a few a pass, the nearest first (watch for holes where a
  parent's impostors go before its children's come).

Measured 2026-11-27 (performance.md §42):

- **The light at the tracked nodes, staged** (`light_period=3`: one stage a
  frame, 0.42 / 0.25 / 0.42 ms against 0.98 ms every third frame). Mission 3,
  six rounds each way in one process: frame mean 8.78 against 9.05 ms (the
  battle moves it more than this), p95 10.96 against 11.49, max 13.1 against
  15.5. The same work, its peak gone.
- **`Terrain._select`'s two per-chunk loops as numpy passes**: Mission 3's
  fly pose, a selection 1.94-2.29 → 1.39-1.44 ms mean (p95 6.9-7.8 → 4.9-5.2),
  of it the collision's reach 0.67-0.81 → 0.29-0.30 (p95 2.5-3.7 → 0.8-0.9)
  and the trees' 0.78-0.89 → 0.46-0.50; the demo's low flight 2.27-2.39 →
  1.68-1.84 (the trees' 1.74-1.87 → 1.13-1.25).
- **One point's terrain query in plain floats**: 167 → 13.5 µs at the
  demo's camera (the swell's struct made once in it).
- **The caster map's fit** in one numpy pass: 197 → 129 µs an update with 16
  ships.
- **The eclipse in the air's march as two loops: no gain, removed.**
  GPU-synced frame, two rounds each, before / after: `ground` 9.49, 8.93 /
  9.36, 9.25; `high` 6.70, 6.68 / 6.57, 6.97; `orbit` 4.73, 4.82 / 4.80, 4.78;
  the pipelined frame alike; and with an eclipse set up (the moon's shadow
  on the planet) the same. The march already gathers the eclipses only
  where a shadow can reach (`eclipseCanReach`), and the shader compiler kept
  the array out of the rest: the 0.2-0.5 ms measured in Part A had been
  taken back then. The shader is back as it was.

Leads measured and closed (2026-11-27):

- **Chunk results through shared memory:** unpickling a chunk (250 KB) is
  18.7 µs, what holds the GIL on the main side: ~3 ms a second at the low
  flight's 150 chunks a second, ~0.03 ms a frame, a tenth of the 0.3 ms that
  would have made it worth it. Its spikes are elsewhere.
- **The mountain map read back from the GPU instead of its twins:** a
  synchronous read-back right after a frame (`extract_texture_data`, Panda
  1.10 has no asynchronous one) took 33 ms for the global map (6 x 1024²,
  13.5 s the first time) and 9.1 ms for the local one (1024²), against a
  3.45 ms frame: worse than the twins' marches even once per bake.

## Savings with a cost (the user's call)

| Change | Gain (measured) | Look |
|---|---|---|
| Terrain `anisotropy` 8 to 4 at HIGH | up to 0.35-0.5 ms GPU (8 to 2 measured) | softer ground at grazing angles |
| Terrain `detail_far` 4000 to 2500 m | up to 0.55-0.75 ms GPU (to 1500 measured) | textures fade to flat colour sooner |
| Terrain `split_factor` 2 to 1.5 | 0.6-1.3 ms (0.4-0.75 of it CPU) | less distant detail |
| The sea reflection's reach (its lens reaches 4x the view's far; chunks past N km out of it) | up to 0.55-0.6 ms CPU | far shores no longer reflected |
| A split by roughness (BACKLOG: flat ground and seabed stay coarse) | most of `split_factor`'s gain where the ground is flat | none intended; a larger change in `lod.c` and the selector |
| The clouds at a quarter of the view instead of a half; 32 steps instead of 64 | ~3.5 ms / ~1.4 ms GPU in the game | coarser, lower clouds (left to the presets) |
| A lighter terrain shader in the reflection (camera tag states: no detail normals, fewer fetches; its coarse mesh is done, §34) | up to ~1 ms | hard to see at half resolution under ripples |
| The terrain's atmosphere per vertex on far chunks | part of its ~2 ms per-pixel atmosphere | banding where the air changes fast (the limb, the terminator) |
| A narrower sea fade band (`OceanLook.near_band`, 12-18 km) | its cost moves down (2.5 ms at 17 km) | where the sea's look changes |

## Generation and loading

- **The planet's composition in C.** A chunk is ~6.7 ms since its geometry
  went native (performance.md §18): ~4.5 ms of native noise, ~2 ms of numpy
  composition (curves and masks around the noise, biome weights 0.6 ms,
  stamps 0.7 ms, paints 0.25 ms). Porting those would make it ~5 ms, at the
  price of the planet's rules in two languages (stamps and paints would need
  their images passed over). Left out by choice: generation no longer limits
  refinement at 60 Hz.
- **The LOD cadence at low frame rates.** A level waits for a pass 3 frames
  after its upload (`select_every_on_upload`): 100 ms a level at 30 fps.
  Time-based instead of frame-based would help slow machines; a pass every
  frame and a deeper queue were within noise at 30 Hz.

## Tried and rejected

- **MSAA in the demo:** Panda picks the window's 16-sample framebuffer config
  whatever count is asked for, which made the start view 40 ms. Exact counts
  need an offscreen buffer (tried, works; the depth camera's scene buffer now
  keeps a game's own count); the user chose no MSAA for the demo.
- **Lower reflection resolution:** 0.5 → 0.25 saved under 1 ms: its cost is
  per chunk, not per pixel.
- **Upload budgets** (3 chunks and 2 ms a frame, or 4 and 3, against 6 and
  4): no lower p95.
- **The state GC's rate** (`garbage-collect-states-rate 0.25`): states pile
  up and it costs more; with `transform-cache false` there is nothing left to
  collect.
- **Panda's per-chunk overhead** (a shared transform per cell, one node less
  per chunk): performance.md §38.
- **The sea without its mirror** (2026-11-14): no reflection pass, the sea
  reflecting the sky-view table along the reflected ray, the clouds where the
  ray meets their layer, the sun as a PBR glint. It looked close to the
  mirror from altitude and a little dimmer along the coast, without the coast
  itself. `terrain_shots.py` at 2560 x 1440, three rounds, median frame ms
  (planar / sky), `open_sea`, `coast`, `swell`:
  - LOW: 4.80-4.85 / 4.75-5.23, 5.47-5.49 / 5.58-5.65, 5.49-5.53 / 5.82-5.89
  - MID: 5.79-5.87 / 5.44-6.04, 7.06-7.10 / 6.59-6.62, 6.95-7.03 / 6.91-6.92
  - HIGH: 6.67-6.72 / 6.27-6.91, 9.11-9.13 / 8.64-8.69, 8.35-8.39 / 8.37-8.41

  About 0.5 ms saved along a coast, nothing over open water or from
  altitude, and 0.1-0.35 ms *more* at LOW. Not worth a setting: removed.
