# Frame profile (WorldBuilder only)

Where a frame's time goes in WorldBuilder's own scenes, per part, and the biggest
culprits, as the reference for RENDERING_PLAN.md's parts B (the atmosphere's
tables) and C (the clouds' light volume). SpaceFlight's frame (its draw of the
ships, its bots) is out of scope here: see [the performance
backlog](performance_backlog.md) for it.

## How it was measured

- RTX 3070, 2560 x 1440 (and 1280 x 720), HIGH, the raymarched clouds, offscreen,
  GPU at P0; VS Code running (nothing else). November 2026, after
  [performance.md §24](performance.md) (one march over both ranges, 4 x 2 at HIGH,
  half-resolution depth passes, the second order of scattering, curled billows).
- `scripts/frame_profile.py`: at each of `terrain_shots.py`'s poses, settled, 80
  GPU-synced frames (a one-pixel read back after each step):
  - **frame**, **p95**: the median frame and its 95th percentile, everything shown;
  - **step**: the median of `taskMgr.step()` alone, before the GPU sync: the CPU's
    part (cull, draw and submission, the demo's update);
  - each part's cost: the median frame with it shown less with it off;
    **terrain** includes the near sea (drawn by the terrain's shader) and the
    terrain's own per-pixel atmosphere;
  - **atmo**: the per-pixel atmosphere's share (sky, terrain, sea, models), from
    its samples at 8 against 2, scaled linearly to all 8.
  Parts overlap (atmo is inside terrain and sky), and a part's cost includes its
  CPU's when the frame is CPU-bound. Run to run: 0.3-0.5 ms.
- `scripts/flight_bench.py --flight low --trace`: the CPU's parts in flight.

## The frame, per pose (2560 x 1440, HIGH)

| Pose | frame | p95 | step (CPU) | terrain | sea refl. | sky | clouds | cirrus | models | atmo |
|---|---|---|---|---|---|---|---|---|---|---|
| ground | 7.26 | 9.19 | 3.64 | 2.72 | 0.78 | 0.50 | 2.17 | 0.24 | 0.07 | 0.64 |
| coast | 7.64 | 11.02 | 3.59 | 3.03 | 0.91 | 0.57 | 2.36 | 0.22 | 0.04 | 1.12 |
| ground_up | 7.05 | 7.73 | 3.86 | 2.06 | 0.90 | 0.89 | 2.74 | 0.26 | -0.06 | 0.78 |
| in_clouds | 8.06 | 13.77 | 3.46 | 2.67 | 1.02 | 0.79 | 3.11 | 0.28 | 0.03 | 1.20 |
| clouds_low | 7.33 | 9.31 | 3.86 | 2.88 | 0.87 | 0.13 | 1.82 | 0.30 | 0.01 | 1.16 |
| sunset | 7.98 | 9.82 | 4.00 | 3.55 | 1.06 | 0.57 | 2.22 | 0.22 | 0.06 | 0.40 |
| high (20 km) | 5.74 | 5.96 | 2.36 | 2.08 | 0.00 | 0.24 | 1.52 | 0.07 | 0.02 | 1.06 |
| ball_noon | 7.27 | 9.26 | 3.66 | 2.73 | 0.77 | 0.48 | 2.18 | 0.23 | 0.06 | 0.63 |
| orbit | 3.31 | 3.37 | 2.05 | 0.47 | 0.00 | 0.20 | 1.31 | 0.69 | 0.05 | 0.23 |

With genera:

| Pose | frame | p95 | step | terrain | sea refl. | sky | clouds | cirrus | atmo |
|---|---|---|---|---|---|---|---|---|---|
| ground_up | 7.83 | 8.39 | 3.46 | 1.70 | 0.71 | 0.80 | 3.63 | 0.18 | 0.72 |
| in_clouds | 10.12 | 15.93 | 3.51 | 2.98 | 1.05 | 0.78 | 4.80 | 0.28 | 1.18 |
| clouds_low | 9.48 | 15.17 | 3.90 | 3.39 | 0.73 | 0.01 | 3.53 | 0.18 | 1.01 |
| sunset | 8.77 | 14.51 | 4.04 | 3.82 | 1.08 | 0.55 | 2.59 | 0.21 | 0.38 |
| high | 7.14 | 7.87 | 2.38 | 2.45 | -0.03 | 0.21 | 2.49 | 0.06 | 1.06 |

With a ring (orbit / high): frame 5.75 / 7.25 ms (2.4 / 1.5 ms over the same poses
without it); the ring's own cost was not split out.

At 1280 x 720 the same poses take 4.8-5.4 ms (orbit 2.7, high 3.8) with the same
CPU step (3.5-4.0 ms): **at 2560 x 1440 the frame is GPU-bound** by 3-4 ms, at
720p nearly CPU-bound. The per-part costs at 720p (terrain 1.8-2.5, clouds
1.0-1.6, sea reflection 0.6-0.8 ms) are largely their CPU's (cull, draw).

In flight (the low flight, 2560 x 1440): mean 6.8 ms, median 6.3, p95 11.3, max
31.5. In the slowest 5 % of frames `renderFrame` takes 7.7 ms (3.5 median) and
`Terrain._select` 1.3 (0.24 median); `TerrainDemo._fly` 0.7, a gen-1 GC 0.3,
`Terrain._upload` 0.5.

## The culprits, ranked

The budget is 6.25 ms (160 fps). At 2560 x 1440, HIGH, the plain poses run 7.0-8.1
ms, with genera 7.8-10.1:

1. **The terrain: 2.1-3.8 ms** (with the near sea and its per-pixel atmosphere),
   about 1.5-2.5 ms of it the CPU's at 720p. Its atmosphere is part B's; its
   draws (per chunk) are in performance_backlog.md ("Panda's state changes").
2. **The clouds: 1.5-3.1 ms, with genera 2.5-4.8.** Within it, **the second order
   of scattering costs 0.7-0.85 ms with genera** (in_clouds 9.72 / 9.86 ms
   compiled out against 10.62 / 10.70 in; clouds_low 9.11 / 9.16 against 9.78 /
   9.86): far more than cloud_bench showed (~0), with the same table fetch in both,
   so it is the cloudy samples' extra work (register pressure, occupancy) rather
   than its arithmetic. Options: drop it at HIGH (keep at ULTRA), or take it from
   the light volume (part C) where the march reads one value for both orders.
3. **The CPU: 3.5-4.0 ms a frame** (2.0-2.4 from orbit and 20 km): Panda's cull
   and draw (`renderFrame` 3.5 ms median in flight, 7.7 in slow frames) and the
   demo's update. Not the limit at 1440p, the floor below which GPU work buys
   nothing.
4. **The per-pixel atmosphere: 0.4-1.2 ms** (sky, terrain, sea, models; most from
   20 km and in clouds). Part B's tables aim at <= 0.5 ms with their bakes: a gain
   of up to ~0.7 ms, and real multiple scattering.
5. **The sea's reflection: 0.8-1.1 ms** wherever the sea is near (it draws the
   scene again: performance_backlog.md's "sea reflection's cull" and reach).
6. **The p95 spikes: 9-16 ms against 7-10 median** at still poses (in_clouds 13.8,
   genera 15-16): periodic work in a still view (not the cloud bakes, which are
   change-driven; candidates: the terrain's selection and uploads, the GC, the
   shadows' re-bakes). Not traced yet; `flight_bench --trace` sees the same in
   flight (`Terrain._select` +1.1 ms, gen-1 GC +0.3).
7. **The sky: 0.1-0.9 ms; the cirrus 0.2-0.3 (0.7 from orbit); models ~0.**

So at HIGH, 1440p, a plain scene needs 1-2 ms off and a genera scene 2.5-4 to
reach 160 fps: the clouds' second order (0.7-0.85 with genera), the atmosphere's
tables (up to ~0.7), and the terrain's draw cost are the levers in that order of
certainty.
