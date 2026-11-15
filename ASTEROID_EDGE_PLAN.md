# Asteroid edge plan: where the drawn rocks give way to the haze

The `ring` preset seen from above its plane (`demo_asteroids.py --preset ring`, 82 km over the plane, 101,000
km from the axis) shows a line where the drawn rocks end and the haze begins (reported 2026-11-15; the
backlog's "A front is still visible" of 2026-10-27, asteroids_backlog.md).

Rules: never commit unless asked; stop for the user's visual check after each step.

## What was measured (2026-11-15)

- **The mean brightness already matches across the line.** Rows of the user's screenshot, either side of it:
  162-166 (0-255) in the haze, 160-166 among the rocks. The haze shades each missing rock as a Lambert sphere
  at the sun's phase angle (`hazeColourLit`, haze.glsl): the "shade the haze like the mean rock at that angle"
  idea is in, and holds. **What jumps is the texture**: the spread per row falls from ~15 to ~5. The eye sees
  discrete bright rocks give way to a flat sheet.
- **The line is where rocks are still a few pixels wide**, not where they shrink under `min_pixels`:
  - At 1,000 km, the field's `far`: the demo passes the depth's far, and `World`'s far in open space is
    `SPACE_FAR_M` = 1,000 km. The two biggest classes (800-2,000 m) are held to 1,400-1,800 km but cut there,
    3-8 px wide. This is the line in the screenshot. A far of 3,000 km moves it out to their class limits,
    and the line fades a lot (scratch shots `edge_default.png` / `edge_far3000.png`).
  - At each class's limit (`Population.limits`: its window's radius), set by the held budget: `held_pixels`
    was 3.8 px there (285,000 rocks held for a budget of 150,000 on HIGH), so each class ends at ~2.9 px.
- **The fade is narrow and dotted**: the last fifth of a limit (`FAR_FADE` = 0.2), by a 4x4 Bayer screen door
  over each rock (asteroid.frag): a fading rock looks speckled, not rarer.

## Options

### 0. The far plane reaches the fields' reach (done 2026-11-15)

`World.depth_ranges` in open space returns a far plane past the farthest rock any field can draw (2 x the
largest radius x pixels per radian / `min_pixels`: ~2,600 km for the `ring` preset at 1600x900), so `far`
no longer cuts the big classes while they are several pixels wide. With the 2 km split, depth steps at
2,600 km are ~200 m: plenty for 2-px rocks. Small change, largest effect measured. With planets, a farther
far plane coarsens their far range a little.

Done as: `AsteroidField.reach(camera)` (the largest rock at the settings' `min_pixels`, at most the field's
far side), and `World.depth_ranges` taking the largest of the bodies' far (or `SPACE_FAR_M` in open space)
and every field's reach; `demo_terrain.py` does the same for its ring. Not tied to the altitude: with bodies
the far already follows the horizon (`planet_ranges`), so in-atmosphere views keep theirs; the line was open
space's fixed 1,000 km. The scratch view's far went from 1,000 to 3,440 km, the line at 800 km is gone, the
frame time unchanged (8.6 ms).

### 1. Rocks thin out whole, over a wider band (tried 2026-11-15, rejected)

Each rock drawn whole or not against its own random threshold (a hash of its place), over the last half of a
class's limit instead of the last fifth, linear in 1 / distance (linear in distance, the band's end was
crammed into a few rows at grazing angles). Measured, it halved the grain's step at the far and spread it at
the class limits; seen, it did not change the render significantly. Removed.

### 2. The clumps in the haze

The haze averages the clumping (coverage 0.7): near the camera rocks gather in pockets 20 km across, past
the line the haze is even. The backlog has the design: the scatter's clumping function computed natively on
a few camera-centred grids (a clipmap in the ring's plane, 3D for a box or a ball), recomputed on the workers
as the camera moves, read by the haze's samples in place of the coverage. A GLSL copy of the noise would not
match the CPU's 64-bit hash: its pockets would sit beside the rocks'. Medium-large.

### 3. Grain in the haze

Noise fixed in space, its contrast from how few missing rocks a pixel holds (strong where each pixel holds a
few big ones, gone where it holds many): the haze gets the rocks' texture statistically. For a thin ring, on
its mid-plane, so it does not swim. Cheap, but a counterfeit: flying in, the speckles do not become the rocks.

### 4. Far rocks as GPU-only points (prototyped 2026-11-15, stashed)

Past the held windows, down to ~1 px, rocks drawn as points or tiny sprites generated on the GPU from the
same placement hash, the CPU holding none: approaching, a point becomes its rock in place. The line moves to
where rocks are sub-pixel and the grain fades out by itself. The real fix; the largest job (the scatter's
placement in GLSL, matching the CPU's for the hand-over).

Prototype in `git stash` ("asteroid edge option 4: far rocks as GPU points"): `points.py`, `points.vert`
/ `.geom` / `.frag`, `u64.glsl` (exact 64-bit hash in 32-bit halves), the clump fbm ported with an early exit.
Drew the far rocks, but at ~45 ms a frame before its rework (66M candidate slots, the clump noise per
candidate) and unmeasured after it (moved to a vertex + geometry shader pair). Seen with the 1,000 km far, the
points sat in the same range as the meshes: the far was the line, not the points' absence. Shelved for 0.

### 5. The budgets

Find what sets `held_pixels` in each view; let the few big classes reach farther than the many small ones (a
per-class held budget). Moves lines back, does not remove them.

## Order

1 rejected, 4 stashed, 0 done (2026-11-15). Next: judge what remains in flight; 2 or 5 if a line shows.
