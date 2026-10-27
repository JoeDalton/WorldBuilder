# Asteroid fields backlog

Improvements found but not made, so the fields could move on (October 2026).
The design is in [Asteroid fields](asteroids.md).

## Rock shapes

The prototypes are potatoes: an ellipsoid, lumpy noise, faint ridges and
craters, with the fine detail in the normal map only. That is enough for now.
Ways to make them better:

- **Sharp ridges.** Ridged noise has V-shaped crests that interpolating
  between vertices cannot follow at any resolution: at the first default
  amplitude (0.05 of the radius), every level of detail stayed about 0.05 off
  the finest, which made the intermediate levels nearly useless. The default
  is down to 0.02. To get crests back, soften the noise's absolute value (a
  smooth `sqrt(x² + ε²)`), or move the ridges into the normal map only.
- **Silhouettes.** The finest mesh (32 quads per face side) is still about
  0.025 of the radius off the full detail: 5 m on a 200 m rock, 50 m on a
  2 km one. The normal map hides it in the shading, not in the outline. The
  64-quad level (ULTRA) halves it. Rocks from 200 m up get the terrain's
  level of detail near the camera (phase 6).
- **Shape variety.** Real asteroids include contact binaries (two lobes),
  flattened spinning tops and angular fragments. The radius function could
  take a second lobe (the union of two ellipsoids as a radius function),
  facets (a few planes clipping the ellipsoid), and a per-rock choice of
  family.
- **Generation time.** About 180 ms per prototype on one core: 32 prototypes
  take about 2 s across the workers at level load. The vertex normals
  evaluate the shape five times per vertex; the finest level could reuse
  neighbouring vertices instead (as the normal map does), and the radius
  function could go native.
- **Normal map seams.** Each face's normal map is clamped at its edge, so
  filtering stops at face seams: a faint line can show on a very close
  rock. A cube-map array (GL 4.0) or a one-texel border copied from the
  neighbouring faces would remove it.
- **Tumbling from the shape.** `Tumble.ratio` is a distribution;
  "from_shape" (I1/I3 - 1 from each prototype's ellipsoid axes) would tie
  tumbling to how elongated each rock is.
- **Unique rocks.** Every rock is one of a few prototypes, turned and
  scaled. Phase 6's terrain rocks add per-rock detail up close; instanced
  rocks could vary too, through a per-instance offset into a noise in the
  shader.
- **Overhangs, arches and caves.** A radius function cannot have them: these
  are the hero rocks (designed for in the plan, not built).

## Drawing

- **Whole table uploads.** Only new rows are written on the CPU, but Panda
  1.10 uploads the whole texture (64 bytes a slot, 20-30 MB for a ring's
  rocks) each time some changed: every third frame while flying through a
  ring. One texture per size class, or a ring of row blocks, would bound
  it.
- **Spikes in fast flight.** At 16 km/s through a ring, frames reach 20-40
  ms when the big windows move (their new cells scattered, the rocks of the
  old freed by a scan of the class) and the table uploads. Spreading a
  move's new cells over frames (the class limit standing in), and a native
  free pass, would flatten them.
- **Rocks held in every direction.** The windows hold rocks all round the
  camera (to turn freely): half a million in a ring for 15,000 drawn. The
  held budget bounds the CPU's share; a window narrower behind the camera
  (filling as it turns, the class limit standing in) would hold far fewer.
- **Texture seams.** Textures follow each cube face's grid, so their
  pattern turns at the face edges of every rock; triplanar mapping in the
  body frame would hide it (needed anyway for hero rocks).

- **Ice in patches, not blended.** A rock's ice fraction now weights a
  rock/snow blend over its whole surface: a uniform beige at low fractions,
  with artefacts where the two textures meet, and a far colour that was
  hard to match. Instead, lay out separate patches: a mostly-rock body gets
  a few patches of ice, a mostly-ice body a few patches of rock, the
  fraction setting how much of the surface each covers. This gives
  contrast and variety rather than a blend.
  - The patches come from a noise of the direction, thresholded at the
    fraction's quantile (as the field's clumping coverage is). Per
    prototype, the patches go in a fourth channel of the normal map (RGB8
    today, RGBA8 then), so the instance shader reads them in the same
    fetch. The rock's terrain
    (`RockBody.weights`) uses the same function, so the swap does not
    change them.
  - The far colour becomes each patch's mean colour: the pixels' share,
    not the weights' blend.
  - The same idea fits planets' biome borders.

## Haze

- **A front is still visible** (reported 2026-10-27, flying the `ring`
  demo after the thresholds were removed): rocks still appear or vanish
  along a visible line. Not investigated yet. Suspects, in order:
  - the far plane: in open space `World`'s far plane is 1,000 km, but at
    2 pixels the biggest rocks are drawn out to about 2,600 km, so the far
    plane cuts them hard (fix: `World.depth_ranges` reaching the fields'
    reach, or the fields' `far` capped at it to fade them into the haze);
  - the held budget's class limits: their fade band (the last fifth of a
    limit) moves 3 % a frame while the budget adjusts, and the rocks there
    are a few pixels wide, brighter than the haze that takes them;
  - the clumps: the haze averages them, so where a pocket of rocks ends the
    haze does not darken with it (see Clumps below);
  - the size cut itself: a 2-pixel rock dithers out over a quarter of its
    distance, a bright dot against a haze of the same mean but no texture.

  To find it: shots of the same view with the far plane pushed out, with
  the held budget off, and with uniform density (no clumps).

- **Shadows.** The haze and the rocks take no shadows: a planet's shadow
  across its ring, the ring's on the planet, the rocks' on each other are
  for the shadows issue.
- **Grazing rays.** 32 midpoints along a ray within a ring's plane span
  thousands of kilometres: its ringlets alias into faint dashes on the far
  rim. More samples where the ray is long, or a per-pixel jitter, would
  smooth them.
- **Clumps.** The haze averages the clumping (its coverage): an even glow
  where the rocks drawn gather in pockets, so the two disagree where a
  pocket's rocks thin out into the haze. The clumping function itself (the
  scatter's fbm and smoothstep, computed natively) sampled on a few
  camera-centred grids (a clipmap: 128² cells per level, each level twice
  as wide, in the ring's plane; a 3D one for a box or a ball), recomputed on
  the workers as the camera moves a cell, and read by the haze's samples in
  place of the coverage, would put the pockets in the haze exactly where
  the rocks are. A GLSL copy of the noise would not match the CPU's bits
  (its 64-bit hash), so the pockets would sit beside the rocks'.
- **Dust near the camera.** The far camera of a split draws the haze from
  the split on: the dust nearer is lost (0.06 of optical depth for the
  `ring` preset's 2 km split). The near camera could draw the dust alone.
- **Forward scattering.** Rocks are Lambert spheres, dark against the sun;
  fine dust scatters forward and glows there (Saturn's rings seen against
  the sun). A phase term for the dust would add it.

## Rocks as terrain

- **A turning rock's surface speed.** A rock drawn as terrain is solid
  through the terrain's `TerrainSurface`, which has no `velocity_at`: a
  game's handlers see its surface as still (a big rock turns slowly, about
  1 m/s at its surface at the default rates).
- **Normal detail on the terrain.** The terrain only beats the instance's
  normal map within 0.8 R of the surface, because its chunks carry no
  normal map of their own. Chunk normal maps of the finer detail (as the
  instances have) would let the swap happen farther, and look finer on the
  way.
- **What terrain rocks cost.** The detail they add up close is modest for
  scattered rocks; they stay mostly for hero rocks. Measured with
  `scripts/asteroid_bench.py --field cube --flight approach --offscreen
  --terrain-rocks N` (HIGH, 1600×900, two runs each, frame ms):

  | `max_terrain_rocks` | Mean | Median | p95 | Max |
  |---|---|---|---|---|
  | 0 | 1.82-1.93 | 1.78-1.83 | 2.06-2.51 | 3.1-3.2 |
  | 8 | 2.63-2.96 | 1.97-2.07 | 5.45-5.46 | 15-159 |

  p95 is more than twice as high, and the first terrain made costs a
  15 ms frame (159 ms in one run, likely a cold shader compile). To try:
  - make one terrain at the field's start, hidden, so its shader and
    inputs are ready;
  - load fewer chunks per frame while the terrain is still hidden;
  - make LOW and MID default to 0 terrain rocks, and keep terrain for hero
    rocks only.
- **Textures.** This checkout has no biome texture sets
  (`terrain/textures/<biome>/`), so rocks show the procedural placeholder,
  whose tiling reads as a knitted pattern up close.
