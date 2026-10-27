# Asteroid fields

[`world_builder.asteroids`](https://github.com/JoeDalton/WorldBuilder/tree/main/src/world_builder/asteroids/)
makes fields of rocks, from a few kilometres across to a planet's ring
(issue #14). No list of rocks is stored: any part of a field is generated from
its spec when asked for, and is the same every time. That is what lets a field
be as large as a ring, drawn and collided with only around the camera.

```
python scripts/demo_asteroids.py            # fly through a field (controls in the script)
python scripts/demo_asteroids.py --preset ring   # a planet ring; Home for the view from afar
python -m world_builder.asteroids.preview   # each preset's rocks from above and the side, and their statistics, to PNG
python scripts/asteroid_gallery.py          # rock prototypes at each level of detail, to PNG
python scripts/asteroid_bench.py --field cube --offscreen   # frame times in fixed flights
pytest tests/test_asteroid_*.py
```

The field's definition, its rocks, their shapes, their drawing and
collision, big rocks drawn as terrain near the camera, rings and the haze of
the rocks not drawn are all built (the plan's phases 1 to 7, see
[Phases](#phases)). Improvements found but not made are in the
[backlog](asteroids_backlog.md).

## A field

A `FieldSpec` (`asteroids/spec.py`) is a frozen, picklable dataclass. Lengths
are metres, angles radians, rates radians per second, all in the **field
frame**: the field's own centre and axes.

| Part | Class | What it sets |
|---|---|---|
| Extent | `Box(half_size, edge)`, `Sphere(radius, edge)`, `Annulus(inner, outer, thickness, edge, radial_edge, radial_profile)` | Where rocks are. The density fades to nothing over `edge` inside the surface. See [Rings](#rings). |
| Density | `Density(count_per_km3, clumping, dust)` | Rocks per km³ where densest; dust only seen as [haze](#haze). |
| Clumping | `Clumping(scale, coverage, softness, octaves)` | Patchiness: rocks gather where a 3D noise is high. |
| Sizes | a `Distribution` | The rocks' bounding radii. |
| Spin | `SpinSpec(rate, axis, size_exponent, tumble)` | How they turn. |
| Shapes | `ShapeSpec(prototypes)` | How many distinct shapes. |
| Material | `MaterialSpec(ice_fraction, patch_scale)` | Each rock's share of ice, laid out in patches of that size. See [Ice patches](#ice-patches). |
| Keep-outs | `KeepOut(centre, radius, end=None)` | Balls or capsules no rock enters. |

`FieldSpec.summary()` gives what a spec amounts to: the expected rock count,
the mean spacing and the share of volume filled where densest, and the
median, mean and largest radius.

`spec.PRESETS` has three examples. `cube` is SpaceFlight's missions 1 and 2
setting: a 15 km cube of rock, power-law sizes from 50 m to 2 km, patchy,
with a 300 m corridor through it. `icy_ball` is a 10 km ball of rock and ice
spinning about one axis, some of it tumbling. `ring` is a planet's ring of
icy rocks, 90,000 to 110,000 km from its axis and 4 km thick, about
3 × 10¹⁰ rocks.

### Sizes and other variables

A rock's `radius` is its bounding radius: no part of it reaches farther. Sizes,
spin rates, cone angles and ice fractions are each a `Distribution`
(`asteroids/distributions.py`): a family truncated to [lo, hi].

| Family | Use |
|---|---|
| `uniform(lo, hi)` | |
| `normal(mean, sd, lo, hi)` | |
| `lognormal(median, sigma, lo, hi)` | One typical size with a spread. |
| `power_law(q, lo, hi)` | Density ∝ x^-q: few big, many small. q = 3.5 is a collisional cascade, as in rings and belts. |
| `u_shaped(a, lo, hi, log=True)` | beta(a, a) across the interval, by default across log x: mostly small and large, few between. |
| `table(xs, pdf)` | Any shape, piecewise linear. |
| `constant(value)` | |

A field draws from each one's inverse CDF tabulated at 1,025 points, by
linear interpolation. numpy and the native scatter do the same arithmetic on
the same table, so they draw the same values.

### Clumping

The idea of "asteroids where a noise is above a threshold" is kept, but it
sets the density, not the rocks: a rock's chance to exist is multiplied by
`smoothstep(t - softness, t + softness, fbm(x))`. `scale` is the noise's
largest wavelength, the clusters' spacing. `coverage` is the share of space
populated; the threshold `t` is derived from it through the measured
distribution of the noise. `softness` is how gradually a cluster's edge
fades, in noise units: 0.02 is sharp, 0.3 very soft.

The rocks' sizes are not tied to the noise, so the density, the clumping and
the size distribution are three independent knobs.

### Spin and tumble

Each rock turns at a constant rate about a fixed axis.

- **Rate** is drawn for the smallest rocks, and scales as (radius /
  smallest)^-`size_exponent`: big rocks turn slower. At the default 1.5, a
  2 km rock turns about 250 times slower than a 50 m one, as in SpaceFlight's
  current field.
- **Axes** are `Isotropic()`, or `AroundAxis(axis, kappa)`: von
  Mises–Fisher around one direction, for a ring's rocks turning with its
  orbit. kappa 10 keeps most within about 25° of it, 100 within about 8°.
- **Tumble** (`Tumble(fraction, cone_angle, ratio)`): a share of the rocks
  move as free symmetric tops. Their body z axis sweeps a cone about the
  fixed axis while they turn about it. `ratio` is I1/I3 - 1 for moments of
  inertia I1 = I2 and I3 about body z; it is positive for an elongated rock.

`asteroids/motion.py` has the closed form: q(t) = rot(axis, precession · t) ·
q0 · rot(z, spin · t), the solution of Euler's equations for a free
symmetric top (`tests/test_asteroid_motion.py` integrates them to check). A
rock that only spins has spin = 0. Time is always given by the caller, from
the game's clock, so a paused game stops the rocks.

## Scattering

`asteroids/scatter.py` gives the rocks of any cells, boxes or neighbourhoods
of a field: `Scatter.rocks_in_cells`, `rocks_in_box`, `rocks_near`,
`is_clear`, `find_clear`. Each rock is a `ROCK_DTYPE` record: id, size
class, prototype, centre, radius, q0, axis, precession, spin, ice.

- **Size classes.** Radii [lo, 2 lo), [2 lo, 4 lo), and so on up to the
  sizes' hi. Class k scatters in cubic cells of edge `cell_factor` × its
  smallest radius (8 by default). Big rocks are sparse in big cells, so each
  class can be asked for out to its own distance: a 2 km rock is seen from
  much farther than a 50 m one.
- **Candidates.** A cell's 64-bit hash (splitmix64 chained over its
  coordinates, from the class's seed) draws a Poisson count of candidates.
  Each candidate's position, radius, prototype, ice, spin and q0 then come
  from the hash of (cell, candidate, slot). Nothing depends on which other
  cells are asked for, or in what order.
- **Existence.** A candidate exists with probability = the density there:
  the extent's profile times the clumping. Candidates inside a keep-out do
  not exist.
- **Hard core (Matérn type II).** A rock is dropped where it overlaps an
  existing candidate that ranks above it: larger, or as large with a smaller
  id. The test is against candidates, not against survivors, so it stays
  local: a cell's rocks depend on its 27 neighbours in its own class and in
  every larger one, nothing farther. No two rocks overlap.
  - At the densities of a game field this removes a few percent: 4 % of the
    `cube` preset.
  - Near packing it removes most, because a candidate goes wherever any
    larger one overlaps it, kept or not. A field whose rocks fill more than
    about a tenth of its volume gets far fewer rocks than its density says.
- **Keep-outs.** Static ones are in the spec. `Scatter.with_keep_outs` adds
  more at runtime, such as around a level's spawn points and routes.
  `find_clear(point, radius, search)` finds the nearest point where a ball
  touches no rock, for ships placed relative to the player.

The native library runs the same steps (`native/scatter.c`, about 100x
faster): the 3,500 rocks of the `cube` preset take 12 ms. numpy runs without
it. The two give the same rocks: every choice uses the same arithmetic on the
same values, and the clumping noise goes through the same per-point fbm.
Without the native noise as well, the clumping's last bits differ, and a rock
on the very edge of existing may differ.

## Rock shapes

Every rock is one of the field's prototypes (`ShapeSpec.prototypes`, 32 by
default), turned and scaled. A prototype is a radius function of direction
(`asteroids/shape.py`), in units of the rock's bounding radius. Its body
depends on its family (`ShapeSpec.family`: the first 40 % of the prototypes,
`round_prototypes`, are round; the others split by `crooked_shares`):

| Family | Body | Like |
|---|---|---|
| round | An ellipsoid, semi-axes (short, middle, 1), its long axis along body z (a tumbling rock's symmetry axis; `elongation` draws short / long), with big bulges and dents (`bulges`: 2 octaves of noise, 2 radii long). | Most big asteroids |
| binary | Two ellipsoidal lobes, the small one `lobe_ratio` of the big one, its axis up to 40° off the big one's, overlapping by `lobe_overlap` at the centre, joined by a smooth maximum (a waist, not a crease), each with its own lumps. | Toutatis, Itokawa |
| bent | An elongated ellipsoid (`crooked_elongation`) seen through bent directions (its ends curve aside, `bend`) and twisted ones (`warp`, low-frequency noise, faded out at the tips). | 54509 YORP |
| fragment | An ellipsoid clipped by `facets` planes (`facet_depth` deep), joined by a smooth minimum: flat faces, rounded edges. Half the lumps and craters. | Collision shards |

On every family:

- lumpy fbm noise (`lumps`, `lump_scale`) and faint ridged noise (`ridges`),
  evaluated on the body's surface (not by direction: on an elongated rock's
  long sides, or a fragment's faces, noise by direction stretches into
  streaks);
- craters with raised rims (`craters` per prototype, `crater_size` their
  angular radius, `crater_depth` over their diameter);
- scaled so its farthest point is 1, measured on a fixed sampling with a 2 %
  margin, so the shape does not depend on the quality settings.

**Size picks the family.** Small rocks are mostly crooked, big ones mostly
round, as real bodies round under their own gravity: a rock is round with
the chance `round_share` (15 % at the smallest size, 85 % at the largest,
log-linear between; `ShapeSpec.round_chance`), then one of its family's
prototypes. The scatter draws both from the same uniform as before,
through a table of the chance over the radius (numpy and C alike, no log).

A radius function has no overhangs, arches or caves: every family is
star-shaped from the centre (a binary's centre is inside both lobes; a
fragment is convex).

`python scripts/asteroid_gallery.py --family binary --views 4` shows one
family's prototypes from four sides.

The workers build each prototype (`asteroids/prototype.py`,
`generate_prototype` on a `ChunkExecutor` channel: about 230 ms for a round
one, 300 for a binary or a bent one, on one core; 32 load in about 3.3 s on
the workers):

- **Meshes**, one per level of detail: a cube-sphere grid of 4, 8, 16 and 32
  quads per face side (64 at ULTRA), the terrain's face frames, equal-angle
  grid, diagonal and winding. A level samples the shape band-limited to four
  times its spacing, so a coarse mesh is a smoothed rock, not an aliased one.
  The faces share no vertices, but seam vertices come from bit-identical
  directions, so the mesh has no cracks.
- **Normals** by central differences about each vertex, in a tangent frame
  of the direction alone: the same on both sides of a seam.
- **Errors**: each level's largest radial distance from the finest mesh (how
  far it can be seen without showing) and from the full detail (the normal
  map's). With the defaults: about 0.08, 0.05, 0.03 and 0 of the radius from
  the finest; the finest is about 0.025 from the full detail.
- **Radius grid**: the finest mesh's vertex radii, for collision on the
  triangles drawn.
- **Normal map**: the full-detail shape's body-frame normals per texel of
  each face's equal-angle grid (6 × 128² RGB8 per prototype). Drawn at every
  level, it lends coarse meshes the fine detail.

`asteroids/mesh.py` turns them into Panda Geoms (one index buffer per level,
shared) and one normal-map texture array (six layers per prototype).

## Drawing

`AsteroidField(app, spec, clock, settings=, ...)` (`asteroids/field.py`)
draws a field around the camera; `update(camera_world, camera_render,
cameras=)` every frame once the camera is placed. It hangs under render and
is placed relative to the camera, as the terrain is. The field's own frame
sits at `position` with `orientation` (world from field) in the world.

- **Prototypes** are built by the terrain's worker pool (its own channel)
  when the field is made: about 2.5 s for 32. `loading` is True until they
  are all in; nothing is drawn before. A game's loading phase waits for it.
- **Rocks in range** (`asteroids/population.py`). A rock is drawn while its
  diameter covers `min_pixels` on screen. Each size class holds a ball of
  cells (its window: every cell within its range of some point of the
  camera's cell) out to its largest rock's drawing distance: small rocks
  near the camera, big ones far. Each rock in those cells holds a slot of
  the instance table until its cell leaves the window. A window is its
  centre cell and its radius, each row of cells one interval: when the
  camera crosses a cell, only the cells entering it are scattered, and the
  rocks of those leaving it freed. It grows toward its range at most about
  30,000 new cells a frame over all classes, so entering a ring is spread
  over frames; `loading` stays True until every window is within 10 % of
  its range.
- **Class limits.** No rock of a class is drawn past its window's radius
  (while it grows) or the field's `far`: they fade out over the last fifth
  of it, and the haze takes them in over the same band, class by class. A
  window lagging behind a fast camera is a soft edge into the haze, not a
  front.
- **The instance table** is a float texture of four texels per slot: the
  rock's centre relative to its cell's corner, its radius, its orientation at
  an epoch, its precession and spin, its cell and its class, prototype and
  ice. The camera is given per class the same way (its cell, its offset in
  it), so a rock's position relative to the camera is a difference of small
  numbers: float32 holds at any distance from the field's centre. New rows
  are written into the texture's RAM image (only those) and uploaded at most
  every three frames; a rock is drawn once its row is on the GPU (new ones
  are at the edge of their range, a few frames late is unseen).
- **The visible pass** (`native/visible.c`, numpy without the library),
  per frame over every rock held: big enough on screen, within its class's
  limit, inside some camera's view (`cameras=`: the main one, and a
  rear-view mirror), at the coarsest level of detail whose error stays
  under `lod_error_px` pixels, with 15 % hysteresis. It reads a compact
  copy of each slot (40 bytes), and lists the big rocks near the camera for
  the terrain rocks. The coarsest mesh (2 quads a face side, 48 triangles)
  draws rocks a few pixels wide. The slots drawn go in an index texture, grouped by
  (prototype, level): one GeomNode each, drawn with
  `set_instance_count`. Nothing per rock goes through the scene graph.
- **Budgets.** Past `instance_budget` rocks drawn (in view: the GPU's),
  the smallest size drawn grows by 10 % a frame until they fit, and shrinks
  back when well under. Past `held_budget` rocks held (in every direction:
  the CPU's), the windows keep only bigger rocks, 3 % a frame: the class
  limits come nearer. The haze takes the rocks either drops.
- **Spin** is computed in the vertex shader at the game's time (`clock=`,
  never Panda's): q(t) as in `motion.py`, from the table's orientation at
  the epoch. The epoch moves every 1,024 s, rewriting the table, so float32
  time stays precise in a long game. A paused clock freezes every rock.
- **Fade.** Rocks fade out by dithering over the last quarter of their
  drawing distance.
- **Shading** (`asteroids/shaders/asteroid.frag`): the prototype's normal map
  (the full-detail shape) turned with the rock, the terrain's snow textures
  on the rock's ice patches and its rock textures elsewhere
  (`terrain_material.glsl`: rock projected from the sides on steep flanks), lit as the terrain lights its
  ground (`object_light.glsl`), no shadows. Then either the world's exposure
  and ACES tonemap, or, with `tonemap=False`, a plain display encoding to
  match models lit by Panda's fixed-function lights. Last, the haze between
  the camera and the rock veils it ([Haze](#haze)): a far rock takes the
  haze's colour as the haze would hide it, so it melts into the haze it
  leaves to.
- **Costs** are the `asteroids` section of the [quality settings](quality.md).

### Measured

`scripts/asteroid_bench.py`, offscreen at 1600×900 on a GTX 1070, each frame
waiting for the GPU, two runs each (October 2026). The legacy field is
SpaceFlight's for missions 1 and 2: 2,520 nodes instancing one 10,000-triangle
glTF, each with a collision sphere, 520 of them turned every frame. The new
one is the `cube` preset, about 3,400 rocks, at HIGH.

| Flight | Legacy p95 | Field p95 | Legacy mean | Field mean |
|---|---|---|---|---|
| cruise (through the field, 250 m/s) | 19.9–20.5 ms | 3.6–4.0 ms | 17.9–18.0 ms | 2.8–3.0 ms |
| sweep (turning at its centre) | 10.1–10.3 ms | 3.0–3.4 ms | 8.7 ms | 2.2–2.4 ms |
| outside (the whole field in view) | 35.3–37.0 ms | 4.1–4.7 ms | 33.4–33.7 ms | 3.2–3.4 ms |

The field's own main-thread work is about 1–1.5 ms a frame, most of it the
visible pass in numpy (see the [backlog](asteroids_backlog.md)).

### Ice patches

A rock's ice is not spread over it: it lies in patches
(`shape.IcePatches`). Each prototype has a smooth noise of the direction (3
octaves, its largest wavelength `MaterialSpec.patch_scale` radii, 1.5 by
default), ranked over its surface: the rank is the share of the surface
lower, 0 to 1, measured on 8,192 directions of equal area. A rock of ice
fraction f is ice where the rank is above 1 - f, so on the share f of its
surface: a few patches of ice on a rocky rock, a few of rock on an icy one,
each pure (the edge ramps over 0.04 of rank).

- **Instances** read the rank in the normal map's fourth channel. Far
  away, where a texel spans many pixels (the mipmaps average the ranks),
  they take the ice's share instead; past `detail_far` the patches keep
  their two mean colours.
- **The rock's terrain** (`RockBody.weights`) and **collision**
  (`material_at`) read the same function: the swap keeps the patches, and a
  ship hits the snow it sees.
- **The haze** takes the rocks' mean colour: snow on the mean ice share,
  rock on the rest.

## Collision

`field.enable_collision(into_mask, name="asteroids", owner_tag="owner",
max_distance=20_000, samples=5)` makes the rocks solid to a game's own
`CollisionTraverser` (traversing render). It adds one into-only
`CollisionNode` holding a native `CollisionAsteroids` solid
(`terrain/native/asteroid_collision.cpp`, in the same library as the
terrain's).

- **Which rocks.** Those within `max_distance` of the camera, registered and
  dropped as the camera moves. Farther rocks are drawn but not solid: a ship
  flying far from the player passes through them, as with the terrain.
- **On what.** Each prototype's finest mesh: spheres (ships, ordnance,
  sensors) and segments (lasers) are tested in each rock's own frame, turned
  to the time it was drawn at, against those very triangles (the radial ray
  under a direction meets one triangle, the meshes' diagonal). A segment is
  marched within the rock's bounding sphere and bisected; a sphere takes the
  nearest of a footprint of samples, refined on its triangle's plane, as the
  terrain's test does.
- **What the game gets.** Ordinary `CollisionEntry` objects, at most one per
  collider per frame: the deepest sphere contact, the first segment hit, with
  the surface point and the triangle's normal.
- **The owner** (`AsteroidSurface`, the node's python tag):
  `material_at(point, relative_to)` gives "snow" on the rock's ice
  patches, else "rock"; `velocity_at(point, relative_to)` gives how fast the
  surface moves there (the rock's angular velocity crossed with the point's
  offset from its centre), for a game whose handlers take the obstacle's
  motion into account.
- **Where the node goes.** Beside the field's root, in the field's axes, at an
  origin snapped to 4 km around the camera: its float32 coordinates stay
  small at any distance from the field's centre.

**Cost:** one `traverse` with SpaceFlight's load (64 ship spheres, a sixth
of their sensors, 300 lasers within 5 km of the camera) takes 2.4 ms over
the `cube` preset's 3,400 rocks, against 32 ms for the legacy field's 2,520
collision spheres (`scripts/asteroid_bench.py --collision`).

## Rocks as terrain

Up close, a big rock is drawn by a real `Terrain`, with the terrain's level
of detail and collision, instead of its instanced mesh
(`asteroids/promotion.py`, `asteroids/rock.py`).

- **The body.** A `RockSpec` names the rock: its field's shapes and seed, its
  prototype, radius and ice, and a seed of its own. Its `RockBody` is the
  prototype's radius function times the radius, plus metre-scale detail of
  its own, band-limited as a planet is: the instanced meshes are the same
  rock, coarser. See [Terrain: other bodies](terrain.md#other-bodies).
- **When.** An instance shows its prototype's full detail through its normal
  map, down to about pi R / `normal_size`. The terrain at a distance d is
  band-limited to about d / 32. So the terrain is the finer only within
  16 × 2 × pi R / `normal_size` of the surface: 0.8 R at HIGH (a 2 km rock
  within 1.6 km of it, a 200 m rock within 160 m).
- **Which.** Rocks of at least `terrain_min_radius` (200 m), nearest first,
  at most `max_terrain_rocks` (8 at HIGH, none at LOW). A rock's terrain
  loads, hidden, from twice that distance. It takes over within it once
  its ground for the camera is drawn: the instance is hidden and its
  collision passes to the terrain's own solid (same node name, mask and
  owner tag). It hands back 15 % farther out, and is released past the
  preload distance.
- **Pool.** The terrains are reused (`set_spec`), all on the field's worker
  pool and texture arrays, each keeping at most a few hundred chunks.
- **Motion.** Every frame each terrain sits at its rock's centre and turns
  as the rock does at the field's time (the terrain's orientation).
- **Look.** Lit as the instances are (as an object, no day and night by
  the vertical), the same textures laid out the same way (whole 128 m
  periods along each face), the same tonemap or display encoding.

**Cost** (`scripts/asteroid_bench.py --flight approach`): flying from 6 to
1.3 radii of the `cube` preset's biggest rock (834 m), which turns into
terrain on the way: median 2.0 ms, p95 5.5 ms, the worst frame 15 ms (the
first terrain made). Without terrain rocks (`--terrain-rocks 0`), median
1.8 ms and p95 2.1-2.5 ms. See the backlog.

## Rings

`Annulus(inner, outer, thickness, edge, radial_edge, radial_profile)` is a
ring around the field's z axis: its density fades over `edge` inside its
faces and over `radial_edge` inside its rims, times `radial_profile`, values
evenly spaced from the inner rim to the outer (ringlets, gaps), linearly
interpolated. The native scatter has it too (interface 11). A field's
`orientation` tilts it.

Nothing in a field depends on its size: a ring is scattered like a cube, only
around the camera. Two things scale with it:

- **Windows** in a 4 km sheet hold up to millions of cells (most of them
  empty) when `min_pixels` is small: they move by their difference and grow
  over frames (see [Drawing](#drawing)). Moving through the `ring` preset
  with 180,000 rocks held cost 230 ms a cell crossed when windows were
  rebuilt whole; now a few ms.
- **Rocks held.** At 2 pixels, the ring holds half a million rocks around
  the camera for 15,000 drawn: the held budget (300,000 at HIGH) keeps the
  CPU's share to about 5 ms a frame.
- **Cells.** Bigger cells (`cell_factor`) mean fewer of them, but the hard
  core compares every candidate with its neighbours' (cost per rock: 5 µs at
  8, 111 µs at 32): keep the default.

`python -m world_builder.asteroids.preview --preset ring` maps a 40 km patch
of a field too big to scatter whole, beside the density of the whole ring.

## Haze

The rocks too small or too far to be drawn are not gone: the haze
(`asteroids/haze.py`, `shaders/haze.frag`) integrates them along each
pixel's ray. From afar a ring is a band of light with its ringlets and gaps;
approaching, rocks appear out of it while it thins by as much.

- **What is missing.** At a distance t, the rocks under r_cut = t ×
  1.125 `min_pixels` / (2 pixels per radian) (the middle of the drawn rocks'
  fade), and the share of each class its limit fades out (all of it past
  the limit). Their cross-section per
  metre is the density there (the shape's profile, the peak density, the
  clumping's coverage: clumps averaged) times the mean of π r² over the
  rocks under r_cut, each size weighted by the share the hard core keeps
  (Matérn II: exp of minus the density times the volume within reach of a
  larger candidate), times a rock's projected area over its bounding disc's
  (a quarter of its prototype's surface, about 0.59). Tested against the
  scatter's own rocks: the haze under any cut plus the rocks over it add up
  to the same cross-section within 1 %.
- **Dust.** `Density(dust=)`, an optical depth per km at full profile, of
  particles under the smallest rock: never drawn nor solid, only haze. A
  thin ring bright from afar needs an optical depth of a few tenths across
  it; rocks alone would make it a wall up close. The `ring` preset has 0.03
  per km: about 0.2 across it, with its rocks.
- **The integral.** For each pixel where a ray leaves the field's hull, over
  the interval that ends there (a ring's annulus cuts a ray into two), by
  32 midpoints spaced geometrically from where something is missing (8 for
  the veil over a rock). What is missing per metre at each distance is
  tabulated each frame on the CPU (256 entries over the log of the
  distance), so a sample is one lookup whatever the classes.
  `haze.optical_depth` is the same quadrature in numpy; the shader's pixels
  match it within 4 levels of 255. The shaders share `haze.glsl`.
- **What it does not follow.** The clumps (their coverage, averaged) and the
  keep-outs: away from the rocks drawn, a patchy field's haze is even. See
  the backlog.
- **Light.** Lambert spheres of the rocks' mean far colour: full with the
  sun behind the camera, dark against it, plus the sky light; then the
  field's exposure and tonemap. No shadows, not even a planet's on its ring.
- **Drawing.** A hull around the shape (1,024 segments for a ring), drawn on
  both sides with depth test and no depth write, after the sky and clouds:
  a planet in front hides it. Hull points beyond the far plane are pulled
  onto it, so a ring across the sky is drawn whole without pushing the far
  plane out. Only the far camera of a split draws it, from its near plane
  (read from its projection): nearer than the split every rock is drawn, so
  nothing but a little dust is lost there.
- `AsteroidField(haze=False)` leaves it out. In a 15 km field it costs
  nothing seen: nothing is missing nearer than about 57 km.

## In a world

`World(fields=[FieldLook(name, spec, position=, orientation=, ...)])`
builds fields beside the bodies: on the world's worker pool, with its
`asteroids` settings, its sun and clock (the rocks turn with the game's
time), and the bodies' textures. They are part of `World.loading`,
`update`, `set_sun`, `set_settings` and `destroy`; `World.field(name)`
finds one (to make it solid, find clear spots...). `scripts/demo_asteroids.py`
is built this way.

### Ring costs

`scripts/asteroid_bench.py --field ring --offscreen` (HIGH, 1600×900, in the
ring's dense band, October 2026): frames 11.8–12.6 ms median, p95
12.5–22 ms; about 290,000 rocks held (the held budget), smallest drawn 2
pixels; the field's own update 5.3–5.5 ms median. The `cube` preset is
unchanged: p95 3.2–4.0 ms.

## Phases

The design (issue #14) builds the rest on this:

1. The field's definition and its rocks. Done.
2. The rocks' shapes: prototypes from the terrain's noise, at several
   resolutions, generated in the terrain's worker processes. Done.
3. Drawing: hardware instancing, one draw per prototype and level of detail,
   spin in the vertex shader, the terrain's materials and lighting
   (`terrain_material.glsl`, `object_light.glsl`). Done.
4. Collision: a native Panda collision solid for the rocks near the camera,
   on the drawn shapes. Done.
5. SpaceFlight's missions 1 and 2 on a field. Done.
6. Rocks above 200 m (`terrain_min_radius`) turn into real terrain bodies
   near the camera, with the terrain's level of detail. Done.
7. Rings: an annulus with a radial profile, and a haze for the rocks too
   small or too far to draw; fields in a `World`. Done.
