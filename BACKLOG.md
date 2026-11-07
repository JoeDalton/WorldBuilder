# Backlog

Features found missing but not made, so the work could move on. Each item
says where it came from. Performance work has its own ranked list,
[docs/source/performance_backlog.md](docs/source/performance_backlog.md);
the asteroid fields', [docs/source/asteroids_backlog.md](docs/source/asteroids_backlog.md).

## Models (shadows' part G, #13)

The model shader (`world_builder.objects`, [Lighting on
objects](docs/source/object_lighting.md)) lights a model's colour, normal
and roughness. Left out:

- **Normal maps.** A model's normal map (Panda's `p3d_Texture1` with a
  normal-map texture stage) is ignored: the vertex normal lights it.
- **Emission.** `p3d_Material.emission` is ignored: a ship's lights or a
  glowing panel are lit like the rest.
- **Metalness.** `p3d_Material.metallic` is ignored: a metal is lit as a
  dielectric (the gloss highlight takes the sun's colour, not the
  albedo's).
- **Light bounced off the ground.** The sky light falls to a tenth on a
  face turned down (`objectLight`'s `0.55 + 0.45 N.up`, the terrain's): a
  ball's underside is near black over bright ground. A term of the ground's
  albedo under the sun, from below, would lift it.
- **The clouds' half-resolution pass.** There a shaded model is drawn whole
  as an occluder (only its depth is kept), where the terrain swaps in a
  depth-only fragment stage (the `PASS_TAG` tag state). A tag and a
  depth-only model stage would do the same.
- **The asteroids.** Their rocks and haze take the bodies' shadows (part
  I: a planet's across its ring), but not the world's other terms (the
  objects' shadows, a ring's own).

## Objects' shadows (shadows' part H, #13)

The casters' map (`world_builder.shadows.casters`) works as planned; left
out:

- **On the CPU.** `World.lights_at` (Panda3D's lights at the camera, a
  tracked node's own) does not see the objects' shadows: a model drawn by
  Panda3D's lighting under a ship stays lit. A read-back of the map at a
  few points, or rays against the casters' spheres, would do it.
- **Casters far apart.** One map spans every caster: two ships kilometres
  apart share its texels (a ship's shadow goes soft). Splitting the
  casters into clusters (a map each, or an atlas) would keep them sharp.
- **The clouds and the sky** do not receive them: a ship over the deck
  casts nothing on it. A fetch per billboard vertex (as the mountains')
  would do it.
- **Cut-outs.** The depth-only stage ignores textures: a model whose alpha
  cuts holes (a lattice) casts a solid shadow.
- **The rocks.** An asteroid neither casts nor receives them (see Models).
- **Casters behind the camera.** A caster whose shadow is too small to see
  is left out, but one near the camera whose shadow falls behind it still
  costs the pass. Testing its shadow's cylinder against the view's
  frustum would save it.

## Rings (shadows' part I, #13)

- **A ring's own shadow on itself.** Its rocks and haze are not shaded by
  the ring between them and the sun (at a low sun across it, its far side
  should be dimmer).
- **The sun through the planet's air.** On its way to the ring, the sun
  grazing the planet's limb is not reddened: the planet's shadow on the
  ring has a penumbra but no red edge.

## Clouds (shadows' part F, #11)

- **The far cloud deck's rim.** The billboards at the deck's distant rim
  show their rectangular edges; the long shadows (`cloudFarDepth`, one value
  per billboard) darken some of them, which makes them stand out a little
  more at dusk (`dusk_in_clouds`).

## The raymarch (MARCH_PLAN)

The raymarch (`CloudSettings.renderer = "march"`, `clouds/march.py`) draws
the low clouds, with genera (two families in one pass), casts their shadows
on the ground, sea and models, and lights itself from the cloud shadow maps
where they hold it. Phases 0-3 are done (MARCH_PLAN.md); Phase 4 (speed,
then the default) is measured, not built. What is left:

### Steadiness

- **Jitter is mostly gone** (Phase 4.1: still p99 under 2.5 levels at HIGH and
  ULTRA, moving grid under 0.05); what is left is the genera at LOW (24 steps
  over a 10 km slab: still p99 6-7) and a softer cumulus at LOW. Before 4.1 it
  was, at the 20 km pose, 1.6 mean / 9 at p99. It sits on thin edges and holes in the
  cloud, where a ray's jittered steps land on or miss the cloud, and in the
  lighting's noise (the flat colour, `--cloud-debug 8`, is clean; the sun
  march and the powder term, which follows the billowy density, are the
  suspects). Ideas, none tried: the powder term from the density without
  billows; the lighting evaluated at the segment's mean position per
  pixel, or only where alpha x transmittance matters (also a speed idea,
  below); a per-pixel blue-noise offset instead of the interleaved gradient
  noise; a lower `MIN_WEIGHT` for far pixels; more samples per segment far
  off (`SEGMENT_SAMPLES`, +0.4-0.9 ms). Measure with `cloud_bench.py
  --stability 30` (still) and the `--turn` / `--move` runs; the
  performance.md §23 "Steadier" tables are the baseline. Tried since and
  dropped (performance.md §24, "Tried and dropped"): a space-time blue
  noise for the jitter (no change), EVE-Redux's exact-pixel reconstruction
  (a grid, steadiness halved), a Catmull-Rom history read (softer, more
  flicker).
- **Sky occlusion from a zenith map: not built** (the EVE salvage plan's
  Phase 4 item 2). The cloud shadow maps' bake marched straight up would
  give each march sample the optical depth above it. But the CPU twin puts
  the genera's convective towers optically *thinner* than an overcast
  cumulus deck (median vertical depth at the base 0.8 against 4.9; p90 3.0
  against 6.9), so one depth-to-sky-light scale that keeps a cumulus base
  as today lightens tower bases; the height ramp (`skyReach`) already follows
  each column's own top. Its gain would be the stratus under an anvil and
  holes in a deck. **Done by the light volume's sky channel** (RENDERING_PLAN.md C.3): the march's sky
  reach is `mix(skyOcclusion, 1, exp2(-depth up))` where the volume holds the sample, the ramp elsewhere
  (`--cloud-debug 512`: the ramp). Watch the tower bases against a cumulus deck's (the twin's warning above)
  in the user's flight checks.
- **Softer in motion at an eighth (4 x 2, HIGH)**: sharpness 0.92-0.94
  plain, 0.75-0.85 in towers, against 0.95-0.97 and 0.84-0.94 at a half.
  Not the jitter, the bilinear mix of the fresh samples or the history's
  filter (all three measured, §24). Ideas, not tried: the history's lag in
  motion (`MIN_WEIGHT`, the clamp's width without `uStill`); tracing more
  pixels a frame while the view moves (a half while turning, an eighth when
  still: the march's cost is in its traced pixels).
- **The clouds lag behind the camera against terrain, when it flies fast.**
  Seen at the demo's camera `{lat: 20.18986, lon: 29.12635, alt: 2769,
  heading: 157.2, pitch: -5.4}`, roll 5.1 (`terrain_shots.py --pose at --at
  20.18986 29.12635 2769 157.2 -5.4`), advancing toward a mountain: the
  cloud's update lag is very visible against it. The same when descending
  fast from orbit. Not looked at yet. Suspects: the history's reprojection by
  the cloud's own distance (`cloud_resolve.frag`) against a near, hard
  edge (the terrain's depth is not part of it), the composite's depth test
  at the view's resolution, and `uStill` (clamp floor in motion). Measure
  with a `cloud_bench.py --move` flight over a terrain pose, the last moving
  frame against the converged one.

### Look

- **Far cloud detail against shimmer** (the shapes lost in Phase 4.1 are back,
  partly): `STEP_FADE` (features finer than a share of the step fade to their
  means) was 0.4 everywhere and the steps are long (the budget lengthens them,
  and 1 % of the distance farther out), so the lumps and the 125 m octave
  faded almost everywhere: soft white lumps even at ULTRA (confirmed by
  rendering with the fade off). It is now `march.STEP_FADES`: 0 at full
  resolution, 0.1 at a half (HIGH, ULTRA), 0.25 at a quarter (LOW). At HIGH the
  20 km shimmer is p99 4.8 levels (9.1 before 4.1, 2.4 with the fade at 0.4);
  the far clouds are crisp again. LOW is softer by design (24 steps at a
  quarter). A better trade may exist: fade by the distance-driven part of the
  step only (done 2026-11-07: the step deepened in a cloud and cut at a
  segment's end drew contours on towers), or only beyond about 5 km, or a
  variance-driven fade in the resolve.
- ~~**The field's own octaves carry the value noise's lattice planes**~~ Done
  (2026-11-07): the two finest are turned (`noise.OCTAVE_TURNS`, the shader's
  `OCTAVE_n_TURN`), twins and placement with them: they drew interference lines
  on towers seen at a grazing angle.
- ~~**Faint horizontal bands in towers** lit from the maps' 4 levels~~ Gone
  with the light volume (C.3), which holds the depth at every voxel. (The
  stacked bands on a tower seen level that remain are the march's steps, not
  its light: RENDERING_PLAN.md "After C".)
- **The map-lit sun depth counts out to the sun march's reach only**
  (`cloudMapSunDepth`'s window): counting a tower's whole 10 km chord left
  it black, and the levels are too coarse for an anvil. `--cloud-debug 64`
  takes the lighting from the sun march alone, to compare. A longer reach
  needs more levels or another representation. With the light volume (C.3)
  the window is kept, from the volume; its whole chord (`--cloud-debug 1024`)
  still blackens towers: a longer reach needs the light scattered more than
  once in the towers, not more resolution.
- **Low sun (about 4 degrees):** the stratus under the convective clouds is
  shaded by the ones sunward (`cloudSunDepthOther`) with hard-looking
  edges (`terrain_shots.py --pose at --at 24.48163 33.77925 5019 -49.1 8.3
  --sun 270 4 --genera`). Not investigated.
- **The sea's reflection shows the convective family's deck only** (the
  stratus is missing), and no clouds' detail: accepted for now. The fix is a
  cheap mirror march (clouds.md "The raymarch": a third march layer from the
  reflection camera, a quarter of its resolution, about 24 steps, no history,
  about 0.3-0.6 ms while the sea is in view; not built).
- **Artifacts in the sea's reflection of the deck** (seen with the march as
  the default, HIGH, `terrain_shots.py --pose at --at 24.49453 35.64275 2029
  19.4 -32.1`, 2 km up over the sea; not reproduced exactly: the weather
  differs). Not investigated, only read from the code:
  - *Stone-like contour lines* inside the light patches: the deck's density
    is the mean of two field samples (at 30 % and 70 % of the slab,
    `cloud.frag` `uDeck == 1`), so a column holding cloud at one height only
    gets half the density and its outline shows. Fix idea: a smoother
    density (one sample, or a smoothed mean) for the deck.
  - *A straight diagonal step in brightness* with a darker triangle behind
    it: probably the terrain shadow on the deck's sunlight (`terrainShadow`
    in `cloud.frag`: the local shadow map is a square around the camera, and
    the reflection views the deck from below). Alternatives: a cube-face seam
    of the weather map, the reflection camera's culling margin. To confirm:
    the same pose with `--shadows-quality low`, then with
    `--cloud-renderer billboards`. Fix idea: the deck takes the global
    terrain shadow only.
  - *Fine diagonal hatching* at the bottom: the ripples undersampled by the
    half-resolution reflection buffer (`reflection_scale`, 0.5 at HIGH); the
    billboards' reflection has it too.
- **With genera the march keeps at most 3 distance bands per family**
  (`MAX_LAYERS` is shared by the two); HIGH asks for more. An octaves-per-band
  table in a small uniform would free layers.
- `noise.sun_depth_from_map`'s local part ignores the convective depth and
  the shelf (the cumulus-only test covers it); optional and never done: the
  billboards' `cloudFarDepth` from the maps.

### Speed (Phase 4.2: measured, partly built)

On the RTX 3070 at 2560 x 1440, HIGH (march at half resolution, 64 steps),
the frame is GPU-bound (the CPU's `step()` is 3.7 ms). Pipelined, with no
clouds 2.7-4.0 ms; the march 6.2-9.2 ms, with genera 8.8-12.8 ms; the
budget is 6.25 ms (160 fps), so the clouds should take about half of what
they do. The cloud time:

- the march pass 75-80 % (the far range's layer is nearly all of it; the near
  layer adds about 0.7 ms of fixed cost even over mostly empty sky);
- the resolve 0.3-0.5 ms, the depth pre-passes 0.1-0.5 ms (and 0.4-0.5 ms of
  CPU: the terrain is drawn again per range), the composite 0.05-0.4, the
  cirrus 0.15-0.5, the ground shadows 0.3-1.5;
- inside the march pass (start pose, half resolution): fixed passes 1.26 ms,
  per-pixel set-up 0.7, lighting 0.57, billows 0.7, the steps the rest.
  **88-98 % of the steps are in empty air.** The weather read per sample
  with genera costs 0.3-0.8 ms. The atmosphere and the flat colour cost
  nothing measurable, and the clouds' Python is nothing.

Built in 4.2 (small): the billows share one density evaluation with the
column (`cloudColumnAt`: 0.1 ms), the resolve early-outs in clear sky (0.1-0.2 ms),
the march stops at 4 % transmittance and its steps grow in dense cloud
(0.1-1.2 ms inside clouds: `inside` 5.1 -> 4.7, `model_in_cloud` 5.5 -> 4.3 at
half resolution, same look). **Tried and dropped:**

- **Skipping clear air by a bound** (the first octave, the others at their
  greatest, the slope of the lattice): the bound holds in 0-7 % of the steps with
  margins under a step; slower than not skipping.
- **An occupancy map** (a column map around the camera, baked on the GPU, per
  column the heights where the field may hold cloud; steps outside it skipped by
  a ray-DDA): it cleared 38-81 % of the steps and cut the loop's iterations by
  26-30 %, but the time by only 5-8 % (0.1-0.4 ms). The cost is not the iterations:
  a GPU warp (32 pixels) runs as long as its busiest lane, so a clear pixel next
  to a cloudy one gains nothing, and cloudy steps (lighting, billows, four
  octaves) cost several empty ones. Not worth its bake, its chart and the risk of
  clipped wisps; the code is in the history of this session only.

So the remaining time is in the cloudy pixels' work and the passes, not in the
empty air. Ideas below are re-ranked by that: (the first of them is the one the
measurements no longer support.)

Ideas, by expected gain (status after 4.2 and 4.3):

1. ~~Skip empty air~~ (tried, 0.1-0.4 ms). Work per *cloudy* step and per
   warp is what is left: billows only at the edge, a cheaper sun depth, and
   tracing at a lower resolution where the view is all sky (a coarse pre-pass
   to find the pixels that see cloud). Not tried.
2. ~~Fewer fixed passes~~: measured small (`--passes` at HIGH: depth 0.08,
   resolve 0.3, composite 0.14 ms), so a half-resolution depth pass or the
   composite folded into the resolve would save about 0.1 ms each. The near
   layer already stops when the slab is out of range. A compute-shader
   resolve: not tried (only if the resolve grows). **Since then** (performance.md §24): one
   march over both ranges, and the depth passes at half resolution, which
   saves 0.45-0.55 ms in a game's scene (Mission 3) though cloud_bench saw
   0.08.
3. **Cheaper set-up per pixel** (0.2-0.4 ms, measured in 4.0): the slabs'
   constants as uniforms, not repeated `getField` unpacks. Not done.
4. ~~Billows~~ done (shared column, 0.1 ms).
5. ~~Lighting reuse~~ tried: every other cloudy step reuses the last's
   radiance, 0.1 ms or less (start 3.78 -> 3.78, inside 4.91 -> 4.78):
   dropped.
6. **Resolution policy (open decision):** a quarter meets the targets
   (plain 2.1-2.3 ms, genera 2.5-3.1 ms at 64 steps, 1440p) and a half does
   not (3.8-4.9, genera 6.8-8.1 ms), at HIGH's price: error against the
   reference about 3x, moving flicker 2-3x, edges softer. MID went to a
   quarter in 4.3; HIGH and ULTRA are at a half. Not tried: a quarter in
   the far range and a half in the near; a denoise (an edge-aware blend in
   the resolve, MARCH_PLAN 4.1.5, and a variance-aware history, 4.1.4:
   neither was built, the smoothing and the Gaussian statistics replaced
   them) to make a quarter look like a half; and a HIGH that is a quarter
   only when the frame is over budget. A game at 160 fps needs one of them.
7. **Ground shadows** (0.3-1.5 ms): re-bake the maps less often
   (`CloudShadowMaps.RECENTRE`, the sun's steps), drop the far map's blend.
   A weather read every 100 m with genera saves 0.3-0.5 ms (check for
   bands). Neither done.

Billboards stay faster over a distant sky (genera `cb_far` 1.35 against
3.7 ms at half resolution). `CloudSettings.renderer` now defaults to
`"march"` at every preset (4.3); billboards are selectable.

The jitter that remains: slightly visible at LOW and MID (a quarter), the
genera at LOW most (p99 6-8 levels still); the user judged it acceptable.

### Docs and wiring

- No test runs the march's GLSL: the shaders are checked as text (the constants
  they share with Python, `marchDensity`'s single field read, the resolve's
  clear-sky early-out) and through the NumPy mirror (`march.march_ray`: the
  opaque stop and the deepening steps against the fine integral). A small
  offscreen harness that renders a march pass on a fixed weather and reads it
  back (clear sky -> alpha 0, a dense cloud -> alpha 1, the billows' column
  equal to `cloudDensityAt`) would close it.
- The hand-built demo (`scripts/demo_terrain.py`) wires the clouds' shadows
  and maps itself beside `World`'s (`bind_shadow_inputs`, `bind_maps`): keep
  the two in step.
