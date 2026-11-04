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
  performance.md §23 "Steadier" tables are the baseline.
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
  step only, or only beyond about 5 km, or a variance-driven fade in the resolve.
- **The field's own octaves carry the value noise's lattice planes** too
  (fainter than the billows' were, and shared with the billboards): a
  rotation of their coordinates would need the CPU twin (`noise.py`),
  `cloud_field.glsl` and the billboards' placement changed together.
- **Faint horizontal bands in towers** lit from the maps: a map texel holds
  4 levels (the total and a quarter, a half, three quarters of the chord),
  linear between them. 8 levels (a second render target, as the resolve's
  aux plane; MARCH_PLAN's K) would remove them.
- **The map-lit sun depth counts out to the sun march's reach only**
  (`cloudMapSunDepth`'s window): counting a tower's whole 10 km chord left
  it black, and the levels are too coarse for an anvil. `--cloud-debug 64`
  takes the lighting from the sun march alone, to compare. A longer reach
  needs more levels or another representation.
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

Ideas, by expected gain:

1. ~~Skip empty air~~ (above: tried, 0.1-0.4 ms). What could still help is work per
   *cloudy* step and per warp: lighting reused across steps, billows only at the
   edge, a cheaper sun depth, and tracing at a lower resolution where the view is
   all sky (a coarse pre-pass to find the pixels that see cloud).
2. **Fewer fixed passes** (0.8-1.2 ms): the resolve early-outs where the four
   nearest samples and the history are empty (or is a compute shader with
   shared memory); no near layer when it has nothing to draw; the march's
   depth at half resolution (the composite tests at full resolution anyway);
   the composite folded into the resolve.
3. **Cheaper set-up per pixel** (0.2-0.4 ms): the slabs' constants as uniforms,
   not repeated `getField` unpacks.
4. **Billows** (0.3-0.5 ms): one density evaluation, its threshold shifted
   twice (`marchDensity` evaluates the field twice per cloudy sample).
5. **Lighting** (0.2-0.4 ms): only where alpha x transmittance matters, or
   reused across adjacent steps in a cloud.
6. **Resolution policy:** a quarter in the far range, a half in the near
   (a quarter everywhere saves about 1.8 ms at the start pose, at a cost in
   motion).
7. **Ground shadows** (0.3-1.5 ms): re-bake the maps less often, drop the
   far map's blend. A weather read every 100 m with genera saves 0.3-0.5 ms
   (check for bands).

Billboards stay faster over a distant sky (genera `cb_far` 1.35 against
3.7 ms at half resolution). `CloudSettings.renderer` still defaults to
`"billboards"` (MARCH_PLAN's switch is not made); the demo's
`--cloud-renderer march` picks the march.

### Docs and wiring

- performance.md "The raymarch with genera" is labelled quarter resolution
  but was measured at HIGH's half: relabel (at a quarter: between_towers
  about 3.0, under_anvil about 3.3 ms of cloud).
- The hand-built demo (`scripts/demo_terrain.py`) wires the clouds' shadows
  and maps itself beside `World`'s (`bind_shadow_inputs`, `bind_maps`): keep
  the two in step.
