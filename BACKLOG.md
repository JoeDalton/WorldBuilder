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

## The raymarch's steadiness (MARCH_PLAN, Phase 1)

- **A little jitter is left**, at `march_resolution` 0.25 and in far clouds
  most (still 20 km pose: spread 1.6 mean / 9 at p99 in 8-bit steps; the
  other poses 0.3-0.8 / 2-3). It sits on thin edges and holes in the
  cloud, where a ray's jittered steps land on or miss the cloud, and in the
  lighting's noise (the flat colour, `--cloud-debug 8`, is clean; the sun
  march and the powder term, which follows the billowy density, are the
  suspects). Ideas, none tried: the powder term from the density without
  billows; the lighting evaluated at the segment's mean position per
  pixel; a per-pixel blue-noise offset instead of the interleaved gradient
  noise; a lower `MIN_WEIGHT` for far pixels; more samples per segment far
  off (`SEGMENT_SAMPLES`, +0.4-0.9 ms). Measure with `cloud_bench.py
  --stability 30` (still) and the `--turn` / `--move` runs; the
  performance.md §23 "Steadier" tables are the baseline.
- **The field's own octaves carry the value noise's lattice planes** too
  (fainter than the billows' were, and shared with the billboards): a
  rotation of their coordinates would need the CPU twin (`noise.py`),
  `cloud_field.glsl` and the billboards' placement changed together.

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
