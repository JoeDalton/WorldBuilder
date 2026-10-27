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
- **The asteroids.** Their shaders include `shadows.glsl` but take none of
  its terms: a rock in a ring is not eclipsed, and not in its planet's
  shadow per pixel. Wiring them is the plan's small follow-up (part G, #14).

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

## Clouds (shadows' part F, #11)

- **The far cloud deck's rim.** The billboards at the deck's distant rim
  show their rectangular edges; the long shadows (`cloudFarDepth`, one value
  per billboard) darken some of them, which makes them stand out a little
  more at dusk (`dusk_in_clouds`).
