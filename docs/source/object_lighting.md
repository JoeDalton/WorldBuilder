# Lighting on objects

How models (ships, props, anything loaded with `loader.load_model`) are lit
compared with the world, and the shader that makes them match: a model
tracked with `World.track_light(node, shader=True)` is drawn by
`world_builder.objects` (see [The model shader](#the-model-shader) below,
part G of [Shadows](shadows.md)). A game's models are the way to judge it: track one with
`World.track_light(node, shader=True)` and with `shader=False` (Panda3D's own
lighting) and compare.

## What the world's shaders do

The terrain, sky, ocean and clouds compute their lighting in physical units,
where 1.0 is display white and a surface can be much brighter than that:
noon sunlight in the shipped worlds is about 2.6 (`sun.color`), so a white surface
facing the sun reflects about 2.6 plus the sky's light. Before a pixel is
written, every one of them applies the same three steps:

1. **Exposure**: the light is multiplied by a camera-like constant,
   `uExposure` (0.9 for the terrain, sky, ocean and clouds).
2. **Tone map**: the ACES curve (`aces_tonemap`) maps the unbounded range
   into 0–1 smoothly. Dark values pass almost unchanged, and bright ones roll
   off gently toward white, so 2.0, 2.5 and 3.0 still come out as slightly
   different near-whites. (The sky and the clouds keep the hue of what is
   brighter than white, `displayColour` in
   [display.glsl](../../src/world_builder/atmosphere/shaders/display.glsl);
   see [atmosphere](atmosphere.md).)
3. **Display encoding**: the `pow(…, 1/2.2)` at the end of `aces_tonemap`
   turns linear light into what a monitor expects.

For example, [terrain.frag](../../src/world_builder/terrain/shaders/terrain.frag)
ends with `aces_tonemap(C * uExposure)`. Over the ground, they also add the
air between the camera and the surface (aerial perspective, `applyAtmospheres`).

## What models get without it

A model the world does not draw has no shader. OpenGL's built-in lighting computes
`ambient + sun × cos(angle)` from Panda3D lights and **clips each channel at
1.0**. The lights are either the world's (a `DirectionalLight` and an
`AmbientLight` on `render`, set from the light at the camera), or, for a
model tracked with `World.track_light(node)`, its own pair, set from the
light where the model is (both every `World(light_period=N)` frames) (`world_builder.lights.TrackedLights`;
a game tracks its models). Nothing else is applied, which shows on the
test ball:

- **Clipping**: with sunlight at about 2.4, every part of a white object facing
  within about 75° of the sun exceeds 1.0 and clips to the same pure white.
  At noon the ball's lit half is flat white, with no gradient until the very
  edge of its shadow.
- **No display encoding**: the linear value goes straight to the screen, so
  mid-tones (the shadow side, dusk light) come out darker and more contrasty
  than the same light on the terrain.
- **No air in front**: no aerial perspective. It doesn't matter at 150 m, but a
  distant model would not fade into the horizon as the ground does.

The **colour** of the light is right: the lights take the sunlight reaching
the camera, or the tracked model, through the atmosphere
(`Atmospheres.light_at`), reddened at sunset, gone in the planet's shadow,
and dimmed in the world's shadows (`World.sun_visibility`: so far the
eclipses, see [Shadows](shadows.md)). At a 2° sun the ball's lit side is salmon and
its shadow side takes the blue sky light, as expected. Only the conversion to
screen colours differs.

## Why a shader of their own

Panda3D's auto-shader (`render.set_shader_auto()`) does not help on its own:
it lights per pixel but does no tone mapping. Models would need a small shader
of their own that:

- lights with the same light values (the sun's colour and direction, the
  ambient), unclipped;
- ends with the same exposure and tone map as the terrain (`uExposure`,
  `aces_tonemap` from `atmosphere/shaders/display.glsl`, which the terrain
  includes too, so the two cannot drift);
- optionally applies the atmosphere between the camera and the model
  (`atmosphere.glsl`'s `applyAtmospheres`, as the terrain does), for distant
  objects.

The terrain's own lighting and textures are includes such a shader can share:
`terrain/shaders/object_light.glsl` (`objectLight`: Lambert, the gloss
highlight and the sky light, in linear light) and
`terrain/shaders/terrain_material.glsl` (`layeredMaterial`: the biome texture
layers blended by weight and height, with the rock projected from the sides
on steep ground). The asteroids (issue #14) are drawn with both.

Textured models would also need their textures decoded to linear light first
(`pow(texel, 2.2)`) so they come out of the display encoding as authored.

## The model shader

`world_builder.objects` is that shader (`model.vert`, `model_main.glsl`
through one `model_<tier>.frag` per shading tier, see
[PBR](#physically-based-shading) below), with every world shadow per pixel:

- **Colour.** The base colour's texture (`p3d_TextureModulate`, white
  without one) times the vertex colour (or `set_color`), the colour scale
  and the material's base colour (the larger of `baseColor` and `diffuse`:
  1 without a material, `baseColor` for a PBR one, whose `diffuse` its
  metalness darkens). A model authored as displayed is decoded with
  `pow(2.2)`; a glTF one is in linear light already (its colour textures
  are sRGB, which the GPU decodes), so `World.track_light` marks its parts
  (`objects.mark_model`, a `uBaseLinear` input on each geom's state) and they are not decoded
  again. The alpha is kept.
- **Light.** At the simple tier, `objectLight`, as the terrain lights its
  ground, with the material's roughness (1, matte, without a material: no
  highlight) and the body's vertical for the sky light's "up" (the normal
  in open space). At the PBR tiers, below.
- **The sun** at each pixel (`model_light.glsl`'s `modelSunlight`): the sun
  above the air, through the body's air (`atmoSunlight`: reddened low, gone
  past its terminator), the other bodies' eclipses, an airless body's own
  horizon (`shadows.glsl`'s `ownShadow`, with the sun's penumbra), the
  body's clouds' shadow (`cloudShadow`, through its shadow maps) and its
  mountains' (`terrainShadow`, in the body's own axes). Its CPU twin is
  `World.lights_at`'s direct light (`Shadows.own_shadow` for the airless
  horizon).
- **The sky light** is the node's own Panda3D ambient light, which
  `TrackedLights` sets to the sky light where the model is (eclipses dim
  it), less `light_scale`. A node with no light at all would read 1: the
  shader is for tracked nodes.
- **Then** the air between the camera and the model (`applyAtmospheres`),
  the body's exposure, and the ACES tonemap: the terrain's.

How a model gets its inputs: they belong to the body it is near (its air,
its maps, its clouds). `ModelLighting` holds them on a node of its own,
bound as the body's terrain root is (`World` makes one per body, and one
for open space); every `update`, `World` gives each shaded node the
`ShaderAttrib` of its nearest body's (`ModelLighting.apply`), so a texture
a part swaps in (a new mountain bake) reaches it, and a model flying from
the planet to the moon changes body. The node's own shader state is set
aside, and put back by `untrack_light` or `track_light(node,
shader=False)`. Without a `World`: make a `ModelLighting` per
body, bind its parts to `lighting.inputs`, track the node with
`TrackedLights.track(node, shader=True)` and call `lighting.update` and
`lighting.apply(node)` every frame.

On a test ball (the white ball, with Panda3D's lighting for comparison): at noon the lit side has a
gradient instead of a flat white disc, at sunset it is a graded warm light
instead of a saturated yellow one, and the shadow side takes the sky's
blue as the ground does.

Cost: the bookkeeping is ~25 µs a frame (two balls); the balls cover a few
thousand pixels. `flight_bench.py --flight low`, four alternating pairs
(mean / p95 ms): the shader 10.2 / 17.3, 9.5 / 15.3, 10.0 / 16.4,
9.9 / 16.2; Panda3D's lighting 9.5 / 14.9, 9.7 / 15.5, 9.6 / 16.0,
9.9 / 16.7: within the runs' spread (on average 0.2 / 0.5 ms more with the
shader). The shots' timed frames differ by less than their noise.

Not done (logged in the repository's `BACKLOG.md`): occlusion for models
without a glTF occlusion map, the ground's own albedo under a model, and a
depth-only stage in the clouds' half-resolution pass.

## Physically based shading

panda3d-simplepbr cannot draw the models as it is: `simplepbr.init()` puts
its own shader on all of `render`, tonemaps in a post-process pass of its
own, and lights by Panda3D lights alone: no world shadows, no air, a second
tonemap over the world's. Its BRDF and its glTF material inputs are
derived instead (`terrain/shaders/pbr_light.glsl`, shared with the ground's
own BRDF, `TerrainSettings.pbr`; after simplepbr 0.13,
itself after Khronos' glTF-Sample-Viewer), inside the model shader, so the
sun, the shadows, the air and the tonemap stay the world's:

- **The sun**: GGX distribution, height-correlated Smith-GGX visibility,
  Schlick's Fresnel (simplepbr's spherical-gaussian form), the roughness
  floored at Filament's 0.045. In the world's units: `objectLight` lights a
  white Lambert surface to the sunlight itself, so the diffuse term drops
  simplepbr's 1/pi and the specular term takes pi. A rough dielectric is as
  bright as at the simple tier (`test_a_rough_dielectric_looks_alike_at_every_tier`).
- **The material**, glTF's as panda3d-gltf loads it: `baseColor`,
  `metallic` and `roughness` times the metal-roughness texture
  (`p3d_TextureSelector`: G roughness, B metal), `emission` times its
  texture (`p3d_TextureEmission`), added as linear light before the air.
- **The sky light**: diffuse as `objectLight` takes it (all of it facing up,
  a tenth facing down); specular, the environment along the reflected ray
  through Karis' analytic split-sum fit (`EnvBRDFApprox`: no lookup
  texture). The environment is the sky light, as on a surface facing the
  ray, so a metal never reflects black under a sky.
- **Normal maps** (`p3d_TextureNormal`, two-channel ones too), in the
  model's tangents (`p3d_Tangent`: panda3d-gltf computes them), or a frame
  from the screen's derivatives (Schüler's) where a model has none.
- **The sky's reflection** (the top tier): inside an air, the sky-view
  table (`atmosphere/shaders/sky_view.glsl`, one read) along the reflected
  ray, sharp on a smooth surface, the even sky on a rough one and below the
  horizon (the table holds the air there, not the ground); the even sky in
  space.

The tiers (`ModelSettings.shading`, the `models` section, live: the next
`World.update` gives each model the new shader):

| Tier | Preset | What |
|---|---|---|
| `SIMPLE` | | `objectLight`, as the terrain lights its ground |
| `PBR` | | the BRDF, metal, roughness, emission |
| `PBR_NORMAL` | | and normal maps |
| `PBR_SKY` | LOW, MID, HIGH | and the sky's reflection |
| `PBR_FULL` | ULTRA | and occlusion maps, the blurred sky, the ground's bounce |

The cheaper tiers cost too little to set the presets apart (see the cost
below), so LOW to HIGH share `PBR_SKY` and stay choices of the knob; the
section reads as the highest preset it equals
(`QualitySettings.DISTINCT_PRESETS`), so its menu and F8 toggle High and
Ultra.

`PBR_FULL` adds:

- **glTF's occlusion maps**, which panda3d-gltf does not load:
  `objects.gltf_occlusion` reads them back from the model's .gltf (its
  `ModelRoot`'s path) by material name. A map that is the material's
  metal-roughness texture is its red channel (glTF's ORM packing); another
  is loaded on its own. `mark_model` gives each geom its map, mode and
  strength; the occlusion darkens the sky light, and the reflected one by
  Lagarde's specular occlusion. A model without a map has none (of
  SpaceFlight's ships: the tie-interceptor and tie-bomber packed, the
  y-wing's exterior on its own).
- **The sky blurred by the roughness**: the sky-view table is mipmapped
  after each bake (`SkyView`; the sky and the stars read its first level),
  and a reflection reads it at level `roughness x 5` (a 256-texel table
  blurred to 8 texels; past it, its two halves would meet). Rough metal
  keeps its sky's gradient, the even sky only at the roughest.
- **The ground's bounce**: the ground under a model, of albedo
  `ModelLighting.set_ground_albedo` (`GROUND_ALBEDO`, 0.2), lit by the
  sunlight where the model is (its own Panda3D sun, which `TrackedLights`
  sets in every world shadow) and the sky, sends its light up: on the
  diffuse from below (`0.5 - 0.5 N.up`) and in the environment a
  downward reflection sees. None in open space.

To look at a model under them, track it with `World.track_light(node,
shader=True)` and fly to it in `demo_world.py` (F8 cycles the models' preset).

Cost, the worst case: SpaceFlight's a-wing cockpit (metal-roughness,
normal map and emission) covering about 95% of a 2560 x 1440 frame
(`terrain_shots.py --offscreen --size "2560 1440" --pose ship_fill --model
.../a-wing/cockpit/scene.gltf --models-quality L --frames 300`, RTX 3070),
four rounds through the tiers in turn, the median frame in ms:

| Tier | Rounds | Mean |
|---|---|---|
| LOW | 10.02, 9.95, 10.74, 9.96 | 10.2 |
| MID | 11.50, 12.00, 10.62, 10.63 | 11.2 |
| HIGH | 9.95, 9.87, 10.64, 10.77 | 10.3 |
| ULTRA | 11.08, 11.06, 10.02, 10.94 | 10.8 |

Those rounds were run with LOW to ULTRA as `SIMPLE`, `PBR`, `PBR_NORMAL`
and `PBR_SKY`. `PBR_FULL` (ULTRA now) against `PBR_SKY` (HIGH now), the same
pose, four rounds in turn: HIGH 12.93, 10.92, 11.35, 11.05 (mean 11.6),
ULTRA 11.74, 11.27, 10.85, 11.16 (mean 11.3): no difference past the noise.
The cockpit has no occlusion map; one map is one more texture read.

MID above HIGH (which does all MID does) puts the noise at about 1 ms: the
PBR tiers cost at most 0.5-1 ms with a model over the whole screen, and
nothing measurable at a ship's usual size (the tie-interceptor whole, the
`ship` pose, within the same noise). Most of the model shader's time is
still its world shadows (`modelSunlight`), at every tier.

A model tracked with `cast=True` also casts shadows (part H of
[Shadows](shadows.md)): on the terrain and the sea, on the other shaded
models and on itself.
