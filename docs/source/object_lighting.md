# Lighting on objects

How models (ships, props, anything loaded with `loader.load_model`) are lit
compared with the world, and the shader that makes them match: a model
tracked with `World.track_light(node, shader=True)` is drawn by
`world_builder.objects` (see [The model shader](#the-model-shader) below,
part G of [Shadows](shadows.md)). The terrain demo's test balls (the white
one, `TEST_BALL_DIAMETER` in `scripts/demo_terrain.py`, 150 m ahead of the
start pose, and the textured smiley beside it) are there to judge it; M
switches them between the shader and Panda3D's own lighting.

## What the world's shaders do

The terrain, sky, ocean and clouds compute their lighting in physical units,
where 1.0 is display white and a surface can be much brighter than that:
noon sunlight in the demo is about 2.6 (`SUN_COLOR`), so a white surface
facing the sun reflects about 2.6 plus the sky's light. Before a pixel is
written, every one of them applies the same three steps:

1. **Exposure**: the light is multiplied by a camera-like constant,
   `uExposure` (0.9 for the terrain, sky, ocean and planet clouds; the flat
   cloud field has its own, 1.15).
2. **Tone map**: the ACES curve (`aces_tonemap`) maps the unbounded range
   into 0–1 smoothly. Dark values pass almost unchanged, and bright ones roll
   off gently toward white, so 2.0, 2.5 and 3.0 still come out as slightly
   different near-whites. (The sky and the planet clouds keep the hue of
   what is brighter than white, `displayColour` in
   [display.glsl](../../src/world_builder/atmosphere/shaders/display.glsl);
   the flat cloud field uses a soft knee on the brightest channel and no
   display encoding; see [atmosphere](atmosphere.md).)
3. **Display encoding**: the `pow(…, 1/2.2)` at the end of `aces_tonemap`
   turns linear light into what a monitor expects.

For example, [terrain.frag](../../src/world_builder/terrain/shaders/terrain.frag)
ends with `aces_tonemap(C * uExposure)`. Over the ground, they also add the
air between the camera and the surface (aerial perspective, `applyAtmospheres`).

## What models get without it

A model the world does not draw has no shader. OpenGL's built-in lighting computes
`ambient + sun × cos(angle)` from Panda3D lights and **clips each channel at
1.0**. The lights are either the world's (a `DirectionalLight` and an
`AmbientLight` on `render`, set each frame from the light at the camera), or,
for a model tracked with `World.track_light(node)`, its own pair, set each
frame from the light where the model is (`world_builder.lights.TrackedLights`;
the demos track the test ball). Nothing else is applied, which shows on the
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

`world_builder.objects` is that shader (`model.vert`, `model.frag`), with
every world shadow per pixel:

- **Colour.** `p3d_Texture0` (white without a texture) times the vertex
  colour (or `set_color`), the colour scale and the material's diffuse
  (1 without a material), authored as displayed: decoded with
  `pow(2.2)`. The alpha is kept.
- **Light.** `objectLight`, as the terrain lights its ground, with the
  material's roughness (1, matte, without a material: no highlight) and
  the body's vertical for the sky light's "up" (the normal in open space).
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
shader=False)`. By hand (the terrain demo): make a `ModelLighting` per
body, bind its parts to `lighting.inputs`, track the node with
`TrackedLights.track(node, shader=True)` and call `lighting.update` and
`lighting.apply(node)` every frame.

On the test balls (`terrain_shots.py`'s `ball_*` poses, with
`--panda-lights` for Panda3D's lighting): at noon the lit side has a
gradient instead of a flat white disc, at sunset it is a graded warm light
instead of a saturated yellow one, and the shadow side takes the sky's
blue as the ground does.

Cost: the bookkeeping is ~25 µs a frame (two balls); the balls cover a few
thousand pixels. `flight_bench.py --flight low`, four alternating pairs
(mean / p95 ms): the shader 10.2 / 17.3, 9.5 / 15.3, 10.0 / 16.4,
9.9 / 16.2; Panda3D's lighting 9.5 / 14.9, 9.7 / 15.5, 9.6 / 16.0,
9.9 / 16.7: within the runs' spread (on average 0.2 / 0.5 ms more with the
shader). The shots' timed frames differ by less than their noise.

Not done: normal maps, emission and metalness (the material's other
fields), and the shadows models cast (part H). In the clouds'
half-resolution pass a shaded model is drawn whole as an occluder, where
the terrain swaps in a depth-only stage.
