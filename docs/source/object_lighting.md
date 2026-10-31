# Lighting on objects

How models (ships, props, anything loaded with `loader.load_model`) are lit
compared with the world, and what it would take to make them match (October
2026). Nothing here is done yet: read it before working on how objects look.
The terrain demo's white test ball (`TEST_BALL_DIAMETER` in
`scripts/demo_terrain.py`, 150 m ahead of the start pose) is there to judge it.

## What the world's shaders do

The terrain, sky, ocean and clouds compute their lighting in physical units,
where 1.0 is display white and a surface can be much brighter than that:
noon sunlight in the demo is about 2.6 (`SUN_COLOR`), so a white surface
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

## What models get

Models have no shader. OpenGL's built-in lighting computes
`ambient + sun × cos(angle)` from the demo's Panda3D lights (a
`DirectionalLight` and an `AmbientLight` on `render`, set each frame from the
light at the camera, `TerrainDemo._update_lights`) and **clips each channel at
1.0**. Nothing else is applied, which shows on the test ball:

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
the camera through the atmosphere (`Atmospheres.light_at`), reddened at sunset
and gone in the planet's shadow. At a 2° sun the ball's lit side is salmon and
its shadow side takes the blue sky light, as expected. Only the conversion to
screen colours differs.

## Making objects match

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
