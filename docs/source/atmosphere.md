# Atmosphere, sky and planet clouds

Three systems make a planet look like one from the ground to orbit:

| System | Module | What it does |
|---|---|---|
| Atmospheres | `world_builder.atmosphere` | Single scattering of sunlight by air and haze: sky colour, sunsets, aerial perspective, the limb from space, reddened sunlight. Applied by every shader along its view ray. |
| Sky | `world_builder.sky` | A star field (procedural or painted) and the sun's disc, seen through the atmospheres. |
| Planet clouds | `world_builder.clouds.sphere` | One cloud density field over the whole planet: volumetric billboards near the camera, a deck surface far away, the same pattern at every distance. |

`scripts/demo_terrain.py` wires all three; `scripts/terrain_shots.py` renders
them from fixed poses (`sunset`, `night_orbit`, `in_clouds`, `moonrise`...).

## Atmospheres

### The model

Each atmosphere is an exponential medium around a sphere: Rayleigh scattering
(molecules: blue sky, red sunsets) with an 8 km scale height, Mie scattering
(aerosols: haze, the glow around the sun) with 1.2 km. The defaults are
Earth's air (Bruneton 2017's constants) over a 60 km shell; every value is in
`AtmosphereParams`.

For a view ray, `atmosphere.glsl` marches 8 samples through each atmosphere
the ray crosses. At each sample:

- the density of both media, for the optical depth back to the camera;
- the optical depth toward the sun, **analytically**: the Chapman function in
  its erfcx form, `Ch(x, χ) = √(πx/2) · erfcx(√(x/2) cos χ)`, with a fitted
  `erfcx` within 0.33 %. Against a numerical integral it is within ~1.5 % at
  every altitude and angle; below the horizon it goes through the tangent
  point, and samples in the planet's shadow get no sun;
- the sunlight reaching it, times the density, times the transmittance back
  to the camera.

The phase functions (Rayleigh, and Cornette-Shanks for Mie) then give the
light scattered toward the eye. The result for anything at distance `d` along
the ray is `seen = colour · transmittance + inscatter`, composed far to near
when a ray crosses two atmospheres.

### Where it applies

- **Terrain:** aerial perspective per pixel, and its sunlight goes through
  its own planet's air (red at sunset, nothing in the planet's shadow).
- **Sky:** the scattered light toward each direction, over the stars and the
  sun's disc. The eye adapts: stars fade as `exp(-sky luminance × 60)`, so they
  are gone by day and out at dusk.
- **Other bodies:** a moon is a terrain like any other, seen through the
  planet's air: dimmed and tinted by day, untouched from space.
- **The ocean** (opt-in, `Atmospheres.bind(ocean.ocean_node)` and
  `Ocean.set_atmosphere_frame`): its distance haze is replaced by the
  atmospheres. Unbound, as in SpaceFlight, it is unchanged.
- **Planet clouds:** per vertex for billboards (the air barely changes across
  one, and per pixel it would be paid for every layer of overdraw), per pixel
  for the deck.

`Atmospheres` holds every atmosphere's uniforms in arrays shared by all bound
nodes and rewrites them in place once a frame, in float64 where it matters
(`|camera|² - radius²` would be noise in float32).

### Left out: multiple scattering (the Bruneton option)

Light scattered more than once is missing. Twilight comes out a little dark,
the sky a little dark near the horizon, and lit clouds a little yellow (the
cloud shader puts half the reddening back, standing in for it).

The reference fix is **precomputed atmospheric scattering**, Bruneton &
Neyret 2008 (revised by Bruneton in 2017): transmittance, multiple-scattering
and irradiance tables computed once per atmosphere (a 4D scattering table
packed in a 3D texture, plus two 2D tables), then a handful of texture reads
per pixel instead of the march. It costs a precomputation step (on the GPU,
or offline with numpy and saved), a table layout to get right, and new
shaders; it gives correct twilight, deep-blue upper skies and ground
irradiance, and makes the per-pixel cost constant. The shader interface here
(`applyAtmospheres`, `atmosphereAlong`, `atmoSunlight`) is what such tables
would plug into.

## Sky

`Sky` draws a camera-centred sphere first, behind everything, with the far
camera only (`hide(HIDE_FROM_NEAR)`), and scales it inside the far camera's
range every frame. Its shader shows:

- **the star field**, a cube map: `make_starfield_texture()` builds one with
  numpy at startup (stars with a power law of brightness and colour
  temperatures, more of them toward a Milky Way band with dust lanes, faint
  nebulae; under a second at 1024 px), or `load_cube_map(pattern, loader)`
  loads a painted one (SpaceFlight's `models/skyboxes/purple_#.png`). Rotate
  it with `star_hpr`;
- **the sun's disc**, twice its real size so it reads at any resolution,
  reddened by the air at sunset;
- through every atmosphere, then the terrain's tonemap.

## Planet clouds

### One density field

The billboard field's density (`clouds.noise`) is moved onto the planet:

- the slab is a **height above the sea-level sphere** (`altitude` sets its
  base);
- the tileable noise is sampled at **planet-frame positions in 3D**, so it has
  no seam anywhere on the sphere;
- a **weather** field of direction (`noise.weather_volume`, a 48³ volume over
  the unit cube sampled on the unit sphere) shifts the fbm, so some regions
  are clear and others overcast (`weather_gain`).

`noise.density_on_sphere` is the CPU copy, exact to the shader's texture
filtering, which places the billboards where the shader draws cloud.

### Near: billboard shells

Billboards are generated per **cube-sphere tile** in the shared worker pool
(`generate_cloud_tile`), by rejection sampling of the density, as the flat
field places its own. Shell *k* uses tiles of level `base + k` (8 km at the
first) and billboards twice the size of shell *k − 1*; adjacent shells
crossfade by distance on optical depth (the flat field's `lod_shells`
layout). Each shell costs about half the previous.

Tiles hang under a root moved by `origin − camera` (the terrain's rebasing),
sort back to front between them in their own bin, and re-sort their
particles a few tiles per frame.

### Far: the deck

Billboards cannot reach orbit: at constant volume fraction their covered area
falls as 1/size. Beyond the last shell, the **deck** takes over: a sphere at
mid-slab height whose fragments evaluate the same density through the
slab's depth (thickness / |cos|). Where a pixel spans more than the clouds'
features (from orbit, a pixel is a kilometre or two), the deck fades from the
sampled field to its **expected coverage**, the share of the pixel's columns
that hold cloud, from the field's calibrated statistics and the local
weather. Self-shadowing goes first (it is smaller than the cloud). So the
pattern from orbit is the one you fly into, without the speckle.

### Not yet

- **Wind:** the clouds are static.
- **Shadows on the ground:** clouds do not shade the terrain.
- **Reflections:** the demo keeps clouds out of the ocean's mirror (it would
  draw every billboard twice).
- **One cloud type per planet** (cumulus in the demo); the flat field mixes
  types, the planet field could take one per slab.
