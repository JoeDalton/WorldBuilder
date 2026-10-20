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
- the **weather** (below) shifts the fbm by direction, by exactly what gives
  the coverage it asks for there.

`noise.density_on_sphere` is the CPU copy, exact to the shader's texture
filtering, which places the billboards where the shader draws cloud. The
field's code is one include, `clouds/shaders/cloud_field.glsl`, shared by the
fragment stage and the billboards' vertex stage.

### The weather

`clouds.weather` builds a cube map (6 × 256², ~3.7 km a texel on a 600 km
planet) of **target coverage**, the share of the slab's columns that hold
cloud, from the planet's own `PlanetSpec` and a `WeatherSpec` (seed and
tunables):

| Component | What it does |
|---|---|
| Zonal circulation | A curve of latitude: wet equatorial band (~0.7), dry subtropical highs (~0.15 at 20–35°), storm tracks (~0.55 at 50–60°), polar caps (~0.4). `season` moves it north or south. |
| Extratropical cyclones | A few in the storm tracks (6 on the 600 km planet, scaled by area in features). Each is a vortex that winds the mesoscale texture into spiral arms (counter-clockwise in the north, clockwise in the south), plus a comma of cloud along one wound arm: its head curls poleward of the low, its tail trails equatorward and west as the cold front, with clear air behind it. |
| Tropical cyclones | One or two over open sea at 10–25°: a tight two-armed spiral, an eyewall and a clear eye. None on a planet without sea. |
| Mesoscale texture | fbm (160 km and finer) wound by the same vortices: ragged edges, streaks around the systems. |
| Climate | The planet's humidity scales the cover (a dry planet is nearly clear); hot, dry land loses most of what is left (deserts stay clear); the sea adds a little. |
| Terrain | Slopes facing the prevailing wind (easterly trades, westerlies, polar easterlies) gain cloud, their lee loses it, in proportion to the humidity. Only the ranges count: the relief is band-limited to 60 km. |

Lengths follow the planet's feature scale, like the terrain's. The climate is
sampled on a coarse 6 × 64² grid and upsampled; the whole map takes ~3 s,
once per process (the main thread for the textures, each worker for
placement, cached by spec).

**Coverage to the field, exactly.** The clouds never use coverage directly.
The sorted column peaks of the field (`noise.column_peaks`, the coverage
calibration) are the inverse of the coverage function: a column is cloud when
its peak plus the shift clears the threshold, so
`shift(c) = threshold_lo − quantile(peaks, 1 − c)`. `weather.shift_table`
tabulates it at 256 coverages; the texture holds both the coverage (r) and
its shift (g, through the table), so the shader pays one fetch. The far deck
reads the coverage, the near field the shift, and the two agree by
construction (a test checks the round trip within 3 % at 0.1–0.9).

`python -m world_builder.terrain.preview --weather` draws each preset's globe
under its weather, and the coverage alone, for tuning.

### Near: billboard shells

Billboards are generated per **cube-sphere tile** in the shared worker pool
(`generate_cloud_tile`), by rejection sampling of the density, as the flat
field places its own. Shell *k* uses tiles of level `base + k` (8 km at the
first) and billboards twice the size of shell *k − 1*; adjacent shells
crossfade by distance on optical depth (the flat field's `lod_shells`
layout). Each shell costs about half the previous.

Per pixel, the cost depends on the quality (`CloudQuality`: HIGH, MID,
LOW; the demo's F5 cycles them and off), which sets the billboard sizes, the
sun march's steps, and:

- **Octaves per shell** (`sphere.SHELL_OCTAVES`): HIGH evaluates all 4 up to
  shell 2 and drops the finest from shell 3 on; MID and LOW drop more,
  sooner. The finest cumulus octave (~125 m features) is still ~6 px wide at
  27 km at 1600×900, so dropping it nearer shows as smoother, blander
  clouds. A dropped octave counts as its mean, and the shell's threshold is
  moved to make up for the lost spread (`octave_offset`), so the coverage
  holds within 2 % (tested) and the crossfades still agree; the placement
  evaluates the same octaves and threshold.
- **Overlap** (`sphere.VOLUME_FRACTIONS`): billboard volume per unit of
  cloud volume, 2.5 at HIGH (the flat field's), less at MID and LOW. The
  extinction is divided by it, so the mean opacity holds, but fewer, more
  opaque billboards look lumpier and hit the per-billboard optical-depth cap
  sooner (thinner dense cores).

At every quality:

- **Early-out:** every density evaluation stops after one or two octaves
  where the rest cannot reach the threshold (they add at most their
  weights). It is exact: the same densities, fewer fetches in clear air, which
  is most of a billboard quad.
- **The atmosphere** in front of a billboard takes 4 samples per vertex
  (`atmosphereAlong(..., steps, ...)`), against 8 per pixel elsewhere.

Tiles hang under a root moved by `origin − camera` (the terrain's rebasing).

**Draw order.** Premultiplied "over" needs back to front. The deck draws
first, then each shell in its own back-to-front bin, the coarsest first.
One bin for all tiles was wrong: a coarse tile holding the camera has its
centre near it, so it drew late, and its billboards, all farther than the
finer shells' (its near ones are faded out), covered nearer clouds. Within
a tile the particles are re-sorted, a few tiles per frame, the stalest first:
new tiles, then those the camera has moved past most relative to their
distance. A stale order composites a cloud's far side over its near side,
which reads as shadows sliding over it.

### Far: the deck

Billboards cannot reach orbit: at constant volume fraction their covered area
falls as 1/size. Beyond the last shell, the **deck** takes over: a sphere at
mid-slab height whose fragments evaluate the same density through the
slab's depth (thickness / |cos|). Where a pixel spans more than the clouds'
features (from orbit, a pixel is a kilometre or two), the deck fades from the
sampled field to its **expected coverage**, the share of the pixel's columns
that hold cloud: the weather map's coverage, which the near field reproduces
by construction. Self-shadowing goes first (it is smaller than the
cloud). So the pattern from orbit is the one you fly into, without the
speckle.

### Not yet

- **Wind:** the clouds are static.
- **Shadows on the ground:** clouds do not shade the terrain.
- **Reflections:** the demo keeps clouds out of the ocean's mirror (it would
  draw every billboard twice).
- **One cloud type per planet** (cumulus in the demo); the flat field mixes
  types, the planet field could take one per slab (the weather map could
  then say which).
- **Weather that moves:** the map is fixed; the cyclones could drift with the
  zonal wind and the map be rebuilt (a few seconds) or advected.
