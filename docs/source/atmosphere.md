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

For a view ray, `atmosphere.glsl` marches 8 samples (`ATMO_STEPS`) through
each atmosphere the ray crosses (a ray that meets the ground stops there).
At each sample:

- the density of both media, for the optical depth back to the camera;
- the optical depth toward the sun, **analytically**: the Chapman function in
  its erfcx form, `Ch(x, χ) = √(πx/2) · erfcx(√(x/2) cos χ)`, with a fitted
  `erfcx` within 0.33 %. Against a numerical integral it is within ~1.5 % at
  every altitude and angle; below the horizon it goes through the tangent
  point (`atmoDepthToward`);
- the sunlight reaching it, times the density, times the transmittance back
  to the camera.

**The planet's shadow** is found analytically along the ray (a cylinder of
the planet's radius behind it, away from the sun), and each sample counts
only the lit share of its stretch, its sun depth taken at a lit point. With
whole samples lit or not, the shadow's edge moved a sample at a time: hard
horizontal bands across the dusk sky, the planet's shadow rising above the
horizon in steps.

**Redder sunsets on a small planet.** Sunlight grazing a 600 km planet
crosses ~11 air masses (Earth's ~38), so it reached the terminator bright and
barely red. `sunset_radius` (default Earth's, 6,371 km) takes the slant path
as over a planet of that radius: the Chapman function's curvature argument
is scaled by `sunset_radius / radius` (never below 1; the `curve` argument of
`atmoDepthToward`, uniform `uAtmoExtra.y`), which lengthens grazing paths
and leaves the sun overhead as it was (within a few tenths of a percent).
`None` is the physical path. It applies everywhere the sun's depth is
taken: the sky, the ground, the clouds, and the CPU's `depth_toward`.

**Twilight fill.** A flat share of the sunlight (`multiple_scattering`,
0.12, uniform `uAtmoExtra.x`) stands in for light scattered more than once:
it reaches every sample, lit or not, dimmed only by the air back to the
camera, and fades out as the sun goes from 6° above the sample's horizon to
17° below. Without it
the planet's shadow on the air at dusk (the band below the pink "Belt of
Venus" opposite the sun) was black, with stars showing through; with it, it
is blue-grey. By day it lifts the sky by a few percent.

The phase functions (Rayleigh, and Cornette-Shanks for Mie) then give the
light scattered toward the eye. The result for anything at distance `d` along
the ray is `seen = colour · transmittance + inscatter`, composed far to near
when a ray crosses two atmospheres.

### Where it applies

- **Terrain:** aerial perspective per pixel, and its sunlight goes through
  its own planet's air (red at sunset, nothing in the planet's shadow).
- **Sky:** the scattered light toward each direction, over the stars and the
  sun's disc. The eye adapts: stars fade as `exp(-sky luminance × 60)`
  (`star_adaptation`), so they are gone by day and out at dusk. Directions
  that meet a planet's ground (`atmoHitsGround`) show no stars or sun, only
  the air in front.
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

### Lighting anything else

`Atmospheres.light_at(index, point_world, sun_direction, sun_color,
sky_light)` gives the light at a point as the terrain's shader lights its
ground: the sunlight through the air (none in the planet's shadow), and the
sky light fading through twilight (`twilight`, the terrain's
`smoothstep(-0.2, 0.2, cos)`). `transmittance_to_sun` is the shader's
`atmoSunlight` on the CPU (the Chapman function, `depth_toward`, with the
same `sunset_radius` scale); `in_shadow` covers airless bodies.

The demo uses them each frame for:

- **Panda3D's lights**: a `DirectionalLight` (the sun) and an `AmbientLight`
  on `render`, as they are at the camera, for anything drawn without the
  world's own shaders (models, ships);
- **the swell ocean**: `Ocean.set_light` scales its water colour (tuned in
  daylight) by the light on the sea under the camera over the daylight's, so
  the sea is dark by night;
- **an ambient scale** (F9 / F10) applied to the terrain's sky light, the
  clouds' (`SphericalClouds.set_sky_light`), the ocean's and Panda3D's.

### Left out: true multiple scattering (the Bruneton option)

Light scattered more than once is only the flat twilight fill above: its
colour and its fading are guesses, and lit clouds come out a little yellow
(the cloud shader puts half the reddening back, standing in for it). Clouds
in the planet's shadow follow the same twilight curve for their sky light.

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

`Sky` draws a camera-centred sphere first, behind everything, without depth,
with the far camera only (`hide(HIDE_FROM_NEAR)`: the near pass would paint
it over the far one), and recentres it every frame (`Sky.update`).
`sky.vert` puts the sphere on the far plane whatever the camera's range, so
the ocean's reflection camera, with a range of its own, still sees it whole.
Its shader shows:

- **the procedural star field** (`Starfield`), in two parts:
  - **point stars** (`make_stars`: 14,000 by default, more toward a Milky
    Way band, a power law of brightness; most white, a share slightly red
    and a share slightly blue, `red=` / `blue=`). Each is a 3×3-pixel point
    whose shader lights at most 2×2 pixels: a tent of one pixel's radius
    around the star's true position, which sums to 1 over the pixels it
    touches, so a star keeps its brightness as it moves across them (no
    twinkling) and keeps its size at any resolution. A cube map would need
    ~4096 px a face to keep stars that small at 1920×1080. Each point carries
    the sky behind it (air, glow) and adds exactly what it changes once
    tonemapped. `star_brightness` scales them (`Sky.set_star_brightness`).
  - **the diffuse light** (`make_glow`): the band, its dust lanes and
    nebulae, smooth, so a 256 px cube map holds it, as parameters (band, nebula
    density, palette coordinate) that the shader colours. The colours are
    live uniforms: a `NebulaStyle` (a three-stop `palette`, `intensity`, the
    Milky Way's colour; `NEBULA_PALETTES` has a few) set with
    `Sky.set_nebula`. The demo's F6 cycles the palettes, F7 / F8 dim and
    brighten. The nebulae's size and coverage are generation parameters
    (`nebula_scale`, `nebula_coverage`).
- or **a painted cube map**: `load_cube_map(pattern, loader)` (SpaceFlight's
  `models/skyboxes/purple_#.png`), its full white at `star_intensity`
  (linear). Rotate either with `star_hpr` (`Sky.set_star_rotation`);
- **the sun's disc**, twice its real size so it reads at any resolution,
  reddened by the air at sunset;
- through every atmosphere, then the terrain's tonemap. The disc and the glow
  around it are much brighter than white, and a per-channel tonemap flattens
  such light toward white whatever its colour (the sunset sun was white while
  it lit everything orange). So above white the sky fades to a tonemap of its
  brightest channel instead, which keeps the hue: the disc shows the colour of
  the sunlight reaching the camera, the light it casts, and so does its
  reflection on the sea. Low, the sun is no brighter on screen than the glow
  around it and got lost in it, so it is drawn as an overexposed sun looks: a
  paler core, a redder rim and a glare of its own light around it
  (`SUN_CORE`, `SUN_REDDEN`, `SUN_GLARE` in sky.frag). How low it is comes
  from its light's colour, so a high sun is untouched.

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
| Zonal circulation | A curve of latitude (`WeatherSpec.zonal`): wet equatorial band (~0.7), dry subtropical highs (~0.15 at 24–33°), storm tracks (~0.55 at 52–62°), polar caps (~0.4). `season` moves it north or south. |
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
LOW; the demo's F5 cycles them and off; ULTRA, above HIGH, keeps all four
octaves on every shell), which sets the billboard sizes, the sun march's
steps, and:

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
- **Off-screen costs nothing.** Every pass (near, far, the ocean's mirror)
  runs the vertex stage for every billboard of every shown tile. One whose
  four corners all lie beyond one clip plane is dropped before its air and
  sunlight are worked out (exact: it could not cover a pixel); ~1.5 ms at a
  dusk view from inside the deck.
- **Zero weight costs nothing.** A billboard wholly outside its shell's
  distance band, or faded out near the camera, is dropped in the vertex
  stage; a fragment's shell weight is known before its density. A coarse
  shell's tile around the camera holds many such billboards, large on screen
  (inside the deck, looking at a nearby cloud: 12.7 → 9.6 ms of clouds).
- **The sun march** reads the weather once per pixel, not at every step.
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

Lighting goes the same way: from afar, the self-shadowing and the powder
term are their **averages** over the clouds' columns
(`noise.deck_light_table`), not the per-pixel values that would alias. The
averages are measured on the CPU when the clouds are built (~0.4 s, cached).
The deck's own close-up lighting (`sun_optical_depth_on_sphere`, the sun
march's mirror) is averaged over 2048 columns, weighted by their alpha seen
from overhead. The result is a 32 × 17 table indexed by the sun's cosine at
the cloud and the weather's coverage. The far deck is then the blur of the
near one, so nothing changes colour when closing in. A low sun marches along
the slab through the clouds around it, which darkens the deck gradually
toward the terminator, in the reddened sunlight's own colour.

An earlier version replaced this with a sheet's falloff with the sun's
incidence. That removed the direct sunlight from about 20° of sun down,
before the air reddens it, so the clouds along the terminator were lit by the
blue sky alone: grey-blue from orbit, turning orange about 200 km closer.

### Not yet

- **Wind:** the clouds are static.
- **Shadows on the ground:** clouds do not shade the terrain.
- **One cloud type per planet** (cumulus in the demo); the flat field mixes
  types, the planet field could take one per slab (the weather map could
  then say which).
- **Weather that moves:** the map is fixed; the cyclones could drift with the
  zonal wind and the map be rebuilt (a few seconds) or advected.
