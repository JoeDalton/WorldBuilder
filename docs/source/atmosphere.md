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

For a view ray, `atmosphere.glsl` marches `uAtmoSteps` samples (8 at HIGH:
`AtmosphereSettings.steps`, live through `Atmospheres.set_settings`; the
stars and the cloud billboards march `vertex_steps`, 4, per vertex) through
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
- **The sea:** drawn by the terrain, so seen through the air like the
  ground, in both its looks.
- **Planet clouds:** per vertex for billboards (the air barely changes across
  one, and per pixel it would be paid for every layer of overdraw), per pixel
  for the deck. They then share the sky's tonemap (`display.glsl`), so a
  cloud deep in haze ends up the colour of the sky behind it
  ([clouds at the horizon](#clouds-at-the-horizon)).

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
  world's own shaders (models, ships). The white test ball ahead of the start
  pose shows them; models skip the world's exposure and tonemap, see
  [Lighting on objects](object_lighting.md);
- **the sea's near look**: `Sea.set_light` scales its water colour (tuned
  in daylight) by the light on the sea under the camera over the daylight's,
  so the sea is dark by night;
- **an ambient scale** (F9 / F10) applied to the terrain's sky light, the
  clouds' (`SphericalClouds.set_sky_light`), the sea's and Panda3D's.

### Left out: true multiple scattering (the Bruneton option)

Light scattered more than once is only the flat twilight fill above: its
colour and its fading are guesses. Clouds
in the planet's shadow follow the same twilight curve for their sky light,
plus a stand-in for the light of the still-lit air above them
([the terminator on the clouds](#the-terminator-on-the-clouds)).

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

Per pixel, the cost depends on the settings (`CloudSettings`, its presets
by `QualityLevel`: LOW, MID, HIGH, ULTRA; the demo's F4 cycles them; ULTRA, above HIGH, keeps all four octaves on every shell;
`SphericalClouds.set_settings` re-plans the shells and regenerates the
tiles on the same worker channel), which sets the billboard sizes, the sun
march's steps, and:

- **Octaves per shell** (`CloudSettings.shell_octaves`): HIGH evaluates all 4 up to
  shell 2 and drops the finest from shell 3 on; MID and LOW drop more,
  sooner. The finest cumulus octave (~125 m features) is still ~6 px wide at
  27 km at 1600×900, so dropping it nearer shows as smoother, blander
  clouds. A dropped octave counts as its mean, and the shell's threshold is
  moved to make up for the lost spread (`octave_offset`), so the coverage
  holds within 2 % (tested) and the crossfades still agree; the placement
  evaluates the same octaves and threshold.
- **Overlap** (`CloudSettings.overlap_near`, `overlap_far`): billboard volume per unit of
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

**At half resolution** (`CloudSettings.half_resolution`, off by default;
`clouds.half_res.HalfResClouds`). The clouds draw into two offscreen layers
half the view's size, one per split-depth range, each by a camera sharing
that range's lens. Everything else draws into them as depth only (the
terrain through a depth-only fragment stage; the sky is left out), so the
ground and a game's models still hide the clouds. Each layer is laid over
its own range's pass by a fullscreen card, premultiplied "over": the far
layer in the far pass, so near ground covers it as it covers anything far.
It saves 1-1.7 ms of GPU time low over the planet and costs ~1.2 ms of
main-thread cull and draw (see [performance](performance.md) §19): a trade
for GPU-limited setups. `World.use_split_depth(depth)` gives the world the
split cameras it needs for it; cloud edges against the ground's silhouettes
are resolved at half resolution.

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

### The terminator on the clouds

A cloud stays sunlit after sunset below it: at 2.7 km over a 600 km planet,
until the sun is 5.4° below its horizon. The air alone dimmed its light only
to about 15 % of red by then, and the planet's shadow then cut it to black,
which drew a hard line across the clouds, close up and from orbit. Two terms
in `cloud_field.glsl` (billboards per vertex, the deck per pixel) spread it:

- **The grazing edge** (`grazingLight`): sunlight whose path dips below the
  cloud's own altitude crosses the haze, the terrain and the cloud deck
  around, more the lower it passes, and none gets through along the ground.
  It is scaled by a smoothstep of the square root of the path's lowest
  altitude over the cloud's, so the light fades to nothing at the planet's
  shadow over its last ~1.5°, not in a step. A sun above the cloud's horizon
  is untouched.
- **Twilight** (`cloudTwilight`, weighted by `TWILIGHT_GLOW` in cloud.frag):
  once a cloud is in the shadow, the air above it is still sunlit for a few
  degrees more and lights it, warm, fading as that air darkens in turn. It
  is taken as the sunlight the air `TWILIGHT_AIR_HEIGHT` (6 km) above
  receives and the cloud does not, which makes it nothing by day. It is
  light from the sky: not phase-weighted, not self-shadowed, and reaching
  undersides as the sky's light does. It carries the clouds from lit to dark
  over about 4° past the line (about 40 km on this planet).

The terrain and Panda3D's lights still use the plain shadow: on the ground
the sun has set long before, so there is no line to see there.

### Clouds at the horizon

Seen down a long, low path through the air (toward the horizon from the
ground, or toward the limb from above), a cloud is mostly hidden by the haze
in front of it. That air lets red through much better than blue: about
(0.44, 0.21, 0.04) over 100 km at cloud height. Sunlit tops are brighter than
the haze, and what gets through of the difference is reddened; shaded sides,
darker than the haze, come out bluish. In linear light this is mild, the
lit tops about (1.48, 1.45, 1.28), as distant cloud looks a little warm.

It showed orange, even with the sun overhead, because the clouds alone
skipped the display transform that the sky and the terrain share. Their soft
knee kept the hue at every level and wrote linear light with no display
gamma, so the warm excess stayed saturated and no tint was softened (a
blue/red ratio of 0.87 displayed as 0.87, not 0.94). Hazed cloud never met
the sky's colour behind it either. Planet clouds now take the sky's own
transform (`displayColour` in `atmosphere/shaders/display.glsl`): ACES per
channel up to white, keeping the hue above it, then the display gamma, at the
sky's exposure (0.9). Their sunlight now takes the air's whole reddening
(`cloudSunTint`): the half it took before showed orange only without the
gamma, and paled sunsets to gold through it.

Measured on the demo at 1.3 km with the sun 80° up, the brightest horizon
tops went from (0.86, 0.84, 0.75) to (0.89, 0.87, 0.83) displayed; near
clouds kept their contrast. Ruled out on the way: the sunlight's own tint
(`sunset_radius`, `grazingLight`, `cloudTwilight`), since with only the air
in front turned off (`uCloudDebug` bit 2) the clouds came out white; and too
few samples of that air per vertex (as orange with 8 as with 4).

### Cirrus

[`clouds/cirrus.py`](../../src/world_builder/clouds/cirrus.py) puts high
cirrus over the cumulus (`CirrusSheet`, `BodyLook.cirrus` in a `World`): thin
ice cloud, 8-9 km up by default, fibrous and streaming along the upper wind.

**A surface, not billboards.** Cirrus is a kilometre deep at most and tens of
kilometres across, and optically thin. Billboards would give it no volume
worth their cost, and their puffs fight its fibres (the flat field's `CIRRUS`
preset shows both). So it is one sphere at the slab's mid-height. Its
fragments evaluate a density through the slab's depth (`thickness / |cos|`,
so it thickens toward the horizon), where the sight line meets the true
sphere: the mesh's facets sag under it. There is no sun march; a sheet this
thin barely shadows itself.

**Fibres along the wind.** The upper wind (`weather.upper_wind`, from
`WeatherSpec.jets`: weak tropical easterlies, the subtropical and polar-front
jets) has a stream function, `psi = -R * integral(u dlat)`
(`weather.stream_table`). The weather systems add one Gaussian each, turning
the wind around each low and over each tropical cyclone's outflow
(`weather.stream_vortices`). The wind blows along psi's contours, so noise
sampled at `(psi / spacing, 0, 0) + along(p) / length` varies fast across the
wind and slowly along it: streaks exactly along the flow, wherever it turns,
and no frame to rotate (a stretched noise turned to follow a direction field
shears).

- Finer fibrils inside a streak are the same construction at a smaller
  spacing; a slow noise added to psi makes the fibres meander off the
  contours.
- Where the wind is weak, psi varies slowly and the same noise turns
  isotropic. That gives a veil instead of streaks, and it is calm air, where
  the weather puts little cirrus anyway.
- The slow along-wind terms go into the noise's y and z only (`along`).
  Added to x as well, they tilted the fibres off the wind by
  `(spacing / length) / speed`, about 15° where the wind blows at a third of
  the jets' speed.
- The noise is computed, not fetched: hashed value noise with quintic
  interpolation, C2 (`value_noise`, `hashNoise`). A streak magnifies its
  noise about 50 times, and through the density's threshold a 3D texture's
  linear filter showed its creases as a lattice of facets. Smoothstepping the
  texel fraction swapped them for flat bands along the texel edges.

**How much.** The cirrus has its own cover map (`weather.cirrus_cover_map`,
6 × 128²):

- `WeatherSpec.cirrus_zonal`: anvils over the wet tropics, streaks along the
  jets;
- a shield poleward and east of each extratropical low (ahead of its warm
  front) and over each tropical cyclone;
- mesoscale texture;
- humidity, which counts for less than for the cumulus.

The noise is stationary, so the raw density's distribution is the same
everywhere. A coverage `c` is `P(raw > threshold)`, its threshold the
`(1 - c)` quantile of `raw` (`calibrate`, as for the cumulus), and the map
stores that threshold. The share of the sheet with cirrus is the map's cover
(a test checks it within 3 %). Far away, the fibrils fade to their mean first,
then the streaks to the cover, so there is no speckle from orbit.

**Draw order.** The cirrus lies above the cumulus slab (the `World` checks
that its base clears the clouds' top, and that its top stays inside the air).
A ray's height along it is convex: it falls, then rises.

- Where a ray *enters* the cirrus's sphere, it is above every cumulus, so the
  cirrus is in front of all of them.
- Where it *exits*, it never comes down again, so the cirrus is behind all of
  them.

This holds for any camera: under both layers, between them, above them, in
orbit, and in the sea's mirror. The same mesh is drawn by two nodes, each
keeping one kind of crossing, found per fragment from the pass's own camera
(`p3d_ModelViewMatrixInverse`, which is right in the mirror too):

| Node | Crossings | Bin |
|---|---|---|
| `behind` | where sight lines leave the sphere (all of them from below) | the sky's (21, fixed), draw order 2: after the dome and the stars, before the deck (22) and every shell |
| `front` | where they enter it (only from above) | Panda's "transparent" (30, back to front by the planet's centre): after every cloud bin, before a game's translucent effects nearer than the planet's centre |

The cloud bins 21-29 are all taken, so no new sort fits between them; these
two need neither a new sort nor a change to Panda's own.

- **The depth split** needs nothing: the far range draws before the near
  one, and everything near is in front of everything far.
- **Half-resolution clouds** leave the cirrus at full resolution (its fibres
  are its look, and it is one cheap surface). It is kept out of the
  half-resolution pass, and the composite (22) lands between its two nodes.
- **The far plane** (`World.depth_ranges`) reaches the cirrus's far limb
  where it is higher than any ground.

**Lit** as the cumulus are, through `cloud_field.glsl`:

- the ice's phase function (the `CIRRUS` preset's optics);
- the sunlight reddened by the air at the cirrus's own height
  (`cloudSunTint`), so it stays lit after the cumulus below has gone into
  the planet's shadow, then twilight (`cloudTwilight`);
- the air in front of it per pixel (`atmosphereAlong`), and the sky's
  display transform.

From inside the slab it fades to nothing near the camera: a veil, not a
volume to fly through.

### Not yet

- **Wind:** the clouds are static.
- **Shadows on the ground:** clouds do not shade the terrain; the cirrus does
  not shade the cumulus either.
- **One cumulus-like type per planet** (cumulus in the demo), with the cirrus
  above it; the flat field mixes types, the planet field could take one per
  slab (the weather map could then say which).
- **Flying through the cirrus:** from inside its slab it is a veil fading
  near the camera, with no volume.
- **Weather that moves:** the map is fixed; the cyclones could drift with the
  zonal wind and the map be rebuilt (a few seconds) or advected.
