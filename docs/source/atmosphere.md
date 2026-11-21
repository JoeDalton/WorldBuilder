# Atmosphere, sky and planet clouds

Three systems make a planet look like one from the ground to orbit:

| System | Module | What it does |
|---|---|---|
| Atmospheres | `world_builder.atmosphere` | Single scattering of sunlight by air and haze: sky colour, sunsets, aerial perspective, the limb from space, reddened sunlight. Applied by every shader along its view ray. |
| Sky | `world_builder.sky` | A star field (procedural or painted) and the sun's disc, seen through the atmospheres. |
| Planet clouds | `world_builder.clouds` | One cloud density field over the whole planet, raymarched from the ground to orbit, with the weather's expected coverage where a pixel spans more than its features: the same pattern at every distance. |

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
stars march `vertex_steps`, 4, per vertex) through
each atmosphere the ray crosses (a ray that meets the ground stops there).
With the camera in the air most pixels read the sky-view table and the
aerial-perspective cells ([the tables](#the-tables)), baked at 16 samples
whatever the preset; `steps` sets the march that remains: from space, past
the cells' reach, the sea's reflection rays, and everything with the tables
off. At each sample:

- the density of both media, for the optical depth back to the camera;
- the sunlight reaching it, from the **transmittance table**
  (`atmoTransmittance`, [the tables](#the-tables)), whose values are the
  optical depth toward the sun taken analytically: the Chapman function in
  its erfcx form, `Ch(x, χ) = √(πx/2) · erfcx(√(x/2) cos χ)`, with a fitted
  `erfcx` within 0.33 %. Against a numerical integral it is within ~1.5 % at
  every altitude and angle; below the horizon it goes through the tangent
  point (`atmoDepthToward`, still used directly by `atmoSunlight`);
- that sunlight times the density, times the transmittance back to the
  camera; and the light scattered more than once, from the
  **multiple-scattering table** (`atmoMultiple`), times the sample's
  scattering.

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

**Multiple scattering.** Light scattered more than once comes from a table
(Hillaire 2020's approximation, [the tables](#the-tables)): per unit of
scattering, what reaches a sample from the air all around it and the ground
below, at its altitude and with the sun at its own elevation, so it is lit
in the planet's shadow (the band below the pink "Belt of Venus" opposite the
sun is blue-grey, not black) and fades with the sky above through twilight.
It is isotropic (no phase function), dimmed by other bodies' eclipses as the
sky light is, and scaled by `multiple_scattering` (1, physical; uniform
`uAtmoExtra.x`). It replaced a flat share of the sunlight (0.12, faded by a
guessed twilight curve): by day the new light is brighter near the horizon
(the ground's bounce, `ground_albedo` 0.3) and bluer overhead.

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
- **Planet clouds:** no air is drawn in front of them (it showed them
  yellow, [clouds at the horizon](#clouds-at-the-horizon)); the air reddens
  their sunlight, per sample in the raymarch, per pixel on the deck. They
  share the sky's tonemap (`display.glsl`).

`Atmospheres` holds every atmosphere's uniforms in arrays shared by all bound
nodes and rewrites them in place once a frame, in float64 where it matters
(`|camera|² - radius²` would be noise in float32).

### Lighting anything else

`Atmospheres.light_at(index, point_world, sun_direction, sun_color,
sky_light)` gives the light at a point (or at an array of points, at once)
as the terrain's shader lights its ground: the sunlight through the air (none in the planet's shadow), and the
sky light fading through twilight (`twilight`, the terrain's
`smoothstep(-0.2, 0.2, cos)`). `transmittance_to_sun` is the shader's
`atmoSunlight` on the CPU (the Chapman function, `depth_toward`, with the
same `sunset_radius` scale); `in_shadow` covers airless bodies. Other
bodies' eclipses are `Shadows.eclipse`'s ([Shadows](shadows.md)), which
`World.light_at` multiplies in.

The demo uses them each frame for:

- **Panda3D's lights**: a `DirectionalLight` (the sun) and an `AmbientLight`
  on `render`, as they are at the camera, for anything drawn without the
  world's own shaders (models, ships). Models drawn that way skip the
  world's exposure and tonemap; the test balls ahead of the start pose are
  drawn by the world's model shader instead (M switches), see
  [Lighting on objects](object_lighting.md);
- **the sea's near look**: `Sea.set_light` scales its water colour (tuned
  in daylight) by the light on the sea under the camera over the daylight's,
  so the sea is dark by night;
- **an ambient scale** (F9 / F10) applied to the terrain's sky light, the
  clouds' (`PlanetClouds.set_sky_light`), the sea's and Panda3D's.

### The tables

`world_builder.atmosphere.tables` (RENDERING_PLAN.md part B; Hillaire 2020,
"A Scalable and Production Ready Sky and Atmosphere Rendering Technique",
rather than Bruneton's 4D tables, which are dearer to build and fixed per
atmosphere). Per atmosphere, both atmospheres stacked in one RGBA16F texture
per table, bound by `Atmospheres.bind` (`uAtmoTransmittance`,
`uAtmoMultiple`; `bind_defaults` gives 1 x 1 stand-ins):

| Table | Texels | Over | Values |
|---|---|---|---|
| Transmittance | 256 x 64 | (mu, r), Bruneton's mapping (texels crowd toward the horizon) | the sunlight through the air, `transmittance_to_sun`'s formula with the `sunset_radius` curvature; 0 into the ground |
| Multiple scattering | 32 x 32 | (mu_s, altitude), linear | Psi_ms: the second order gathered from 64 directions (40 steps each, the ground's Lambertian bounce with them), the higher orders as `1 / (1 - f_ms)`, f_ms the share the air scatters back |

They depend on the atmosphere's parameters only, fixed once added, so they
are computed on the CPU from the twin's own formulas when an atmosphere is
added (`Atmospheres.add`: ~0.5 s for the multiple scattering, cached per
planet and air) and uploaded: no window needed, and the transmittance table
is the twin by construction. Filtered as GL does, it holds
`transmittance_to_sun` within 2 % or 0.005 everywhere, 99 % of rays within
1.2 % (the worst: grazing rays a few metres over a small planet, whose
curvature x10.6 makes the horizon's tail steepest).

The CPU reference for a view ray is `tables.inscatter_reference` (a fine
march of single scattering plus the table's multiple scattering). The
shader's 16-sample march is within 3 % of it from the ground at noon, a low
sun and in twilight (1 %), except within ~10° of the sun's disc, where Mie's
forward peak at 16 midpoint samples is up to 15 % low
(`tests/test_atmosphere_tables.py`).

Cost: reading the transmittance table instead of the Chapman function is
even (within run-to-run noise at 2560 x 1440, HIGH); the multiple-scattering
read (one more fetch a sample) and the ozone (its density a sample) cost
together ~0.2-0.7 ms a frame at 2560 x 1440, HIGH (coast +0.2, in clouds
+0.7, 20 km +0.2, sunset ~0; means of two alternated runs, +-0.4 noise),
until the sky-view table and the aerial-perspective cells (below) took the
march off the sky and the ground wherever the camera is in the air.

**The sky-view table** (B.3, `atmosphere.sky_view`, `shaders/sky_view.glsl`):
the air between the camera and space for 256 x 128 directions, baked on the
GPU every frame while the camera is inside an atmosphere, read by the sky
(`sky.frag`, and so the sea's reflection, which draws the sky) and the stars
(`stars.vert`) in place of their march; from space they march as before.
Hillaire's layout: the camera's vertical and the sun's azimuth (so a sun
turning in azimuth alone turns the table, not its content), texels crowded
toward the horizon (square roots either side of it, the dip below the
horizontal from `|cam|^2 - R^2` in float64) and toward the sun (its Mie glow);
in-scatter in the lower half, transmittance in the upper (the sun's disc and
the stars are dimmed by it). Each texel is the sky's own `atmosphereAlong` at
16 samples, so the eclipses, the rings' shadow and the planet's shadow are in
it. Against that march per pixel: within 4 % or 1 % of the noon zenith (5 %
within 10° of the sun), the transmittance within 3 %
(`tests/test_atmosphere_tables.py`). Baked each frame rather than on change:
it depends on the camera's altitude and the sun's elevation, but also on the
eclipses, the rings and a second atmosphere, and the bake is cheap
(performance.md). `Sky.attach_view(window, shadows)` turns it on (World, the
demo and cloud_bench do); without it the sky marches. Its size is a power of
two: Panda pads a render target's texture up to one, the content in a
corner (192 x 108 read back garbled).

**The aerial-perspective cells** (B.4, `atmosphere.aerial`): the air between
the camera and every cell of a 32 x 32 x 32 grid over the main camera's view,
baked on the GPU each frame while the camera is inside an atmosphere, read
by `applyAtmospheres` (`aerialAlong`: the terrain, the near sea, the models)
in place of their per-pixel march. Each cell is `atmosphereAlong` from the
camera to its slice's end at 16 samples (eclipses, rings, the planet's shadow
with it); the slices reach past the horizon or 128 km
(`aerial.AERIAL_REACH`), the nearer, spaced as the square of their index;
the reader interpolates linearly between cells and slices (an atlas of 8 x 4
tiles a half, in-scatter and transmittance; `uAerialToClip` maps a direction
into the grid). Two things the first version got wrong:

- **The ground.** A cell's march stopped at the planet's ground like any
  view ray's, so the slice past a point on the ground held the ground's
  value, and the point, between two slices, was interpolated toward it: up
  to ~10 % low near the camera. The bake goes on through the ground, the air
  held at the ground's density (`atmoThroughGround`), and the cells past a
  surface continue its value smoothly.
- **From high up.** The air a ray crosses lies in a thin shell at its far end
  when the ground is far below; slices from the camera over the longest path
  the air allows (300 km and more) were ~16 km apart there, and the haze of
  a ground seen from orbit was up to 38 % off. Hence the 128 km reach, and no
  cells from space.

A point outside the main camera's view (the sea's reflection), past the
reach, or with the camera in space is marched as before. Against the march
(`tests/test_atmosphere_tables.py`), in front of the ground: within 6 % or 1 %
of the noon zenith (8 % for the ground 45 km below), the transmittance within
3 %. `AtmosphereSettings.aerial` turns it off (the march everywhere).

**Presets** (B.5). Both tables stay on at every preset, LOW included: at
LOW's 4 samples a pixel the march is cheaper, but the cells still save
0.2-0.8 ms a frame and the sky-view table up to 0.25 ms at 2560 x 1440
(performance.md §25), and their 16-sample bakes look better than the
march's 4. Their sizes are the same at every preset. From space the march stays, with the multiple-scattering
fetch and the ozone a sample: +0.24 ms on the orbit pose, the cheapest frame
(3.6 ms), left as it is.

**God rays** (RENDERING_PLAN.md C.5, `AtmosphereSettings.god_rays`: off at
LOW and MID). The two bakes that march the air per texel, the aerial cells
and the sky-view table, take the shadows of the clouds and the mountains on
it: each sample's direct sunlight times `atmoGodRays` (`god_rays.glsl`,
compiled into those bakes only, `ATMO_GOD_RAYS`). The light scattered more
than once stays, as the sky's fill does under a cloud.

- **The mountains**: `shadows.glsl`'s `terrainShadow(planetDir, altitude)`,
  which answers for any point, not only the ground (its altitude against the
  ceiling the ridges set toward the sun there: the global cube map, the
  local map's finer one near the camera). At a low sun the valleys' air
  behind a ridge goes dark and shafts fan from the gaps, across the ground
  and up the sky.
- **The clouds**, each family's optical depth toward the sun: the light
  volume's ([shadows](shadows.md#the-cloud-light-volume)) at the point, or
  where the sun's ray enters the slab's base for a point under it; the cloud
  shadow maps' total there beyond the volume; nothing above the slab.
- **Binding.** A bake registers with `Atmospheres.god_ray_root`;
  `Atmospheres.bind_god_rays(atmosphere, binders)` hands it, now and later,
  the planet's inputs (its `ShadowMap.bind`, the clouds'
  `bind_shadow_inputs`, their maps' and light volume's `bind`). `World` binds
  the first body with air; the demo its planet.
- **Resolution.** The shafts are as fine as the bakes: the aerial cells'
  32 x 32 x 32 (soft vertical banding in a shaft, no blocks at 2560 x 1440)
  and the sky-view table's 256 x 128. The plan's finer grid is not built.
- **Cost** (RTX 3070, 2560 x 1440, HIGH; `frame_profile.py`'s `godrays`):
  0.09-0.17 ms, 0.31 ms with genera (both bakes).
- **Test**: under a ceiling over every point the cells lose their direct
  sunlight and keep the fill; off, they are B's whatever is bound
  (`tests/test_atmosphere_tables.py`). Not tested on the GPU: the clouds'
  part (the readers it uses are, in `test_cloud_light_volume.py`).

**The ozone layer.** Without it the twilight sky went olive-green near the
zenith and the horizon (green and blue scattered about equally once the sun's
light has crossed the whole air). Ozone absorbs orange and green (the
Chappuis band; Hillaire's Earth values, `AtmosphereParams.ozone`), in a tent
peaking at 25 km, 15 km half-width (`ozone_altitude`, `ozone_width`): with
the sun 4° down, the zenith's green over blue goes from ~1 to ~0.6. It only
absorbs: it is in the view ray's extinction (per sample, `atmoOzone`), in the
multiple-scattering table's, and on the sunlight's path. The Chapman function
has no form for a tent, so the ozone's depth toward the sun is marched per
texel of the transmittance table (512 steps, over the same curved planet
`sunset_radius` takes) and kept in its alpha (km): `atmoSunlight` multiplies
it in with the analytic Rayleigh and Mie, the CPU twin reads it back
(`tables.ozone_toward`). Overhead it takes ~3 % of the sun's green, more at a
low sun (a slightly pinker sunset light on the ground). `ozone=(0, 0, 0)`
turns it off.

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

A planet's clouds are one part, `clouds.planet.PlanetClouds` (`CloudLook` in
a `World`): the **volume** (`clouds.march.CloudMarch`), the low clouds,
cumulus by default (any cloud type, or the [genera](clouds.md#genera)),
raymarched from the ground to orbit ([the raymarch](clouds.md#the-raymarch)),
with a deck sphere (`clouds.deck`) for the sea's reflection, and optionally
(`cirrus=True`) high cirrus above them in the march's high layer
([Cirrus](#cirrus)). Everything below up to "Clouds at the horizon" is about
the low clouds. (Until 2026-11-15 they could also be drawn as billboard
shells near the camera and the deck beyond; until Task 24 of the cloud
genera, the cirrus was a sheet of baked stamps of its own. Both were
removed.)

The part drives the march (sun, sky light, settings, pixel angle, the split
cameras, update, removal) and holds the rule the layers must keep: the high
layer's slab must clear the low clouds' slabs and its top stay inside the
air (`planet.check_layers`).

### One density field

The clouds' density field (`clouds.noise`) is moved onto the planet:

- the slab is a **height above the sea-level sphere** (`altitude` sets its
  base);
- the tileable noise is sampled at **planet-frame positions in 3D**, so it has
  no seam anywhere on the sphere;
- the **weather** (below) shifts the fbm by direction, by exactly what gives
  the coverage it asks for there.

`noise.density_on_sphere` is the CPU copy, exact to the shader's texture
filtering, which the coverage calibration and the tests measure the drawn
field with. The field's code is one include,
`clouds/shaders/cloud_field.glsl`, shared by the march, the deck and the
shadow maps.

### The weather

The weather is **synoptic and analytic** (`clouds.synoptic`): jets, lows,
highs and tropical cyclones are objects on closed-form tracks, so the
weather at any `WeatherSpec.time_s` is a pure function of the spec, with
no simulation and no state. The design, the literature behind it and every
equation are in
[the synoptic weather review](../literature/synoptic-weather-review.tex)
(section 11, "The model's equations").

| Part | What it does |
|---|---|
| Scales (`synoptic.Scales`) | Follow the planet's radius, its rotation (`PlanetSpec.day_length_h`) and its stratification (`PlanetSpec.stratification`, N·H): the deformation radius sets the cyclones' size (Eady wavelength, ~1000 km on Earth); the thermal Rossby number the Hadley cell's edge (Ω^-1/2, 30° on Earth); the Rhines/Williams spacing the number of jets, each with its storm band. The 600 km demo planet with a 6 h day behaves like a slowly turning Earth: a Hadley cell to ~49°, two or three large lows a hemisphere. |
| Jets (`synoptic.jets`) | Each jet's axis meanders as three Rossby waves, `c = U − β/K²`: long waves near still, short ones travelling east. Troughs and ridges come from it. |
| Systems (`synoptic.systems`) | A train of lows per storm band, born just downstream of their jet's troughs and riding its wave (Norwegian or Shapiro–Keyser life cycles); migratory, blocking and permanent subtropical highs (on the eastern flanks of the ocean basins); cut-off lows; tropical cyclones born over warm sea, recurving, dying at landfall. Each has a closed-form track and life; the population at a time comes from seeded slots, with no state. |
| Pressure and wind (`synoptic.fields`) | A zonal climatology plus a kernel per system (Holland's profile for tropical cyclones); geostrophic wind, the gradient wind in a tropical cyclone, friction at the surface, the trades near the equator. |
| Fronts (`synoptic.fronts`) | Warm, cold and occluded fronts per low, shaped by its stage. |
| Genera (`synoptic.rules`) | What each zone holds: the warm front's Ci → Cs → As → Ns shield, the cold front's Cb line, the warm sector's St/Sc, the comma head, the dry slot, the cold air's open cells, a high's clear core and its Sc deck over cold sea, a tropical cyclone's eye, eyewall and bands, air-mass convection. Stable air (under a shield, in a warm sector) calms the fair-weather cumulus. |
| Background | Fair-weather cumulus by latitude (the wet tropics, the storm tracks, the clear subtropics between: `synoptic.rules.zonal_cover`), scaled by the planet's humidity, hot dry land cleared, the sea adding a little, slopes facing the systems' surface wind lifting cloud (their lee losing it); a mesoscale texture curled round the systems, which also breaks up their decks and shields. The sum goes through a contrast (`weather.contrast`: a smooth step at `CLOUD_EDGE`), so the sky is clear or cloudy, with the texture drawing the boundaries, not a thin cover everywhere. **Tunables:** `WeatherSpec.cloud_amount` (a bias: +0.1 is about 10 points more cloud, -0.1 less), `cloud_contrast` (the step's half-width: larger, softer and more uniform), `cirrus_amount` (the cirrus's bias). On Earth the drawn low cloud covers 0.41-0.43 of the planet, 37-39 % clear, 32-35 % overcast, 70-73 % of the subtropics clear (`scripts/weather_report.py`'s bands). |
| Presets (`synoptic.presets`) | `WeatherSpec.pins`: a fair high, a warm front approaching, a cold front, a squall line, overcast and drizzle, a hurricane or thunderstorms pinned at a coordinate. Each is a role relative to a system, placed so; the jets bend round it and the free systems form round it. A pin the planet cannot hold raises with the reason. |

`weather.synoptic_cube` (6 × 256² × 11, ~3.7 km a texel on a 600 km planet)
holds the convective coverage, depth and anvil shelf and the eight other
genera; `coverage_map`, `weather_cube` and `genera.genus_planes` read it.
The climate is sampled on a coarse 6 × 64² grid and upsampled; the whole
cube takes ~4 s, once per process (cached by spec).

**Coverage to the field, exactly.** The clouds never use coverage directly.
The sorted column peaks of the field (`noise.sphere_column_peaks`, the coverage
calibration) are the inverse of the coverage function: a column is cloud when
its peak plus the shift clears the threshold, so
`shift(c) = threshold_lo − quantile(peaks, 1 − c)`. `weather.shift_table`
tabulates it at 256 coverages; the texture holds both the coverage (r) and
its shift (g, through the table), so the shader pays one fetch. The far
field reads the coverage, the near field the shift, and the two agree by
construction (a test checks the round trip within 3 % at 0.1–0.9).

`python -m world_builder.terrain.preview --weather [--time S] [--pin LAT LON
PRESET ...]` draws each planet preset's weather three ways, for tuning: the
globe under all its cloud, the genera by colour, and a surface chart
(isobars every 4 hPa, fronts with their symbols, the jets' axes, the
systems' centres). `scripts/demo_terrain.py` takes the same `--time` and
`--pin`.

### Near: the raymarch

Near the camera the march samples the field itself, along each pixel's ray
([the raymarch](clouds.md#the-raymarch)), with fewer octaves the farther it
reaches (`CloudSettings.shell_octaves` over `octave_bands` distance bands,
each twice as far as the last: `field_setup.octave_bands`). HIGH evaluates
all 4 octaves up to the second band and drops the finest from the third on;
MID and LOW drop more, sooner; ULTRA keeps all four everywhere. The finest
cumulus octave (~125 m features) is still ~6 px wide at 27 km at 1600×900,
so dropping it nearer shows as smoother, blander clouds. A dropped octave
counts as its mean, and the band's threshold is moved to make up for the
lost spread (`field_setup.octave_offset`), so the coverage holds within 2 %
(tested). Every density evaluation also stops after one or two octaves
where the rest cannot reach the threshold (they add at most their weights):
exact, the same densities, fewer fetches in clear air.

(Until 2026-11-15 the near clouds could also be **billboards**: quads placed
per cube-sphere tile by rejection sampling of the density, in LOD shells
crossfaded on optical depth, sorted back to front within a tile, optionally
drawn at half resolution. Their cost grew with the cloud's volume on screen,
and they were removed; [performance](performance.md) §13-22 measure them.)

### Far: the deck

Sampling the field stops working where a pixel spans more than the clouds'
features (from orbit, a pixel is a kilometre or two): it speckles. There the
march, and the **deck** (a sphere at mid-slab height whose fragments evaluate
the same density through the slab's depth, thickness / |cos|; the sea's
reflection draws it, `clouds.deck`), fade from the sampled field to its
**expected coverage**, the share of the pixel's columns
that hold cloud: the weather map's coverage, which the near field reproduces
by construction. Self-shadowing goes first (it is smaller than the
cloud). So the pattern from orbit is the one you fly into, without the
speckle. That coverage is then drawn as opaque masses with sharp edges and
clear gaps rather than a translucent slab ([clouds.md](clouds.md), "Masses,
not a wash"), and the weather's mesoscale texture keeps its physical size
(`WeatherSpec.meso_wavelength`, 160 km, on every planet: scaled by the
planet's size it was 1700 km on Earth).

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
which drew a hard line across the clouds, close up and from orbit. Three terms
(per sample in the march, per pixel on the deck) spread it:

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

- **The sky's light** (`cloudSkyFade`, `noise.sky_light_fade` on the CPU)
  fades with the sun's height (`smoothstep(-0.3, 0.1)` of its cosine) and,
  once the sun is below the horizon, with the sunlit share of the air above
  the cloud: the air is sunlit only above the planet's shadow, at radius
  `ground / sin(sun's zenith angle)`, so the share is
  `exp(-(that - cloud's radius) / H)`, H the Rayleigh scale height (8 km).
  The planet's size sets how long it lasts: with the sun 8° down, a cloud 5
  km up keeps ~90 % of the curve on the 600 km planet (the shadow 0.9 km
  over it) and ~0.1 % on Earth (60 km). The curve alone, tuned on the small
  planet, kept 35 % on Earth there: its night clouds glowed grey against a
  black sky and sea.

Another body's eclipse multiplies both, and the clouds' sky light dims with
it (`eclipseVisibility`, see [Shadows](shadows.md)).

The clouds' own shadow falls on the terrain, the far water and the near sea
(`cloudShadowDepth`, through the deck's field: see [Shadows](shadows.md)).

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
in front turned off the clouds came out white; and too
few samples of that air per vertex (as orange with 8 as with 4).

None of the clouds (the raymarch, the deck, the cirrus)
draws any air in front of them, since that haze still showed them yellow:
seen at a grazing angle (from orbit, anywhere on the lit side, or at the
horizon and in the sea's reflection) they turned yellow even at noon, and
whitened as the camera closed in. The lit top is far brighter than the sky,
and the extinction of a long path removes its blue, while the light the air
adds is far too dim to restore it; desaturating either (keeping only the air's
luminance) did not look natural either. A cloud keeps its own colour at any
distance, and the sun's reddening (`cloudSunTint`) still turns a dusk cloud
orange. The cost: no distance haze, so far clouds are as contrasty as near
ones.

### Cirrus

High cirrus over the cumulus (`PlanetClouds(cirrus=True)`, or
`CloudLook(cirrus=True)` in a `World`) is drawn by the march's **high
layer** ([clouds.md, "The high layer"](clouds.md#the-high-layer)): thin ice
cloud in a slab of its own, 11-12.5 km up (`genera.HIGH_SLAB`), crossed in
one read per crossing, not in steps, as flow streaks along the upper wind
([clouds.md, "The cirrus"](clouds.md#the-cirrus)). It follows the planet's
weather, from the same systems as the cumulus:

- **Cover**: the cirrus cover map (`weather.cirrus_cover_map`, 6 × 128²),
  the share of the layer that holds cirrus:
  - anvils over the wet tropics and streaks along the meandering jets
    (or `WeatherSpec.cirrus_zonal`);
  - the systems' own cirrus (`synoptic_cube`'s Ci and Cc,
    `weather.CIRRUS_SHEET_GENERA`): the jets' streaks ahead of the warm
    fronts, the tropical cyclones' outflow. Their cirrostratus is a genus
    of its own in the high layer (`genera.WEATHER_GENERA`);
  - mesoscale texture;
  - humidity, which counts for less than for the cumulus.
- **The upper wind** the streaks follow, by its stream function
  (`genera.upper_stream`, on the GPU `genera.high_wind_cube`), the
  weather's (`weather.upper_stream`): each jet a Gaussian profile across its
  meandering axis (measured across it, not along the meridian), plus one
  Gaussian per weather system (`synoptic.jets.system_rows`), turning the
  wind around each low and over each high and tropical cyclone's outflow.

**Draw order** needs nothing of its own: the march composites the high
layer's crossings with the low clouds' along each ray (the downward one
before the bands', the upward one after), so the cirrus is behind the
cumulus from below and in front of it from above, for any camera. The far
plane (`World.depth_ranges`) reaches the high layer's middle where it is
higher than any ground.

**Lit** as a thin slab (`thinSlabRadiance`): the ice's phase function (the
`CIRRUS` preset's optics), no sun march, the sunlight reddened at the
layer's own height, so it stays lit after the cumulus below has gone into
the planet's shadow, then twilight; no air in front of it (see
[clouds at the horizon](#clouds-at-the-horizon)).

**History.** Until Task 24 of the cloud genera the cirrus was a sheet
of its own: two dozen patches of cirrus uncinus baked offline into an
atlas, stamped as quads on a sphere along the upper wind, drawn in two
nodes (where sight lines leave the sphere, in the sky's bin; where they
enter it, in the transparent bin). Its square patches showed, and it cost
0.35-0.55 ms against the march's 0.2 ms; the march's streaks replaced it.
Two designs before it were dropped: noise across the stream function's
contours (regular parallel bands, no head and no tail), and stamping in the
shader (texture bombing: about 30 candidates a pixel on a 3D lattice).

### Not yet

- **Wind:** the clouds are static.
- **The cirrus casts no shadow**, on the cumulus or on the ground.
- **One cumulus-like type per planet** without the genera (cumulus in the
  demo), with the cirrus above it.
- **Flying through the cirrus:** from inside its slab the high layer draws
  only the crossing of its middle ahead, with no volume.
- **The cirrus in the sea's reflection:** the reflection draws the deck
  alone.
- **Weather that moves:** the weather is a function of time
  (`WeatherSpec.time_s`), but a running scene is drawn at one time; moving it
  means rebuilding the maps (a few seconds) as time goes on.
