# Clouds

[`world_builder.clouds`](https://github.com/JoeDalton/WorldBuilder/tree/main/src/world_builder/clouds/)
draws a planet's clouds. The low clouds are **raymarched**: each pixel steps
along its own view ray through one density field and integrates its light
([the raymarch](#the-raymarch)). One idea runs through it, and most design
decisions follow from it:

**A cloud is not an object.** A cloud *type* is one density field spanning a
slab of sky, and the clouds you see are its features. That is what lets clouds
shadow each other, and why their silhouettes are exactly as crisp as the field.
Outline, opacity and colour are continuous functions of world position; nothing
about a cloud's look is carried by a particle, a mesh or a sprite.

The low clouds used to be drawn as **billboards** too: camera-facing quads
placed in the field by rejection sampling, each a sample of the volume, in LOD
shells per cube-sphere tile. They were removed on 2026-11-15. They were the
cheaper choice over a distant open sky on a weak GPU, but their cost grew with
the cloud's volume on screen (overdraw: 15 ms inside the slab on a laptop's
GPU, 41 ms in a dense cloud, 60-70 ms near the genera's big clouds), and the
march, the default at every preset since #45, was smoother and cheaper
wherever clouds fill the view. Their
measurements stay in [performance](performance.md) as history.

## The parts

A planet's clouds are one part,
[`PlanetClouds`](../../src/world_builder/clouds/planet.py), drawn as two layers
by two techniques:

- a **volume**: the low clouds, cumulus by default (any `CloudType`), or the
  [genera](#genera), raymarched as one union
  ([`march.py`](../../src/world_builder/clouds/march.py), `CloudMarch`) from
  the ground to orbit. It needs the split cameras
  (`PlanetClouds.use_split_depth`, which a `World` calls). Around it:
  - [`field_setup.py`](../../src/world_builder/clouds/field_setup.py): what
    the field is built from, shared by the march and its tables: the noise
    volume (`_noise`), the resolved field and its weather (`planet_field`,
    `planet_weather`), the coverage calibration (the column peaks, `_peaks`,
    and the shift tables by depth), the deck's light table, the march's
    distance bands of octaves (`OctaveBand`, `octave_bands`) and
    `sun_visibility`, the clouds' shade on the CPU for what the CPU lights;
  - [`passes.py`](../../src/world_builder/clouds/passes.py): the march's depth
    passes' render states and cameras (`pass_states`, `pass_camera`) and the
    cards that lay its result over the view (`composite_card`);
  - [`deck.py`](../../src/world_builder/clouds/deck.py): the **deck**, a
    sphere at mid-slab carrying the far field (`deck_mesh`, `deck_shader`,
    its bin `CLOUD_BIN`), drawn only in the sea's reflection;
    `shaders/cloud.frag` is its fragment stage;
- optionally (`PlanetClouds(cirrus=True)`), the weather's **cirrus** in the
  march's [high layer](#the-high-layer), streaks along the upper wind.

The part checks the rules the layers must keep (`check_layers`): the high
layer's slab lies above the volume's slabs and inside the air.

[`weather.py`](../../src/world_builder/clouds/weather.py) builds the weather
they follow: a cube map of target coverage (`WeatherSpec`), turned into the fbm
shift that gives it exactly (`shift_table`); the upper wind; and the cirrus's
cover. The planet-wide design (the weather, the deck's far field, the
terminator, the haze, the cirrus) is described in
[Atmosphere, sky and planet clouds](atmosphere.md#planet-clouds). This page
covers the field and the march that samples it.

## The field

[`noise.py`](../../src/world_builder/clouds/noise.py) owns what a type looks
like:

- **`DensityField`** is the whole shape: the slab, the per-axis noise scale
  (hence the feature size), the four fbm octaves, how much of the sky is
  cloud, and the vertical profile. Frozen, so it cannot be mutated behind the
  shader's back.
- **`coverage`** is *how much* cloud there is: the fraction of the type's own
  slab that is cloud, zenith-projected (a column counts if it holds cloud at
  any height). The weather map moves it by direction.

  The smoothstep window the shader uses is *derived* from it:
  `sphere_column_peaks` measures the per-column peak field strength over the
  planet and sorts it, which is the exact inverse of the coverage function, so
  the threshold is a quantile of that array, with no solver and no fitting
  (`threshold_from_peaks`, `with_threshold`). It has to be derived for two
  reasons:
  - which raw fbm value means "30% of the sky" depends on the noise volume
    and therefore the seed, so it cannot be baked into a preset;
  - deriving it decouples *amount* from *shape*, so retuning the octave
    weights does not silently change how much cloud there is.

  Coverage is not orthogonal to cloud type. Thresholding a 1/f field is a
  percolation transition: low coverage gives isolated puffs, high a connected
  sheet with holes (`CUMULUS` at 0.9 is genuinely stratus-like). It costs
  too: a covered column is also cloudy over a taller stretch of its slab, so
  more of the march's steps land in cloud and fewer rays turn opaque early
  over open sky.
- **`edge_softness`** is the window's width, in standard deviations of the
  field's own fbm (`fbm_sigma`), so it means the same for every type and
  survives a change of octave weights. A narrow window is the entire source of
  the crisp cauliflower edge.
- **`value_noise_volume`** and **`build_noise_texture`** generate the single
  tileable octave the shader taps four times, as a `sampler3D`. It is not a
  baked fbm: the octaves are sampled at 1/2/7/16 with independent offsets,
  baking would freeze their relative phase, and resolving the 16x octave
  would need 1024³. The two finest are also **turned** off the noise's axes
  (`OCTAVE_TURNS`, the shader's `OCTAVE_3_TURN` / `OCTAVE_4_TURN`): value
  noise carries its lattice's planes, flat spots and creases, which at their
  few hundred metres drew interference lines across a tower's face seen at a
  grazing angle (the billows are turned for the same reason).
- **`density_on_sphere`** evaluates the field on the CPU as the shader does:
  the same trilinear half-texel addressing, the same 8-bit quantisation. The
  tests measure the drawn field through it (the congestus's and the anvil's
  shapes, the coverage), so what they measure is what the march draws. (The
  billboards, removed 2026-11-15, also placed their quads
  with it, which made its exact equality with the shader load-bearing.) The
  shader side is one include, `shaders/cloud_field.glsl`. Its `cloudDensity`
  stops after one or two octaves where the rest (at most their weights)
  cannot reach the threshold: exact, and most of the samples along a ray are
  clear air.
- Types share one volume, decorrelated by a constant noise-space offset
  (`field_offset`): an independent field with identical statistics, for the
  cost of an add.
- **Per-genus fields on the GPU** ride in a `layerParams` uniform array, an
  entry per genus (`GENUS_VEC4S`, sixteen `vec4`s: the field's
  `LAYER_VEC4S` thirteen, then its coverage row) for up to `MAX_GENERA`
  (ten), then the reflection's deck's. Each genus has `DETAIL_LEVELS` (eight)
  detail levels, its octave bands by distance and its far level, which differ
  only in their octaves and threshold offset (`detailOctaves`,
  `genusLevelOffset`). They are packed by `noise.pack_genus_params` and read
  by `getField(genus * DETAIL_LEVELS + level)` in `cloud_field.glsl`. A
  uniform array rather than a texture: a sample needs about two dozen of
  its type's parameters, which as `texelFetch`es would be two dozen texture
  reads per sample.

## The types and their cost

[`cloud.py`](../../src/world_builder/clouds/cloud.py) holds the types and
what they cost:

- **`PRESETS`** maps `CloudType` (`CUMULUS`/`STRATUS`/`CIRRUS`/`CUMULONIMBUS`,
  and `CONVECTIVE`, [the genera](#genera)'s) to a `CloudSpec`: a field and its
  optics. Every shape parameter lives in the field.
- **Extinction** is the field's own: the view ray takes the field's density
  per metre of path, times `VIEW_EXTINCTION_SCALE` (below), and the sun march
  the field's density unscaled. (The billboards, removed 2026-11-15, had to
  divide it by their overlap and their sprites' mean alpha, measured from
  their placement's accept rate, so that a ray through overlapping quads took
  the field's optical depth.)
- **Quality** (`CloudSettings`, [Quality settings](quality.md); the levels are
  the shared `QualityLevel`) scales a type's cost, by preset across
  LOW/MID/HIGH/ULTRA or knob by knob. HIGH is the identity: what the presets
  ship. The march's own knobs are its most steps a ray (`march_steps`), the
  share of the view's pixels it traces a frame (`march_block`), the octaves
  per distance band (`shell_octaves`, over `octave_bands` bands) and the sun
  march (`march_scale`, through `scale_march`; `with_settings` applies it).
  The sun march is stepped gently (×2/3 a level), because it makes clouds
  shadow themselves and carries most of their form. Its **reach**
  (`sun_steps × sun_step_length`) is held constant while only its resolution
  changes. Beer-Lambert exponentiates the integral, so a coarser sampling of
  the same path approximates it well; a shorter path approximates nothing.
  (The billboards' count, the frame time's knob while they lived, was changed
  through their radius at a fixed overlap; the march has no count.)

## Lighting

Each sample is lit as the reference shader lights a cloud:

- a dual-lobe Henyey-Greenstein phase for the forward-scatter silver lining;
- a short march toward the sun through the field, which is what makes clouds
  shadow each other (near the camera; the light volume and the shadow maps
  give the rest, [the raymarch](#the-raymarch));
- the "powder" dark-edge term and a multiple-scattering fill;
- a second order of scattering (`CloudOptics.octave_gain`, 0.3, and
  `octave_attenuation`, 0.5; Wrenninge's octaves, as EVE-Redux's clouds light
  theirs with a multiple-scattering lobe of their own): 30 % of the sunlight
  sees only half the sunward depth, with a phase half-way to isotropic. It is
  a share taken out of the first order, so a side-lit cloud in full sun keeps
  its brightness and only shadowed sides and anvil undersides open up; the far
  clouds read its column mean from the deck table's fourth channel
  (`noise.deck_light_table`). At 0.5 of the light and a quarter of the depth it
  flattened towers;
- the sky's display transform.

All of these are live uniforms: `set_sun` moves the sun with no rebuild. These
rules live in the Python and in `cloud_shading.glsl` (included by the march's
`cloud_march.frag` and the deck's `cloud.frag`), and each has a visible
signature when broken:

- **The display transform** is the sky's (`displayColour`,
  `atmosphere/shaders/display.glsl`): ACES per channel and display gamma,
  keeping the hue only above white, at the sky's exposure (0.9). With a soft
  knee instead (hue kept at every level, no gamma), the warm excess of lit
  tops seen through a long low path of air showed orange at noon, and hazed
  cloud never met the sky's colour behind it
  ([atmosphere](atmosphere.md#clouds-at-the-horizon)).
- The multiple-scattering fill is **sky-coloured and not phase-weighted**:
  light reaching a shadowed interior comes from the sky and neighbouring
  cloud, not down a beam. Sun-tinted, dusk shadows go orange instead of
  blue-grey; phase-weighted, shadowed cloud toward the sun outshines side-lit
  cloud.
- The fill and the second order **fade with the cloud
  above** (`DIFFUSE_FADE` in `cloud_shading.glsl`): times
  `1 / (1 + 0.2 * depth)` of the column's optical depth straight up, from the
  light volume, a diffusion's slow fall. The sun's depth a sample sees stops
  at its window (640 m), so before this a cumulonimbus's base under
  kilometres of tower was lit as if under 640 m of cloud: at noon its
  anvil's underside was 0.74 of a lit cumulus's brightness (display levels),
  0.57 now; thin cumulus are unchanged. The column up, not toward the sun:
  toward a low sun it crosses tens of kilometres of other clouds, and towers
  went brown at dusk. The sky's floor under the clouds (`skyOcclusion`) is
  not faded: a tower's side sees the open sky around it, and without that
  blue its dusk side browned too (the anvil's underside 0.48 with it).
  `--cloud-debug 2048`: without (the old look).
- The **powder** term takes the *local* density over a fixed reference length.
  Fed the sunward depth it inverts (the lit surface darkens most); fed a
  length the renderer chooses (the billboards' quad chord, while they lived;
  the march's step) it would change with the renderer's resolution.
- The **sun march** uses only the two lowest octaves, affinely calibrated onto
  the full field (`noise.shadow_calibration`). Plain renormalisation leaves
  mean and spread too high: the march sees 1.65x the cloud and greys the tops
  in full sun.

Two rules of the billboards' lighting went with them: sampling the field on
the quad's plane rather than mid-chord (overlapping quads otherwise disagreed
about where the cloud was), and crossfading their LOD shells on optical depth
rather than alpha (a light ring swept with the camera). The march samples the
field where it is and has no shells to crossfade.

## Genera

[`genera.py`](../../src/world_builder/clouds/genera.py) adds the low and
middle genera (stratus, stratocumulus, nimbostratus, altostratus and
altocumulus) and the convective continuum from fair-weather cumulus through
congestus to cumulonimbus (`PlanetClouds(genera=GeneraSpec(...))`, `CloudLook.genera`,
`--genera` in the demo and the scripts sharing its options, on by default
there, `--no-genera` for the plain field; opt-in in `cloud_bench.py`). Its maps are
a **placeholder** for the weather (issue #8): features laid out around one
point. One `CloudMarch` draws them all, as one union of genera.

**The genera and their sky bands.** A genus is one field (one cloud type)
over its own slab, with its own coverage, shift table and far table
(`GeneraSpec.genera`, each a `genera.GENUS_PRESETS` entry: its cloud type,
coverage source, `far_shadow` and slab). There are six, in
`GeneraSpec.genera`'s order:

| Genus | Slab | Coverage | Density |
|---|---|---|---|
| convective (`CloudType.CONVECTIVE`; humilis, mediocris, congestus, cumulonimbus) | 2.2-10.5 km | the weather cube (coverage, shift, depth, spread) | its own field: depth, cell octave, anvil, shelf |
| stratus | 0.9-1.4 km (`GeneraSpec.stratus_altitude`, the type's 500 m) | cube A, channel 0 | recipe: value fbm, stretched 3x, wispy base |
| stratocumulus | 0.8-1.8 km | cube A, channel 1 | recipe: closed cells 1.5 km, billowy |
| nimbostratus | 0.6-5 km | cube A, channel 2 | recipe: value fbm 20 km, dense, ragged base |
| altostratus | 4.7-6 km | cube A, channel 3 | recipe: value fbm stretched 5x, thin |
| altocumulus | 3.5-3.8 km | cube B, channel 0 | recipe: closed cells 350 m, a lens |

The convective genus is the primary one (genus 0): the cumulus field, its
clouds growing with the convective **depth** (0 to 1, a map of its own); the
cumulus species are depths of it, not genera of the march. The others are
[recipe genera](#the-recipe-genera), one generic density driven by knobs.
The default (`GeneraSpec()`) lists all six but has features only for the
convective clouds and the stratus; the others are not marched there
([an empty genus is not marched](#the-recipe-genera)). The high layer
(cirrus, cirrostratus, cirrocumulus) is [a band of the march's own](#the-high-layer)
above them; it draws the weather's cirrus too
([the weather's cirrus](#the-cirrus)).

Slabs may overlap: `march.sky_bands` merges them into at most four
(`MAX_SKY_BANDS`) disjoint bands of sky, each with a bit mask of the genera
in it (`uSkyBands`' z); a test marches a third, stratus-like genus at 3-4 km
inside the convective slab. `planet.check_layers` refuses only what the march
cannot hold: more than `noise.MAX_GENERA` (10) genera, or slabs making more
than four bands, and a cirrus whose base is not above the **union's** top
(the highest band's; `PlanetClouds.top_radius` takes the same) or whose top
is not inside the air.

**The raymarch with genera.** `CloudMarch(genera=...)` marches every genus in
one pass, so a stratus cloud in front of a tower and one behind it composite
correctly. Each sky band gives up to two stretches of a ray
(`march.band_segments`, the CPU mirror), at most eight, sorted along it; the
steps are laid as for one slab, and a gap between bands costs no steps. At
each sample the march loops over the band's genera whose slab holds it and
whose coverage there is above 0: each takes its own detail level of its own
entry, its own weather (once per `uWeatherStretch` of ray, per genus), its
own footprint fades, billows and far mean, and adds its extinction
`σ_g`; the sample's alpha is that of `Σσ_g`, its light `Σ (σ_g / σ)` of
each genus's `cloudRadianceAt(genus, ...)` (one lighting path, its far table
the genus's). A band of one genus (the demo's: the convective clouds, the
stratus) takes its samples in a loop of its own, without the genus loop:
through the genus loop the same light cost 0.4-0.6 ms more between the
towers (`cloud_bench.py --genera`, 2560 × 1440), its lighting inside a second
loop; a single (sample, genus) loop, and slots of weather per genus, cost
0.2-0.4 ms more too. A band shared by several genera pays the genus loop,
and each genus in it about 0.35-0.4 ms more where the band is in view
(performance §37): in the showcase the convective band holds the
nimbostratus, the altostratus and the altocumulus too. Every genus has all the
`octave_bands` asked for (with two marches, the convective clouds' and the
stratus's, each had used to halve them to fit eight layers). The reflection's deck is genus 0's
alone.

**The weather of the other genera.** Genus 0 reads the weather cube
(coverage, shift, depth, spread), so a planet without genera is one genus
and unchanged. The others read their coverage from a channel of **coverage
cube A** (`genusCoverA`, RGBA16F, `genera.coverage_cube`: the stratus,
the stratocumulus, the nimbostratus and the altostratus in channels 0-3) or
of **cube B** (`genusCoverB`: the altocumulus in channel 0, three channels
free for the high layer), and turn it into their fbm shift through their row of the
**shift table** (`genusShift`, `SHIFT_TABLE_SIZE` × `MAX_GENERA`, R32F;
`weather.build_shift_texture`), their channel and row in row 13 of their
entry (`genusWeatherAt`); no depth, no spread. Their far tables are pages of
`genusDeckLight` (a 2D array, page = genus; `genusFarTable`). The shift is
taken of the filtered coverage, not filtered from a cube of shifts: at a
stratus's edge it differs slightly from the stratus's old cube of its own, and
`genera.genus_weather_at` is its CPU twin (the ground's shade,
`field_setup.genus_sun_depth`). A full weather cube per genus was rejected:
four channels of 6 × 512² per genus, where one channel carries all a stratus
needs.

**One light volume and one map pair for all the genera.** The light volume
and the cloud shadow maps ([Shadows](shadows.md#the-cloud-light-volume)) hold
the genera's **summed** depth: their bakes march the sky bands
(`march.sky_bands`: the genera's slabs merged, a gap between two crossed in no
steps; `cloud_field.glsl`'s `skyShadowLevels`), every genus in a band at the
same samples by its own field and weather, and a depth is a sum. The light
volume keeps 32 slices a band, so the 500 m stratus deck keeps the 16 m
slices it had alone. Each genus takes its own sun window from the volume and
its **`far_shadow`** (`noise.GenusField`, `genera.GENUS_PRESETS`; row 13's w
of its entry) of the depth beyond it: 0 for the convective clouds (their
window carries their form; the whole chord blackens a tower), 1 for the
recipe genera: a stratus so takes the whole chord of the towers above it. Under the
towers its sky light dims too (the volume's up depth holds them). The ground,
sea and models take one shadow, `exp2(-sum)`, the product of the genera's
shades (`PlanetClouds.shadow_caster()`, `planetCloudShadow`;
`CloudMarch.genus_caster(g)` is one genus's part, on the CPU). A volume and a
map pair for the convective clouds and another for the stratus, before, read
the towers' shadow on the stratus from the convective clouds' maps and made every reader carry a second set of inputs
(`uCloudShadowOther`, `otherLight`, `otherShadowNear`...). The CPU's shade
sums each genus's depth through its own slab
(`field_setup.sun_visibility`); the GPU marches a band's genera at the band's
shared steps, which differs only by the steps where genera overlap.
The convective weather is read at every sample (`WEATHER_STRETCH_GENERA`): a
tower's depth, which sets its top, changes over tens of metres, and read once
a kilometre it showed as horizontal bands across every tower. The fair-weather
march keeps the kilometre.

**The depth** (`DensityField.min_top`, `anvil`, `cell_weight`, all neutral
by default):

- the clouds' **top** rises from the cumulus's (1 km above the base, a
  share `min_top` of the slab) to the slab's (10.5 km). The lift, which bulges
  the tops, and the base's fill follow the local top (`noise.top_fraction`).
  Above it nothing is cloud;
- **a congestus is a tower** (`DensityField.deep_exponent`, `deep_erosion`,
  `cell_power`): the threshold's rise up the column follows the depth, from
  the cumulus's (`top_exponent` 3, `top_erosion` 0.4) to 10 and 0.25 at
  depth 1, so it stays low up the column and rises late, and the cell
  octave below grows as the square root of the depth. With the cumulus's
  rise at every depth, a depth-½ column stopped 1.5-2 km under its local
  top (6.85 km; cloud in 2 % of the samples at 5.3 km, none above 5.8 km)
  and what rose was scattered (53 % of the cloud above 4 km stood over
  cloud at 3 km): the congestus read as cumulus mediocris
  (`tests/test_clouds_congestus.py`). The demo's congestus features are at
  coverage 0.65: at 0.45 they were detached blobs, as the cumulonimbus was
  (below);
- a **cell octave**, four times coarser than the first (8 km features),
  adds to the fbm in proportion to the depth, about its mean: it gathers the
  deep clouds into towers;
- **the anvil is smooth** (`noise.anvil_weight`, `anvil_fbm`): from half the
  local top of a deep cloud (`ANVIL_SMOOTH_BAND`, ahead of the fall below),
  its fine octaves and the raymarch's billows fade to their means, its two
  coarse octaves are read at a quarter of their frequency (`ANVIL_SCALE`,
  ~8 km features, as the shelf's are) and the threshold's window widens to
  twice itself (`ANVIL_SOFTEN`): smooth over kilometres and soft-edged, not
  the tower's cauliflower spread sideways. The sun march, the shadow maps
  and the light volume see the same (`noise.march_pairs`, `mix_pairs`). The
  anvil used to be the threshold's fall alone, on the tower's own fbm: its
  fine octaves and billows printed lumps, and with them gone the coarse
  octaves still drew kilometre-wide ones (measured: the anvil band held 2.3
  times the tower's energy under 500 m, then 0.75 times under 2 km;
  `tests/test_clouds_anvil.py`). With the fade over the fall's own band the
  spread's underside kept the tower's lumps;
- under a deep cloud's top, the threshold falls (`noise.ANVIL_BAND`): its
  **anvil**, which spreads past the tower as a **shelf** (below).

The weather cube carries the depth beside the coverage and the shift (its
blue channel), and the anvil's spread (alpha; one fetch). The shader and
`density_on_sphere` read it, and for every other field it changes nothing.

**Coverage by depth.** A deep column may grow taller, so it holds cloud at
any height more often: the same coverage needs less shift. The shift is
tabulated by depth too (`genera.depth_shift_tables`: one `shift_table` per
depth of `genera.DEPTHS`, the column peaks measured through each depth's
local top, as many more levels as it is taller) and interpolated
(`depth_shift`). Tested within 4 % at depths 0, ½ and 1.

**A cumulonimbus is coverage, not noise.** At the core's 65 % coverage, a
deep column held scattered blobs at every height (the cumulus noise
decorrelates every 2 km): a quarter of the volume below 6 km was cloud. At
100 % the core fills (87-97 %) and the noise only carves its edges. So the
maps make a cumulonimbus a disc of coverage 1 and depth 1, the depth staying
1 to 1.4 radii where the coverage falls (the anvil's band holds most of the
cloud there), then a ring of congestus (depth ½) to 2.2 radii. Those change
over 2-4 km: the genus maps are twice the weather's resolution
(`GENUS_SIZE`, 1.85 km texels on a 600 km planet).

**The anvil's shelf.** A cumulonimbus's anvil spreads under the
tropopause well past its tower: a smooth lens, thick by the tower and thin
at its rim, unlike the cauliflower column under it. The shelf is drawn
wherever its own map says, over whatever columns lie below:

- **its spread**, a fourth map (`genus_planes`, `genera.anvil_spread`): 1
  to a radius of a cumulonimbus, 0 by 3 (36 km across for a 6 km tower) in
  calm air;
- **on the upper wind**: the cirrus's (`genera.upper_wind_at`: the jets,
  swirled by the weather systems; it blows along the stream function's
  contours), read at the tower. In the jet's fastest the reach is 3.5 times
  longer downwind (63 km for a 6 km tower), half as long upwind (a short,
  sharp overhang) and a quarter shorter across: a plume, thinning
  downwind as its spread falls. `GeneraSpec.wind` sets one wind for all;
- **a lens**: its underside (`noise.shelf_underside`) is at 60 % of the
  slab (7.2 km) at spread 1 and rises to the slab's top as the spread falls.
  Over it the threshold, a depth-1 column's (`noise.shelf_lift`), falls by
  `SHELF_CLEAR` over 1 km; none under it;
- **its top fades in density**, not in threshold (from 76 % of the slab to
  its top): a flat, smooth top cut by the threshold is a hard plane (the
  billboards, removed 2026-11-15, were each cut along one line across it, and
  from above the sheet was a stack of pills);
- **smooth**: the two coarse octaves only, at the anvil's quarter frequency
  (`noise.anvil_coarse`), the others their mean (`noise.shelf_fbm`; no
  cauliflower), its window the anvil's softened one, and its own fbm shift
  (`DensityField.shelf_shift`, `noise.with_shelf`): 97 % of its middle is
  cloud, so from below it is an opaque ceiling;
- the field takes the column's density or the shelf's, whichever is
  denser: over the tower nothing changes under the shelf, over a cumulus
  the shelf floats above it.

Tried on the way, with the billboards: a squash of the noise in the shelf's
band (features three times wider than tall) sliced into pills by the
billboards; a softer threshold window left every billboard half transparent,
and their round sprites showed from below as bubbles; a shelf coverage
counted on columns (any cloud through its band) left separate puffs.

From below it is a grey ceiling; under it, the sky term is occluded from
the shelf's own underside. The deck samples it at the top of its rise.

**Their cost** is the march's, which stops once a ray is opaque, so a big
cloud near the camera spares it the ones behind. Between and under the
towers it was 2-2.4 times cheaper than the billboards' overdraw (7.6-8.8 ms
at HIGH on an RTX 3070 at 2560 × 1440; performance §23, "The raymarch with
genera"); §37 is the genera's baseline from the user's start.

**The showcase.** [`scripts/demo_genera.py`](../../scripts/demo_genera.py)
(the terrain demo with `--showcase`) shows every genus alone, for look
development: `genera.showcase_features` lays out one feature per genus and
cumulus species, the cumulus in a row along the northern horizon
(`SHOWCASE_ROW`: humilis, mediocris, congestus, cumulonimbus, west to east,
35-45 km from the start), the stratiform and mid-level genera along the
southern (`SHOWCASE_SOUTH`, east to west: stratocumulus at bearing 120,
stratus 150, nimbostratus 180, altostratus 210 and altocumulus 240, 38
km out, sheets 14 km wide; a slot whose genus is not a preset is left out,
`showcase_slots`), apart (a test checks no
feature's cloud reaches another's, the cumulonimbus's anvil counted at 4
radii, the slots to come too, and that sheets 15 km wide fit every
southern slot). Ruling: one horizon cannot hold sheets 10-15 km wide and
the cumulus at a readable distance. The low sheets sit east of south, away
from the sun (azimuth 210 at the start), the altostratus at it ("ground
glass"), the altocumulus beyond it. A stratiform feature fills its own genus's coverage map
(`genera.genus_coverage`), and
`GeneraSpec(background=False)` removes the fair-weather cumulus between
them. The cumulus species are convective depths (`GENUS_DEPTHS`: humilis 0,
mediocris 0.2, congestus 0.5). Keys 1-9 hide or show each genus of the march
(`CloudMarch.set_genus_visible`, a bit of `uGenusMask` the view's samples
test before any noise); a hidden genus still shades the others and the
ground, and the cumulus species, one genus, go together.
`terrain_shots.py --showcase` has a `genus_<name>` pose per feature: a
cumulus from 1 km over the start, along its bearing; a low sheet (St, Sc,
Ns) from 60 m over the ground under its centre, looking along its bearing 6
degrees over its far edge (its underside the ceiling, its outline in the
lower third); the altostratus from by its near edge, looking at it against
the sun (`SHOWCASE_AGAINST_SUN`: its "ground glass"); the altocumulus from
under its centre, as the low sheets (its pattern of patches is its
identity). From the start the
stratus, 38 km off, was a line 1.5 degrees up; from just outside its near
edge, a slab side-on, a lenticular's look; the WMO's photos of St and Sc are
from under them. At 50-60 km the features were specks in the
haze; at 30 km the cumulus species gave a cloud or two each, hence wider,
sparser discs a little farther. At 2560 x 1440 HIGH, offscreen, every pose
takes 8.6-12.5 ms, above the showcase's 60 FPS: the cumulus row 12.2-12.5
ms, the southern sheets 8.6 (St) to 10.4 ms (Sc) (performance §37, "Phase
5 summary").

The high genera have an overhead row (`SHOWCASE_HIGH`): the high layer
(11-12.5 km) lies over the rows, so each is shot from the start's ground
looking up at its own pitch (`SHOWCASE_HIGH_PITCH`), with the rows' clouds
under the view's bottom edge or off its sides. The cirrostratus is a veil
64 km wide to the west (bearing 280, 25 km), in the gap between the
altocumulus (240) and the humilis (318). Its pose `genus_cirrostratus`
looks 35° up at the sun, set at azimuth 280 for it, so the veil and its
halo are seen around the sun. The cirrocumulus is a patch 12 km wide to
the east-south-east (bearing 120, 16 km), over the low sheets and away
from the cumulonimbus's anvil (60). Its pose `genus_cirrocumulus` looks 34°
up at it. Ruling: the high discs never overlap each other (a test checks,
and that every low feature's top is under each pose's view or off its
sides). The low clouds under the cirrostratus's west edge (the humilis)
are never in its pose's view. The cirrus's slot is Task 23's.

**Not yet, for the convective clouds.** The anvil does not lean downwind with height (a shear; it is
the genus framework's, docs/superpowers/plans/2026-11-15-cloud-genera.md).
The plume is straight: the wind is read at the tower, not
along the plume. Its underside is lit as the rest of the cloud: no darker base
for deep clouds. The tower's stem, at coverage 1, has
straight sides. From the ground, the air hazes out a cumulonimbus's lower half at
40 km, and its top half reads as a large cumulus.

### The recipe genera

The convective clouds are a field of their own (the cumulus field's depth,
cell octave, anvil and shelf). Every other genus (the stratus, the
stratocumulus, the nimbostratus, the altostratus and the altocumulus) is instead
a **recipe genus**: one generic density, `genusRecipeDensity` in `cloud_field.glsl`
(`noise.recipe_density_on_sphere` on the CPU), driven by knobs of its
`noise.GenusField` (a `GenusPreset.recipe` dict, and `GenusPreset.field` for
its `DensityField`'s own: window, noise scale, density):

- **base value**: the field's value fbm and/or Worley cells at
  `cell_size`, mixed in by `cell_fill`, from `noise.worley_volume` (a
  tileable F1/F2 volume, 8 cells of 8 texels a side, the `cloudCells`
  texture): `noise_kind` `"worley"` reads `1 - F1` (round blobs about the
  cells' points, which merge into one mass as the coverage rises), `"cells"`
  reads `F2 - F1` (0 on the walls between cells, highest at their points:
  however high the coverage, the walls stay thinner than the cells, a
  tessellation; its mean, `noise.cell_means`, in the deck entry's row 14);
- **anisotropy**: coordinates squeezed along the genera's focus's east by
  `stretch` (a fixed axis, the deck entry's row 15: a direction that turned
  with the position would tear the field) and in height by
  `vertical_stretch`, about the slab's base;
- the weather's **shift** and the field's **threshold window and lift**, as
  every genus; the coverage is calibrated on the recipe's own column peaks
  (`noise.recipe_column_peaks`, `field_setup.recipe_field`; within 5 % of
  the coverage asked, tested);
- a **profile** multiplying the density: a soft bottom and a soft top (slab
  shares), or the lens `4h(1 - h)`;
- **erosion** at the edge only: small Worley cells (`cell_size / 6`) move the
  threshold about their mean, billowy (> 0, the cells' centres stay) or
  wispy (< 0, their walls stay), over the window and a skin past it, so the
  core holds; with `erosion_base`, only near the slab's base (whole at the
  base, gone by that share of the slab: a stratus's fractus). The erosion's
  noise still lifts each column's highest value, so the coverage
  calibration takes it in (`noise.recipe_column_peaks`: the cells'
  deviation, a window's share of it, where a column's cloud begins);
  without, the stratus's wispy erosion gave 5 % too much cover at 0.3.

Octaves, cells and erosion finer than a pixel (or a step) resolves fade to
their means, as the view's `resolved()`. The knobs pack into rows 14-15 of
the genus's entry and row 10's free x (`noise.pack_genus_params`: zero for a
genus without the recipe); the deck entry's row 15 holds the shared frame
(the stretch axis, the cells' mean F1). A genus's coverage may come from
coverage cube A or B (`genusCoverB`).

**The stratus** (`GENUS_PRESETS["stratus"]`, 0.9-1.4 km, cube A channel
0): WMO's "flat, uniform base", grey, fog-like, its outline sharp. Value fbm
of 10 km features (`noise_scale` 0.0004; the research's 10-30 km) stretched
three times along the focus's east, its two fine octaves weak (weights
0.6, 0.3, 0.08, 0.02: at full weight, stretched, they drew streaks under
it, an altostratus's striations), a window of 1.2 sigma (2.5 left the
inside half-dense, a light, thin-looking veil; narrower, the inside is
solid, so the base is a uniform grey and the outline sharp), a soft top
(`profile` 0.03, 0.3), a flat base, and wispy erosion (-1.5) near the base
only (`erosion_base` 0.4): its rim ragged low down, as fractus. From under
it: a grey ceiling to its edge, the sky under that.

**The stratocumulus** (`GENUS_PRESETS["stratocumulus"]`, 0.8-1.8 km, cube A
channel 1): tessellations or rounded masses, non-fibrous, flat-based,
lumpy-topped, grey or white with "always some dark parts". Closed cells
(`"cells"`, `F2 - F1`) 1.5 km wide, twice as tall as wide (columns through
the slab: `vertical_stretch` 2), three quarters of the base value, the
rest fbm of 2 km features; the threshold rising 0.4 by the top with
exponent 1.5 (`top_erosion`, `top_exponent`): a cell's centre, its value
highest, reaches higher than its rim, so its top is a dome; a flat base
(`profile` 0.08, a soft top 0.1); billowy erosion (3 windows: at 1 the
domes stayed smooth). `density` 0.045 a metre: the research gives it more
water than the stratus (0.45 against 0.25-0.3 g/m³), but the stratus's
type is already at 0.05, and at 0.07 the stratocumulus's undersides went
navy; its closed cells and narrow window (0.8 sigma) make its cores
denser than the stratus's wider-windowed sheet.
`1 - F1` was tried first: at the showcase's coverage the blobs merged into
one mass with holes, a cumulus field's look, not a tessellation. The cells
at its edge, where the coverage fades, are thin wisps at the base (the
lift cuts them low, flat-topped) and sunlit, brighter than the shadowed
bases behind them; real decks break up the same way, so they stay.

**The nimbostratus** (`GENUS_PRESETS["nimbostratus"]`, 0.6-5 km, cube A
channel 2): WMO's thick, dark, diffuse rain-cloud layer that blots out the
sun. Value fbm of 20 km features (`noise_scale` 0.0002; the research's
20-50 km) stretched three times along east, its fine octaves faint (0.6,
0.3, 0.08, 0.02: featureless), a narrow window (1 sigma: solid inside, a
uniform ceiling), a top-hat with soft ends (`profile` 0.12, 0.15: a diffuse
base), wispy erosion of 500 m cells over its lowest 15 % (ragged where it
thins), and the threshold rising 0.15 toward the top (`top_erosion`: where
the coverage fades its top slopes down; at 0.05 the showcase's disc stood
as a dark brick, at 0.3 a dome). The densest genus (`density` 0.08 a
metre, against the stratus's 0.05): its far table lets under 2 % of the
noon sun through a full sheet (tested). It is lit darker than the stratus
it shares a type with (`GenusPreset.optics`: sky fill 0.05, forward gain
1.6) and steps the bakes 250 m toward the sun. From under it: a dark grey
ceiling, the sky under its far edge.

Its slab spans the low sheets' and the convective clouds' altitudes, which
merged all of them into one 9.9 km sky band: the stratus and the
stratocumulus took that band's light-volume slices (310 m, not 31) and its
bakes' steps, and in the showcase their shading turned grainy and the
stratocumulus's ground shadow patchy (mean difference 10-15 levels against
before). `march.sky_bands` now splits a band at a gap that only one slab
bridges (not the band's thickest: the convective slab holding mid-level
sheets stays whole), each part holding its side's genera and the bridge
(`band_masks`: every genus reaching into a band); the low sheets' shots are
back within 1-3 levels of before. The nimbostratus is marched in both
bands; the cut runs through it at 2 km, unseen in its dense body.

**The altostratus** (`GENUS_PRESETS["altostratus"]`, 4.7-6 km, cube A
channel 3): a grey or bluish, uniform or striated sheet through whose
thinner parts the sun shows as through ground glass. Value fbm of 16 km
features stretched five times along east, a wide window (2 sigma: its
density varies, thin and thicker parts), soft at both ends (`profile` 0.3,
0.3), no erosion (at -0.5 the 500 m cells, squeezed along east like the
rest, sawed its rim into teeth), its last octave dropped (weights 0.6, 0.3,
0.1, 0: the march's far levels drop it, and in a window this wide the cut
drew a contour across the sheet). Thin (`density` 0.0015: at 0.004 the
sheet hid the sun; now its disc shows, dimmed, and the far table lets
8-60 % of the noon sun through a full sheet, tested) and lit as ice
(`GenusPreset.optics`: sky fill 0.2, an ice cloud's forward lobe). Its slab
sits above the altocumulus's 3.5-3.8 km: a sample inside two genera's
slabs evaluates both. Its showcase pose looks at it against the sun
(`terrain_shots.SHOWCASE_AGAINST_SUN`).

A faint contour crossed the sheet near that pose's far side, in the
flat-colour debug view too (an alpha step, not lighting). Not the steps
(each segment takes at least `SEGMENT_SAMPLES`, and the sheet is thin
enough that a step's alpha is linear in its density): the march stops a
ray at transmittance `OPAQUE` (0.04) and counted it opaque, its alpha
snapped from 0.96 to 1. A dense cloud crosses that point inside itself;
a thin sheet seen at a grazing angle, whose path through it lengthens
across the view, crosses it along a line on the screen: a step of 0.04 in
alpha there, about 1.7 levels in the flat view, a brighter line lit (its
far side, past the line, a flat plateau: opaque). The light let through is
now eased to 0 over [OPAQUE, 2 OPAQUE] (`march.opaque_residual`,
`opaqueResidual` in `cloud_march.glsl`: value and slope continuous; the
transmittance itself above 0.08), so only rays already nearly opaque
change: the convective scenes' shots stayed at the noise floor and their
time did not move (performance §37).

**The altocumulus** (`GENUS_PRESETS["altocumulus"]`, 3.5-3.8 km, cube B
channel 0): WMO's white or grey patches, rounded masses or rolls,
regularly arranged, with shading, small clear gaps between them, most of
the elements 1-5 degrees wide seen more than 30 degrees up. Closed cells
(`"cells"`, `F2 - F1`) 350 m apart, 80 % of the base value, the rest fbm of
3 km features (the cells' sizes jitter); half again taller than wide
(`vertical_stretch` 1.5: one element per column) in a lens (`profile`
`4h(1 - h)`: rounded tops and bases) 300 m thick; billowy erosion (3
windows) at a sixth of the cells; unstretched; its last octave dropped; a
narrow window (0.8 sigma); `density` 0.05 a metre. Choices made on the
shots (its pose from under its centre, as the low sheets: its pattern is
its identity, and the sun, 30 degrees off the view, lights the rims):

- An element is about 1.2 cells wide (a Voronoi cell's). At the plan's 800
  m cells the elements were 10 degrees across overhead, cumulus humilis's
  size; at 350 m they are 3-7 degrees, the research's 50-450 m.
- A slab 1 km thick (the plan's 3.5-4.5 km) made elements thicker than
  wide: seen at 30-50 degrees up, tall pebbles; at 300 m they are flat
  patches (the research's 0.1-1 km).
- `1 - F1` (round blobs) merged into chains and clumps with wide gaps; the
  closed cells keep the regular arrangement and the small gaps at any
  coverage. Erosion 2 left their rims polygonal; 3 rounds them.
- At `density` 0.012 the patches were all white; at 0.05 their cores are
  grey and the rims lit. Its far table: three-quarter cover lets 61 % of
  the noon sun through (the stratocumulus 40 %), a full layer (opacus)
  4 %, less than the altostratus's 41 % (tested).
- Stretched, the erosion's cells (read in the recipe's squeezed
  coordinates) would saw the rim into teeth, as on the altostratus: no
  stretch (rolls, undulatus, would need one; not done).
- The Worley volume (8 cells a side) repeats every 2.8 km at this size.
  From 26 km up over the 14 km disc no repeat shows (the autocorrelation
  of its cloud mask has no peak at the period): the sheet is a sphere
  cutting the 3D volume obliquely, and the fbm and the coverage vary over
  it. Nothing is done to break it ([not yet](#not-yet-for-the-recipe-genera)).

Why one recipe instead of shader code per genus: the research's recipe table
(cloud_genera_research.md §3) differs between the stratiform and mid-level
genera only in these knobs, and one code path keeps one CPU mirror, one
coverage calibration and one cost; a new genus is a preset, not shader work.
Rejected: a `switch` on the genus with a hand-written density each, which
multiplies the CPU twins and the shader's size (and, measured, every branch
in the shared loops costs the convective clouds).

**Where the recipe is evaluated.** In the view march, in its variant
`cloud_march_recipes.frag` (`cloud_march.glsl` with `CLOUD_RECIPES`), which
`CloudMarch` compiles only when one of its genera is a recipe genus; and in
the bakes of the summed depths (the light volume's and the shadow maps',
`CLOUD_RECIPE_SHADOWS` in `skyShadowLevels`). Everywhere else (the
terrain's and the models' per-pixel shadow march, the sun march's local
steps) a recipe genus contributes nothing of its own: its depth comes from
the light volume and the maps, from the sample itself (`cloudSunDepth`
takes no local steps for a recipe genus: the march's local steps skip it,
so the volume read past them lost the first 180 m toward the sun, the
densest at a cloud's base, and the stratocumulus's undersides came out
light). Measured at 2560 x 1440, a never-taken
branch to the recipe in those shared loops cost the convective clouds
0.15-0.2 ms in the view's sample loop and 0.3-1 ms in the shadow march
(register pressure in the inlined loops), hence the variants.

The CPU's are the bakes': `field_setup.genus_sun_depth` (hence
`sun_visibility`) marches a recipe genus's own density as the maps do
(`noise.recipe_shadow_on_sphere`: the chord, steps and weather of
`cloud_shadow_on_sphere`, the recipe at its far level, the footprint one
step), and the light volume's twin too (`light_volume.GenusShade`'s
`recipe`). The terrain's per-pixel march without maps leaves recipe genera
out, so there the CPU's shade (the test balls') is darker than the
ground's under a stratus: the maps cover the view's surroundings, where it
shows. A recipe genus's far table (its far shading and the ground's mean
shadow, `genusDeckLight`) is its recipe's (`noise.recipe_deck_light_table`,
`field_setup._recipe_deck_light`): from its field's fbm, the ground's mean
shadow under the stratocumulus was 0.2 off at a quarter cover and a low
sun.

**An empty genus is not marched.** `GeneraSpec.genera` lists the genera a
planet may have; the march takes those holding cloud somewhere
(`genera.genus_specs(covered=True)`, `has_coverage`: the primary genus
always, another where a feature is its). The default's stratocumulus,
nimbostratus, altostratus and altocumulus have no feature, so the demo marches the
convective clouds and the stratus alone. An empty stratocumulus sharing the
stratus's band cost the demo 0.65 ms at 2560 x 1440: a band of two genera
takes the generic genus loop, not the band's own (performance §37).
`planet.check_layers` still counts them all.

#### Not yet, for the recipe genera

- **Altostratus striations.** The stretch is there (five times along east)
  but at its density (0.0015) no streak shows: the sheet reads uniform.
- **Nimbostratus virga and pannus.** The virga tail under the base needs a
  second, lower slab (not cheap); pannus, a stratus under it, would be a
  feature of its own.
- **Altocumulus species.** Lenticularis, castellanus, undulatus (rolls:
  they need a stretch, which saws the erosion's cells) and virga. From
  below its patches can read polygonal and flat-white.
- **No tiling breaker.** The Worley volume repeats every 8 cells (2.8 km
  for the altocumulus, 12 km for the stratocumulus); nothing hides the
  period, which may show over a cube face's centre.
- **The terrain's per-pixel shadow march** (and the models', and the sun
  march's local steps) leave the recipe genera out (above, "Where the
  recipe is evaluated"): they shade the ground only through the light
  volume and the shadow maps.

## The high layer

The high genera (cirrus, cirrostratus, cirrocumulus: `genera.HIGH_CHANNELS`,
`GeneraSpec.high`) are thin layers: a cirrostratus's optical depth is
0.03-0.3. They are not genera of the union march. They have **a sky band of
their own**, `GeneraSpec.high_slab` (`genera.HIGH_SLAB`, 11-12.5 km), which
the march crosses in **one sample**, not in steps (a "thin-layer
path", as cloud_genera_research.md §4 recommends for Ci, Cs and Cc).

**Its slab** is 11-12.5 km, above the convective clouds' top (10.5 km) with a
gap of 500 m, so `march.sky_bands` never merges it with their band, and an
anvil's top and the high layer never share a sample. Real cirrus spans about
6-12 km at mid-latitudes; here it must clear the anvils. Inside the slab one
band could give each genus a height of its own later.
`planet.check_layers` refuses a high slab whose base is not above the union's
top, or whose top is not inside the air (with the weather's cirrus, without
genera too: `check_layers(..., cirrus=True)`). `PlanetClouds.top_radius`
counts the high slab while it is drawn.

Rejected:

- **High genera as genera of the union march.** In a band shared with
  others, each genus costs 0.35-0.4 ms in the genus loop (performance §37),
  and a band of their own would still march it in steps (the step budget,
  the light volume's 32 slices, the bakes' samples), where one or two
  samples hold a layer 1.5 km thick.
- **Inside the convective band** (cirrus at 8-10 km, as the research's
  table has it): the anvils and the cirrus would share the genus loop.

**The crossings** (`highCrossing` in `cloud_march.glsl`; the CPU mirrors are
`march.band_segments`' `high`, `slab_segments` and `thin_optical_depth`):

- The high slab gives up to two segments of the ray, like a band's. They are
  **not sorted among the bands' segments**: a ray's height along it is
  convex and the slab is above every band, so its crossing on the way down
  comes before every band's segment and its crossing on the way up after
  every one (`band_segments`' order, its test). The march composites the
  first before its segment loop and the second after it. An anvil in front
  of the high layer hides it (from the ground). The high layer in front of
  a tower veils it (from above the slab). Sorted among them (two more
  segments, the crossing inlined in the segments' loop) the high variant
  cost the low clouds 0.1-0.2 ms more, with the crossing's own work about
  nothing (performance §37, "Task 23").
- Everything is read once a crossing, at its **sheet point**: where the ray
  crosses the slab's middle (`sheetCrossing`, `march.sheet_crossing`).
  A crossing that never meets the middle (a camera inside the slab,
  looking away from it) draws nothing: read at the crossing's middle
  instead, another height for every ray, it drew a second, offset copy of
  the cirrus seen from inside the slab. The coverage (cube B's channels
  1-3, cirrus, cirrostratus and cirrocumulus, in one fetch) and every
  genus's pattern, as a thin sheet's: jittered samples through the slab
  would wander a slant ray over its 1.5 km, many grains of 200 m or fibres
  of 180 m, and the visits would blur them (and shimmer in motion).
- Each genus has optics of its own (`genera.HIGH_OPTICS`): a full veil's
  optical depth straight through the slab (`uHighExtinction`, per metre)
  and what a crossing's saturates toward (`uHighCap`). The genera's depths
  over the crossing's path add. The cap is theirs, weighted by their
  shares (`march.high_optical_depth`). The path is at most
  `THIN_GRAZING_PATH` (4) slab depths, and the depth saturates softly, as
  the sheet's does. Otherwise a veil turns into a flat plate toward the
  horizon.
- The high segments take none of the step budget, and the march stops at
  them, as at any step, once the ray is opaque.
- A feature finer than a pixel's footprint fades to its mean (`resolved()`,
  as the low clouds' octaves do). On the sheet, the footprint is the view
  pixel's stretched by the slant, over the cosine of the ray's angle to the
  vertical (at least `HIGH_MIN_INCIDENCE`, 0.05: `march.slab_footprint`).
  At a grazing angle a pixel spans many grains along the view, though few
  across it. The step's fade does not apply: the samples are not a comb of
  steps.

**The cirrostratus** (`cover.g`; research §3: a transparent whitish veil,
the sun's outline plain through it, its 22° halo):

- density `cover × swell × fibres`, each about 1. The swells are value
  noise with features 25 km across the upper jet and 6 times longer along
  it (`CS_FEATURE`, `CS_STRETCH`: the research's 50-200 km, stretched 5-10
  times), moving the density by about a fifth. The fibres are 1.5 km
  across and 12 times longer (`CS_FIBRE`), fainter still. Both fade out
  where the footprint does not resolve them.
- Its full veil's optical depth is 0.2 straight up (the research's 0.03-0.3
  for the thin layers), its cap 1.2: about 13 % of the light, so the sun
  stays plain through it, and up to about 0.6 at a grazing angle, milky
  toward the horizon.
- **The 22° halo** is a term of its phase (`haloRing` in
  `cloud_shading.glsl`, by the cirrostratus's share of the crossing's
  depth, `HighOptics.halo` 1): a ring 22° from the sun, sharp inside
  (light refracts through the crystals' 60° prisms by no less), fading out
  over a few degrees outside. It costs one `acos` a crossing.
- **The jet** (`uHighJet`, `genera.high_jet`) is the upper wind at the
  genera's focus (`GeneraSpec.wind`, else `upper_wind_at`), one axis for the
  whole layer, as the recipe genera's shared axis. Far from the focus
  (thousands of kilometres) the stretch drifts off the local wind. The
  cirrus follows the wind's own form on the GPU (below); the cirrostratus
  and the cirrocumulus keep the one axis (their stretch, `alongJet`, and
  the cirrocumulus's ripple phase, `dot(q, jet)`, are linear in the
  position: with an axis that turns over the planet, `|q|` of 600 km times
  its turn would shear them).

**The cirrocumulus** (`cover.b`; the WMO: small white flakes, grains or
ripples without shading, regularly arranged, each under 1° wide):

- grains of the Worley cells (`cloudCells`) 200 m across (`CC_CELL`: 1°
  from 11 km below), `smoothstep(CC_GRAIN, 1 - F1)`. These are round
  flakes with sky between, their sizes varied by the slice through the
  3D cells.
- In rows: the grains' density is moved by a sine whose crests lie across
  the jet, 600 m apart (`CC_RIPPLE`, undulatus), bent and thinned by a
  slow noise 4 km across (`CC_FILL`).
- A grain is 1.2 optical depth at its core. Lit as a thin slab it has no
  self-shading: white flakes, not altocumulus's shaded lumps (5 times
  larger, at 350 m, and 3.5-3.8 km up: about 6° each).
- **The fade:** where a pixel's footprint resolves fewer than 2-4 of a
  cell, the grains are their mean (`uHighShape.x`,
  `march.cirrocumulus_grain_mean`: about a fifth), and the ripple and the
  fill their 1. A far patch is then a faint veil of the same optical depth
  (`march.cirrocumulus_grains`, its test).

#### The cirrus

The cirrus (`cover.r`; `cirrusStreaks` in `cloud_march.glsl`, its CPU
mirror `march.cirrus_streaks`) replaced the old cirrus sheet's baked
stamps, whose square patches showed (the sheet is gone: Task 24). WMO:
detached white filaments, patches or bands,
fibrous or silky, often with hooks (uncinus) or tufts (floccus), fall
streaks slanted by the shear. The research's recipe (§2, "Cirrus in
practice"): noise warped along the flow plus curl, stretched 8-20 times
along it, tails from taps along the flow, heads as tufts. The earlier
per-pixel cirrus (noise across the stream function's contours) gave
regular parallel bands; each part below breaks them.

**The upper wind on the GPU** (`highWind`, `genera.high_wind_cube`): an
RGBA32F cube 128 texels a face (7.4 km on a 600 km planet), stored as a
2D texture array of its six faces and blended in float by the shader
(`highWindAt`, as `weather.sample_cube`), not as a cube map: a cube map's
hardware filter weighs its texels in 8 bits, which stepped the stream
function by 1/256 of a texel (about 30 m) and drew the fibres as stairs
up close. It holds the
wind (planet frame, the jets' fastest 1, `genera.upper_wind_field`, which
is `upper_wind_at` for many directions at once) and its **stream
function** (`genera.upper_stream`, metres: the weather's
`stream_function`, or a uniform wind's with `GeneraSpec.wind` set). It is
built in the pool with coverage cube B. 32 bits: the stream function
spans hundreds of kilometres and the fibres need it to metres.

**The flow's frame** (`cirrusFlow`, `march.cirrus_flow`): the noise is
read at `(q + δ) / length + axis × ψ / width`. The stream function ψ is the
across-the-wind coordinate. It is constant along a streamline, so the
features follow the wind as it turns, around the lows too, with no seam
and no local frame: the streaks curve with the jets. The position gives
the along coordinate (and a little of the across). A width W across and
length L along make features L/W times longer along the wind where it
blows at the jets' speed; slower wind widens them (ψ's gradient is the
speed). `axis` (`CI_PSI_AXIS`) is off the noise's lattice, so the
pattern never repeats across the wind. A shift δ moves ψ by
`-speed × across · δ` (ψ's gradient), so the taps and the warp need no
other fetch.

**The domain warp**: the curl noise (`uCurlNoise`) moves the frame by up
to 1.5 km with features 12 km across (`CI_WARP`, `CI_CURL`): the streaks
meander and their spacing varies. A second, fine warp (180 m, features
2.5 km: `CI_WARP_FINE`) bends the fibres, strands and tufts: they wiggle,
merge and split. Without it the strands were straight, evenly spaced
lines: the old regular bands.

**The parts** (`march.CI_*`, each its width across the wind and length
along it, and its smoothstep window on the value noise):

- **Patches** 9 x 30 km (`CI_PATCH`), moved down by 0.55 × the coverage
  (`CI_PATCH_COVER`): their share of the sheet is about the coverage (0.2
  gives 0.20, 0.5 gives 0.52, 0.8 gives 0.84), the meaning of the
  weather's cirrus cover map. The coverage thins the
  patches. It does not scale the density, which only fades in over its
  first 0.1 (`CI_COVER_EDGE`): a sky of 0.2 cover holds fewer streaks,
  not fainter ones.
- **Fibres** 180 m x 6 km (`CI_FIBRE`, 33 times longer: the research's
  8-20 and more), in **bundles** 1.2 x 4.5 km (`CI_CLUMP`) that taper out
  along the wind.
- **Strands** 70 m x 900 m (`CI_SHEEN`, `CI_STRAND_WINDOW`): thin lines,
  the fibrous texture of the fibres and the tails (`CI_STRAND_FLOOR` 0.15
  of their density between the strands).
- **Tufts** 400 x 650 m (`CI_TUFT`), sparse: where the heads are
  (uncinus hooks, floccus). **Tails**: the tufts at 5 taps 380 m apart
  downwind, each `35 k²` m across the wind at tap k (`CI_TAIL_STEP`,
  `CI_HOOK`: the fall streak's shear bends the tail into a hook),
  weighted `(1 - k/6)^1.3`, averaged (a line integral: a smooth tail, not
  the head's copies, which the maximum of the taps showed), times the
  strands. The tufts lead and the tails lag upwind, as the old sheet's did.
- **Heads** (`CI_HEAD_*`, `CI_FAN`; reworked after Task 23, whose heads,
  the tuft noise through a narrow window at density 1, read as solid
  rounded blobs and, from overhead and 14 km up, as smeared chevrons). The
  WMO's heads, and the user's reference photo, are fibrous themselves: the
  fibres open up at the head. So the head is no separate lobe:
  - **The fan.** The taps fan out toward the head. Each fibre shifts them
    across the wind by its share of 500 m (the fibre noise about its mean,
    signed), most at tap 0 and none at the tail's end. The fibres meet the
    head at different places and splay out there like a broom, and run
    together downstream. This is a shape at the fibres' and the tufts'
    scale (180-650 m), so a pixel still resolves it from 14 km.
  - **The head's density**: a wide window (0.60-1.00) on the tuft noise at
    tap 0, so its edge falls off over the whole tuft, with no outline. Its
    outline is frayed along the fibres (their noise moves it, 0.25). It is
    striated by the strands (0.2 between them) and at most 1.2: about 1.8
    optical depth at a strand's core, translucent around it.
  - The fan takes the fibre noise (180 m), not the strands' (70 m): with
    the strands' the taps of neighbouring pixels scattered in the texture
    cache, +0.4 ms at `genus_cirrus_overhead`. It adds no fetch.
  - **Rejected, by the shots**: a splay of the strands by the head's mask
    (300 m folded them into loops and eyelets, 90 m did nothing); fine curly
    wisps for the head's texture (90 x 220 m read as cottony puffs, ridged
    70 x 500 m as marbled rings and crescents, 120 x 700 m as crescents).
    The mask's outline showed through every texture. Far away the fine
    noise is faded anyway (below), and only the coarse shape remains.
- The density: `patches × min(head + 1.6 tail × strands + fibres ×
  strands, 1.3)`: mostly sky, the heads and the fibres 0.2-1.

**Fading** (`resolved()`, 2-4 footprints a feature): each part fades to
its mean over the sheet (`march.cirrus_means`: `uHighCirrus`, and the
tufts' `uHighShape.w`), so a far patch keeps its optical depth, and the
tails' taps are skipped where the tufts are not resolved. The fibres, the
strands and the patches fade by the footprint **across and along the wind
apart** (`footprintAlong`, `march.footprint_along`): the slant stretches
a pixel's footprint along the view, not across it, so fibres running
across a grazing view stay sharp toward the horizon, and fibres running
along it fade before they alias. The tufts (nearly round) fade by the
slant's footprint. Without this the far cirrus was a faint uniform haze
where the old sheet showed streaks.

**Optics** (`HIGH_OPTICS["cirrus"]`): a full density 1.5 optical depth
straight through, so a head's strands up to about 1.8 and the fibres 0.3-1 (the research's
0.03-3, SIRTA's typical 0.1-1), a cap of 2.5, and a faint 22° halo
(0.4, the cirrostratus's 1): it shows only where a streak is, "fragments
of a halo".

**The showcase**: a cirrus field 80 km wide 90 km north
(`genera.SHOWCASE_HIGH`), beyond the cumulus row (the cirrostratus's veil
reaches over the start, the cirrocumulus and the anvil fill the east).
Its poses stand under its near edge, 55 km north
(`SHOWCASE_HIGH_EYE`), the row behind: `genus_cirrus` (40° up),
`genus_cirrus_overhead` (under its centre, straight up: its pattern
top-down), `genus_cirrus_grazing` (10° up).

**The weather's cirrus** (`--cirrus`, on by default; `--no-cirrus`): the
planet's cirrus, from `weather.cirrus_cover_map` and the upper wind, is
the march's (`PlanetClouds(cirrus=True)`, `CloudLook(cirrus=True)`). With
the genera it sets `GeneraSpec.cirrus_weather`: the cover map fills cube
B's channel 1, under the features, and compiles the high layer. Without
genera (one cloud type: the demo without `--genera`, `demo_world.py`,
`cloud_bench.py`) the march takes a high layer of the weather's cirrus
alone (`genera.weather_cirrus`: a `GeneraSpec` with no genus and no
feature, only its high layer read), over its one type, on the clouds' own
weather. With `--showcase` the default is off: the showcase has a cirrus
feature of its own, which the weather's would bury.

The **C key** shows or hides the march's cirrus, the weather's and the
features' (`PlanetClouds.set_genus_visible("cirrus", ...)`: the high
genera have bits of `uGenusMask` past the low genera's,
`march.HIGH_MASK_SHIFT`, and `highCrossing` drops a hidden one's
coverage); the HUD's "march cirrus on/off". It replaced the old cirrus
sheet (baked stamps on a sphere at 11-12 km), deleted with
its atlas, its bake and preview scripts and its shaders in Task 24: the
march's draws the same cover, in less time (performance §37, "Task
24").

#### The light, the shadows, the cost

**The light** is a thin slab's (`thinSlabRadiance` in
`cloud_shading.glsl`, moved there from the old cirrus sheet's shader):

- no sun march, the sunward half of the slab's own depth instead;
- the cirrus optics' phase (ice, strongly forward);
- the sky, plus the sun's light scattered about as fill;
- the twilight of the lit air above once the slab is in the planet's shadow.

The sun's reddening and the twilight come from `planetCloudLight` at the
crossing's middle, at the high layer's own height. So it stays lit, and
reddens, after the cumulus below has gone grey. Like the low clouds, it
takes the eclipses, the mountains' and the rings' shadows on its sunlight
(`cloudSunAndSky`, below).

**Shadows: none.** The high layer casts no shadow, into the light volume or
the cloud shadow maps, so it does not darken the ground or the clouds below.
The old cirrus sheet cast none either. At 0.03-0.3 a full veil would take
2-19 % of the sun. Through the light volume it would cost a band of 32
slices for that. If a thick cirrostratus deck should dim the ground, its depth fits the
shadow maps' chord alone. That would be a term in the bakes, not in the
volume.

**What costs nothing.** `genera.high_genera(spec, covered=True)` lists the
high genera a feature covers (and the cirrus with the weather's). With
none (`--no-cirrus` and no high feature) the march compiles no high layer
at all: it keeps `cloud_march.frag`
(`march.march_fragment`, as for the recipes). With one, it compiles
`cloud_march_high.frag` or `cloud_march_recipes_high.frag` (`CLOUD_HIGH`)
and builds coverage cube B (and the upper wind's cube) even if no other
genus reads it. At 2560 x 1440 (performance §37, "Task 23"): a
cirrostratus filling the view costs about 0.3 ms, the showcase's
cirrocumulus patch 0.2-0.25 ms, the cirrus field filling the view 0.3-0.35
ms; the weather's cirrus by the march 0.2 ms, against the old sheet's
0.35-0.55 ms (Task 24's comparison: performance §37). (Tasks 21-22 measured about 1.0 and 0.8 ms in a session
that drifted by 1 ms; the crossings sorted among the segments cost
0.1-0.2 ms of the difference.)

With `--genera`, the `high_cb`, `high_overhead`, `high_grazing` and
`high_above` poses show the weather's cirrus toward the cumulonimbus:
behind the anvil, overhead, at a grazing angle and from 14 km up through
it.

#### Not yet, for the high layer

- **The cirrus in the sea's reflection**: the reflection draws the low and
  middle genera's decks alone, so the weather's cirrus, which the old sheet
  drew there, is not reflected.
- **The cirrus's species as choices** (fibratus, uncinus, spissatus,
  castellanus, floccus): one recipe mixes the fibres and the hooked tufts;
  no spissatus (dense, greyish) or castellanus (turrets).
- **The heads' turbulent texture**: the reference photo's dense, curly
  floccus wisps at the head (every texture tried showed the mask's
  outline, or made puffs, rings or crescents). Far away the heads are soft
  lobes: their fine noise fades to its mean before the mask's window, so
  only the coarse shape remains (a general fix, widening a window by the
  faded terms' spread, is to be discussed).
- **The fall streaks' parallax**: the hook is in the sheet (read at its
  middle); seen from the side no streak hangs below the heads.
- **The cirrostratus and the cirrocumulus on the GPU wind** (one jet axis:
  their patterns are linear in the position).
- **A height of each genus's own** inside the slab (the cirrocumulus
  lowest, the cirrus highest): all share its middle.
- **The cirrocumulus's species** (stratiformis, lenticularis,
  castellanus, floccus) and the cirrostratus's other halo arcs (the
  parhelia, the 46° halo).
- **Its own toggle in the showcase** (keys 1-9 hide the march's genera, not
  the high layer).
- **The sea's reflection** has no high layer (the decks are the low and
  middle genera's only).

## The raymarch

The raymarch draws the low clouds at every quality preset; the presets set
its resolution and steps (`march_block`: one pixel in 16 traced a frame at LOW
and MID, one in 8 at HIGH, one in 4 at ULTRA; `march_steps` 24, 40, 64, 96).

**The sea's reflection** sees the decks (`deck.deck_nodes`): one sphere a
genus at its slab's middle, shaded by `cloud.frag` as that genus
(`uDeckGenus`): genus 0 its own deck entry, every other genus its far level,
its weather (`genusWeatherAt`), its density (a recipe genus's
`genusRecipeDensity`) and its far table. A hidden genus (`uGenusMask`) draws
none. They are drawn highest first, since the reflection sees them from
below. Before, the deck was genus 0's alone, so only the cumulus family was
reflected: no stratus, stratocumulus or nimbostratus. The high layer has no
deck. The decks give no 3D cloud shapes; that is good enough under the
ripples. If shapes are wanted, a cheap mirror march would add a third march
layer from the reflection camera into the reflection's buffer, at a quarter
of its resolution, about 24 steps, with no history and no composite, using
the reflection's own depth as the scene's depth. It would run only while the
sea is in view and `reflected` is on, for about 0.3-0.6 ms (a guess).

**Eclipses, mountains and rings** (`cloudSunAndSky`, `cloud_march.glsl`):
the light reaching a cloud is the air's (`planetCloudLight`: its reddening,
twilight, the sky's fade), times what else hides the sun there, as every
other receiver takes it (shadows.glsl):
- other bodies (`eclipseVisibility`), which dim the sky's light too, down to
  `ECLIPSE_SKY_FLOOR`;
- the planet's mountains (`terrainShadow`);
- its rings (`ringVisibility`).

It is evaluated once a pixel, at the first lit sample, where the air's light
was already taken; the high layer's crossing takes it too. The world binds
the inputs to every pass (`World`: `Shadows.bind` and `ShadowMap.bind`
through `PlanetClouds.bind_maps`). Before, the march took none of them, since
the billboards' `cloud.frag` was the only cloud shader that did: in an
eclipse the air darkened around sunlit clouds. Its cost is a few shadow-map
taps a pixel.

[`march.py`](../../src/world_builder/clouds/march.py) (`CloudMarch`) draws
the field, its weather and its lighting, and the deck shares them. The
billboards it replaced were a raymarch whose steps were quads: every one of
them was drawn, shaded and blended, so their cost grew with the cloud's
volume on screen (15 ms inside the slab on a laptop's GPU, 41 ms in a dense
cloud). Here each pixel steps along its own ray:

- **Through the slab**, between the spheres at its base and top, less the
  planet: up to two segments (a ray skimming under the slab leaves it and
  comes back), from the near plane to the **scene's depth** (the near
  range's surface where it has one, else the far range's). With genera, the
  slab is each **sky band** (`uSkyBands`, the genera's slabs merged): up to
  two segments a band, eight in all, sorted along the ray (insertion sort),
  and inside a segment a **genus loop** per sample over the band's genera
  (bit mask), each adding its own density and light ([Genera](#genera)).
  Rejected: segments per genus (overlapping slabs would cross the same
  stretch twice, and the steps of two segments could not be one comb), and
  a march pass per genus (they would composite in the wrong order where a
  stratus lies in front of a tower and behind another).
  A game's model inside a cloud is hidden by just the cloud in front of it,
  and the camera inside a cloud sees it all around (the billboards faded out
  near the camera, which left a clear bubble).
- **In steps** of `MIN_STEP_FEATURE` of the finest feature near the camera
  (50 m for cumulus), `STEP_GROWTH` of the distance farther, all lengthened
  alike when a ray needs more than `CloudSettings.march_steps`, and at
  least `SEGMENT_SAMPLES` (8) a stretch of the slab: a far ray crossing it
  steeply in a few steps longer than its clouds' edges are thick hits or
  misses thin cloud by chance (the sparkle seen from 20 km up). A step's
  length is its distance's alone: it no longer lengthens as the ray loses
  its light in a cloud. That made the steps' grid depend on what the samples
  found, and the visits' jitter no longer averaged out the contours where
  the grid met a cloud's surface (bands across a tower's face, strongest
  behind a thin veil). Each sample
  is jittered within its step by a **blue noise** (`blue_noise`, a 64 x 64
  tileable void-and-cluster texture) read at the *march's* pixel, shifted by
  which pixel of its block this frame traces (R2) and moved on by the golden
  ratio at each **visit** of the pixel (`visit_of`), not each frame: traced
  once a block's cycle, a per-frame jitter skipped half a step a visit and
  never stratified it. (The interleaved gradient noise at the view pixel, a
  stride of `uBlock` apart, aliased its own diagonals into a lattice of
  stripes that showed in motion, worst at a quarter resolution.) The steps'
  **boundaries** move too: each stretch's first step is a random share of its
  length (another draw of the blue noise, moved on each visit), so neighbouring
  pixels do not share them; a long step's light, not linear in its density,
  carries a bias that depends on where a cloud's surface falls between two
  boundaries, which drew bands up a tower's face seen level. The march stops once the cloud is opaque. The steps' geometry
  has a NumPy mirror that the tests run (`slab_segments`, `march_steps`,
  `march_transmittance`).
- **No longer steps in clear air.** A step longer than the finest features
  (85-125 m) samples thin cloud by chance, and the jitter changes that
  chance every frame: thin cloud blinks, and edges come out ragged. Up to
  four-times-longer steps in clear air did exactly that. Nor can clear air
  be *proven* clear and skipped: the fbm sits within a narrow band, the
  threshold inside it, and its two fine octaves can each reach their
  maximum somewhere in any 500 m, so no bound on it falls under the
  threshold (the exact bound, from the texels each octave's filter
  touches, did in 0.004 % of the cells).
- **The field** with the octaves of its distance band
  (`CloudSettings.shell_octaves` over the `octave_bands` bands of
  `field_setup.octave_bands`, each twice as far as the last; a band's
  threshold moved to keep the coverage, `field_setup.octave_offset`), and where a view pixel spans
  more than half a fine octave's feature (the 3rd and 4th, 285 m and 125 m
  for cumulus) or the billows', that one fades to its mean (as an octave
  left out does: the coverage holds): it would alias, a different value at
  every jitter and every move. The view pixel alone decides this, not the
  step's length as well.
  - **The step fade, removed.** Features finer than a share of the step
    (`uStepFade`: 0.1 at a half resolution, 0.25 at a quarter) used to fade
    too, against the shimmer of far clouds when the steps grow with the
    distance. It cost far clouds their lumps: billows and the fine octaves
    were gone a few tens of kilometres out, and towers and cumulus read as
    smooth blobs.
  - **Why it could go.** With the resolve's history, far clouds shimmer
    *less* without it: `cloud_bench --jitter` at `start` gave error 6.15 →
    2.51 and flicker 3.28 → 2.32 against the full-resolution reference, and
    at `cb_high` 4.28 → 3.17 and 5.36 → 3.91. Still frames got slightly less
    steady (spread 0.82 → 0.99).
  - **Its cost**: 0.1-0.25 ms (performance.md §37).
  - **What was not enough**: widening the edge by the faded detail's spread
    (`edgeWidening`, still in) and keeping the far self-shadowing longer.

  The weather is read
  once per `uWeatherStretch` of ray (a kilometre without genera), each
  genus's its own, not at every sample.
  Two departures from the plain field keep the crisp look the billboards
  had (removed 2026-11-15; the march was tuned against them):
  - the view ray takes **twice the field's extinction**
    (`VIEW_EXTINCTION_SCALE`): a billboard carried its sample's density over
    a whole chord, which kept a cloud's edge crisp, and the field's own
    soft edge, integrated, reads as a blur;
  - **billows** carve the edges, as the billboards' lumpy sprites did: a
    billowy noise 24 times finer than the first octave moves the threshold
    by up to half its window about its mean (`DETAIL_EROSION`), so the core
    stays and the coverage holds.
    The billows are two layers of the noise itself (the second 2.3 times
    finer, turned about, half the weight: `BILLOW_FINER_*`), carving where it
    is high, with the ridged noise's mean and spread (`billow_gain`) so the
    coverage and the erosion's depth hold. The first version used the ridged
    noise `1 - |2n - 1|`, which carves along sheets: seen edge-on they showed
    as parallel streaks across every cloud. The first layer is also sampled
    through a rotation off the field's axes (`BILLOW_TURN`), which hides the
    value noise's lattice planes. Both layers are then **bent by a curl
    noise** (`noise.curl_noise_volume`, 32³, the curl of three value-noise
    potentials; `CURL_STRENGTH` 2 of the billows' noise units, features about
    two billows across), as EVE-Redux bends its detail noise: wisps and small
    fragments curl into tufts instead of round lumps. Curl noise has no
    divergence, so the billows' mean and spread (and the coverage) hold. At 4
    it stretched fragments into streaks.
- **Lit** per sample as in [lighting](#lighting) (`cloud_shading.glsl`,
  which the deck's `cloud.frag` includes too), with the sunward depth and the
  sky's reach coming from the **light volume** around the camera
  ([shadows](shadows.md#the-cloud-light-volume)) where it is whole, the
  cloud shadow maps' levels without it, the sun march beyond both. Past a
  view footprint of `march.LIGHT_LOCAL_FOOTPRINT` (0.04 of the clouds'
  feature) a sample takes its sunlight from the volume alone, without the sun
  march's two local steps. No air is drawn in front of the cloud
  ([atmosphere](atmosphere.md#clouds-at-the-horizon)); the sun's reddening
  is in its light already. The transmittance-weighted distance is written
  beside the colour, for the resolve and the composite's depth.
- **Far away**, where a pixel spans more than the features, it samples the
  deck's expected coverage as a homogeneous slab whose vertical crossing
  takes the deck's alpha, lit by the deck's averages. The march reaches
  orbit with no deck, and nothing changes colour closing in.

**The passes.** One march covers both depth ranges of `SplitDepthCameras`
(`CloudMarch.attach`; `PlanetClouds.use_split_depth`, which a `World` calls
from `use_split_depth`), as EVE-Redux marches once over KSP's two cameras'
depths:

1. a camera per range, sharing its lens, draws the **scene's depth** at half
   the view's resolution per axis (`march.DEPTH_SCALE`) into a texture (the
   march reads it at the pixels it traces; the cards, to tell the near range's
   pixels; static shots differ from full resolution in a few pixels a frame). Everything but the clouds and the sky
   is drawn, the terrain through its depth-only stage
   (`passes.pass_states`, `passes.pass_camera`; they were the
   half-resolution billboards' pass's, removed with them);
2. a fullscreen card **marches** one pixel of every block
   (`CloudSettings.march_block`: 2 × 2 at a half resolution, 4 × 2 for an
   eighth of the pixels, 3 × 3 for a ninth, 4 × 4 at a quarter), a different
   one each frame (`march.block_order`: consecutive frames never trace
   touching pixels, even across the blocks' edges; 4 × 4 is a Bayer order,
   4 × 2 and 3 × 3 were found by an exhaustive search, as EVE-Redux's own
   sequences for them trace touching pixels in turn), from the near range's near plane to the near range's
   surface where its depth pass drew one, else to the far range's (EVE's
   distance buffer of both cameras, read in the march itself). It stops once
   the cloud is opaque, so a cloud near the camera spares the march the ones
   behind it (two marches, one per range, could not: the far one did not know
   the near one's cloud). It writes the premultiplied cloud and its distance;
3. a **resolve** gathers them into the view's resolution over the frames.
   Every pixel blends this frame's samples around it (the four nearest,
   bilinearly) into the last frame's result, by the same share whatever
   its distance to a traced pixel (a weight that fell with it put a grid of
   the traced pixels on the cloud). The history is reprojected along the
   view ray to the pixel's own cloud distance (this frame's first, then the
   history's there; float64 for the camera's move, float32 only near the
   camera) and held within 1.25 standard deviations of the samples around.
   While the view is still (`uStill`, from the camera's move and
   turn), that range is widened by a floor and a share of the mean, or the
   steps' own noise would clamp the history and start its count over; in
   motion a quarter of it stays (`MOTION_FLOOR`): with none a cloud coming
   closer kept less of the thin look it had, but where the samples agree the
   range was their noisy mean, and a tower went grainy whenever the camera
   moved. In motion this frame's samples are taken as their smooth mean, and
   each frame takes at least about a cycle of the block (`MOVING_WEIGHTS`:
   1/8 at 4 × 2): with the full count's `MIN_WEIGHT` alone, ~33 frames of
   history lagged the view of a cloud whose look changes fast as the camera
   flies (a lumpy anvil's base at a grazing angle from inside the slab), and
   the clamp cut that lag into grain wherever its box of noisy samples
   happened to be narrow. Unreal's clouds refresh each pixel once a cycle for
   flying through them. `CloudSettings.march_refresh_moving` (off) traces
   twice the pixels while moving (`MOVING_BLOCKS`, back after `MOVING_HOLD`
   still frames): no less grain measured, each sample as noisy. The samples'
   mean and spread are taken over the 4 × 4 about the pixel, each weighted by
   a radial bump of its distance, (1 − (d / 2)²)², nothing from two samples
   away (`STATS_REACH`; `march.stats_window` on the CPU): a sample enters and
   leaves the window at no weight, so the statistics run on unbroken from one
   of the block's cells to the next. The 3 × 3 about the nearest sample (a
   Gaussian) jumped at each cell's edge, and where the clamp cut the history
   its box printed the jumps: a mosaic of the block's cells in motion
   (Karis' "3x3 box artifacts"; performance.md §30). The distance the history
   is moved by takes the same weights. A pixel that takes most of this
   frame's samples (its history lost or short) takes their smooth mean instead
   of the bilinear mix of four, by `uSmooth` (`march.SMOOTHING`: all of it at a
   quarter resolution, half at a half, none at full): each sample's noise was a
   dot of its own. Each pixel counts the samples it has gathered (the distance
   target's second channel): this frame's share falls as 1 / (count + 1)
   down to `MIN_WEIGHT` (0.03), and what the clamp takes away takes the
   count down (to 4 at least, so a noisy sample is not taken
   whole). The history's distance is held within the fresh samples' around
   (their least and most): mixed like the colour it lagged them by up to
   1 / `MIN_WEIGHT` frames, and flying forward (the clouds nearer every
   frame) it moved the history from too far, where the clamp threw it away
   (small clouds lost in a fast descent). The history is kept across a move of
   `HISTORY_JUMP` (2 km) or `HISTORY_JUMP_SHARE` (2 %) of the camera's distance
   to the slab, the more: from orbit a descent moves kilometres a frame. Clear
   sky (no cloud in the samples around, none in the history) returns early.
   Two results alternate, each frame's reading the other's;
4. two cards (`passes.composite_card`) lay the result over the view,
   premultiplied "over", at the
   **cloud's own depth**: the near range's card over the pixels where the
   near range's depth pass drew something (the near camera draws its
   surfaces over the far pass afterwards, and the clouds in front of them
   must come after), the far range's card over the rest, in the far pass's
   clouds' slot (after the sky, before the cirrus). Each range's depth test
   hides the cloud pixel by pixel at the view's resolution, behind anything
   nearer, which also drops any stale history a moving model would leave
   behind it. The depth is **no farther than the scene's** (each card's own
   range's depth pass, `uSceneDepth`): the march stopped there, so all it
   returned lies in front. The resolved distance alone lagged the fresh
   samples, and flying forward it put the clouds behind the ground under them:
   the depth test cut 90 % of a stratus and slid its edge against the
   mountains, and a fast descent lost half its clouds (performance.md §28;
   `CloudMarch` layers' `scene_cut` switches it off, to compare).

The near range's depth pass and card run only when the slab reaches into the
near range; otherwise the march stops at the far range's scene, and what the
near camera draws covers the clouds behind it. The history goes on across
that switch.

**Their shadow on the ground** comes from one more pass of the march's
(`_GroundShadow`, `ground_shadow.frag`): from the same two depth passes, at
their resolution, the clouds' shadow at each texel's scene point (all the
genera's: `planetCloudShadow`, as the terrain's per pixel), and the distance there.
The terrain and the sea read it back by their distance (`groundShadowAt`) and
shade it per pixel only where no texel agrees, and in the sea's reflection
(shadows.md, "The clouds' shadow on the view's ground"; performance.md §33).
The depth passes draw the terrain's coarse mesh (terrain.md, "LOD").

The sea's reflection sees the decks (one a genus; genus 0's far layer is packed for it).
The textures may be padded to powers of two, so the shaders address them
in pixels.

**What it costs** (performance §20, §23, §32, §35): measured against the
billboards while they lived, 1.3-2 times their cost over open sky on a weak
GPU, the same inside the slab, two-fifths in a dense cloud.

## Testing

The headless tests check the CPU side: the field and its calibration
(`tests/test_cloud_noise.py`, `test_clouds_field.py`), the march's geometry,
steps and resolve through their NumPy mirrors (`test_clouds_march.py`), the
genera's maps and shapes (`test_clouds_genera.py`, `test_clouds_congestus.py`,
`test_clouds_anvil.py`, measured through `density_on_sphere`), the light
volume and the shadows. They cannot check the pixels, nor even whether the
GLSL compiles (`Shader.load` succeeds with no graphics context). That needs
[`scripts/terrain_shots.py`](../../scripts/terrain_shots.py) (its `cirrus_*`,
`genera_*` and `in_clouds` poses among others) or an offscreen GL context.

[`scripts/cloud_bench.py`](../../scripts/cloud_bench.py) times and shoots
the clouds alone (the air, the sky, no terrain) from fixed poses, still or
moving (`--move`). Its `model_in_cloud` pose puts a black ball, standing in
for a game's model, half into a cloud, and `--interactive` flies it in and
out of the clouds. `--stability N` scores the clouds' steadiness over N
frames read back (each pixel's brightness spread, and the raymarch's block
grid: see the script), and each pose's frame comes with its 95th percentile
([performance §23](performance.md)). On WSL, run it with
`GALLIUM_DRIVER=d3d12`, or Mesa draws on the CPU.
