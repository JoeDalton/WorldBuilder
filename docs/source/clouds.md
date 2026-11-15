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
- optionally, a **sheet** above it
  ([`cirrus.py`](../../src/world_builder/clouds/cirrus.py), `CloudSheet`): high
  cirrus, baked patches stamped along the upper wind.

They share the weather, the sun, the settings and the camera, and the part
checks the one rule the pair must keep: the sheet lies above the volume's slab
and inside the air.

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

[`genera.py`](../../src/world_builder/clouds/genera.py) adds stratus, and the
convective continuum from fair-weather cumulus through congestus to
cumulonimbus (`PlanetClouds(genera=GeneraSpec(...))`, `CloudLook.genera`,
`--genera` in the demo and the scripts sharing its options, on by default
there, `--no-genera` for the plain field; opt-in in `cloud_bench.py`). Its maps are
a **placeholder** for the weather (issue #8): features laid out around one
point. One `CloudMarch` draws them all, as one union of genera.

**The genera and their sky bands.** A genus is one field (one cloud type)
over its own slab, with its own coverage, shift table and far table
(`GeneraSpec.genera`, each a `genera.GENUS_PRESETS` entry: its cloud type,
coverage source, `far_shadow` and slab). Today there are two:

- **convective** (`CloudType.CONVECTIVE`), the primary genus (genus 0): the
  cumulus field, its clouds growing with the convective **depth** (0 to 1, a
  map of its own);
- **stratus** (`CloudType.STRATUS`): a sheet 500 m thick, its base at
  `GeneraSpec.stratus_altitude`.

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
the genus's). A band of one genus (every band today) takes its samples in a
loop of its own, without the genus loop: through it the same light cost
0.4-0.6 ms more between the towers (`cloud_bench.py --genera`, 2560 × 1440),
its lighting inside a second loop; a single (sample, genus) loop, and slots
of weather per genus, cost 0.2-0.4 ms more too. Every genus has all the
`octave_bands` asked for (with two marches, the convective clouds' and the
stratus's, each had used to halve them to fit eight layers). The reflection's deck is genus 0's
alone.

**The weather of the other genera.** Genus 0 reads the weather cube
(coverage, shift, depth, spread), so a planet without genera is one genus
and unchanged. The others read their coverage from a channel of **coverage
cube A** (`genusCoverA`, RGBA16F, `genera.coverage_cube`: the stratus in
channel 0), and turn it into their fbm shift through their row of the
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
stratus, which so takes the whole chord of the towers above it. Under the
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

**Not yet.** The anvil does not lean downwind with height (a shear; it is
the genus framework's, docs/superpowers/plans/2026-11-15-cloud-genera.md).
The plume is straight: the wind is read at the tower, not
along the plume. Its underside is lit as the rest of the cloud: no darker base
for deep clouds. The tower's stem, at coverage 1, has
straight sides. From the ground, the air hazes out a cumulonimbus's lower half at
40 km, and its top half reads as a large cumulus.

## The raymarch

The raymarch draws the low clouds at every quality preset; the presets set
its resolution and steps (`march_block`: one pixel in 16 traced a frame at LOW
and MID, one in 8 at HIGH, one in 4 at ULTRA; `march_steps` 24, 40, 64, 96).

**The sea's reflection** sees only the deck (`deck.deck_mesh`, the far
field): no cloud shapes and, with genera, no stratus. It is
good enough under the ripples, so it is left as is. If it is wanted: a cheap
mirror march, a third march layer from the reflection camera into the
reflection's buffer (a quarter of its resolution, about 24 steps, no history,
no composite; the reflection's own depth as the scene's depth), active only
while the sea is in view and `reflected` is on; about 0.3-0.6 ms (a guess).

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
  every jitter and every move. The same goes for features finer than the
  **step** (`uStepFade`, a share of its length, over the same ramp: 0 at full
  resolution, 0.1 at a half, 0.25 at a quarter, `march.STEP_FADES`: more is steadier
  and softer): beyond a few
  kilometres the steps grow with the distance (1 %), and an octave they cannot
  sample aliases into the jitter however finely a pixel resolves it (the
  shimmer of far clouds, which more steps did not remove). The step's length
  there is its distance's alone (`k * max(dtMin, t * growth)`), not the step
  taken, which is cut at a stretch's end: the octaves a sample kept then
  depended on where its ray
  entered the cloud, contours up a tower's face. The weather is read
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

The sea's reflection sees the deck alone (the far field's layer is packed for it).
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
`genera_*` and `in_clouds` poses among others) or an offscreen GL context;
the cirrus's shapes can be previewed without one
([`scripts/cirrus_preview.py`](../../scripts/cirrus_preview.py)).

[`scripts/cloud_bench.py`](../../scripts/cloud_bench.py) times and shoots
the clouds alone (the air, the sky, no terrain) from fixed poses, still or
moving (`--move`). Its `model_in_cloud` pose puts a black ball, standing in
for a game's model, half into a cloud, and `--interactive` flies it in and
out of the clouds. `--stability N` scores the clouds' steadiness over N
frames read back (each pixel's brightness spread, and the raymarch's block
grid: see the script), and each pose's frame comes with its 95th percentile
([performance §23](performance.md)). On WSL, run it with
`GALLIUM_DRIVER=d3d12`, or Mesa draws on the CPU.
