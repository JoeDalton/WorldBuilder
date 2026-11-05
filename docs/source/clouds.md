# Clouds

[`world_builder.clouds`](https://github.com/JoeDalton/WorldBuilder/tree/main/src/world_builder/clouds/)
draws a planet's clouds. It is built on two ideas, and almost every design
decision in it follows from one of them.

**A billboard is a sample of a volume, not a sprite.** Its geometry says only
where in the volume it sits and how many metres of cloud lie behind each of its
pixels; outline, opacity and colour are continuous functions of world position,
evaluated per fragment. Nothing about appearance may be carried per particle:
anything that is shows up as quad-shaped patches once the field gives each
billboard a crisp footprint.

**A cloud is not an object.** A cloud *type* is one density field spanning a
slab of sky, and the clouds you see are its features. That is what lets clouds
shadow each other, and why their silhouettes are exactly as crisp as the field.

## The parts

A planet's clouds are one part,
[`PlanetClouds`](../../src/world_builder/clouds/planet.py), drawn as two layers
by two techniques:

- a **volume** ([`sphere.py`](../../src/world_builder/clouds/sphere.py),
  `CloudVolume`): the low clouds, cumulus by default (any `CloudType`),
  billboard LOD shells generated per cube-sphere tile in the terrain's worker
  pool, and a deck sphere beyond the last shell, up to orbit. By default
  (`CloudSettings.renderer = "march"`, at every preset) the same field is
  **raymarched** instead
  ([`march.py`](../../src/world_builder/clouds/march.py), `CloudMarch`, see
  [the raymarch](#the-raymarch)); `renderer = "billboards"` selects the
  billboards (a rebuild; they stay the cheaper choice over a distant open
  sky on a weak GPU);
- optionally, a **sheet** above it
  ([`cirrus.py`](../../src/world_builder/clouds/cirrus.py), `CloudSheet`): high
  cirrus, baked patches stamped along the upper wind.

They share the weather, the sun, the settings and the camera, and the part
checks the one rule the pair must keep: the sheet lies above the volume's slab
and inside the air.

[`weather.py`](../../src/world_builder/clouds/weather.py) builds the weather
they follow: a cube map of target coverage (`WeatherSpec`), turned into the fbm
shift that gives it exactly (`shift_table`); the upper wind; and the cirrus's
cover. The planet-wide design (the shells, the deck, the weather, the
terminator, the haze, the cirrus) is described in
[Atmosphere, sky and planet clouds](atmosphere.md#planet-clouds). This page
covers the field and the billboards that sample it.

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
  sheet with holes (`CUMULUS` at 0.9 is genuinely stratus-like). It is also
  the most expensive knob: billboards fill a *volume* while coverage measures
  a *projection*, and a covered column is also cloudy over a taller stretch of
  its slab, so doubling coverage roughly triples the billboard count.
- **`edge_softness`** is the window's width, in standard deviations of the
  field's own fbm (`fbm_sigma`), so it means the same for every type and
  survives a change of octave weights. A narrow window is the entire source of
  the crisp cauliflower edge.
- **`value_noise_volume`** and **`build_noise_texture`** generate the single
  tileable octave the shader taps four times, as a `sampler3D`. It is not a
  baked fbm: the octaves are sampled at 1/2/7/16 with independent offsets,
  baking would freeze their relative phase, and resolving the 16x octave
  would need 1024³.
- **`density_on_sphere`** evaluates the field on the CPU *exactly* as the
  fragment shader does: the same trilinear half-texel addressing, the same
  8-bit quantisation. That equality is load-bearing, because placement is
  driven by the very field that is drawn. The shader side is one include,
  `shaders/cloud_field.glsl`. Its `cloudDensity` stops after one or two
  octaves where the rest (at most their weights) cannot reach the threshold:
  exact, and most of a billboard quad is clear air.
- Types share one volume, decorrelated by a constant noise-space offset
  (`field_offset`): an independent field with identical statistics, for the
  cost of an add.
- **Per-type fields on the GPU** ride in a `layerParams` uniform array indexed
  by a flat layer id (`LAYER_VEC4S`, twelve `vec4`s each, for up to
  `MAX_LAYERS`, eight, layers: the shells and the deck). They are packed by
  `noise.pack_layer_params` and read by `getField` in `cloud_field.glsl`. A
  uniform array rather than a texture: a fragment needs about two dozen of
  its type's parameters, which as `texelFetch`es would be two dozen texture
  reads per pixel, and as varyings would burn most of the interpolator budget.

## The types and their cost

[`cloud.py`](../../src/world_builder/clouds/cloud.py) holds the types and
what they cost:

- **`PRESETS`** maps `CloudType` (`CUMULUS`/`STRATUS`/`CIRRUS`/`CUMULONIMBUS`,
  and `CONVECTIVE`, [the genera](#genera)'s) to a `CloudSpec`: a field, its
  optics, a billboard radius range and a billboard aspect. Every shape
  parameter lives in the field.
- **Placement** (`sphere.generate_cloud_tile`) rejection-samples the slab and
  keeps the points the field calls cloud, so the field alone bounds each
  cloud. The accept rate measures the cloud's volume directly, which makes the
  extinction calibration exact rather than tuned: extinction is the field's
  density divided by the billboards' overlap (sphere volume per unit of cloud
  volume) and the sprites' mean alpha. The billboard budget is then a
  performance decision, not a look change.
- **Quality** (`CloudSettings`, [Quality settings](quality.md); the levels are
  the shared `QualityLevel`) scales a type's cost, by preset across
  LOW/MID/HIGH/ULTRA or knob by knob (`count_scale`, `march_scale`, through
  `scale_count` and `scale_march`; `with_settings` applies them). HIGH is the
  identity: what the presets ship. Two knobs are scaled differently on
  purpose:
  - **The billboard count** takes the full ×0.25/×0.5/×1/×2 spread, because
    that is where the frame time is. It changes through the **radius**, not
    the overlap: at a fixed overlap the count falls as the cube of the
    radius, the trade each LOD shell makes. Cutting the overlap instead holds
    optical depth only where billboards still reach; at a cloud's fringe it
    removes the silhouette rather than thinning it (measured at LOW: 18% of
    the cloud's screen area lost, against 8.8% via the radius, with more grain
    and no saving). Bigger billboards cost little because density is
    evaluated per fragment, so lateral detail and the silhouette do not depend
    on quad size; what coarsens is the **chord**, the resolution along the
    view ray.
  - **The sun march** is stepped far more gently (×2/3 a level), because it
    makes clouds shadow themselves and carries most of their form. Its
    **reach** (`sun_steps × sun_step_length`) is held constant while only its
    resolution changes. Beer-Lambert exponentiates the integral, so a coarser
    sampling of the same path approximates it well; a shorter path
    approximates nothing.

## Lighting

The billboards follow the reference shader:

- a dual-lobe Henyey-Greenstein phase for the forward-scatter silver lining;
- a short per-fragment march toward the sun through the field, which is what
  makes clouds shadow each other;
- the "powder" dark-edge term and a multiple-scattering fill;
- the sky's display transform.

All of these are live uniforms: `set_sun` moves the sun with no rebuild. These
rules live in the Python and in `cloud.frag`, and each has a visible signature
when broken:

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
- The field is sampled **on the quad plane**, not offset to mid-chord. The
  offset moves the lookup by up to a particle radius (most of a noise
  feature), so overlapping billboards disagree about where the cloud is and
  cut holes in each other.
- The **powder** term takes the *local* density over a fixed reference length.
  Fed the sunward depth it inverts (the lit surface darkens most); fed the
  quad's chord it depends on billboard size.
- The **sun march** uses only the two lowest octaves, affinely calibrated onto
  the full field (`noise.shadow_calibration`). Plain renormalisation leaves
  mean and spread too high: the march sees 1.65x the cloud and greys the tops
  in full sun.
- The **LOD crossfade** weights *optical depth*, after the per-billboard cap.
  On depth the shells' transmittances multiply out exactly; on alpha two
  half-weight billboards at depth 1.5 composite to 0.544 instead of 0.65, a
  ~16% light ring that sweeps with the camera. Weighting before the cap would
  let each shell reach it and sum to twice it.

## Genera

[`genera.py`](../../src/world_builder/clouds/genera.py) adds stratus, and the
convective continuum from fair-weather cumulus through congestus to
cumulonimbus (`PlanetClouds(genera=GeneraSpec(...))`, `CloudLook.genera`,
`--genera` in the demo, `terrain_shots.py` and `cloud_bench.py`). Its maps are
a **placeholder** for the weather (issue #8): features laid out around one
point. Either renderer draws them: billboards (two `CloudVolume`s), or the
raymarch (one `CloudMarch` of two families, below).

**The raymarch with genera.** `CloudMarch(genera=...)` marches both families
in one pass, so a stratus cloud in front of a tower and one behind it composite
correctly: each family's slab gives up to two stretches of a ray
(`march.family_segments`), the four sorted along it, and every sample takes the
field, the weather (the shift, the convective depth, the anvil's spread:
`texture(weatherMap | otherWeather, p).gba`), the light table and the shadow
maps of the stretch's family. The families' layers share `layerParams` (at
most `MAX_LAYERS` between them, so with two the shells' bands are at most 3
each: `march._configure`); a stratus cloud takes the convective clouds' shadow
from their maps (`cloudSunDepthOther`). Both families cast their shadows on the
ground, sea and models (`PlanetClouds.shadow_caster(family)`, the maps' pairs
as for the billboards). The reflection's deck is the convective family's alone.
The convective weather is read at every sample (`WEATHER_STRETCH_GENERA`): a
tower's depth, which sets its top, changes over tens of metres, and read once
a kilometre it showed as horizontal bands across every tower. The fair-weather
march keeps the kilometre.

**Two families, one sort.** Billboards mix cloud types by superposition: each
samples its own type's field, and the blender adds them. So the genera are two
families, each a `CloudVolume` (one field), in bands of sky that do not
overlap:

- **stratiform** (`CloudType.STRATUS`): a sheet 500 m thick, its base at
  `GeneraSpec.stratus_altitude`;
- **convective** (`CloudType.CONVECTIVE`): the cumulus field, its clouds
  growing with the convective **depth** (0 to 1, a map of its own).

Their billboards share the shells' bins, which sort them back to front as
one cloud's: the bands do not overlap (`check_layers` refuses a stratus
reaching the convective base). Their two decks are spheres one inside the
other, drawn in a fixed bin farthest first: the stratus's first from above,
the convective one's from under both.

**The depth** (`DensityField.min_top`, `anvil`, `cell_weight`, all neutral
by default):

- the clouds' **top** rises from the cumulus's (1 km above the base, a
  share `min_top` of the slab) to the slab's (10.5 km). The lift, which bulges
  the tops, and the base's fill follow the local top (`noise.top_fraction`).
  Above it nothing is cloud;
- a **cell octave**, four times coarser than the first (8 km features),
  adds to the fbm in proportion to the depth, about its mean: it gathers the
  deep clouds into towers;
- under a deep cloud's top, the threshold falls (`noise.ANVIL_BAND`): its
  **anvil**, which spreads past the tower as a **shelf** (below);
- **billboards grow with the depth** (`CloudSpec.deep_radius`, 2.5 times at
  depth 1) and are thinned as the cube of their size, so the overlap, and
  the extinction, hold. A cumulonimbus takes as many billboards per
  kilometre of its features as a cumulus, not per cubic metre.

The weather cube carries the depth beside the coverage and the shift (its
blue channel), and the anvil's spread (alpha; one fetch). The shader and
`density_on_sphere` read it, and for every other field it changes nothing
(the billboards' images are within one 8-bit step of before).

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
  its top): a flat, smooth top cut by the threshold cuts every billboard
  across it along one line, and from above the sheet was a stack of pills;
- **smooth**: the two coarse octaves only, the others their mean
  (`noise.shelf_fbm`; no cauliflower), and its own fbm shift
  (`DensityField.shelf_shift`, `noise.with_shelf`): 97 % of its middle is
  cloud, so from below it is an opaque ceiling;
- the field takes the column's density or the shelf's, whichever is
  denser: over the tower nothing changes under the shelf, over a cumulus
  the shelf floats above it.

Tried on the way: a squash of the noise in the shelf's band (features
three times wider than tall) sliced into pills by the billboards; a softer
threshold window left every billboard half transparent, and their round
sprites showed from below as bubbles; a shelf coverage counted on columns
(any cloud through its band) left separate puffs.

From below it is a grey ceiling; under it, the sky term is occluded from
the shelf's own underside. The deck samples it at the top of its rise.

**Placement** draws a tile's candidates up to its highest local top (the
depth read over the tile; the whole slab under a shelf), not through the
whole 8.3 km slab. A billboard over its column's top is in a shelf, and as
large as a tower's.

**Their cost** is overdraw (performance §22): near the big clouds, half
resolution (`CloudSettings.half_resolution`, on at LOW and MID) makes
frames 2.5-2.9 times faster on a weak GPU. At LOW and MID, billboards deep inside a cloud grow
and thin as the cube (`CloudSettings.interior_growth`,
`sphere.interior_levels`): the same overlap, so the same extinction, for
14-16 % less near a cumulonimbus and softer detail inside it.

**Not yet.** The plume is straight: the wind is read at the tower, not
along the plume. Its underside is lit as the rest of the cloud: no darker base
for deep clouds. Close up from below, its billboards' round sprites show
faintly, as the stratus's do. The tower's stem, at coverage 1, has
straight sides. From the ground, the air hazes out a cumulonimbus's lower half at
40 km, and its top half reads as a large cumulus. The stratus shows its billboards' outlines close up
from above: a uniformly dense sheet has no field edge to hide the sprites
behind. And the cost near the big clouds (performance §21).

## The raymarch

The raymarch is the default renderer at every quality preset
(`CloudSettings.renderer`); the presets set its resolution and steps (LOW and
MID a quarter of the view's resolution, HIGH and ULTRA a half).

**The sea's reflection** sees only the deck (`sphere.deck_mesh`, the
billboards' far field): no cloud shapes and, with genera, no stratus. It is
good enough under the ripples, so it is left as is. If it is wanted: a cheap
mirror march, a third march layer from the reflection camera into the
reflection's buffer (a quarter of its resolution, about 24 steps, no history,
no composite; the reflection's own depth as the scene's depth), active only
while the sea is in view and `reflected` is on; about 0.3-0.6 ms (a guess).

[`march.py`](../../src/world_builder/clouds/march.py) (`CloudMarch`) draws
the same field, weather and lighting as the billboards and the deck, without
their overdraw. Billboards are a raymarch whose steps are quads: every one
of them is drawn, shaded and blended, so their cost grows with the cloud's
volume on screen (15 ms inside the slab on a laptop's GPU, 41 ms in a dense
cloud). Here each pixel steps along its own ray:

- **Through the slab**, between the spheres at its base and top, less the
  planet: up to two segments (a ray skimming under the slab leaves it and
  comes back), from the near plane to the **scene's depth** (the near
  range's surface where it has one, else the far range's).
  A game's model inside a cloud is hidden by just the cloud in front of it,
  and the camera inside a cloud sees it all around (the billboards fade out
  near the camera, which leaves a clear bubble).
- **In steps** of `MIN_STEP_FEATURE` of the finest feature near the camera
  (50 m for cumulus), `STEP_GROWTH` of the distance farther, all lengthened
  alike when a ray needs more than `CloudSettings.march_steps`, and at
  least `SEGMENT_SAMPLES` (8) a stretch of the slab: a far ray crossing it
  steeply in a few steps longer than its clouds' edges are thick hits or
  misses thin cloud by chance (the sparkle seen from 20 km up). Each sample
  is jittered within its step by a **blue noise** (`blue_noise`, a 64 x 64
  tileable void-and-cluster texture) read at the *march's* pixel, shifted by
  which pixel of its block this frame traces (R2) and moved on by the golden
  ratio at each **visit** of the pixel (`visit_of`), not each frame: traced
  once a block's cycle, a per-frame jitter skipped half a step a visit and
  never stratified it. (The interleaved gradient noise at the view pixel, a
  stride of `uBlock` apart, aliased its own diagonals into a lattice of
  stripes that showed in motion, worst at a quarter resolution.) The march stops once the cloud is opaque. The steps' geometry
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
- **The field** with the octaves the billboard shells would have there
  (`shell_octaves` over their distance bands), and where a view pixel spans
  more than half a fine octave's feature (the 3rd and 4th, 285 m and 125 m
  for cumulus) or the billows', that one fades to its mean (as an octave
  left out does: the coverage holds): it would alias, a different value at
  every jitter and every move. The same goes for features finer than the
  **step** (`uStepFade`, a share of its length, over the same ramp: 0 at full
  resolution, 0.1 at a half, 0.25 at a quarter, `march.STEP_FADES`: more is steadier
  and softer): beyond a few
  kilometres the steps grow with the distance (1 %), and an octave they cannot
  sample aliases into the jitter however finely a pixel resolves it (the
  shimmer of far clouds, which more steps did not remove). The weather's shift is read
  once per kilometre of ray (`cloudDensityShifted`), not at every sample.
  Two departures, march-only, give back the billboards' look:
  - the view ray takes **twice the field's extinction**
    (`VIEW_EXTINCTION_SCALE`): a billboard carries its sample's density over
    a whole chord, which keeps a cloud's edge crisp, and the field's own
    soft edge, integrated, reads as a blur;
  - **billows** carve the edges, as the billboards' lumpy sprites do: a
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
    value noise's lattice planes.
- **Lit** per sample exactly as the billboards are (`cloud_shading.glsl`,
  which both renderers include). The air is applied once per pixel, at the
  cloud's transmittance-weighted distance, with the billboards' per-vertex
  sample count.
- **Far away**, where a pixel spans more than the features, it samples the
  deck's expected coverage as a homogeneous slab whose vertical crossing
  takes the deck's alpha, lit by the deck's averages. The march reaches
  orbit with no deck, and nothing changes colour closing in.

**The passes.** One march covers both depth ranges of `SplitDepthCameras`
(`CloudMarch.attach`; `PlanetClouds.use_split_depth`, which a `World` calls
from `use_split_depth`), as EVE-Redux marches once over KSP's two cameras'
depths:

1. a camera per range, sharing its lens, draws the **scene's depth** at the
   view's resolution into a texture. Everything but the clouds and the sky
   is drawn, the terrain through its depth-only stage, as for the
   half-resolution pass;
2. a fullscreen card **marches** one pixel of every block (2 × 2 at a half resolution, 4 × 4 at a quarter)
   (`march_resolution` ½; 4 × 4 at ¼), a different one each frame in a
   Bayer order, from the near range's near plane to the near range's
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
   camera) and held within 1.25 standard deviations of the 3 × 3 samples
   around. While the view is still (`uStill`, from the camera's move and
   turn), that range is widened by a floor and a share of the mean, or the
   steps' own noise would clamp the history and start its count over; in
   motion the floor is gone, or a cloud coming closer would keep the thin
   look it had. The samples' mean and spread are Gaussian-weighted by their
   distance to the pixel (`STATS_REACH`), not a box about the nearest sample
   (whose centre moves every half block and puts the lattice into the clamp), and
   so is the distance the history is moved by. A pixel that takes most of this
   frame's samples (its history lost or short) takes their smooth mean instead
   of the bilinear mix of four, by `uSmooth` (`march.SMOOTHING`: all of it at a
   quarter resolution, half at a half, none at full): each sample's noise was a
   dot of its own. Each pixel counts the samples it has gathered (the distance
   target's second channel): this frame's share falls as 1 / (count + 1)
   down to `MIN_WEIGHT` (0.03), and what the clamp takes away takes the
   count down (to 4 at least, so a noisy sample is not taken
   whole). Two results alternate, each frame's reading the other's;
4. two cards lay the result over the view, premultiplied "over", at the
   **cloud's own depth**: the near range's card over the pixels where the
   near range's depth pass drew something (the near camera draws its
   surfaces over the far pass afterwards, and the clouds in front of them
   must come after), the far range's card over the rest, in the far pass's
   clouds' slot (after the sky, before the cirrus). Each range's depth test
   hides the cloud pixel by pixel at the view's resolution, behind anything
   nearer, which also drops any stale history a moving model would leave
   behind it.

The near range's depth pass and card run only when the slab reaches into the
near range; otherwise the march stops at the far range's scene, and what the
near camera draws covers the clouds behind it. The history goes on across
that switch.

The sea's reflection sees the deck alone (the far field's layer is packed for it).
The textures may be padded to powers of two, so the shaders address them
in pixels.

**What it costs** (performance §20): 1.3-2 times the billboards' over open
sky on a weak GPU, the same inside the slab, two-fifths in a dense cloud.

## Testing

The headless tests cannot check the pixels, nor even whether the GLSL compiles
(`Shader.load` succeeds with no graphics context). That needs
[`scripts/terrain_shots.py`](../../scripts/terrain_shots.py) (its `cirrus_*`
and `in_clouds` poses among others, `--cloud-renderer` to pick the
renderer) or an offscreen GL context; the cirrus's shapes can be previewed
without one ([`scripts/cirrus_preview.py`](../../scripts/cirrus_preview.py)).

[`scripts/cloud_bench.py`](../../scripts/cloud_bench.py) times and shoots
the clouds alone (the air, the sky, no terrain) from fixed poses, with
either renderer, still or moving (`--move`). Its `model_in_cloud` pose puts
a black ball, standing in for a game's model, half into a cloud, and
`--interactive` flies it in and out of the clouds, switching renderers with
M. `--stability N` scores the clouds' steadiness over N frames read back
(each pixel's brightness spread, and the raymarch's block grid: see the
script), and each pose's frame comes with its 95th percentile
([performance §23](performance.md)). On WSL, run it with `GALLIUM_DRIVER=d3d12`, or Mesa draws on the CPU.
