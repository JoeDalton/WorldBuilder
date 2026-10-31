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
  pool, and a deck sphere beyond the last shell, up to orbit. Or, with
  `CloudSettings.renderer = "march"`, the same field **raymarched**
  ([`march.py`](../../src/world_builder/clouds/march.py), `CloudMarch`, see
  [the raymarch](#the-raymarch)), an experimental second code path;
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
point. Billboards draw them; the raymarch has no depth.

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
  **anvil**;
- **billboards grow with the depth** (`CloudSpec.deep_radius`, 2.5 times at
  depth 1) and are thinned as the cube of their size, so the overlap, and
  the extinction, hold. A cumulonimbus takes as many billboards per
  kilometre of its features as a cumulus, not per cubic metre.

The weather cube carries the depth beside the coverage and the shift (its
blue channel; one fetch). The shader and `density_on_sphere` read it, and
for every other field it changes nothing (the billboards' images are within
one 8-bit step of before).

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

**Placement** draws a tile's candidates up to its highest local top (the
depth read over the tile), not through the whole 8.3 km slab.

**Their cost** is overdraw (performance §22): near the big clouds, half
resolution (`CloudSettings.half_resolution`, on at LOW and MID) makes
frames 2.5-2.9 times faster on a weak GPU. At LOW and MID, billboards deep inside a cloud grow
and thin as the cube (`CloudSettings.interior_growth`,
`sphere.interior_levels`): the same overlap, so the same extinction, for
14-16 % less near a cumulonimbus and softer detail inside it.

**Not yet.** The anvil thickens the tops but makes no flat overhang (the
noise would need stretching across the anvil's band). From the ground, the
air hazes out a cumulonimbus's lower half at 40 km, and its top half reads
as a large cumulus. The stratus shows its billboards' outlines close up
from above: a uniformly dense sheet has no field edge to hide the sprites
behind. And the cost near the big clouds (performance §21).

## The raymarch

[`march.py`](../../src/world_builder/clouds/march.py) (`CloudMarch`) draws
the same field, weather and lighting as the billboards and the deck, without
their overdraw. Billboards are a raymarch whose steps are quads: every one
of them is drawn, shaded and blended, so their cost grows with the cloud's
volume on screen (15 ms inside the slab on a laptop's GPU, 41 ms in a dense
cloud). Here each pixel steps along its own ray:

- **Through the slab**, between the spheres at its base and top, less the
  planet: up to two segments (a ray skimming under the slab leaves it and
  comes back), from the depth range's near plane to the **scene's depth**.
  A game's model inside a cloud is hidden by just the cloud in front of it,
  and the camera inside a cloud sees it all around (the billboards fade out
  near the camera, which leaves a clear bubble).
- **In steps** of `MIN_STEP_FEATURE` of the finest feature near the camera
  (50 m for cumulus), `STEP_GROWTH` of the distance farther, all lengthened
  alike when a ray needs more than `CloudSettings.march_steps`. Each sample
  is jittered within its step, the jitter moving on by the golden ratio
  each frame. The march stops once the cloud is opaque. The steps' geometry
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
  (`shell_octaves` over their distance bands). The weather's shift is read
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
- **Lit** per sample exactly as the billboards are (`cloud_shading.glsl`,
  which both renderers include). The air is applied once per pixel, at the
  cloud's transmittance-weighted distance, with the billboards' per-vertex
  sample count.
- **Far away**, where a pixel spans more than the features, it samples the
  deck's expected coverage as a homogeneous slab whose vertical crossing
  takes the deck's alpha, lit by the deck's averages. The march reaches
  orbit with no deck, and nothing changes colour closing in.

**The passes.** There is one set per depth range of `SplitDepthCameras`
(`CloudMarch.attach`; `PlanetClouds.use_split_depth`, which a `World` calls
from `use_split_depth`):

1. a camera sharing the range's lens draws the **scene's depth** at the
   view's resolution into a texture. Everything but the clouds and the sky
   is drawn, the terrain through its depth-only stage, as for the
   half-resolution pass;
2. a fullscreen card **marches** one pixel of every 2 × 2 block
   (`march_resolution` ½; 4 × 4 at ¼), a different one each frame in a
   Bayer order. It writes the premultiplied cloud and its distance;
3. a **resolve** gathers them into the view's resolution over the frames:
   the other pixels come from the last frame's result, reprojected along the
   view ray to the cloud's distance (float64 for the camera's move, float32
   only near the camera) and held within the range of this frame's
   neighbouring samples. A traced pixel blends into its history, which
   averages the jitter away. Two results alternate, each frame's reading
   the other's;
4. a card lays the result over the range's own pass, premultiplied "over",
   at the **cloud's own depth**. The scene's depth test hides it pixel by
   pixel at the view's resolution, behind anything nearer, which also
   drops any stale history a moving model would leave behind it.

The near range's clouds are close, large and soft on screen. It traces the
middle of each block every frame, averages only its jitter, at the march's
resolution, and the composite upsamples it: a full-resolution pass less. It
runs only when the slab reaches into the near range.

The sea's reflection sees the deck alone (the far layer is packed for it).
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
M. On WSL, run it with `GALLIUM_DRIVER=d3d12`, or Mesa draws on the CPU.
