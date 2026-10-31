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
  pool, and a deck sphere beyond the last shell, up to orbit;
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

- **`PRESETS`** maps `CloudType` (`CUMULUS`/`STRATUS`/`CIRRUS`/`CUMULONIMBUS`)
  to a `CloudSpec`: a field, its optics, a billboard radius range and a
  billboard aspect. Every shape parameter lives in the field.
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

## Testing

The headless tests cannot check the pixels, nor even whether the GLSL compiles
(`Shader.load` succeeds with no graphics context). That needs
[`scripts/terrain_shots.py`](../../scripts/terrain_shots.py) (its `cirrus_*`
and `in_clouds` poses among others) or an offscreen GL context; the cirrus's
shapes can be previewed without one
([`scripts/cirrus_preview.py`](../../scripts/cirrus_preview.py)).
