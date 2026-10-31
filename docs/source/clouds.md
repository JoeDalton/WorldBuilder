# Clouds

[`world_builder.clouds`](https://github.com/JoeDalton/WorldBuilder/tree/main/src/world_builder/clouds/) is built on two ideas,
and almost every design decision in it follows from one of them.

**A billboard is a sample of a volume, not a sprite.** Its geometry says only
where in the volume it sits and how many metres of cloud lie behind each of its
pixels; outline, opacity and colour are continuous functions of world position,
evaluated per fragment. Nothing about appearance may be carried per particle:
anything that is shows up as quad-shaped patches once the field gives each
billboard a crisp footprint.

**A cloud is not an object.** A cloud *type* is one density field spanning a
slab of sky, and the clouds you see are its features. That is what lets clouds
shadow each other, and why their silhouettes are exactly as crisp as the field.

- **[`noise.py`](../../src/world_builder/clouds/noise.py)** owns what a
  type looks like:
  - **`DensityField`** is the whole shape: the slab, the per-axis noise scale
    (hence the feature size), the four fbm octaves, how much of the sky is
    cloud, and the vertical profile. Frozen, so it cannot be mutated behind the
    shader's back.
  - **`coverage`** is *how much* cloud there is: the fraction of the type's own
    slab that is cloud, zenith-projected (a column counts if it holds cloud at
    any height). Per type, so a cumulus deck at 0.3 under a cirrus veil at 0.3
    leaves `0.7 × 0.7` of the sky clear, not `0.4`.

    The smoothstep window the shader uses is *derived* from it: `column_peaks`
    measures the per-column peak field strength over one noise period and sorts
    it, which is the exact inverse of the coverage function, so the threshold
    is a quantile of that array — no solver, no fitting. It has to be derived
    for two reasons: which raw fbm value means "30% of the sky" depends on the
    noise volume and therefore the seed, so it cannot be baked into a preset;
    and deriving it decouples *amount* from *shape*, so retuning the octave
    weights does not silently change how much cloud there is. `resolve_field`
    fills the window in and `measure_coverage` reads it back, a round trip the
    tests assert at an independent resolution.

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
  - **`value_noise_volume`/`build_noise_texture`** generate the single tileable
    octave the shader taps four times, as a `sampler3D`. Not a baked fbm: the
    octaves are sampled at 1/2/7/16 with independent offsets, baking would
    freeze their relative phase, and resolving the 16x octave would need 1024³.
  - **`density`** evaluates the field on the CPU *exactly* as the fragment
    shader does — same trilinear half-texel addressing, same 8-bit quantisation.
    That equality is load-bearing: placement is driven by the very field that
    is drawn. (`density_on_sphere` is its planet-mode twin.) The shader side is
    one include, `shaders/cloud_field.glsl`, and its `cloudDensity` stops after
    one or two octaves where the rest (at most their weights) cannot reach the
    threshold: exact, and most of a billboard quad is clear air.
  - Types share one volume, decorrelated by a constant noise-space offset
    (`field_offset`): an independent field with identical statistics, for the
    cost of an add.
- **[`cloud.py`](../../src/world_builder/clouds/cloud.py)** decides where a
  type's billboards go, entirely on the CPU:
  - **`PRESETS`** maps `CloudType` (`CUMULUS`/`STRATUS`/`CIRRUS`/
    `CUMULONIMBUS`) to a `CloudSpec`: a field, its optics, a billboard radius
    range, a target volume fraction and a billboard aspect. Every shape
    parameter lives in the field.
  - **`sample_field_particles`** rejection-samples the slab and keeps the points
    the field calls cloud, so the field alone bounds each cloud. The accept rate
    measures the cloud's volume directly, which makes the extinction calibration
    exact rather than tuned: extinction is the field's density divided by the
    achieved sphere-volume fraction (`volume_fraction`) and the sprites' mean
    alpha, so the billboard budget is a performance decision, not a look change.
  - **`snap_to_noise_period`** trims a layer's recycle box to a whole number of
    the field's noise periods. A cell teleporting by that width lands where the
    field is bit-identical, so the recycle is invisible and needs no fade —
    `wrapFadeBand` is zero for an isotropic layer.
  - **Quality** (`CloudSettings`, [Quality settings](quality.md); the levels
    are the shared `QualityLevel`) scales a type's cost: by
    preset across LOW/MID/HIGH/ULTRA, or knob by knob (`count_scale`,
    `march_scale`, through `scale_count` and `scale_march`; `with_settings`
    applies them). With no settings given, the field is HIGH.
    `CloudField.set_settings` changes the march live, and re-places the
    billboards for a new count. The re-placement keeps the wind's drift and
    the live `set_optics` / `set_coverage` edits.
    HIGH is the identity — what the presets ship — so a caller with no settings
    (the demo) gets the authored look rather than a silent downgrade. Two knobs,
    scaled differently on purpose:
    - Billboard count takes the full ×0.25/×0.5/×1/×2 spread, because that is
      where the frame time is (measured 2.3 / 3.6 / 5.8 / 11.4 ms on a six-shell
      deck). It changes through the **radius**, not `volume_fraction`: since
      `volume_fraction` is `(4/3)·π·n·<r³>`, holding it fixed makes the count
      fall as the cube of the radius — the trade `lod_shells` makes per shell,
      applied globally. Cutting `volume_fraction` instead holds optical depth
      only where billboards still reach; at a cloud's fringe it removes the
      silhouette rather than thinning it (measured at LOW: 18% of the cloud's
      screen area lost, against 8.8% via the radius, with more grain and no
      saving). Bigger billboards cost little because density is evaluated per
      fragment, so lateral detail and the silhouette do not depend on quad size;
      what coarsens is the **chord**, i.e. resolution along the view ray.
    - The sun march is stepped far more gently (×2/3 a level), because it makes
      clouds shadow themselves and carries most of their form; its **reach**
      (`sun_steps × sun_step_length`) is held constant while only its resolution
      changes. Beer-Lambert exponentiates the integral, so a coarser sampling of
      the same path approximates it well; a shorter path approximates nothing.
- **[`field.py`](../../src/world_builder/clouds/field.py)** turns placements
  into a drawable, animated field — the GPU/runtime half:
  - **`CloudLayer`** is one type's worth of sky. It carries no cloud count and
    no altitude range: the count follows from the field's coverage and the
    altitude *is* the field's slab.
  - **`lod_shells`** nests camera-centred layers of one type out to the
    horizon, each doubling the last's radius, box, cell size and optical-depth
    cap. At fixed `volume_fraction` each shell costs half the previous, so six
    shells reach 512 km for about 5% more than four. Adjacent shells crossfade
    over a shared band so their weights sum to 1, and share one field and
    threshold so they agree where the cloud is.
  - **Per-type fields on the GPU** ride in a `layerParams` uniform array indexed
    by a flat layer id (`LAYER_VEC4S`, twelve `vec4`s each, for up to
    `MAX_LAYERS`, eight, layers; packed by `noise.pack_layer_params`, read by
    `getField` in `cloud_field.glsl`). A uniform array rather than a
    texture: a fragment needs about two dozen of its type's parameters, which as
    `texelFetch`es would be two dozen texture reads per pixel and as varyings
    would burn most of the interpolator budget.
  - **Wind/recycling without touching vertex data.** Particle positions are
    stored relative to their *cell's* centre; only a small per-cell texture is
    updated each frame (wind drift, plus toroidal wraparound within each layer's
    camera-centred box). The vertex buffer never changes after the build; the
    vertex shader looks the centre up and builds the camera-facing quad itself.
    The same wind advects the noise coordinate, so a billboard keeps its place
    in the field, and its shape, as it drifts.
  - **Depth-correct transparency without a full re-sort.** Premultiplied-alpha
    "over" needs back-to-front order. `_restage` re-sorts a `1/resort_frames`
    slice of *cells* per frame (round-robin) against a draw order snapshotted
    once per cycle, and uploads the index buffer once per completed cycle. Cell
    populations vary, so the particle arrays are ragged with a `cell_start`
    offsets table (padding to the largest cell would cost about 3x the vertex
    memory at cumulus coverage); the gather and sort is still one `lexsort` on
    (cell rank, distance), with no Python loop over cells. The field draws in
    its own "cloud" bin (sort 25, between "opaque" and "transparent") with no
    depth write: opaque geometry occludes it, every translucent effect
    composites over it.
  - **Lighting** follows the reference shader: a dual-lobe Henyey-Greenstein
    phase for the forward-scatter silver lining, a short per-fragment march
    toward the sun through the field (which is what makes clouds shadow each
    other), the "powder" dark-edge term, a multiple-scattering fill and a
    soft-knee roll-off (planet clouds: the sky's tonemap instead, below). All
    live uniforms: `set_sun` moves the sun with no rebuild. Rules in the
    Python and in `cloud.frag`, each with a visible signature when broken:
    - `sun_brightness` and `exposure` are not independent gains: the knee
      saturates above about 3, so too bright a pair flattens lit and shadowed
      sides onto the same white.
    - The knee is applied to the **peak channel** and scales the colour. A
      per-channel knee drives every channel to the same asymptote, so an orange
      dusk sun at silver-lining intensity maps to white.
    - **Planet clouds** go through the sky's display transform instead
      (`displayColour`, `atmosphere/shaders/display.glsl`): ACES per channel
      and display gamma, keeping the hue only above white, at the sky's
      exposure (0.9). With the knee (hue kept at every level, no gamma), the
      warm excess of lit tops seen through a long low path of air showed
      orange at noon, and hazed cloud never met the sky's colour behind it
      ([atmosphere](atmosphere.md#clouds-at-the-horizon)). The flat field
      keeps the knee: it has no air, and draws over the caller's own colours,
      set for it.
    - The multiple-scattering fill is **sky-coloured and not phase-weighted**:
      light reaching a shadowed interior comes from the sky and neighbouring
      cloud, not down a beam. Sun-tinted, dusk shadows go orange instead of
      blue-grey; phase-weighted, shadowed cloud toward the sun outshines
      side-lit cloud.
    - The field is sampled **on the quad plane**, not offset to mid-chord. The
      offset moves the lookup by up to a particle radius (most of a noise
      feature), so overlapping billboards disagree about where the cloud is and
      cut holes in each other.
    - The **powder** term takes the *local* density over a fixed reference
      length. Fed the sunward depth it inverts (the lit surface darkens most);
      fed the quad's chord it depends on billboard size.
    - The **sun march** uses only the two lowest octaves, affinely calibrated
      onto the full field (`noise.shadow_calibration`). Plain renormalisation
      leaves mean and spread too high: the march sees 1.65x the cloud and greys
      the tops in full sun.
    - The **LOD crossfade** weights *optical depth*, after the per-billboard
      cap. On depth the shells' transmittances multiply out exactly; on alpha two
      half-weight billboards at depth 1.5 composite to 0.544 instead of 0.65, a
      ~16% light ring that sweeps with the camera. Weighting before the cap
      would let each shell reach it and sum to twice it.
    - **Aerial perspective** tends to the haze *colour*, never to transparent:
      fading alpha makes a deck seen from above take on the ground behind it.
  - **An open idea** for cirrus: the atlas sprites are painted cumulus puffs,
    and stretching magnifies their detail along the long axis — backwards for a
    wisp, which wants fine detail along and softness across. A cheap fix would
    modulate the sprite's thickness by a 1-D tap of the noise volume along the
    stretched axis where aspect > 1 (passing the aspect to the fragment stage).
    Not done on the flat field. Planets draw their cirrus as a surface instead
    (`cirrus.py`, [atmosphere](atmosphere.md#cirrus)): fibres along the upper
    wind's stream function, no billboards.
  - **Live vs rebuild** is decided by whether placement depends on it.
    `set_sun` and `set_optics` are pure uniform writes. `set_coverage`
    re-derives the threshold from the cached column peaks (an array index, not
    a measurement) and re-packs *without* re-placing. Placement was
    rejection-sampled at the built coverage, so lowering coverage is exact (the
    surplus billboards discard, and extinction is per billboard so the rest keep
    the right optical depth), while raising it past the built value has no
    billboards to draw with and grows holes. A tuning knob: committing a higher
    coverage means rebuilding.
  - **`Clouds`** is the thin game-facing wrapper with the usual scene contract
    (construct with `game`, `update` in `game.method_lists`, `clean()`),
    delegating everything else to `CloudField`, which builds synchronously in
    its constructor.

## Planet clouds

A planet's clouds are one part,
[`PlanetClouds`](../../src/world_builder/clouds/planet.py), drawn as two
layers by two techniques: a **volume** (`CloudVolume`, the low clouds) and,
optionally, a **sheet** above it (`CloudSheet`, high cirrus). They share the
weather, the sun, the settings and the camera, and the part checks the one
rule the pair must keep: the sheet lies above the volume's slab and inside
the air.

[`sphere.py`](../../src/world_builder/clouds/sphere.py) carries the same field
over a whole planet (`CloudVolume`): the slab becomes a height above the
sea-level sphere, billboard LOD shells are generated per cube-sphere tile in
the terrain's worker pool, and a deck sphere takes over beyond the last shell,
up to orbit. [`weather.py`](../../src/world_builder/clouds/weather.py) builds
the weather they follow: a cube map of target coverage (`WeatherSpec`), turned
into the fbm shift that gives it exactly (`shift_table`). The same shaders draw
it, with `uSpherical` set (`field.flat_mode_inputs` switches that mode off for
the flat field); in that mode their colour goes through the sky's tonemap
instead of the flat field's knee (Lighting, above). The design is described in
[Atmosphere, sky and planet clouds](atmosphere.md#planet-clouds). The cirrus
sheet ([`cirrus.py`](../../src/world_builder/clouds/cirrus.py)): see
[Cirrus](atmosphere.md#cirrus).

## Testing and the demo

The headless tests cannot check the pixels, nor even whether the GLSL compiles
(`Shader.load` succeeds with no graphics context); that needs
[`scripts/demo_clouds.py`](../../scripts/demo_clouds.py) (the planet's:
[`scripts/terrain_shots.py`](../../scripts/terrain_shots.py)) or an offscreen
GL context.

The demo builds its deck with **coverage headroom** above the authored value,
because `set_coverage` is only exact downward. Measured on six shells at
1280x720 from a 0.29 deck: no headroom is 100k billboards and 9.6 ms, 0.15 is
170k and 15.3 ms, 0.25 is 221k and 17.6 ms; 0.15 buys a 0.29 → 0.44 sweep,
which crosses from isolated puffs toward the sheet-with-holes regime. Its
`forward_gain` sweep stops at 15: above about 10 the view into the sun clips in
the tonemap, and the phase solve needs a negative isotropic weight past about
20 at 0.4 anisotropy (about 30 at the cumulus preset's 0.46). The ground is a
curved polar mesh for the same reason as the ocean — a flat plane clips the
drooped deck beyond about 113 km — with 256 segments (0.26 arcmin of flat spot,
against 1.0 at 128) out to 550 km, good for eye heights to about 24 km. Its
cirrus layer (`WITH_CIRRUS`) is toggled by zeroing the layer's optical-depth
cap, since every layer shares one Geom: a look toggle, not a cost saving.
