# Cloud genera: a literature review

What each of the ten WMO cloud genera looks like and where it sits, and which
density functions the graphics literature uses to draw it. This review grounds
the genus model in [Clouds](clouds.md) (design spec:
`docs/superpowers/specs/2026-11-15-cloud-genera-design.md`).

**Verification flags.** Several key decks (Nubis 2015/2017/2022/2023,
Frostbite 2016, Webanck 2018) were too large to read during the review.
Each claim is marked:

- **[V]**: verified from a fetched page or a search snippet;
- **[R]**: recalled from the source, not re-read;
- **[U]**: unverified or our own synthesis;
- **(est.)**: an estimated figure, not a sourced one.

Check the PDFs before relying on any [R] number.

## 1. The genera, meteorologically

### Étages (WMO, temperate regions)

| Étage | Genera | Base range |
|---|---|---|
| High | Ci, Cc, Cs | 5–13 km |
| Middle | Ac, As, Ns | 2–7 km |
| Low | Sc, St, Cu, Cb | 0–2 km |

WMO lists three exceptions to the table:

- As often reaches above the middle étage.
- Ns usually extends into the other two.
- Cu and Cb have low bases, but their tops may reach the middle and high
  étages.

Source: <https://cloudatlas.wmo.int/en/useful-concepts.html>.

### Element size: the apparent-width rule

WMO tells the cellular genera apart by the angle one element subtends, measured
more than 30° above the horizon
(<https://cloudatlas.wmo.int/en/ac-compared-with-sc.html>). The width in metres
is roughly base height × θ in radians.

| Genus | Apparent width | At a typical base | Width |
|---|---|---|---|
| Cc | < 1° | 6–12 km | < ~100–200 m |
| Ac | 1–5° | 3–5 km | ~50–450 m |
| Sc | > 5° | 0.5–1.5 km | > ~45–130 m |

That rule is only a lower bound for Sc. Real Sc rolls and cells are 0.5–2 km
wide. Mesoscale open and closed cells are 10–40 km across, with width-to-depth
ratios of 10–40:1 (est.).

### Summary

Optical depths (τ) are anchored on the ISCCP satellite bins (τ < 3.6, 3.6–23,
> 23), but the values below are estimates. Water contents: cirrus ~0.03 g/m³,
St and Cu 0.25–0.30, Sc 0.45, Cb 1–3
(<https://en.wikipedia.org/wiki/Liquid_water_content>, after Thompson 2007), Ns
~0.2–0.5.

| Genus | Base | Thickness | τ (est.) | Phase | Sun through it | Look |
|---|---|---|---|---|---|---|
| Ci | 6–12 km | 0.2–3 km, typically 0.5–1¹ | 0.03–3¹ | Ice | Yes; halo fragments | White, silky, fibrous |
| Cc | 6–12 km | 0.1–0.5 km (est.) | 0.3–3 | Ice, brief supercooled drops | Yes; corona, iridescence | White grains or ripples, **no shading** |
| Cs | 6–12 km | 0.5–3 km (est.) | 0.1–5 | Sparse ice | Outline visible; **22° halo** | Transparent whitish veil |
| Ac | 2–6 km | 0.1–1 km (est.) | 2–15 | Water (ice when very cold) | Thin parts: corona | White or grey, **with shading** |
| As | 2–6 km, tops higher | Hundreds to thousands of m; tens to hundreds of km wide | 5–25 | Ice over mixed over water | "Ground glass", **never a halo** | Grey or bluish, striated or uniform |
| Ns | 0.5–3 km, spans étages | 2–6 km (est.) | > 23, often 50+ | Drops, rain, snow | **Blots it out** | Dark grey, diffuse; pannus below |
| Sc | 0.5–2 km | 0.2–1 km (est.) | 3–23 | Water | Corona when thin | Grey or white, **always some dark parts** |
| St | 0–0.5 km (est.) | 0.1–0.5 km (est.) | 1–30 | Small drops | Outline **sharp** | Grey, uniform base, fog-like |
| Cu | 0.5–2 km | 0.3–6 km by species | 5–100 | Water | Opaque | Brilliant white tops, **dark flat base** |
| Cb | 0.5–2 km | To the tropopause, 9–13 km (est.) | 50–200+ | Water below, ice above | Opaque, dark beneath | Towers, flattened top |

¹ SIRTA midlatitude cirrus climatology, <https://arxiv.org/pdf/0705.2517>.

WMO definitions: `https://cloudatlas.wmo.int/en/definition-<genus>-<abbr>.html`.
Physical constitution: `https://cloudatlas.wmo.int/en/physical-constitution-<genus>.html`.

### Shapes a density function must reproduce

| Genus | Shape |
|---|---|
| **Ci** | Detached filaments, patches or bands; fibrous or silky. Fall streaks hang below, curved and slanted by wind shear. Narrow, so only fragments of a halo show. |
| **Cc** | Very small grains or ripples, regularly arranged. No self-shadowing. |
| **Cs** | A wide, near-uniform veil, sometimes visible only by its halo. |
| **Ac** | Laminae, rounded masses or rolls, regularly arranged and shaded. **Stratiformis** is most common. **Lenticularis** is a smooth, sharp-edged lens. **Castellanus** has turrets on a common flat base. |
| **As** | A featureless or striated sheet. Its base is diffuse where precipitation falls inside it. |
| **Ns** | A thick, dark, diffuse layer. Continuous precipitation reaches the ground, with ragged pannus below the base. |
| **Sc** | Tessellations, rolls or rounded masses. Non-fibrous, flat-based, lumpy-topped. |
| **St** | A flat, uniform base, sometimes fractus. May give drizzle. |
| **Cu** | Detached, dense, sharp-edged mounds, domes or towers. Cauliflower tops, flat dark base. |
| **Cb** | Its upper part is smooth, fibrous or striated, and almost always flattened into an **anvil** (incus) or a vast plume. |

Cumulus species:

- **humilis:** flattened, width:height ≥ 2:1 (est.).
- **mediocris:** about as wide as tall, small bumps on top.
- **congestus:** "strongly sprouting… generally sharp outlines and often great
  vertical extent". Taller than wide, 2–6 km deep (est.).
- **fractus:** ragged, rapidly changing.

Source: <https://cloudatlas.wmo.int/en/species-cumulus-cu.html>.

Cumulonimbus species:

- **calvus:** tops indistinct and flattened, not fibrous.
- **capillatus:** fibrous, cirrus-like tops, often an anvil.

Mammatus hang mostly under the overhanging anvil. The overshooting top above
the anvil, and anvil spans of tens to more than 100 km, are (est.).

Sources:

- <https://cloudatlas.wmo.int/en/species-cumulonimbus-cb.html>
- <https://cloudatlas.wmo.int/en/species-cumulonimbus-capillatus-cb-cap.html>
- <https://cloudatlas.wmo.int/en/supplementary-features-and-accessory-clouds-cumulonimbus.html>

### Coexistence and transitions

- **Cb** most often grows from Cu congestus. It can also grow from Ac or Sc
  castellanus, or from part of an As or Ns layer
  (<https://cloudatlas.wmo.int/en/cumulonimbus-may-form.html>).
- **Ci** forms from Cc or Ac virga, from Cb anvils, or from Cs evaporating
  unevenly (<https://cloudatlas.wmo.int/en/cirrus-may-form.html>).
- **As** forms from Cs thickening, Ns thinning, Ac virga, or a Cb spreading out
  (<https://cloudatlas.wmo.int/en/altostratus-may-form.html>).
- **Warm front** sequence: Ci → Cs → As → Ns, with St or Cu fractus under the
  Ns (est.).
- **Afternoon convection:** Cu humilis → mediocris → congestus → Cb.

### Reference photos

All are under `https://cloudatlas.wmo.int/images/compressed/`.

| Genus | Photos |
|---|---|
| Ci | `4843_main_cirrus-fibratus-increasing_clouds.jpg` |
| Cc | `5091_main_cirrocumulus-stratiformis-undulatus_clouds.JPG` |
| Cs | `4805_main_halo-phenomena-in-cirrostratus-nebulosus_clouds-photometeors.JPG` |
| Ac | `5479_main_altocumulus-stratiformus-translucidus_clouds.jpg`, `5149_main_altocumulus-lenticularis_clouds-special-clouds-and-other-features.JPG`, `4722_main_altocumulus-castellanus-radiatus_clouds.JPG` |
| As | `4785_main_altostratus-translucidus_clouds.JPG` |
| Ns | `5145_main_nimbostratus-praecipitatio-and-stratus-fractus-of-wet-weather_clouds-hydrometeors.jpg` |
| Sc | `5949_main_stratocumulus-stratiformis-opacus_clouds-lithometeors.JPG` |
| St | `5875_main_stratus-nebulosus-opacus_clouds.JPG` |
| Cu | `2635_main_cumulus-humilis_clouds.jpg`, `4993_main_cumulus-mediocris_clouds.JPG`, `4766_main_cumulus-congestus_clouds.JPG`, `4756_main_cumulus-fractus_clouds.jpg` |
| Cb | `5993_main_cumulonimbus-capillatus-praecipitatio-incus-mamma_clouds.jpg`, `5042_main_cumulonimbus-calvus-and-cumulus-congestus_clouds-hydrometeors.JPG`, `5500_main_cumulonimbus-capillatus-praecipitatio_clouds.JPG` |

## 2. The graphics literature

### Guerrilla: Horizon Zero Dawn (2015), GPU Pro 7 (2016), Nubis (2017)

Sources:

- [Advances 2015 slides](https://www.advances.realtimerendering.com/s2015/The%20Real-time%20Volumetric%20Cloudscapes%20of%20Horizon%20-%20Zero%20Dawn%20-%20ARTR.pdf)
- [GPU Pro 7, ch. "Real-Time Volumetric Cloudscapes"](https://gameenginegems.com/gemsdb/article.php?id=1319)
- [Nubis 2017](https://advances.realtimerendering.com/s2017/Nubis%20-%20Authoring%20Realtime%20Volumetric%20Cloudscapes%20with%20the%20Decima%20Engine%20-%20Final%20.pdf)

**Height gradients per type.** Each type (stratus, stratocumulus, cumulus) has
a "gradient of probability of change in density over height". A continuous
cloud-type value blends the three **[V]**. The talk contrasts this with the
"very procedural looking" result of layered Perlin noise times a single
gradient **[V]**.

**Noise [R].**

- A 128³ base texture. Its red channel is Perlin-Worley: Perlin remapped by
  inverted Worley, which gives connected billows. Its other three channels are
  Worley octaves combined into a low-frequency fbm.
- A 32³ Worley detail texture.
- A 2D curl texture that distorts the detail lookup.

**The density chain [R].** The base is remapped by the fbm, multiplied by the
height gradient, then remapped by coverage and multiplied by it, so small
clouds are lighter **[V,
[forum quote](https://gamedev.net/forums/topic/680832-horizonzero-dawn-cloud-system/?page=4)]**.
The detail then erodes it: `remap(base, detail·k, 1, 0, 1)`. The detail is
wispy (inverted Worley) at the base and billowy higher up **[R]**.

**Weather map [R].**

- 2015: coverage, precipitation, type.
- 2017: low and high coverage, peak height, density. Authoring towers became
  a matter of painting peak height.

**Anvil bias [R].**
`coverage = pow(coverage, remap(h, 0.7, 0.8, 1, lerp(1, 0.5, anvil_bias)))`
bulges coverage outward near the top. The anvil therefore comes from the
*same* base noise, which is exactly why a threshold-lowering anvil looks
lumpy.

### Nubis Evolved (2022) and Nubis³ (2023)

Sources:

- [Nubis Evolved, Advances 2022](https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-NubisEvolved-NoVideos.pdf),
  [Guerrilla page](https://www.guerrilla-games.com/read/nubis-evolved)
- [Nubis³, Advances 2023](https://advances.realtimerendering.com/s2023/Nubis%20Cubed%20(Advances%202023).pdf),
  [Guerrilla page](https://www.guerrilla-games.com/read/nubis-cubed)

**Vertical profile.** The vertical profile model has cloud min and max heights,
coverage, and separate **bottom type** and **top type** **[V, snippet]**. A
flat base can have any top: the key to telling congestus from Cb.

**Dimensional profile.** A **dimensional profile** comes from a signed distance
field of the cloud's envelope, which gives an inside-to-outside gradient. Detail
noise erodes only where the profile is positive **[V, secondary]**.

**Nubis³.** It uses voxel clouds modelled by fluid simulation and raymarched
through compressed SDFs **[V]**. The smooth anvils of Forbidden West are
authored or simulated envelopes, not thresholded fbm **[U]**.

**Lesson.** Smooth anvils and bulbous congestus come from a *low-frequency
envelope* eroded only near its surface.

### Frostbite (Hillaire 2016), Unreal and Unity

Sources:

- [Frostbite 2016 slides](https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/s2016_pbs_frostbite_sky_clouds.pdf)
- [Unreal volumetric cloud docs](https://dev.epicgames.com/documentation/unreal-engine/volumetric-cloud-component-properties-in-unreal-engine)
- [Unity HDRP volumetric clouds](https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@16.0/manual/volumetric-clouds-volume-override-reference.html)
- [Unity HDRP cloud layer](https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@12.1/manual/Override-Cloud-Layer.html)

- **Frostbite** authors clouds in Schneider's way: a weather texture
  (coverage, type), a type × height profile table, base and erosion noise,
  and a quarter-resolution march with reprojection **[R]**.
- **Unreal** has one spherical slab (bottom, height) and a Volume-domain
  material **[V]**. Several genera are expressed inside one material by
  height-conditional density **[R]**: the single march with many genera this
  design adopts.
- **Unity HDRP** has:
  - a separate cumulonimbus (anvil) map, erosion factor and scale, and a
    height erosion curve **[V]**;
  - cirrus in a separate **2D cloud layer** distorted by a **flowmap**
    **[V]**.

### Red Dead Redemption 2 (Bauer 2019)

Source: [Advances 2019](https://advances.realtimerendering.com/s2019/index.htm).

- Clouds are raymarched **[V]**.
- Multiple scattering is a sum of octaves of attenuated direct light **[V,
  cited by
  [Disney's Raya paper](https://media.disneyanimation.com/technology/publications/2021/the-atmosphere-of-raya-and-the-last-dragon.pdf)]**.
- High cirrus and altostratus are believed to be 2D or 2.5D layers **[U]**.

### Theses

- **Häggström 2018** (Umeå): a Schneider-style model with Perlin-Worley and a
  weather map **[V, secondary]**.
- **Högfeldt 2016** (Chalmers/DICE): a type height gradient **[R]**.
- [*Optimisations for Real-Time Volumetric Cloudscapes*](https://arxiv.org/pdf/1609.05344):
  early-out and step strategies.

### Academic modelling

- **Webanck et al. 2018**, [*Procedural Cloudscapes*](https://perso.liris.cnrs.fr/eric.galin/Articles/2018-cloudscapes.pdf)
  (CGF 37(2)):
  - cumuliform clouds are blended ellipsoid primitives; stratiform clouds are
    horizontal layers of varying thickness; noise adds detail **[V]**;
  - per-altitude layers with per-genus fields, anisotropic cirrus, and
    cellular altocumulus **[R/U]**.
- **Bouthors & Neyret 2004**, [*Modeling clouds shape*](https://evasion.inrialpes.fr/Publications/2004/BN04):
  stacked quasi-spherical blobs whose implicit field deforms under its
  neighbours, after Gardner **[V]**.
- **Bouthors et al. 2008**: a mesh with a thin hypertexture skin (noise
  displaced along the normal) **[R]**. The origin of "sharp envelope, eroded
  skin".
- **Gardner 1985**: textured ellipsoids, with transparency modulated by sums of
  sines and fading to the rim **[R]**. Cheap for distant cells.
- **Ebert** (*Texturing & Modeling*): cirrus as turbulence advected along a
  direction **[R]**.
- **Others:**
  - Schpok 2003: metaballs and noise **[R]**;
  - Dobashi 2000: cellular automata **[R]**;
  - Vimont 2020: stratiform clouds as "slabs of multiresolution noise" **[V]**;
  - Goswami's survey, DOI 10.1007/s00371-020-01953-y **[V, abstract]**.

### Cirrus in practice

No primary MSFS/Asobo source exists. Forum consensus is 2D opacity maps
**[V, forum]**, and one shipped method uses tiled cirrus photographs
(<https://gamedev.net/forums/topic/680832-horizonzero-dawn-cloud-system/5357932/>).
Unity uses a 2D layer with a flowmap **[V]**. EVE-Redux's raymarch shaders are
closed.

The procedural recipe below is a synthesis **[U]**:

1. Domain-warp noise along a flow (the jet plus curl).
2. Stretch it 4–20× along the flow.
3. Average 4–8 taps along the flow, a line-integral convolution, for tails.
4. Lay it in a 0.3–1.5 km slab or a 2D layer with parallax.

Our own earlier per-pixel cirrus (noise across the stream function's
contours) gave regular parallel bands with no heads or tails
([Atmosphere](atmosphere.md#cirrus)). Tufts (heads) and tails need explicit
modelling.

### Cells and waves [U, standard practice]

- **Closed cells** (bright cells, dark gaps) come from thresholded 1 − F1
  Worley. **Open cells** (dark centres, bright rims) come from F2 − F1.
- A low-frequency coverage field varies how full each cell is.
- **Kelvin-Helmholtz billows and undulatus:** warp the domain with a sine
  across the wind, `x' = x + A sin(k y)`, or stretch the cells across the wind
  into rolls.
- Roll spacing: 1–3 km for Ac undulatus, 5–20 km for Sc rolls.

## 3. Recipes per genus

Notation: `h` is the normalized height in the genus's slab and L the element
size. Each genus takes the form `density = erode(profile(h) · coverage_remap(base), detail)`.
The frequency rule is that the base period is about L, with shape octaves at
L, L/2 and L/4 (stop there for anvils, As and Ns). Billow erosion runs at
L/6–L/10, with a second erosion octave at L/20–L/30 only on convective towers.
The detail is wispy at bases, anvil undersides and cirrus, and billowy on
convective tops **[U]**.

| Genus | Slab (km) | Profile | Base noise | Anisotropy | Erosion |
|---|---|---|---|---|---|
| St | 0.1–0.8 | Flat top-hat, soft top | fbm, 10–30 km; high coverage, wide window | 2–4× horizontal | Wispy, at the base only (fractus) |
| Sc | 0.5–2 | Flat base, domed tops | Worley cells, L 1–3 km, × fbm; open or closed by weather | Optional rolls along the wind | Billowy, L/8 |
| Cu hum/med | 0.8–2+ | Sharp base, rounded top | Existing value fbm with cell octave | ~1.2× vertical | Wispy at the base → billowy above, curl-bent |
| Cu con | top 3–7 | Separate top type: tapered radius, parabolic dome | Coarse cell field with a height lift | 1.5–2× vertical | Two billow scales, **only in the outer skin** |
| Cb | top 9–14 | Tower as congestus to h ≈ 0.75; **anvil a separate term**, union | Anvil: 2 low-frequency octaves (10–30 km), no cell octave | Anvil 4–8× downwind, shear-advected | Anvil: wispy on underside and rim only; top fades in density |
| Ns | 0.5–5 | Thick top-hat, ragged lowered base | fbm, 20–50 km, coverage ~0.9 | 3× horizontal | Wispy base, virga tail below |
| As | 2.5–6 | Top-hat, soft both ends | fbm, 20–50 km | 4–6× horizontal | Minimal; striations along the wind |
| Ac | 2.5–5 | Thin lens `4h(1−h)` | Worley F1, L 0.5–1.5 km, size jitter | Undulatus: sine across the wind; lenticularis: stretched ellipses | Billowy, L/6 |
| Cs | 8–10 | Thin layer | Low-frequency veil, 50–200 km | 5–10× along the jet | Faint fibres along the flow |
| Cc | 7–9 | Thin layer | Worley grains, L 100–300 m, × fbm | Sine ripple, 0.5–1 km | Fades to its mean below 2 px a cell |
| Ci | 8–11 | Thin layer | Flow-warped noise plus Worley tufts; tails by taps along the flow | 8–20× | Hooks from the curl; fall streaks sheared downwind |

## 4. Many genera in one march

- **Bands and slab bounds.** Clip each ray to the active bands (low, mid,
  high), using the coverage maps **[V/R]**. Only Cu con and Cb span bands.
- **Cheap sample first.** Skip the 3D noise where a column's coverage is 0
  **[R]**.
- **Thin layers.** Ci, Cs and Cc take one or two samples where a ray crosses
  their slab, with thin-slab lighting and no sun march. Ac takes a few.
- **Erode only near the surface.** Skip the detail noise deep inside a cloud
  and far outside it **[V/U]**.
- **Reprojection.** Temporal accumulation at a fraction of the resolution is
  already in place ([Clouds, the raymarch](clouds.md#the-raymarch)).

## 5. Risks

- **Anvil smoothness:** low-frequency noise can look plastic, and production
  anvils are authored. Wispy erosion on the rim and underside sells scale.
- **Aliasing:** anisotropic cirrus streaks and Cc/Ac cells alias at grazing
  angles and distance. Fade the octaves by footprint, as the march already
  does.
- **Genus transitions:** blending types works from St to Sc to Cu, but not
  from Cu to Cb (the anvil's topology). A union is safer.
- **Lighting carries identity** for As, Ns and Cs: extinction,
  multiple-scattering octaves and powder. Density alone does not tell As from
  thin Ns.
- **No primary source** was found for MSFS or EVE cirrus, or for a procedural
  Kelvin-Helmholtz technique. Those recipes are our synthesis.
