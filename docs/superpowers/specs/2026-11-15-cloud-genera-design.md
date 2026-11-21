# Design: cloud genera (all ten WMO genera in one raymarch)

Status: approved 2026-11-15 (plan review). Research basis:
[docs/source/cloud_genera_research.md](../../source/cloud_genera_research.md).

## Context
Three visible defects:
- lumpy cumulonimbus anvils;
- no visible congestus in demo_terrain;
- square cirrus patches.

There is also a capability gap: only the Cu→Cb continuum, stratus and cirrus exist today.

Goal: all ten WMO genera (Ci, Cc, Cs, Ac, As, Ns, Sc, St, Cu incl. humilis/mediocris/congestus,
Cb with anvil) coexisting in **one raymarch**, viewable side by side in a dedicated demo,
grounded in a literature review.

User decisions:
- **Remove the billboard renderer entirely.** Keep the deck sphere and deck tables: they serve
  the far field, the sea's reflection and orbit views.
- **Cirrus: procedural flow streaks inside the march.** Drop the baked stamp sheet. Cc and Cs
  use the same thin-layer technique.
- **Literature review delivered twice:** `docs/source/cloud_genera_research.md` and a published
  artifact page.
- **Demo layout:** every genus in a row along one horizon, each at its real altitude.
- **Performance:** >160 FPS at 1440p on an RTX 3070 is required for normal scenes
  (demo_terrain). The showcase needs >60 FPS; further compromises can come later if needed.

## Root causes found
- **Lumpy anvil.** `thresholdLift` lowers the threshold under a deep cloud's top
  ([cloud_field.glsl:280](src/world_builder/clouds/shaders/cloud_field.glsl#L280)). It does so on
  the column's full four-octave fbm, which the march's billows also erode
  ([cloud_march.frag:129-161](src/world_builder/clouds/shaders/cloud_march.frag#L129-L161)).
  Only the separate shelf is smooth, and `max(column, shelf)` lets the lumpy column win over the
  tower. The literature agrees: smooth anvils come from a separate low-frequency envelope
  joined to the tower as a union, not from lowering a noise threshold. Nubis models
  bottom-type and top-type separately and erodes only an SDF-like skin.
- **Congestus.** Two `Feature`s exist (`genera.DEFAULT_FEATURES`: depth 0.5, coverage 0.45,
  about 30 km west). Diagnose with shots before fixing. Likely causes: they read as cumulus,
  since they have no tapered tower or skin-only erosion, or the haze hides them.
- **Cirrus squares.** The squares come from the stamp quads, and that design is being replaced.
  The draw-order rule also breaks: the sheet must lie above every cloud, but anvils reach
  10.5 km and the cirrus sits at 8-9 km.

## Approaches (architecture)
- **Chosen: a union march.** Each sample adds up the densities of every genus whose slab
  contains it. Genus slabs may overlap, and each genus has its own coverage. Rays are clipped to
  the union of the active bands (low / mid / high), the same scheme as Unreal and Nubis.
- **Rejected: more families in disjoint bands.** This is today's model. A Cb tower crosses all
  three bands, and Ns spans low and mid.
- **Rejected: one march per band, composited.** It composites wrongly where clouds cross
  bands, at roughly triple the fixed cost.

## Design

### Genus field
A new `GenusField` dataclass in `noise.py` describes each genus. It is packed into
`layerParams` (widening `LAYER_VEC4S` if needed). Its parameters:
- the slab's base and top;
- bottom and top profile curves (Nubis-style);
- the base noise kind: value fbm (existing), Worley cells (new 3D texture), or flow streaks
  (new);
- feature scale, horizontal and along-wind anisotropy, and octave count;
- erosion: billowy or wispy, amount, and skin width (erode only near the threshold);
- the smoothstep window, extinction, and phase/optics (`CloudOptics`);
- the coverage channel it reads.

The convective genus keeps its specialised path (depth, cell octave, shelf) with the fixes
below. A single `genusDensity(p)` GLSL function adds up the active genera. The view samples,
the sun march, the cloud shadow-map bake and the light-volume bake all call it, so every genus
shadows every other.

### Weather maps
Genus cube maps, RGBA (`genera.genus_planes`, extended):
- convective: coverage, shift, depth, anvil spread (as today);
- stratiform: St, Sc, Ns, As;
- mid/high: Ac, Ci, Cs, Cc.

Each genus's coverage→shift calibration (`weather.shift_table`) goes to the shader as a small
1D table per genus, so the cube channels hold coverage only. Each genus keeps a NumPy mirror
for calibration and tests. The strict 8-bit CPU/GPU equality, needed only by billboard
placement, is relaxed to a tolerance.

### Recipes per genus
From the review. L is the element size.

| Genus | Slab (km) | Recipe |
|---|---|---|
| St | 0.1–0.8 | Low-frequency fbm (10–30 km), stretched 2–4× horizontally, wide window, wispy erosion on the base only (fractus) |
| Sc | 0.5–2 | Worley cells L 1–3 km × fbm; flat base, domed top; billowy erosion at L/8 |
| Cu hum/med | 0.8–2+ | Existing field; sharp base; erosion wispy at the base, billowy above |
| Cu con | top 3–7 | Tapered tower `r(h)=1-0.3h`, parabolic dome (cell field − k·h²), vertical stretch 1.5×, erosion only in the outer skin |
| Cb | top ~10.5 | Tower as congestus up to h≈0.75; anvil a separate 2-octave low-frequency lens, stretched and shear-advected downwind, joined as a union; wispy erosion on its underside and rim only; top fades in density |
| Ns | 0.5–5 | Very low-frequency fbm, coverage ~0.9, thick top-hat, ragged base, virga tail below the base (precipitation × exp fall-off) |
| As | 2.5–6 | Low-frequency fbm stretched 4–6×, low extinction ("ground glass"), faint striations along the wind |
| Ac | 2.5–5 | Thin lens `4h(1-h)` of Worley cells L 0.5–1.5 km, size jitter, billowy erosion at L/6; 2–6 samples |
| Cs | 8–10 | Thin layer: very low-frequency veil, faint fibres stretched along the jet, very low optical depth (0.03–0.3) |
| Cc | 7–9 | Thin layer: Worley grains L 100–300 m, sine ripple; fades to the mean density below 2 px per cell |
| Ci | 8–11 | Thin layer: domain warped by the upper wind (`genera.upper_wind_at` / `weather.stream_function`) plus curl; noise 8–20× anisotropic; Worley tufts as heads; 4–8 taps along the flow for tails; shear-offset fall streaks. Heads and tails must avoid the earlier per-pixel cirrus's failure, regular bands |

### High layer in the march
Ci, Cs and Cc are evaluated where each ray crosses their thin slab. They take 1–2 jittered
samples per crossing, with no sun march (the thin-slab lighting from `cirrus.frag`: phase,
reddened sun tint, twilight), and are sorted into the segment list. Distance filtering fades
sub-pixel octaves to their mean, reusing `resolved()` and the footprint logic.

### Cb and congestus fixes (on the convective path)
- Remove the anvil term from `thresholdLift`. In the anvil band, fade octaves 3–4 and the
  billows to their means with the anvil weight. Stretch the noise horizontally there and union
  with the shelf. Extend the shelf with shear advection along the upper wind, and wispy erosion
  on its underside only.
- Congestus gets top-type parameters: taper, dome and skin-only erosion. These keep the
  coverage calibration valid, because the depth tables are recomputed.

### Showcase demo
- `genera.showcase_features()` lays out an arc about 40–70 km from the start: one strip per
  genus, ~15–25 km wide, each at its real altitude. The high genera sit over their own strips.
- `scripts/demo_genera.py` reuses demo_terrain's app class with this `GeneraSpec`, a fixed
  start pose and C-style toggles per genus.
- `scripts/terrain_shots.py` gets a named pose per genus. Shots must pass the user's
  `--start 29.7159 29.8232` with `--at` (from memory).

## Phases
Each phase ends with shots and tests passing.
0. **Docs.** Write the design spec (`docs/superpowers/specs/2026-11-15-cloud-genera-design.md`)
   and the literature review (`docs/source/cloud_genera_research.md`), merging the WMO sheet and
   the graphics survey with citations and verification flags. Publish the review as an artifact.
   Then invoke `superpowers:writing-plans` for the detailed task plan.
1. **Cb anvil and congestus fixes** on the current code, so the visible wins come early.
   Before/after shots.
2. **Remove billboards.**
   - Drop `sphere.py` placement and shells, keeping `deck_mesh` and the deck tables.
   - Drop `cloud.frag`, `cloud_sphere.vert` and `half_res.py`.
   - Drop `CloudSettings.renderer`, `count_scale`, `interior_growth` and `half_resolution`.
   - Drop the cloud tile jobs in `terrain/workers.py`, the renderer switch in `cloud_bench.py`
     and `demo_terrain.py`, and the related tests (`test_clouds_sphere.py` and others).
   - Update `clouds.md`.
3. **Union-march framework.** Add `GenusField`, the genus cubes, the per-genus shift tables
   and `genusDensity`. Port the convective and stratus genera. Their shots should be visually
   the same as after phase 1. Shadow maps and the light volume switch to the summed density.
4. **Showcase demo** with the existing genera (St, Cu, Cu con, Cb).
5. **Volumetric genera:** Sc, Ns, As, Ac. Add Worley 3D texture generation to `noise.py`.
6. **High layer:** Ci, Cs, Cc. Remove `CloudSheet`, the atlas, `bake_cirrus_atlas.py` and
   `cirrus_preview.py`. Keep `cirrus_cover_map` and the upper wind.
7. **Docs and performance pass:** `clouds.md`, `atmosphere.md` and `performance.md`.

## Critical files
- `src/world_builder/clouds/`:
  - `shaders/cloud_field.glsl`, `shaders/cloud_march.frag`, `shaders/cloud_shading.glsl`
  - `noise.py`, `genera.py`, `weather.py`, `march.py`, `planet.py`, `cloud.py`
  - `light_volume.py`, `shadow_map.py`, `sphere.py`, `cirrus.py`
- `scripts/demo_terrain.py`, `scripts/terrain_shots.py`, `scripts/cloud_bench.py`, and the new
  `scripts/demo_genera.py`
- `tests/test_clouds_*.py`, `tests/test_cloud_*.py`

## Verification
- **Tests:** `PYTHONPATH=src ../SpaceFlight/.venv/bin/pytest tests -q` after every phase. New
  tests cover each genus's coverage calibration (within 5 %), its slab bounds, the union
  segment clipping, and that `genusDensity`'s NumPy mirror matches the shader formulas within
  tolerance.
- **Shots:** each defect gets before/after shots in `terrain_shots.py` (Cb anvil close and far,
  the congestus line, the cirrus overhead and at grazing angles), plus a per-genus showcase pose
  compared against the WMO reference photos listed in the review.
- **Steadiness:** `cloud_bench.py --stability` on the new genera, especially Ci and Cc, which
  are prone to aliasing.
- **Performance:** demo_terrain at HIGH, 2560×1440, from the user's start pose, must stay above
  160 FPS. The showcase (`demo_genera.py`, all genera in view) must stay above 60 FPS on the
  same setup. Record a baseline before phase 1 and compare after phases 3, 5 and 6. Add the
  numbers to performance.md.
