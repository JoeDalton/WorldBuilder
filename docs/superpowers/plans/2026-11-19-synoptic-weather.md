# Synoptic Weather Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder weather with a closed-form W(t) model:
- jets as Rossby waves;
- lows, highs and tropical cyclones on analytic tracks;
- pressure and wind derived from those, plus fronts;
- genus coverages from conceptual-model rules;
- local presets pinned at coordinates;
- on the Kerbin-like, Earth and rocky-Jupiter planets.

**Architecture:** A new package `world_builder.clouds.synoptic` computes the scales, the jets, the
objects, the fields, the fronts and the genus rules as pure numpy. `weather.py` and `genera.py` keep
their public functions and GPU channel layouts, and are rewired to fill them from it. Pins are solved
before the free population is placed.

**Tech Stack:** Python 3, numpy, Panda3D (textures only), pytest. Run with
`cd /home/joe/development/WorldBuilder-weather && PYTHONPATH=src ../SpaceFlight/.venv/bin/python -m pytest …`.

**Spec:** `docs/superpowers/specs/2026-11-19-synoptic-weather-design.md`

## Global Constraints
- Work only in the worktree `/home/joe/development/WorldBuilder-weather` (branch `weather-synoptic`).
- **Never `git add` or `git commit`.** The user commits.
- **Stop after every task.** Report the changed files, the pytest output and the preview images, then
  wait for the user's go-ahead.
- GPU interface unchanged:
  - `weatherMap` RGBA = coverage, shift, convective depth, anvil shelf;
  - `genusCoverA` = St, Sc, Ns, As (channels 0-3);
  - `genusCoverB` = Ac, Ci, Cs, Cc (channels 0-3).
- No edits to `src/world_builder/clouds/shaders/*`, nor to `march.py` beyond what is strictly needed.
  Ask before touching either.
- Pure numpy, deterministic, picklable. Specs are frozen dataclasses. Caches use `functools.cache`.
- Earth preset: R = 6 371 km, `day_length_h` = 23.934. Rocky Jupiter: R = 69 911 km, `day_length_h` =
  9.925. Kerbin-like default: `day_length_h` = 6.0. Default `stratification` NH = 100 m/s.
- Bake time on the 600 km default stays ≤ 2× today's `coverage_map` (≈ 3 s). Measure it in Task 4.
- Pins never move silently. An infeasible pin raises `ValueError` with the reason.

## Review Focus
1. **Pins and objects near the poles and across the ±180° meridian.** Frames, longitude differences
   and jet phases must wrap. A pin at lon 179.9° behaves like one at 0°. Test in Task 5.
2. **Planets without ocean** (`desert`, `moon`, `ice`): no TCs, no cold-sea Sc decks, a HURRICANE pin
   raises, and nothing divides by an empty ocean mask. Test in Tasks 2 and 5.
3. **Huge populations** on rocky Jupiter (~10³ objects): bake time stays bounded (spatial culling).
   Test in Task 6.
4. **Two pins close together** (closer than the spacing rule): raise, unless they can share one object
   (for example COLD_FRONT and SQUALL_LINE on the same low). Test in Task 5.
5. **Large `time_s`** (1e9 s): the population is still steady and the phases are precise (reduce t
   modulo each period in float64 before the trig). Test in Task 2.

## File Structure
- Create `src/world_builder/clouds/synoptic/__init__.py`: re-exports.
- Create `…/synoptic/scales.py`: `Scales.from_planet`.
- Create `…/synoptic/jets.py`: `JetWave`, jet axis, troughs and ridges, upper-wind stream function.
- Create `…/synoptic/systems.py`: `Kind`, `SynopticObject`, tracks, stages, `population`.
- Create `…/synoptic/fields.py`: pressure, wind, spatial culling, `synoptic_fields`.
- Create `…/synoptic/fronts.py`: front curves per LOW and signed distances.
- Create `…/synoptic/rules.py`: zones → genus coverages, convective depth, shelf.
- Create `…/synoptic/presets.py`: `WeatherPin`, `Preset`, the solver.
- Modify `src/world_builder/terrain/planet.py`: `day_length_h` and `stratification` on `PlanetSpec`;
  presets `earth` and `jupiter_rocky`.
- Modify `src/world_builder/clouds/weather.py`: `WeatherSpec` gains `time_s` and `pins`; the tables
  become `None` = derived; `_compose`, `vortices`, `stream_*` and `_cirrus_compose` are rewired.
- Modify `src/world_builder/clouds/genera.py`: `genus_coverage` and `genus_planes` take their values
  from the rules, with features as an additive override.
- Modify `src/world_builder/terrain/preview.py` and `scripts/demo_terrain.py`: weather layers,
  `--pin`, `--planet`.
- Create `scripts/weather_report.py`: the Earth climatology report.
- Tests: `tests/test_weather_scales.py`, `test_weather_systems.py`, `test_weather_rules.py`,
  `test_weather_presets.py`, `test_weather_planets.py`.
- Docs: `docs/source/atmosphere.md` §weather, `docs/source/clouds.md` (drop the "placeholder (issue #8)"
  wording), `CHANGELOG.md`.

---

### Task 1: Planet rotation and scales

**Files:**
- Modify: `src/world_builder/terrain/planet.py` (`PlanetSpec`: add `day_length_h: float = 6.0` and
  `stratification: float = 100.0` after `has_clouds`, with docstring ivars)
- Create: `src/world_builder/clouds/synoptic/__init__.py`, `…/synoptic/scales.py`
- Test: `tests/test_weather_scales.py`

**Interfaces:**
- Produces: `Scales` (frozen dataclass), all SI except latitudes, which are in degrees:
  - `omega`, `f45`, `beta45`;
  - `deformation_radius` (m), `wavelength` (m), `wavenumber` (float, k* at 45°);
  - `cyclone_radius` (m), `tc_radius` (m);
  - `hadley_edge` (deg);
  - `jet_lats: tuple[float, ...]` (deg, one hemisphere: the subtropical jet first, then the
    eddy-driven ones poleward);
  - `storm_track: tuple[float, float]` (deg), `tc_band: tuple[float, float]` (deg);
  - `jet_speed` (m/s), `steering` (m/s).
- Produces: `Scales.from_planet(spec: PlanetSpec) -> Scales`, cached.

- [ ] **Step 1: Write the failing tests**

```python
EARTH = PlanetSpec("e", radius=6.371e6, day_length_h=23.934)
JUP = PlanetSpec("j", radius=69.911e6, day_length_h=9.925)
KERBIN = PlanetSpec("k")  # 600 km, 6 h

def test_earth_anchors():
    s = Scales.from_planet(EARTH)
    assert 800e3 < s.deformation_radius < 1200e3
    assert 5.0 < s.wavenumber < 9.0
    assert 27.0 < s.hadley_edge < 33.0
    assert len(s.jet_lats) == 2 and 25 < s.jet_lats[0] < 35 and 40 < s.jet_lats[1] < 60
    assert 5.0 <= s.tc_band[0] and s.tc_band[1] <= 30.0

def test_faster_spin_smaller_storms_more_jets():
    slow, fast = Scales.from_planet(EARTH), Scales.from_planet(replace(EARTH, day_length_h=10.0))
    assert fast.deformation_radius < slow.deformation_radius
    assert len(fast.jet_lats) >= len(slow.jet_lats)
    assert fast.hadley_edge < slow.hadley_edge

def test_jupiter_many_jets_kerbin_broad_hadley():
    assert len(Scales.from_planet(JUP).jet_lats) >= 4
    k = Scales.from_planet(KERBIN)
    assert k.hadley_edge > 40.0 and k.wavenumber < 4.0

def test_belts_ordered():
    for p in (EARTH, JUP, KERBIN):
        s = Scales.from_planet(p)
        assert 0 < s.tc_band[1] <= s.hadley_edge <= s.storm_track[0] < s.storm_track[1] <= 85
```

- [ ] **Step 2: Run** `pytest tests/test_weather_scales.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement `Scales.from_planet`.** The relations are in the spec §1:
  - f = 2Ω sinφ, β = 2Ω cosφ / R;
  - L_R = NH / f45; λ = 3.9 L_R; k* = 2πR cos45° / λ;
  - jet spacing L_β = π·sqrt(2U/β45) (Williams 1978) with U = 20 m/s, giving the number of eddy-driven jets poleward of
    the Hadley edge (≥ 1, ≤ 8);
  - φ_H = 30° · (Ro_T / Ro_T,Earth)^α with α = 1/4 (schneider2006, Eq. 3: ∝ Ω^-1/2) and Ro_T ∝ (NH/ΩR)², clamped 10–60°.

  Name the Earth calibration constants at module top, with the source key from the review's .bib in a
  comment.
- [ ] **Step 4: Run the tests.** Expected: PASS. Also run `pytest tests/ -q -x`; nothing else uses
  the new fields yet.
- [ ] **Step 5: Stop.** Report the printed `Scales` for the three planets.

### Task 2: Jets, systems on tracks, pressure and wind

**Files:**
- Create: `…/synoptic/jets.py`, `…/synoptic/systems.py`, `…/synoptic/fields.py`
- Test: `tests/test_weather_systems.py`

**Interfaces:**
- Consumes: `Scales`.
- Produces, in `jets.py`:
  - `JetWave(hemisphere: int, lat0: float, k: np.ndarray, amp: np.ndarray, omega: np.ndarray,
    phase: np.ndarray, speed: float)`;
  - `jet_waves(scales, seed, phases=None) -> tuple[JetWave, ...]`;
  - `jet_lat(jet, lon: np.ndarray, t: float) -> np.ndarray` (deg);
  - `troughs(jet, t) -> np.ndarray` (lon deg, equatorward extremes);
  - `ridges(jet, t) -> np.ndarray`;
  - `upper_stream(jets, systems, t, dirs) -> np.ndarray` (m; same sign convention as
    `weather.stream_table`).
- Produces, in `systems.py`:
  - `Kind(Enum)`: LOW, HIGH, TROPICAL, CUTOFF;
  - `SynopticObject(kind, birth: float, lifetime: float, lat0: float, lon0: float, radius: float,
    strength: float, variant: str, heading: float, seed: int)`;
  - `position(obj, t) -> (lat_deg, lon_deg)`, closed form;
  - `stage(obj, t) -> float` in [0, 1];
  - `active(obj, t) -> bool`;
  - `population(spec: WeatherSpec, scales, jets, planet, t, pinned=()) -> list[SynopticObject]`.
- Produces, in `fields.py`:
  - `pressure(dirs, systems, scales, t) -> np.ndarray` (hPa anomaly from 1013);
  - `wind(dirs, systems, scales, t, surface=True) -> np.ndarray` (N, 2), east and north m/s;
  - `near(dirs, obj, t, reach) -> np.ndarray` (index mask, spatially culled).
- Add `WeatherSpec.time_s: float = 0.0`. `population` reads the ocean and temperature through
  `weather._sampler`.

- [ ] **Step 1: Write the failing tests**
  - `test_jet_meanders_and_moves`: `jet_lat` varies with longitude by 3–15°. At t and t + 1 day the
    short-wave troughs have moved east.
  - `test_track_continuous`: for every object, `position` at t and t + 60 s differs by
    < 60 s × 3 × `scales.steering`.
  - `test_population_steady`: the counts of active LOWs per hemisphere at 20 times spread over 100
    lifetimes lie within [0.5, 1.5] × `round(scales.wavenumber)`. Same at t = 1e9 s (Review Focus 5).
  - `test_lows_downstream_of_troughs`: for ≥ 70 % of LOWs at stage < 0.5, the nearest trough lies
    within 0–0.5 wavelength to the west.
  - `test_tc_over_warm_sea_only`: every TROPICAL at genesis has `elevation < -200` and |lat| in
    `tc_band`. On `PRESETS["desert"]` there are none (Review Focus 2).
  - `test_circulation_sense`: around a northern LOW the surface wind's curl is > 0 (CCW), and < 0 for
    a southern one. The surface wind is turned toward low pressure (the angle between the wind and the
    isobar lies in 10–40°).
  - `test_spacing`: no two active objects are closer than 1.5 × (r_a + r_b) / 2.
- [ ] **Step 2: Run** `pytest tests/test_weather_systems.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `jets.py`.**
  - ω from Rossby dispersion, per spec §2.
  - Reduce `omega * t` modulo 2π with `math.fmod` in float64.
  - `upper_stream` is the jet-relative ψ: integrate a Gaussian speed profile across the distance to
    the meander axis, plus each system's Gaussian bump (the current `stream_vortices` sign rules).
- [ ] **Step 4: Implement `systems.py`.**
  - Seeded time slots per sector; a LOW spawns at the downstream side of a trough.
  - Tracks: LOW `lon0 + u t` along the jet axis with a poleward drift. TROPICAL: west and poleward,
    recurving at the subtropical ridge (a closed-form parabola in local coords). HIGH: subtropical
    static; migratory in ridges; ~10 % blocking (3× the lifetime, 0.2× the speed, and they push the
    local jet poleward). CUTOFF: a LOW detached equatorward of a deep trough, moving at 0.2× the
    speed.
  - Spacing resolved by birth order. `pinned` objects are kept first.
- [ ] **Step 5: Implement `fields.py`.**
  - Gaussian kernels elongated along the jet; the Holland profile for TROPICAL.
  - Geostrophic wind via finite differences on the sphere, ρ = 1.2.
  - Friction turn of 25° and × 0.7 at the surface.
  - Below |lat| 10° blend to trades; easterly −5 m/s.
- [ ] **Step 6: Run the tests.** Expected: PASS.
- [ ] **Step 7: Stop.** Report the object counts per kind for the three planets, at three times.

### Task 3: Fronts and genus rules wired into the cubes

**Files:**
- Create: `…/synoptic/fronts.py`, `…/synoptic/rules.py`
- Modify: `src/world_builder/clouds/weather.py`:
  - `_compose`, `vortices`, `stream_table`, `stream_vortices`, `stream_function`, `_cirrus_compose`;
  - the `zonal`, `storm_tracks`, `jets`, `cyclones`, `tropical`, `cyclone_radius`,
    `tropical_radius` fields default to `None` (derived).
- Modify: `src/world_builder/clouds/genera.py`: `genus_coverage`, `genus_planes`, `upper_wind_at`,
  `high_jet`.
- Test: `tests/test_weather_rules.py`

**Interfaces:**
- Consumes: Task 2.
- Produces, in `fronts.py`:
  - `Fronts(warm, cold, occluded)`, each an (M, 2) polyline in the LOW's local (east, poleward)
    metres;
  - `fronts(obj, t, scales) -> Fronts`;
  - `front_distance(points_local, polyline) -> np.ndarray` (signed m; positive on the cold side).
- Produces, in `rules.py`:
  - `GenusFields(convective, depth, shelf, layers: dict[str, np.ndarray])`, where the `layers` keys
    are the `GENUS_PRESETS` names: stratus, stratocumulus, nimbostratus, altostratus, altocumulus,
    cirrus, cirrostratus, cirrocumulus;
  - `genus_fields(spec: WeatherSpec, dirs, planet, climate) -> GenusFields`.
- `weather.coverage_map` = `genus_fields(...).convective` combined with the climate terms. The
  signatures of `coverage_map`, `weather_cube`, `coverage_at`, `cirrus_cover_map`, `stream_table`,
  `stream_vortices` and `stream_function` stay compatible. If `stream_table` and `stream_function` can
  no longer express ψ, change `cirrus.py`'s caller (CPU) in the same task. Do not touch the shader.
- `genera.genus_coverage(spec, name)` = `max(rules layer, features)`.
  `genera.genus_planes` = `max(rules convective/depth/shelf, features)`. The background flag still
  zeroes the rules.

- [ ] **Step 1: Write the failing tests**
  - `test_warm_front_sequence`: along a transect from 1.2 L_R ahead of a mature LOW's warm front
    toward the front, the argmax genus goes cirrus → cirrostratus → altostratus → nimbostratus, in
    that order (allowing repeats).
  - `test_cold_front_cb`: on the cold front, depth > 0.8 and shelf > 0.
  - `test_high_core_clear_and_sc_deck`: a HIGH core over land has all coverages < 0.15. A HIGH over
    sea colder than 18 °C (the eastern basin) has stratocumulus > 0.5.
  - `test_tc_eye_clear`: TROPICAL eye convective < 0.1; eyewall depth > 0.8.
  - `test_dry_slot`: behind the cold front, poleward of the triple point, all layers < 0.2.
  - `test_interfaces_unchanged`: `weather_cube(spec, table).shape == (6, n, n, 2)`;
    `genera.genus_cube(...)` and `coverage_cube(...)` keep their shapes. The existing `Feature` showcase
    still shows each genus at its feature (≥ the feature coverage).
- [ ] **Step 2: Run** `pytest tests/test_weather_rules.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `fronts.py`.** Stage-dependent shapes, per spec §4. Shapiro–Keyser variant
  for `variant == "sk"`.
- [ ] **Step 4: Implement `rules.py`.**
  - The zone table from spec §4, smoothstep falloffs.
  - Climate terms kept (`climate_factor`, orographic using `fields.wind`, maritime, fbm warped by
    systems).
- [ ] **Step 5: Rewire `weather.py` and `genera.py`.** Delete the dead `Vortex` comma code once nothing
  uses it.
- [ ] **Step 6: Run** `pytest tests/ -q`. Expected: all pass. Where an existing test asserts old
  placeholder numbers (for example the zonal tables), update it to the new behaviour and list each
  change in the report.
- [ ] **Step 7: Measure** `coverage_map` plus `coverage_cube` A/B bake time on the default planet.
  Expected: ≤ 2× the baseline measured before the change.
- [ ] **Step 8: Stop.** Report: test output, timings, and a quick `preview --weather` image.

### Task 4: Local presets and the solver

**Files:**
- Create: `…/synoptic/presets.py`
- Modify: `weather.py` (`WeatherSpec.pins: tuple = ()`); `systems.population` (honours the pinned
  objects); `jets.jet_waves` (phase refit)
- Test: `tests/test_weather_presets.py`

**Interfaces:**
- Produces:
  - `Preset(Enum)`: FAIR_HIGH, WARM_FRONT_APPROACH, COLD_FRONT, THUNDERSTORMS, SQUALL_LINE, HURRICANE,
    OVERCAST_DRIZZLE;
  - `WeatherPin(lat: float, lon: float, preset: Preset, heading_deg: float | None = None)` (frozen);
  - `solve(spec, scales, planet) -> tuple[tuple[SynopticObject, ...], tuple[JetWave, ...]]`, which
    raises `ValueError`;
  - `PRESET_GENUS: dict[Preset, str]`: the dominant genus expected overhead at the pin.

- [ ] **Step 1: Write the failing tests**
  - `test_each_preset_overhead`: on `earthlike` at a feasible pin per preset, the dominant genus at
    the pin equals `PRESET_GENUS`:

    | Preset | Expected genus |
    |---|---|
    | FAIR_HIGH | none above 0.2 |
    | WARM_FRONT_APPROACH | altostratus |
    | COLD_FRONT | cumulonimbus (depth > 0.8) |
    | THUNDERSTORMS | cumulonimbus |
    | SQUALL_LINE | cumulonimbus |
    | HURRICANE | eye clear, eyewall within 1.5 tc_radius |
    | OVERCAST_DRIZZLE | stratus or stratocumulus |

  - `test_pin_consistent_neighbourhood`: with a WARM_FRONT_APPROACH pin, the nearest trough is
    upstream of the pinned low (jet refit), and no free object overlaps the pinned one.
  - `test_infeasible_raises`: HURRICANE at lat 60° → `ValueError` mentioning "latitude". HURRICANE on
    `desert` → "ocean". LOW pins outside the storm track by more than one L_R → "storm track".
  - `test_wrap_and_poles`: a pin at lon 179.9° gives the same local genus as the same pin rotated to
    lon 0° (within 0.1). A FAIR_HIGH at 85° does not crash.
  - `test_close_pins`: COLD_FRONT and SQUALL_LINE 0.3 L_R apart share one LOW (exactly one pinned
    LOW). Two HURRICANE pins 0.5 tc_radius apart raise.
  - `test_pins_hold_at_time`: the same spec with a different `time_s` places the pinned object at the
    pin at the new time.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement `solve`.** Implement it per spec §5, in four parts:
  1. Role offsets per preset, in units of `Scales`.
  2. Invert the track for `lat0`, `lon0`, `birth` so that `stage(t)` matches the preset (mature ≈
     0.4).
  3. Linear least-squares refit of the phases on sin/cos pairs.
  4. Feasibility checks with clear messages.
- [ ] **Step 4: Run** `pytest tests/test_weather_presets.py tests/ -q`. Expected: PASS.
- [ ] **Step 5: Stop.** Report: one preview image per preset.

### Task 5: Preview, CLI, docs

**Files:**
- Modify: `src/world_builder/terrain/preview.py`:
  - `render_weather(planet, height, seed, time_s=0.0, pins=())` gains panels for isobars every 4 hPa,
    fronts (blue cold with triangles, red warm with half-discs, purple occluded), the jet axes and a
    genus colour map;
  - `--pin LAT LON PRESET` and `--time`.
- Modify: `scripts/demo_terrain.py`: `--pin LAT LON PRESET` (repeatable) and `--time SECONDS`, passed
  into `WeatherSpec` / `GeneraSpec`.
- Modify: `docs/source/atmosphere.md` §"The weather"; `docs/source/clouds.md` (genera §, the "Not yet"
  lists); `CHANGELOG.md` (Unreleased).

- [ ] **Step 1:** Run `PYTHONPATH=src python -m world_builder.terrain.preview --weather --seed 11`.
  The image is written, and the panels show isobars closed around the lows, fronts attached to the
  lows, and genus colours.
- [ ] **Step 2:** Run `scripts/demo_terrain.py --start 29.7159 29.8232 --pin 29.7 29.8 warm_front_approach --help`.
  The argument parses, and an invalid preset name gives an argparse error listing the valid names.
- [ ] **Step 3:** `pytest tests/ -q`: all pass.
- [ ] **Step 4: Stop.** Report: preview images for 3 seeds and 7 presets (paths in the scratchpad).

### Task 6: Earth and rocky-Jupiter planets in game

**Files:**
- Modify: `src/world_builder/terrain/planet.py` `PRESETS`:
  - `"earth"`: `radius=6.371e6`, `day_length_h=23.934`, Earth-like climate;
  - `"jupiter_rocky"`: `radius=69.911e6`, `day_length_h=9.925`.
- Modify: `scripts/demo_terrain.py`: `--planet NAME` (if it does not exist already); camera and far
  plane scale with the radius.
- Modify, only if broken: atmosphere scale, `grid_for`, precision paths. List each fix. Shader changes
  → ask first.
- Test: `tests/test_weather_planets.py`

- [ ] **Step 1: Write the failing tests**
  - `test_presets_exist`: radii and day lengths as above.
  - `test_jupiter_bake_bounded`: `coverage_map(WeatherSpec(PRESETS["jupiter_rocky"]))` completes in
    < 4× the Kerbin time (Review Focus 3).
  - `test_grid_levels`: `grid_for` works at both radii, and the level-0 spacing lies within 1–4 m.
- [ ] **Step 2: Run.** Expected: FAIL. Then implement, then run again. Expected: PASS.
- [ ] **Step 3:** Launch `demo_terrain.py --planet earth` and `--planet jupiter_rocky` at an orbit
  pose and a ground pose. Take screenshots with `scripts/terrain_shots.py`. Check for precision
  jitter, holes, the atmosphere look, and FPS against the >160 FPS target.
- [ ] **Step 4: Stop.** Report: screenshots, FPS and the list of fixes. Flag any `march.py` or shader
  need (for example a local high-resolution weather window at Jupiter's ~215 km per texel).

### Task 7: Draw the weather's other genera

Since the rebase on #59 the march draws a genus per coverage map. The weather fills only the
convective field, the stratus (`genera.WEATHER_GENERA`: its stratus and stratocumulus) and the
cirrus cover (its Ci, Cs and Cc together). Its nimbostratus, altostratus, altocumulus,
cirrostratus and cirrocumulus are computed but not drawn: those genera show their features alone.

**Files:**
- Modify: `src/world_builder/clouds/genera.py`:
  - `WEATHER_GENERA` gains `stratocumulus`, `nimbostratus`, `altostratus` and `altocumulus`, each
    its own channel; the stratus takes the weather's stratus alone;
  - the high layer: the cirrostratus and cirrocumulus channels of cube B take the weather's;
    `high_genera(covered=True)` lists them where the weather has them.
- Modify: `src/world_builder/clouds/weather.py` `_cirrus_compose`: the cirrus cover takes the
  weather's Ci alone (Cs and Cc have their own channels now).
- Modify: the tests that assume a genus without a feature is empty (`test_clouds_genera.py`
  `test_the_march_takes_the_genera_that_hold_cloud`, `test_clouds_high.py`,
  `test_clouds_union.py`, `test_cloud_shadows.py`), each with a ruling.
- Docs: clouds.md (Genera, The high layer), atmosphere.md (Cirrus), CHANGELOG.

- [ ] **Step 1: Benchmark first.** Warn the user before running it. Measure the march at
  2560×1440 (quality high), today versus every genus drawn, at the ground, 20 km and orbit poses,
  Kerbin-like and Earth. Main measured +0.65 ms per extra genus drawn.
- [ ] **Step 2: Stop.** Report the costs. If it is over budget (>160 FPS), propose options, for
  example skipping a genus whose weather cover is under a floor within the view's reach, or
  merging Ns into As. The user picks.
- [ ] **Step 3:** Implement the chosen option (TDD: each genus's coverage is its weather channel
  under its features; a genus that holds no cloud is still skipped).
- [ ] **Step 4:** Shots of a pinned `warm_front_approach` and `overcast_drizzle` (ground and flight)
  on Earth and Kerbin-like, before and after. Full test suite.
- [ ] **Step 5: Stop.** Report the shots, the FPS and the test output.

### Task 8: Earth credibility check, then in-game showcase

**Files:**
- Create: `scripts/weather_report.py`. It prints and checks bands for `PRESETS["earth"]` over 5 seeds
  and 5 times.

**Checks:**

| Quantity | Band |
|---|---|
| Subtropical jet latitude | 25–35° |
| Eddy-driven jet latitude | 40–60° |
| Hadley edge | 27–33° |
| Active extratropical lows per hemisphere | 5–10 |
| Low diameter | 1 000–2 000 km |
| Subtropical highs | ≥ 60 % centred on the eastern half of their ocean basin |
| TCs | 0–5 per hemisphere, 5–25° latitude, all over sea |

It also samples the warm-front genus order and the CCW/CW sense. Exit code 1 on failure.

- [ ] **Step 1:** Run `PYTHONPATH=src python scripts/weather_report.py`. Expected: all bands pass.
  If not, tune within the literature ranges and re-run. Record each tuning change.
- [ ] **Step 2:** Produce the in-game shots: Kerbin-like, Earth and rocky Jupiter, each at an orbit,
  a flight and a ground pose. On Kerbin use `--start 29.7159 29.8232` with a pin there.
- [ ] **Step 3: Stop.** Hand the user the report output and the shots, plus the exact commands so
  they can fly around themselves.

Rocky Jupiter is dropped from Task 8's shots (it runs out of memory in game; the user's ruling,
2026-11-20): Kerbin-like and Earth only.
