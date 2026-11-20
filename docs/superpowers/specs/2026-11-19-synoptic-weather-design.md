# Design: synoptic weather (jets, fronts, lows, highs, cyclones, local presets)

Status: approved 2026-11-19 (plan review). Literature basis:
[docs/literature/synoptic-weather-review.tex](../../literature/synoptic-weather-review.tex).
Branch `weather-synoptic` (worktree `../WorldBuilder-weather`), kept apart from the cloud-shape work on
`49-better-looking-cloud-shapes`.

## Context
`clouds/weather.py` bakes one static coverage cube. It combines:
- zonal belts × climate;
- comma-shaped vortices placed at random in a fixed latitude band;
- a jet that depends only on latitude;
- fbm.

Every genus except fair-weather Cu and cirrus is drawn only where a hand-placed `Feature` disc sits
around one focus point (`genera.py`, issue #8).

What is missing:
- a pressure field;
- fronts;
- a wind field;
- any relation between the weather's scales and the planet;
- time.

## Intent
Weather that reads as credible to anyone who has looked at a satellite loop or a surface analysis. That
means jets that meander, lows downstream of troughs with the right fronts and cloud sequence, highs that
are clear or capped by a stratocumulus deck, and tropical cyclones over warm sea. It must hold from
orbit, from an aircraft, and from the ground. It must cost nothing per frame. The user must be able to
pin a chosen local situation at a coordinate, with the rest of the planet arranged around it so the pin
is not an anomaly.

## Goals
- Weather scales come from the planet: radius, rotation and a stratification knob. They are not
  scaled-down Earth tables.
- A closed-form weather state W(t): any time t can be evaluated without stepping. For now it is
  evaluated at one fixed `WeatherSpec.time_s`, because cloud shapes are static.
- Every genus channel is filled by the weather. `Feature` discs become an optional override.
- Seven local presets, pinned at (lat, lon) at time t.
- Three planets viewable in game:
  - the 600 km Kerbin-like one;
  - Earth (6 371 km, 23.934 h);
  - a rocky Jupiter-scale world (69 911 km, 9.925 h).
- On the Earth preset, the weather passes a climatology credibility report.

## Non-goals
- No simulation of any kind: no fluid solve and no time stepping.
- No precipitation rendering, no surface wind gameplay and no animation in this work. W(t) makes those
  possible later.
- No gas-giant mode. "Jupiter" here is a rocky planet.
- No cloud-shape or shader changes. Those belong to the other agent; anything unavoidable is flagged
  first.

## Anti-goals (failure even if it "works")
- Pins that look pasted on: a front that ends at the pin's edge, or a hurricane next to a high with
  nothing in between.
- Earth-tuned constants hidden in the code, so that Kerbin or Jupiter look like resized Earth.
- Fronts that disagree with the wind or pressure: a cold front on the wrong side, or a clockwise
  northern low.
- A per-frame cost, or a bake so slow it hurts iteration (more than about 2× today's ~3 s on
  Kerbin).

## Constraints
- The GPU interface stays the same:
  - `weatherMap` RGBA = coverage, shift, convective depth, anvil shelf;
  - `genusCoverA` = St, Sc, Ns, As;
  - `genusCoverB` = Ac, Ci, Cs, Cc;
  - `genusShift`;
  - the cirrus cover map, plus the stream function the cirrus evaluates on the CPU (`cirrus.py`).
- Everything is pure numpy, deterministic, picklable and cached per spec. It runs in the cloud workers
  as today.
- The planet is the terrain `Planet`: `sample_dirs` gives elevation (below 0 is sea), temperature and
  humidity. +z is the north pole.

## Approaches considered
1. **Analytic synoptic objects (chosen).** A jet expressed as Rossby waves, a seeded population of
   lows, highs and tropical cyclones on closed-form tracks, and fields derived from them by
   conceptual-model rules. It is cheap, any t can be evaluated, and the objects are what make pinning
   natural.
2. **Pre-baked GCM time series** (ExoPlaSim, Isca). It has real dynamics, but needs gigabytes of
   fields. It is too coarse on a 600 km planet (the whole planet is a few grid cells at T21–T42). It
   cannot be pinned without nudging and re-running. At most it could check a climatology.
3. **Noise plus texture (status quo extended).** It is the cheapest option, but the systems do not
   hold together: fronts and lows would be texture, not structure.
4. **Reanalysis analogues** (ERA5 cut-outs pasted onto the planet). The shapes are realistic, but the
   scale cannot match a non-Earth planet and the seams break consistency.

## Design

### 1. Scales: `clouds/synoptic.py::Scales.from_planet(spec)`
Inputs:
- radius R;
- `PlanetSpec.day_length_h` (new; Ω = 2π/day). Defaults: Kerbin-like 6 h, Earth 23.934 h,
  Jupiter-rocky 9.925 h;
- `PlanetSpec.stratification` NH (new; buoyancy frequency × tropopause height, default 100 m/s, the
  Earth value).

Derived quantities, each calibrated so the Earth preset gives the observed value:

| Quantity | Relation | Earth anchor |
|---|---|---|
| Deformation radius L_R | NH / f(45°) | ≈ 1 000 km |
| Synoptic wavelength | Eady λ ≈ 3.9 L_R → storm-track wavenumber k* = 2πR cos45° / λ | k* ≈ 7 |
| Cyclone radius | ≈ 0.5–0.6 λ / 2 (seeded spread) | ~1 000 km |
| Jet spacing (Rhines/Williams) | L_β = π sqrt(2U/β), U ≈ 20 m/s eddy velocity | ≈ 1 eddy-driven jet per hemisphere |
| Hadley edge φ_H | ∝ Ro_T^α with Ro_T ∝ (NH / ΩR)², α = 1/4 (eddy-limited, ∝ Ω^-1/2; Held–Hou's 1/2 is the upper bound), clamp 10–60° | 30° |
| Belts | ITCZ at 0 (+season), subtropical highs at φ_H, one storm band per eddy-driven jet, centred on it, ± min(half the jet spacing, 12.5°) (Earth 40–65°, rocky Jupiter 8 bands); the storm track spans them | |
| TC radius | ∝ the Rossby-like scale V/f at 20°, clamped | ≈ 300–500 km |

Consequence to show the user: Kerbin with a 6 h day has ΩR/NH ≈ 1.7 against Earth's 4.6. It behaves
like a slowly rotating Earth: a broad Hadley cell (about 49°) and only 2–3 large lows per hemisphere.
`day_length_h` (about 2.3 h for Earth-like ΩR) and `stratification` are the knobs. A rocky Jupiter
gives about 5 jets per hemisphere and hundreds of small lows.

`WeatherSpec` keeps optional overrides (`zonal`, `jets`, `storm_tracks`, counts, radii). `None` means
derived.

### 2. Time model and the jet
Per hemisphere and per jet j:

φ_j(λ, t) = φ0_j + Σ_m A_m sin(k_m λ − ω_m t + θ_m)

- k_m spans about k*/2 … 1.5 k*.
- ω_m = k_m (U_j − β/K_m²)/(R cos φ0), from Rossby dispersion. Long waves are near-stationary;
  short waves move east.
- A_m are seeded, with a total meander amplitude of about 5–10°.

The upper wind is jet-relative: its speed falls off with distance from φ_j as a Gaussian, and it points
along the meander. Its stream function replaces `stream_table`: ψ is a function of the distance to the
meandering axis, plus the Gaussian rows for the systems as today. `cirrus.py` evaluates it on the CPU,
so no shader changes.

### 3. Synoptic objects
`SynopticObject(kind, birth_t, lifetime, track params, radius, strength, seed)`, where kind is LOW,
HIGH, TROPICAL or CUTOFF.

Every object has a closed-form position p(t). A **stage** s = (t − birth)/lifetime in [0, 1] picks the
life cycle:
- LOW: incipient wave → open wave → mature → occluding → occluded / decaying (Norwegian sequence,
  Shapiro–Keyser variant for some by seed). Strength rises, then falls.
- HIGH: builds, persists, decays. Some are blocking: longer life, slower, and they bend the local jet.
- TROPICAL: forms at 5–20° over sea warm enough (temperature proxy from `sample_dirs`). It drifts west
  and poleward, recurves into the westerlies, and becomes extratropical (it turns into a LOW) or decays
  over land or cold sea.
- CUTOFF: a low detached equatorward of the jet, slow-moving.

**Population.**
- Time is divided into seeded slots per storm-track sector (sector width = one synoptic wavelength) per
  hemisphere.
- Each slot may spawn a LOW at the sector's trough-downstream point. The slot length is chosen so that
  about k* lows are active per hemisphere.
- Lows move with the steering flow: ~0.6 × jet speed along the jet, plus a poleward drift.
- HIGHs: the subtropical ones (with longitudes biased to the eastern side of ocean basins, which the
  land mask finds) are long-lived and nearly static. Migratory ones sit between lows in ridges.
- Placement keeps a minimum spacing of about 1.5 × the radius sum. A conflict is resolved by dropping
  the later-born object. The rule is deterministic, so W(t) is a pure function.
- The active set at t comes from slots overlapping t, with no state.

### 4. Derived fields
Fields are evaluated on cube grids; objects are culled spatially (bins by face/tile), so a rocky
Jupiter with ~1–2 k objects stays tractable.

- **Pressure** (diagnostic and used for wind):
  p = climatological zonal p(φ; belts) + Σ kernels.
  - Lows and highs use a Gaussian kernel elongated along the jet.
  - TCs use the Holland profile.
- **Wind**:
  - geostrophic, from ∇p/(ρ f), with the gradient-wind correction near TCs;
  - the surface wind turned ~25° toward low pressure and slowed;
  - near the equator, where f → 0, it blends to the trade-wind climatology.

  This replaces `_wind_east` in `orographic` and the focus-east axis for anvils (`upper_wind_at`
  becomes a lookup).
- **Fronts** are analytic curves in each LOW's frame. Their shape depends on stage:
  - the warm front runs east from the triple point;
  - the cold front trails equatorward and west;
  - the occlusion wraps poleward as the stage passes about 0.5;
  - Shapiro–Keyser variants have a fractured front and a T-bone.

  Each point's signed distance to each front gives its zone.
- **Genus rules** (conceptual model; coverages in [0, 1], then the existing shift tables):

  | Zone | Genera (channel) |
  |---|---|
  | Warm-front band, cold side, over ~1 L_R ahead of the surface front | Ci → Cs → As → Ns as the front gets closer (B1, B2, A3, A2) |
  | Cold front, narrow band | Cb (convective depth 1, anvil shelf) and Cu congestus (depth 0.5) |
  | Warm sector | St / Sc and drizzle (A0, A1); Sc over sea |
  | Comma head and occlusion | Ns, As, embedded Cb |
  | Dry slot | clear |
  | Post-frontal cold air over sea | open-cell Cu/Sc (convective, depth ~0.3; Sc) |
  | HIGH core | clear (subsidence) |
  | HIGH over cold sea, eastern basins | Sc deck under the inversion (A1), St/fog near the coast (A0) |
  | HIGH flank | Ac (B0), fair-weather Cu |
  | TC | clear eye, eyewall Cb (depth 1), spiral bands (Cb/Cu), CDO and outflow cirrostratus (B2) |
  | ITCZ and wet tropics; humid land in the daytime | Cu → Cb by instability (convective depth from humidity × temperature) |

  The existing climate terms stay: humidity scaling, desert clearing, the maritime term, orographic
  lift (now using the true wind) and mesoscale fbm warped by the vortices.
- **Outputs.** The existing arrays keep their channel meaning:
  - `coverage_map`;
  - `weather_cube`;
  - `genus_planes` (convective coverage, depth, —, shelf);
  - `coverage_cube("A"/"B")`;
  - `cirrus_cover_map`;
  - `stream_function` and `upper_wind_at`.

  New: `synoptic_fields(spec, size)` gives pressure, the surface wind (east, north) and a zone id. It
  is for debugging and previews; it is not uploaded.

### 5. Local presets: `clouds/presets.py`
`WeatherPin(lat, lon, preset, heading_deg=None)` goes in `WeatherSpec.pins`, all at `time_s`. Each
preset is a role relative to an object:

| Preset | Pinned object | The pin sits at |
|---|---|---|
| FAIR_HIGH | HIGH, mature | its core (0–0.3 r) |
| WARM_FRONT_APPROACH | LOW, mature (s≈0.4) | ~0.5 L_R ahead of the warm front, cold side (As/Cs overhead) |
| COLD_FRONT | LOW, mature | on the cold front |
| THUNDERSTORMS | none; a warm humid sector or tropics | the convective depth is raised locally (airmass Cb cluster) and the location is required to be warm and humid |
| SQUALL_LINE | LOW, mature | ~0.3 L_R ahead of its cold front, warm sector (a pre-frontal line) |
| HURRICANE | TROPICAL, mature | its eye (or eyewall with `heading`) |
| OVERCAST_DRIZZLE | LOW, mature | its warm sector, near the warm front (St/Sc and drizzle), or a stalled front |

**Solver** (deterministic):
1. Build each pinned object so that p(t) puts the pin at its role offset. Tracks are closed-form, so
   birth and position are solved backwards. `heading` rotates the frame (for example which way the
   front faces). The default is the climatological orientation.
2. Refit the jet phases θ_m by small linear least squares in sin/cos, so that pinned LOWs sit just
   downstream of a trough and pinned HIGHs under a ridge. Free waves keep their seeded values.
3. Generate the free population, rejecting objects that conflict with pinned ones. Slots that would
   overlap a pin are dropped, so the population stays consistent around it.
4. Check feasibility and **raise with a reason**; never move a pin silently:
   - a HURRICANE needs sea and |lat| inside the TC band;
   - a LOW pin needs to be inside or near the storm track (within one L_R);
   - THUNDERSTORMS needs humid land or warm sea;
   - two pins must not share an object role that the spacing forbids.

`--pin LAT LON PRESET` on `scripts/demo_terrain.py` can be repeated. `GeneraSpec.features` stays as an
additive override; the showcase and its tests are unchanged.

### 6. Planets
`PRESETS` gains:
- `earth`: R 6 371 km, 23.934 h, with an ocean and an Earth-like climate;
- `jupiter_rocky`: R 69 911 km, 9.925 h.

Existing presets get `day_length_h`; Kerbin-like keeps 6 h.

Step 7 checks rendering at those radii:
- float32 precision;
- `grid_for`;
- the atmosphere scale;
- cloud cube texel size (Earth 512² ≈ 20 km per texel, Jupiter ≈ 215 km per texel; a local
  high-resolution window may be needed later and would touch `march.py`, so flag it, not do it here);
- camera, far plane and speed.

## What's left to the builder
Every relation and starting constant is in §"The model's equations" of the literature review
(`docs/literature/synoptic-weather-review.tex`), which the code must match. Constants marked † there
are starting values for tuning.
- Exact kernel shapes, front widths, the strength curve per stage, and noise amplitudes. Tune them on
  preview maps.
- The spatial binning scheme.
- Exactly how jet-relative ψ is built (a distance to the meander curve, or a summed wave stream
  function).
- Where the Hadley-scaling exponent and the U of the Rhines scale end up within the literature ranges,
  provided the Earth anchors are met.

## Verification
- **Unit tests** (`tests/test_weather_*.py`):
  - Scales: the Earth anchors; monotonic in Ω.
  - Time: continuity of p(t) and steady population counts over long t.
  - Lows downstream of troughs; TCs only over warm sea; CCW circulation around northern lows and CW
    in the south.
  - The warm-front genus sequence along a transect.
  - Each preset gives its expected genus at the pin.
  - Infeasible pins raise.
  - Existing cloud tests still pass.
- **Preview**: `python -m world_builder.terrain.preview --weather` gains layers for pressure isobars,
  fronts (standard symbols), the jet axis and genus colours. Check it for several seeds, presets and
  all three planets.
- **Earth report**: a script prints jet latitudes, the Hadley edge, low count and size per hemisphere,
  subtropical high longitudes against the ocean basins, and TC count and latitude, against the
  climatology bands listed in the plan.
- **In game**: orbit, flight and ground poses on all three planets with one pin each. FPS is unchanged
  (target > 160 FPS at 1440p on an RTX 3070).
