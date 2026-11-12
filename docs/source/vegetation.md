# Vegetation

The plants on a planet with vegetation: where they stand, and how they are drawn from the ground to orbit. The
plan, its sources and its decisions are in `VEGETATION_PLAN.md` (citation keys such as [SS4 s.29] and
[BN12 §6] are its own); this page describes what is built. Two looks draw the forest: the trees one by
one (`terrain.trees`: each asset's model out to `VegetationSettings.mesh_range`, point-sprite impostors out to
`tree_range`, both from the first trials, now fed by part V1's placement), and past them the far forest (part
V2). The ray-marched canopy trial was removed on 2026-11-12. The settings are `terrain.vegetation`'s.

## Placement (part V1)

`world_builder.terrain.forest`, in the terrain's worker processes, chunk by chunk (Serious Sam 4's jittered
grid and rules [SS4 s.22, s.29]).

**The density.** Each vertex has a vegetation density in [0, 1] (`ChunkData.density`): the planet's own
procedural field (fbm of a 2.4 km first wavelength, three octaves, its own seed, through a smoothstep: mostly
dense, with wide thinner stretches and clearings, never below 0.15), then the level's density paints. A
density paint (`stamps.DensityPaint`, a level's `density:` entries) is a grayscale image, black bare and white
full, its alpha (if any) saying where it paints: Serious Sam 4's "spray-painting" of forests, as an override,
never required.

**The ground's shape.** Beside the slope, `chunkgen` measures the convexity (`ChunkData.convexity`): minus the
elevation's Laplacian over the biome slope's 8 m, in 1/m, positive on crests and hilltops, negative in hollows
and valleys [SS4 s.22]. On the tiny planet's gentle ground it spans about ±0.004 1/m (5th to 95th percentile).

**The grids.** Each cube face carries two square grids of cells in its texture coordinates: the trees'
(`forest.TREE_CELL`, 4.6 m with today's assets) and the small plants' and rocks' (`SMALL_CELL`, 3.1 m). A whole
number of cells fills a UV period, so cells match across the wrap. Their sides are derived from the baked
crowns (`trees/crowns.json`, measured by `scripts/bake_tree_impostors.py`: each asset's area seen from
straight above and its height): a tree cell is one mean crown diameter of the tree groups, a small cell 2.5
mean diameters of the small groups. A cell holds one sample at most, by an integer hash of its global index
(the same on the GPU later): kept or not and its group, its asset, a thinning draw (two hashes: most cells are
rejected on the first), then its point (jittered up to 0.4 cell from the centre), turn, scale and tint.

**The rules.** Per group (`forest.SPECIES`), MSFS-style: instances per hectare in each biome (forest,
grassland, desert, rock, snow), spawn ratios among its assets, a scale range, a clumping strength (groves and
clearings, a 90 m value noise), and ramps on the slope, the convexity and the altitude. A group's probability
per cell at a vertex is its biome-weighted instances per hectare times the cell's area, times the density and
its ramps; the groups of one grid share its cells (their sum capped at 1). Today's groups:

| Group | Grid | Per hectare (forest, grass, desert, rock, snow) | Rules |
|---|---|---|---|
| conifers (far) | tree | 420, 0, 0, 0.2, 0 | not on slopes past 26-40°, thinner on sharp crests |
| birches (far) | tree | 50, 1, 0, 0, 0 | in hollows rather than on crests; clumped |
| snowy pines (far) | tree | 0, 0, 0, 0, 6 | not on slopes past 24-36° |
| bushes | small | 0, 10, 0, 5, 0 | not past 30-45°, fewer on crests |
| dry bushes | small | 0, 0, 1, 0, 0 | scorched tint |
| rocks | small | 0, 4, 6, 60, 0 | more on slopes and crests |

A model swap changes the crowns (baked), hence the cells and the far forest's cover; what a new model may still
need by hand is a group's instances per hectare.

**What a chunk gets.** Its plants as a list (`ChunkData.forest`: base, turn, scale, asset, tint): the samples
whose point lies in its uv square, on its ground, for the trees drawn one by one. Coarser chunks keep a share of
their trees, bigger (half per doubling of the vertex spacing past 8 m, none past 32 m); the small plants grow
on chunks up to 4 m only (they are drawn near the camera only). And per vertex, four bytes for the far forest
(the vertex record grew from 36 to 40 bytes, `chunkgen.VERTEX_DTYPE`, `chunk.c`):

- the trees' **cover**, the far groups' summed crown area per ground area (Bruneton's Γ, overlaps counted
  twice), over 2: on chunks fine enough to place every tree (8 m and finer), the actual trees' (each crown a
  square of its area, its overlap with each vertex's footprint added, from a margin of cells round the chunk
  so that neighbours agree); coarser, its expected value;
- each **far group's density**: its expected share of the trees' cells (its probability times the clumping's
  mean thinning).

Placement adds 2 to 7 ms to a chunk's generation (10 to 20 ms in all on one worker).

## The far forest (part V2)

The ground's shader draws the forest past the trees drawn one by one as Bruneton and Neyret's shader-map [BN12 §6]
(`shaders/vegetation_far.glsl`, `terrain.far_forest`). A pixel of forest is split into sunlit ground, shadowed
ground, sunlit crowns and shadowed crowns [§3], their shares read from tables measured on our own trees:

**The tables** (`scripts/bake_forest_tables.py`, `datafiles/trees/forest_tables.npz`). Per far group and tree
density λ (8 samples in [0, 1]: the share of the trees' cells holding a tree, the game's own jittered grid,
assets, spawn ratios and scales), a flat patch of 24 x 24 tree cells is rendered offscreen, tiled 13 x 13,
from 16 view zenith angles x 16 azimuths (from the sun's), each pixel's first hit and its position; and from
the sun at 16 zenith angles, a depth map of the central patch (any point wraps into it). Per view and sun:

| Table | What | Paper |
|---|---|---|
| G(θv, θl, φ, λ) | kg, the visible and sunlit ground | Eq. 14 |
| G(θv, θv, 0, λ) | kg + kḡ, the visible ground | Eq. 15 |
| T(θv, θl, φ, λ) | the visible crowns' sunlit share | Eq. 16 |
| E(θv, λ) | the visible crowns' sky visibility (uniform over the hemisphere) | Eq. 21 |
| D(λ) | the ground's sky visibility under the trees (cosine weighted) | Eq. 18 |

and the crowns' mean albedo seen from above. One 3D texture holds G, T and E for the three far groups (x θv,
y θl, z the group, λ and φ: 786 kB of 16-bit texels), linearly filtered (the shader mixes two λ itself). The bake takes
about ten minutes on the RTX 3070.

**The radiance** [§6]: Īg and J̄g are the ground's own sunlit and sky light (`objectLight`), the latter times D;
the crowns' Īt = ρ F Lsun, with the leaves' hotspot F = 1 - a1 (1 - exp(-a2 acos(v·l))) (a1 0.35, a2 5) [Eq. 8],
and J̄t = ρ E Lsky. Several groups in a pixel are taken as independent: the ground is seen through each (the
products of their kg and kg + kḡ), the crowns' terms are averaged by their shares of the view.

**The transition** [§7]. Past the trees' range R (`tree_range`) the pixel is Eq. 13:
kg Īg + (kg + kḡ) J̄g + (1 - kg - kḡ)(T Īt + J̄t). Within it, as the impostors fade out over 0.8-1 R (their
dither), Eqs. 26-27: (1 - μ)(kg / (kg + kḡ) Īg + J̄g) + μ (T Īt + J̄t), μ = (1 - kg - kḡ) smoothstep(0.8 R, R, s).
The ground between the drawn trees thus takes the visible ground's statistical shadow (from the meshes'
range out; nearer, the trees' own shadows are part V10's) and its sky occlusion D, darker close to the trees by
the contrast term Δ = 1 - a3 (Γ - Λ A) (a3 0.6) [Eq. 10]. The density used is the vertex's actual cover over its
expected one times the groups' densities (Γ / A for Λ [§7]), so the far forest is as dense, place by place, as
the trees it replaces.

**The forest floor** [SS4 s.21]: the forest biome's ground texture lies where its trees stand (their expected
cover past 0.02-0.2), grassland's in its clearings.

**Per vertex.** The shares (the table lookups, D, the crowns' albedo, the hotspot) are reckoned in the
ground's vertex stage (`terrain.vert`, `forest_table.glsl`) and interpolated; the pixel only sums the
radiances (per pixel the lookups cost +0.5 ms, per vertex +0.15 ms: performance.md §36).

**Simplifications.** Angles are taken from the local vertical, not the ground's normal (trees stand vertical;
Bruneton's slope transform is left out). The tables are measured with the trees' alpha-tested cards as opaque
(no translucency, no light through the foliage). The cover's resolution is the vertices' (2 m on the finest
chunks): with `tree_range` 0 (the far forest everywhere) the gaps look blocky near the camera.

## Cost

Measured with `scripts/vegetation_bench.py --ab` on the tiny planet, RTX 3070, 2560 x 1440, HIGH: see
[performance](performance.md) §36: the far forest alone +0.15-0.19 ms over forest (V2's share: +0.15 ms); the
whole, with the trees one by one, +2.4 to +4.2 ms among and over the forest (the impostors: part V3's).
