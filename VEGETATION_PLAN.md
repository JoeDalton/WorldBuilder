# Vegetation plan: forests, plants, rocks, grass, flowers and fields

The second pass at WorldBuilder's vegetation, after the first two trials (both kept, below) were judged slow
and ugly (2026-11-10). It follows the literature review of how flight simulators and large open-world games
draw forests. The user's choices: **Serious Sam 4, Bruneton & Neyret, and Mantler et al. are the main leads**;
other references stay where they have a good trick; **no wind for now**; ~~both existing paths are kept~~ (the ray-marched canopy removed after V2, 2026-11-12);
Serious Sam 4's rocks, grass, flowers and fields are in scope.

Rules: never commit unless asked; stop for the user's visual check after each part; every cost measured with
`scripts/vegetation_bench.py --ab` (each pose timed with the vegetation off and on in turn, in one run) on the
tiny planet (`--planet tiny`), plus the 600 km demo level before a part closes; RTX 3070, 2560 x 1440, HIGH,
the machine idle. Numbers go to docs/source/performance.md and a new docs/source/vegetation.md; the CHANGELOG
as parts close.

## Sources

Kept in [sources/](sources/README.md). Citation keys used below:

| Key | Reference |
|---|---|
| **[SS4 s.N]** | A. Ladavac, *Four Million Acres, Seriously: GPU-Based Procedural Terrains in Serious Sam 4: Planet Badass*, GDC 2019, slide N: [sources/croteam_gdc2019_4_million_acres_serious_sam_4.pdf](sources/croteam_gdc2019_4_million_acres_serious_sam_4.pdf) ([gpuopen.com](https://gpuopen.com/gdc-presentations/2019/gdc-2019-agtd2-4-million-acres-serious-sam-4.pdf)) |
| **[Mantler §N]** | S. Mantler, S. Jeschke, M. Wimmer, *Interactive Landscape Visualization Using GPU Ray Casting*, 2006, section N: [sources/mantler2006_interactive_landscape_visualization.pdf](sources/mantler2006_interactive_landscape_visualization.pdf) ([TU Wien](https://www.cg.tuwien.ac.at/research/publications/2006/Mantler-06-landscape/Mantler-06-landscape-paper.pdf)) |
| **[BN12 §N]** | E. Bruneton, F. Neyret, *Real-time Realistic Rendering and Lighting of Forests*, Computer Graphics Forum 31(2), 2012, pp. 373-382, DOI 10.1111/j.1467-8659.2012.03016.x, section N (Eq. M: equation M): [sources/bruneton_neyret2012_forests.pdf](sources/bruneton_neyret2012_forests.pdf) ([HAL hal-00650120](https://inria.hal.science/hal-00650120v1), [Eurographics](https://diglib.eg.org/handle/10.1111/v31i2pp373-382)) |
| **[vD08]** | J. van Dongen, *Interior Mapping*, CGI 2008: [sources/vandongen2008_interior_mapping.pdf](sources/vandongen2008_interior_mapping.pdf) |
| **[XP12]** | Laminar Research, *Next generation trees and OpenGL* (2021) and *Building custom 3-D trees*: [sources/web/xplane_next_generation_trees.html](sources/web/xplane_next_generation_trees.html), [sources/web/xplane_building_custom_3d_trees.html](sources/web/xplane_building_custom_3d_trees.html) |
| **[MSFS]** | MSFS SDK, *Species definitions*: [sources/web/msfs_sdk_species_definitions.html](sources/web/msfs_sdk_species_definitions.html) |
| **[UE-IMP]** | Epic, *Impostor Baker plugin*: [sources/web/unreal_impostor_baker.html](sources/web/unreal_impostor_baker.html); R. Brucks, *Octahedral Impostors*, <https://shaderbits.com/blog/octahedral-impostors> (not saved: rendered by script) |
| **[HZD]** | Guerrilla, *GPU-based procedural placement in Horizon Zero Dawn*, GDC 2017: [sources/web/guerrilla_hzd_gpu_placement.html](sources/web/guerrilla_hzd_gpu_placement.html) |
| **[FINS]** | Falcon BMS forum, *"Fins" for 3D forests* (IL-2's far forests): [sources/web/falconbms_fins_3d_forests.html](sources/web/falconbms_fins_3d_forests.html) |

## The first two trials

The ray-marched canopy (1) was removed from the code on 2026-11-12, after V1 and V2 (the user's decision; in
git history before that date). The instanced trees (2) stay, fed by V1's placement, in front of V2's far
forest, until V3 and V5 replace them.

1. **The ray-marched canopy** (`vegetation.Canopy`, `canopy_*.glsl`): per-cell ellipsoids marched in a tight
   shell over each chunk, the far look in `vegetation_far.glsl`. Off by default (`range` 0); its worker lift
   is no longer computed (`chunkgen`), so it needs that turned back on to be compared again. Measured at 400 m
   range: +0.5-0.6 ms over forest, +3.2 ms among the trees; blobby up close, grid-aligned.
2. **The instanced trees** (`terrain.forest`, `terrain.trees`, `tree_*.glsl`; assets in
   `world_builder.datafiles`): blue-noise placement in the workers, instanced meshes to 200 m, point-sprite
   impostors (nearest of 8 x 8 hemi-octahedral frames, no depth) to 3.5 km. Measured: +0.9-1.0 ms over forest,
   +2.5 ms among the trees, ~2.5 ms CPU per LOD pass; dark, flat, sparse from above.

What the review says is wrong with 2 (each point is a part below): thousands of near meshes where X-Plane
fills the near view with "a few hundred or so complex 3D trees" [XP12]; impostors without frame blending,
depth or perspective [SS4 s.31-32, UE-IMP]; light reckoned per vertex; nothing on the ground past the
impostors that matches them (IL-2's "bubble", [FINS]); low-poly game assets seen from above.

## The target: Serious Sam 4's stack

Serious Sam 4 layers its vegetation in subsystems: bare terrain, forest imposters, near trees, misc plants and
crops, grass blades (geometry), flowers (alpha-keyed) [SS4 s.23-28]. Its terrain is 128 x 128 km, viewed to
~50-100 km [SS4 s.2-4], which is our problem at a smaller scale. The plan builds the same stack, with
Bruneton's lighting at every scale and Mantler's canopy as the alternative middle layer. Bruneton & Neyret's
two representations map onto it directly: z-fields for the nearer trees (one camera-facing quad per tree,
reconstructed from precomputed depth views), shader-maps for the farther ones (a shader on the terrain,
modulated by a coverage map of the trees as tiny disks), with a seamless transition between them [BN12 §4,
§7]:

| Layer | From | Range (HIGH, to tune) |
|---|---|---|
| Terrain forest material (far forest, forest floor) | [SS4 s.21], [BN12 §6] shader-maps | everywhere; alone where a tree is under ~3 pixels [BN12 §4] |
| Forest imposters (views with depth, perspective-corrected) | [SS4 s.30-32], [BN12 §5] z-fields, [UE-IMP] | ~100 m to where a tree is ~3 pixels |
| Canopy heightfield (alternative to imposters, from the ray-marched shell) | [Mantler §3-4] | ~100 m to 2-3 km |
| Near trees (meshes) | [SS4 s.23], [XP12] | to ~100 m, a few hundred |
| Misc plants, crops, rocks (instanced meshes) | [SS4 s.36] | to ~150 m |
| Grass blades (geometry) | [SS4 s.34] | to ~50-80 m |
| Flowers (alpha-keyed, reshaped placeholders) | [SS4 s.35] | to ~50 m |
| Roads, fields and patches (masks over vegetation and material) | [SS4 s.37-38] | n/a (data) |

**Budget (HIGH):** all of it at most **+1.0 ms GPU** in the worst flight pose (forest_low: 150 m over forest),
**+0.5 ms** more at ground level for grass and flowers, **under 0.15 ms CPU a frame** on average and **no LOD
pass over 2 ms**. Each part has its own share, below; a part over its share stops for a decision.

---

## Part V1. Placement and data, Serious Sam 4's way

What Serious Sam 4 stores and decides (all [SS4]):

- **Coarse painted data plus rules.** Pre-created data at 32 m (elevation, a "far texture" albedo, vegetation
  density), ~128-200 MB for the whole 128 x 128 km terrain [s.9]; finer detail procedural; edits stored in
  quadtrees down to 25 cm [s.15]; vegetation density has a default (0.28) where unpainted [s.15].
- **Density with elevation, slope and convexity** controls the materials (and through them grass and
  flowers), the misc plants' distributions and the forest's (tree probability): "trees not spawned on steep
  slopes, or in drylands"; convexity is the second derivative of elevation, to spawn on hilltops (> 0) or in
  valleys (< 0) [s.22]. Huge diverse forests are made by "spray-painting" density roughly [s.22].
- **Material rules.** Elevation regions (seaside, grasslands, grass and ground, bare rocks, snow) cut by "a
  simple noisy ramp" [s.18]; 2-8 materials per region picked by noise x slope ramp x vegetation ramp, the
  highest chance winning per pixel [s.19]; rocky material slopes matched to the elevation's angle of repose
  [s.19]; coarse materials mixed by a height-based transition, alpha x noise x height [s.42].
- **Forest:** a jittered grid, ~8 m between trees, in 1 km blocks of ~16 000 trees; one tile ~128 x 128 texels,
  each sample one tree: offset, rotation, tree type, hue [s.29].

For WorldBuilder:

1. **The vegetation density field.** Today the biome weights (`biomes.weights`: temperature, humidity,
   elevation, slope) stand for it. Add **convexity** (the elevation's Laplacian at a few metres, computed in
   `chunkgen` beside the slope) and a **per-planet density modulation** at low frequency (procedural
   value noise). **Decided (2026-11-12): density is procedural, with optional stamps, as the biomes**: a
   level file may paint density over the procedural field the way it paints biomes today
   (`terrain.stamps.Paint`, applied after the weights in `chunkgen`), Serious Sam 4's "spray-painting"
   [SS4 s.22] as an override, never required. Output per vertex: forest
   density in [0, 1] (a new byte in the chunk's weights, or replacing the forest weight's meaning).
2. **Trees on a jittered grid, as Serious Sam 4.** Replace the blue-noise tile (`forest.poisson_tile`) with a
   jittered grid of ~6-8 m cells keyed by global cell index (as the ray-marched canopy's cells), one sample
   per cell: kept by density, then offset, rotation, type, hue [SS4 s.29]. It is cheaper to evaluate, the same
   on the CPU and the GPU (one hash per cell), and its regularity is broken by the jitter, the density's
   clumping and the per-cell type. (The trial's blue noise is kept as an option if the grid shows.)
3. **Species by rules, MSFS-style tables.** Per biome: species, instances per hectare, spawn ratios [MSFS];
   per species: slope, convexity and altitude ramps [SS4 s.22]. Outerra's distribution from elevation, slope
   and curvature with fractal noise is the same idea ([Outerra forum](https://forum.outerra.com/index.php?topic=15.0)).
4. **Blocks, not lists.** Per chunk (finest levels) the samples as a texture of cells (SS4's ~128 x 128 per
   km) rather than a list of plants: the impostors' and the canopy's shaders read it directly.

Budget: placement in the workers only (< 10 ms a chunk); no per-pass Python over plants.

## Part V2. The terrain forest material (the far forest), Serious Sam 4 + Bruneton's shader-maps

Serious Sam 4: "Far forest: splatted into terrain at distance beyond forest imposters"; "Forest floor: each
region has separate material sets for inside the forest" [SS4 s.21]; texture cascades near (1-4 m tiles),
mid (100+ m tiles), and one far texture for the whole terrain [SS4 s.20].

Bruneton & Neyret's **shader-map** is the far forest done with lighting that holds at every scale [BN12 §6]:

- **What it must reproduce** [BN12 §1]: view-dependent reflectance (brighter looking away from the sun: lit
  parts seen), slope-dependent reflectance, the opposition hotspot opposite the sun, silverlining (backlit
  crown edges brighter), and sky occlusion (tree bases and interiors darker, the ground darker under trees).
- **The model** extends Strahler et al.'s remote-sensing model [BN12 §3]: a pixel is split into sunlit ground,
  shadowed ground, sunlit crowns and shadowed crowns, fractions kg, kḡ, kt, kt̄, and L = kg Lg + kḡ Lḡ + kt Lt +
  kt̄ Lt̄ [Eq. 1]. A vertical scaling by 2/h (h = crown height / radius) and a rotation make every case spherical
  trees on flat ground, so the fractions depend on 6 parameters only: θv, θl, φ (view, sun and their azimuth
  in transformed space), λ (tree density), A (mean crown area) and β [§3, Eqs. 2-6]. With a Poisson
  distribution the visible ground is exp(-λ P(θv)), P = A sec θv [§3].
- **Precomputed tables, not formulas** (any tree shape, any isotropic distribution): render a forest patch
  of z-field trees, measure the fractions for many angles and densities: G(θv, θl, φ, λ) gives kg and kg + kḡ
  [Eqs. 14-15], T(θv, θl, φ, λ) the sunlit share of the visible crowns [Eq. 16], D(λ) the ground's ambient
  occlusion [Eq. 18], E(θv, λ) the crowns' [Eq. 21]. **131 kB in all**; 8 samples of λ, 16 of each angle;
  ~10 minutes on a GTX 470 [§9, §10].
- **Radiances** [§6]: sunlit ground Īg ≈ r̄ max(n̄·l, 0) Lsun [Eq. 17]; sky on the ground J̄g ≈ (1 + n̄·uz)/2
  D(λ) r̄ π Lsky [Eq. 19]; sunlit crowns Īt = ρ p(v, l) F(v, l) Lsun [Eq. 20], F the hotspot term
  1 - a1 [1 - exp(-a2 acos(v·l))] [Eq. 8]; sky on crowns J̄t [Eq. 21]. The pixel: L = kg Īg + (kg + kḡ) J̄g
  + (1 - kg - kḡ) (kt / (kt + kt̄) Īt + J̄t) [Eq. 13].
- **The coverage map** Γ(x): per terrain tile, the share of ground covered by trees in top view, made by
  drawing one disk per tree (or the density times A when a disk is under 1/16 of a texel), plus one density
  channel per species [§4]. Using Γ/A in place of the density near the transition makes the disks shaded as
  trees and the ground between them as ground, so the shader-map matches the trees it replaces [§7].
- **The switch**: tile data turns from tree seeds into coverage tiles where a tree's apparent size is about
  3 pixels (1 would avoid popping; 3 saves trees) [§4].
- **Cost:** 0.6 ms for the shader-map in a 1024 x 768 frame on a GTX 470 [§9].

Mantler et al. warn that a plain diffuse model makes canopies look "plastic", and ambient light with soft
accumulated shadows was their compromise [Mantler §4.5]: Bruneton's model is the principled answer.

For WorldBuilder (the statistical far look in `vegetation_far.glsl` is a crude shader-map, with a Poisson
exp(-λP) and no tables):

1. **The coverage map:** per terrain chunk (finest levels, a layer of a texture array, as Bruneton's cache
   [§4]), the trees of V1 drawn as disks of their crown area, plus density per species; coarser chunks the
   density times the mean crown area. Made in the workers or on the GPU when the chunk is made.
2. **The tables G, T, D, E** baked once per species group (or once for all) from our trees by
   `scripts/bake_tree_impostors.py`'s successor: render patches of imposter trees (V3) at the 8 x 16^3
   samples and measure the four fractions. 131 kB.
3. **The shader** in the terrain's fragment stage (forest weight > 0): Eqs. 13-21, the hotspot term, the sky
   occlusion; our sun, sky and shadows from `terrain_light.glsl`.
4. **The forest floor** as its own terrain material layer under and between the trees [SS4 s.21], and the
   ground darkened near trees by Bruneton's contrast term Δ(x) = 1 - a3 (Γ(x) - Λh(x) A) [§5, Eq. 10].
5. **The transition** (V3's imposters fade into it): §7's three parts, below in V3.

Budget: +0.15 ms (Bruneton's 0.6 ms on a 2010 GPU at 1024 x 768 [§9]; the terrain shader's share was
0.0-0.2 ms in the trials).

## Part V3. Forest imposters, Serious Sam 4 + Bruneton's z-fields

Serious Sam 4 [SS4 s.30-32]:

- **Imposter blocks:** pregenerated geometry, a grid of 128 x 128 standalone quads, each tree 2 triangles, per
  block a table of per-type parameters [s.30].
- **The choice:** "8-direction crossed geometry or viewer facing? Crossed geometry is slower. Viewer facing
  lacks parallax. So... use 8-direction textures. Face quad to viewer. Compensate for perspective in pixel
  shader (see Interior Mapping by Joost van Dongen). Blend 2 nearest directions." [s.31]
- **Textures** in a texture array for the different trees: albedo, subsurface, normal, AO, depth; the depth
  texture "for proper shadows" [s.32].

Bruneton & Neyret's near representation, the **z-field** [BN12 §5], is the same idea with depth and light:

- **Precomputed views** per tree model: 4 values per texel, min and max depth (z, z̄), ambient occlusion δ
  and opacity α (premultiplied); depths from the plane through the tree's centre facing the view; AO for an
  isolated tree on flat ground. Theirs: **181 views** over the upper hemisphere, **256 x 256 RGBA8** each,
  trees up to 256 pixels without artefacts, **45 MB per model** [§5].
- **One camera-facing quad per tree** covering its bounding box [§5, Fig. 4], from cached seeds (position,
  species, rotation, scale, colour), culled and expanded in a geometry shader [§4, §8].
- **Shape reconstruction** (Max et al. interpolate 3-4 depth views and ghost): take the 3 views nearest the
  view direction with barycentric weights; intersect the view ray with the plane through the centre; project
  into the 3 views, interpolate depth over opacity to step along the ray; once more from the new point; the
  result sets the fragment's depth. Two iterations for the view ray, one for the light ray from that point
  toward the sun [§5, steps 1-5].
- **Light** [§5]: foliage as a participating medium, sun It = ρ p(v, l) exp(-τ |pl - pv|) F(v, l) V(pl) Lsun
  [Eq. 7] (self-shadows, silverlining, the hotspot); sky Jt = the tree's own AO times a hole term for the
  other trees, 1 / (1 + d² Λh / 4) with d the depth below the canopy top, times (1 + n·uz)/2 [Eq. 9]; shadows
  from cascaded shadow maps drawn with the light step only, storing opacity for soft shadows [§5].
- **Transparency** without sorting: alpha-to-coverage, with a different coverage mask per tree (a table
  of masks per alpha, picked by the tree's seed) so n overlapping trees give 1 - (1 - α)^n and the opacity
  matches the shader-map's at the transition [§8].
- **The transition** to the shader-map [§7]: the z-field's light terms are forced toward the shader-map's
  averages with distance, ν = smoothstep(0, 0.8, s / smax) [Eqs. 22-25]; the trees' opacity is faded out over
  0.8-1 of smax while the ground under them takes the forest radiance [Eqs. 26-27]; Γ/A in the shader-map
  (V2).
- **Cost:** ~180 000 z-field trees in a 1024 x 768 frame: 18 ms on a GTX 470 (10.6 ms on a GTX 580), plus 4.4
  ms (1.4) for the shadow maps [§9]. Too dear as is for our budget; the views, the reconstruction and the
  light are what we take, at fewer views and texels.
- No wind with z-fields: the data is too big to animate [§10] (the user's choice anyway).

Octahedral impostors (MSFS draws all its vegetation so [MSFS]; Fortnite its trees, upper hemisphere [UE-IMP],
8 x 8 frames for small trees and 12 x 12 for the main pines, per the
[Unity forum](https://discussions.unity.com/t/hlod-streaming-octahedral-impostors-using-megacity-subscenes/742223))
cover views from above, which Serious Sam 4's 8 side directions (a ground-level game [SS4 s.5]) do not.

For WorldBuilder:

1. **Views:** a hemi-octahedral grid (8 x 8 small trees, 12 x 12 the main ones: 64-144 views where Bruneton
   uses 181 [BN12 §5]), not 8 side directions, for the flight's views from above; the **3 nearest views
   blended** with barycentric weights [BN12 §5] (Brucks' octahedral triangle [UE-IMP]; SS4 blends 2 [s.31]).
2. **Perspective compensation** per pixel from the baked depth: Bruneton's two-step reconstruction [BN12 §5]
   (the per-pixel ray against depth that SS4 cites from Interior Mapping [vD08]) gives the true hit, so
   parallax and contact with the ground are right, and its depth is written. The light step toward the sun
   gives the self-shadow (Eq. 7).
3. **Bake:** albedo, normal, min and max depth, AO and opacity [BN12 §5], plus translucency (SS4's
   subsurface [s.32]; X-Plane's in a normal map's blue channel [XP12]), per species, in texture arrays at
   64-128 px a view (Bruneton's 256 x 256 x 181 = 45 MB a tree [§5] is out of our budget), compressed.
4. **Geometry:** quads, not point sprites (trees are not square), from per-chunk blocks of standalone quads
   expanded in the vertex shader [SS4 s.30]; the plant data read from V1's cell textures. One draw per chunk,
   one shared state; frustum-culled with the chunk.
5. **Light:** Bruneton's Eqs. 7-9 per pixel (cheap: the light step's length, the hotspot term, the AO and the
   hole term), with our sun, sky and shadows reckoned once per tree in the vertex shader (four vertices), not
   per mesh vertex.
6. **Transparency:** alpha-to-coverage with per-tree masks [BN12 §8] where the leaves' alpha is soft; the
   alpha test otherwise.
7. **The transition to V2** exactly as [BN12 §7] (Eqs. 22-27), at an apparent size of ~3 pixels [§4].
8. **Assets:** the bake works for any model; better assets (a few thousand triangles a tree, as X-Plane's
   1-4k [XP12]) are a separate decision for the user. The current packs are re-baked meanwhile.

Budget: +0.5 ms over forest (forest_low), +0.2 ms at 800 m.

## Part V4. The canopy heightfield, Mantler (the ray-marched shell, reworked)

Mantler et al. render "a special vegetation heightfield ... on top of the terrain thus providing parallax and
occlusions like a real forest canopy" [Mantler §1]:

- **Canopy proxy:** a generic, tileable patch of canopy height (theirs from a LIDAR scan of a forest),
  multiplied by the vegetation cover map and added to the terrain's height; non-repeating by Wang tiles;
  several proxies for different forests (deciduous, coniferous, bushes) [§3.2]. 16-bit heights, 0.75-1.5 m
  horizontal resolution [§3.2].
- **Rays:** a coarse hull of the terrain, raised to enclose the canopy (in the vertex shader), starts the
  rays; tiles with vertical "skirts" avoid missed intersections [§4.1, §4.3]. A fixed step against the
  heightfield, clipped where nothing is hit [§4.2].
- **Acceleration:** per texel "safety zones", the highest height within radii (they used 4 and 32 texels),
  let a ray skip ahead safely [§4.4].
- **Light:** ambient plus shadows accumulated along a ray toward the sun with growing steps, softer and more
  natural than hard ray-cast shadows [§4.5].
- **Cost (2006, ATI X1900, 1280 x 1024):** 30 intersection + 10 shadow steps acceptable at 35 fps; one 16-bit
  texture is ~4x faster than two 8-bit ones [§3.2, §5.2].

Bruneton & Neyret note the limits of the canopy heightfield: easier user control, but "only far views are
supported", and its shadows and AO from the canopy height "cannot account for self-shadowing inside trees"
[BN12 §2, on Mantler et al.]. Hence V4 as an alternative to compare, not the default.

The removed trial's shell was this algorithm already (a hull with skirts over each chunk, a conservative top,
exit tests: `canopy_*.glsl`, in git history before 2026-11-12), with ellipsoids per cell instead of a
heightfield. For WorldBuilder:

1. **Replace the ellipsoids by a canopy proxy texture baked from our trees** (the imposters' top-view depth,
   V3, composited on V1's jittered grid into a tileable canopy height tile per forest type), times the forest
   density.
2. **Safety zones** (two max-filtered levels) in place of per-cell tests: a ray costs its steps, not its
   cells.
3. **Ambient + accumulated soft shadow** light, as [Mantler §4.5], with the colour from V2's baked table.
4. **Compare with V3** at 100 m to 3 km, A/B: look (grazing views over hills, from above) and cost. Both stay
   selectable (`VegetationSettings`: a middle-layer choice).

Budget: as V3 (+0.5 ms over forest); whichever wins the look within it becomes HIGH's default.

## Part V5. Near trees

- A few hundred trees as meshes near the camera, the billboard (here: the imposter) for the rest [XP12];
  additive mesh LODs, the underside of the branches dropped past 100 m [XP12].
- In WorldBuilder: meshes to ~80-120 m at HIGH, capped (the nearest N); per-tree light (V3.5); CPU frustum
  culling per chunk batch (X-Plane culls on the GPU with compute and indirect draws [XP12]; Panda 1.10 has
  compute shaders but no indirect draws, so the GPU path is later work).

Budget: +0.3 ms among the trees (forest_edge).

## Part V6. Misc plants, crops and rocks

Serious Sam 4: "procedurally instanced meshes": bigger flowers, bushes in the plains and forests, forest
overgrowth (logs, roots), crops [SS4 s.36]; "doodads", plants and rocks by riversides, at the edges of crop
fields [s.37-38]. Its tiles stop at 0.5 m and add geometry from photo-scanned height textures [s.10, s.16-17]
(rocks as terrain detail rather than objects, which is a separate project).

For WorldBuilder: the instanced meshes of the trial (`terrain.trees`), limited to the near range, with V1's
rules: bushes and overgrowth in forest and grassland, dry scrub in desert, rocks by slope and convexity
(bare rock regions, scree under cliffs), the forest pack's 9 rocks first.

Budget: +0.15 ms near the ground.

## Part V7. Grass blades

Serious Sam 4 [SS4 s.34]: true geometry, not alpha-keyed, ~7 triangles per blade; pre-batched geometry in grids,
different density grids overlapped; the vertex shader samples a splatted mask and colour texture; blades
perfectly match the grass material's position on the floor; albedo coloured from the floor; a per-blade
ambient shadow on the floor.

For WorldBuilder: a few pre-batched grid patches (blades at fixed offsets, a few densities) drawn around the
camera, each blade's height and colour from the terrain at its foot (the grass weight and the ground's
albedo, as SS4); the patches' density falls with distance (the coarser grid alone farther). No wind (the
user's choice). The ground under the grass darkened per blade (SS4's ambient shadow).

Budget: +0.3 ms at ground level, nothing above ~80 m.

## Part V8. Flowers

Serious Sam 4 [SS4 s.35]: like grass but far less dense, alpha-keyed; pre-batched placeholder geometry (up to
32 triangles a model, a triangle soup), the real geometry in separate buffers; the distribution from the
materials, splatted to a cached texture; the vertex shader reshapes each placeholder into a real flower, its
index from the splatted distribution.

For WorldBuilder: the same reshaping trick for flowers and small plants (a handful of models in one buffer),
the distribution from the grass and meadow weights. Assets needed (the user's decision).

Budget: +0.1 ms at ground level.

## Part V9. Roads, fields and patches

Serious Sam 4 [SS4 s.37-38]: splines (roads, rivers, trails) and patches (fields, lakes, yards) fade the
vegetation map out over a transition border, force their own material, level or smooth the elevation, add a
custom height texture (wheel ruts, crop rows), and place doodads at their edges.

For WorldBuilder: patches as level-file entries beside the existing stamps and paints (`terrain.stamps`): a
mask that clears or sets vegetation density and a material, crop rows as a height texture; roads later. This is
data work more than rendering; it waits for a level that needs it.

## Part V10. Shadows and the other passes

- Tree shadows: from the imposters' depth [SS4 s.32]; near trees into the casters' map or a vegetation shadow
  map out to ~500 m (X-Plane's default shadow distance [XP12]); far, the shader-map's shadow (V2).
- The sea's reflection, the clouds' depth passes, the objects' casters: the imposters' depth-only variant.
- docs/source/vegetation.md, the CHANGELOG, the backlog (collision with trees: X-Plane has none [XP12]; wind).

---

## Order and stops

V1 (data) → **V2 (far material)** → **V3 (imposters)** and **V4 (canopy heightfield)**, compared → V5 (near
trees) → V6 → V7 → V8 → V9 → V10. A stop for the user's look after V2, after the V3/V4 comparison, and after
each of V5-V8. The far material first: it is what a flight game shows most, and the cheapest.

## Decisions on data (2026-11-12)

1. **Density:** procedural, with optional stamps (paints) in level files, as the biomes (V1).
2. **Assets:** start with the current packs (forest, birch, snowy pines); better models come later as a
   data swap. To keep that swap cheap, everything that depends on a model is baked or declared, never tuned in
   code:
   - **Baked from the model** (re-run `scripts/build_tree_assets.py` and the bake script): the imposter views
     (colour, normal, depths, AO, opacity, V3), Bruneton's tables G, T, D, E (measured on patches of the
     actual trees [BN12 §6]), the far colours (V2), the canopy proxy tile (V4), crown radius, height and
     aspect h per asset.
   - **Declared per pack or asset** in a small metadata file beside the models: the colour space of its
     textures (the current packs store linear albedo; most store sRGB), translucency, foliage density τ,
     mesh LOD distances, and which species group it belongs to.
   - **Declared per species group** (`forest.SPECIES`): biome densities and spawn ratios, slope, convexity and
     altitude ramps, scale ranges. Spacing is derived from the crown sizes (baked), not set by hand.
   - What a new model can still change by hand: the per-species densities if crowns are much wider or
     narrower, and the near-mesh range if its triangle count is far from today's (300-650 for the near
     conifers). Both are a line each in data.
