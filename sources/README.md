# Sources

The references behind [VEGETATION_PLAN.md](../VEGETATION_PLAN.md), kept as downloaded (November 2026).
Third-party material, for reading only (their own licences apply): nothing here is shipped or built.

## Papers and presentations

| File | Reference | What it gives us |
|---|---|---|
| [croteam_gdc2019_4_million_acres_serious_sam_4.pdf](croteam_gdc2019_4_million_acres_serious_sam_4.pdf) | A. Ladavac (Croteam), *Four Million Acres, Seriously: GPU-Based Procedural Terrains in Serious Sam 4: Planet Badass*, GDC 2019 Advanced Graphics Techniques Tutorial. <https://gpuopen.com/gdc-presentations/2019/gdc-2019-agtd2-4-million-acres-serious-sam-4.pdf> | The whole vegetation stack: density maps and rules, material rules, forest on a jittered grid, imposter blocks, 8-direction imposters with perspective compensation, far forest splatted into the terrain, grass blades, flowers, misc plants, roads and patches |
| [mantler2006_interactive_landscape_visualization.pdf](mantler2006_interactive_landscape_visualization.pdf) | S. Mantler, S. Jeschke, M. Wimmer, *Interactive Landscape Visualization Using GPU Ray Casting*, TU Wien, 2006. <https://www.cg.tuwien.ac.at/research/publications/2006/Mantler-06-landscape/Mantler-06-landscape-paper.pdf> | A forest canopy as a heightfield over the terrain (a tileable canopy proxy times a cover map), ray cast per pixel from a coarse hull with skirts, safety-zone acceleration, ambient light and accumulated soft shadows |
| [bruneton_neyret2012_forests.pdf](bruneton_neyret2012_forests.pdf) | E. Bruneton, F. Neyret, *Real-time Realistic Rendering and Lighting of Forests*, Computer Graphics Forum 31(2), 2012, pp. 373-382, DOI 10.1111/j.1467-8659.2012.03016.x. HAL hal-00650120: <https://inria.hal.science/hal-00650120v1> (downloaded by hand) | Near trees as z-fields (precomputed depth views, one camera-facing quad, iterative reconstruction, light as a participating medium); far trees as shader-maps (Strahler's four fractions from precomputed tables, a coverage map); the seamless transition; per-tree coverage masks; costs |
| [vandongen2008_interior_mapping.pdf](vandongen2008_interior_mapping.pdf) | J. van Dongen, *Interior Mapping: A new technique for rendering realistic buildings*, CGI 2008. <https://www.proun-game.com/Oogst3D/CODING/InteriorMapping/InteriorMapping.pdf> | The per-pixel ray against planes that Serious Sam 4 cites for its imposters' perspective compensation |

**Not saved:** the paper's appendix on view interpolation
(<https://artis.inrialpes.fr/Publications/2011/BN11a/more_details.pdf>), unreachable; the paper's §5 describes
the reconstruction fully.

**Not available:** the GDC 2019 talk's video (GDC Vault, members only:
<https://www.gdcvault.com/play/1026349/Advanced-Graphics-Techniques-Tutorial-Four>). The slides above carry
its content.

## Web pages (`web/`)

| File | Page | What it gives us |
|---|---|---|
| [croteam_gdcvault_four_million_acres.html](web/croteam_gdcvault_four_million_acres.html) | The GDC Vault abstract of the Serious Sam 4 talk | Scope: 128 x 128 km terrain tessellated to 1 cm; GPU-based procedural forests and shrubs, grass, rocks, fine details |
| [croteam_wccftech_interview.html](web/croteam_wccftech_interview.html) | Wccftech, interview with A. Ladavac and D. Lucic, 21 April 2019 | Context only (why a terrain engine) |
| [croteam_pcgamer_behind_the_scenes.html](web/croteam_pcgamer_behind_the_scenes.html) | PC Gamer, *Behind the scenes with Serious Sam 4* | Context only |
| [xplane_next_generation_trees.html](web/xplane_next_generation_trees.html) | Laminar Research, *Next generation trees and OpenGL*, 2021 | Millions of trees, a few hundred 3D near, a camera-facing billboard for the rest; GPU compute culling, indirect draws, per-tree LOD on the GPU |
| [xplane_building_custom_3d_trees.html](web/xplane_building_custom_3d_trees.html) | Laminar Research, *Building custom 3-D trees* (X-Plane 12) | Additive mesh LODs (the underside dropped past 100 m), 500 m shadow distance, 1-4k triangles a tree, translucency in the normal map's blue channel |
| [msfs_sdk_species_definitions.html](web/msfs_sdk_species_definitions.html) | MSFS SDK, *Species definitions* | Vegetation drawn as octahedral impostors; species per biome by instances per hectare and spawn ratios |
| [unreal_impostor_baker.html](web/unreal_impostor_baker.html) | Epic, *Impostor Baker plugin* | Fortnite's trees as upper-hemisphere octahedral impostors (without Nanite) |
| [guerrilla_hzd_gpu_placement.html](web/guerrilla_hzd_gpu_placement.html) | Guerrilla, *GPU-based procedural placement in Horizon Zero Dawn* (GDC 2017) | All vegetation placed on the GPU at run time from painted density maps and rules; deterministic, locally stable |
| [falconbms_fins_3d_forests.html](web/falconbms_fins_3d_forests.html) | Falcon BMS forum, *Concept: "Fins" for 3D forests* | IL-2's far forests as stacked textured planes over the ground |

Cited but not saved (their pages are rendered by script, or forum posts only): R. Brucks, *Octahedral
Impostors* (<https://shaderbits.com/blog/octahedral-impostors>); the Fortnite frame counts, 8 x 8 small
trees, 12 x 12 the main pines (Unity forum,
<https://discussions.unity.com/t/hlod-streaming-octahedral-impostors-using-megacity-subscenes/742223>);
Outerra's tree distribution and densities (<https://forum.outerra.com/index.php?topic=15.0>); IL-2's tree
"bubble" against its forest texture (<https://forum.il2sturmovik.com/topic/61283-improved-forest-settings>);
Ghost of Tsushima's far foliage as an authored texture (secondary sources; talk:
<https://gdcvault.com/play/1027214/contactUs>).
