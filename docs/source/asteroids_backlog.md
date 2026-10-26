# Asteroid fields backlog

Improvements found but not made, so the fields could move on (October 2026).
The design is in [Asteroid fields](asteroids.md).

## Rock shapes

The prototypes are potatoes: an ellipsoid, lumpy noise, faint ridges and
craters, with the fine detail in the normal map only. That is enough for now.
Ways to make them better:

- **Sharp ridges.** Ridged noise has V-shaped crests that interpolating
  between vertices cannot follow at any resolution: at the first default
  amplitude (0.05 of the radius), every level of detail stayed about 0.05 off
  the finest, which made the intermediate levels nearly useless. The default
  is down to 0.02. To get crests back, soften the noise's absolute value (a
  smooth `sqrt(x² + ε²)`), or move the ridges into the normal map only.
- **Silhouettes.** The finest mesh (32 quads per face side) is still about
  0.025 of the radius off the full detail: 5 m on a 200 m rock, 50 m on a
  2 km one. The normal map hides it in the shading, not in the outline. The
  64-quad level (ULTRA) halves it. Rocks from 200 m up get the terrain's
  level of detail near the camera (phase 6).
- **Shape variety.** Real asteroids include contact binaries (two lobes),
  flattened spinning tops and angular fragments. The radius function could
  take a second lobe (the union of two ellipsoids as a radius function),
  facets (a few planes clipping the ellipsoid), and a per-rock choice of
  family.
- **Generation time.** About 180 ms per prototype on one core: 32 prototypes
  take about 2 s across the workers at level load. The vertex normals
  evaluate the shape five times per vertex; the finest level could reuse
  neighbouring vertices instead (as the normal map does), and the radius
  function could go native.
- **Normal map seams.** Each face's normal map is clamped at its edge, so
  filtering stops at face seams: a faint line can show on a very close
  rock. A cube-map array (GL 4.0) or a one-texel border copied from the
  neighbouring faces would remove it.
- **Tumbling from the shape.** `Tumble.ratio` is a distribution;
  "from_shape" (I1/I3 - 1 from each prototype's ellipsoid axes) would tie
  tumbling to how elongated each rock is.
- **Unique rocks.** Every rock is one of a few prototypes, turned and
  scaled. Phase 6's terrain rocks add per-rock detail up close; instanced
  rocks could vary too, through a per-instance offset into a noise in the
  shader.
- **Overhangs, arches and caves.** A radius function cannot have them: these
  are the hero rocks (designed for in the plan, not built).

## Drawing

- **A native visible pass.** The per-frame pass (size on screen, frusta,
  level of detail, grouping) is numpy: about 0.75 ms for the `cube`
  preset's 3,400 rocks, and it grows with the rocks in range. A ring at
  tens of thousands of rocks wants it in C (`native/field.c`, as the plan
  had it), or the pass every other frame for the far classes.
- **Partial table uploads.** A change uploads the whole instance table
  (64 bytes a slot): nothing in the `cube` preset, where the whole field is
  in range, but a ring streams cells in as the camera flies. Panda 1.10 has
  no sub-image upload; one texture per size class, or a ring of row blocks,
  would bound each upload.
- **Box windows.** Each class holds a box of cells around the camera, about
  twice the cells of the sphere of its range. Holding only the cells the
  sphere touches halves the rocks held in an open field.
- **Texture seams.** Textures follow each cube face's grid, so their
  pattern turns at the face edges of every rock; triplanar mapping in the
  body frame would hide it (needed anyway for hero rocks).
