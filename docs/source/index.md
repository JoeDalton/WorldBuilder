# Welcome to the WorldBuilder documentation!

WorldBuilder procedurally builds worlds in Panda3D: cube-sphere planetary
terrain seen from orbit to the ground, single-scattering atmospheres, a
procedural sky (stars, nebulae, the sun), a curved reflective ocean, and
volumetric billboard clouds (a flat field, or over a whole planet, following
its weather). It started inside
[SpaceFlight](https://github.com/JoeDalton/SpaceFlight), which uses it for its
ocean and clouds.

These pages are developer documentation: how each system is organised and how
it works.

- [A whole world](world.md): `World` builds every part a game draws around
  its camera (bodies, air, clouds, ocean, sky, sun, lights) from the game's
  looks, and updates them all every frame.
- [Terrain](terrain.md): the planet's layers, the cube-sphere and its LOD,
  precision and depth, several bodies, and the demo.
- [Atmosphere, sky and planet clouds](atmosphere.md): the scattering model,
  the star field and the clouds over a whole planet, with their weather.
- [Ocean](ocean.md): the reflective ocean and its swell.
- [Clouds](clouds.md): billboard clouds, their density fields and shading.
- [Quality settings](quality.md): every cost/quality knob, the LOW to ULTRA
  presets, and how a game's settings menu reads and saves them.
- [Performance](performance.md): where the time goes, measured, and what was
  done about it; the [performance backlog](performance_backlog.md) lists the
  savings found but not yet made.

The per-class API is in the auto-generated [code reference](apidocs/index.rst).

```{toctree}
:maxdepth: 2
:caption: Contents

world
terrain
atmosphere
ocean
clouds
quality
performance
performance_backlog
object_lighting
apidocs/index
```
