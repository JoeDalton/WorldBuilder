
❯ Big task ahead : rebase the current branch on main. Don't drop any feature.
  Ask any question you need.
  At first, you can let the "marched clouds" feature aside and concentrate on
  the billboards. We'll tackle the marched clouds later

● I'll start by surveying how far the branch and main have diverged.

  Ran 1 shell command (ctrl+o to expand)

● Local main is stale; origin/main has moved. Let me compare against
  origin/main.

  Ran 2 shell commands (ctrl+o to expand)

● Main picked up three squash-merges (#42, #44, #46) that heavily rework the
  clouds code, including deleting clouds/field.py, which our branch also
  modifies. Next I'll check each commit's scope and where it overlaps with the
  branch.

  Ran 2 shell commands (ctrl+o to expand)

● Before deciding on an approach I need to understand how main's new cloud
  architecture relates to what the branch's cloud-shadow work was built on.

  Searched for 1 pattern, ran 2 shell commands (ctrl+o to expand)

● Now the shape of main's cloud module, vs the branch's.

  Ran 5 shell commands (ctrl+o to expand)

● Let me check the test and render tooling available before asking questions.

  Listed 1 directory (ctrl+o to expand)

● Main has moved to b825304 with the cloud code rebuilt, so the rebase is a real
  port of the cloud changes rather than a conflict fix. A few decisions on that
  port are left for you.

● User answered Claude's questions:
  ⎿  · How should the 11 branch commits land on origin/main (b825304)? The cloud
     conflicts recur in most of them: field.py deleted, sphere.py,
     cloud_field.glsl, world.py, demo_terrain.py and terrain_shots.py rewritten
     on main. → Replay all 11 (Recommended)
     · With genera on, the low clouds are two CloudVolumes (stratus under
     convective), each with its own field and weather map. The branch's cloud
     shadow marches one layer. Which of them should shade the ground, sea and
     objects? → Both families (Recommended)
     · The cirrus sheet is new on main, so the branch never shadowed it. What
     should it take part in? → Receives only (Recommended)
     · When should I stop for you to look at it in the demo? → At the end
     (Recommended)

● I'm replaying all 11 commits with both genera casting shadows and cirrus only
  receiving, and I'll pause for your visual check at the end. Meanwhile I'm
  kicking off a background baseline test run on origin/main in a separate
  worktree to see which failures were already there before the rebase.

  Ran 3 shell commands (ctrl+o to expand)

● Bash(git config rerere.enabled; git config merge.conflictstyle; git rebase
      origin/main 2>&1 | tail -30)
  ⎿  Rebasing (1/11)Auto-merging CHANGELOG.md
     CONFLICT (content): Merge conflict in CHANGELOG.md
     Auto-merging docs/source/atmosphere.md
     … +25 lines (ctrl+o to expand)
  ⎿  Deleted BACKLOG.md (+0 -70)
  ⎿  Updated src/world_builder/clouds/field.py (+0 -46)
  ⎿  Deleted src/world_builder/clouds/shaders/cloud.vert (+0 -120)
  ⎿  Deleted src/world_builder/clouds/shaders/cloud_shadow_map.frag (+0
     -30)
  ⎿  Deleted src/world_builder/clouds/shaders/cloud_shadow_map.vert (+0 -8)
  ⎿  … 121 more files changed (part of the diff is unavailable)
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● Commit 1/11 (Eclipses) conflicts. I'll redo the conflict markers in diff3
  style so the base is visible, then work through the files one at a time.

● Bash(git config merge.conflictstyle zdiff3 && git checkout --conflict=zdiff3
      CHANGELOG.md docs/source/atmosphere.md scripts/demo_terrain.py
      scripts/terrain_shots.py…)
  ⎿  Recreated 5 merge conflicts
     diff --git a/src/world_builder/clouds/field.py b/src/world_builder/clo
     uds/field.py
     … +24 lines (ctrl+o to expand)
  ⎿  Updated CHANGELOG.md (+6 -4)
  ⎿  Updated docs/source/atmosphere.md (+6 -2)
  ⎿  Updated scripts/demo_terrain.py (+7 -2)
  ⎿  Updated scripts/terrain_shots.py (+23 -6)
  ⎿  Updated src/world_builder/clouds/shaders/cloud.frag (+43 -2)
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● The flat field is gone on main, so its one-line change has nowhere to go. I'm
  deleting the file and will make sure main's new cloud shaders bind the shadow
  defaults instead.

● Bash(git rm -q src/world_builder/clouds/field.py && git diff 8375966~1 8375966
      -- src/world_builder/clouds/sphere.py src/world_builder/clouds/shaders/
      src/world_buil…)
  ⎿  diff --git a/src/world_builder/clouds/shaders/cloud.frag b/src/world_b
     uilder/clouds/shaders/cloud.frag
     index d189b5b..21bdfc4 100644
     … +126 lines (ctrl+o to expand)
  ⎿  1 file changed (binary, mode only or too large to show)

● Now the conflicted cloud.frag:

  Ran 1 shell command (ctrl+o to expand)

● That lighting block moved from cloud.frag into cloud_shading.glsl on main. Let
  me read both main-side files.

  Ran 1 shell command (ctrl+o to expand)

● Main's cloud.frag now handles only the planet, and its deck lighting runs
  through planetCloudLight, which the raymarch also uses. I'll add the eclipse
  in the deck branch of cloud.frag and leave the march code alone for now.

● Bash(git checkout --ours src/world_builder/clouds/shaders/cloud.frag && python
      - <<'EOF'
      p='src/world_builder/clouds/shaders/cloud.frag'…)
  ⎿  Updated 1 path from the index
     diff --git a/src/world_builder/clouds/shaders/cloud_sphere.vert b/src/
     world_builder/clouds/shaders/cloud_sphere.vert
     … +33 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/shaders/cloud.frag (+2 -86)
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● The cirrus.py and march.py shaders on main pull in shadows.glsl through
  atmosphere.glsl, so they have to bind the shadow defaults or Panda won't draw
  them. I'm checking every caller of the atmosphere bind_defaults next.

  Ran 2 shell commands (ctrl+o to expand)

● Bash(python - <<'EOF'
      import re…)
  ⎿  src/world_builder/clouds/cirrus.py:102:from world_builder.atmosphere i
     mport Atmospheres, bind_defaults
     src/world_builder/clouds/cirrus.py:103:from world_builder.clouds.cloud
     … +19 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/cirrus.py (+1 -0)
      758              )
      759          )
      760          bind_defaults(root)
      761 +        bind_shadow_defaults(root)                            
      762          if atmospheres is not None:
      763              atmospheres.bind(root)
      764          self._sun = PTA_LVecBase3f.empty_array(1)
  ⎿  Updated src/world_builder/clouds/march.py (+1 -0)
      795          """Give a node every input the march and the deck read
            (kept in step
      796          by :meth:`_set_inputs`)."""
      797          bind_defaults(node)
      798 +        bind_shadow_defaults(node)                            
      799          if self._atmospheres is not None:
      800              self._atmospheres.bind(node)
      801          node.set_shader_inputs(**self._inputs)

● Bash(sed -i '119i from world_builder.shadows import bind_defaults as
      bind_shadow_defaults' src/world_builder/clouds/cirrus.py && sed -i '135i
      from world_builder.shad…)  ⎿  )
     from world_builder.depth import REFLECTION_CAMERA_MASK
     from world_builder.shadows import bind_defaults as bind_shadow_default
     … +10 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/cirrus.py (+1 -0)
      116      stream_vortices,
      117  )
      118  from world_builder.depth import REFLECTION_CAMERA_MASK
      119 +from world_builder.shadows import bind_defaults as bind_shadow
          +_defaults                                                     
      120  from world_builder.sky.sky import SKY_BIN, ensure_sky_bin
      121  from world_builder.terrain.planet import PlanetSpec
      122  
  ⎿  Updated src/world_builder/clouds/march.py (+1 -0)
      132      REFLECTION_CAMERA_MASK,
      133      SplitDepthCameras,
      134  )
      135 +from world_builder.shadows import bind_defaults as bind_shadow
          +_defaults                                                     
      136  from world_builder.terrain.cubesphere import grid_for
      137  from world_builder.terrain.planet import PlanetSpec
      138  

● You chose "cirrus receives", so the Eclipses commit should also dim the
  cirrus. Let me see how its shaders light it.

  Ran 2 shell commands (ctrl+o to expand)

● Background command "cd /tmp/claude-1000/-home-gl-development-WorldBuilder/d412
4c88-f133-488e-aa3c-ec7b6e8c7853/scratchpad/main_wt && PYTHONPATH=src timeout
1800 python -m pytest tests/ -n auto -q -p no:cacheprovider 2>&1 | tail -30 >
../main_baseline.txt; tail -30 ../main_baseline.txt" completed (exit code 0)

  Read 1 file (ctrl+o to expand)

● The baseline on origin/main passes all 741 tests in 52 s. For the cirrus
  eclipse I need the main camera's position rather than the per-pass vEye, so
  I'm checking what the cirrus already knows about the camera.

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)

● The cirrus node sits at the planet's centre in render orientation, so the main
  camera's planet-frame position (cam in update) gives the eclipse what it
  needs. I'm adding it as a camPos uniform.

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/cirrus.py'…)
  ⎿  diff --git a/src/world_builder/world.py b/src/world_builder/world.py
     index 7ddc350..4a3faec 100644
     --- a/src/world_builder/world.py
     … +147 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/cirrus.py (+4 -0)
      764              atmospheres.bind(root)
      765          self._sun = PTA_LVecBase3f.empty_array(1)
      766          self._sun[0] = LVecBase3f(0.0, 0.0, 1.0)
      767 +        # The (main) camera, planet frame: what an eclipse is 
          +seen from.                                                    
      768 +        self._camera = PTA_LVecBase3f.empty_array(1)          
      769          o = spec.optics
      770          forward, backward, isotropic = phase_weights(o)
      771          root.set_shader_inputs(
     ...
      776              uMaxOpticalDepth=float(spec.max_optical_depth),
      777              uOwnAtmosphere=-1 if atmosphere is None else int(a
           tmosphere),
      778              sunDir=self._sun,
      779 +            camPos=self._camera,                              
      780              sunColor=Vec3(1.0, 1.0, 1.0),
      781              skyColor=Vec3(0.45, 0.6, 0.85),
      782              sunBrightness=float(sun_brightness),
     ...
      862              self._finish()
      863          cam = np.asarray(camera_world, dtype=np.float64) - sel
           f.position
      864          self.root.set_pos(*(np.asarray(camera_render, dtype=np
           .float64) - cam))
      865 +        self._camera[0] = LVecBase3f(*cam)                    
      866  
      867      def remove(self) -> None:
      868          """Take the cirrus off the scene."""
  ⎿  Updated src/world_builder/clouds/shaders/cirrus.vert (+6 -0)
       9  uniform mat4 p3d_ModelViewProjectionMatrix;
      10  uniform mat4 p3d_ModelViewMatrixInverse;
      11  uniform int uOwnAtmosphere;
      12 +uniform vec3 camPos;         // the main camera, planet frame (
         +eclipses)                                                      
      13  
      14  in vec4 p3d_Vertex;           // on the sheet, planet frame
      15  in vec2 p3d_MultiTexCoord0;   // in the atlas
     ...
      53          vTwilight = cloudTwilight(uOwnAtmosphere, p, sunDir, vS
          unTint);
      54          vSkyTint = smoothstep(-0.3, 0.1, dot(normalize(p), sunD
          ir));
      55      }
      56 +    // Other bodies eclipsing the sun (as cloud_sphere.vert).  
      57 +    float eclipse = eclipseVisibility(p - camPos, sunDir, uShad
         +owSelf);                                                       
      58 +    vSunTint *= eclipse;                                       
      59 +    vTwilight *= eclipse;                                      
      60 +    vSkyTint *= mix(ECLIPSE_SKY_FLOOR, 1.0, eclipse);          
      61  }

● That cirrus change is my own edit. Next, world.py auto-merged; let me check
  that it binds the shadows onto PlanetClouds correctly, and that the march's
  off-tree nodes are covered.

  Searched for 1 pattern (ctrl+o to expand)

● On main, body.clouds.root is the parent of each layer's root, so binding
  shadows there would be overridden by the layers' own defaults. I'm reading the
  rest of planet.py to find the right place to bind them.

  Ran 1 shell command (ctrl+o to expand)

● I'll add a bind_shadows(shadows, body) method to PlanetClouds that binds each
  layer's root and rebinds after a renderer change. First I'm checking
  everything the branch's final world.py binds to the clouds, so the API covers
  the later commits as well.

  Ran 1 shell command (ctrl+o to expand)

● The final branch attaches the eclipses, terrain shadow map and rings to the
  clouds' root. I'm adding a generic bind_inputs hook to PlanetClouds that
  applies that binding to each layer and to any rebuilt volume, and I'm leaving
  the march for a later phase.

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/planet.py'…)
  ⎿  57:from __future__ import annotations
     59:import math
     60:from collections.abc import Callable, Iterable, Sequence
     … +14 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/planet.py (+24 -0)
      24  
      25  from __future__ import annotations
      26  
      27 +from collections.abc import Callable                           
      28  from typing import TYPE_CHECKING, Any, ClassVar
      29  
      30  import numpy as np
     ...
      229          self._sky_light: float | None = None
      230          self._debug = 0
      231          self._half_res: HalfResClouds | None = None
      232 +        # What every layer's shader also takes (bind_inputs), 
          +again for a                                                   
      233 +        # new volume.                                         
      234 +        self._binders: list[Callable[[NodePath], None]] = []  
      235          self.volume = self._make_volume(settings)
      236          #: The genera's stratus, under the convective volume; 
           None = none
      237          self.stratus: CloudVolume | None = None
     ...
      310              if self._sky_light is not None:
      311                  self.volume.set_sky_light(self._sky_light)
      312              self.volume.set_debug(self._debug)
      313 +            self._bind_layer(self.volume)                     
      314              if self._half_res is not None and isinstance(self.
           volume, CloudVolume):
      315                  self._half_res.add(self.volume.root)
      316              if self.sheet is not None:
     ...
      319          for layer in self._layers():
      320              layer.set_settings(settings)
      321  
      322 +    def bind_inputs(self, bind: Callable[[NodePath], None]) ->
          + None:                                                        
      323 +        """Have every layer's shader take more inputs (what sh
          +ades the                                                      
      324 +        clouds: the eclipses, a body's shadow maps): *bind* is
          + called with                                                  
      325 +        each layer's node now, and with a new volume's on a ch
          +ange of                                                       
      326 +        renderer. Each layer binds its own defaults on its own
          + node, which a                                                
      327 +        binding on :attr:`root` would not override.           
      328 +                                                              
      329 +        :param bind: sets inputs on a node (e.g. ``Shadows.bin
          +d``)                                                          
      330 +        """                                                   
      331 +        self._binders.append(bind)                            
      332 +        for layer in self._layers():                          
      333 +            self._bind_layer(layer, (bind,))                  
      334 +                                                              
      335 +    def _bind_layer(self, layer, binders=None) -> None:       
     … 8 more lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/world.py (+3 -1)
      461              body = Body(look=look, terrain=terrain, atmosphere
           =index)
      462              if look.clouds is not None:
      463                  body.clouds = self._make_clouds(look, index)
      464 -                self.shadows.bind(body.clouds.root, len(self.b
          -odies))                                                       
      464 +                body.clouds.bind_inputs(                      
      465 +                    functools.partial(self.shadows.bind, body=
          +len(self.bodies))                                             
      466 +                )                                             
      467              if look.ocean is not None:
      468                  kwargs = {
      469                      name: getattr(look.ocean, name)

● Bash(sed -i '59s/^import math$/import functools\nimport math/'
      src/world_builder/world.py && sed -n 57,62p src/world_builder/world.py &&
      git diff --name-only --diff-…)
  ⎿  from __future__ import annotations

     import functools
     … +8 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/world.py (+1 -0)
      56  
      57  from __future__ import annotations
      58  
      59 +import functools                                               
      60  import math
      61  from collections.abc import Callable, Iterable, Sequence
      62  from dataclasses import dataclass, field

● Now the remaining conflicts in this commit: CHANGELOG, atmosphere.md, and the
  two scripts.

  Ran 1 shell command (ctrl+o to expand)

● CHANGELOG hunks where both sides only add entries resolve to "main's, then the
  branch's". I'll script that for empty-base hunks and handle the rest by hand.

● Bash(mkdir -p /tmp/claude-1000/-home-gl-development-WorldBuilder/d4124c88-f133
      -488e-aa3c-ec7b6e8c7853/scratchpad/tools && cat >
      /tmp/claude-1000/-home-gl-development…)
  ⎿  CHANGELOG.md remaining 0
  ⎿  Updated CHANGELOG.md (+0 -8)
       9  
      10  ### Added
      11  
      12 -<<<<<<< ours                                                   
      12  - Clouds by genus (`world_builder.clouds.genera`,
      13    docs/source/clouds.md#genera; #45): stratus, and cumulus grow
          ing with the
      14    convective depth through congestus to cumulonimbus, drawn by 
          billboards.
     ...
      101      `cirrus_dusk` poses.
      102    - Its cost is not measured yet: see docs/source/performance_
           backlog.md,
      103      "Cirrus".
      105 -||||||| base                                                  
      106 -=======                                                       
      104  - `world_builder.shadows`: what hides the sun from a point, fo
           r every
      105    receiver (docs/source/shadows.md). Its first term is the ecl
           ipse: each
      106    body hides the sun's disc for the others, with a physical pe
           numbra
     ...
      119  - `scripts/terrain_shots.py`: a pose's sun can be aimed at a b
           ody
      120    (`AimedSun`), and four eclipse poses: `eclipse_ground`, `ecl
           ipse_total`,
      121    `eclipse_orbit`, `lunar_eclipse`. (#13)
      125 ->>>>>>> theirs                                                
      122  - `world_builder.gravity.Gravity` and `World.gravity_field(g_c
           onstant)`: the
      123    bodies' raw gravity vector at many points at once, from poin
           t masses
      124    (`g = -G sum m (x - p) / |x - p|^3`, capped inside a body at
            its surface
     ...
      216  
      217  ### Changed
      218  
      223 -<<<<<<< ours                                                  
      219  - A planet's clouds are one part, `world_builder.clouds.planet
           .PlanetClouds`,
      220    with two layers named by how they are drawn:
      221    `clouds.sphere.SphericalClouds` is now `CloudVolume` (the lo
           w clouds:
     ...
      266    desert landform (sand seas, canyons by depth, mesas by relie
           f), and the demos (`demo_terrain.py`, `demo_world.py`,
      267    `terrain_shots.py`) take `--planet PRESET`, `--seed N` and
      268    `--start LAT LON`.
      274 -||||||| base                                                  
      275 -=======                                                       
      269  - The terrain demo's moon is 21,000 km out (was 10,000 km): as
            wide in the
      270    sky as the sun, so it can eclipse it. (#13)
      271  - `sky.SUN_ANGULAR_RADIUS` moves to `world_builder.shadows` (`
           sky` still
      272    imports it). (#13)
      280 ->>>>>>> theirs                                                
      273  - Asteroid fields without thresholds (#14):
     … 2 more lines (ctrl+o to expand)

  Ran 3 shell commands (ctrl+o to expand)

● CHANGELOG is fine (that joined line is main's own). Next, atmosphere.md:

  Ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='docs/source/atmosphere.md'…)
  ⎿  570:- **Shadows on the ground:** clouds do not shade the terrain (plan
     ned:
     571-  [Shadows](shadows.md)); the cirrus does not shade the cumulus ei
     … +20 lines (ctrl+o to expand)
  ⎿  Updated docs/source/atmosphere.md (+4 -14)
      567  ### Not yet
      568  
      569  - **Wind:** the clouds are static.
      570 -<<<<<<< ours                                                  
      571 -- **Shadows on the ground:** clouds do not shade the terrain; 
          -the cirrus does                                               
      572 -  not shade the cumulus either.                               
      570 +- **Shadows on the ground:** clouds do not shade the terrain (
          +planned:                                                      
      571 +  [Shadows](shadows.md)); the cirrus does not shade the cumulu
          +s either.                                                     
      572 +  Eclipses by other bodies already darken the air, the ground 
          +and the                                                       
      573 +  clouds.                                                     
      574  - **One cumulus-like type per planet** (cumulus in the demo), 
           with the cirrus
      575    above it; the planet field could take one per slab (the weat
           her map could
      575 -||||||| base                                                  
      576 -- **Shadows on the ground:** clouds do not shade the terrain. 
      577 -- **One cloud type per planet** (cumulus in the demo); the fla
          -t field mixes                                                 
      578 -  types, the planet field could take one per slab (the weather
          - map could                                                    
      579 -=======                                                       
      580 -- **Shadows on the ground:** clouds do not shade the terrain (
          -planned:                                                      
      581 -  [Shadows](shadows.md)). Eclipses by other bodies already dar
          -ken the air,                                                  
      582 -  the ground and the clouds.                                  
      583 -- **One cloud type per planet** (cumulus in the demo); the fla
          -t field mixes                                                 
      584 -  types, the planet field could take one per slab (the weather
          - map could                                                    
      585 ->>>>>>> theirs                                                
      576    then say which).
      577  - **Flying through the cirrus:** from inside its slab it is a 
           veil fading
      578    near the camera, with no volume.

● Bash(python - <<'EOF'
      p='scripts/demo_terrain.py'…)
  ⎿  diff --git a/scripts/demo_terrain.py b/scripts/demo_terrain.py
     index 68bdb7f..9c29e19 100644
     --- a/scripts/demo_terrain.py
     … +169 lines (ctrl+o to expand)
  ⎿  Updated scripts/demo_terrain.py (+3 -15)
       3  
       4  Flies a free camera around a small planetary system: a Kerbin-s
          ized planet
       5  (600 km) with hand-made stamps (world_builder/terrain/levels/de
          mo.yaml), its
       6 -<<<<<<< ours                                                   
       6  atmosphere, its clouds (cumulus over the whole planet, followin
          g its weather,
       7  and high cirrus streaming along its jets) and its sea, an airle
          ss sand moon
       9 -10,000 km away, and a procedural star field, from orbit down to
         - the ground of                                                 
      10 -either. Run it from an environment where world_builder is impor
         -table::                                                        
      11 -||||||| base                                                   
      12 -atmosphere, its clouds (cumulus over the whole planet, followin
         -g its                                                          
      13 -weather) and its sea, an airless sand moon 10,000 km away,     
      14 -and a procedural star field, from orbit down to the ground of e
         -ither. Run it                                                  
      15 -from an environment where world_builder is importable::        
      16 -=======                                                        
      17 -atmosphere, its clouds (cumulus over the whole planet, followin
         -g its                                                          
      18 -weather) and its sea, an airless sand moon 21,000 km away (as w
         -ide in the                                                     
      19 -sky as the sun: it can eclipse it), and a procedural star field
         -, from orbit                                                   
      20 -down to the ground of either. Run it from an environment where 
         -world_builder                                                  
      21 -is importable::                                                
      22 ->>>>>>> theirs                                                 
       8 +21,000 km away (as wide in the sky as the sun: it can eclipse i
         +t), and a                                                      
       9 +procedural star field, from orbit down to the ground of either.
         + Run it from                                                   
      10 +an environment where world_builder is importable::             
      11  
      12      python scripts/demo_terrain.py [--level FILE] [--planet PRE
          SET]
      13          [--seed N] [--start LAT LON] [--radius METRES]

  Searched for 1 pattern (ctrl+o to expand)

● Bash(sed -i '845s/.*/
      self.clouds.bind_inputs(functools.partial(self.shadows.bind, body=0))/'
      scripts/demo_terrain.py && sed -n 843,847p scripts/demo_terrain.…)
  ⎿              channel="clouds",
             )
             self.clouds.bind_inputs(functools.partial(self.shadows.bind, b
     … +7 lines (ctrl+o to expand)
  ⎿  Updated scripts/demo_terrain.py (+1 -1)
      842              settings=self.world.clouds,
      843              channel="clouds",
      844          )
      845 -        self.shadows.bind(self.clouds.root, 0)                
      845 +        self.clouds.bind_inputs(functools.partial(self.shadows
          +.bind, body=0))                                               
      846          self.clouds.set_debug(self._cloud_debug)
      847          self.clouds.set_sun(self._sun)
      848          if getattr(self, "depth", None) is not None:

● Bash(sed -i '109s/^import math$/import functools\nimport math/'
      scripts/demo_terrain.py && sed -n '/<<<<<<</,/>>>>>>>/p'
      scripts/terrain_shots.py)
  ⎿  <<<<<<< ours
     high (20 km up), swell / sea_surface (13 and 17 km up, inside the sea'
     s 12-18 km
     … +85 lines (ctrl+o to expand)
  ⎿  Updated scripts/demo_terrain.py (+1 -0)
      106  from __future__ import annotations
      107  
      108  import argparse
      109 +import functools                                              
      110  import math
      111  import sys
      112  from dataclasses import dataclass, replace

● Still on commit 1/11 (Eclipses): the shader, CHANGELOG and docs conflicts are
  resolved and the cirrus now takes the eclipse. I'm merging the shot poses in
  terrain_shots.py now.

  Ran 1 shell command (ctrl+o to expand)
● Bash(python - <<'EOF'
      import re…)
  ⎿      poses = {
             "ground": (
                 lat0,
     … +103 lines (ctrl+o to expand)
  ⎿  Updated scripts/terrain_shots.py (+7 -47)
      47  cloud billboards, the LOD pass and the median frame time.
      48  
      49  Poses: ground (the level's start), coast (2 km up over the refe
          rence point),
      50 -<<<<<<< ours                                                   
      50  high (20 km up), swell / sea_surface (13 and 17 km up, inside t
          he sea's 12-18 km
      51  fade from its near look to its flat far one), stratosphere (200
           km up),
      52  orbit (3 radii up, looking down), corner (over a cube corner, w
          here three faces
     ...
       64  be as dark as the sky), beach (40 m up, 300 m off the coast ne
           arest the
       65  reference point, looking at it: the sea's foam, colour by dept
           h and swell),
       66  open_sea (150 m over open water, looking along it: the swell a
           nd how it
       68 -flattens with distance). Every other pose has the sun at azimu
          -th 210, elevation                                             
       69 -35, as seen there.                                            
       70 -||||||| base                                                  
       71 -high (20 km up), swell / sea_surface (13 and 17 km up, inside 
          -the sea's                                                     
       72 -12-18 km fade from its near look to its flat far one), stratos
          -phere (200 km                                                 
       73 -up), orbit (3 radii up, looking down), corner (over a cube cor
          -ner, where three                                              
       74 -faces meet), lod (the LOD debug view from 20 km), in_clouds (i
          -nside the cloud                                               
       75 -deck, 2.7 km up), sunset (600 m up, looking west at the sun 3 
          -degrees up),                                                  
       76 -moonrise (the moon over the northern horizon, from the start a
          -rea),                                                         
       77 -moon_orbit (3 moon radii up, looking down), moon_ground (2 km 
          -over the                                                      
       78 -moon's sand), night_orbit (1.5 radii up, looking down at the t
          -erminator: the                                                
       79 -sun 60 degrees below the nadir's horizon), night_sea (3 m over
          - the sea, the                                                 
       80 -sun 8 degrees down: the sky's reflection must be as dark as th
          -e sky), beach                                                 
       81 -(40 m up, 300 m off the coast nearest the reference point, loo
          -king at it: the                                               
       82 -sea's foam, colour by depth and swell), open_sea (150 m over o
          -pen water,                                                    
       83 -looking along it: the swell and how it flattens with distance)
          -. Every                                                       
       84 -other pose has the sun at azimuth 210, elevation 35, as seen t
          -here.                                                         
       85 -=======                                                       
       86 -high (20 km up), swell / sea_surface (13 and 17 km up, inside 
          -the sea's                                                     
       87 -12-18 km fade from its near look to its flat far one), stratos
          -phere (200 km                                                 
       88 -up), orbit (3 radii up, looking down), corner (over a cube cor
          -ner, where three                                              
       89 -faces meet), lod (the LOD debug view from 20 km), in_clouds (i
          -nside the cloud                                               
       90 -deck, 2.7 km up), sunset (600 m up, looking west at the sun 3 
          -degrees up),                                                  
       91 -moonrise (the moon over the northern horizon, from the start a
          -rea),                                                         
       92 -moon_orbit (3 moon radii up, looking down), moon_ground (2 km 
          -over the                                                      
       93 -moon's sand), night_orbit (1.5 radii up, looking down at the t
          -erminator: the                                                
       94 -sun 60 degrees below the nadir's horizon), night_sea (3 m over
          - the sea, the                                                 
       95 -sun 8 degrees down: the sky's reflection must be as dark as th
          -e sky), beach                                                 
       96 -(40 m up, 300 m off the coast nearest the reference point, loo
          -king at it: the                                               
       97 -sea's foam, colour by depth and swell), open_sea (150 m over o
          -pen water,                                                    
     … 55 more lines (ctrl+o to expand)

  Ran 1 shell command (ctrl+o to expand)

● Let me update shadows.md for the cirrus receiver and the bind_inputs hook.

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='docs/source/shadows.md'…)
  ⎿  M  CHANGELOG.md
     M  docs/source/atmosphere.md
     M  docs/source/index.md
     … +2 lines (ctrl+o to expand)
  ⎿  Updated docs/source/shadows.md (+8 -2)
      46  
      47  `bind_defaults(node)` gives "nothing casts" inputs. `atmosphere
          .glsl`
      48  includes `shadows.glsl`, so every owner of a shader that sees t
          hrough air
      49 -(terrain, sky, planet and flat clouds) calls it next to the atm
         -osphere's                                                      
      50 -`bind_defaults`, and every shader links without a `Shadows`.   
      49 +(terrain, sky, each layer of a planet's clouds) calls it next t
         +o the                                                          
      50 +atmosphere's `bind_defaults`, and every shader links without a 
         +`Shadows`.                                                     
      51 +A planet's clouds take a `Shadows` through `PlanetClouds.bind_i
         +nputs(bind)`:                                                  
      52 +each layer binds its defaults on its own node, which a binding 
         +on the                                                         
      53 +part's root would not override, so the binding is applied to ev
         +ery layer's                                                    
      54 +node (and to a volume rebuilt on a change of renderer).        
      55  
      56  ## Eclipses
      57  
     ...
      82  - **Cloud billboards** (`cloud_sphere.vert`): per vertex, at th
          e
      83    billboard's centre, into `vSunTint` and `vTwilight`.
      84  - **Cloud deck** (`cloud.frag`): per fragment.
      85 +- **Cirrus** (`cirrus.vert`): per vertex, as the billboards, fr
         +om the main                                                    
      86 +  camera (`camPos`; the sea's mirror draws it from its own).   
      87  - **The air** (`atmoSegment`): each sample's direct sunlight, a
          t the
      88    sample. The sky darkens, and so does the aerial perspective o
          ver the
      89    ground.

● Before continuing, a test run on this commit:

  Ran 1 shell command (ctrl+o to expand)

● 752 pass, but the tests don't compile GLSL. I'll render a couple of shots to
  catch shader errors.

  Read 2 files, ran 3 shell commands (ctrl+o to expand)

● Both render without shader errors. In the total eclipse the nearby clouds and
  cirrus go dark and the distant ones stay lit. On to commit 2.

● Bash(GIT_EDITOR=true git rebase --continue 2>&1 | tail -20)
  ⎿  [detached HEAD 1c18318] Eclipses
      25 files changed, 1200 insertions(+), 61 deletions(-)
      create mode 100644 docs/source/shadows.md
     … +17 lines (ctrl+o to expand)
  ⎿  Updated CHANGELOG.md (+5 -0)
  ⎿  Updated scripts/demo_terrain.py (+101 -3)
  ⎿  Updated scripts/demo_world.py (+16 -2)
  ⎿  Updated scripts/flight_bench.py (+1 -0)
  ⎿  Updated scripts/terrain_shots.py (+22 -4)
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

  Ran 1 shell command (ctrl+o to expand)

● All unions, apart from the demo's docstring, which I merge by hand.

● Bash(T=/tmp/claude-1000/-home-gl-development-WorldBuilder/d4124c88-f133-488e-a
      a3c-ec7b6e8c7853/scratchpad/tools/union.py; python - <<'EOF'
      p='scripts/demo_terrain.py…)
  ⎿  scripts/demo_terrain.py remaining 0
     scripts/demo_world.py remaining 0
     scripts/terrain_shots.py remaining 0
     … +13 lines (ctrl+o to expand)
  ⎿  Updated scripts/demo_terrain.py (+2 -33)
       3  
       4  Flies a free camera around a small planetary system: a Kerbin-s
          ized planet
       5  (600 km) with hand-made stamps (world_builder/terrain/levels/de
          mo.yaml), its
       6 -<<<<<<< HEAD                                                   
       6  atmosphere, its clouds (cumulus over the whole planet, followin
          g its weather,
       8 -and high cirrus streaming along its jets) and its sea, an airle
         -ss sand moon                                                   
       7 +and high cirrus streaming along its jets) and its sea, a small 
         +ring of dust                                                   
       8 +and rocks in its equator's plane (1.5 to 2.1 radii out), an air
         +less sand moon                                                 
       9  21,000 km away (as wide in the sky as the sun: it can eclipse i
          t), and a
      10  procedural star field, from orbit down to the ground of either.
           Run it from
      11  an environment where world_builder is importable::
      12 -||||||| parent of 8386b42 (ring in the demo terrain)           
      13 -atmosphere, its clouds (cumulus over the whole planet, followin
         -g its                                                          
      14 -weather) and its sea, an airless sand moon 21,000 km away (as w
         -ide in the                                                     
      15 -sky as the sun: it can eclipse it), and a procedural star field
         -, from orbit                                                   
      16 -down to the ground of either. Run it from an environment where 
         -world_builder                                                  
      17 -is importable::                                                
      18 -=======                                                        
      19 -atmosphere, its clouds (cumulus over the whole planet, followin
         -g its                                                          
      20 -weather) and its sea, a small ring of dust and rocks in its equ
         -ator's                                                         
      21 -plane (1.5 to 2.1 radii out), an airless sand moon 21,000 km aw
         -ay (as wide                                                    
      22 -in the sky as the sun: it can eclipse it), and a procedural sta
         -r field,                                                       
      23 -from orbit down to the ground of either. Run it from an environ
         -ment where world_builder                                       
      24 -is importable::                                                
      25 ->>>>>>> 8386b42 (ring in the demo terrain)                     
      12  
      13      python scripts/demo_terrain.py [--level FILE] [--planet PRE
          SET]
      14          [--seed N] [--start LAT LON] [--radius METRES]
     ...
      40      --quality        every part's quality preset at start (defa
          ult high;
      41                       world_builder.settings.WorldSettings)
      42      --<part>-quality one part's preset, over --quality
      57 -<<<<<<< HEAD                                                   
      43      --no-clouds      no clouds at all (neither cumulus nor cirr
          us)
      59 -||||||| parent of 8386b42 (ring in the demo terrain)           
      44      --no-clouds      no clouds at all
      61 -=======                                                        
      62 -    --no-clouds      no clouds at all                          
      45      --no-ring        no ring around the planet
      64 ->>>>>>> 8386b42 (ring in the demo terrain)                     
      46  
     … 46 more lines (ctrl+o to expand)
  ⎿  Updated scripts/demo_world.py (+0 -8)
      148          clouds: bool = True,
      149          cloud_debug: int = 0,
      150          ocean_debug: int = 0,
      151 -<<<<<<< HEAD                                                  
      151          preset: str | None = None,
      152          seed: int | None = None,
      153          start: tuple[float, float] | None = None,
      155 -||||||| parent of 8386b42 (ring in the demo terrain)          
      156 -=======                                                       
      154          ring: bool = True,
      158 ->>>>>>> 8386b42 (ring in the demo terrain)                    
      155      ) -> None:
      156          ShowBase.__init__(self)
      157          self.setFrameRateMeter(True)
     ...
      314          moon,
      315          settings=settings_from_args(args),
      316          clouds=not args.no_clouds,
      321 -<<<<<<< HEAD                                                  
      317          preset=args.planet,
      318          seed=args.seed,
      319          start=args.start,
      325 -||||||| parent of 8386b42 (ring in the demo terrain)          
      326 -=======                                                       
      320          ring=not args.no_ring,
      328 ->>>>>>> 8386b42 (ring in the demo terrain)                    
      321      ).run()
      322  
      323  
  ⎿  Updated scripts/terrain_shots.py (+0 -8)
      191          lat, lon = (float(v[0]) for v in latlon(under[None, :]
           ))
      192          poses["eclipse_orbit"] = (lat, lon, 3.0 * radius, 0.0,
            -89.9, 0, 0)
      193          poses["lunar_eclipse"] = poses["moon_orbit"]
      194 -<<<<<<< HEAD                                                  
      194      # The cirrus (8-9 km) against the cumulus (2.2-3.2 km): un
           der both, between,
      195      # above both, and at dusk (looking west, toward the set su
           n).
      196      poses["cirrus_up"] = (lat0, lon0, start.get("alt", 800.0),
            0.0, 35.0, 0, 0)
      197      poses["cirrus_between"] = (ref_lat, ref_lon, 5000.0, 0.0, 
           3.0, 0, 0)
      198      poses["cirrus_above"] = (ref_lat, ref_lon, 15000.0, 0.0, -
           8.0, 0, 0)
      199      poses["cirrus_dusk"] = (ref_lat, ref_lon, 600.0, 90.0, 15.
           0, 0, 0)
      201 -||||||| parent of 8386b42 (ring in the demo terrain)          
      202 -=======                                                       
      200      if demo.ring is not None:
      201          poses["ring_ground"] = (ref_lat, ref_lon, 1500.0, 180.
           0, 12.0, 0, 0)
      202          poses["ring_orbit"] = (35.0, lon0, 6.0 * radius, 0.0, 
           -89.9, 0, 0)
      206 ->>>>>>> 8386b42 (ring in the demo terrain)                    
      203      # The terminator from orbit: the sun 60 degrees below the 
           nadir's horizon.
      204      poses["night_orbit"] = (lat0, lon0, 1.5 * radius, 0.0, -89
           .9, 0, 0)
      205      # At sea level by night: the sky's reflection must be as d
           ark as the sky.
     ...
      336          args.radius,
      337          settings=settings,
      338          clouds=not args.no_clouds,
      343 -<<<<<<< HEAD                                                  
      339          genera=args.genera,
      345 -||||||| parent of 8386b42 (ring in the demo terrain)          
      346 -=======                                                       
      340          ring=not args.no_ring,
      348 ->>>>>>> 8386b42 (ring in the demo terrain)                    
      341          cloud_debug=args.cloud_debug,
      342          ocean_debug=args.ocean_debug,
      343          preset=args.planet,

  Ran 1 shell command (ctrl+o to expand)

● Bash(sed -i '44d' scripts/demo_terrain.py && sed -n 40,47p
      scripts/demo_terrain.py && git diff HEAD -- scripts/demo_terrain.py | grep
      '^[-+]' | head -120)
  ⎿      --quality        every part's quality preset at start (default hig
     h;
                          world_builder.settings.WorldSettings)
     … +81 lines (ctrl+o to expand)
  ⎿  Updated scripts/demo_terrain.py (+0 -1)
      41                       world_builder.settings.WorldSettings)
      42      --<part>-quality one part's preset, over --quality
      43      --no-clouds      no clouds at all (neither cumulus nor cirr
          us)
      44 -    --no-clouds      no clouds at all                          
      44      --no-ring        no ring around the planet
      45  
      46  The default backend generates terrain in spawned worker process
          es, which

● Commit 2 only touches scripts and they compile, so on to commit 3.

● Bash(git add -A scripts && GIT_EDITOR=true git rebase --continue 2>&1 | tail
      -20)
  ⎿  Rebasing (3/11)Auto-merging CHANGELOG.md
     Auto-merging docs/source/atmosphere.md
     Auto-merging docs/source/object_lighting.md
     … +17 lines (ctrl+o to expand)
  ⎿  Updated CHANGELOG.md (+12 -3)
  ⎿  Updated docs/source/atmosphere.md (+2 -2)
  ⎿  Updated docs/source/object_lighting.md (+11 -6)
  ⎿  Updated docs/source/shadows.md (+25 -0)
  ⎿  Updated docs/source/world.md (+10 -1)
  ⎿  … 10 more files changed
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

  Ran 1 shell command (ctrl+o to expand)

● These just apply the branch's --no-ring → --ring flip on top of main's lines.
  Doing it by hand:

● Bash(python - <<'EOF'
      import re…)
  ⎿  scripts/demo_world.py:322:        ring=args.ring,
     scripts/terrain_shots.py:340:        ring=args.ring,
     scripts/flight_bench.py:351:        ring=args.ring,
     … +3 lines (ctrl+o to expand)
  ⎿  Updated scripts/demo_terrain.py (+5 -49)
       3  
       4  Flies a free camera around a small planetary system: a Kerbin-s
          ized planet
       5  (600 km) with hand-made stamps (world_builder/terrain/levels/de
          mo.yaml), its
       6 -<<<<<<< HEAD                                                   
       6  atmosphere, its clouds (cumulus over the whole planet, followin
          g its weather,
       8 -and high cirrus streaming along its jets) and its sea, a small 
         -ring of dust                                                   
       9 -and rocks in its equator's plane (1.5 to 2.1 radii out), an air
         -less sand moon                                                 
      10 -21,000 km away (as wide in the sky as the sun: it can eclipse i
         -t), and a                                                      
      11 -procedural star field, from orbit down to the ground of either.
         - Run it from                                                   
      12 -an environment where world_builder is importable::             
      13 -||||||| parent of 0dac844 (Game objects are shadowed)          
      14 -atmosphere, its clouds (cumulus over the whole planet, followin
         -g its                                                          
      15 -weather) and its sea, a small ring of dust and rocks in its equ
         -ator's                                                         
      16 -plane (1.5 to 2.1 radii out), an airless sand moon 21,000 km aw
         -ay (as wide                                                    
      17 -in the sky as the sun: it can eclipse it), and a procedural sta
         -r field,                                                       
      18 -from orbit down to the ground of either. Run it from an environ
         -ment where world_builder                                       
      19 -is importable::                                                
      20 -=======                                                        
      21 -atmosphere, its clouds (cumulus over the whole planet, followin
         -g its                                                          
      22 -weather) and its sea (and, with --ring, a small ring of dust an
         -d rocks in                                                     
      23 -its equator's plane, 1.5 to 2.1 radii out), an airless sand moo
         -n 21,000 km                                                    
      24 -away (as wide in the sky as the sun: it can eclipse it), and a 
         -procedural                                                     
      25 -star field, from orbit down to the ground of either. Run it fro
         -m an                                                           
      26 -environment where world_builder is importable::                
      27 ->>>>>>> 0dac844 (Game objects are shadowed)                    
       7 +and high cirrus streaming along its jets) and its sea (and, wit
         +h --ring, a                                                    
       8 +small ring of dust and rocks in its equator's plane, 1.5 to 2.1
         + radii out),                                                   
       9 +an airless sand moon 21,000 km away (as wide in the sky as the 
         +sun: it can                                                    
      10 +eclipse it), and a procedural star field, from orbit down to th
         +e ground of                                                    
      11 +either. Run it from an environment where world_builder is impor
         +table::                                                        
      12  
      13      python scripts/demo_terrain.py [--level FILE] [--planet PRE
          SET]
      14          [--seed N] [--start LAT LON] [--radius METRES]
     ...
      40      --quality        every part's quality preset at start (defa
          ult high;
      41                       world_builder.settings.WorldSettings)
      42      --<part>-quality one part's preset, over --quality
      59 -<<<<<<< HEAD                                                   
      43      --no-clouds      no clouds at all (neither cumulus nor cirr
          us)
      61 -    --no-ring        no ring around the planet                 
      62 -||||||| parent of 0dac844 (Game objects are shadowed)          
     … 61 more lines (ctrl+o to expand)
  ⎿  Updated scripts/demo_world.py (+0 -12)
      149          clouds: bool = True,
      150          cloud_debug: int = 0,
      151          ocean_debug: int = 0,
      152 -<<<<<<< HEAD                                                  
      152          preset: str | None = None,
      153          seed: int | None = None,
      154          start: tuple[float, float] | None = None,
      156 -        ring: bool = True,                                    
      157 -||||||| parent of 0dac844 (Game objects are shadowed)         
      158 -        ring: bool = True,                                    
      159 -=======                                                       
      155          ring: bool = False,
      161 ->>>>>>> 0dac844 (Game objects are shadowed)                   
      156      ) -> None:
      157          ShowBase.__init__(self)
      158          self.setFrameRateMeter(True)
     ...
      316          moon,
      317          settings=settings_from_args(args),
      318          clouds=not args.no_clouds,
      325 -<<<<<<< HEAD                                                  
      319          preset=args.planet,
      320          seed=args.seed,
      321          start=args.start,
      329 -        ring=not args.no_ring,                                
      330 -||||||| parent of 0dac844 (Game objects are shadowed)         
      331 -        ring=not args.no_ring,                                
      332 -=======                                                       
      322          ring=args.ring,
      334 ->>>>>>> 0dac844 (Game objects are shadowed)                   
      323      ).run()
      324  
      325  
  ⎿  Updated scripts/terrain_shots.py (+0 -6)
      336          args.radius,
      337          settings=settings,
      338          clouds=not args.no_clouds,
      339 -<<<<<<< HEAD                                                  
      339          genera=args.genera,
      341 -        ring=not args.no_ring,                                
      342 -||||||| parent of 0dac844 (Game objects are shadowed)         
      343 -        ring=not args.no_ring,                                
      344 -=======                                                       
      340          ring=args.ring,
      346 ->>>>>>> 0dac844 (Game objects are shadowed)                   
      341          cloud_debug=args.cloud_debug,
      342          ocean_debug=args.ocean_debug,
      343          preset=args.planet,

● Scripts compile. The core of commit 3 is world.py (lights), which auto-merged;
  let me check its cloud-related parts.

  Ran 1 shell command (ctrl+o to expand)

● Bash(git add -A scripts && timeout 900 python -m pytest tests/test_world.py
      tests/test_lights.py tests/test_atmosphere_sky.py tests/test_shadows.py -n
      auto -q -p no:…)
  ⎿  -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.htm
     l
     62 passed, 5 warnings in 5.65s
     … +24 lines (ctrl+o to expand)
  ⎿  Updated CHANGELOG.md (+15 -0)
  ⎿  Updated docs/source/atmosphere.md (+14 -0)
  ⎿  Updated docs/source/ocean.md (+4 -1)
  ⎿  Updated docs/source/performance_backlog.md (+23 -0)
  ⎿  Updated docs/source/quality.md (+6 -0)
  ⎿  … 21 more files changed
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● Commit 4/11, Cloud shadows, is the first real port. Let me read what it does
  to the cloud code.

  Ran 1 shell command (ctrl+o to expand)

● Now main's version of the sun march, which gained the depth and shelf:

  Ran 1 shell command (ctrl+o to expand)

● The sun march now runs through sunOpticalDepthAt, with separate fields and
  weather maps for each family. Next I'll check how volume layer parameters,
  wind and weather are stored and updated on main before designing the
  two-family shadow.

  Searched for 1 pattern (ctrl+o to expand)

● The branch's bind_shadow_inputs / sun_visibility auto-merged into sphere.py.
  Let me read the merged regions.

  Ran 2 shell commands (ctrl+o to expand)
● Main's planet clouds have no wind, so both families share one noise volume and
  offset. Shadows from both families will be built once at the end as
  uncommitted changes for you to review, and I'm moving on to the rest of the
  branch's cloud-shadow code.

  Ran 2 shell commands (ctrl+o to expand)

● The flat field held the "no clouds" textures and shadow defaults that the
  terrain and models import. I'm moving them into their own small module,
  clouds/shadow_defaults.py, which depends only on noise and weather so it
  avoids import cycles with the terrain.

  Ran 1 shell command (ctrl+o to expand)

● Main dropped uSpherical entirely, so the defaults lose it. Now the noise.py
  conflicts:

● Bash(git rm -q src/world_builder/clouds/field.py; grep -n
      '<<<<<<<\|>>>>>>>\|^=======$\|^||||||| '
      src/world_builder/clouds/noise.py)
  ⎿  21:<<<<<<< HEAD
     23:||||||| parent of ff11889 (Cloud shadows)
     24:=======
     26:>>>>>>> ff11889 (Cloud shadows)
  ⎿  1 file changed (binary, mode only or too large to show)

  Ran 1 shell command (ctrl+o to expand)

● The imports just need both. The CPU twin auto-merged, but it calls
  _slab_density, fbm and shadow_calibration, which main changed for depth and
  shelf. Let me compare with main's CPU sun march.

● Bash(python /tmp/claude-1000/-home-gl-development-WorldBuilder/d4124c88-f133-4
      88e-aa3c-ec7b6e8c7853/scratchpad/tools/union.py
      src/world_builder/clouds/noise.py --all…)
  ⎿  132:    def feature_size(self) -> tuple:
     178:def delta_eddington(asymmetry: float, applicability: float = 1.0)
     -> float:
     … +40 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/noise.py (+0 -4)
      18  
      19  from __future__ import annotations
      20  
      21 -<<<<<<< HEAD                                                   
      21  import math
      23 -||||||| parent of ff11889 (Cloud shadows)                      
      24 -=======                                                        
      22  from collections.abc import Callable
      26 ->>>>>>> ff11889 (Cloud shadows)                                
      23  from dataclasses import dataclass, replace
      24  
      25  import numpy as np

  Ran 1 shell command (ctrl+o to expand)

● I'm porting main's sun march to the GPU by generalising its optical-depth
  function so the branch's cloud shadow can build on it, with a matching CPU
  split in noise.py. Next I'll move on to the merged sphere.py, starting with
  the deck-light job.

  Ran 1 shell command (ctrl+o to expand)

● sphere.py merged cleanly, but its bind_shadow_inputs still sets uSpherical,
  which main removed, and sun_visibility passes only the weather shift. I'm
  searching for any other references to uSpherical.

  Searched for 1 pattern (ctrl+o to expand)

● Now I'll write the new defaults module and point the terrain at it.

● Write(src/world_builder/clouds/shadow_defaults.py)
  ⎿  Wrote 61 lines to src/world_builder/clouds/shadow_defaults.py
       1 """
       2 "No clouds overhead": the inputs a shader that receives the clou
         ds' shadow
       3 (cloud_field.glsl's ``cloudShadow``: the terrain's) needs before
          any clouds
       4 are bound to it, or without any. :meth:`sphere.CloudVolume.bind_
         shadow_inputs`
       5 overrides them.
       6 """
       7
     … +54 lines (ctrl+o to expand)

● Bash(sed -i 's/from world_builder.clouds.field import
      cloud_shadow_defaults/from world_builder.clouds.shadow_defaults import
      cloud_shadow_defaults/' src/world_builde…)
  ⎿  Traceback (most recent call last):
       File "<string>", line 1, in <module>
       File "/home/gl/development/WorldBuilder/src/world_builder/world.py",
     … +23 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/terrain/terrain.py (+1 -1)
      63  
      64  from world_builder import SHALLOW_WATER_COLOR, SHALLOW_WATER_DE
          PTH_M
      65  from world_builder.atmosphere import Atmospheres, bind_defaults
      66 -from world_builder.clouds.field import cloud_shadow_defaults   
      66 +from world_builder.clouds.shadow_defaults import cloud_shadow_d
         +efaults                                                        
      67  from world_builder.depth import PASS_TAG, REFLECTION_CAMERA_MAS
          K
      68  from world_builder.ocean.ocean import bind_sea_defaults
      69  from world_builder.ocean.swell import Swell
  ⎿  Updated src/world_builder/clouds/sphere.py (+0 -1)
      1268              noiseSize=float(NOISE_SIZE),
      1269              noiseOffset=Vec3(0, 0, 0),
      1270              deckLightCosMin=DECK_LIGHT_COS_MIN,
      1271 -            uSpherical=1,                                    
      1271              uCloudDebug=0,
      1272              uCloudShadowLayer=self._shadow_layer,
      1273          )

● Now world.py:

  Ran 1 shell command (ctrl+o to expand)

● body.clouds is now a PlanetClouds, so it must forward bind_shadow_inputs and
  sun_visibility to its volume. On a renderer change, the nodes either rebind
  the new volume or reset to "no clouds" when that volume is a march.

● Bash(python - <<'EOF'
      p='src/world_builder/world.py'…)
  ⎿  ok
  ⎿  Updated src/world_builder/world.py (+0 -6)
      464              body = Body(look=look, terrain=terrain, atmosphere
           =index)
      465              if look.clouds is not None:
      466                  body.clouds = self._make_clouds(look, index)
      467 -<<<<<<< HEAD                                                  
      467                  body.clouds.bind_inputs(
      468                      functools.partial(self.shadows.bind, body=
           len(self.bodies))
      469                  )
      471 -||||||| parent of ff11889 (Cloud shadows)                     
      472 -                self.shadows.bind(body.clouds.root, len(self.b
          -odies))                                                       
      473 -=======                                                       
      474 -                self.shadows.bind(body.clouds.root, len(self.b
          -odies))                                                       
      470                  body.clouds.bind_shadow_inputs(terrain.root)  
           # their shadow
      476 ->>>>>>> ff11889 (Cloud shadows)                               
      471              if look.ocean is not None:
      472                  kwargs = {
      473                      name: getattr(look.ocean, name)
  ⎿  Updated src/world_builder/clouds/planet.py (+42 -0)
      36  from world_builder.clouds.cloud import PRESETS, CloudSettings, 
          CloudType
      37  from world_builder.clouds.genera import CONVECTIVE, STRATIFORM,
           GeneraSpec
      38  from world_builder.clouds.march import CloudMarch
      39 +from world_builder.clouds.shadow_defaults import cloud_shadow_d
         +efaults                                                        
      40  from world_builder.clouds.sphere import CLOUD_BIN, CloudVolume
      41  from world_builder.clouds.weather import WeatherSpec
      42  from world_builder.depth import HIDE_FROM_CLOUD_PASS
     ...
      233          # What every layer's shader also takes (bind_inputs), 
           again for a
      234          # new volume.
      235          self._binders: list[Callable[[NodePath], None]] = []
      236 +        # The nodes the clouds shade (bind_shadow_inputs), aga
          +in for a new                                                  
      237 +        # volume.                                             
      238 +        self._shadow_nodes: list[NodePath] = []               
      239          self.volume = self._make_volume(settings)
      240          #: The genera's stratus, under the convective volume; 
           None = none
      241          self.stratus: CloudVolume | None = None
     ...
      315                  self.volume.set_sky_light(self._sky_light)
      316              self.volume.set_debug(self._debug)
      317              self._bind_layer(self.volume)
      318 +            self._shade(self._shadow_nodes)                   
      319              if self._half_res is not None and isinstance(self.
           volume, CloudVolume):
      320                  self._half_res.add(self.volume.root)
      321              if self.sheet is not None:
     ...
      343          for bind in self._binders if binders is None else bind
           ers:
      344              bind(layer.root)
      345  
      346 +    def bind_shadow_inputs(self, node: NodePath) -> None:     
      347 +        """Let the shaders on a node (the planet's terrain) da
          +rken the sun                                                  
      348 +        under the clouds: the volume's shadow                 
      349 +        (:meth:`sphere.CloudVolume.bind_shadow_inputs`), kept 
          +across a change                                               
      350 +        of renderer. A raymarched volume casts none yet.      
      351 +                                                              
      352 +        :param node: the node whose shaders include cloud_fiel
          +d.glsl's                                                      
      353 +            ``cloudShadow``                                   
      354 +        """                                                   
      355 +        self._shadow_nodes.append(node)                       
      356 +        self._shade([node])                                   
      357 +                                                              
      358 +    def _shade(self, nodes: list[NodePath]) -> None:          
      359 +        if isinstance(self.volume, CloudVolume):              
     … 26 more lines (ctrl+o to expand)

● Now the GLSL conflicts in cloud_field.glsl and cloud.frag:

  Ran 1 shell command (ctrl+o to expand)

● cloud.frag takes main's side. For cloud_field.glsl I'll start from main's file
  and apply the port on it, rather than splice hunks.

● Bash(git checkout --ours src/world_builder/clouds/shaders/cloud_field.glsl
      src/world_builder/clouds/shaders/cloud.frag && python - <<'EOF'
      p='src/world_builder/cloud…)
  ⎿  Updated 2 paths from the index
  ⎿  Updated src/world_builder/clouds/shaders/cloud.frag (+0 -10)
  ⎿  Updated src/world_builder/clouds/shaders/cloud_field.glsl (+42 -67)
  ⎿  Updated src/world_builder/clouds/shaders/cloud_shading.glsl (+1 -5)
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● The GLSL is ported. Now the CPU twin in noise.py: one _march_density helper
  shared by the sun march and the chord march, so the shadow keeps the depth,
  cell octave and shelf as the GPU does.

  Ran 2 shell commands (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/noise.py'…)
  ⎿  def _march_density(
         spec: DensityField,
         h: np.ndarray,
     … +25 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/noise.py (+56 -14)
      877      return h, fbm(volume, coords, spec, octaves=2, octave_mean
           =0.0)
      878  
      879  
      880 -def _sun_depth(                                               
      880 +def _march_density(                                           
      881      spec: DensityField,
      882 -    optics: CloudOptics,                                      
      882      h: np.ndarray,
      883      pair: np.ndarray,
      884      shift: ArrayLike,
     ...
      886      depth: ArrayLike = 0.0,
      887      shelf: tuple | None = None,
      888  ) -> np.ndarray:
      890 -    """:func:`sun_optical_depth_on_sphere` from :func:`_sun_ma
          -rch_samples`."""                                              
      889 +    """The density at a march's samples (..., steps), from the
          +ir slab                                                       
      890 +    heights and two-octave sums: the column's calibrated onto 
          +the full                                                      
      891 +    field, the shelf's its own (cloud_field.glsl's            
      892 +    ``slabOpticalDepthCell``). *shift*, *depth* and *shelf* ar
          +e the                                                         
      893 +    march's, one per march (...)."""                          
      894      gain, bias = shadow_calibration(spec, octave_mean)
      895      shift = np.asarray(shift, dtype=np.float64)[..., None]
      896      depth = np.asarray(depth, dtype=np.float64)[..., None]
     ...
      901          value = pair + octave_mean * (w[2] + w[3]) + cell + sp
           ec.shelf_shift
      902          shelf = (spread, value)
      903      raw = pair * gain + shift + bias
      901 -    density = _slab_density(spec, h, raw, depth, shelf).sum(ax
          -is=-1)                                                        
      902 -    return density * optics.sun_step_length * spec.density * o
          -ptics.shadow_strength                                         
      904 +    return _slab_density(spec, h, raw, depth, shelf)          
      905  
      906  
      907 +def _sun_depth(                                               
      908 +    spec: DensityField,                                       
      909 +    optics: CloudOptics,                                      
      910 +    h: np.ndarray,                                            
      911 +    pair: np.ndarray,                                         
      912 +    shift: ArrayLike,                                         
      913 +    octave_mean: float,                                       
      914 +    depth: ArrayLike = 0.0,                                   
      915 +    shelf: tuple | None = None,                               
      916 +) -> np.ndarray:                                              
     … 82 more lines (ctrl+o to expand)

● The CPU twin is now ported, but the GLSL comment cites a nonexistent
  noise.slab_optical_depth. I'm redirecting it to the real twins and wiring
  depth and spread into CloudVolume.sun_visibility.

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/shaders/cloud_field.glsl'…)
  ⎿  909:        self.weather: np.ndarray | None = None
     1359:        self.weather = self._weather_job.result()
     1124:        self._deck_field = deck.field_of(field)
     1314:                self._deck_field,
  ⎿  Updated src/world_builder/clouds/shaders/cloud_field.glsl (+2 -1)
      343  // through the two lowest octaves calibrated onto the full fie
           ld
      344  // (noise.shadow_calibration). The weather varies over kilomet
           res: given
      345  // once (weatherAt at start) for every step, and the cell octa
           ve (eight
      346 -// kilometres, cellShift at start) too. noise.slab_optical_dep
          -th on the CPU.                                                
      346 +// kilometres, cellShift at start) too. noise._march_density o
          +n the CPU                                                     
      347 +// (sun_optical_depth_on_sphere, cloud_shadow_on_sphere).     
      348  float slabOpticalDepthCell(Field f, vec3 start, vec3 dir, floa
           t stepLength,
      349                             int steps, vec3 weather, float cell
           ) {
      350      vec3 increment = dir * stepLength;
  ⎿  Updated src/world_builder/clouds/sphere.py (+10 -2)
      1302          cells = [tuple(c) for c in np.round(flat / SHADE_CELL
            ).astype(np.int64)]
      1303          missing = [i for i, cell in enumerate(cells) if cell 
            not in self._shades]
      1304          if missing:
      1305 -            plane = self.weather[..., 1]                     
      1305 +            weather = self.weather                           
      1306 +                                                             
      1307 +            def channel(k):                                  
      1308 +                if weather.shape[-1] <= k:                   
      1309 +                    return None                              
      1310 +                return lambda p: sample_cube(weather[..., k],
           + p)                                                          
      1311 +                                                             
      1312              depth = cloud_shadow_on_sphere(
      1313                  _noise(self._seed),
      1314                  self._deck_field,
     ...
      1317                  sun,
      1318                  self.radius,
      1319                  max_steps,
      1314 -                shift_at=lambda p: sample_cube(plane, p),    
      1320 +                shift_at=channel(1),                         
      1321                  octave_mean=self._octave_mean,
      1322 +                depth_at=channel(2),                         
      1323 +                spread_at=channel(3),                        
      1324              )
      1325              if len(self._shades) > SHADE_CACHE_SIZE:
      1326                  self._shades = {}

● Remaining conflict in this commit: demo_terrain.py.

  Ran 1 shell command (ctrl+o to expand)
● Bash(python - <<'EOF'
      import re…)
  ⎿  80:    F1 - F5         cycle one part's quality preset, low / mid / hi
     gh / ultra:
     83:    F6              cycle every part's preset at once
     … +16 lines (ctrl+o to expand)
  ⎿  Updated scripts/demo_terrain.py (+0 -10)
      81                      terrain, air, sky, clouds, ocean (each part
          's own
      82                      settings; a cloud or ocean change may refil
          l a moment)
      83      F6              cycle every part's preset at once
      84 -<<<<<<< HEAD                                                   
      84      C               show / hide the high cirrus
      86 -||||||| parent of ff11889 (Cloud shadows)                      
      87 -=======                                                        
      85      F7              cycle the shadows' preset (the clouds' shad
          ow on the
      86                      ground: off at low, more samples above)
      90 ->>>>>>> ff11889 (Cloud shadows)                                
      87      F               print the camera pose and terrain stats
      88      ESC             quit
      89  
     ...
      949              settings=self.world.clouds,
      950              channel="clouds",
      951          )
      956 -<<<<<<< HEAD                                                  
      952          self.clouds.bind_inputs(functools.partial(self.shadows
           .bind, body=0))
      958 -||||||| parent of ff11889 (Cloud shadows)                     
      959 -        self.shadows.bind(self.clouds.root, 0)                
      960 -=======                                                       
      961 -        self.shadows.bind(self.clouds.root, 0)                
      953          self.clouds.bind_shadow_inputs(self.terrain.root)  # t
           heir shadow on it
      963 ->>>>>>> ff11889 (Cloud shadows)                               
      954          self.clouds.set_debug(self._cloud_debug)
      955          self.clouds.set_sun(self._sun)
      956          if getattr(self, "depth", None) is not None:

  Ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='docs/source/atmosphere.md'…)
  ⎿  diff --git a/src/world_builder/terrain/shaders/terrain_frag.glsl b/src
     /world_builder/terrain/shaders/terrain_frag.glsl
     index 0a49568..133b9e9 100644
     … +77 lines (ctrl+o to expand)
  ⎿  Updated docs/source/atmosphere.md (+1 -15)
      570  ### Not yet
      571  
      572  - **Wind:** the clouds are static.
      573 -<<<<<<< HEAD                                                  
      574 -- **Shadows on the ground:** clouds do not shade the terrain (
          -planned:                                                      
      575 -  [Shadows](shadows.md)); the cirrus does not shade the cumulu
          -s either.                                                     
      576 -  Eclipses by other bodies already darken the air, the ground 
          -and the                                                       
      577 -  clouds.                                                     
      573 +- **The cirrus casts no shadow**, on the cumulus or on the gro
          +und.                                                          
      574  - **One cumulus-like type per planet** (cumulus in the demo), 
           with the cirrus
      575    above it; the planet field could take one per slab (the weat
           her map could
      580 -||||||| parent of ff11889 (Cloud shadows)                     
      581 -- **Shadows on the ground:** clouds do not shade the terrain (
          -planned:                                                      
      582 -  [Shadows](shadows.md)). Eclipses by other bodies already dar
          -ken the air,                                                  
      583 -  the ground and the clouds.                                  
      584 -- **One cloud type per planet** (cumulus in the demo); the fla
          -t field mixes                                                 
      585 -  types, the planet field could take one per slab (the weather
          - map could                                                    
      586 -=======                                                       
      587 -- **One cloud type per planet** (cumulus in the demo); the fla
          -t field mixes                                                 
      588 -  types, the planet field could take one per slab (the weather
          - map could                                                    
      589 ->>>>>>> ff11889 (Cloud shadows)                               
      576    then say which).
      577  - **Flying through the cirrus:** from inside its slab it is a 
           veil fading
      578    near the camera, with no volume.

● The terrain side merged cleanly. Comments, docs and tests from this commit
  still name SphericalClouds, which is now CloudVolume:

  Searched for 1 pattern (ctrl+o to expand)

● Bash(git ls-files docs/source/apidocs | head -2; for f in
      src/world_builder/clouds/sphere.py
      src/world_builder/terrain/shaders/terrain_frag.glsl src/world_builder/sh…)
  ⎿    straight stretch of the field, through the two lowest octaves calibr
     ated
       onto the full field, the weather fetched once at the start.
     … +37 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/sphere.py (+1 -1)
      258      gen_ms: float
      259  
      260  
      261 -#: SphericalClouds.sun_visibility keeps a point's shade per cu
          -be of this                                                    
      261 +#: CloudVolume.sun_visibility keeps a point's shade per cube o
          +f this                                                        
      262  #: side, metres, and at most this many of them.
      263  SHADE_CELL = 25.0
      264  SHADE_CACHE_SIZE = 4096
  ⎿  Updated src/world_builder/terrain/shaders/terrain_frag.glsl (+1 -1)
      59  uniform vec2  uStrataRange;       // the altitudes its first an
          d last texels span
      60  uniform int   uDebugMode;  // 0 shaded, 1 LOD, 2 biomes, 3 slop
          e, 4 normals, 5 skirts
      61  // The planet's clouds' layer in layerParams whose shadow falls
           here, -1 none
      62 -// (SphericalClouds.bind_shadow_inputs).                       
      62 +// (CloudVolume.bind_shadow_inputs).                           
      63  uniform int   uCloudShadowLayer;
      64  uniform int   uMaxLevel;
      65  
  ⎿  Updated src/world_builder/shadows/__init__.py (+2 -2)
      21  
      22  The second term is the planet's clouds' shadow on what is under
           them,
      23  marched through their density field (cloud_field.glsl's
      24 -``cloudShadowDepth``; :meth:`SphericalClouds.bind_shadow_inputs
      25 -<world_builder.clouds.sphere.SphericalClouds.bind_shadow_inputs
         ->` gives a                                                     
      24 +``cloudShadowDepth``; :meth:`CloudVolume.bind_shadow_inputs    
      25 +<world_builder.clouds.sphere.CloudVolume.bind_shadow_inputs>` g
         +ives a                                                         
      26  node's shaders the field). :class:`ShadowSettings` sets what it
           costs.
      27  
      28  Most of the time no body's shadow is near another: :meth:`Shado
          ws.set_sun`
  ⎿  Updated docs/source/shadows.md (+3 -3)
      109    marches the chord between them in steps of about the layer's
      110    `sunStepLength`, at most `maxSteps` (a long, low chord takes
            longer steps
      111    rather than more of them). The shadow is `exp2(-depth)`.
      112 -- **The field.** The deck layer's (`SphericalClouds._deck_laye
          -r`, the full                                                  
      112 +- **The field.** The deck layer's (`CloudVolume._deck_layer`, 
          +the full                                                      
      113    field with per-metre extinction, and the cumulus's `shadow_s
           trength`), so
      114    the shadow matches the clouds as drawn.
      115 -- **Inputs.** `SphericalClouds.bind_shadow_inputs(node)` gives
          - a node (the                                                  
      115 +- **Inputs.** `CloudVolume.bind_shadow_inputs(node)` gives a n
          +ode (the                                                      
      116    planet's terrain root) the field's inputs and `uCloudShadowL
           ayer`, its
      117    layer: -1 until the weather is made, or after `remove`.
      118    `clouds.field.cloud_shadow_defaults` (the terrain binds it) 
           is "no clouds
     ...
      125    the CPU's light on the sea leaves the clouds out. The sky li
           ght is not
      126    dimmed: under a cloud the air all round is still lit.
      127  - **CPU twin.** `noise.cloud_shadow_on_sphere`, and
      128 -  `SphericalClouds.sun_visibility(points, sun, steps)`, which 
      128 +  `CloudVolume.sun_visibility(points, sun, steps)`, which     
      129    `World.sun_visibility` and `lights_at` multiply in: Panda3D'
           s lights and
      130    the tracked models darken under a cloud.
      131  - **Knob.** `ShadowSettings.cloud_shadow_steps` (the `shadows`
            section of

● Bash(python - <<'EOF'
      p='docs/source/shadows.md'…)
  ⎿      scale = f"const float CLOUD_SHADOW_STEP_SCALE = {CLOUD_SHADOW_STEP
     _SCALE:.1f};"
         assert scale in source
     … +64 lines (ctrl+o to expand)
  ⎿  Updated docs/source/shadows.md (+12 -3)
      115  - **Inputs.** `CloudVolume.bind_shadow_inputs(node)` gives a n
           ode (the
      116    planet's terrain root) the field's inputs and `uCloudShadowL
           ayer`, its
      117    layer: -1 until the weather is made, or after `remove`.
      118 -  `clouds.field.cloud_shadow_defaults` (the terrain binds it) 
          -is "no clouds                                                 
      119 -  overhead". `World` binds each body's clouds to its terrain; 
          -the                                                           
      118 +  `clouds.shadow_defaults.cloud_shadow_defaults` (the terrain 
          +binds it) is                                                  
      119 +  "no clouds overhead". `PlanetClouds.bind_shadow_inputs` bind
          +s its volume's                                                
      120 +  (and binds a new volume again on a change of renderer; a ray
          +marched one                                                   
      121 +  casts none yet). `World` binds each body's clouds to its ter
          +rain; the                                                     
      122    hand-built demo does the same.
      123 +- **The weather** is fetched where the chord enters the slab, 
          +as the                                                        
      124 +  clouds' own march fetches it where it starts: the coverage's
          + shift, and                                                   
      125 +  with genera ([Clouds](clouds.md#genera)) the convective dept
          +h, the cell                                                   
      126 +  octave and the anvil's shelf. The march's early-out (exact: 
          +the two                                                       
      127 +  octaves at their most under the threshold at its lowest) all
          +ows for                                                       
      128 +  the anvil's fall and the shelf. With genera, the convective 
          +volume casts                                                  
      129 +  the shadow.                                                 
      130  - **Receivers.** The terrain's sunlight (`terrain_frag.glsl`, 
           at the planet
      131    frame position `uAtmoCam[own] + vRelPos`: clouds come with a
           ir), so land
      132    and far water. The near sea takes a per-pixel `sunShade` in 
           `seaShade`,
     ...
      134    the CPU's light on the sea leaves the clouds out. The sky li
           ght is not
      135    dimmed: under a cloud the air all round is still lit.
      136  - **CPU twin.** `noise.cloud_shadow_on_sphere`, and
      128 -  `CloudVolume.sun_visibility(points, sun, steps)`, which     
      137 +  `CloudVolume.sun_visibility(points, sun, steps)` (`PlanetClo
          +uds`'), which                                                 
      138    `World.sun_visibility` and `lights_at` multiply in: Panda3D'
           s lights and
      139    the tracked models darken under a cloud.
      140  - **Knob.** `ShadowSettings.cloud_shadow_steps` (the `shadows`
            section of

● The tests still construct SphericalClouds and set uSpherical. Let me see how
  they compare the GLSL to the CPU twin, to port them and add a genera case.

  Ran 4 shell commands (ctrl+o to expand)

● I'm porting the tests: removing uSpherical and the arguments main dropped from
  pack_layer_params, renaming the class, and parametrizing the GPU readback
  over a convective field so it's checked against the CPU twin.

● Bash(python - <<'EOF'
      p='tests/test_cloud_shadows.py'…)
  ⎿  308:        octaves=2,
  ⎿  Updated tests/test_cloud_shadows.py (+33 -8)
      28      ground_shadow_table,
      29      pack_layer_params,
      30      sphere_column_peaks,
      31 +    with_shelf,                                                
      32      with_threshold,
      33  )
      34  from world_builder.clouds.sphere import _noise
     ...
      182      return showbase
      183  
      184  
      184 -def test_the_shader_cloud_shadow_is_the_cpus(window, field, ta
          -ble):                                                         
      185 +@pytest.fixture(scope="module")                               
      186 +def convective():                                             
      187 +    """The genera's convective field (clouds.genera): its top 
          +rising with                                                   
      188 +    the depth, its cell octave, its anvil and shelf."""       
      189 +    raw = PRESETS[CloudType.CONVECTIVE].field                 
      190 +    volume = _noise(7)                                        
      191 +    peaks = sphere_column_peaks(volume, raw, R)               
      192 +    return with_shelf(with_threshold(raw, peaks, fbm_sigma(vol
          +ume, raw)), volume, R)                                        
      193 +                                                              
      194 +                                                              
      195 +def _genus_weather(cube):                                     
      196 +    """A four-channel weather cube from a two-channel one: a c
          +onvective                                                     
      197 +    depth and an anvil's spread that vary with the coverage.""
          +"                                                             
      198 +    depth = np.clip(1.4 * cube[..., 0] - 0.1, 0.0, 1.0)       
      199 +    spread = np.clip(2.0 * cube[..., 0] - 0.8, 0.0, 1.0)      
      200 +    return np.concatenate(                                    
      201 +        [cube, depth[..., None], spread[..., None]], axis=-1  
      202 +    ).astype(np.float32)                                      
      203 +                                                              
      204 +                                                              
      205 +@pytest.mark.parametrize("genus", [False, True])              
      206 +def test_the_shader_cloud_shadow_is_the_cpus(window, field, ta
          +ble, convective, genus):                                      
      207      """cloudShadowDepth (float32, the noise and the weather th
           rough the
      208      GPU's filtering) matches cloud_shadow_on_sphere under a re
           al weather
      187 -    map, from overhead to a low sun."""                       
      209 +    map, from overhead to a low sun: a cumulus field's, and a 
          +convective                                                    
      210 +    one's with its depth, cell octave and anvil shelf."""     
      211      from panda3d.core import (
      212          Geom,
     … 54 more lines (ctrl+o to expand)

  Ran 1 shell command (ctrl+o to expand)

● The convective case is far off, at the 99th percentile 0.55. Main has no
  numeric GPU-vs-CPU check, so the gap could be from my port or main's own twin,
  and I'm switching terms off one at a time to narrow it down.

  Ran 4 shell commands (ctrl+o to expand)

● The GPU result ignores the depth channel, which points to Panda storing RAM
  images in BGR(A) order while build_weather_texture writes RGBA, so the GPU
  reads coverage and depth swapped. I'll confirm that with a direct readback
  next.

  Ran 1 shell command (ctrl+o to expand)

● The float readback came back garbled. Retrying with an 8-bit target:

  Ran 1 shell command (ctrl+o to expand)

● The shader reads depth from .r and coverage from .b on a 4-channel cube, which
  is an existing bug on origin/main (#46) and not caused by the rebase. I'm now
  checking my port against Panda's texture input order before deciding how to
  handle it.

  Ran 1 shell command (ctrl+o to expand)

● With the texture in Panda's order, my port matches the CPU twin within float
  error, so the gap was main's bug, which also affects the far deck's depth
  reads. I'll keep that fix out of your "Cloud shadows" commit, commit only the
  cumulus readback, and leave the convective readback plus the fix as an
  uncommitted change for you to review.

● Bash(S=/tmp/claude-1000/-home-gl-development-WorldBuilder/d4124c88-f133-488e-a
      a3c-ec7b6e8c7853/scratchpad && cp tests/test_cloud_shadows.py
      $S/test_cloud_shadows_gen…)
  ⎿  diff --git a/tests/test_cloud_shadows.py b/tests/test_cloud_shadows.py
     new file mode 100644
     index 0000000..86c623b
     … +7 lines (ctrl+o to expand)
  ⎿  Updated tests/test_cloud_shadows.py (+2 -30)
      28      ground_shadow_table,
      29      pack_layer_params,
      30      sphere_column_peaks,
      31 -    with_shelf,                                                
      31      with_threshold,
      32  )
      33  from world_builder.clouds.sphere import _noise
     ...
      181      return showbase
      182  
      183  
      185 -@pytest.fixture(scope="module")                               
      186 -def convective():                                             
      187 -    """The genera's convective field (clouds.genera): its top 
          -rising with                                                   
      188 -    the depth, its cell octave, its anvil and shelf."""       
      189 -    raw = PRESETS[CloudType.CONVECTIVE].field                 
      190 -    volume = _noise(7)                                        
      191 -    peaks = sphere_column_peaks(volume, raw, R)               
      192 -    return with_shelf(with_threshold(raw, peaks, fbm_sigma(vol
          -ume, raw)), volume, R)                                        
      193 -                                                              
      194 -                                                              
      195 -def _genus_weather(cube):                                     
      196 -    """A four-channel weather cube from a two-channel one: a c
          -onvective                                                     
      197 -    depth and an anvil's spread that vary with the coverage.""
          -"                                                             
      198 -    depth = np.clip(1.4 * cube[..., 0] - 0.1, 0.0, 1.0)       
      199 -    spread = np.clip(2.0 * cube[..., 0] - 0.8, 0.0, 1.0)      
      200 -    return np.concatenate(                                    
      201 -        [cube, depth[..., None], spread[..., None]], axis=-1  
      202 -    ).astype(np.float32)                                      
      203 -                                                              
      204 -                                                              
      205 -@pytest.mark.parametrize("genus", [False, True])              
      206 -def test_the_shader_cloud_shadow_is_the_cpus(window, field, ta
          -ble, convective, genus):                                      
      184 +def test_the_shader_cloud_shadow_is_the_cpus(window, field, ta
          +ble):                                                         
      185      """cloudShadowDepth (float32, the noise and the weather th
           rough the
      186      GPU's filtering) matches cloud_shadow_on_sphere under a re
           al weather
      209 -    map, from overhead to a low sun: a cumulus field's, and a 
          -convective                                                    
      210 -    one's with its depth, cell octave and anvil shelf."""     
      187 +    map, from overhead to a low sun."""                       
      188      from panda3d.core import (
      189          Geom,
     … 18 more lines (ctrl+o to expand)

● All 778 tests pass. Now this commit's CHANGELOG entry, for the new
  PlanetClouds hooks:

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='CHANGELOG.md'…)
  ⎿  clouds_noon   settled  352 chunks, finest 4, 57k bb, LOD 0.4 ms, 7.19
     ms (139 fps) -> /tmp/claude-1000/-home-gl-development-WorldBuilder/d41
     24c88-f133-488e-aa3c-ec7b6e8c7853/scratchpad/shots4/clouds_noon.png
     … +3 lines (ctrl+o to expand)
  ⎿  Updated CHANGELOG.md (+9 -4)
      117    water and the near sea, marched per pixel through the deck's
            density
      118    field toward the sun (`cloud_field.glsl`'s `cloudShadowDepth
           `, its CPU twin
      119    `noise.cloud_shadow_on_sphere`), so the shadow follows the c
           louds drawn
      120 -  and the sun. `SphericalClouds.bind_shadow_inputs(node)` lets
          - a node's                                                     
      121 -  shaders read them (`World` binds each body's clouds to its t
          -errain);                                                      
      122 -  `SphericalClouds.sun_visibility` is the CPU side, in        
      123 -  `World.sun_visibility` and `lights_at`. `ShadowSettings`    
      120 +  and the sun. `CloudVolume.bind_shadow_inputs(node)` lets a n
          +ode's                                                         
      121 +  shaders read them, and `PlanetClouds.bind_shadow_inputs` its
          + volume's,                                                    
      122 +  again after a change of renderer (`World` binds each body's 
          +clouds to                                                     
      123 +  its terrain; a raymarched volume casts none yet);           
      124 +  `CloudVolume.sun_visibility` (and `PlanetClouds'`) is the CP
          +U side, in                                                    
      125 +  `World.sun_visibility` and `lights_at`. The march takes the 
          +genera's                                                      
      126 +  weather (convective depth, cell octave, anvil shelf) where i
          +ts chord                                                      
      127 +  enters the slab. `PlanetClouds.bind_inputs(bind)` gives ever
          +y layer's                                                     
      128 +  shader more inputs (the eclipses: `World` binds them so). `S
          +hadowSettings`                                                
      129    (`cloud_shadow_steps`) is a new `shadows` section of `WorldS
           ettings`.
      130    `Sea.set_light(light, sun=...)` takes the sun's share apart,
            which the
      131    shadow dims per pixel. Far away and under a grazing sun the 
           march gives

  Read 1 file (ctrl+o to expand)

● Cloud shadows are on the ground (the dark patches on the right). The fine dot
  pattern over the sand is new to me, though. Let me check whether main has it
  too.

  Read 2 files, ran 1 shell command (ctrl+o to expand)

● The rebased commit renders the same view as main, with faint new cloud shadows
  on the sea, and the frame cost rose from 4.5 to 6.0 ms, as the branch's docs
  predicted for the per-pixel march. The next-but-one commit, shadow maps,
  should recover that cost, so I'm moving on.

● Bash(GIT_EDITOR=true git rebase --continue 2>&1 | tail -22)
  ⎿  Auto-merging docs/source/quality.md
     Auto-merging docs/source/shadows.md
     Auto-merging docs/source/terrain.md
     … +19 lines (ctrl+o to expand)
  ⎿  Updated CHANGELOG.md (+11 -0)
  ⎿  Updated SHADOWS_PLAN.md (+33 -2)
  ⎿  Updated docs/source/quality.md (+1 -0)
  ⎿  Updated docs/source/shadows.md (+68 -4)
  ⎿  Updated docs/source/terrain.md (+3 -0)
  ⎿  … 15 more files changed
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● Commit 5/11, terrain shadow map. Let me see the conflict and how the commit
  puts mountain shadows on the clouds.

  Ran 1 shell command (ctrl+o to expand)
● The shadow map binding hits the same override issue, so it's going through
  bind_inputs instead. The cirrus will also take the mountain shadow, like the
  billboards, and the shots conflict is another union problem.

● Bash(python - <<'EOF'
      import re…)
  ⎿  165:from world_builder.terrain.shadow_map import ShadowMap
     353:    :ivar shadow_map: its mountains' shadow
     359:    shadow_map: ShadowMap | None = None
     … +16 lines (ctrl+o to expand)
  ⎿  Updated scripts/terrain_shots.py (+2 -6)
      206          lat, lon = (float(v[0]) for v in latlon(under[None, :]
           ))
      207          poses["eclipse_orbit"] = (lat, lon, 3.0 * radius, 0.0,
            -89.9, 0, 0)
      208          poses["lunar_eclipse"] = poses["moon_orbit"]
      209 -<<<<<<< HEAD                                                  
      209 +        # The moon's craters at its terminator: their rims' sh
          +adows.                                                        
      210 +        poses["moon_terminator"] = (lat, lon, 1.0 * moon.radiu
          +s, 0.0, -89.9, 0, 1)                                          
      211      # The cirrus (8-9 km) against the cumulus (2.2-3.2 km): un
           der both, between,
      212      # above both, and at dusk (looking west, toward the set su
           n).
      213      poses["cirrus_up"] = (lat0, lon0, start.get("alt", 800.0),
            0.0, 35.0, 0, 0)
      214      poses["cirrus_between"] = (ref_lat, ref_lon, 5000.0, 0.0, 
           3.0, 0, 0)
      215      poses["cirrus_above"] = (ref_lat, ref_lon, 15000.0, 0.0, -
           8.0, 0, 0)
      216      poses["cirrus_dusk"] = (ref_lat, ref_lon, 600.0, 90.0, 15.
           0, 0, 0)
      216 -||||||| parent of 16669b3 (terrain shadow map)                
      217 -=======                                                       
      218 -        # The moon's craters at its terminator: their rims' sh
          -adows.                                                        
      219 -        poses["moon_terminator"] = (lat, lon, 1.0 * moon.radiu
          -s, 0.0, -89.9, 0, 1)                                          
      220 ->>>>>>> 16669b3 (terrain shadow map)                          
      217      if demo.ring is not None:
      218          poses["ring_ground"] = (ref_lat, ref_lon, 1500.0, 180.
           0, 12.0, 0, 0)
      219          poses["ring_orbit"] = (35.0, lon0, 6.0 * radius, 0.0, 
           -89.9, 0, 0)
  ⎿  Updated src/world_builder/world.py (+1 -1)
      478                      functools.partial(self.shadows.bind, body=
           len(self.bodies))
      479                  )
      480                  body.clouds.bind_shadow_inputs(terrain.root)  
           # their shadow
      481 -                body.shadow_map.bind(body.clouds.root)  # the 
          -peaks' on them                                                
      481 +                body.clouds.bind_inputs(body.shadow_map.bind) 
          + # the peaks' on them                                         
      482              if look.ocean is not None:
      483                  kwargs = {
      484                      name: getattr(look.ocean, name)

● Bash(sed -i '972s/.*/
      self.clouds.bind_inputs(self.home.shadow_map.bind)/'
      scripts/demo_terrain.py && sed -n 968,974p scripts/demo_terrain.py &&
      python -…)
  ⎿          )
             self.clouds.bind_inputs(functools.partial(self.shadows.bind, b
     ody=0))
     … +10 lines (ctrl+o to expand)
  ⎿  Updated scripts/demo_terrain.py (+1 -1)
      969          self.clouds.bind_inputs(functools.partial(self.shadows
           .bind, body=0))
      970          self.clouds.bind_shadow_inputs(self.terrain.root)  # t
           heir shadow on it
      971          if self.home.shadow_map is not None:  # the peaks' on 
           them
      972 -            self.home.shadow_map.bind(self.clouds.root)       
      972 +            self.clouds.bind_inputs(self.home.shadow_map.bind)
      973          self.clouds.set_debug(self._cloud_debug)
      974          self.clouds.set_sun(self._sun)
      975          if getattr(self, "depth", None) is not None:
  ⎿  Updated src/world_builder/clouds/shaders/cirrus.vert (+2 -1)
      55      }
      56      // Other bodies eclipsing the sun (as cloud_sphere.vert).
      57      float eclipse = eclipseVisibility(p - camPos, sunDir, uShad
          owSelf);
      58 -    vSunTint *= eclipse;                                       
      58 +    // And the mountains between the cirrus and a low sun (the 
         +body's map).                                                   
      59 +    vSunTint *= eclipse * terrainShadow(normalize(p), length(p)
         + - planetRadius);                                              
      60      vTwilight *= eclipse;
      61      vSkyTint *= mix(ECLIPSE_SKY_FLOOR, 1.0, eclipse);
      62  }

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='docs/source/shadows.md'…)
  ⎿  -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.htm
     l
     785 passed, 5 warnings in 61.28s (0:01:01)
  ⎿  Updated docs/source/shadows.md (+3 -2)
      204  - **Receivers.** The terrain (land, far water) per pixel at it
           s planet-frame
      205    position (a varying from terrain.vert: an airless moon has i
           ts craters'
      206    shadows too) and its drawn height; the near sea through `sun
           Shade`; the
      207 -  clouds (billboards per vertex, the deck per fragment: peaks 
          -between a                                                     
      208 -  low sun and the clouds). The CPU twin, `ShadowMap.visibility
          -` (kept per                                                   
      207 +  clouds (billboards and cirrus per vertex, the deck per fragm
          +ent: peaks                                                    
      208 +  between a low sun and the clouds; bound through             
      209 +  `PlanetClouds.bind_inputs(shadow_map.bind)`). The CPU twin, 
          +`ShadowMap.visibility` (kept per                              
      210    25 m cell until the next bake), is in `World.sun_visibility`
            and
      211    `lights_at`.
      212  - **Knob.** `ShadowSettings.terrain_map_size` (rebuild): 256 a
           t LOW, 512,

  Read 1 file, ran 1 shell command (ctrl+o to expand)

● I'm at commit 5 of 11 (terrain shadow map). It passes all 785 tests, but the
  cirrus dusk render has the colours reversed, so I'm comparing it against main.

  Read 1 file, ran 1 shell command (ctrl+o to expand)

● Identical to main, so that look is main's own and not a regression. Continuing
  to commit 6.

● Bash(GIT_EDITOR=true git rebase --continue 2>&1 | grep -v '^Auto-merging' |
      tail -20)
  ⎿  [detached HEAD d1ba1b1] terrain shadow map
      21 files changed, 978 insertions(+), 45 deletions(-)
      create mode 100644 src/world_builder/terrain/shadow_map.py
     … +13 lines (ctrl+o to expand)
  ⎿  Updated CHANGELOG.md (+12 -0)
  ⎿  Updated docs/source/performance_backlog.md (+9 -1)
  ⎿  Updated docs/source/quality.md (+3 -0)
  ⎿  Updated docs/source/shadows.md (+66 -1)
  ⎿  Updated docs/source/world.md (+3 -1)
  ⎿  … 14 more files changed
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● Commit 6/11 adds the cloud shadow maps. The field.py part goes to the new
  defaults module:

● Bash(git diff f30ad5f~1 f30ad5f -- src/world_builder/clouds/field.py | grep
      '^[-+]'; git rm -q src/world_builder/clouds/field.py; sed -n
      '/<<<<<<</,/>>>>>>>/p' src/w…)
  ⎿  --- a/src/world_builder/clouds/field.py
     +++ b/src/world_builder/clouds/field.py
     +        cloudShadowNear=_no_cloud_map(),
     … +58 lines (ctrl+o to expand)
  ⎿  1 file changed (binary, mode only or too large to show)

● CloudShadowMaps is handed the clouds object itself. Let me see what it reads
  from it, since it will now get a PlanetClouds.

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)

● The maps read private attributes (_shadow_layer, _deck_field) along with
  position and radius. I'll add public shadow_layer and shadow_field properties
  to CloudVolume, have PlanetClouds pass them through, and switch the maps to
  use them.

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/sphere.py'…)
  ⎿  tests/test_cloud_shadows.py:376:        assert clouds._shadow_layer[0]
      == -1  # the set-up is not done
     tests/test_cloud_shadows.py:383:        assert clouds._shadow_layer[0]
     … +14 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/sphere.py (+11 -0)
      1252          """
      1253          self.root.set_shader_input("uCloudDebug", int(flags))
      1254  
      1255 +    @property                                                
      1256 +    def shadow_layer(self) -> int:                           
      1257 +        """The layer of ``layerParams`` the clouds' shadow is
           + marched in                                                  
      1258 +        (the deck's), -1 until the set-up is done or after :m
           +eth:`remove`."""                                             
      1259 +        return int(self._shadow_layer[0])                    
      1260 +                                                             
      1261 +    @property                                                
      1262 +    def shadow_field(self) -> DensityField:                  
      1263 +        """The field the clouds' shadow is marched in (the de
           +ck's)."""                                                    
      1264 +        return self._deck_field                              
      1265 +                                                             
      1266      def bind_shadow_inputs(self, node: NodePath) -> None:
      1267          """Let the shaders on a node (the planet's terrain) d
            arken the sun
      1268          under these clouds (cloud_field.glsl's ``cloudShadowD
            epth``, through
  ⎿  Updated src/world_builder/clouds/planet.py (+22 -0)
      36  from world_builder.clouds.cloud import PRESETS, CloudSettings, 
          CloudType
      37  from world_builder.clouds.genera import CONVECTIVE, STRATIFORM,
           GeneraSpec
      38  from world_builder.clouds.march import CloudMarch
      39 +from world_builder.clouds.noise import DensityField            
      40  from world_builder.clouds.shadow_defaults import cloud_shadow_d
          efaults
      41  from world_builder.clouds.sphere import CLOUD_BIN, CloudVolume
      42  from world_builder.clouds.weather import WeatherSpec
     ...
      116          :class:`march.CloudMarch`, by ``settings.renderer``)
      117      :ivar sheet: the cirrus (:class:`cirrus.CloudSheet`), or N
           one
      118      :ivar weather: the weather both follow
      119 +    :ivar radius: the planet's sea-level radius, metres       
      120 +    :ivar position: its centre, world frame                   
      121      """
      122  
      123      #: The settings it takes (``WorldSettings.apply``, ``setti
           ngs_for``).
     ...
      203              planet = planet or PlanetSpec(name="clouds", radiu
           s=float(radius))
      204              weather = WeatherSpec(planet, seed=weather_seed)
      205          self.weather = weather
      206 +        self.radius = float(radius)                           
      207 +        self.position = np.asarray(position, dtype=np.float64)
      208          self._settings = settings
      209          self.root = parent.attach_new_node("planet_clouds")
      210          # What the volume is built from, again on a change of 
           renderer.
     ...
      368              for node in nodes:
      369                  cloud_shadow_defaults(node)
      370  
      371 +    @property                                                 
      372 +    def shadow_layer(self) -> int:                            
      373 +        """The volume's layer its shadow is marched in        
      374 +        (:attr:`sphere.CloudVolume.shadow_layer`); -1 = none (
          +not ready, or                                                 
      375 +        a raymarch)."""                                       
      376 +        if isinstance(self.volume, CloudVolume):              
      377 +            return self.volume.shadow_layer                   
      378 +        return -1                                             
      379 +                                                              
      380 +    @property                                                 
      381 +    def shadow_field(self) -> DensityField | None:            
      382 +        """The field the volume's shadow is marched in; None w
          +ith a                                                         
      383 +        raymarch."""                                          
      384 +        if isinstance(self.volume, CloudVolume):              
     … 6 more lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/shadow_map.py (+4 -3)
      145          """
      146          :param output: the window (its GSG draws the maps)
      147          :param clouds: the planet's clouds
      148 -            (:class:`~world_builder.clouds.sphere.SphericalClo
          -uds`)                                                         
      148 +            (:class:`~world_builder.clouds.planet.PlanetClouds
          +`, or a                                                       
      149 +            :class:`~world_builder.clouds.sphere.CloudVolume`)
      150          :param size: ``ShadowSettings.cloud_map_size`` (0: non
           e)
      151          :param to_ram: copy each map to RAM when drawn (tests)
      152          """
     ...
      212          :param steps: ``ShadowSettings.cloud_shadow_steps``
      213          """
      214          clouds = self._clouds
      214 -        layer = int(clouds._shadow_layer[0])                  
      215 +        layer = clouds.shadow_layer                           
      216          if self.near is None or layer < 0 or steps <= 0:
      217              self._on[0] = 0
      218              return
      219          relative = np.asarray(camera_world, dtype=np.float64) 
           - clouds.position
      220          up = relative / np.linalg.norm(relative)
      221          sun = np.asarray(sun, dtype=np.float64)
      221 -        base = clouds.radius + clouds._deck_field.slab[0]     
      222 +        base = clouds.radius + clouds.shadow_field.slab[0]    
      223          for shadow_map in (self.near, self.far):
      224              if shadow_map.wanted(up, sun, steps, layer, base):
      225                  shadow_map.draw(up, sun, steps, layer, base)

● Now world.py, demo_terrain.py, and this commit's tests:

● Bash(python - <<'EOF'
      import re…)
  ⎿  src/world_builder/world.py:487:                body.clouds.bind_inputs
     (body.shadow_map.bind)  # the peaks' on them
     src/world_builder/world.py-488-                output = getattr(app, "
     … +71 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/world.py (+0 -12)
      81  from world_builder.clouds.cloud import CloudType
      82  from world_builder.clouds.genera import GeneraSpec
      83  from world_builder.clouds.half_res import HalfResClouds
      84 -<<<<<<< HEAD                                                   
      84  from world_builder.clouds.planet import PlanetClouds, check_lay
          ers
      86 -||||||| parent of f30ad5f (Detailes local shadow maps for terra
         -in and clouds)                                                 
      87 -from world_builder.clouds.sphere import SphericalClouds        
      88 -=======                                                        
      85  from world_builder.clouds.shadow_map import CloudShadowMaps
      90 -from world_builder.clouds.sphere import SphericalClouds        
      91 ->>>>>>> f30ad5f (Detailes local shadow maps for terrain and clo
         -uds)                                                           
      86  from world_builder.clouds.weather import WeatherSpec
      87  from world_builder.depth import HIDE_FROM_NEAR, SplitDepthCamer
          as
      88  from world_builder.gravity import Gravity
     ...
      484                      functools.partial(self.shadows.bind, body=
           len(self.bodies))
      485                  )
      486                  body.clouds.bind_shadow_inputs(terrain.root)  
           # their shadow
      493 -<<<<<<< HEAD                                                  
      487                  body.clouds.bind_inputs(body.shadow_map.bind) 
            # the peaks' on them
      495 -||||||| parent of f30ad5f (Detailes local shadow maps for terr
          -ain and clouds)                                               
      496 -                body.shadow_map.bind(body.clouds.root)  # the 
          -peaks' on them                                                
      497 -=======                                                       
      498 -                body.shadow_map.bind(body.clouds.root)  # the 
          -peaks' on them                                                
      488                  output = getattr(app, "win", None)
      489                  if output is not None:
      490                      body.cloud_maps = CloudShadowMaps(
      491                          output, body.clouds, self.settings.sha
           dows.cloud_map_size
      492                      )
      493                      body.cloud_maps.bind(terrain.root)
      505 ->>>>>>> f30ad5f (Detailes local shadow maps for terrain and cl
          -ouds)                                                         
      494              if look.ocean is not None:
      495                  kwargs = {
      496                      name: getattr(look.ocean, name)
  ⎿  Updated scripts/demo_terrain.py (+1 -15)
      143  from world_builder.clouds.cirrus import CirrusSpec
      144  from world_builder.clouds.genera import GeneraSpec
      145  from world_builder.clouds.half_res import HalfResClouds
      146 -<<<<<<< HEAD                                                  
      146  from world_builder.clouds.planet import PlanetClouds
      147 +from world_builder.clouds.shadow_map import CloudShadowMaps   
      148  from world_builder.clouds.weather import WeatherSpec
      149  from world_builder.depth import HIDE_FROM_CLOUD_PASS, HIDE_FRO
           M_NEAR, SplitDepthCameras
      150 -||||||| parent of f30ad5f (Detailes local shadow maps for terr
          -ain and clouds)                                               
      151 -from world_builder.clouds.sphere import SphericalClouds       
      152 -from world_builder.depth import HIDE_FROM_NEAR, SplitDepthCame
          -ras                                                           
      153 -=======                                                       
      154 -from world_builder.clouds.shadow_map import CloudShadowMaps   
      155 -from world_builder.clouds.sphere import SphericalClouds       
      156 -from world_builder.depth import HIDE_FROM_NEAR, SplitDepthCame
          -ras                                                           
      157 ->>>>>>> f30ad5f (Detailes local shadow maps for terrain and cl
          -ouds)                                                         
      150  from world_builder.lights import TrackedLights
      151  from world_builder.ocean import Sea
      152  from world_builder.quality import QualityLevel
     ...
       982          self.clouds.bind_inputs(functools.partial(self.shadow
            s.bind, body=0))
       983          self.clouds.bind_shadow_inputs(self.terrain.root)  # 
            their shadow on it
       984          if self.home.shadow_map is not None:  # the peaks' on
             them
       993 -<<<<<<< HEAD                                                 
       985              self.clouds.bind_inputs(self.home.shadow_map.bind
            )
       995 -||||||| parent of f30ad5f (Detailes local shadow maps for ter
           -rain and clouds)                                             
       996 -            self.home.shadow_map.bind(self.clouds.root)      
       997 -=======                                                      
       998 -            self.home.shadow_map.bind(self.clouds.root)      
       986          # Their shadow marched once per texel of two maps (Sh
            adows, part E).
       987          self.cloud_maps = CloudShadowMaps(
       988              self.win, self.clouds, self.world.shadows.cloud_m
            ap_size
       989          )
       990          self.cloud_maps.bind(self.terrain.root)
      1004 ->>>>>>> f30ad5f (Detailes local shadow maps for terrain and c
           -louds)                                                       
       991          self.clouds.set_debug(self._cloud_debug)
       992          self.clouds.set_sun(self._sun)
       993          if getattr(self, "depth", None) is not None:

● Bash(sed -i 's/from world_builder.clouds.sphere import SphericalClouds/from
      world_builder.clouds.sphere import CloudVolume/; s/    clouds =
      SphericalClouds(/    clou…)
  ⎿  =========================== short test summary info ==================
     ==========
     FAILED tests/test_scene.py::test_the_gpu_reads_each_chunks_values - As
     … +2 lines (ctrl+o to expand)
  ⎿  Updated tests/test_cloud_shadows.py (+2 -2)
      423      moves a quarter of it."""
      424      from world_builder.clouds.cloud import CloudSettings
      425      from world_builder.clouds.shadow_map import NEAR_SPACING, 
           CloudShadowMaps
      426 -    from world_builder.clouds.sphere import SphericalClouds   
      426 +    from world_builder.clouds.sphere import CloudVolume       
      427      from world_builder.terrain.workers import ChunkExecutor
      428  
      429      executor = ChunkExecutor(backend="inline", inline_budget=6
           4)
      430 -    clouds = SphericalClouds(                                 
      430 +    clouds = CloudVolume(                                     
      431          window.render,
      432          R,
      433          executor,

  Ran 1 shell command (ctrl+o to expand)

● Exactly what the commit's field.py change added: the map defaults. Adding them
  to the new defaults module.

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/shadow_defaults.py'…)
  ⎿  -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.htm
     l
     791 passed, 5 warnings in 61.24s (0:01:01)
  ⎿  Updated src/world_builder/clouds/shadow_defaults.py (+14 -0)
      58          noiseOffset=Vec3(0.0, 0.0, 0.0),
      59          uCloudDebug=0,
      60          uCloudShadowLayer=-1,
      61 +        cloudShadowNear=_no_cloud_map(),                       
      62 +        cloudShadowFar=_no_cloud_map(),                        
      63 +        cloudNearChart=PTA_LVecBase4f.empty_array(4),          
      64 +        cloudFarChart=PTA_LVecBase4f.empty_array(4),           
      65 +        cloudShadowMapOn=0,                                    
      66      )
      67 +                                                               
      68 +                                                               
      69 +@functools.cache                                               
      70 +def _no_cloud_map() -> Texture:                                
      71 +    """A 1-texel cloud shadow map, for shaders with none (off).
         +"""                                                            
      72 +    tex = Texture("no_cloud_shadow_map")                       
      73 +    tex.setup_2d_texture(1, 1, Texture.T_float, Texture.F_rgba3
         +2)                                                             
      74 +    tex.set_ram_image(bytes(16))                               
      75 +    return tex                                                 

● That edit is mine; all 791 tests pass. Now a render check that the maps work
  through PlanetClouds:

  Read 1 file, ran 1 shell command (ctrl+o to expand)

● Identical to the per-pixel march, as the maps should be. Continuing to commit
  7.

● Bash(GIT_EDITOR=true git rebase --continue 2>&1 | grep -v '^Auto-merging' |
      tail -20)
  ⎿  [detached HEAD c9a7297] Detailes local shadow maps for terrain and clo
     uds
      21 files changed, 1356 insertions(+), 32 deletions(-)
     … +15 lines (ctrl+o to expand)
  ⎿  Updated CHANGELOG.md (+22 -0)
  ⎿  Updated docs/source/quality.md (+1 -0)
  ⎿  Updated docs/source/shadows.md (+74 -4)
  ⎿  Updated scripts/terrain_shots.py (+139 -3)
  ⎿  Updated src/world_builder/clouds/noise.py (+68 -0)
  ⎿  … 21 more files changed
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● Commit 7 applied cleanly, but its cloudFarDepth calls slabOpticalDepth, which
  my port renamed to slabOpticalDepthAt, so it likely won't compile. I'm going
  to check that next.

  Ran 1 shell command (ctrl+o to expand)

● Commit 7 still calls slabOpticalDepth, and its CPU twin far_cloud_depth
  ignores depth and shelf. I'll finish the rebase, then fold a fix into commit 7
  with a fixup and autosquash so every replayed commit stays buildable. Commit
  8 of 11, the custom model shader, is next.
  Ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      import re…)
  ⎿  +        # The models' shader's inputs near each (the clouds' are boun
     d with
     +        self.clouds.bind_shadow_inputs(self.home.models.inputs)  # on
     … +7 lines (ctrl+o to expand)
  ⎿  Updated scripts/demo_terrain.py (+0 -4)
      585              ("end", self._reset_camera, []),
      586              ("r", self._reset_camera, []),
      587              ("f", self._print_pose, []),
      588 -<<<<<<< HEAD                                                  
      588              ("c", self._toggle_cirrus, []),
      590 -||||||| parent of 71b2c5d (Custom model shader)               
      591 -=======                                                       
      589              ("m", self._toggle_model_shader, []),
      593 ->>>>>>> 71b2c5d (Custom model shader)                         
      590              ("escape", self._quit, []),
      591              (QUALITY_ALL_KEY, self._cycle_all_quality, []),
      592          ]
  ⎿  Updated scripts/demo_world.py (+0 -4)
      226              ("end", self._reset_camera, []),
      227              ("r", self._reset_camera, []),
      228              ("f", self._print_pose, []),
      229 -<<<<<<< HEAD                                                  
      229              ("c", self._toggle_cirrus, []),
      231 -||||||| parent of 71b2c5d (Custom model shader)               
      232 -=======                                                       
      230              ("m", self._toggle_model_shader, []),
      234 ->>>>>>> 71b2c5d (Custom model shader)                         
      231              ("escape", self._quit, []),
      232              (QUALITY_ALL_KEY, self._cycle_all_quality, []),
      233          ]
  ⎿  Updated src/world_builder/world.py (+1 -7)
      581              bodies[0].position if bodies else (0.0, 0.0, 0.0),
            dtype=np.float64
      582          )
      583  
      584 -<<<<<<< HEAD                                                  
      585 -    def _make_clouds(self, look: BodyLook, atmosphere: int) ->
          - PlanetClouds:                                                
      586 -||||||| parent of 71b2c5d (Custom model shader)               
      587 -    def _make_clouds(self, look: BodyLook, atmosphere: int) ->
          - SphericalClouds:                                             
      588 -=======                                                       
      584      def _model_lighting(
      585          self,
      586          exposure: float,
     ...
      597          lighting.set_sun(self._sun, self._sun_color)
      598          return lighting
      599  
      605 -    def _make_clouds(self, look: BodyLook, atmosphere: int) ->
          - SphericalClouds:                                             
      606 ->>>>>>> 71b2c5d (Custom model shader)                         
      600 +    def _make_clouds(self, look: BodyLook, atmosphere: int) ->
          + PlanetClouds:                                                
      601          cl = look.clouds
      602          clouds = PlanetClouds(
      603              self.app.render,

  Ran 1 shell command (ctrl+o to expand)

● Bash(sed -i 's/from world_builder.clouds.field import
      cloud_shadow_defaults/from world_builder.clouds.shadow_defaults import
      cloud_shadow_defaults/' src/world_builde…)
  ⎿  262-#: side, metres, and at most this many of them.
     263-SHADE_CELL = 25.0
     264:SHADE_CACHE_SIZE = 4096
     … +13 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/objects/__init__.py (+1 -1)
      54  )
      55  
      56  from world_builder.atmosphere import bind_defaults
      57 -from world_builder.clouds.field import cloud_shadow_defaults   
      57 +from world_builder.clouds.shadow_defaults import cloud_shadow_d
         +efaults                                                        
      58  from world_builder.shadows import bind_defaults as bind_shadow_
          defaults
      59  
      60  SHADER_PATH = Path(__file__).parent / "shaders"

● Bash(sed -i
      's/(SphericalClouds.bind_shadow_inputs)/(CloudVolume.bind_shadow_inputs)/'
      src/world_builder/objects/shaders/model_light.glsl && grep -rn
      'SphericalCloud…)
  ⎿  FAILED tests/test_objects.py::test_the_model_shader_lights_and_tonemap
     s_as_the_terrain
     6 failed, 794 passed, 5 warnings in 59.35s
  ⎿  Updated src/world_builder/objects/shaders/model_light.glsl (+1 -1)
      10  uniform mat3 uPlanetAxes;     // the body's own axes to the ren
          der frame
      11                                // (Terrain.rotation): its shadow
           map's
      12  // The body's clouds' layer in layerParams whose shadow falls h
          ere, -1 none
      13 -// (SphericalClouds.bind_shadow_inputs).                       
      13 +// (CloudVolume.bind_shadow_inputs).                           
      14  uniform int  uCloudShadowLayer;
      15  
      16  // The body's vertical at a point rel (from the camera, render 
          frame), or

  Ran 1 shell command (ctrl+o to expand)

● Commit 8 is resolved, but six failures trace back to a leftover
  slabOpticalDepth call in commit 7. I'll stash commit 8, add a fixup for commit
  7 on top, restore the stash, and squash the fixup into commit 7 at the end.

● Bash(git diff --name-only --diff-filter=U; git stash push -m
      "commit8-resolution" 2>&1 | tail -2; git status --short | head; git log
      --oneline -1 | cat)
  ⎿  Saved working directory and index state On (no branch): commit8-resolu
     tion
     b4674e4 clouds shadow clouds
  ⎿  Deleted src/world_builder/objects/__init__.py (+0 -162)
  ⎿  Deleted src/world_builder/objects/shaders/model.frag (+0 -59)
  ⎿  Deleted src/world_builder/objects/shaders/model.vert (+0 -31)
  ⎿  Deleted src/world_builder/objects/shaders/model_light.glsl (+0 -48)
  ⎿  Deleted tests/test_objects.py (+0 -308)
  ⎿  … 16 more files changed
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● Now the commit 7 fix. The GLSL far segment, built on slabOpticalDepthAt:

  Ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/shaders/cloud_field.glsl'…)
  ⎿  25:517-        shift_at=lambda p: np.full(len(p), shift),
     57:600-            shift_at=lambda p: sample_cube(cube[..., 1], p),
  ⎿  Updated src/world_builder/clouds/shaders/cloud_field.glsl (+6 -3)
      492  // sunStepLength) out to reach metres, or to where the ray lea
           ves the
      493  // slab's top if sooner, in steps of about CLOUD_FAR_STEP (at 
           most
      494  // CLOUD_FAR_STEPS): a low sun's ray runs along the slab for k
           ilometres,
      495 -// through the clouds sunward. Slowly varying across a billboa
          -rd, so taken                                                  
      496 -// at its centre (cloud_sphere.vert). noise.far_cloud_depth on
          - the CPU.                                                     
      495 +// through the clouds sunward. The weather is taken where the 
          +segment                                                       
      496 +// starts. Slowly varying across a billboard, so taken at its 
          +centre                                                        
      497 +// (cloud_sphere.vert). noise.far_cloud_depth on the CPU.     
      498  const float CLOUD_FAR_STEP = 600.0;
      499  const int CLOUD_FAR_STEPS = 12;
      500  float cloudFarDepth(Field f, vec3 p, vec3 dir, float reach) {
     ...
      512          return 0.0;
      513      }
      514      int steps = clamp(int(ceil((end - near) / CLOUD_FAR_STEP))
           , 1, CLOUD_FAR_STEPS);
      514 -    return slabOpticalDepth(f, p + dir * near, dir, (end - nea
          -r) / float(steps), steps);                                    
      515 +    vec3 from = p + dir * near;                               
      516 +    return slabOpticalDepthAt(f, from, dir, (end - near) / flo
          +at(steps), steps,                                             
      517 +                              weatherAt(from));               
      518  }
      519  
      520  // One cloud shadow map at a point of the base's sphere (unit 
           dir): its
  ⎿  Updated src/world_builder/clouds/noise.py (+22 -6)
      1121      reach: float,
      1122      shift_at: Callable[[np.ndarray], np.ndarray] | None = Non
            e,
      1123      octave_mean: float | None = None,
      1124 +    depth_at: Callable[[np.ndarray], np.ndarray] | None = Non
           +e,                                                           
      1125 +    spread_at: Callable[[np.ndarray], np.ndarray] | None = No
           +ne,                                                          
      1126  ) -> np.ndarray:
      1127      """The shader's ``cloudFarDepth``: the clouds along the s
            un's ray from
      1128      points (in or under the slab) past their own march (``sun
            _steps`` of
      1129      ``sun_step_length``) out to *reach* metres, or to where t
            he ray leaves
      1128 -    the slab's top, in steps of about :data:`CLOUD_FAR_STEP`.
      1130 +    the slab's top, in steps of about :data:`CLOUD_FAR_STEP`,
           + the weather                                                 
      1131 +    (and the cell octave) taken where the segment starts.    
      1132  
      1133      :param volume: the noise octave (quantised, to match the 
            GPU)
      1134      :param spec: a resolved field (the billboards' layer's)
     ...
      1140      :param shift_at: the weather's fbm shift at planet-frame 
            positions
      1141          ((m, 3) -> (m,)), taken where the segment starts; Non
            e = none
      1142      :param octave_mean: the noise's mean; None = *volume*'s
      1143 +    :param depth_at: the convective depth there, the same way
           +; None = 0                                                   
      1144 +    :param spread_at: the anvil shelf's spread there, the sam
           +e way; None =                                                
      1145 +        no shelf                                             
      1146      :returns: (...) optical depth
      1147      """
      1148      _require_resolved(spec)
     ...
      1169      h = (np.linalg.norm(positions, axis=-1) - radius - base) 
            / max(thickness, 1e-3)
      1170      coords = positions * np.asarray(spec.noise_scale) + np.as
            array(spec.offset)
      1171      pair = fbm(volume, coords, spec, octaves=2, octave_mean=0
            .0)
      1166 -    shift = np.zeros(points.shape[:-1])                      
      1167 -    if shift_at is not None:                                 
      1168 -        shift = np.reshape(shift_at(start.reshape(-1, 3)), po
           -ints.shape[:-1])                                             
      1169 -    gain, bias = shadow_calibration(spec, octave_mean)       
      1170 -    density = _slab_density(spec, h, pair * gain + shift[...,
           - None] + bias)                                               
      1172 +                                                             
      1173 +    def at(sample):                                          
      1174 +        if sample is None:                                   
      1175 +            return np.zeros(points.shape[:-1])               
      1176 +        return np.reshape(sample(start.reshape(-1, 3)), point
           +s.shape[:-1])                                                
      1177 +                                                             
      1178 +    convective = at(depth_at)                                
      1179 +    origin = start * np.asarray(spec.noise_scale) + np.asarra
           +y(spec.offset)                                               
     … 10 more lines (ctrl+o to expand)

  Ran 1 shell command (ctrl+o to expand)

● Bash(sed -i '652,653d' tests/test_cloud_shadows.py && sed -i '/^
      uSpherical=1,$/d' tests/test_cloud_shadows.py && timeout 300 python -m
      pytest tests/test_clou…)
  ⎿  14 passed, 1 warning in 4.32s
  ⎿  Updated tests/test_cloud_shadows.py (+0 -3)
      649      packed = pack_layer_params(
      650          field,
      651          extinction=field.density,
      652 -        wrap_radius=1e12,                                     
      653 -        wrap_fade_band=0.0,                                   
      652          octave_mean=float(_noise(7).mean()),
      653          optics=OPTICS,
      654          octaves=2,
     ...
      666          noiseOffset=Vec3(0.0, 0.0, 0.0),
      667          sunDir=Vec3(0.0, 0.0, 1.0),
      668          layerParams=params,
      671 -        uSpherical=1,                                         
      669          planetRadius=R,
      670          weatherMap=build_weather_texture(cube),
      671          uCloudDebug=0,

● Commit 7's tests pass again. Committing the fixup and restoring commit 8's
  resolution:

● Bash(git add -A && git commit -q -m "fixup! clouds shadow clouds" && git stash
      pop 2>&1 | tail -3; git status --short | head -30)
  ⎿  mixed line ending.....................................................
     ...Passed
     trim trailing whitespace..............................................
     … +29 lines (ctrl+o to expand)
  ⎿  Updated CHANGELOG.md (+16 -0)
  ⎿  Updated docs/source/atmosphere.md (+3 -2)
  ⎿  Updated docs/source/object_lighting.md (+68 -7)
  ⎿  Updated docs/source/shadows.md (+39 -4)
  ⎿  Updated docs/source/world.md (+11 -5)
  ⎿  … 16 more files changed
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● Pre-commit (ruff and the rest) passed on the fixup. Tests for commit 8:

  Ran 1 shell command (ctrl+o to expand)

● Bash(grep -n 'models.inputs\|bind_shadow_inputs\|cloud_maps.bind'
      src/world_builder/world.py scripts/demo_terrain.py; GIT_EDITOR=true git
      rebase --continue 2>&1 | gr…)
  ⎿  scripts/demo_terrain.py:522:            self.atmospheres.bind(body.mod
     els.inputs)
     scripts/demo_terrain.py:523:            self.shadows.bind(body.models.
     … +26 lines (ctrl+o to expand)
  ⎿  Created BACKLOG.md (+61 -0)
  ⎿  Updated CHANGELOG.md (+10 -0)
  ⎿  Updated README.md (+2 -1)
  ⎿  Updated docs/source/object_lighting.md (+7 -4)
  ⎿  Updated docs/source/quality.md (+2 -0)
  ⎿  … 16 more files changed
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

  Ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      import re…)
  ⎿  803 passed, 5 warnings in 58.52s
     [detached HEAD 3550d1f] Objects cast shadows
      21 files changed, 979 insertions(+), 22 deletions(-)
     … +13 lines (ctrl+o to expand)
  ⎿  Updated scripts/terrain_shots.py (+23 -5)
  ⎿  Updated BACKLOG.md (+12 -3)
  ⎿  Updated CHANGELOG.md (+22 -0)
  ⎿  Updated docs/source/asteroids.md (+18 -2)
  ⎿  Updated docs/source/performance_backlog.md (+6 -7)
  ⎿  … 26 more files changed
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● Commit 10 (ring shadow, shadow maps on GPU) applied cleanly. I'll verify it
  once commit 11's conflict is resolved, and the cirrus still needs the ring
  shadow. The commit 11 conflict:

  Ran 1 shell command (ctrl+o to expand)

● Main already exposes ensure_sky_bin publicly, so only the docstring needs
  merging. The ring-haze split needs a top_radius, and I'm checking how that
  split works now that main has a cirrus sheet.

  Ran 1 shell command (ctrl+o to expand)

● The haze's back pass sits in the sky bin at order 2 and the front pass after
  the clouds at bin 30, and since the cirrus shares the sky bin, the order
  matters. I'm checking where its two nodes are drawn next.

  Ran 1 shell command (ctrl+o to expand)

● The haze and cirrus layers are both at order 2, so their draw order is
  undefined, and the haze must go first since it sits farther out. I'm moving
  the cirrus to order 3 and setting the split radius to the top layer's radius,
  since either split height works.

  Ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      import re…)
  ⎿  src/world_builder/asteroids/haze.py:33:  (:meth:`FieldHaze.set_clouds_
     top`) is drawn in two passes, split where the
     src/world_builder/asteroids/haze.py:589:    def set_clouds_top(self, r
     … +7 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/sky/sky.py (+0 -6)
      370  
      371  
      372  def ensure_sky_bin() -> None:
      373 -<<<<<<< HEAD                                                  
      374 -    """Register the sky's cull bin (once)."""                 
      375 -||||||| parent of a26b4dc (Fixed ring-cloud interaction)      
      376 -def _ensure_sky_bin() -> None:                                
      377 -=======                                                       
      373      """The sky's cull bin (after the terrain, before the cloud
           s): the
      374      sky, the stars, and what a ring's haze holds beyond the cl
           ouds."""
      380 ->>>>>>> a26b4dc (Fixed ring-cloud interaction)                
      375      manager = CullBinManager.get_global_ptr()
      376      if manager.find_bin(SKY_BIN) == -1:
      377          manager.add_bin(SKY_BIN, CullBinManager.BT_fixed, SKY_
           BIN_SORT)
  ⎿  Updated src/world_builder/clouds/cirrus.py (+4 -3)
      124      from world_builder.terrain.workers import ChunkExecutor
      125  
      126  _SHADERS = Path(__file__).parent / "shaders"
      127 -#: The behind node's draw order in the sky's bin: after the do
          -me (0) and the                                                
      128 -#: stars (1).                                                 
      129 -BEHIND_DRAW_ORDER = 2                                         
      127 +#: The behind node's draw order in the sky's bin: after the do
          +me (0), the                                                   
      128 +#: stars (1) and a ring's haze beyond the clouds              
      129 +#: (``asteroids.haze.HAZE_BEHIND_ORDER``, farther still).     
      130 +BEHIND_DRAW_ORDER = 3                                         
      131  #: Panda's own bin for translucent objects, sorted back to fro
           nt.
      132  FRONT_BIN = "transparent"
      133  #: The baked stamps (scripts/bake_cirrus_atlas.py): patches of
            tufts at +x
  ⎿  Updated src/world_builder/clouds/sphere.py (+1 -1)
      145  #: front.
      146  CLOUD_BIN = "cloud_sphere"
      147  #: A cloud's billboard may stand this far over the slab's top,
            metres (its
      148 -#: radius): SphericalClouds.top_radius.                       
      148 +#: radius): CloudVolume.top_radius.                           
      149  CLOUD_TOP_MARGIN = 2000.0
      150  _CLOUD_BIN_SORT = 22
      151  #: The shells' bin sorts, shell 0 (the nearest, drawn last) fi
           rst: at most
  ⎿  Updated src/world_builder/clouds/planet.py (+17 -1)
      38  from world_builder.clouds.march import CloudMarch
      39  from world_builder.clouds.noise import DensityField
      40  from world_builder.clouds.shadow_defaults import cloud_shadow_d
          efaults
      41 -from world_builder.clouds.sphere import CLOUD_BIN, CloudVolume 
      41 +from world_builder.clouds.sphere import CLOUD_BIN, CLOUD_TOP_MA
         +RGIN, CloudVolume                                              
      42  from world_builder.clouds.weather import WeatherSpec
      43  from world_builder.depth import HIDE_FROM_CLOUD_PASS
      44  from world_builder.terrain.planet import PlanetSpec
     ...
      369                  cloud_shadow_defaults(node)
      370  
      371      @property
      372 +    def top_radius(self) -> float:                            
      373 +        """How far from the planet's centre the clouds reach, 
          +metres: the                                                   
      374 +        highest layer's top (the cirrus's, when there is one).
          + What a ray                                                   
      375 +        meets past where it enters this sphere is behind them 
          +(a ring's                                                     
      376 +        haze, :meth:`asteroids.haze.FieldHaze.set_clouds_top`)
          +."""                                                          
      377 +        tops = [                                              
      378 +            volume.top_radius                                 
      379 +            if isinstance(volume, CloudVolume)                
      380 +            else self.radius + sum(volume.field.slab) + CLOUD_
          +TOP_MARGIN                                                    
      381 +            for volume in self._volumes()                     
      382 +        ]                                                     
      383 +        if self.sheet is not None:                            
      384 +            tops.append(self.radius + self.sheet.spec.top)    
      385 +        return max(tops)                                      
      386 +                                                              
      387 +    @property                                                 
      388      def shadow_layer(self) -> int:
      389          """The volume's layer its shadow is marched in
      390          (:attr:`sphere.CloudVolume.shadow_layer`); -1 = none (
           not ready, or

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)
● Bash(python - <<'EOF'
      p='docs/source/asteroids.md'…)
  ⎿  Updated docs/source/asteroids.md (+5 -4)
      475    seen from the ground covered them (and dimmed the clouds rou
           nd a peak
      476    the ring passes behind). Around a planet with clouds
      477    (`FieldHaze.set_clouds_top`, which `World` and the demo set 
           for a ring
      478 -  centred on it, `SphericalClouds.top_radius`), the haze is dr
          -awn in two                                                    
      479 -  passes split where each ray enters the sphere of the clouds'
          - tops: what                                                   
      480 -  lies beyond it in the sky's bin, after the stars, before the
          - clouds; what                                                 
      481 -  lies before it after them, as before. Along a ray a ring and
          - the clouds                                                   
      478 +  centred on it, `PlanetClouds.top_radius`: its highest layer'
          +s, the                                                        
      479 +  cirrus's when there is one), the haze is drawn in two passes
          + split where                                                  
      480 +  each ray enters the sphere of the clouds' tops: what lies be
          +yond it in                                                    
      481 +  the sky's bin, after the stars, before the clouds (the cirru
          +s's far side                                                  
      482 +  included); what lies before it after them, as before. Along 
          +a ray a ring and the clouds                                   
      483    never overlap, so the split is exact. No cost seen (`ring_gr
           ound` 13.09
      484    / 13.12 ms, `ring_orbit` 5.93 / 5.58); `terrain_shots.py --r
           ing` gains
      485    `ring_behind_clouds`.
  ⎿  Updated CHANGELOG.md (+3 -3)
      573    write no depth, the ring's haze was drawn after them), veili
           ng and
      574    dimming them where it passed behind. Its haze is now split a
           t its
      575    planet's clouds' tops (`FieldHaze.set_clouds_top`,
      576 -  `SphericalClouds.top_radius`): what lies beyond them is draw
          -n before                                                      
      577 -  them. (#14)                                                 
      578 -- `SphericalClouds.sun_visibility` raised a `KeyError` once it
          -s cache of                                                    
      576 +  `PlanetClouds.top_radius`, the cirrus's included): what lies
          + beyond them                                                  
      577 +  is drawn before them, and before the cirrus's far side. (#14
          +)                                                             
      578 +- `CloudVolume.sun_visibility` raised a `KeyError` once its ca
          +che of                                                        
      579    shades had grown past its size, on a call with a cell alread
           y in it and
      580    a new one (in flight, after a few thousand 25 m cells): the 
           cache is now
      581    dropped before the lookup, not in it. (#11)
  ⎿  Updated src/world_builder/asteroids/haze.py (+2 -1)
      32    there, so nothing is counted twice. A ring around a planet wi
          th clouds
      33    (:meth:`FieldHaze.set_clouds_top`) is drawn in two passes, sp
          lit where the
      34    ray enters the sphere of the clouds' tops: what lies beyond i
          t before the
      35 -  clouds (in the sky's bin, after the stars), what lies before 
         -it after                                                       
      35 +  clouds (in the sky's bin, after the stars, before the cirrus'
         +s far side),                                                   
      36 +  what lies before it after                                    
      37    them. The clouds write no depth: drawn after them, the ring s
          een from the
      38    ground covered them. Along a ray a ring (well outside the pla
          net) and
      39    the clouds never overlap. Depth-tested, not written: a planet
           in

  Ran 1 shell command (ctrl+o to expand)

● The last commit's ring-cloud fix is resolved and all 813 tests pass. I'm now
  checking commit 10, which applied cleanly; its ring shadow reaches the
  billboards and deck but not the cirrus yet.

  Ran 1 shell command (ctrl+o to expand)

● HEAD is commit 10 right now, so its cirrus ring term goes in as a second
  fixup, the same way as before.

● Bash(git stash push -q -m commit11 && git status --short | head -3; grep -n
      'ringVisibility\|cloud' docs/source/shadows.md | sed -n 1,200p | grep -n
      'ring\|Ring' | h…)
  ⎿  5:14:| Ring shadow | rings | planet, clouds, air, objects | planned (I
     ) |
     59:494:  `shadows.glsl`'s `ringVisibility`: where a point's ray toward
     … +11 lines (ctrl+o to expand)
  ⎿  Updated CHANGELOG.md (+1 -7)
  ⎿  Updated docs/source/asteroids.md (+0 -12)
  ⎿  Updated scripts/demo_terrain.py (+0 -5)
  ⎿  Updated scripts/terrain_shots.py (+2 -6)
  ⎿  Updated src/world_builder/asteroids/haze.py (+3 -31)
  ⎿  … 7 more files changed
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='docs/source/shadows.md'…)
  ⎿   M CHANGELOG.md
      M docs/source/asteroids.md
      M scripts/demo_terrain.py
      12 files changed, 152 insertions(+), 13 deletions(-)
  ⎿  Updated docs/source/shadows.md (+2 -1)
  ⎿  Updated src/world_builder/clouds/shaders/cirrus.vert (+2 -1)
  ⎿  Updated CHANGELOG.md (+7 -1)
  ⎿  Updated docs/source/asteroids.md (+12 -0)
  ⎿  Updated scripts/demo_terrain.py (+5 -0)
  ⎿  … 9 more files changed
  ⎿  (a git step in this command can move the working tree, so these may be
     its changes, not edits) (ctrl+o to expand)

● Before continuing, ring shots (with --ring), to check the haze split against
  the cirrus:

  Read 3 files, ran 1 shell command (ctrl+o to expand)

● The ring looks right behind the clouds, but it's invisible in ring_ground and
  the planet looks mottled in ring_orbit. I'm rendering those same poses from
  the original branch tip (a26b4dc) to compare.

  Read 2 files, ran 1 shell command (ctrl+o to expand)

● No regression: ring_orbit matches the branch, and the empty ring_ground comes
  from main's #42 moving the level's start point. The ring still sits above that
  pose's frame, which I'll re-aim later without blocking the rebase, so I'm
  finishing commit 11 now.

  Ran 1 shell command (ctrl+o to expand)

● The rebase is done. Now I'll fold the two fixup! commits into their targets
  with an autosquash:

  Rebased onto origin/main (ctrl+o to expand)

● Git 2.43 ignores --autosquash unless the rebase is interactive. Running it as
  -i with GIT_SEQUENCE_EDITOR=true, which accepts the generated todo with no
  editor:

  Rebased onto origin/main (ctrl+o to expand)

● The working tree matches exactly before and after the rebase, and the branch's
  11 commits now sit on origin/main. I'm running the full test suite on each
  rebased commit in the background in the scratch worktree to confirm every one
  builds and passes.

  Ran 2 shell commands (ctrl+o to expand)

● While that runs, I'm back on the channel-order bug: if Panda reads 3- and
  4-channel RAM images as BGR(A), every numpy-built RGB/RGBA texture would be
  affected, including the branch's 3-channel deck-light table. I'll list all of
  those textures next.

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)

● The codebase confirms Panda's RAM layout is BGR(A), but two texture builders
  still read channels in RGB order: weather.build_weather_texture on main and
  noise.build_deck_light_texture on this branch, which makes the far ground
  shadow read the deck's sun shadowing. Next I'm checking the terrain's chunk
  table.

  Searched for 1 pattern (ctrl+o to expand)

● The chunk table already reorders. Before fixing the two builders, let me
  confirm the 2-channel order is R,G and the 3-channel one is B,G,R:

  Ran 1 shell command (ctrl+o to expand)

● Confirmed: 2-channel images are R,G, and 3-channel ones come back B,G,R.
  Fixing both builders in the working tree (uncommitted):

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/weather.py'…)
  ⎿  2 channels, expect [0.1, 0.2] got rgb [0.098 0.2   0.   ]
     3 channels, expect [0.1, 0.2, 0.30000000000000004] got rgb [0.098 0.2
       0.298]
     … +2 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/weather.py (+2 -1)
      244      else:
      245          tex.setup_cube_map(size, Texture.T_float, Texture.F_rg
           ba32)
      246          pad = np.zeros(cube.shape[:-1] + (4 - cube.shape[-1],)
           , dtype=np.float32)
      247 -        cube = np.concatenate([cube, pad], axis=-1)           
      247 +        # Panda keeps an RGBA texture's RAM image as BGRA.    
      248 +        cube = np.concatenate([cube, pad], axis=-1)[..., [2, 1
          +, 0, 3]]                                                      
      249      # Pages in face order, rows bottom-up = t up, as the star 
           field's.
      250      tex.set_ram_image(np.ascontiguousarray(cube, dtype=np.floa
           t32).tobytes())
      251      tex.set_magfilter(SamplerState.FT_linear)
  ⎿  Updated src/world_builder/clouds/noise.py (+2 -0)
      1372      tex = Texture("cloud_deck_light")
      1373      layout = Texture.F_rgb32 if channels == 3 else Texture.F_
            rg32
      1374      tex.setup_2d_texture(cosines, coverages, Texture.T_float,
             layout)
      1375 +    if channels == 3:                                        
      1376 +        table = table[..., ::-1]  # Panda keeps an RGB textur
           +e's RAM image as BGR                                         
      1377      tex.set_ram_image(np.ascontiguousarray(table, dtype=np.fl
            oat32).tobytes())
      1378      tex.set_magfilter(SamplerState.FT_linear)
      1379      tex.set_minfilter(SamplerState.FT_linear)

● Both now read back in order. Next, regression tests: restore the convective
  readback I stashed, and add a direct channel-order check for both builders.

  Ran 2 shell commands (ctrl+o to expand)

● Bash(python - <<'EOF'
      import re…)
  ⎿  Updated tests/test_cloud_shadows.py (+2 -0)
      22      MAX_LAYERS,
      23      NOISE_FEATURE,
      24      NOISE_SIZE,
      25 +    build_deck_light_texture,                                  
      26      build_noise_texture,
      27      cloud_shadow_on_sphere,
      28      far_cloud_depth,
     ...
      30      ground_shadow_table,
      31      pack_layer_params,
      32      sphere_column_peaks,
      33 +    with_shelf,                                                
      34      with_threshold,
      35  )
      36  from world_builder.clouds.sphere import SHADE_CACHE_SIZE, _nois
          e

● The earlier edit aborted before writing anything, so only the imports went in.
  Re-running the body edits:

● Bash(S=/tmp/claude-1000/-home-gl-development-WorldBuilder/d4124c88-f133-488e-a
      a3c-ec7b6e8c7853/scratchpad && cat > $S/tools/genus_test_edit.py <<'PYEOF'
      p='tests/tes…)
  ⎿  16 passed, 1 warning in 5.93s
  ⎿  Updated tests/test_cloud_shadows.py (+84 -2)
      184      return showbase
      185  
      186  
      187 -def test_the_shader_cloud_shadow_is_the_cpus(window, field, ta
          -ble):                                                         
      187 +@pytest.fixture(scope="module")                               
      188 +def convective():                                             
      189 +    """The genera's convective field (clouds.genera): its top 
          +rising with                                                   
      190 +    the depth, its cell octave, its anvil and shelf."""       
      191 +    raw = PRESETS[CloudType.CONVECTIVE].field                 
      192 +    volume = _noise(7)                                        
      193 +    peaks = sphere_column_peaks(volume, raw, R)               
      194 +    return with_shelf(with_threshold(raw, peaks, fbm_sigma(vol
          +ume, raw)), volume, R)                                        
      195 +                                                              
      196 +                                                              
      197 +def _genus_weather(cube):                                     
      198 +    """A four-channel weather cube from a two-channel one: a c
          +onvective                                                     
      199 +    depth and an anvil's spread that vary with the coverage.""
          +"                                                             
      200 +    depth = np.clip(1.4 * cube[..., 0] - 0.1, 0.0, 1.0)       
      201 +    spread = np.clip(2.0 * cube[..., 0] - 0.8, 0.0, 1.0)      
      202 +    return np.concatenate(                                    
      203 +        [cube, depth[..., None], spread[..., None]], axis=-1  
      204 +    ).astype(np.float32)                                      
      205 +                                                              
      206 +                                                              
      207 +def _read_back(window, colour, uniform, texture):             
      208 +    """One 4x4 offscreen draw of a card whose colour is *colou
          +r* (an rgb                                                    
      209 +    expression of the sampler *uniform*, *texture*), as its fi
          +rst pixel's                                                   
      210 +    8-bit rgb."""                                             
      211 +    from panda3d.core import CardMaker, Shader, Texture       
      212 +                                                              
      213 +    vertex = """#version 330                                  
      214 +in vec4 p3d_Vertex;                                           
      215 +uniform mat4 p3d_ModelViewProjectionMatrix;                   
      216 +void main() { gl_Position = p3d_ModelViewProjectionMatrix * p3
          +d_Vertex; }                                                   
      217 +"""                                                           
      218 +    fragment = f"""#version 330                               
      219 +{uniform}                                                     
      220 +out vec4 colour;                                              
      221 +void main() {{ colour = vec4({colour}, 1.0); }}               
      222 +"""                                                           
     … 66 more lines (ctrl+o to expand)

● The convective readback now matches the CPU twin as well (16 passed), so the
  genera shadow port is verified end to end on the GPU. I'm next checking every
  place that reads the clouds' shadow layer, covering shadows from both genera
  families.

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)
● I'm wiring every cloud receiver through the shadow functions and the map bake,
  with each family getting its own pair of maps. The shadow terms, the
  per-volume parameter slot and the CPU sun visibility all need to know about
  the other family, and I'm checking the slot budget before going further.

  Searched for 1 pattern (ctrl+o to expand)

● MAX_SHELLS is 6, so the deck sits at slot 6 at most and slot 7 is free.
  Writing the GLSL side:

  Ran 1 shell command (ctrl+o to expand)

● The branch's own commit 7 left cloudShadow's doc comment stranded above
  cloudFarDepth. I'll move it back while rewriting this section.

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/shaders/cloud_field.glsl'…)
  ⎿  src/world_builder/terrain/shaders/terrain_frag.glsl:64:uniform int   u
     CloudShadowLayer;
     src/world_builder/clouds/shaders/cloud_field.glsl:48:uniform int   uCl
     … +5 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/shaders/cloud_field.glsl (+84 -28)
      43  uniform vec4  cloudNearChart[4];
      44  uniform vec4  cloudFarChart[4];
      45  uniform int   cloudShadowMapOn;
      46 +// The clouds' shadow's layer in layerParams (-1: none), for it
         +s receivers                                                    
      47 +// (the terrain, the models: planetCloudShadow) and the maps' b
         +ake.                                                           
      48 +uniform int   uCloudShadowLayer;                               
      49 +// The other family's (clouds.genera: the stratus's for the con
         +vective                                                        
      50 +// clouds and their receivers, theirs for the stratus): its dec
         +k's layer in                                                   
      51 +// layerParams (sphere.OTHER_LAYER; -1: none), its weather, its
         + far table                                                     
      52 +// and its shadow maps.                                        
      53 +uniform int   uCloudShadowOther;                               
      54 +uniform samplerCube otherWeather;                              
      55 +uniform sampler2D otherDeckLight;                              
      56 +uniform sampler2D otherShadowNear;                             
      57 +uniform sampler2D otherShadowFar;                              
      58 +uniform vec4  otherNearChart[4];                               
      59 +uniform vec4  otherFarChart[4];                                
      60 +uniform int   otherShadowMapOn;                                
      61  
      62  const int LAYER_VEC4S = 13;
      63  // A noise lattice cell, in noise coordinates (noise.NOISE_FEAT
          URE).
     ...
      462  
      463  // The planet's clouds between a point (planet frame) and the 
           sun toward dir
      464  // (unit): the optical depth of their chord (cloudShadowChord)
           , marched in
      450 -// cloudShadowSteps, the weather taken where the chord enters 
          -the slab. 0                                                   
      451 -// above the slab, or where the weather leaves no cloud. What 
          -a cloud's                                                     
      452 -// shadow on the ground is: exp2(-depth). noise.cloud_shadow_o
          -n_sphere on                                                   
      453 -// the CPU.                                                   
      454 -float cloudShadowDepth(Field f, vec3 p, vec3 dir, int maxSteps
          -) {                                                           
      465 +// cloudShadowSteps, the weather (the field's own map) taken w
          +here the chord                                                
      466 +// enters the slab. 0 above the slab, or where the weather lea
          +ves no cloud.                                                 
      467 +// What a cloud's shadow on the ground is: exp2(-depth).      
      468 +// noise.cloud_shadow_on_sphere on the CPU.                   
      469 +float cloudShadowDepthIn(Field f, samplerCube weatherCube, vec
          +3 p, vec3 dir,                                                
      470 +                         int maxSteps) {                      
      471      vec3 start;
      472      float chord;
      473      if (maxSteps <= 0 || !cloudShadowChord(f, p, dir, start, c
           hord)) {
      474          return 0.0;
      475      }
     … 129 more lines (ctrl+o to expand)

● Declaring uCloudShadowLayer in three files broke the build, since a uniform
  can't be declared twice in one shader. Those files will now take it from
  cloud_field.glsl, and the terrain and models will use planetCloudShadow
  instead.

  Ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      edits = {…)
  ⎿  #version 330
     // The cloud shadow map's bake (clouds.shadow_map): per texel, the clo
     uds'
     … +19 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/terrain/shaders/terrain_frag.glsl (+8 -9)
      59  uniform sampler2D uStrata;        // the rock strata's tints by
           altitude (one row)
      60  uniform vec2  uStrataRange;       // the altitudes its first an
          d last texels span
      61  uniform int   uDebugMode;  // 0 shaded, 1 LOD, 2 biomes, 3 slop
          e, 4 normals, 5 skirts
      62 -// The planet's clouds' layer in layerParams whose shadow falls
         - here, -1 none                                                 
      63 -// (CloudVolume.bind_shadow_inputs).                           
      64 -uniform int   uCloudShadowLayer;                               
      62  uniform int   uMaxLevel;
      63  
      64  out vec4 fragColor;
     ...
      78  }
      79  
      80  // The share of the sun the planet's clouds leave here: their s
          hadow, marched
      84 -// through the deck's field toward the sun, or its mean where t
         -he pixel's                                                     
      85 -// footprint (metres) spans several clouds (cloud_field.glsl's 
         -cloudShadow).                                                  
      86 -// Clouds only come with air: the planet frame is the air's.   
      81 +// through the deck's field toward the sun (both families' with
         + genera), or                                                   
      82 +// its mean where the pixel's footprint (metres) spans several 
         +clouds                                                         
      83 +// (cloud_field.glsl's planetCloudShadow; uCloudShadowLayer and
         + Other, -1                                                     
      84 +// none: CloudVolume.bind_shadow_inputs). Clouds only come with
         + air: the                                                      
      85 +// planet frame is the air's.                                  
      86  float cloudShade(vec3 L, float footprint) {
      88 -    if (uCloudShadowLayer < 0 || uCloudShadowSteps <= 0 || uOwn
         -Atmosphere < 0) {                                              
      87 +    if ((uCloudShadowLayer < 0 && uCloudShadowOther < 0) || uCl
         +oudShadowSteps <= 0                                            
      88 +            || uOwnAtmosphere < 0) {                           
      89          return 1.0;
      90      }
      91      vec3 planet = uAtmoCam[uOwnAtmosphere].xyz + vRelPos;
      92 -    return cloudShadow(getField(uCloudShadowLayer), planet, L, 
         -uCloudShadowSteps,                                             
      93 -                       footprint);                             
      92 +    return planetCloudShadow(planet, L, uCloudShadowSteps, foot
         +print);                                                        
      93  }
      94  
      95  void main() {
  ⎿  Updated src/world_builder/objects/shaders/model_light.glsl (+4 -6)
       9  uniform int  uOwnAtmosphere;  // the body's atmosphere index, -
          1 = airless
      10  uniform mat3 uPlanetAxes;     // the body's own axes to the ren
          der frame
      11                                // (Terrain.rotation): its shadow
           map's
      12 -// The body's clouds' layer in layerParams whose shadow falls h
         -ere, -1 none                                                   
      13 -// (CloudVolume.bind_shadow_inputs).                           
      14 -uniform int  uCloudShadowLayer;                                
      12 +// The body's clouds' shadow: cloud_field.glsl's uCloudShadowLa
         +yer (and, with                                                 
      13 +// genera, uCloudShadowOther), -1 none (CloudVolume.bind_shadow
         +_inputs).                                                      
      14  
      15  // The body's vertical at a point rel (from the camera, render 
          frame), or
      16  // fallback where there is no body (open space).
     ...
      35      if (uOwnAtmosphere >= 0) {
      36          // Its air reddens the sun, and hides it past its termi
          nator.
      37          sunlight *= atmoSunlight(uOwnAtmosphere, planet, L);
      39 -        if (uCloudShadowLayer >= 0 && uCloudShadowSteps > 0) { 
      40 -            sunlight *= cloudShadow(getField(uCloudShadowLayer)
         -, planet, L,                                                   
      41 -                                    uCloudShadowSteps, footprin
         -t);                                                            
      38 +        if (uCloudShadowSteps > 0) {                           
      39 +            sunlight *= planetCloudShadow(planet, L, uCloudShad
         +owSteps, footprint);                                           
      40          }
      41      } else {
      42          sunlight *= ownShadow(rel, L, self);
  ⎿  Updated src/world_builder/clouds/shaders/cloud_shadow_map.frag (+0 -1)
       9  #pragma include "../../atmosphere/shaders/atmosphere.glsl"
      10  #pragma include "cloud_field.glsl"
      11  
      12 -uniform int  uCloudShadowLayer;                                
      12  uniform int  uMapSteps;
      13  uniform vec4 uMapChart[4];
      14  

● On the billboards, the stratus casts its shadow toward the sun using the full
  set of steps, so none at LOW. Next I'm moving to sphere.py to find where the
  root's inputs are set up.

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/shaders/cloud_sphere.vert'…)
  ⎿          # they are in (see _finish).
             if self._genera is not None:
                 self._weather_job = executor.submit_once(_genus_weather, *
     … +59 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/shaders/cloud_sphere.vert (+7 -0)
      133      if (uCloudFarReach > 0.0) {
      134          vSunTint *= exp2(-cloudFarDepth(getField(vLayer), plan
           et, sunDir, uCloudFarReach));
      135      }
      136 +    // And, with genera, the other family's clouds between it 
          +and the sun                                                   
      137 +    // (the convective clouds over the stratus): their whole c
          +hord, as on                                                   
      138 +    // the ground under them (none over the convective clouds'
          + own).                                                        
      139 +    if (uCloudShadowOther >= 0) {                             
      140 +        vSunTint *= exp2(-cloudShadowDepthIn(getField(uCloudSh
          +adowOther), otherWeather,                                     
      141 +                                             planet, sunDir, u
          +CloudShadowSteps));                                           
      142 +    }                                                         
      143      vTwilight *= eclipse;
      144      vSkyTint *= mix(ECLIPSE_SKY_FLOOR, 1.0, eclipse);
      145  }

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/sphere.py'…)
  ⎿  1014:            uCloudDebug=0,
     1306:            uCloudDebug=0,
     130:from world_builder.shadows import bind_defaults as bind_shadow_def
     aults
  ⎿  Updated src/world_builder/clouds/sphere.py (+14 -0)
      265  #: side, metres, and at most this many of them.
      266  SHADE_CELL = 25.0
      267  SHADE_CACHE_SIZE = 4096
      268 +#: The layer of ``layerParams`` that holds the other genus fam
          +ily's deck                                                    
      269 +#: field (:meth:`CloudVolume.take_shadow_of`): the last, past 
          +every shell's                                                 
      270 +#: and the deck's (at most ``cloud.MAX_SHELLS`` + 1 of them). 
      271 +OTHER_LAYER = MAX_LAYERS - 1                                  
      272  
      273  
      274  @functools.cache
     ...
      926          # weather they read, once made.
      927          self._shadow_nodes: list[NodePath] = []
      928          self._weather_texture = None
      929 +        # With genera, the other family (take_shadow_of): whos
          +e shadow this                                                 
      930 +        # one's billboards and receivers take too (its deck's 
          +field in                                                      
      931 +        # OTHER_LAYER, uCloudShadowOther once it is ready), an
          +d the volumes                                                 
      932 +        # that take this one's.                               
      933 +        self._other: CloudVolume | None = None                
      934 +        self._takers: list[CloudVolume] = []                  
      935 +        self._other_layer = PTA_int.empty_array(1)            
      936 +        self._other_layer[0] = -1                             
      937          # Their uCloudShadowLayer, shared: the deck's layer on
           ce ready
      938          self._shadow_layer = PTA_int.empty_array(1)
      939          # sun_visibility's shades by cell, for one sun and ste
           p count
     ...
      1009              exposure=float(exposure),
      1010              uPixelAngle=0.001,
      1011              nearFadeRadii=float(near_fade_radii),
      1012 +            uCloudShadowOther=self._other_layer,             
      1013 +            otherWeather=_no_weather(),                      
      1014              uCloudDebug=0,
      1015          )
      1016  

● Now the methods: take_shadow_of, keeping the mirrored slot and textures in
  sync, binding the receivers, and dropping the link on remove.

  Ran 1 shell command (ctrl+o to expand)
  ⎿  You've hit your 5-hour limit · resets 11pm (Europe/Paris)

● Background command "/tmp/claude-1000/-home-gl-development-WorldBuilder/d4124c8
8-f133-488e-aa3c-ec7b6e8c7853/scratchpad/tools/percommit.sh >
/tmp/claude-1000/-home-gl-development-WorldBuilder/d4124c88-f133-488e-aa3c-ec7b6
e8c7853/scratchpad/percommit.txt 2>&1" completed (exit code 0)
  ⎿  You've hit your 5-hour limit · resets 11pm (Europe/Paris)

❯ resume

  Read 1 file (ctrl+o to expand)

● Every one of the 11 rebased commits passes the full suite on its own (752 →
  813 tests). Back to the uncommitted two-family work; writing the CloudVolume
  methods:

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/sphere.py'…)
  ⎿  78:    PTA_int,
     936:        self._other_layer = PTA_int.empty_array(1)
     1421-                node.set_shader_input("deckLight", self._deck_lig
     … +8 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/sphere.py (+53 -1)
      127  )
      128  from world_builder.depth import REFLECTION_CAMERA_MASK
      129  from world_builder.quality import REBUILD
      130 +from world_builder.clouds.shadow_defaults import _no_deck_ligh
          +t, _no_weather                                                
      131  from world_builder.shadows import bind_defaults as bind_shadow
           _defaults
      132  from world_builder.terrain.cubesphere import QUADS, CubeGrid, 
           face_dirs, grid_for
      133  from world_builder.terrain.noise import native_library
     ...
      1075              self.deck.set_shader_input("uDeckLayer", self._de
            ck_layer)
      1076          for i, row in enumerate(params.reshape(-1, 4)):
      1077              self._layer_params[i] = LVecBase4f(*(float(v) for
             v in row))
      1078 +        self._pull_other()  # its row again: written over jus
           +t now                                                        
      1079 +        self._push_takers()                                  
      1080          self._apply_live(settings)
      1081  
      1082      def _apply_live(self, settings: CloudSettings) -> None:
     ...
      1308              deckLightCosMin=DECK_LIGHT_COS_MIN,
      1309              uCloudDebug=0,
      1310              uCloudShadowLayer=self._shadow_layer,
      1311 +            uCloudShadowOther=self._other_layer,             
      1312 +            otherWeather=_no_weather(),                      
      1313 +            otherDeckLight=_no_deck_light(),                 
      1314          )
      1315          self._shadow_nodes.append(node)
      1316          self._update_shadow_nodes()
      1317 +        self._pull_other()                                   
      1318  
      1319 +    def take_shadow_of(self, other: CloudVolume) -> None:    
      1320 +        """Take another volume's shadow too (the genera's oth
           +er family,                                                   
      1321 +        :mod:`genera`): its deck's field in :data:`OTHER_LAYE
           +R` of this                                                   
      1322 +        one's ``layerParams``, its weather and far table, for
           + this one's                                                  
      1323 +        billboards (the convective clouds over the stratus sh
           +ade them) and                                                
      1324 +        the nodes it shades (``uCloudShadowOther``: the groun
           +d under both).                                               
      1325 +        Kept in step with the other's set-up and settings.   
      1326 +                                                             
      1327 +        :param other: the volume whose shadow this one takes 
      1328 +        """                                                  
      1329 +        self._other = other                                  
      1330 +        other._takers.append(self)                           
      1331 +        self._pull_other()                                   
      1332 +                                                             
     … 50 more lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/shadow_defaults.py (+8 -0)
      63          cloudNearChart=PTA_LVecBase4f.empty_array(4),
      64          cloudFarChart=PTA_LVecBase4f.empty_array(4),
      65          cloudShadowMapOn=0,
      66 +        uCloudShadowOther=-1,                                  
      67 +        otherWeather=_no_weather(),                            
      68 +        otherDeckLight=_no_deck_light(),                       
      69 +        otherShadowNear=_no_cloud_map(),                       
      70 +        otherShadowFar=_no_cloud_map(),                        
      71 +        otherNearChart=PTA_LVecBase4f.empty_array(4),          
      72 +        otherFarChart=PTA_LVecBase4f.empty_array(4),           
      73 +        otherShadowMapOn=0,                                    
      74      )
      75  
      76  

● Now PlanetClouds: link the two volumes, bind receivers per family, and
  multiply the CPU shades.

  Searched for 1 pattern (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/planet.py'…)
  ⎿  262:            self.volume.take_shadow_of(self.stratus)
     263:            self.stratus.take_shadow_of(self.volume)
     357:    def shadow_families(self) -> int:
     442:        for family in range(self.shadow_families):
  ⎿  Updated src/world_builder/clouds/planet.py (+39 -10)
      256                      "channel": None if channel is None else f"
           {channel}:stratus",
      257                  },
      258              )
      259 +        if self.stratus is not None:                          
      260 +            # Each family takes the other's shadow (the convec
          +tive clouds'                                                  
      261 +            # on the stratus under them), and so does what the
          +y shade.                                                      
      262 +            self.volume.take_shadow_of(self.stratus)          
      263 +            self.stratus.take_shadow_of(self.volume)          
      264          self._deck_order: tuple | None = None
      265          self.sheet: CloudSheet | None = None
      266          if cirrus is not None:
     ...
      353          for bind in self._binders if binders is None else bind
           ers:
      354              bind(layer.root)
      355  
      351 -    def bind_shadow_inputs(self, node: NodePath) -> None:     
      356 +    @property                                                 
      357 +    def shadow_families(self) -> int:                         
      358 +        """How many volumes cast a shadow: two with genera (th
          +e convective                                                  
      359 +        one, then the stratus), else one."""                  
      360 +        return 2 if self.stratus is not None else 1           
      361 +                                                              
      362 +    def shadow_caster(self, family: int = 0) -> CloudVolume | 
          +None:                                                         
      363 +        """The volume whose shadow is *family*'s (0: :attr:`vo
          +lume`, 1: the                                                 
      364 +        stratus); None for a raymarch, which casts none yet.""
          +"                                                             
      365 +        caster = self.volume if family == 0 else self.stratus 
      366 +        return caster if isinstance(caster, CloudVolume) else 
          +None                                                          
      367 +                                                              
      368 +    def bind_shadow_inputs(self, node: NodePath, family: int =
          + 0) -> None:                                                  
      369          """Let the shaders on a node (the planet's terrain) da
           rken the sun
      353 -        under the clouds: the volume's shadow                 
      354 -        (:meth:`sphere.CloudVolume.bind_shadow_inputs`), kept 
          -across a change                                               
      355 -        of renderer. A raymarched volume casts none yet.      
      370 +        under the clouds: the volume's shadow, and with genera
          + the stratus's                                                
      371 +        too (:meth:`sphere.CloudVolume.bind_shadow_inputs`,   
      372 +        ``uCloudShadowOther``), kept across a change of render
          +er. A                                                         
      373 +        raymarched volume casts none yet.                     
      374  
      375          :param node: the node whose shaders include cloud_fiel
           d.glsl's
      358 -            ``cloudShadow``                                   
      376 +            ``planetCloudShadow``                             
     … 37 more lines (ctrl+o to expand)

● Now CloudShadowMaps: one near/far pair per family, the second bound to the
  receivers as other*.

  Ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/shadow_map.py'…)
  ⎿  tests/test_cloud_shadows.py:546:        assert maps._on[0] == 1 and ma
     ps.near.drawn is not None
     tests/test_cloud_shadows.py:549:        texture = maps.near.texture
     … +10 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/shadow_map.py (+97 -44)
      14  (mountain tops in the clouds) still marches: its ray does not s
          tart on the
      15  base.
      16  
      17 +With genera (:mod:`genera`), each family has its own pair: the 
         +stratus's                                                      
      18 +maps are on its own base's sphere, and the ground under both ta
         +kes the two                                                    
      19 +shadows' product (cloud_field.glsl's ``planetCloudShadow``, the
         + second                                                        
      20 +pair as ``otherShadowNear`` and the rest).                     
      21 +                                                               
      22  A map is drawn again (one frame of its buffer) once the camera 
          has moved a
      23  quarter of it, the sun has turned, the steps or the clouds' lay
          er changed:
      24  with the sun still, the shadow costs a fetch; with the sun swin
          ging, a
     ...
      62      def __init__(
      63          self,
      64          output: Any,
      60 -        clouds: Any,                                           
      65 +        bind: Any,                                             
      66          name: str,
      67          size: int,
      68          spacing: float,
     ...
       96          card.set_depth_test(False)
       97          card.set_depth_write(False)
       98          card.set_shader(_bake_shader())
       94 -        clouds.bind_shadow_inputs(card)                       
       99 +        bind(card)  # the field's inputs: its volume's bind_sh
          +adow_inputs                                                   
      100          self.chart = PTA_LVecBase4f.empty_array(4)  # the rece
           ivers' too
      101          self.steps = PTA_int.empty_array(1)
      102          card.set_shader_inputs(uMapChart=self.chart, uMapSteps
           =self.steps)
     ...
      144          least 256); 0 = none (the march per pixel)
      145      """
      146  
      147 +    #: The receivers' inputs of each family's pair: (near map,
          + far map,                                                     
      148 +    #: near chart, far chart, on).                            
      149 +    _INPUTS = (                                               
      150 +        ("cloudShadowNear", "cloudShadowFar", "cloudNearChart"
          +, "cloudFarChart",                                            
      151 +         "cloudShadowMapOn"),                                 
      152 +        ("otherShadowNear", "otherShadowFar", "otherNearChart"
          +, "otherFarChart",                                            
      153 +         "otherShadowMapOn"),                                 
      154 +    )                                                         
      155 +                                                              
      156      def __init__(
     … 170 more lines (ctrl+o to expand)

● Bash(sed -i 's/assert maps._on\[0\] == 1 and/assert maps._on[0][0] == 1 and/'
      tests/test_cloud_shadows.py && grep -rn '_on\[0\]' tests/ | head; timeout
      900 python -m…)
  ⎿  tests/test_casters.py:73:        assert casters._on[0] == 0  # no map:
      nothing to read
     tests/test_casters.py:89:        assert casters.casters == [] and cast
     … +14 lines (ctrl+o to expand)
  ⎿  Updated tests/test_cloud_shadows.py (+1 -1)
      543          sun = 0.8 * frame.up + 0.6 * frame.east
      544          clouds.set_sun(sun)
      545          maps.update(camera, sun / np.linalg.norm(sun), steps)
      546 -        assert maps._on[0] == 1 and maps.near.drawn is not Non
          -e                                                             
      546 +        assert maps._on[0][0] == 1 and maps.near.drawn is not 
          +None                                                          
      547          window.graphicsEngine.render_frame()
      548          window.graphicsEngine.render_frame()
      549          texture = maps.near.texture

● 815 pass. Now a test for the genera wiring. Let me see how existing tests
  build PlanetClouds with genera.

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)

● Bash(cat >> tests/test_clouds_genera.py <<'EOF'…)
  ⎿              assert clouds.shadow_families == 2
                 assert convective.shadow_layer >= 0 and stratus.shadow_lay
     er >= 0
     … +8 lines (ctrl+o to expand)
  ⎿  Updated tests/test_clouds_genera.py (+60 -0)
      609          _tile_under(clear[0] / np.linalg.norm(clear[0]), level
           ),
      610      )
      611      assert far.particles[:, 3].max() <= 140.0 * INTERIOR_GROWT
           H
      612 +                                                              
      613 +                                                              
      614 +def test_both_families_cast_their_shadow(showbase, executor, g
          +enera):                                                       
      615 +    """With genera each volume takes the other's shadow (its d
          +eck's field                                                   
      616 +    in OTHER_LAYER, its weather and far table), and so do the 
          +nodes the                                                     
      617 +    clouds shade: the ground under both takes the product of t
          +he two, on                                                    
      618 +    the CPU as in the shaders; a change of settings carries ov
          +er."""                                                        
      619 +    from panda3d.core import NodePath                         
      620 +                                                              
      621 +    from world_builder.clouds.cloud import CloudSettings      
      622 +    from world_builder.clouds.noise import LAYER_VEC4S        
      623 +    from world_builder.clouds.planet import PlanetClouds      
      624 +    from world_builder.clouds.sphere import OTHER_LAYER       
      625 +                                                              
      626 +    clouds = PlanetClouds(                                    
      627 +        showbase.render,                                      
      628 +        R,                                                    
      629 +        executor,                                             
      630 +        altitude=2200.0,                                      
      631 +        settings=CloudSettings(shells=2),                     
      632 +        genera=genera,                                        
      633 +    )                                                         
      634 +    ground = NodePath("ground")                               
      635 +    try:                                                      
      636 +        assert clouds.shadow_families == 2                    
      637 +        clouds.bind_shadow_inputs(ground)                     
      638 +        other = ground.get_shader_input("uCloudShadowOther")  
      639 +        frame = TangentFrame.at(*genera.focus)                
      640 +        clouds.update(frame.up * (R + 1000.0))                
      641 +        convective, stratus = clouds.volume, clouds.stratus   
      642 +        assert convective.shadow_layer >= 0 and stratus.shadow
          +_layer >= 0                                                   
      643 +        assert convective._other_layer[0] == stratus._other_la
          +yer[0] == OTHER_LAYER                                         
      644 +        assert ground.get_shader_input("uCloudShadowOther") ==
          + other  # in place                                            
      645 +        assert (                                              
      646 +            ground.get_shader_input("otherWeather").get_textur
          +e()                                                           
      647 +            is stratus._weather_texture                       
      648 +        )                                                     
     … 23 more lines (ctrl+o to expand)
● Bash(python - <<'EOF'
      p='tests/test_clouds_genera.py'…)
  ⎿  1 passed, 33 deselected, 1 warning in 6.20s
  ⎿  Updated tests/test_clouds_genera.py (+2 -2)
      644          assert ground.get_shader_input("uCloudShadowOther") ==
            other  # in place
      645          assert (
      646              ground.get_shader_input("otherWeather").get_textur
           e()
      647 -            is stratus._weather_texture                       
      647 +            == stratus._weather_texture                       
      648          )
      649  
      650          def rows(volume, layer):
     ...
      668          assert rows(stratus, OTHER_LAYER) == rows(convective, 
           convective._deck_layer)
      669      finally:
      670          clouds.remove()
      671 -    assert ground.get_shader_input("uCloudShadowOther").get_va
          -lue() is not None                                             
      671 +    assert convective._other_layer[0] == stratus._other_layer[
          +0] == -1                                                      

● Passes. Now a visual check with genera. Which shot poses look at the genera's
  features?

  Searched for 2 patterns, ran 1 shell command (ctrl+o to expand)

● cloud_bench.py draws clouds without terrain, so I'll use terrain_shots.py
  --genera, whose features sit around the start pose. Rendering the same poses
  on main and on the current tree:

  Read 2 files, ran 1 shell command (ctrl+o to expand)

● The cumulonimbus looks different: lumpier anvil, separate puffs in the tower.
  To tell whether that's the weather-channel fix or the new shadows, I'll render
  again with only the weather fix taken out.

  Read 1 file, ran 1 shell command (ctrl+o to expand)

● The tower's lumpiness persists without the weather fix, so I suspect commit
  7's far segment (cloudFarDepth) is the culprit, and I'm checking with shadows
  set to LOW to turn it off.

  Ran 1 shell command (ctrl+o to expand)
  ⎿  Interrupted · What should Claude do instead?

❯ Lumpiness of tower and anvil is expected at this point (with billboards)

● The rebase is done: 11-cloud-and-mountain-shadows now sits on origin/main
  (b825304). Your local main was stale at #32, so I rebased onto origin/main.
  All 11 commits are there with their original messages, and I ran the full test
  suite on each one separately: all pass (752 tests on the first, 813 on the
  last). I haven't pushed. Your remote branch still has the old tip (a26b4dc),
  so publishing will need a force-push.

  What I had to adapt inside the commits, because main rebuilt the clouds:
  - field.py is gone on main. The "no clouds" shadow defaults moved to a new
    clouds/shadow_defaults.py.
  - Main's PlanetClouds is the parent node of the cloud layers. Each layer sets
    its own defaults, so binding shadows onto the parent was silently ignored. I
    added PlanetClouds.bind_inputs(...), which binds onto each layer and binds
    again when the renderer changes.
  - PlanetClouds now forwards bind_shadow_inputs, sun_visibility, shadow_layer,
    shadow_field and top_radius to its volume. The shadow maps use these instead
    of private attributes.
  - Cloud shadow march: rebuilt on main's new march, so it handles the
    cumulonimbus's depth, cell octave and anvil shelf, and the CPU copy matches.
    Without genera it reduces to your original formula.
  - Cirrus now receives the eclipse, mountain and ring shadows, and is drawn
    after the ring's far haze (its draw order went from 2 to 3).
  - Commits 7 and 10 applied without conflicts but needed fixes. Commit 7's
    shader called a function my port had renamed, and commit 10 lacked the
    cirrus ring term. Both fixes are folded into those commits, so neither is
    broken on its own.

  Left uncommitted on top, for you to review:
  1. A texture-channel bug, in two places. Panda stores 3- and 4-channel
     textures in BGR(A) order:
     - Main's 4-channel genera weather texture (#46) never accounted for that,
       so the shader has been reading the convective depth as coverage and the
       coverage as depth.
     - Your 3-channel far shadow table has the same bug, so the far ground
       shadow and the deck's far lighting each read the other's channel.

     Both are fixed, with a test that reads the channels back from the GPU. This
     changes how main's genera look, so it's your call.
  2. Shadows from both genera layers, as you chose. The ground, sea and models
     take the product of the stratus and convective shadows, and each layer gets
     its own pair of shadow maps. Stratus billboards are also shaded by the
     convective clouds above them. New tests cover this, including a GPU check
     of the convective shadow against the CPU copy. The full suite passed before
     I added the last test, which passes on its own.

  Still open:
  - Marched clouds: not handled yet, as agreed. They cast no shadow and take no
    eclipse, mountain or ring shadow.
  - The stratus's distant flat layer isn't darkened by the convective clouds
    above it. Only the nearby stratus billboards are.
  - Cirrus casts no shadow, as agreed.
  - The ring_ground shot misses the ring. Main's #42 moved the start point
    south, and from there the ring stands 37–52° up, above the frame. The pose
    needs re-aiming.

  For your visual check: run python scripts/demo_terrain.py --genera --ring. F7
  cycles the shadow quality and C toggles the cirrus. All the shots I rendered
  are in /tmp/claude-1000/-home-gl-development-WorldBuilder/d4124c88-f133-488e-a
  a3c-ec7b6e8c7853/scratchpad/. The matching shots from main are in shots_main
  and shots_gen_main there.

  I also registered two scratch git worktrees there, main_wt and branch_wt. Tell
  me when to remove them.

[Image #1]

❯ I've committed and pushed your rebase. Any bugs left that you found ?
  If not, tackle the marched clouds. The feature was initially a bit buggy
  (jittering), point-grid patterns in the clouds. Can you fix it, and by the
  same occasion unify stratus and convective clouds ?
  I wonder if the marching of the clouds wouldn't also give the godrays for
  free ? Also it feels like the cloud shadow maps are kinda the same problem,
  aren't they ?
  Analyze the problem, benchmark. The idea behind marched clouds was primarily
  to reduce frame time when many or big clouds were on screen (e.g. several
  cumulonimbus). Offload as much work as possible on the GPU.

  Challenge my framing of the issue if you need to, be critic, ask any question
  you need

  Searched for 1 pattern, read 3 files, ran 1 shell command (ctrl+o to expand)

● I'm reading the raymarch code (march.py, the march, resolve and composite
  shaders) to find where the jitter and the dot grid come from, before writing
  the plan. A few more spots to check:

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)

● I've traced the march, resolve and composite shaders. One thing to check
  before asking you anything: which GPU this machine has, since §20's numbers
  come from an Intel UHD under WSL.

  Ran 1 shell command (ctrl+o to expand)

● No new bugs beyond what I listed last time:

  - The ring_ground pose needs re-aiming since #42 moved the start point.
  - The stratus's distant flat layer isn't darkened by the convective clouds
    above it.
  - The cirrus casts no shadow.

  On the marched clouds, the code points to the causes of what you're seeing. On
  the jitter:

  - The jitter cycle and the trace pattern work against each other. A pixel is
    traced every 4th frame, and the golden-ratio jitter advances once per frame,
    so each pixel's jitter jumps by 4 × 0.618 per visit, about half a step
    every time. Its samples never average out.
  - Pixels that aren't traced are reprojected with their block's sample
    distance, not their own, which smears edges.
  - Pixels with no history copy their block's one sample, which makes 2×2
    blocks.

  The dot grid most likely comes from the fresh pixels (1 in 4, given ⅓ weight).
  They aren't filtered spatially, so they stand out against the history around
  them.

  Where the time likely goes: each sample runs the sun march again (6–12 steps,
  2 fetches each), on top of 2–4 density fetches plus the billows. That's why a
  dense cloud costs so much.

  On your framing:

  - The shadow maps and the march's lighting are the same problem. Both need
    "transmittance toward the sun at a point". One sun-space map storing optical
    depth at several heights could light the march's samples (a fetch instead
    of a sun march), shade the ground, sea and models, and replace the far
    segment. That is the real unification.
  - Godrays are not free. They live in the air outside the slab, and the march
    only steps inside it. They become cheap once that map exists, if the march
    also steps the air below the clouds and its temporal accumulation removes
    the noise. Two catches: the scene would have to be darkened where the air is
    shadowed, and the shafts only show in haze or at dusk on this thin air.
  - Stratus and convective: the march today supports neither depth nor shelf. It
    uses a single slab with cloudDensityShifted, which is why genera fall back
    to billboards. Their bands never overlap, so a march can pick the family by
    height and pay for one field per sample.

  A few decisions that shape the plan:

● User answered Claude's questions:
  ⎿  · What role should the raymarch play once it is fixed? This decides whether
     it must keep imitating the billboards' look (VIEW_EXTINCTION_SCALE x2,
     billows standing in for sprite lumps) and whether the billboard tiles and
     the worker pool's cloud jobs can eventually go. → Default, billboards
     fallback (Recommended)
     · What does 'unify stratus and convective' mean to you? → One march, both
     families (Recommended)
     · Should godrays be part of this work? (As explained above, they are cheap
     but not free: the march must also step the air below the clouds against a
     sun-transmittance map.) → Not wanted
     · Which hardware and budget should the benchmarks target? This machine has
     a GTX 1070; §20's march numbers came from an Intel UHD under WSL. → GTX
     1070, 1600x900 (Recommended)

● The march is now the default with billboards as fallback, one march covers
  both families, no godrays, tuned for the GTX 1070 at 1600×900. Next I'm
  checking the march's Python side and its tests so I can plan against them.

  Ran 1 shell command (ctrl+o to expand)

● Writing the plan.

● Updated plan
  ⎿  /plan to preview
  ⎿  Error: The user doesn't want to proceed with this tool use. The tool use
     was rejected (eg. if it was a file edit, the new_string was NOT written to
     the file). The user provided the following reason for the rejection:  Write
     this in a file at project's root

  Ran 1 shell command (ctrl+o to expand)

● I wrote the plan to MARCH_PLAN.md at the project root, next to
  SHADOWS_PLAN.md. It's uncommitted.

  The plan rests on reading the code, not on measurements yet; benchmarking is
  Phase 0. In short:

  - Phase 0 – baseline: add flicker and dot-grid scores to cloud_bench.py, then
    time billboards against the march on the GTX 1070 at 1600×900, cumulus and
    genera poses, mean and p95.
  - Phase 1 – jitter and dot grid: I see three causes in the code:
    - The jitter advances every frame, but each pixel is traced only every 4th
      frame, so each visit lands about half a step away from the last and the
      samples never even out.
    - Untraced pixels are reprojected with their block's distance instead of
      their own.
    - Fresh pixels get a fixed ⅓ weight and no spatial filtering, so they stand
      out as a grid.

    The fixes are per-pixel jitter sequences, reprojection by each pixel's own
    distance, spatial reconstruction, and a variance clamp with a running sample
    count.
  - Phase 2 – one sun map: you were right that the shadow maps and the march's
    lighting are the same problem. Each map texel would store optical depth at 8
    heights instead of one total. That map lights the march's samples (a short
    local march plus 2 fetches, instead of 12–24 fetches) and shades the ground
    and models, for each genera family.
  - Phase 3 – genera in the march: pull the field and shadow state out of
    CloudVolume so both renderers share it. Then one march covers both slabs and
    picks the family by height. The march also gains the shadows it lacks
    today: none cast on the ground, none received from eclipses, mountains or
    rings.
  - Phase 4 – speed, then default: try a few speed levers, each kept only if it
    measures well, then make the march the default renderer. With the march, the
    sea's reflection would show only the distant flat cloud layer; the plan
    proposes a cheap ¼-resolution march for the reflection.

  I'll stop after each phase for you to check it in the demo. Shall I start
  Phase 0?

❯ go for 0

● Starting Phase 0. First, the bench script in full:

  Read 1 file (ctrl+o to expand)

● I've made the Phase 0 changes to cloud_bench.py (p95, stability and grid
  scores, steps-per-pixel readout, pool shutdown on every exit) and added a "no
  billows" debug bit to the march shader. Next I'll check the presets and what
  flight_bench.py accepts.

  Searched for 1 pattern, ran 2 shell commands (ctrl+o to expand)

● flight_bench.py already takes --cloud-renderer through the shared arguments.
  Now the cloud_bench.py edits:

● Bash(python - <<'EOF'
      p='scripts/cloud_bench.py'…)
  ⎿      if "llvmpipe" in renderer:
             print("warning: CPU rasteriser; on WSL set GALLIUM_DRIVER=d3d1
     2 for the GPU")
     … +19 lines (ctrl+o to expand)
  ⎿  Updated scripts/cloud_bench.py (+152 -25)
       8          [--half-res | --no-half-res]
       9          [--uniform-coverage C] [--no-cirrus] [--cloud-debug BIT
          S]
      10          [--move M] [--genera] [--interior-growth | --no-interio
          r-growth]
      11 -        [--interactive]                                        
      11 +        [--stability N] [--interactive]                        
      12  
      13      --out              where the PNGs go (default ./cloud_bench
          )
      14      --pose             the poses to time (default: all, in the 
          order below)
     ...
      32                         (CloudSettings.interior_growth), over th
          e preset's
      33      --move             fly the camera (and the probe) along the
           view at M
      34                         metres a frame while timing: reprojectio
          n under motion
      35 +    --stability        also read N frames back (moving with --m
         +ove) and score                                                 
      36 +                       the clouds' steadiness and the raymarch'
         +s block grid                                                   
      37 +                       (below)                                 
      38      --interactive      fly the probe in and out of the clouds i
          n a window
      39                         (keys below); no timing
      40  
     ...
      66  (2 km up over the stratus deck, looking down), stratus_below (3
          00 m up under
      67  it), genera_orbit (3 radii up).
      68  
      69 +Each pose prints the frame's median and 95th percentile (ms), t
         +he frame                                                       
      70 +with the clouds hidden, and the clouds' cost (the medians' diff
         +erence). With                                                  
      71 +debug bit 16 (the raymarch's steps per pixel, red) it also prin
         +ts the mean                                                    
      72 +steps over the clouds' pixels.                                 
      73 +                                                               
      74 +With --stability N, over the clouds' pixels (those that change 
         +when the                                                       
      75 +clouds are hidden), from N frames read back after the timing:  
      76 +                                                               
      77 +* **spread**: each pixel's brightness range over the frames, 8-
         +bit steps:                                                     
      78 +  its mean and 99th percentile (the billboards' still frames: 0
         +). Moving,                                                     
      79 +  the view changes too: compare renderers at the same --move;  
      80 +* **grid**: how much one pixel of each march block (block x blo
         +ck pixels,                                                     
      81 +  CloudSettings.march_resolution) stands out from the others: e
         +ach frame's                                                    
      82 +  brightness less its 3 x 3 mean, its RMS by the pixel's place 
         +in its block;                                                  
      83 +  the largest over the smallest, less 1, averaged over the fram
         +es (0: no                                                      
      84 +  pattern; the billboards measure it at the march's block too).
      85 +                                                               
      86  Interactive (``--interactive``): the probe flies, the camera fo
          llows it.
      87  
      88      W/S A/D R/F   the probe forward/back, left/right, up/down (
          along the view)
     … 208 more lines (ctrl+o to expand)
● Now debug bit 32 (no billows) in the march shader, and its documentation:

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/shaders/cloud_march.frag'…)
  ⎿  366:        props.set_rgba_bits(16, 16, 16, 16)
     369:        props.set_rgba_bits(8, 8, 8, 8)
     372:    props.set_depth_bits(24 if GraphicsOutput.RTP_depth in targets
     … +5 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/shaders/cloud_march.frag (+2 -1)
      55  // the coverage holds on average. Clear air is found first with
           the most the
      56  // billows could add: cloud alone costs the noise twice over.
      57  float marchDensity(Field f, vec3 p, float shift) {
      58 -    float window = (f.threshHi - f.threshLo) * uDetailErosion; 
      58 +    float window = (uCloudDebug & 32) != 0 ? 0.0  // no billows
         + (profiling)                                                   
      59 +                                           : (f.threshHi - f.th
         +reshLo) * uDetailErosion;                                      
      60      float most = cloudDensityShifted(f, p, shift + window * uBi
          llowMean);
      61      if (most <= 0.0 || window <= 0.0) {
      62          return most;
  ⎿  Updated src/world_builder/clouds/shaders/cloud_field.glsl (+2 -2)
      23  // two-channel map).
      24  uniform samplerCube weatherMap;
      25  // A/B switches for profiling (bits, 0 = normal): 1 no sun marc
          h, 2 no
      26 -// atmosphere, 4 two octaves only, 8 flat colour after the dens
         -ity, 16 (the                                                   
      27 -// raymarch) its steps per pixel, red over the budget.         
      26 +// atmosphere, 4 two octaves only, 8 flat colour after the dens
         +ity; the                                                       
      27 +// raymarch's: 16 its steps per pixel, red over the budget, 32 
         +no billows.                                                    
      28  uniform int   uCloudDebug;
      29  // The far clouds' averages, by the sun's cosine (x, from deckL
          ightCosMin to
      30  // 1) and the weather's coverage (y): the deck's lighting

● Quick smoke run of the tooling, on two poses with few frames:

  Ran 1 shell command (ctrl+o to expand)

● The tooling works: about 6–7 ms of cloud cost at these poses, with p95 about 4
  ms above the median. Next I'll run the full sequential baseline in the
  background, covering cumulus, genera, the march breakdown and flight_bench.py.

  Ran 2 shell commands (ctrl+o to expand)

● The baseline is running in the background. I'll keep the GPU and CPU idle
  until it's done, so the timings stay clean. Meanwhile, a lint check on the
  script, which costs almost nothing:

  Ran 1 shell command (ctrl+o to expand)

● Bash(ruff format scripts/cloud_bench.py && git diff --stat)
  ⎿  1 file reformatted
      scripts/cloud_bench.py                            | 180 +++++++++++++
     ++++++---
     … +3 lines (ctrl+o to expand)
  ⎿  Updated scripts/cloud_bench.py (+8 -5)
      225      lum = luminance(frame)
      226      padded = np.pad(lum, 1, mode="edge")
      227      h, w = lum.shape
      228 -    box = sum(                                                
      229 -        padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]      
      230 -        for dy in (-1, 0, 1)                                  
      231 -        for dx in (-1, 0, 1)                                  
      232 -    ) / 9.0                                                   
      228 +    box = (                                                   
      229 +        sum(                                                  
      230 +            padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]  
      231 +            for dy in (-1, 0, 1)                              
      232 +            for dx in (-1, 0, 1)                              
      233 +        )                                                     
      234 +        / 9.0                                                 
      235 +    )                                                         
      236      residual = lum - box
      237      y, x = np.mgrid[0:h, 0:w]
      238      place = (y % block) * block + (x % block)

● The baseline is still running. It covers cumulus with both renderers at HIGH
  and MID, still and moving, then genera, the march breakdown, and flight_bench.
  I'll write it up when it finishes.

● Background command "/tmp/claude-1000/-home-gl-development-WorldBuilder/d4124c8
8-f133-488e-aa3c-ec7b6e8c7853/scratchpad/p0/run.sh >
/tmp/claude-1000/-home-gl-development-WorldBuilder/d4124c88-f133-488e-aa3c-ec7b6
e8c7853/scratchpad/p0/results.txt 2>&1" completed (exit code 0)

  Searched for 1 pattern (ctrl+o to expand)

● Two results already contradict the plan's premise on this GPU, so I'm adding a
  --march-resolution option to the bench to split the march's cost into
  per-pixel and fixed-per-frame parts.

● Bash(python - <<'EOF'
      p='scripts/cloud_bench.py'…)
  ⎿  === march high resolution 0.25
     pose               frame       p95  no clouds    clouds  billboards
     below             3.20ms    4.03ms     1.00ms    2.19ms         0k
     … +17 lines (ctrl+o to expand)
  ⎿  Updated scripts/cloud_bench.py (+11 -1)
       8          [--half-res | --no-half-res]
       9          [--uniform-coverage C] [--no-cirrus] [--cloud-debug BIT
          S]
      10          [--move M] [--genera] [--interior-growth | --no-interio
          r-growth]
      11 -        [--stability N] [--interactive]                        
      11 +        [--march-resolution R] [--stability N] [--interactive] 
      12  
      13      --out              where the PNGs go (default ./cloud_bench
          )
      14      --pose             the poses to time (default: all, in the 
          order below)
     ...
      20      --renderer         billboards (the default) or march (the r
          aymarch)
      21      --half-res         the clouds at half resolution (CloudSett
          ings
      22                         .half_resolution), over the preset's cho
          ice
      23 +    --march-resolution the raymarch's resolution over the view'
         +s (1, 0.5,                                                     
      24 +                       0.25: CloudSettings.march_resolution), o
         +ver the preset's                                               
      25      --uniform-coverage the same coverage everywhere (no weather
           systems,
      26                         climate or terrain): comparisons that do
           not depend
      27                         on where the pose is
     ...
      698          help="larger billboards inside clouds, over the preset
           's choice",
      699      )
      700      parser.add_argument(
      701 +        "--march-resolution",                                 
      702 +        type=float,                                           
      703 +        default=None,                                         
      704 +        help="the raymarch's resolution over the view's, over 
          +the preset's",                                                
      705 +    )                                                         
      706 +    parser.add_argument(                                      
      707          "--stability",
      708          type=int,
      709          default=0,
     ...
      721          settings = replace(settings, interior_growth=args.inte
           rior_growth)
      722      if args.half_res is not None:
      723          settings = replace(settings, half_resolution=args.half
           _res)
      724 +    if args.march_resolution is not None:                     
      725 +        settings = replace(settings, march_resolution=args.mar
          +ch_resolution)                                                
      726      bench = CloudBench(
      727          settings,
      728          cirrus=not args.no_cirrus,

● The march costs about 2.2 ms per frame plus a part that scales with traced
  pixels: 3.1, 6.5 and 18.9 ms at 1/16, ¼ and all pixels. I'm writing this
  baseline up as §23 of performance.md next.

  Ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='docs/source/performance.md'…)
  ⎿  172:| 20 km | ~290 | 1.5 ms |
     211:| 20 km | ~270–320 (to the horizon) | 5–6 |
     212:| 200 km | ~120 | 9 |
     … +7 lines (ctrl+o to expand)
  ⎿  Updated docs/source/performance.md (+71 -0)
      1113  Under the anvil the sheet fills the view: 1.6 times the frame
             at MID, 2.4
      1114  at HIGH. Elsewhere it is a few milliseconds at most.
      1115  
      1116 +## 23. The raymarch on a strong GPU: the baseline (#45)      
      1117 +                                                             
      1118 +Before reworking the raymarch ([the raymarch](clouds.md#the-r
           +aymarch)), the                                               
      1119 +same comparison as §20–22 on a **GTX 1070** (Linux, 1600×900,
           + offscreen, one                                              
      1120 +session), with `scripts/cloud_bench.py`'s new columns: the fr
           +ame's 95th                                                   
      1121 +percentile, and with `--stability 16` the clouds' **spread** 
           +(each cloud                                                  
      1122 +pixel's brightness range over 16 frames, 8-bit steps: mean / 
           +99th                                                         
      1123 +percentile) and **grid** (how much one pixel of each march bl
           +ock stands out                                               
      1124 +from the others, 0 = none; the billboards measure their own t
           +exture's                                                     
      1125 +0.003–0.05 at the same block). Cloud cost = the frame less th
           +e frame with                                                 
      1126 +the clouds hidden (0.7–1.6 ms).                              
      1127 +                                                             
      1128 +**Cumulus, HIGH, still** (cloud cost ms; spread mean / p99; g
           +rid):                                                        
      1129 +                                                             
      1130 +| Pose | Billboards | March | March spread | March grid |    
      1131 +|---|---|---|---|---|                                        
      1132 +| Below | 1.92 | 3.08 | 2.8 / 18 | 0.066 |                   
      1133 +| Start | 2.67 | 4.84 | 2.0 / 14 | 0.042 |                   
      1134 +| Inside the slab | 4.95 | 6.50 | 1.3 / 15 | 0.035 |         
      1135 +| Above | 3.23 | 4.77 | 2.8 / 16 | 0.057 |                   
      1136 +| 20 km up | 1.64 | 2.45 | 8.6 / 57 | 0.135 |                
      1137 +| Orbit | 2.21 | 1.62 | 0.4 / 3 | 0.006 |                    
      1138 +| Sunset | 2.88 | 3.57 | 1.2 / 8 | 0.054 |                   
      1139 +| A model in a dense cloud | 10.92 | 6.95 | 0.7 / 8 | 0.009 |
      1140 +                                                             
      1141 +The billboards' still frames do not change (spread 0). At MID
           + the                                                         
      1142 +billboards (half resolution, a quarter of the billboards) cos
           +t 0.7–2.3 ms;                                                
      1143 +the march, whose resolution MID does not lower, 3.1–5.5 ms, a
           +s unsteady as                                                
      1144 +at HIGH (grid 0.07 everywhere).                              
      1145 +                                                             
      1146 +**Moving** (10 m a frame, HIGH): the view's own change domina
           +tes the                                                      
      1147 +spread for both (billboards 6–12 mean, march 5–13), but the m
           +arch's                                                       
      1148 +**grid** rises to 0.13–0.29 (billboards 0.01–0.02, the sea's 
           +glints                                                       
      1149 +0.21 at `below`): the dot grid seen in flight.               
      1150 +                                                             
      1151 +**Genera, billboards** (the march cannot draw them yet): HIGH
           + 2.0–2.7 ms                                                  
      1152 +far off, 7.8 ms over the anvil and in the congestus, 12.3 / 1
           +4.0 ms above /                                               
     … 37 more lines (ctrl+o to expand)

● Now one line in clouds.md's Testing section for the new options, and the plan
  revised to match what the numbers showed:

● Bash(python - <<'EOF'
      p='docs/source/clouds.md'…)
  ⎿  816 passed, 5 warnings in 62.56s (0:01:02)
  ⎿  Updated docs/source/clouds.md (+4 -1)
      408  either renderer, still or moving (`--move`). Its `model_in_clo
           ud` pose puts
      409  a black ball, standing in for a game's model, half into a clou
           d, and
      410  `--interactive` flies it in and out of the clouds, switching r
           enderers with
      411 -M. On WSL, run it with `GALLIUM_DRIVER=d3d12`, or Mesa draws o
          -n the CPU.                                                    
      411 +M. `--stability N` scores the clouds' steadiness over N frames
          + read back                                                    
      412 +(each pixel's brightness spread, and the raymarch's block grid
          +: see the                                                     
      413 +script), and each pose's frame comes with its 95th percentile 
      414 +([performance §23](performance.md)). On WSL, run it with `GALL
          +IUM_DRIVER=d3d12`, or Mesa draws on the CPU.                  
  ⎿  Updated MARCH_PLAN.md (+10 -0)
      25  
      26  Phases stop for your demo check after each one (memory: stop-af
          ter-each-phase, visual-confirmation).
      27  
      28 +## Phase 0 findings (done: `docs/source/performance.md` §23)   
      29 +                                                               
      30 +Measured on the GTX 1070. They change the plan's priorities:   
      31 +                                                               
      32 +- **Over fair-weather cumulus the billboards are cheaper** (HIG
         +H cloud cost 1.6–4.9 ms vs the march's 2.5–6.5). The march wins
         + in a dense cloud (6.9 vs 10.9 ms) and near the genera, where t
         +he billboards cost 12–27 ms at HIGH. In the world it already wi
         +ns on the main thread: low flight 11.2 / 19.1 vs 12.2 / 21.4 ms
         + mean / p95, the billboards' tiles and sorts spared.           
      33 +- **The march's cost is a fixed ~2.2 ms a frame** (depth passes
         +, full-resolution resolve, composite, near range) **plus its tr
         +aced pixels.** At `inside`: 3.1 / 6.5 / 18.9 ms tracing 1/16, ¼
         + and all of them per frame.                                    
      34 +- **Lighting is not the lever here.** The per-sample sun march 
         +is 0.4–1.4 ms, the billows 0.4–1.4 ms, the air nothing measurab
         +le. So Phase 2's sun-transmittance map stays for its other reas
         +ons (shadows, both families, the far segment). It moves after P
         +hase 3, and is kept only if the genera's tall sun marches make 
         +lighting matter.                                               
      35 +- **Instability, quantified.** Still frames flicker (spread 1.2
         +–3 mean, p99 8–20 levels; 8.6 / 57 at 20 km). The block grid is
         + 0.04–0.13 still and **0.13–0.29 in motion**, against the billb
         +oards' 0.01–0.02.                                              
      36 +- **The levers, in order:** Phase 1's reconstruction, good enou
         +gh to trace 1/16 of the pixels a frame at HIGH (≈2.5–3.4 ms, th
         +e billboards' cost); then the fixed cost; then the genera (Phas
         +e 3).                                                          
      37 +                                                               
      38  ## Phase 0: Baseline and measuring tools
      39  
      40  - `scripts/cloud_bench.py` gains:

● Phase 0 is done. Everything was measured on the GTX 1070 at 1600×900 and is
  written up in §23 of performance.md (docs/source/performance.md). The tests
  still pass (816).

  cloud_bench.py now reports:
  - the frame's p95 next to the median;
  - --stability N: each cloud pixel's brightness range over N frames read back,
    mean and 99th percentile;
  - a grid score: how much one pixel of each march block stands out from the
    others;
  - the mean steps per pixel, with debug bit 16.

  It also takes --march-resolution, and shuts the worker pool down on every exit
  path. The march shader gained debug bit 32 (no billows).

  On this GPU, several findings go against the plan:

  - Over fair-weather cumulus the billboards are cheaper. At HIGH they cost
    1.6–4.9 ms against the march's 2.5–6.5 ms.
  - The march wins where it was meant to: in a dense cloud (6.9 vs 10.9 ms) and
    near the genera, where billboards cost 12–27 ms at HIGH. At MID the
    billboards stay under 4.3 ms even there.
  - In the full world the march already wins, on the CPU. Low flight takes 11.2
    ms mean / 19.1 ms p95 against 12.2 / 21.4 with billboards, because the main
    thread no longer handles the billboard tiles and sorting.
  - The march costs about 2.2 ms fixed per frame, from its extra passes, plus a
    part that scales with the pixels it traces. At the inside pose: 3.1, 6.5 and
    18.9 ms tracing 1/16, ¼ and all of the pixels per frame.
  - Lighting isn't the expensive part here. The per-sample sun march is 0.4–1.4
    ms, the billows about the same, the air nothing measurable. So the
    sun-transmittance map won't buy speed on this GPU; it remains worth doing
    for shadows and the two genera families.
  - The instability now has numbers. Still frames flicker: on average a pixel
    moves 1.2–3 brightness levels out of 255, and the worst 1 % move 8–20 (57 at
    20 km up). The dot grid scores 0.04–0.13 still and 0.13–0.29 in motion,
    against 0.01–0.02 for billboards. The grid you saw is mainly a motion
    problem.

  Revised priorities, recorded in MARCH_PLAN.md:
  1. Phase 1: fix the temporal reconstruction, well enough to trace only 1/16 of
     the pixels per frame at HIGH. That's about 2.5–3.4 ms, the billboards'
     cost.
  2. Then the fixed cost of the march's passes.
  3. Phase 3: genera in the march, plus the march's shadows.
  4. Phase 2 last: the sun-transmittance map, kept only if the cumulonimbus's
     tall sun marches make lighting matter.

  Nothing is committed: the bench script, the debug bit, the docs and the plan
  update are all uncommitted. The raw results are in
  /tmp/claude-1000/-home-gl-development-WorldBuilder/d4124c88-f133-488e-aa3c-ec7
  b6e8c7853/scratchpad/p0/results.txt, the shots next to it.

  Do you want Phase 1 next, with this new order?

❯ go for 1

● Starting Phase 1. First the march's pass machinery in march.py:

  Read 1 file (ctrl+o to expand)

● The near range traces every pixel each frame and only averages its jitter,
  while the far range handles the gathering with the Bayer order and history.
  I'm checking what the tests pin down before changing the resolve.

  Ran 1 shell command (ctrl+o to expand)

● I've got a plan for the new resolve: each pixel blends its clamped history
  with a reconstruction from the four nearest traced samples, using a
  distance-based Gaussian. Next I'll write it, with history reprojected in two
  steps and the sample count stored in the distance target's spare channel.

● Write(src/world_builder/clouds/shaders/cloud_resolve.frag)
  ⎿  Added 109 lines, removed 38 lines
       1  #version 330
       2  // The raymarch's full resolution (clouds.march), gathered over
           the frames:
       3  // cloud_march.frag traces one pixel of every uBlock x uBlock b
          lock a frame;
       4 -// the rest come from the last frame's result, moved to where t
         -he camera now                                                  
       5 -// sees them (reprojected along the view ray to the cloud's dis
         -tance), held                                                   
       6 -// within the range of this frame's samples around them so a ch
         -ange shows at                                                  
       7 -// once rather than trailing behind. A traced pixel blends into
         - its history,                                                  
       8 -// which averages the steps' jitter away over the visits.      
       4 +// every pixel here blends this frame's samples around it (reco
         +nstructed                                                      
       5 +// bilinearly from the four nearest, weighted by how near the n
         +earest is)                                                     
       6 +// into the last frame's result, moved to where the camera now 
         +sees it                                                        
       7 +// (reprojected along the view ray to the pixel's own cloud dis
         +tance) and                                                     
       8 +// held within the spread of this frame's samples around it, so
         + a change                                                      
       9 +// shows at once rather than trailing behind. How much a pixel 
         +takes from                                                     
      10 +// this frame falls with how many samples it has gathered (its 
         +count), down                                                   
      11 +// to uMinWeight: still, the steps' jitter averages away over m
         +any visits;                                                    
      12 +// no pixel of a block is treated apart from the others, so no 
         +grid shows.                                                    
      13  //
      10 -// The cloud is premultiplied; its distance (kilometres, 0: non
         -e) goes on with                                                
      11 -// it, for the next frame and for the composite's depth test. T
         -he textures may be larger                                      
      12 -// than what is drawn in them (padded to powers of two): pixels
         -, not UVs,                                                     
      13 -// except for the history's filtered read.                     
      14 +// The cloud is premultiplied; its distance (kilometres, 0: non
         +e) and its                                                     
      15 +// count go on with it (r and g), for the next frame and for th
         +e composite's                                                  
      16 +// depth test. The textures may be larger than what is drawn in
         + them (padded                                                  
      17 +// to powers of two): pixels, not UVs, except for the history's
         + filtered                                                      
      18 +// read.                                                       
      19  uniform sampler2D uMarch;            // this frame's samples, o
          ne a block
      20  uniform sampler2D uMarchDistance;
      21  uniform sampler2D uHistory;          // the last frame's result
          , full resolution
      17 -uniform sampler2D uHistoryDistance;                            
      22 +uniform sampler2D uHistoryDistance;  // r: distance, km; g: sam
         +ples gathered                                                  
      23  uniform mat4  uClipToRel;            // clip space -> relative 
          to the camera (world axes)
      24  uniform mat4  uPrevRelToClip;        // relative to the camera 
          -> the last frame's clip space
      25  uniform int   uBlock;
      26  uniform ivec2 uOffset;
      27  uniform vec2  uViewSize;
      28  uniform int   uHistoryValid;         // 0: no usable last frame
           (first, resized, jumped)
      24 -uniform float uFreshWeight;          // a traced pixel's share 
         -over its history                                               
      29 +uniform float uMinWeight;            // this frame's least shar
         +e, with a full count                                           
      30  
      31  layout(location = 0) out vec4 fragColor;
      32  layout(location = 1) out vec4 fragDistance;
      33  
      34  // How far to look along a ray with no cloud on it (the sky at 
          infinity).
      35  const float NO_CLOUD = 1.0e7;
      36 +// The history is held within this many standard deviations of 
         +this frame's                                                   
      37 +// samples around the pixel.                                   
      38 +const float CLAMP_SIGMA = 1.25;                                
      39 +// A sample's reach, in its spacing (uBlock pixels): the Gaussi
         +an weight a                                                    
      40 +// pixel gives the nearest one.                                
      41 +const float REACH = 0.3;                                       
      42  
      43  vec3 rayDir(vec2 ndc) {
      44      vec4 near = uClipToRel * vec4(ndc, -1.0, 1.0);
     ...
       46      return normalize(far.xyz / far.w - near.xyz / near.w);
       47  }
       48  
       49 +// Where the point along this pixel's ray at distance km was i
          +n the last                                                    
       50 +// frame's view, as a texel coordinate of the history; false o
          +ff it.                                                        
       51 +bool reproject(vec3 dir, float km, out vec2 at) {             
       52 +    vec3 point = dir * (km > 0.0 ? km * 1000.0 : NO_CLOUD);   
       53 +    vec4 previous = uPrevRelToClip * vec4(point, 1.0);        
       54 +    vec2 uv = previous.xy / previous.w * 0.5 + 0.5;           
       55 +    at = uv * uViewSize;                                      
       56 +    return previous.w > 0.0 && all(greaterThanEqual(uv, vec2(0
          +.0)))                                                         
       57 +        && all(lessThanEqual(uv, vec2(1.0)));                 
       58 +}                                                             
       59 +                                                              
       60  void main() {
       61      ivec2 pixel = ivec2(gl_FragCoord.xy);
       40 -    ivec2 block = pixel / uBlock;                             
       62      ivec2 last = ivec2(ceil(uViewSize / float(uBlock))) - 1;
       42 -    bool fresh = pixel - block * uBlock == uOffset;           
       43 -    vec4 current = texelFetch(uMarch, min(block, last), 0);   
       44 -    float distance = texelFetch(uMarchDistance, min(block, las
          -t), 0).r;                                                     
       63 +    float block = float(uBlock);                              
       64  
       46 -    // The range of this frame's four samples around the pixel
          -.                                                             
       47 -    ivec2 corner = ivec2(floor((vec2(pixel) + 0.5) / float(uBl
          -ock) - 0.5));                                                 
       48 -    vec4 lo = current;                                        
       49 -    vec4 hi = current;                                        
       65 +    // The march's samples around: sample k sits at view pixel
          + k * block +                                                  
       66 +    // uOffset. The four nearest, bilinearly; the 3 x 3 around
          +, for the clamp.                                              
       67 +    vec2 f = (vec2(pixel) - vec2(uOffset)) / block;           
       68 +    ivec2 base = ivec2(floor(f));                             
       69 +    vec2 w = f - vec2(base);                                  
       70 +    vec4 reconstructed = vec4(0.0);                           
       71 +    float reconstructedKm = 0.0;                              
       72 +    float kmWeight = 0.0;                                     
       73 +    float nearest = 1e9;                                      
       74      for (int i = 0; i < 4; ++i) {
       51 -        ivec2 at = clamp(corner + ivec2(i & 1, i >> 1), ivec2(
          -0), last);                                                    
       75 +        ivec2 corner = ivec2(i & 1, i >> 1);                  
       76 +        ivec2 at = clamp(base + corner, ivec2(0), last);      
       77          vec4 v = texelFetch(uMarch, at, 0);
       53 -        lo = min(lo, v);                                      
       54 -        hi = max(hi, v);                                      
       78 +        float km = texelFetch(uMarchDistance, at, 0).r;       
       79 +        vec2 bw = mix(vec2(1.0) - w, w, vec2(corner));        
       80 +        float weight = bw.x * bw.y;                           
       81 +        reconstructed += weight * v;                          
       82 +        if (km > 0.0) {                                       
       83 +            reconstructedKm += weight * v.a * km;             
       84 +            kmWeight += weight * v.a;                         
       85 +        }                                                     
       86 +        nearest = min(nearest, length(vec2(base + corner) * bl
          +ock + vec2(uOffset)                                           
       87 +                                      - vec2(pixel)));        
       88      }
       89 +    reconstructedKm = kmWeight > 0.0 ? reconstructedKm / kmWei
          +ght : 0.0;                                                    
       90 +    ivec2 centre = clamp(ivec2(floor(f + 0.5)), ivec2(0), last
          +);                                                            
       91 +    vec4 mean = vec4(0.0);                                    
       92 +    vec4 square = vec4(0.0);                                  
       93 +    for (int y = -1; y <= 1; ++y) {                           
       94 +        for (int x = -1; x <= 1; ++x) {                       
       95 +            vec4 v = texelFetch(uMarch, clamp(centre + ivec2(x
          +, y), ivec2(0), last), 0);                                    
       96 +            mean += v;                                        
       97 +            square += v * v;                                  
       98 +        }                                                     
       99 +    }                                                         
      100 +    mean /= 9.0;                                              
      101 +    vec4 sigma = sqrt(max(square / 9.0 - mean * mean, vec4(0.0
          +)));                                                          
      102 +    vec4 lo = mean - CLAMP_SIGMA * sigma;                     
      103 +    vec4 hi = mean + CLAMP_SIGMA * sigma;                     
      104  
      105 +    // How much this pixel takes from this frame: its nearest 
          +sample's                                                      
      106 +    // Gaussian weight (1 on a traced pixel at block 1 and 2..
          +. none far off).                                              
      107 +    float reach = REACH * block;                              
      108 +    float near = block <= 1.0 ? 1.0 : exp(-0.5 * nearest * nea
          +rest / (reach * reach));                                      
      109 +                                                              
      110      vec2 ndc = (vec2(pixel) + 0.5) / uViewSize * 2.0 - 1.0;
       58 -    vec3 point = rayDir(ndc) * (distance > 0.0 ? distance * 10
          -00.0 : NO_CLOUD);                                             
       59 -    vec4 previous = uPrevRelToClip * vec4(point, 1.0);        
       60 -    vec2 uv = previous.xy / previous.w * 0.5 + 0.5;           
       61 -    bool valid = uHistoryValid != 0 && previous.w > 0.0       
       62 -              && all(greaterThanEqual(uv, vec2(0.0))) && all(l
          -essThanEqual(uv, vec2(1.0)));                                 
      111 +    vec3 dir = rayDir(ndc);                                   
      112 +    // Reprojected by this frame's distance first, then by the
          + history's own                                                
      113 +    // there (the pixel's cloud as it was): an edge keeps its 
          +place.                                                        
      114 +    vec2 at;                                                  
      115 +    bool valid = uHistoryValid != 0 && reproject(dir, reconstr
          +uctedKm, at);                                                 
      116 +    vec2 historyKm = vec2(0.0);                               
      117 +    if (valid) {                                              
      118 +        historyKm = texelFetch(uHistoryDistance, ivec2(at), 0)
          +.rg;                                                          
      119 +        if (historyKm.r > 0.0) {                              
      120 +            valid = reproject(dir, historyKm.r, at);          
      121 +            historyKm = valid ? texelFetch(uHistoryDistance, i
          +vec2(at), 0).rg : historyKm;                                  
      122 +        }                                                     
      123 +    }                                                         
      124  
      125      vec4 colour;
       65 -    float outDistance;                                        
      126 +    float km;                                                 
      127 +    float count;                                              
      128      if (!valid) {
       67 -        // Nothing to go on: this frame's nearest sample.     
       68 -        colour = current;                                     
       69 -        outDistance = distance;                               
      129 +        // Nothing to go on: this frame's samples, a sample's 
          +worth.                                                        
      130 +        colour = reconstructed;                               
      131 +        km = reconstructedKm;                                 
      132 +        count = near;                                         
      133      } else {
       71 -        vec2 at = uv * uViewSize / vec2(textureSize(uHistory, 
          -0));                                                          
       72 -        vec4 history = clamp(texture(uHistory, at), lo, hi);  
       73 -        float historyDistance = texture(uHistoryDistance, at).
          -r;                                                            
       74 -        colour = fresh ? mix(history, current, uFreshWeight) :
          - history;                                                     
       75 -        outDistance = fresh || historyDistance <= 0.0 ? distan
          -ce : historyDistance;                                         
      134 +        vec2 uv = at / vec2(textureSize(uHistory, 0));        
      135 +        vec4 history = texture(uHistory, uv);                 
      136 +        vec4 held = clamp(history, lo, hi);                   
      137 +        // What the clamp took away counts as samples lost: a 
          +change starts                                                 
      138 +        // the count over, where the scene moved or appeared. 
      139 +        float moved = length(held - history) / max(length(hist
          +ory), 0.05);                                                  
      140 +        count = historyKm.g * clamp(1.0 - 2.0 * moved, 0.0, 1.
          +0);                                                           
      141 +        float weight = near * max(1.0 / (count + 1.0), uMinWei
          +ght);                                                         
      142 +        colour = mix(held, reconstructed, weight);            
      143 +        km = historyKm.r > 0.0 && reconstructedKm > 0.0       
      144 +           ? mix(historyKm.r, reconstructedKm, weight)        
      145 +           : (weight >= 0.5 || historyKm.r <= 0.0 ? reconstruc
          +tedKm : historyKm.r);                                         
      146 +        count = min(count + near, 1.0 / uMinWeight);          
      147      }
      148      if (colour.a <= 0.0) {
       78 -        outDistance = 0.0;                                    
      149 +        km = 0.0;                                             
      150      }
      151      fragColor = colour;
       81 -    fragDistance = vec4(outDistance, 0.0, 0.0, 1.0);          
      152 +    fragDistance = vec4(km, count, 0.0, 1.0);                 
      153  }

● Fixing a comment, then the march.py side: uMinWeight replaces the fixed fresh
  weight, and each layer gets its own visit index for the jitter.

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/shaders/cloud_resolve.frag'…)
  ⎿  513:        self.card.set_shader_input("uFrame", self.visit)
     787:            "uFrame": self._frame,
                 self._finish()
     … +25 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/shaders/cloud_resolve.frag (+1 -1)
      103      vec4 hi = mean + CLAMP_SIGMA * sigma;
      104  
      105      // How much this pixel takes from this frame: its nearest 
           sample's
      106 -    // Gaussian weight (1 on a traced pixel at block 1 and 2..
          -. none far off).                                              
      106 +    // Gaussian weight (1 on a traced pixel, and at block 1 ev
          +erywhere).                                                    
      107      float reach = REACH * block;
      108      float near = block <= 1.0 ? 1.0 : exp(-0.5 * nearest * nea
           rest / (reach * reach));
      109  
  ⎿  Updated src/world_builder/clouds/march.py (+17 -4)
      312          )
      313      ),
      314  }
      315 -#: A traced pixel's share over its history, by block size: at 
          -full                                                          
      316 -#: resolution every pixel is traced every frame, so it average
          -s longer.                                                     
      317 -_FRESH_WEIGHT = {1: 0.2, 2: 1.0 / 3.0, 4: 0.5}                
      315 +#: This frame's least share in a pixel that has gathered its f
          +ull count of                                                  
      316 +#: samples (cloud_resolve.frag): it averages the steps' jitter
          + over about                                                   
      317 +#: this many of its visits' inverse.                          
      318 +MIN_WEIGHT = 0.1                                              
      319  #: The camera may move this far between frames and keep its hi
           story, metres
      320  #: (more is a jump: a new pose, a teleport).
      321  HISTORY_JUMP = 2000.0
     ...
      327      return max(1, round(1.0 / resolution))
      328  
      329  
      330 +def visit_of(block: int, frame: int, gather: bool = True) -> i
          +nt:                                                           
      331 +    """:returns: how many times the pixel traced on *frame* ha
          +s been traced                                                 
      332 +    before (once a cycle of *block* x *block* frames when gath
          +ering, every                                                  
      333 +    frame otherwise): what moves its steps' jitter on, so that
          + each pixel's                                                 
      334 +    visits, not the frames, run through the jitter's sequence"
          +""                                                            
      335 +    return frame // (block * block) if gather else frame      
      336 +                                                              
      337 +                                                              
      338  def bayer_offset(block: int, frame: int) -> tuple[int, int]:
      339      """:returns: the pixel of each *block* x *block* block the
            march traces
      340      on *frame*"""
     ...
      508              self.buffer, name, _shader("cloud_march.vert", "cl
           oud_march.frag")
      509          )
      510          self.card.set_shader_input("uSceneDepth", self.depth)
      511 +        # The jitter's index: this layer's visits (visit_of). 
      512 +        self.visit = PTA_int.empty_array(1)                   
      513 +        self.card.set_shader_input("uFrame", self.visit)      
      514  
      515          # The result, gathered over the frames.
      516          self.resolves = [
     ...
      525                  uMarchDistance=self.distance,
      526                  uHistory=history.colour,
      527                  uHistoryDistance=history.distance,
      516 -                uFreshWeight=_FRESH_WEIGHT[block if gather els
          -e 1],                                                         
     … 11 more lines (ctrl+o to expand)

● The layer's card now sets its own uFrame. I need to check that
  CloudMarch._bind doesn't overwrite it afterwards on the same node:

  Ran 1 shell command (ctrl+o to expand)

● It does: _bind sets the shared uFrame on the card after the layer's own. The
  march card is the only reader of uFrame, so I'll take it out of the shared
  inputs:

  Ran 1 shell command (ctrl+o to expand)
● Bash(python - <<'EOF'
      p='src/world_builder/clouds/march.py'…)
  ⎿  pose               frame       p95  no clouds    clouds  billboards  s
     pread    p99   grid
     below             4.81ms    5.58ms     1.04ms    3.77ms         0k
     … +15 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/march.py (+0 -1)
      784              "uStepGrowth": STEP_GROWTH,
      785              "uLayerFar": self._layer_far,
      786              "uLayerCount": self._layer_count,
      787 -            "uFrame": self._frame,                            
      787              "uDetailScale": DETAIL_SCALE,
      788              "uDetailErosion": DETAIL_EROSION,
      789              "uBillowMean": float(np.mean(1.0 - np.abs(2.0 * vo
           lume - 1.0))),
  ⎿  Updated src/world_builder/clouds/shaders/cloud_march.frag (+4 -2)
      35  // the last one beyond.
      36  uniform float uLayerFar[MAX_MARCH_LAYERS];
      37  uniform int   uLayerCount;
      38 -uniform int   uFrame;            // moves the steps' jitter (0:
         - fixed)                                                        
      38 +uniform int   uFrame;            // this pixel's visits: moves 
         +its steps' jitter (march.visit_of)                             
      39  // The edges' billows (what the billboards' sprites carry): a f
          ine billowy
      40  // noise, uDetailScale times the field's first octave, moving t
          he threshold
      41  // by up to uDetailErosion of its window, about its mean uBillo
          wMean.
     ...
      169          growth *= k;
      170      }
      171      // Where in its step each sample lies: no banding, a fine 
           noise instead,
      172 -    // moved on by the golden ratio each frame so the frames a
          -verage it away.                                               
      172 +    // moved on by the golden ratio each visit of this pixel (
          +not each frame:                                               
      173 +    // traced once a block's cycle, it would skip through the 
          +sequence), so                                                 
      174 +    // its visits stratify the step and the resolve averages t
          +hem away.                                                     
      175      float jitter = fract(interleavedGradient(vec2(pixel)) + fl
           oat(uFrame) * 0.618034);
      176  
      177      float feature = NOISE_FEATURE_UNITS / (f.noiseScale.x * f.
           octaveScales.x);

● The flicker is roughly halved, but the grid score got worse at three shots and
  resolve costs up to 1.4 ms more. I'm looking at a crop of the high shot to
  see what the grid is before tuning.

  Read 2 files, ran 1 shell command (ctrl+o to expand)

● The new resolve is clearly smoother than the old one, though a few bright
  specks remain on distant thin cloud. The grid score was misleading because
  it's relative, so I'll switch to an absolute measure in 8-bit brightness steps
  and compare both resolves again.

● Bash(python - <<'EOF'
      p='scripts/cloud_bench.py'…)
  ⎿  ok
  ⎿  Updated scripts/cloud_bench.py (+4 -4)
      82  * **grid**: how much one pixel of each march block (block x blo
          ck pixels,
      83    CloudSettings.march_resolution) stands out from the others: e
          ach frame's
      84    brightness less its 3 x 3 mean, its RMS by the pixel's place 
          in its block;
      85 -  the largest over the smallest, less 1, averaged over the fram
         -es (0: no                                                      
      86 -  pattern; the billboards measure it at the march's block too).
      85 +  the largest less the smallest, 8-bit steps, averaged over the
         + frames (0:                                                    
      86 +  no pattern; the billboards measure it at the march's block to
         +o).                                                            
      87  
      88  Interactive (``--interactive``): the probe flies, the camera fo
          llows it.
      89  
     ...
      221  def grid_score(frame: np.ndarray, mask: np.ndarray, block: int
           ) -> float:
      222      """How much one place in each *block* x *block* block stan
           ds out: the
      223      brightness less its 3 x 3 mean, its RMS per place in the b
           lock over the
      224 -    masked pixels, the largest over the smallest, less 1 (0: n
          -one)."""                                                      
      224 +    masked pixels, the largest less the smallest, 8-bit steps 
          +(0: none)."""                                                 
      225      if block <= 1:
      226          return 0.0
      227      lum = luminance(frame)
     ...
      244          if chosen.sum() < 64:
      245              return 0.0
      246          rms.append(np.sqrt(np.mean(residual[chosen] ** 2)))
      247 -    return float(max(rms) / max(min(rms), 1e-9) - 1.0)        
      247 +    return float(max(rms) - min(rms))                         
      248  
      249  
      250  def local_orientation(position: np.ndarray, heading: float, pi
           tch: float):

● For the old resolve I'll use the scratch worktree at HEAD, with only the new
  bench script copied in:

  Ran 2 shell commands (ctrl+o to expand)

● The scratchpad was cleared since yesterday, and the branch has moved to
  01c6376. Let me check what changed:

  Ran 1 shell command (ctrl+o to expand)

● You committed Phase 0 as 01c6376, so HEAD has the old resolve plus the bench.
  I'll prune the dead worktrees and make a fresh one at HEAD for the A/B:

  Ran 2 shell commands (ctrl+o to expand)

● The A/B is running: old resolve, then new, both still and moving at 10 m a
  frame.

● Background command "S=/tmp/claude-1000/-home-gl-development-WorldBuilder/d4124
c88-f133-488e-aa3c-ec7b6e8c7853/scratchpad; $S/p1/ab.sh old $S/head_wt >
$S/p1/old.txt 2>&1; $S/p1/ab.sh new > $S/p1/new.txt 2>&1; cat $S/p1/old.txt
$S/p1/new.txt" completed (exit code 0)

  Ran 1 shell command (ctrl+o to expand)

● The A/B shows the new resolve roughly halves the moving-scene grid error and
  cuts still-scene spread about in half at similar cost. The remaining worst
  case, the 20 km pose, I suspect is aliasing from the finest billow detail
  rather than the resolve, so I'm testing with the billows turned off next.

  Ran 1 shell command (ctrl+o to expand)

● Billows off cuts the grid error at high and above, but the 3rd and 4th octaves
  still alias since they're sub-pixel at tens of kilometres, and temporal
  accumulation can't fix that. I'm checking what uPixelAngle measures in the
  march so I can fade those octaves to their mean.

  Ran 1 shell command (ctrl+o to expand)

● Coarse octave about 2 km, then 1 km, 286 m, 125 m; the billows about 83 m. The
  view pixel's angle is uPixelAngle / uBlock. Adding the fade to marchDensity:

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/shaders/cloud_march.frag'…)
  ⎿  high              4.60ms    6.90ms     1.17ms    3.43ms         0k
     2.89   32.5  0.385
     above             6.57ms   10.14ms     1.17ms    5.40ms         0k
     … +9 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/shaders/cloud_march.frag (+26 -4)
      49  layout(location = 0) out vec4 fragColor;
      50  layout(location = 1) out vec4 fragDistance;
      51  
      52 +// What a pixel of the view resolves: a feature spanning fewer 
         +view pixels                                                    
      53 +// than this fades to its mean (it would alias, a different val
         +ue every                                                       
      54 +// frame's jitter and every move), over to twice as many.      
      55 +const vec2 RESOLVED = vec2(2.0, 4.0);                          
      56 +                                                               
      57 +// How much of a feature of this many metres a view pixel of fo
         +otprint                                                        
      58 +// metres resolves: 1 wholly, 0 not at all.                    
      59 +float resolved(float feature, float footprint) {               
      60 +    return smoothstep(RESOLVED.x, RESOLVED.y, feature / max(foo
         +tprint, 1e-3));                                                
      61 +}                                                              
      62 +                                                               
      63  // The density, its edges carved by billows (march.DETAIL_SCALE
          ): the
      64  // threshold moves by the billowy noise about its mean, in unit
          s of its
      65  // window, so a cloud's edge breaks into rounded lumps, its cor
          e stays, and
      66  // the coverage holds on average. Clear air is found first with
           the most the
      56 -// billows could add: cloud alone costs the noise twice over.  
      57 -float marchDensity(Field f, vec3 p, float shift) {             
      67 +// billows could add: cloud alone costs the noise twice over. T
         +he two fine                                                    
      68 +// octaves and the billows fade to their means where a view pix
         +el (footprint                                                  
      69 +// metres) cannot resolve them: the coverage holds (as an octav
         +e left out                                                     
      70 +// does), the sparkle goes.                                    
      71 +float marchDensity(Field f, vec3 p, float shift, float footprin
         +t) {                                                           
      72 +    float unit = NOISE_FEATURE_UNITS / f.noiseScale.x;         
      73 +    vec2 kept = vec2(resolved(unit / f.octaveScales.z, footprin
         +t),                                                            
      74 +                     resolved(unit / f.octaveScales.w, footprin
         +t));                                                           
      75 +    shift += dot((1.0 - kept) * f.octaveWeights.zw, vec2(f.octa
         +veMean));                                                      
      76 +    f.octaveWeights.zw *= kept;                                
      77 +    float billows = resolved(unit / (f.octaveScales.x * uDetail
         +Scale), footprint);                                            
      78      float window = (uCloudDebug & 32) != 0 ? 0.0  // no billows
           (profiling)
      59 -                                           : (f.threshHi - f.th
         -reshLo) * uDetailErosion;                                      
      79 +                 : (f.threshHi - f.threshLo) * uDetailErosion *
         + billows;                                                      
      80      float most = cloudDensityShifted(f, p, shift + window * uBi
          llowMean);
      81      if (most <= 0.0 || window <= 0.0) {
      82          return most;
     ...
      232              float footprint = t * uPixelAngle / feature;
      233              float detail = 1.0 - smoothstep(0.15, 0.6, footpri
           nt);
      234              float shading = 1.0 - smoothstep(0.04, 0.2, footpr
           int);
     … 7 more lines (ctrl+o to expand)

● Spread on high dropped only slightly, since each ray's jitter is landing on or
  missing thin cloud across 5–9 coarse steps. I'll try averaging more visits
  (MIN_WEIGHT) and then shorter steps far out (STEP_GROWTH).

  Ran 1 shell command (ctrl+o to expand)

● Raising the minimum samples per slab segment from 2 to 8 targets only the
  cheap long-step rays, while MIN_WEIGHT 0.05 gets kept since it helps
  everywhere at no cost. Halving STEP_GROWTH across the board would cost 0.4–0.9
  ms and make start worse.

  Searched for 1 pattern (ctrl+o to expand)

● Trying 8 samples per segment, with the shader and its NumPy mirror changed
  together:

  Ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/march.py'…)
  ⎿  Updated src/world_builder/clouds/march.py (+7 -2)
      145  MIN_STEP_FEATURE = 0.4
      146  #: Farther, a step's length per metre of distance from the cam
           era.
      147  STEP_GROWTH = 0.01
      148 +#: The fewest samples a stretch of the slab takes (cloud_march
          +.frag's                                                       
      149 +#: SEGMENT_SAMPLES): a far ray crossing it steeply, in a few s
          +teps longer                                                   
      150 +#: than its clouds' edges are thick, would hit or miss thin cl
          +oud by the                                                    
      151 +#: jitter's chance, frame after frame (sparkle seen from 20 km
          + up).                                                         
      152 +SEGMENT_SAMPLES = 8                                           
      153  #: The view ray's extinction over the field's (the sun march k
           eeps the
      154  #: field's own). The billboards' look: each of their quads car
           ries its
      155  #: sample's density over a whole chord, which keeps a cloud's 
           edge crisp;
     ...
      269      for t0, t1 in segments:
      270          if t1 <= t0:
      271              continue
      267 -        least = 0.5 * (t1 - t0)                               
      272 +        least = (t1 - t0) / SEGMENT_SAMPLES                   
      273          a = t0
      274          while a < t1 and len(ts) < budget + 8:
      275              dt = min(max(min_step, a * growth), least, t1 - a)
     ...
      320  #: This frame's least share in a pixel that has gathered its f
           ull count of
      321  #: samples (cloud_resolve.frag): it averages the steps' jitter
            over about
      322  #: this many of its visits' inverse.
      318 -MIN_WEIGHT = 0.1                                              
      323 +MIN_WEIGHT = 0.05                                             
      324  #: The camera may move this far between frames and keep its hi
           story, metres
      325  #: (more is a jump: a new pose, a teleport).
      326  HISTORY_JUMP = 2000.0
  ⎿  Updated src/world_builder/clouds/shaders/cloud_march.frag (+3 -1)
      86      return cloudDensityShifted(f, p, shift + window * (uBillowM
          ean - billow));
      87  }
      88  
      89 +// The fewest samples a stretch of the slab takes (march.SEGMEN
         +T_SAMPLES).                                                    
      90 +const float SEGMENT_SAMPLES = 8.0;                             
      91  // The march stops at this transmittance, and counts the cloud 
          opaque.
      92  const float OPAQUE = 0.01;
      93  // A cloudy column's alpha seen from afar: the deck's.
     ...
      212      for (int s = 0; s < 2; ++s) {
      213          vec2 seg = s == 0 ? seg0 : seg1;
      214          float a = seg.x;
      213 -        float least = 0.5 * (seg.y - seg.x);  // two samples a
          - segment at least                                             
      215 +        float least = (seg.y - seg.x) / SEGMENT_SAMPLES;  // t
          +he fewest a segment takes                                     
      216          while (a < seg.y && steps < uMarchSteps + 8 && transmi
           ttance > OPAQUE) {
      217              float dt = min(min(max(dtMin, a * growth), least),
            seg.y - a);
      218              float t = a + dt * jitter;

  Ran 1 shell command (ctrl+o to expand)

● Segment minimum is now 8 samples with MIN_WEIGHT 0.05, which cut the spread
  across all five poses, though below still sits at 0.32 in motion because of
  the sea's glints. The frame cost rose by up to 0.7 ms, so I'm taking the clamp
  statistics from the 4 bilinear samples instead of a separate 3×3 to save 9
  fetches per pixel.

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/shaders/cloud_resolve.frag'…)
  ⎿  move0 offscreen 1600x900, NVIDIA Corporation NVIDIA GeForce GTX 1070/P
     CIe/SSE2, clouds high march
     move0 below             4.57ms    5.49ms     1.06ms    3.51ms
     … +22 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/shaders/cloud_resolve.frag (+8 -15)
      34  // How far to look along a ray with no cloud on it (the sky at 
          infinity).
      35  const float NO_CLOUD = 1.0e7;
      36  // The history is held within this many standard deviations of 
          this frame's
      37 -// samples around the pixel.                                   
      38 -const float CLAMP_SIGMA = 1.25;                                
      37 +// four samples around the pixel (four: few, so a wide band).  
      38 +const float CLAMP_SIGMA = 1.5;                                 
      39  // A sample's reach, in its spacing (uBlock pixels): the Gaussi
          an weight a
      40  // pixel gives the nearest one.
      41  const float REACH = 0.3;
     ...
      63      float block = float(uBlock);
      64  
      65      // The march's samples around: sample k sits at view pixel 
          k * block +
      66 -    // uOffset. The four nearest, bilinearly; the 3 x 3 around,
         - for the clamp.                                                
      66 +    // uOffset. The four nearest, bilinearly, and their spread 
         +for the clamp.                                                 
      67      vec2 f = (vec2(pixel) - vec2(uOffset)) / block;
      68      ivec2 base = ivec2(floor(f));
      69      vec2 w = f - vec2(base);
     ...
      71      float reconstructedKm = 0.0;
      72      float kmWeight = 0.0;
      73      float nearest = 1e9;
      74 +    vec4 mean = vec4(0.0);                                     
      75 +    vec4 square = vec4(0.0);                                   
      76      for (int i = 0; i < 4; ++i) {
      77          ivec2 corner = ivec2(i & 1, i >> 1);
      78          ivec2 at = clamp(base + corner, ivec2(0), last);
     ...
       85              reconstructedKm += weight * v.a * km;
       86              kmWeight += weight * v.a;
       87          }
       88 +        mean += 0.25 * v;                                     
       89 +        square += 0.25 * v * v;                               
       90          nearest = min(nearest, length(vec2(base + corner) * bl
           ock + vec2(uOffset)
       91                                        - vec2(pixel)));
       92      }
       93      reconstructedKm = kmWeight > 0.0 ? reconstructedKm / kmWei
           ght : 0.0;
       90 -    ivec2 centre = clamp(ivec2(floor(f + 0.5)), ivec2(0), last
          -);                                                            
       91 -    vec4 mean = vec4(0.0);                                    
       92 -    vec4 square = vec4(0.0);                                  
       93 -    for (int y = -1; y <= 1; ++y) {                           
       94 -        for (int x = -1; x <= 1; ++x) {                       
     … 11 more lines (ctrl+o to expand)

● Fetches weren't the cost: the resolve change made things worse and no faster,
  so I'm reverting to the 3×3 clamp. Next I'll time the per-sample octave fade
  on and off to check whether it explains the extra 0.3–0.7 ms.

● Bash(python - <<'EOF'
      p='src/world_builder/clouds/shaders/cloud_resolve.frag'…)
  ⎿  with_fade inside            8.49ms    9.55ms     1.15ms    7.34ms
         0k
     with_fade model_in_cloud    8.58ms    9.41ms     1.15ms    7.43ms
     … +19 lines (ctrl+o to expand)
  ⎿  Shell cwd was reset to /home/gl/development/WorldBuilder
  ⎿  Updated src/world_builder/clouds/shaders/cloud_resolve.frag (+15 -8)
      34  // How far to look along a ray with no cloud on it (the sky at 
          infinity).
      35  const float NO_CLOUD = 1.0e7;
      36  // The history is held within this many standard deviations of 
          this frame's
      37 -// four samples around the pixel (four: few, so a wide band).  
      38 -const float CLAMP_SIGMA = 1.5;                                 
      37 +// samples around the pixel.                                   
      38 +const float CLAMP_SIGMA = 1.25;                                
      39  // A sample's reach, in its spacing (uBlock pixels): the Gaussi
          an weight a
      40  // pixel gives the nearest one.
      41  const float REACH = 0.3;
     ...
      63      float block = float(uBlock);
      64  
      65      // The march's samples around: sample k sits at view pixel 
          k * block +
      66 -    // uOffset. The four nearest, bilinearly, and their spread 
         -for the clamp.                                                 
      66 +    // uOffset. The four nearest, bilinearly; the 3 x 3 around,
         + for the clamp.                                                
      67      vec2 f = (vec2(pixel) - vec2(uOffset)) / block;
      68      ivec2 base = ivec2(floor(f));
      69      vec2 w = f - vec2(base);
     ...
      71      float reconstructedKm = 0.0;
      72      float kmWeight = 0.0;
      73      float nearest = 1e9;
      74 -    vec4 mean = vec4(0.0);                                     
      75 -    vec4 square = vec4(0.0);                                   
      74      for (int i = 0; i < 4; ++i) {
      75          ivec2 corner = ivec2(i & 1, i >> 1);
      76          ivec2 at = clamp(base + corner, ivec2(0), last);
     ...
       83              reconstructedKm += weight * v.a * km;
       84              kmWeight += weight * v.a;
       85          }
       88 -        mean += 0.25 * v;                                     
       89 -        square += 0.25 * v * v;                               
       86          nearest = min(nearest, length(vec2(base + corner) * bl
           ock + vec2(uOffset)
       87                                        - vec2(pixel)));
       88      }
       89      reconstructedKm = kmWeight > 0.0 ? reconstructedKm / kmWei
           ght : 0.0;
       94 -    vec4 sigma = sqrt(max(square - mean * mean, vec4(0.0)));  
       90 +    ivec2 centre = clamp(ivec2(floor(f + 0.5)), ivec2(0), last
          +);                                                            
       91 +    vec4 mean = vec4(0.0);                                    
       92 +    vec4 square = vec4(0.0);                                  
       93 +    for (int y = -1; y <= 1; ++y) {                           
     … 11 more lines (ctrl+o to expand)

● Session drift accounts for the cost gap: re-measured in the same session, the
  old and new resolves cost about the same, so the new pipeline is roughly three
  times steadier at equal cost. Next I'll check whether HIGH at
  march_resolution 0.25 (1/16 of pixels) can match the old ¼ steadiness.

  Ran 1 shell command (ctrl+o to expand)
● Tracing 1/16 of the pixels runs about as fast as the billboards at HIGH, and
  the still frames are steadier than before, except near-range poses like inside
  and model_in_cloud, where the spread rises. The near range is upsampled
  bilinearly from the march resolution, which shows as a visible 4×4 block; I'm
  showing a crop of inside next.

  Read 2 files, ran 1 shell command (ctrl+o to expand)

● That crop was empty sky. The whole frame shows the near cloud (right) very
  soft at ¼ resolution, and something blocky at the left around (150, 430).
  Zooming there, at 1/16 against ¼:

  Read 1 file, ran 1 shell command (ctrl+o to expand)

● At 1/16 the near-range clouds break into 4×4 blocks, while the ¼ version stays
  smooth, so I'm trying frame-gathering for the near range next; it adds one
  full-resolution resolve pass but drops its march to 1/16.

● Bash(sed -i 's/                gather=bit == CLOUD_FAR_BIT,/
      gather=True,/' src/world_builder/clouds/march.py && grep -n 'gather=True'
      src/world_build…)
  ⎿  869:                gather=True,
     res0.25 move0 below             3.56ms    4.54ms     1.12ms    2.45ms
             0k    0.75    3.9  0.085
     … +27 lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/march.py (+1 -1)
      866                  block,
      867                  sort,
      868                  states,
      869 -                gather=bit == CLOUD_FAR_BIT,                  
      869 +                gather=True,                                  
      870              )
      871              self._bind(layer.card)
      872              # Nothing to march until the weather is in (_finis
           h).

● Gathering in the near range cuts the 1/16 grid error to about 0.09 and 0.06
  for about 0.3 ms, while ¼ is unchanged, so both ranges now gather. Next I'll
  update the shared constants and visit_of, then run the full test suite.

  Ran 1 shell command (ctrl+o to expand)

● Bash(python - <<'EOF'
      p='tests/test_clouds_march.py'…)
  ⎿  I001 [*] Import block is un-sorted or un-formatted
       --> tests/test_clouds_march.py:9:1
        |
     … +61 lines (ctrl+o to expand)
  ⎿  Updated tests/test_clouds_march.py (+22 -3)
      21      STEP_GROWTH,
      22      VIEW_EXTINCTION_SCALE,
      23      CloudMarch,
      24 +    SEGMENT_SAMPLES,                                           
      25      bayer_offset,
      26      block_of,
      27      march_steps,
     ...
      30      slab_segments,
      31      sphere_interval,
      32      step_scale,
      33 +    visit_of,                                                  
      34  )
      35  from world_builder.clouds.noise import (
      36      MAX_LAYERS,
     ...
      167      assert step_scale(segments, 50.0, 0.01, 10) > 1.0
      168  
      169  
      168 -def test_a_short_segment_still_takes_two_samples():           
      170 +def test_a_short_segment_still_takes_its_fewest_samples():    
      171      # From orbit, the slab is a sliver of a ray thousands of k
           m long
      172      origin, d = _ray(2 * R, -90.0)
      173      segments = slab_segments(origin, d, R, SLAB)
      172 -    t, _ = march_steps(segments, 50.0, 0.5, 64)               
      173 -    assert len(t) >= 2                                        
      174 +    t, dt = march_steps(segments, 50.0, 0.5, 64)              
      175 +    assert len(t) >= SEGMENT_SAMPLES                          
      176 +    assert dt.max() <= (segments[0, 1] - segments[0, 0]) / SEG
          +MENT_SAMPLES + 1e-6                                           
      177  
      178  
      179  # ── Compositing ─────────────────────────────────────────────
           ─────────────────
     ...
      244      assert int(re.search(r"MAX_MARCH_LAYERS = (\d+);", text).g
           roup(1)) == MAX_LAYERS
      245      units = float(re.search(r"NOISE_FEATURE_UNITS = ([\d.]+);"
           , text).group(1))
      246      assert units == NOISE_FEATURE
      247 +    fewest = float(re.search(r"SEGMENT_SAMPLES = ([\d.]+);", t
          +ext).group(1))                                                
      248 +    assert fewest == SEGMENT_SAMPLES                          
      249  
      250  
      251  # ── Settings and the renderer choice ────────────────────────
           ─────────────────
     ...
      334      assert bayer_offset(block, block * block) == cycle[0]
      335  
     … 18 more lines (ctrl+o to expand)

● Bash(ruff check --fix -q tests/test_clouds_march.py && ruff format -q
      scripts/cloud_bench.py src/world_builder/clouds/march.py
      tests/test_clouds_march.py && ruff che…)
  ⎿  All checks passed!
      scripts/cloud_bench.py                             |   8 +-
      src/world_builder/clouds/march.py                  |  31 ++++-
     … +5 lines (ctrl+o to expand)
  ⎿  Updated tests/test_clouds_march.py (+1 -1)
      18  from world_builder.clouds.cloud import PRESETS, CloudSettings, 
          CloudType
      19  from world_builder.clouds.march import (
      20      MIN_STEP_FEATURE,
      21 +    SEGMENT_SAMPLES,                                           
      22      STEP_GROWTH,
      23      VIEW_EXTINCTION_SCALE,
      24      CloudMarch,
      24 -    SEGMENT_SAMPLES,                                           
      25      bayer_offset,
      26      block_of,
      27      march_steps,

● Tests pass (819) and ruff is clean. A visual check before the docs: the start
  pose with the new pipeline at ¼ (HIGH's current resolution) and at 1/16.

  Read 2 files (ctrl+o to expand)

● Still frames at 1/16 are indistinguishable from ¼. Motion is what you'll need
  to judge. Now the docs: clouds.md "The raymarch", the _MarchLayer docstring,
  and a Phase 1 subsection in §23.

● Bash(python - <<'EOF'
      p='docs/source/clouds.md'…)
  ⎿  ok
  ⎿  Updated docs/source/clouds.md (+29 -15)
      324    near the camera, which leaves a clear bubble).
      325  - **In steps** of `MIN_STEP_FEATURE` of the finest feature nea
           r the camera
      326    (50 m for cumulus), `STEP_GROWTH` of the distance farther, a
           ll lengthened
      327 -  alike when a ray needs more than `CloudSettings.march_steps`
          -. Each sample                                                 
      328 -  is jittered within its step, the jitter moving on by the gol
          -den ratio                                                     
      329 -  each frame. The march stops once the cloud is opaque. The st
          -eps' geometry                                                 
      327 +  alike when a ray needs more than `CloudSettings.march_steps`
          +, and at                                                      
      328 +  least `SEGMENT_SAMPLES` (8) a stretch of the slab: a far ray
          + crossing it                                                  
      329 +  steeply in a few steps longer than its clouds' edges are thi
          +ck hits or                                                    
      330 +  misses thin cloud by chance (the sparkle seen from 20 km up)
          +. Each sample                                                 
      331 +  is jittered within its step, the jitter moving on by the gol
          +den ratio at                                                  
      332 +  each **visit** of the pixel (`visit_of`), not each frame: tr
          +aced once a                                                   
      333 +  block's cycle, a per-frame jitter skipped half a step a visi
          +t and never                                                   
      334 +  stratified it. The march stops once the cloud is opaque. The
          + steps' geometry                                              
      335    has a NumPy mirror that the tests run (`slab_segments`, `mar
           ch_steps`,
      336    `march_transmittance`).
      337  - **No longer steps in clear air.** A step longer than the fin
           est features
     ...
      344    threshold (the exact bound, from the texels each octave's fi
           lter
      345    touches, did in 0.004 % of the cells).
      346  - **The field** with the octaves the billboard shells would ha
           ve there
      342 -  (`shell_octaves` over their distance bands). The weather's s
          -hift is read                                                  
      347 +  (`shell_octaves` over their distance bands), and where a vie
          +w pixel spans                                                 
      348 +  more than half a fine octave's feature (the 3rd and 4th, 285
          + m and 125 m                                                  
      349 +  for cumulus) or the billows', that one fades to its mean (as
          + an octave                                                    
      350 +  left out does: the coverage holds): it would alias, a differ
          +ent value at                                                  
      351 +  every jitter and every move. The weather's shift is read    
      352    once per kilometre of ray (`cloudDensityShifted`), not at ev
           ery sample.
      353    Two departures, march-only, give back the billboards' look:
      354    - the view ray takes **twice the field's extinction**
     ...
      379  2. a fullscreen card **marches** one pixel of every 2 × 2 bloc
           k
      380     (`march_resolution` ½; 4 × 4 at ¼), a different one each fr
           ame in a
      381     Bayer order. It writes the premultiplied cloud and its dist
           ance;
      373 -3. a **resolve** gathers them into the view's resolution over 
          -the frames:                                                   
      374 -   the other pixels come from the last frame's result, reproje
          -cted along the                                                
      375 -   view ray to the cloud's distance (float64 for the camera's 
          -move, float32                                                 
      376 -   only near the camera) and held within the range of this fra
          -me's                                                          
      377 -   neighbouring samples. A traced pixel blends into its histor
          -y, which                                                      
      378 -   averages the jitter away. Two results alternate, each frame
          -'s reading                                                    
      379 -   the other's;                                               
      382 +3. a **resolve** gathers them into the view's resolution over 
          +the frames.                                                   
     … 27 more lines (ctrl+o to expand)
  ⎿  Updated src/world_builder/clouds/march.py (+5 -6)
      449      """One depth range's march: its depth pass, its march, its
            two
      450      alternating results and the card that lays them over the r
           ange's pass.
      451  
      452 -    A *gathering* layer traces a different pixel of each block
          - every frame                                                  
      453 -    and gathers them into the view's resolution; the other tra
          -ces the same                                                  
      454 -    one and only averages its jitter over the frames, at the m
          -arch's own                                                    
      455 -    resolution, upsampled by the composite (the near range: it
          -s clouds are                                                  
      456 -    close, large and soft on screen, and it costs a full-resol
          -ution pass                                                    
      457 -    less).                                                    
      452 +    A *gathering* layer (both ranges') traces a different pixe
          +l of each                                                     
      453 +    block every frame and gathers them into the view's resolut
          +ion; the                                                      
      454 +    other traces the same one and only averages its jitter ove
          +r the frames,                                                 
      455 +    at the march's own resolution, upsampled by the composite 
          +(a                                                            
      456 +    full-resolution pass less, but 4 x 4 squares at a quarter 
          +resolution).                                                  
      457      """
      458  
      459      def __init__(

● Now the Phase 1 results in §23:

● Bash(python - <<'EOF'
      p='docs/source/performance.md'…)
  ⎿  ok
  ⎿  Updated docs/source/performance.md (+44 -0)
      1184  the billboards, a third of it fixed. Its unsteadiness (still 
            frames that
      1185  flicker, a block grid in motion) is the first thing to fix.
      1186  
      1187 +### Steadier: the reconstruction (Phase 1)                   
      1188 +                                                             
      1189 +What made the march unsteady, and what changed (clouds.md, "T
           +he                                                           
      1190 +raymarch"): the jitter moved on every frame while a pixel was
           + traced once                                                 
      1191 +every 4 (it skipped through the sequence: its visits never st
           +ratified the                                                 
      1192 +step), so it moves on per visit; the resolve copied a block's
           + one sample                                                  
      1193 +into the pixels with no history, reprojected the others by th
           +e block's                                                    
      1194 +distance, and gave the traced pixel a fixed third over its hi
           +story, so                                                    
      1195 +1 pixel in 4 stood apart every frame: now every pixel blends 
           +the four                                                     
      1196 +nearest samples by the nearest's Gaussian weight, reprojects 
           +by its own                                                   
      1197 +distance, clamps its history to ±1.25 σ of the 3 × 3 samples,
           + and counts                                                  
      1198 +its samples (this frame's share 1 / (count + 1), down to 0.05
           +); far off,                                                  
      1199 +the fine octaves and the billows fade to their means where a 
           +pixel cannot                                                 
      1200 +resolve them, and a stretch of the slab takes 8 samples at le
           +ast. The near                                                
      1201 +range gathers too.                                           
      1202 +                                                             
      1203 +The **grid** below is absolute now (the largest class RMS les
           +s the                                                        
      1204 +smallest, in 8-bit steps: the ratio of the first table rose a
           +s the noise                                                  
      1205 +fell). Same session, HIGH, cloud cost ms; spread (mean / p99)
           + still; grid                                                 
      1206 +still and moving 10 m a frame:                               
      1207 +                                                             
      1208 +| Pose | Cost, old | Cost, new ½ | Cost, new ¼ | Spread old |
           + Spread new ½ | Spread new ¼ | Grid moving old / ½ / ¼ |     
      1209 +|---|---|---|---|---|---|---|---|                            
      1210 +| Below | 3.98 | 3.81 | 2.45 | 2.8 / 18 | 0.9 / 8 | 0.8 / 4 |
           + 0.32 / 0.33 / 0.51 (the sea's glints) |                     
      1211 +| Start | 5.35 | 5.49 | 3.08 | 2.0 / 14 | 0.7 / 7 | 0.6 / 3 |
           + 0.19 / 0.06 / 0.06 |                                        
      1212 +| Inside | 6.43 | 7.26 | 3.48 | 1.3 / 15 | 0.5 / 5 | 0.4 / 3 
           +| 0.09 / 0.05 / 0.06 |                                       
      1213 +| Above | 5.14 | 5.36 | 2.86 | 2.8 / 15 | 0.9 / 6 | 0.8 / 4 |
           + 0.35 / 0.09 / 0.07 |                                        
      1214 +| 20 km up | 2.50 | 2.84 | 1.55 | 8.6 / 57 | 1.6 / 20 | 1.4 /
           + 11 | 1.06 / 0.29 / 0.22 |                                   
      1215 +| Model in cloud | 6.93 | 7.62 | 3.78 | 0.7 / 8 | 0.2 / 1 | 0
           +.2 / 1 | 0.01 / 0.07 / 0.08 |                                
      1216 +                                                             
      1217 +(½ and ¼: `march_resolution`, a quarter and a sixteenth of th
           +e pixels                                                     
      1218 +traced a frame.) The cost moved within this machine's drift (
           +the old                                                      
      1219 +pipeline measured again in one session with the new: 7.19 / 7
           +.15 ms                                                       
      1220 +inside, 2.04 / 1.98 at orbit). Tried and dropped: the clamp's
           + statistics                                                  
      1221 +from the 4 bilinear samples instead of the 3 × 3 (twice the s
           +pread, no                                                    
      1222 +faster); steps half as long far off (`STEP_GROWTH` 0.005: the
           + 20 km sparkle                                               
      1223 +gone too, but 0.4–0.9 ms dearer and longer rays stretched to 
           +the budget).                                                 
     … 10 more lines (ctrl+o to expand)

● Recording Phase 1's outcome in the plan:

● Bash(python - <<'EOF'
      p='MARCH_PLAN.md'…)
  ⎿   MARCH_PLAN.md                                      |   2 +
      docs/source/clouds.md                              |  44 +++---
      docs/source/performance.md                         |  44 ++++++
     … +7 lines (ctrl+o to expand)
  ⎿  Updated MARCH_PLAN.md (+2 -0)
      46  
      47  ## Phase 1: Temporal pipeline (the jitter and the grid)
      48  
      49 +**Done** (performance §23, "Steadier"): flicker 2–5 times lower
         +, the grid in motion 2–4 times lower, at the same cost. Beyond 
         +the items below, two more causes turned up far off: the fine oc
         +taves and the billows alias (now faded to their means by the pi
         +xel's footprint), and steep rays crossed the slab in 2 long ste
         +ps (now at least 8, `SEGMENT_SAMPLES`). The near range gathers 
         +too. At `march_resolution` ¼ (1/16 of the pixels a frame) the m
         +arch is steadier than it was at ½ and costs 1.5–3.8 ms, about t
         +he billboards' HIGH cost. Whether HIGH moves to ¼ is decided af
         +ter you look at it in flight.                                  
      50 +                                                               
      51  In `cloud_march.frag`, `cloud_resolve.frag` and `march.py` (`_M
          archLayer`, `bayer_offset`, `_FRESH_WEIGHT`):
      52  
      53  1. **Jitter per visit, not per frame.** Index the sequence by `
          frame / block²`, with a per-pixel R2 or blue-noise offset, so e
          ach pixel's visits stratify its step.

● Phase 1 is done: the march's flicker is 2–5 times lower and the dot grid in
  motion 2–4 times lower, at the same cost. All 819 tests pass; the changes are
  uncommitted.

  Causes found and fixed:
  - Jitter that never averaged out. It advanced every frame, but each pixel is
    traced only once every 4 frames, so it jumped about half a step per visit.
    It now advances once per visit of the pixel, so each pixel's samples cover
    its step evenly.
  - The resolve singled out the traced pixel. Every pixel now blends the 4
    nearest samples, weighted by how close the nearest one is, and is
    reprojected by its own last distance. Its history is held within ±1.25
    standard deviations of the 3×3 samples around it. Each pixel counts the
    samples it has gathered, so a still view keeps averaging down to a 5 % share
    for new samples.
  - Aliasing far away. The two finest noise octaves and the billows fade to
    their average once a pixel can't resolve them.
  - Steep far rays crossed the slab in only 2 long steps; they now take at
    least 8.
  - The near range now accumulates over frames like the far one, instead of
    being upscaled. At low resolution the upscale showed as 4×4 squares.

  Results at HIGH, same session (spread = how much a still pixel's brightness
  wanders, out of 255):

  ┌──────────────────────────────┬──────────────┬───────────────────────────┐
  │                              │    Before    │           After           │
  ├──────────────────────────────┼──────────────┼───────────────────────────┤
  │ Still spread, mean / worst 1 │ 1.2–2.8 /    │ 0.2–0.9 / 1–8             │
  │  %                           │ 8–18         │                           │
  ├──────────────────────────────┼──────────────┼───────────────────────────┤
  │ Still spread at 20 km up     │ 8.6 / 57     │ 1.6 / 20                  │
  ├──────────────────────────────┼──────────────┼───────────────────────────┤
  │ Dot grid in motion (start,   │ 0.19, 0.35,  │ 0.06, 0.09, 0.29          │
  │ above, 20 km)                │ 1.06         │                           │
  ├──────────────────────────────┼──────────────┼───────────────────────────┤
  │ Cloud cost                   │ —            │ unchanged, within         │
  │                              │              │ run-to-run drift          │
  └──────────────────────────────┴──────────────┴───────────────────────────┘

  The below pose's grid score stays around 0.3–0.5; that's the sea's glints, not
  the clouds.

  The bigger win: tracing only 1/16 of the pixels per frame (march_resolution
  0.25) is now steadier than the old ¼ was. It costs 1.5–3.8 ms, about what the
  billboards cost at HIGH, against 2.5–7 ms before. Still frames look the same
  as at ¼. In motion, fine detail should be coarser until the history fills back
  in, which I can't judge from stills.

  Tried and dropped:
  - Taking the resolve's clamp statistics from 4 samples instead of 9: twice the
    flicker, no faster.
  - Shorter steps everywhere far away: 0.4–0.9 ms dearer, and long grazing rays
    got worse.

  Please fly it, at both resolutions, to compare motion; M switches renderers:
  python scripts/cloud_bench.py --renderer march --interactive
  python scripts/cloud_bench.py --renderer march --march-resolution 0.25
  --interactive
  python scripts/demo_terrain.py --cloud-renderer march

  If ¼ resolution holds up in flight, I'd make it the HIGH preset. I've left the
  presets untouched until you've looked. The write-up is in §23 of
  performance.md (docs/source/performance.md) ("Steadier"), and clouds.md
  (docs/source/clouds.md) and MARCH_PLAN.md are updated.