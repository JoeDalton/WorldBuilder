# Render scale, anti-aliasing and FSR 1

This page covers how the view reaches the screen: at what resolution it is
drawn, how its edges are anti-aliased, and how it is stretched to the window.
It lives in [`world_builder.post`](../../src/world_builder/post/) and is set by
the `render` section of `WorldSettings`. It came from SpaceFlight's
post-process pipeline (issue #68), with AMD's FSR 1 added.

## The settings

`RenderSettings` has no presets of its own, like the `display` section. Every
knob is a choice, so picking LOW to ULTRA leaves it as it is. The defaults
draw the view at the window's size with no anti-aliasing, as before.

| Id | Label | Applies | Range or choices | Default |
|---|---|---|---|---|
| `scale` | Render scale | rebuild | 0.5 to 1 | 1 |
| `msaa` | MSAA | rebuild | Off, 2x, 4x, 8x, 16x | Off |
| `fxaa` | FXAA | rebuild | on / off | off |
| `upscaler` | Upscaler | rebuild | Bilinear, FSR 1 | Bilinear |
| `sharpness` | FSR sharpness | live | 0 to 2 stops | 0.1 |

- **`scale`:** the view is drawn at `round(region × scale)` pixels per axis.
  `FSR_MODES` holds AMD's names for 0.77, 0.67, 0.59 and 0.5 (ultra quality,
  quality, balanced, performance).
- **`msaa`:** the scene buffer's samples. The driver may grant fewer;
  `ReversedDepthCamera.describe()` shows what it got. `msaa` Off keeps the
  window's own samples, which is none in the demo.
- **`sharpness`:** RCAS's strength in stops. 0 is the strongest, and each +1
  halves it.
- **FXAA under FSR 1:** without MSAA, FSR 1 forces FXAA (`fxaa_forced`).
  FSR 1 stretches its input's edges, so it wants an anti-aliased input. AMD's
  preferred input is TAA, which needs motion vectors, so there is none here.
  With MSAA, FXAA runs only if asked.

## The passes

The world's `ReversedDepthCamera` draws the view into its float-depth scene
buffer at the render size, with `msaa` samples. A `PostChain` then shows that
buffer in the window, in a region where the app camera's was, under the
game's HUD.

| Upscaler | Passes |
|---|---|
| Bilinear | view → window region: a copy, or FXAA. The GPU stretches it bilinearly. |
| FSR 1 | view → [FXAA, view size] → EASU (output size) → RCAS → window region |

- **EASU** (edge-adaptive spatial upsampling) builds each output pixel from
  12 view texels. Its Lanczos-like kernel is stretched along the local edge
  and clamped to the nearest 2×2 texels (deringing).
- **RCAS** (robust contrast-adaptive sharpening) sharpens with a 5-tap
  cross, limited so that it never overshoots its neighbours.
- **The ports:** both are ports of AMD's `ffx_fsr1.h` (MIT; its notice is in
  each shader) to GLSL 1.40.
  - AMD's `textureGather` becomes `texelFetch` taps, so the textures' padding
    to powers of two doesn't matter.
  - The reciprocals are guarded, so pure black or white neighbourhoods don't
    divide 0 by 0.

The view must hold display values. WorldBuilder's shaders tone map and gamma
encode in the shader, into 8-bit buffers. FSR 1 and FXAA both assume such
input.

FSR 1 is spatial: it needs the view's colour only. FSR 2 and later are
temporal: they need jittered projections, per-pixel motion vectors and
history, which Panda3D's shaders don't write.

## What follows the view's size

`depth.view_size(app)` is the view's size: the live reversed camera's render
size, else the app camera's region, else the window. These parts follow it,
so a lower `scale` costs less than its pixels alone:

- **the clouds' raymarch** and its passes, through the camera's region;
- **the depth copy** the clouds read;
- **the sea's reflection,** made again when the view's size changes;
- **the per-pixel detail:**
  - the trees' and the clouds' pixel angle;
  - the sea's pixel angle, which sets how far wave slopes are averaged per
    pixel;
  - the asteroid fields' pixels per radian, which set the smallest rock
    drawn.

  A pixel of the view spans more at a lower scale, so the detail drops with
  it.

## Changes

`ReversedDepthCamera.set_settings(render)` changes only what it must:

- `sharpness` applies live;
- `scale`, `fxaa` and `upscaler` build a new chain over the same scene
  buffer;
- `msaa` needs a new scene buffer. `set_settings` then returns True, and
  `World.set_settings` attaches the clouds' raymarch again.

The world hands the camera its `render` section in `load_world` and
`World.use_depth_camera`. A game changes it like any other section, through
`world.set_settings(settings)`.

A window resize is followed each frame in `update()`: the scene buffer and
the chain's passes take the new sizes.

## A chain that ends in a texture

`PostChain(..., output_size=(w, h))` ends in a texture of that size
(`chain.texture`) instead of a window region. This is for a view shown on a
card, such as a game's rear-view mirror:

- the mirror is drawn small, at its own buffer's size;
- it is upscaled with FSR 1 to the card's size on screen;
- the card shows the result texel for texel, using the texture's
  `get_tex_scale()` should the GSG pad it to powers of two.

`chain.resize(render_size, output_size)` follows a new size. The passes' sorts
start at `buffer_sort`, one per pass (up to three): after the view's buffer,
before whatever shows the texture.

## Measured

The measurements are from `frame_profile.py --offscreen`, three_planets.yaml
at HIGH, 2560×1440, RTX 3070 / Ryzen 5600X, at 2026-11-27's code. In GPU-synced
mode a frame is the CPU's step plus the GPU's work, so frame less step is
about the GPU's share. The table gives that share as ground / in_clouds, in
ms:

| Settings | GPU (frame − step) | Clouds' share (ground) |
|---|---|---|
| native | 5.15 / 4.94 | 3.50 |
| native, MSAA 4 | 5.38 / 5.24 | 3.26 |
| 0.67, bilinear | 2.98 / 2.61 | 2.10 |
| 0.77, FSR 1 | 3.89 / 3.36 | 2.56 |
| 0.67, FSR 1 | 2.96 / 2.91 | 1.81 |
| 0.67, FSR 1, MSAA 4 | 3.21 / 2.96 | 2.04 |
| 0.5, FSR 1 | 1.96 / 1.83 | 1.31 |

- **The chain's cost:** at 0.67 the FXAA, EASU and RCAS passes cost within
  the noise of a bilinear stretch, at most about 0.3 ms.
- **MSAA 4:** about +0.25 ms.
- **The game's frame:** pipelined (`--pipelined`), the CPU's frame and the
  GPU's last one overlap. That frame is about 3.3 ms in every setting, so
  this scene is CPU-bound and the GPU's savings are headroom.
- **In a window:** the compositor holds a frame for the display's refresh,
  even with vsync off. Native measured 7.0 ms there, and every scaled
  setting 5.6 ms, the 180 Hz display's refresh. Time offscreen.

## The demo

`demo_world.py` and every script taking the quality arguments accept
`--render-scale`, `--msaa`, `--fxaa`, `--fsr` and `--sharpness`. The demo's
keys:

| Key | Action |
|---|---|
| F9 | cycle the scale: native, then FSR 1's four modes |
| F10 | FSR 1 on or off |
| F11 | cycle MSAA: off, 2, 4, 8, 16 |
| F12 | FXAA on or off |
| U / I | sharper / softer, in 0.1 stop steps |
| N | native (scale 1, no AA, no FSR), and back |

The HUD's last line is the camera's `describe()`, for example
`1715x965 -> 2560x1440 | no MSAA | FXAA (forced) -> EASU -> RCAS 0.10`.
