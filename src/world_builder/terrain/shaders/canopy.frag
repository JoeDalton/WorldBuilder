#version 330
// The canopy's march from where the view ray enters a chunk's shell (its
// front faces): the hit is never nearer than the shell, so the depth written
// is conservative (greater) and the ground's early depth test still culls
// the shell's hidden pixels.
#extension GL_ARB_conservative_depth : enable
layout(depth_greater) out float gl_FragDepth;

#pragma include "canopy_frag.glsl"
