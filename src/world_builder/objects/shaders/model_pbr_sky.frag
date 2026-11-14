#version 330
// The model shader's "pbr_sky" tier (ModelSettings.shading): PBR, normal
// maps, and the sky reflected from the sky-view table.
#define MODEL_PBR
#define MODEL_NORMAL_MAP
#define MODEL_SKY_REFLECTION
#pragma include "model_main.glsl"
