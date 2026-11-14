#version 330
// The model shader's "pbr_full" tier (ModelSettings.shading, ULTRA): PBR,
// normal maps, the sky reflected and blurred by the roughness, glTF's
// occlusion maps, and the light the ground bounces up.
#define MODEL_PBR
#define MODEL_NORMAL_MAP
#define MODEL_SKY_REFLECTION
#define MODEL_FULL
#pragma include "model_main.glsl"
