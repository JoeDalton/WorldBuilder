#version 330
// The clouds' light volume's bake (clouds.light_volume): per voxel of the
// sky bands around the camera (cloud_field.glsl's lightVoxelAt chart), the
// genera's summed optical depth toward the sun (r) and straight up (g), from
// the voxel to where the ray leaves the highest band's top
// (skyShadowLevels: each band ahead in its own steps, every genus in it by
// its own field and weather).
//
// Unlike the ground's shadow march (skyShadowDepth), the weather (the fbm's
// shift, the convective depth, the anvil's spread) and the cell octave are
// taken at every step, not once where the chord starts, and the chord is not
// cut at that column's top: the depth is then a function of the point alone,
// so two voxels on one sun ray differ by the clouds between them, which the
// readers take (cloudLightSunDepth). With the weather of the chord's start,
// a voxel beside a tower marched through it with its own column's
// (shallower) depth, and the tower's face streaked.
// light_volume.light_depth_union on the CPU.
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "cloud_field.glsl"

uniform int  uLightSteps;
// The march's steps against cloudShadowSteps' (about two sun steps each).
uniform float uLightStepScale;
uniform vec4 uLightChart[4];

out vec4 fragColor;

float lightDepth(vec3 p, vec3 dir) {
    vec3 levels;
    return skyShadowLevels(p, dir, uLightSteps, uLightStepScale, false, false, levels);
}

void main() {
    int slice;
    vec3 p = lightVoxelAt(uLightChart, gl_FragCoord.xy, slice);
    fragColor = vec4(lightDepth(p, sunDir), lightDepth(p, normalize(p)), 0.0, 1.0);
}
