#version 330
// The clouds' light volume's bake (clouds.light_volume): per voxel of the
// field's slab around the camera (cloud_field.glsl's lightVoxelAt chart),
// the clouds' optical depth toward the sun (r) and straight up (g), from the
// voxel to where the ray leaves the slab's top.
//
// Unlike the ground's shadow march (cloudShadowDepthIn), the weather (the
// fbm's shift, the convective depth, the anvil's spread) and the cell octave
// are taken at every step, not once where the chord starts, and the chord is
// not cut at that column's top: the depth is then a function of the point
// alone, so two voxels on one sun ray differ by the clouds between them,
// which the readers take (cloudLightSunDepth). With the weather of the
// chord's start, a voxel beside a tower marched through it with its own
// column's (shallower) depth, and the tower's face streaked.
// light_volume.light_depth_on_sphere on the CPU.
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "cloud_field.glsl"

uniform int  uLightSteps;
// The march's steps against cloudShadowSteps' (about two sun steps each).
uniform float uLightStepScale;
uniform vec4 uLightChart[4];

out vec4 fragColor;

float lightDepth(Field f, vec3 p, vec3 dir, int maxSteps) {
    vec3 start;
    float chord;
    if (!cloudShadowChord(f, p, dir, start, chord)) {
        return 0.0;
    }
    int steps = cloudShadowSteps(f, chord, maxSteps);
    float stepLength = chord / float(steps);
    vec3 increment = dir * stepLength;
    vec3 position = start + increment * 0.5;
    float density = 0.0;
    float rest = f.octaveMean * (f.octaveWeights.z + f.octaveWeights.w);
    for (int i = 0; i < steps; i++, position += increment) {
        vec3 weather = texture(weatherMap, position).gba;
        float cell = cellShift(f, position * f.noiseScale + f.noiseOrigin, weather.y);
        density += slabSampleDensity(f, position, weather, f.shadowBias + cell,
                                     rest + cell);
    }
    return density * stepLength * f.fieldExtinction * f.shadowStrength;
}

void main() {
    int slice;
    vec3 p = lightVoxelAt(uLightChart, gl_FragCoord.xy, slice);
    Field f = getField(uCloudShadowLayer);
    f.sunStepLength *= uLightStepScale;
    fragColor = vec4(lightDepth(f, p, sunDir, uLightSteps),
                     lightDepth(f, p, normalize(p), uLightSteps), 0.0, 1.0);
}
