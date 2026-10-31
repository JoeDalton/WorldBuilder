#version 330

// The cirrus sheet's stamps (clouds.cirrus): baked patches of cirrus uncinus
// (scripts/bake_cirrus_atlas.py) lying on one sphere. A stamp's texel is a
// thickness; a sight line crosses thickness / |cos| of the slab through it.
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "cloud_field.glsl"

// 0: the behind node, keeping the crossings where sight lines leave the
// sheet's sphere (behind every cumulus); 1: the front node, where they enter
// it (in front of every cumulus). See cirrus.py.
uniform int   uFront;
uniform float uThickness;       // the slab's depth, metres
uniform float uExtinction;      // optical depth per metre at thickness 1
uniform float uMaxOpticalDepth;
uniform sampler2D cirrusAtlas;  // r: thickness, mipmapped
uniform vec4  uPhase;           // noise.phase_weights, and the forward g
// x: multiple scattering, y: shadow strength.
uniform vec4  uLight;

uniform vec3  sunColor;
uniform vec3  skyColor;
uniform float sunBrightness;
uniform float skyStrength;
uniform float exposure;

// The longest path through the slab, in slab depths (see main).
const float GRAZING_PATH = 4.0;

in vec3 vPosition;
flat in vec3 vEye;
in vec2 vUv;
in float vOpacity;
in vec3 vAtmoT;
in vec3 vAtmoS;
in vec3 vSunTint;
in float vSkyTint;
in vec3 vTwilight;
out vec4 fragColor;

void main() {
    // Which crossing of the sphere this is, seen from this pass's camera:
    // a sight line enters it exactly where the camera is above the tangent
    // plane (from inside the sphere it never is).
    vec3 up = normalize(vPosition);
    vec3 toEye = vEye - vPosition;
    bool entering = dot(up, toEye) > 0.0;
    if (entering != (uFront == 1)) discard;

    float thickness = texture(cirrusAtlas, vUv).r * vOpacity;
    if (thickness < 1.0 / 255.0) discard;
    float distance = length(toEye);
    // A veil from inside the slab: fade near the camera.
    float nearFade = smoothstep(0.5 * uThickness, 2.0 * uThickness, distance);
    if (nearFade <= 0.0) discard;
    vec3 dir = -toEye / distance;

    // The path through the slab, at most GRAZING_PATH times its depth: past
    // that every stamp saturates toward the horizon into a flat plate, its
    // fibres gone. Then a soft saturation toward the cap, not a clamp, which
    // flattened each tuft to one white with a hard rim.
    float cosView = max(abs(dot(dir, up)), 1.0 / GRAZING_PATH);
    float opticalDepth = uMaxOpticalDepth * (1.0 - exp(
        -uExtinction * thickness * uThickness / cosView / uMaxOpticalDepth));
    float alpha = (1.0 - exp2(-opticalDepth)) * nearFade;
    if (alpha < 1.0 / 512.0) discard;

    if ((uCloudDebug & 8) != 0) {
        fragColor = vec4(vec3(0.8) * alpha, alpha);
        return;
    }
    // Thin: no march, the sunward half of the slab at this thickness.
    float cosSun = dot(up, sunDir);
    float sunDepth = uExtinction * thickness * 0.5 * uThickness / max(cosSun, 0.1);
    float sunVisibility = exp2(-uMaxOpticalDepth
        * (1.0 - exp(-sunDepth * uLight.y / uMaxOpticalDepth)));
    float ambient = skyStrength + uLight.x * sunBrightness * (1.0 - sunVisibility);
    float phase = lobePhase(uPhase, dot(sunDir, dir));

    vec3 colour = sunColor * vSunTint * (sunBrightness * sunVisibility * phase)
                + skyColor * vSkyTint * ambient
                + sunColor * vTwilight * (sunBrightness * TWILIGHT_GLOW);
    // The air in front, weighted by the alpha like the cirrus itself.
    colour = colour * vAtmoT + vAtmoS;
    colour = displayColour(colour * exposure);
    fragColor = vec4(colour * alpha, alpha);   // premultiplied "over"
}
