#version 330

// A rock prototype's level of detail, drawn once per rock (hardware
// instancing: world_builder.asteroids.field). Instance k of the draw is the
// slot uIndex[uOffset + k]; the slot's four texels in uInstances
// (population.Population.table_rows) place, size and turn it:
//   0: centre relative to its cell's corner (field frame), radius
//   1: orientation at the epoch (w, x, y, z)
//   2: precession (axis times rate, field frame), spin about body z
//   3: the cell's indices, then class << 18 | prototype << 6 | ice in 64ths
// The camera is given per size class the same way (its cell, and its offset
// in it), so the rock's position relative to it is a difference of small
// numbers: float32 holds at any distance from the field's centre.

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform sampler2D uInstances;  // RGBA32F, TABLE_WIDTH texels a row
uniform sampler2D uIndex;      // R32F, slots, TABLE_WIDTH a row
uniform int uOffset;           // this draw's first entry in uIndex
uniform vec3 uCamRender;       // the camera, render frame
uniform mat3 uFieldRot;        // field frame to render frame
uniform vec3 uCamCell[16];     // per size class: the camera's cell
uniform vec3 uCamLocal[16];    // and its offset in it
uniform float uCellSize[16];
uniform float uTime;           // seconds since the table's epoch
uniform float uPixelsPerRadian;
uniform float uMinPixels;      // the smallest diameter drawn
uniform float uClassFar[16];   // per size class: none farther is drawn

in vec4 p3d_Vertex;   // body frame, bounding radius 1
in vec3 p3d_Normal;
in vec3 face_uv;      // equal-angle face parameters a, b; face

out vec3 vRelPos;     // from the camera, render frame
out vec3 vNormal;
out vec3 vUp;         // the radial direction, render frame
out vec3 vFaceUv;
out float vHeight;    // metres above the bounding radius (negative)
flat out vec3 vTu;    // the face's texture axes, render frame
flat out vec3 vTv;
flat out vec4 vQuat;  // body to field frame
flat out vec4 vRock;  // radius, ice, prototype, fade (1 = whole)

#pragma include "../../terrain/shaders/cube_faces.glsl"

const int TABLE_WIDTH = 4096;

vec4 fetch(sampler2D table, int i) {
    return texelFetch(table, ivec2(i % TABLE_WIDTH, i / TABLE_WIDTH), 0);
}

vec4 qmul(vec4 a, vec4 b) {
    return vec4(a.x * b.x - dot(a.yzw, b.yzw),
                a.x * b.yzw + b.x * a.yzw + cross(a.yzw, b.yzw));
}

vec3 qrot(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.yzw, v);
    return v + q.x * t + cross(q.yzw, t);
}

// The turn by |w| * t about w (motion.axis_angle).
vec4 turn(vec3 w, float t) {
    float rate = length(w);
    if (rate < 1e-12) return vec4(1.0, 0.0, 0.0, 0.0);
    float half_angle = 0.5 * rate * t;
    return vec4(cos(half_angle), w / rate * sin(half_angle));
}

void main() {
    int slot = int(fetch(uIndex, uOffset + gl_InstanceID).r + 0.5);
    vec4 t0 = fetch(uInstances, 4 * slot);
    vec4 q0 = fetch(uInstances, 4 * slot + 1);
    vec4 t2 = fetch(uInstances, 4 * slot + 2);
    vec4 t3 = fetch(uInstances, 4 * slot + 3);
    int bits = int(t3.w + 0.5);
    int cls = bits >> 18;
    int proto = (bits >> 6) & 4095;
    float ice = float(bits & 63) / 63.0;
    float radius = t0.w;

    // motion.orientation: rot(axis, precession t) q0 rot(z, spin t)
    vec4 q = qmul(qmul(turn(t2.xyz, uTime), q0), turn(vec3(0.0, 0.0, t2.w), uTime));
    vec3 centre = (t3.xyz - uCamCell[cls]) * uCellSize[cls] + (t0.xyz - uCamLocal[cls]);
    vec3 field = centre + qrot(q, p3d_Vertex.xyz * radius);
    vRelPos = uFieldRot * field;
    gl_Position = p3d_ModelViewProjectionMatrix * vec4(uCamRender + vRelPos, 1.0);

    vNormal = uFieldRot * qrot(q, p3d_Normal);
    vUp = uFieldRot * qrot(q, normalize(p3d_Vertex.xyz));
    vFaceUv = face_uv;
    vHeight = (length(p3d_Vertex.xyz) - 1.0) * radius;
    int face = int(face_uv.z + 0.5);
    vTu = uFieldRot * qrot(q, FACE_U[face]);
    vTv = uFieldRot * qrot(q, FACE_V[face]);
    vQuat = q;
    // Fade out over the last quarter of the drawing distance, and over the
    // last fifth of the class's limit (Population.limits: the field's far, a
    // window still growing; population.FAR_FADE): into the haze.
    float dist = max(length(centre), 1e-3);
    float pixels = 2.0 * radius * uPixelsPerRadian / dist;
    float limit = uClassFar[cls];
    float fade = min(clamp((pixels - uMinPixels) / (0.25 * uMinPixels), 0.0, 1.0),
                     clamp((limit - dist) / max(0.2 * limit, 1e-6), 0.0, 1.0));
    vRock = vec4(radius, ice, float(proto), fade);
}
