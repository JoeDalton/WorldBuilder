// The sky-view table (world_builder.atmosphere.sky_view; Hillaire 2020): the
// air between the camera and space, per direction, baked once a frame around
// the camera while it is inside an atmosphere, read by the sky and the stars
// instead of marching each pixel or vertex.
//
// Include after atmosphere.glsl. Its frame: the camera's vertical (from its
// atmosphere's centre) and the sun's azimuth (u 0.5 toward the sun, 0 and 1
// away from it, crowded toward the sun by a square root for its glow); v
// from the zenith (0) through the horizon (0.5, the ground's tangent) to the
// nadir (1), crowded toward the horizon by square roots. Rows [0, H): the
// in-scatter; rows [H, 2H): the transmittance.

uniform sampler2D uSkyView;
// 1: the table is baked (SkyView); 0: march.
uniform int uSkyViewOn;
// Powers of two: Panda pads a render target's texture up to them, its
// content in a corner (192 x 216 read as 256 x 256 was garbled).
const vec2 SKY_VIEW_SIZE = vec2(256.0, 128.0);

// The atmosphere the camera is in (the table's), -1 none (from space: no
// table, the march).
int skyViewAtmosphere() {
    for (int i = 0; i < MAX_ATMOSPHERES; ++i) {
        if (i < uAtmoCount && uAtmoCam[i].w < 0.0) {
            return i;
        }
    }
    return -1;
}

// The table's axes: the camera's vertical, toward the sun's azimuth, and the
// third.
void skyViewFrame(int a, vec3 sunDir, out vec3 up, out vec3 east, out vec3 north) {
    up = normalize(uAtmoCam[a].xyz);
    vec3 e = sunDir - up * dot(sunDir, up);
    if (dot(e, e) < 1e-10) {  // the sun at the zenith or nadir: any azimuth
        e = abs(up.x) < 0.9 ? cross(up, vec3(1.0, 0.0, 0.0))
                            : cross(up, vec3(0.0, 1.0, 0.0));
    }
    east = normalize(e);
    north = cross(up, east);
}

// The horizon's dip below the horizontal, radians (from |cam|^2 - R^2 in
// float64: as float32 R / r it would be noise near the ground).
float skyViewDip(int a) {
    return atan(sqrt(max(uAtmoMie[a].w, 0.0)), uAtmoShape[a].x);
}

// (u, v) in [0, 1] of a direction.
vec2 skyViewUv(int a, vec3 dir, vec3 sunDir) {
    vec3 up, east, north;
    skyViewFrame(a, sunDir, up, east, north);
    float theta = acos(clamp(dot(dir, up), -1.0, 1.0));
    float dip = skyViewDip(a);
    float horizon = 0.5 * ATMO_PI + dip;  // from the zenith
    float below = 0.5 * ATMO_PI - dip;    // from the horizon to the nadir
    float v = theta < horizon
            ? 0.5 * (1.0 - sqrt(max(1.0 - theta / horizon, 0.0)))
            : 0.5 + 0.5 * sqrt(clamp((theta - horizon) / max(below, 1e-6), 0.0, 1.0));
    float phi = atan(dot(dir, north), dot(dir, east));
    float u = 0.5 + 0.5 * sign(phi) * sqrt(abs(phi) / ATMO_PI);
    return vec2(u, v);
}

// The direction at (u, v): skyViewUv's inverse.
vec3 skyViewDirection(int a, vec2 uv, vec3 sunDir) {
    vec3 up, east, north;
    skyViewFrame(a, sunDir, up, east, north);
    float dip = skyViewDip(a);
    float horizon = 0.5 * ATMO_PI + dip;
    float below = 0.5 * ATMO_PI - dip;
    float theta;
    if (uv.y < 0.5) {
        float x = 1.0 - 2.0 * uv.y;
        theta = horizon * (1.0 - x * x);
    } else {
        float x = 2.0 * uv.y - 1.0;
        theta = horizon + below * x * x;
    }
    float s = 2.0 * uv.x - 1.0;
    float phi = sign(s) * s * s * ATMO_PI;
    return cos(theta) * up + sin(theta) * (cos(phi) * east + sin(phi) * north);
}

// The air from the camera to space along dir, from the table: false where
// it holds none (not baked, or the camera in no atmosphere: march then).
bool skyViewAlong(vec3 dir, vec3 sunDir, out vec3 transmittance, out vec3 inscatter) {
    transmittance = vec3(1.0);
    inscatter = vec3(0.0);
    if (uSkyViewOn == 0) {
        return false;
    }
    int a = skyViewAtmosphere();
    if (a < 0) {
        return false;
    }
    vec2 uv = skyViewUv(a, dir, sunDir);
    // Each half's rows only (the two meet at v 0.5 of the texture).
    float v = 0.5 * clamp(uv.y, 0.5 / SKY_VIEW_SIZE.y, 1.0 - 0.5 / SKY_VIEW_SIZE.y);
    inscatter = textureLod(uSkyView, vec2(uv.x, v), 0.0).rgb;
    transmittance = textureLod(uSkyView, vec2(uv.x, v + 0.5), 0.0).rgb;
    return true;
}
