// Each cube face's u and v axes (cubesphere.FACE_FRAMES), in the body's
// frame: the textures' directions, once projected on the tangent plane.
// Include once per shader with #pragma include.
const vec3 FACE_U[6] = vec3[6](
    vec3(0, 1, 0), vec3(0, -1, 0), vec3(-1, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 1, 0));
const vec3 FACE_V[6] = vec3[6](
    vec3(0, 0, 1), vec3(0, 0, 1), vec3(0, 0, 1), vec3(0, 0, 1), vec3(-1, 0, 0), vec3(1, 0, 0));
