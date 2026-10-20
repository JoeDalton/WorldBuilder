#version 330

// Sky sphere: a unit sphere around the camera, axes = world axes.

in vec4 p3d_Vertex;
uniform mat4 p3d_ModelViewProjectionMatrix;

out vec3 vDir;

void main() {
    vDir = p3d_Vertex.xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    // On the far plane, whatever the camera's range: the sphere is drawn
    // first, without depth, so only its direction matters, and a camera
    // with a shorter range (the ocean's reflection) still sees it whole.
    gl_Position.z = gl_Position.w * 0.999999;
    // Never under the water: the reflection pass's clip plane passes it.
    gl_ClipDistance[0] = 1.0;
}
