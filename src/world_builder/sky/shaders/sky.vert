#version 330

// Sky sphere: a unit sphere around the camera, axes = world axes.

in vec4 p3d_Vertex;
uniform mat4 p3d_ModelViewProjectionMatrix;

out vec3 vDir;

void main() {
    vDir = p3d_Vertex.xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
