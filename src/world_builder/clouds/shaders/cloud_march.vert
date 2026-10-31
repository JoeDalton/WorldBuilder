#version 330
// The raymarch's fullscreen card (clouds.march): its x and z span -1..1, clip
// space directly, whatever the lens.
in vec4 p3d_Vertex;
out vec2 vNdc;

void main() {
    vNdc = p3d_Vertex.xz;
    gl_Position = vec4(p3d_Vertex.x, p3d_Vertex.z, 0.0, 1.0);
}
