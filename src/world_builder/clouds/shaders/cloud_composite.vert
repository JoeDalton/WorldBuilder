#version 330
// The raymarch's composite (clouds.march): a fullscreen card in one depth
// range's pass; its x and z span -1..1, clip space directly.
in vec4 p3d_Vertex;
out vec2 vNdc;

void main() {
    vNdc = p3d_Vertex.xz;
    gl_Position = vec4(p3d_Vertex.x, p3d_Vertex.z, 0.0, 1.0);
}
