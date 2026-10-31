#version 330
// The cirrus sheet (clouds.cirrus): a sphere at the slab's mid-height, shaded
// by cirrus.frag on the true sphere. Positions are in the planet frame (the
// node sits at its centre).
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrixInverse;

in vec4 p3d_Vertex;

out vec3 vPosition;
// The camera drawing this pass, planet frame: the sea's mirror has its own,
// and which crossing of the sheet a fragment is depends on it.
flat out vec3 vEye;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    gl_ClipDistance[0] = 1.0;  // above the sea (see cloud_sphere.vert)
    vPosition = p3d_Vertex.xyz;
    vEye = (p3d_ModelViewMatrixInverse * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
}
