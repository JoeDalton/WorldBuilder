#version 330
// A game's model (world_builder.objects): its own geometry, with the
// position relative to the main camera for the world's shadows and air.

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;          // model -> render
uniform mat4 p3d_TextureMatrix[1];
uniform vec4 p3d_ClipPlane[1];         // the sea's reflection pass only (view space)
uniform vec3 uCamRender;               // the MAIN camera, render frame (the mirror's too)

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;
in vec4 p3d_Color;

out vec3 vRelPos;   // from the camera, render frame
out vec3 vNormal;   // render frame
out vec2 vUv;
out vec4 vColor;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    gl_ClipDistance[0] = dot(p3d_ClipPlane[0], p3d_ModelViewMatrix * p3d_Vertex);
    vec4 world = p3d_ModelMatrix * p3d_Vertex;
    vRelPos = world.xyz / world.w - uCamRender;
    // Scaled unevenly too: the inverse transpose.
    vNormal = transpose(inverse(mat3(p3d_ModelMatrix))) * p3d_Normal;
    vUv = (p3d_TextureMatrix[0] * vec4(p3d_MultiTexCoord0, 0.0, 1.0)).xy;
    vColor = p3d_Color;
}
