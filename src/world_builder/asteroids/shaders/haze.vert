#version 330

// The hull around a field's shape (world_builder.asteroids.haze): each
// vertex relative to the camera, field frame, for the fragments' rays.
// Points beyond the far plane are pulled onto it, as the sky's are, so a
// ring across the sky is drawn whole.

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform vec3 uCamRender;   // the camera, render frame
uniform mat3 uFieldRot;    // field frame to render frame
uniform vec3 uCamField;    // the camera, field frame

in vec4 p3d_Vertex;        // field frame
in vec3 p3d_Normal;        // the hull's outward normal, field frame

out vec3 vRay;             // from the camera, field frame
out vec3 vNormal;

void main() {
    vRay = p3d_Vertex.xyz - uCamField;
    vNormal = p3d_Normal;
    gl_Position = p3d_ModelViewProjectionMatrix * vec4(uCamRender + uFieldRot * vRay, 1.0);
    if (gl_Position.w > 0.0 && gl_Position.z > gl_Position.w)
        gl_Position.z = gl_Position.w * 0.999999;
}
