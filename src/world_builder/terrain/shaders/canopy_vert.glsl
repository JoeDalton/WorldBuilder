// The canopy's shell (world_builder.terrain.vegetation.Canopy): a chunk's own
// vertices (its vertex data, drawn again by the shell's index buffer), its
// grid lifted over the crowns around each vertex (the canopy layer's lift
// rows: vegetation.shell_lift; none where nothing grows) and its skirt set
// on the edge's ground: a closed volume over the chunk's plants, tight
// enough that most view rays never enter it. canopy_frag.glsl marches what
// grows inside, from where the view ray enters it.
//
// The light is reckoned here, per vertex of the shell, not per pixel: the
// sunlight with every shadow but the plants' own (terrain_light.glsl's
// surfaceLight), the sky light and the air to the camera. They change over
// hundreds of metres, the shell's vertices are a few metres apart, and the
// hit lies a few metres behind its shell: a third of the canopy's cost per
// pixel saved. The chunk's frame (its canopy layer's last row, its table
// rows) goes on flat.

in vec4 p3d_Vertex;     // relative to the chunk centre, planet axes (metres)
in float height;        // metres above sea level (a skirt's own, below its edge)
in vec4 biome_a;        // snow, rock, grass, forest
in vec4 biome_b;        // desert, strata mask, slot low byte, slot high | skirt

#define TERRAIN_LIGHT_VERTEX
#pragma include "terrain_light.glsl"
#pragma include "chunk_grid.glsl"

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;   // chunk -> render: the body's orientation, then a translation
uniform sampler2D uChunkTable;  // terrain.py: the chunks' own values, by slot
uniform sampler2DArray uCanopy; // the canopy layers (vegetation.canopy_layer)

out vec3 vLocal;     // the shell's point, from the chunk's centre, planet axes
out vec3 vRay;       // from the camera to it, planet axes
out float vLift;     // the shell's height over the ground (0: nothing grows)
out vec3 vSun;       // the sunlight there (every shadow but the plants')
out vec3 vSky;       // the sky light there
out vec3 vAirT;      // the air to the camera: transmittance
out vec3 vAirS;      // and inscatter
// The chunk (canopy_frag.glsl's frame).
flat out vec4 vEu;      // metres per grid step along u (tangent), w: the radius
flat out vec3 vEv;
flat out vec3 vFu;      // grid steps per metre (the dual axes)
flat out vec3 vFv;
flat out vec3 vOffset;  // O (the middle vertex at sea level) from the centre
flat out vec3 vUpO;     // the vertical at O
flat out vec3 vCentre;  // the chunk's centre, from the planet's
flat out vec3 vUvStep;  // its first vertex's uv, uv metres per grid step
flat out ivec4 vIds;    // layer, face, period u, v

void main() {
    int high = int(biome_b.a * 255.0 + 0.5);
    int slot = int(biome_b.b * 255.0 + 0.5) + 256 * (high & 127);
    bool skirt = high >= 128;
    vec4 centre = texelFetch(uChunkTable, ivec2(slot, 0), 0);
    vec4 chunk = texelFetch(uChunkTable, ivec2(slot, 1), 0);
    vec4 ids = texelFetch(uChunkTable, ivec2(slot, 2), 0);
    vec4 vegetation = texelFetch(uChunkTable, ivec2(slot, 3), 0);
    vec3 up = normalize(centre.xyz + p3d_Vertex.xyz);
    int layer = int(vegetation.z + 0.5);
    ivec2 g = ivec2(gridCoord(gl_VertexID) + 0.5);
    float lift = texelFetch(uCanopy, ivec3(g.x, VERTS + 1 + g.y, layer), 0).r;
    vec4 local = p3d_Vertex;
    // The grid rises by the lift, the skirt to its edge's ground: walls.
    local.xyz += up * (skirt ? chunk.w : lift);
    gl_Position = p3d_ModelViewProjectionMatrix * local;
    mat3 rotation = mat3(p3d_ModelMatrix);
    vec3 world = (p3d_ModelMatrix * local).xyz;
    vec3 relPos = world - uCamPos;
    vLocal = local.xyz;
    vRay = transpose(rotation) * relPos;
    vLift = lift;

    vEu = texelFetch(uCanopy, ivec3(0, VERTS, layer), 0);
    vEv = texelFetch(uCanopy, ivec3(1, VERTS, layer), 0).xyz;
    vFu = texelFetch(uCanopy, ivec3(2, VERTS, layer), 0).xyz;
    vFv = texelFetch(uCanopy, ivec3(3, VERTS, layer), 0).xyz;
    vOffset = texelFetch(uCanopy, ivec3(4, VERTS, layer), 0).xyz;
    vUpO = texelFetch(uCanopy, ivec3(5, VERTS, layer), 0).xyz;
    vCentre = centre.xyz;
    vUvStep = chunk.xyz;
    vIds = ivec4(layer, int(ids.y + 0.5), int(vegetation.x + 0.5), int(vegetation.y + 0.5));

#ifdef CANOPY_INSIDE
    // The inside is the camera's chunk's only: elsewhere every triangle is
    // dropped (clipped whole), or the shells near the camera would each
    // cover the view with fragments that find nothing.
    vec3 cam = transpose(rotation) * (uCamPos - (p3d_ModelMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz)
             - vOffset;
    vec2 camGrid = vec2(32.0) + vec2(dot(vFu, cam), dot(vFv, cam));
    if (any(lessThan(camGrid, vec2(-1.0))) || any(greaterThan(camGrid, vec2(65.0)))) {
        gl_Position = vec4(0.0, 0.0, 0.0, -1.0);
    }
#endif

    // The light at the shell, half way down to the ground (where the
    // crowns' middles are), lifted above the ground's own shadow bias.
    vec3 L = normalize(uSunDir);
    vec3 upR = normalize(rotation * up);
    vec3 at = relPos - upR * (0.5 * lift);
    vec3 planetPos = centre.xyz + local.xyz - up * (0.5 * lift);
    float sunDay, shade;
    float footprint = max(1.0, 0.002 * length(relPos));
    surfaceLight(at, planetPos, height + 0.5 * lift, upR, upR, upR, L, footprint, true,
                 vSun, vSky, sunDay, shade);
    vSun *= sunDay;
    airAlong(at, L, vAirT, vAirS);
}
