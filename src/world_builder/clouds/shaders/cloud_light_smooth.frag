#version 330
// The clouds' light volume's smoothing (clouds.light_volume): a whole bake,
// each voxel the [1, 2, 1] / 4 tent of its neighbours along the tile's two
// axes and across the slices (27 voxels, by 4 bilinear reads half a texel
// off in each of three slices), the edges clamped (a tile's border texels
// and the lowest and highest slices weigh themselves in for the missing
// side).
//
// Why: the bake samples the field at each voxel's centre, and far out
// (the chart's texels hundreds of metres wide) a feature a texel across
// is one voxel with a deep depth beside shallow ones. The readers jitter
// their reads to hide the texels' grid; across such a step a jittered read
// came out dark or bright, a speck the resolve spread over its block in
// motion (performance.md §28). light_volume.smooth_volume on the CPU.

uniform sampler2D uRaw;
// (tile, slices, columns): the atlas's layout (cloud_field.glsl's chart).
uniform vec3 uLayout;

out vec4 fragColor;

// One slice's bilinear read at texel coordinates `local` of its tile
// (clamped to the tile's texel centres: no bleed from the next tile).
vec2 sliceAt(float slice, vec2 local, vec2 size) {
    float tile = uLayout.x;
    float columns = uLayout.z;
    local = clamp(local, vec2(0.5), vec2(tile - 0.5));
    vec2 corner = vec2(mod(slice, columns), floor(slice / columns)) * tile;
    return texture(uRaw, (corner + local) / size).rg;
}

// A slice's 3 x 3 tent around the texel centre `local`: the four bilinear
// reads at the diagonal half-texel offsets (each the mean of a 2 x 2).
vec2 tentAt(float slice, vec2 local, vec2 size) {
    return 0.25 * (sliceAt(slice, local + vec2(-0.5, -0.5), size)
                 + sliceAt(slice, local + vec2(0.5, -0.5), size)
                 + sliceAt(slice, local + vec2(-0.5, 0.5), size)
                 + sliceAt(slice, local + vec2(0.5, 0.5), size));
}

void main() {
    float tile = uLayout.x;
    float slices = uLayout.y;
    float columns = uLayout.z;
    vec2 size = vec2(columns, slices / columns) * tile;
    vec2 at = floor(gl_FragCoord.xy / tile);
    float slice = at.y * columns + at.x;
    vec2 local = gl_FragCoord.xy - at * tile;  // the texel's centre
    // Across the slices: [1, 2, 1] / 4.
    vec2 below = tentAt(max(slice - 1.0, 0.0), local, size);
    vec2 above = tentAt(min(slice + 1.0, slices - 1.0), local, size);
    vec2 own = tentAt(slice, local, size);
    fragColor = vec4(0.25 * below + 0.5 * own + 0.25 * above, 0.0, 1.0);
}
