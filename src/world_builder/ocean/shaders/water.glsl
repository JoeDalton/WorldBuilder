// The water's colour by depth, shared by the sea's near look (sea.glsl) and
// its far look (terrain_frag.glsl), so the sea keeps its colour as they fade.

// Share of the deep water's colour over a sea floor this deep (metres), seen
// cosView from the vertical, the rest being the shallow colour: Beer-Lambert
// along the light's way down to the floor (1) and the view's back up
// (1 / cosView), with an e-folding depth of depthScale.
float deepWaterShare(float depth, float cosView, float depthScale) {
    float path = max(depth, 0.0) * (1.0 + 1.0 / max(cosView, 0.1));
    return 1.0 - exp(-path / max(depthScale, 1e-3));
}
