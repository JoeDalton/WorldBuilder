// The clouds in front of a game's translucent surface (world_builder.clouds.cover).
//
// The clouds are laid over the view as a card that writes no depth, so a
// translucent surface that writes none either (a shield bubble: depth write
// off, drawn after the clouds) would be blended over every cloud on screen,
// one in front of it included. This gives the surface the cloud in front of
// it, per pixel, to fade itself by:
//
//     #pragma include "cloud_cover.glsl"
//     ...
//     float cover = cloudCoverBefore(gl_FragCoord.xy, length(viewPos));
//     colour *= 1.0 - cover;   // straight alpha: colour and alpha;
//     alpha *= 1.0 - cover;    // premultiplied: the colour alone
//
// A dense cloud in front hides the surface, a thin one lets it through, a
// cloud behind it stays behind (it is drawn over the clouds). Its inputs are
// set by World.track_translucent(node); with no cloud (open space, no window,
// a view other than the main one) cloudCoverBefore is 0. One distance per
// pixel (the cloud's representative one, not its front): a surface inside a
// cloud is all in front of it or all behind. The cirrus decks are not
// covered. Works in GLSL 1.40 and later.

#ifndef CLOUD_COVER_GLSL
#define CLOUD_COVER_GLSL

uniform sampler2D uCloudCover;          // the clouds over the view, premultiplied
uniform sampler2D uCloudCoverDistance;  // their distance from the camera, km; 0 none
uniform vec2 uCloudCoverDownscale;      // view pixels per pixel of theirs
uniform int uCloudCoverOn;              // 0: no clouds in this view

// The opacity of the cloud between the camera and a fragment.
//   fragCoord: gl_FragCoord.xy (the main view's pixels)
//   distance:  the fragment's distance from the camera, metres (the length
//              of its view-space position)
float cloudCoverBefore(vec2 fragCoord, float distance) {
    if (uCloudCoverOn == 0) {
        return 0.0;
    }
    vec2 uv = fragCoord / uCloudCoverDownscale / vec2(textureSize(uCloudCover, 0));
    float alpha = texture(uCloudCover, uv).a;
    float cloud = texture(uCloudCoverDistance, uv).r * 1000.0;
    if (alpha <= 0.0 || cloud <= 0.0 || cloud >= distance) {
        return 0.0;
    }
    return clamp(alpha, 0.0, 1.0);
}

#endif
