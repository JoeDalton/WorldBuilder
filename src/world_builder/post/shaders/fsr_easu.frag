#version 140
// FSR 1 EASU (Edge Adaptive Spatial Upsampling), ported to GLSL 1.40 from
// AMD FidelityFX Super Resolution 1 (ffx_fsr1.h, v1.20210629, FsrEasuF).
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// Port notes:
// - The 4 textureGather calls (GLSL 4.00) become 12 texelFetch taps in pixel
//   coordinates, which also makes power-of-two padding irrelevant.
// - AMD's bit-trick approximate reciprocals need floatBitsToUint (GLSL 3.30);
//   exact 1.0/x and inversesqrt are used instead, guarded against 0 so flat
//   (e.g. black) regions don't produce 0 * inf = NaN.
// - GL's bottom-left origin mirrors the kernel vertically for both input and
//   output, which the vertically symmetric 12-tap footprint is unaffected by.

uniform sampler2D inputTex;
uniform vec2 inputSize;   // pixels of valid content in inputTex
uniform vec2 outputSize;  // pixels of the target

out vec4 p3d_FragColor;

float rcpSafe(float x) { return 1.0 / max(x, 1e-30); }

vec3 tap(ivec2 p) {
    ivec2 hi = ivec2(inputSize) - 1;
    return texelFetch(inputTex, clamp(p, ivec2(0), hi), 0).rgb;
}

float luma2(vec3 c) { return c.b * 0.5 + (c.r * 0.5 + c.g); }

void easuTap(inout vec3 aC, inout float aW, vec2 off, vec2 dir, vec2 len,
             float lob, float clp, vec3 c) {
    vec2 v;
    v.x = (off.x * ( dir.x)) + (off.y * dir.y);
    v.y = (off.x * (-dir.y)) + (off.y * dir.x);
    v *= len;
    float d2 = min(v.x * v.x + v.y * v.y, clp);
    float wB = (2.0 / 5.0) * d2 - 1.0;
    float wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = (25.0 / 16.0) * wB - (25.0 / 16.0 - 1.0);
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

// a: above, b: left, c: centre, d: right, e: below; w: bilinear weight.
void easuSet(inout vec2 dir, inout float len, float w,
             float lA, float lB, float lC, float lD, float lE) {
    float dc = lD - lC;
    float cb = lC - lB;
    float lenX = rcpSafe(max(abs(dc), abs(cb)));
    float dirX = lD - lB;
    dir.x += dirX * w;
    lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);
    lenX *= lenX;
    len += lenX * w;

    float ec = lE - lC;
    float ca = lC - lA;
    float lenY = rcpSafe(max(abs(ec), abs(ca)));
    float dirY = lE - lA;
    dir.y += dirY * w;
    lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);
    lenY *= lenY;
    len += lenY * w;
}

void main() {
    // FsrEasuCon, with the viewport equal to the input size.
    vec2 scale = inputSize / outputSize;
    vec2 ip = floor(gl_FragCoord.xy);
    vec2 pp = ip * scale + (0.5 * scale - 0.5);
    vec2 fp = floor(pp);
    pp -= fp;
    ivec2 f0 = ivec2(fp);

    //    b c
    //  e f g h
    //  i j k l
    //    n o
    vec3 b = tap(f0 + ivec2( 0, -1));
    vec3 c = tap(f0 + ivec2( 1, -1));
    vec3 e = tap(f0 + ivec2(-1,  0));
    vec3 f = tap(f0);
    vec3 g = tap(f0 + ivec2( 1,  0));
    vec3 h = tap(f0 + ivec2( 2,  0));
    vec3 i = tap(f0 + ivec2(-1,  1));
    vec3 j = tap(f0 + ivec2( 0,  1));
    vec3 k = tap(f0 + ivec2( 1,  1));
    vec3 l = tap(f0 + ivec2( 2,  1));
    vec3 n = tap(f0 + ivec2( 0,  2));
    vec3 o = tap(f0 + ivec2( 1,  2));

    float bL = luma2(b), cL = luma2(c), eL = luma2(e), fL = luma2(f);
    float gL = luma2(g), hL = luma2(h), iL = luma2(i), jL = luma2(j);
    float kL = luma2(k), lL = luma2(l), nL = luma2(n), oL = luma2(o);

    vec2 dir = vec2(0.0);
    float len = 0.0;
    easuSet(dir, len, (1.0 - pp.x) * (1.0 - pp.y), bL, eL, fL, gL, jL);
    easuSet(dir, len,        pp.x  * (1.0 - pp.y), cL, fL, gL, hL, kL);
    easuSet(dir, len, (1.0 - pp.x) *        pp.y , fL, iL, jL, kL, nL);
    easuSet(dir, len,        pp.x  *        pp.y , gL, jL, kL, lL, oL);

    vec2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    bool zro = dirR < (1.0 / 32768.0);
    dirR = zro ? 1.0 : inversesqrt(dirR);
    dir.x = zro ? 1.0 : dir.x;
    dir *= dirR;
    len = len * 0.5;
    len *= len;
    float stretch = (dir.x * dir.x + dir.y * dir.y) / max(abs(dir.x), abs(dir.y));
    vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp = 1.0 / lob;

    vec3 min4 = min(min(f, g), min(j, k));
    vec3 max4 = max(max(f, g), max(j, k));

    vec3 aC = vec3(0.0);
    float aW = 0.0;
    easuTap(aC, aW, vec2( 0.0, -1.0) - pp, dir, len2, lob, clp, b);
    easuTap(aC, aW, vec2( 1.0, -1.0) - pp, dir, len2, lob, clp, c);
    easuTap(aC, aW, vec2(-1.0,  1.0) - pp, dir, len2, lob, clp, i);
    easuTap(aC, aW, vec2( 0.0,  1.0) - pp, dir, len2, lob, clp, j);
    easuTap(aC, aW, vec2( 0.0,  0.0) - pp, dir, len2, lob, clp, f);
    easuTap(aC, aW, vec2(-1.0,  0.0) - pp, dir, len2, lob, clp, e);
    easuTap(aC, aW, vec2( 1.0,  1.0) - pp, dir, len2, lob, clp, k);
    easuTap(aC, aW, vec2( 2.0,  1.0) - pp, dir, len2, lob, clp, l);
    easuTap(aC, aW, vec2( 2.0,  0.0) - pp, dir, len2, lob, clp, h);
    easuTap(aC, aW, vec2( 1.0,  0.0) - pp, dir, len2, lob, clp, g);
    easuTap(aC, aW, vec2( 1.0,  2.0) - pp, dir, len2, lob, clp, o);
    easuTap(aC, aW, vec2( 0.0,  2.0) - pp, dir, len2, lob, clp, n);

    // Normalise and dering against the nearest 2x2.
    vec3 pix = min(max4, max(min4, aC / aW));
    p3d_FragColor = vec4(pix, 1.0);
}
