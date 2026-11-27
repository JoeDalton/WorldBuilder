#version 140
// FSR 1 RCAS (Robust Contrast Adaptive Sharpening), ported to GLSL 1.40 from
// AMD FidelityFX Super Resolution 1 (ffx_fsr1.h, v1.20210629, FsrRcasF).
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
// Port notes: no FSR_RCAS_DENOISE / PASSTHROUGH_ALPHA. Drawn into a window
// region that need not start at the window's corner: outputOrigin. The limiter
// reciprocals are guarded: pure black (mx4 == 0) or pure white (mn4 == 1)
// neighbourhoods would otherwise compute 0 / 0.

uniform sampler2D inputTex;
uniform vec2 inputSize;   // pixels of valid content in inputTex (== output)
uniform vec2 outputOrigin;  // the output region's lower-left pixel in its window
uniform float sharpness;  // stops: 0 = maximum, each +1 halves the sharpening

out vec4 p3d_FragColor;

const float FSR_RCAS_LIMIT = 0.25 - (1.0 / 16.0);

vec3 load(ivec2 p) {
    ivec2 hi = ivec2(inputSize) - 1;
    return texelFetch(inputTex, clamp(p, ivec2(0), hi), 0).rgb;
}

void main() {
    //    b
    //  d e f
    //    h
    ivec2 sp = ivec2(gl_FragCoord.xy - outputOrigin);
    vec3 b = load(sp + ivec2( 0, -1));
    vec3 d = load(sp + ivec2(-1,  0));
    vec3 e = load(sp);
    vec3 f = load(sp + ivec2( 1,  0));
    vec3 h = load(sp + ivec2( 0,  1));

    vec3 mn4 = min(min(b, d), min(f, h));
    vec3 mx4 = max(max(b, d), max(f, h));

    vec3 hitMin = min(mn4, e) / max(4.0 * mx4, vec3(1e-30));
    vec3 hitMax = (1.0 - max(mx4, e)) / min(4.0 * mn4 - 4.0, vec3(-1e-30));
    vec3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(-FSR_RCAS_LIMIT, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0))
                 * exp2(-sharpness);

    vec3 pix = (lobe * (b + d + h + f) + e) / (4.0 * lobe + 1.0);
    p3d_FragColor = vec4(pix, 1.0);
}
