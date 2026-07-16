package com.agx.camera.gpu

object AgxCoreGlsl {

    const val CORE_HELPERS = """
float spowf(float a, float b) {
    return sign(a) * pow(abs(a), b);
}

vec3 spowf3(vec3 x, float p) {
    return sign(x) * pow(abs(x), vec3(p));
}

vec3 lin2log(vec3 rgb, float logMin, float logMax) {
    float logFloor = 0.18 * pow(2.0, logMin);
    rgb = max(rgb, vec3(logFloor));
    rgb = log2(rgb / 0.18);
    rgb = clamp(rgb, logMin, logMax);
    return (rgb + abs(logMin)) / (abs(logMin) + abs(logMax));
}

float sigmoid(float x, float sp, float tp, float slope, float px, float py) {
    float s0 = 1.0;
    float t0 = 0.0;
    float ss = spowf(
        ((spowf((slope * ((s0 - px) / (1.0 - py))), sp) - 1.0) * (spowf(slope * (s0 - px), -sp))),
        -1.0 / sp);
    float ms = slope * (x - px) / ss;
    float fs = ms / spowf(1.0 + spowf(ms, sp), 1.0 / sp);
    float ts = spowf(
        ((spowf((slope * ((px - t0) / py)), tp) - 1.0) * (spowf(slope * (px - t0), -tp))),
        -1.0 / tp);
    float mr = (slope * (x - px)) / -ts;
    float ft = mr / spowf(1.0 + spowf(mr, tp), 1.0 / tp);
    return x >= px ? ss * fs + py : (-ts * ft) + py;
}

vec3 compensateLowSide(vec3 rgb) {
    const vec3 lumCoeffs = vec3(0.2589235355689848, 0.6104985346066525, 0.13057792982436284);
    vec3 rgb2020 = u_709_to_2020 * rgb;
    float Y = dot(rgb2020, lumCoeffs);
    float maxRGB = max(rgb.r, max(rgb.g, rgb.b));
    vec3 inverseRGB = vec3(maxRGB - rgb.r, maxRGB - rgb.g, maxRGB - rgb.b);
    float maxInvRGB = max(inverseRGB.r, max(inverseRGB.g, inverseRGB.b));
    vec3 inv2020 = u_709_to_2020 * inverseRGB;
    float Yinv = dot(inv2020, lumCoeffs);
    float yCompensate = (maxInvRGB - Yinv + Y);
    float minRGB = min(rgb.r, min(rgb.g, rgb.b));
    float offset = max(-minRGB, 0.0);
    vec3 rgbOffset = rgb + offset;
    float maxOffset = max(rgbOffset.r, max(rgbOffset.g, rgbOffset.b));
    vec3 invOffset = vec3(maxOffset - rgbOffset.r, maxOffset - rgbOffset.g, maxOffset - rgbOffset.b);
    float maxInvOff = max(invOffset.r, max(invOffset.g, invOffset.b));
    vec3 invOff2020 = u_709_to_2020 * invOffset;
    float YinvOff = dot(invOff2020, lumCoeffs);
    vec3 off2020 = u_709_to_2020 * rgbOffset;
    float Ynew = dot(off2020, lumCoeffs);
    float yNewCompensate = (maxInvOff - YinvOff + Ynew);
    float ratio = (yNewCompensate > yCompensate) ? (yCompensate / yNewCompensate) : 1.0;
    return max(rgbOffset * ratio, vec3(0.0));
}

vec3 srgbOETF(vec3 linear) {
    vec3 result;
    for (int i = 0; i < 3; i++) {
        float c = linear[i];
        if (c <= 0.0031308) {
            result[i] = c * 12.92;
        } else {
            result[i] = 1.055 * pow(c, 1.0/2.4) - 0.055;
        }
    }
    return result;
}

vec3 srgbEOTF(vec3 srgb) {
    vec3 result;
    for (int i = 0; i < 3; i++) {
        float c = srgb[i];
        if (c <= 0.04045) {
            result[i] = c / 12.92;
        } else {
            result[i] = pow((c + 0.055) / 1.055, 2.4);
        }
    }
    return result;
}
"""

    const val AGX_FORMATION = """
vec3 agxFormation(vec3 sensorLinear) {
    vec3 rgb = sensorLinear / (u_white_level - u_black_level);
    rgb = u_scene_linear_to_709 * rgb;
    rgb = compensateLowSide(rgb);
    rgb = u_insetmat * rgb;
    rgb = lin2log(rgb, u_log_min, u_log_max);
    rgb.r = sigmoid(rgb.r, u_shoulder, u_toe, u_contrast, u_log_midgray, u_display_midgray);
    rgb.g = sigmoid(rgb.g, u_shoulder, u_toe, u_contrast, u_log_midgray, u_display_midgray);
    rgb.b = sigmoid(rgb.b, u_shoulder, u_toe, u_contrast, u_log_midgray, u_display_midgray);
    rgb = spowf3(rgb, 2.4);
    rgb = u_outsetmat * rgb;
    rgb = clamp(rgb, 0.0, 1.0);
    rgb = srgbOETF(rgb);
    return rgb;
}
"""
}
