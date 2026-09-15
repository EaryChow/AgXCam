package com.agx.camera.gpu

/**
 * Shared GLSL helpers for the multi-stage denoise pipeline (Stages 1/2/3/5).
 *
 * All passes operate on the demosaic output grid ("sparse Bayer grid"): each
 * output texel covers one 2x2 sensor cell and carries the four CFA phases in
 * R/G/B/A. This is the SAME indexing contract the placeholder denoiser used
 * (C3 constraint) — no second indexing system is introduced anywhere.
 */
object DenoiseGlsl {

    /** Mosaic sampling + black level + ISO noise-model helper header.
     *  Requires uniforms: u_cropOrigin, u_cropSize, u_viewSize(vec2),
     *  u_sensorSize, u_black_level_pattern(ivec4), u_iso_model_a / u_iso_model_b (float). */
    const val MOSAIC_HELPERS = """
// sensor coordinate of this output texel (cell origin)
ivec2 cellOrigin() {
    vec2 sensorUV = u_cropOrigin + v_texCoord * u_cropSize;
    sensorUV = clamp(sensorUV, vec2(0.0), u_sensorSize - vec2(1.0));
    return ivec2(floor(sensorUV));
}

// output-grid texel index covering the 2x2 cell that contains sensor position p
ivec2 cellTexelFor(ivec2 sensorPos) {
    ivec2 cell = ivec2(sensorPos.x - abs(sensorPos.x % 2), sensorPos.y - abs(sensorPos.y % 2));
    vec2 uv = (vec2(cell) - u_cropOrigin) / u_cropSize;
    ivec2 oc = ivec2(floor(uv * u_viewSize));
    return clamp(oc, ivec2(0), ivec2(u_viewSize) - ivec2(1));
}

// black-level-subtracted value (kept signed) of the given phase member of the
// given 2x2 cell, sampled from the sparse mosaic texture.
float phaseValue(float phaseVal, int phase) {
    return phaseVal - float(u_black_level_pattern[phase]);
}

float isoModelSigma(float signal) {
    float s2 = u_iso_model_a * max(signal, 0.0) + u_iso_model_b;
    return sqrt(max(s2, 1.0e-6));
}

float isoModelSigmaSq(float signal) {
    return max(u_iso_model_a * max(signal, 0.0) + u_iso_model_b, 1.0e-6);
}
"""

}
