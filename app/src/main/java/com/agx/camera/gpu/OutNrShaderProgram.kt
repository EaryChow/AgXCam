package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Stage 5 — output-domain denoise (plan §3 Stage 5): weighted YC1C2 + SWGF
 * self-guided filter (I=p, 8 side windows, min-variance window chosen) with a
 * Fast-GF style separable box-statistic prefilter. The pass is designed to
 * run twice with a β=0.3 noise return and κ×1.4 on round 2, but the host
 * currently runs a single iteration (round 1 only) while double-pass behavior
 * is still being tuned.
 *
 * Passes (all one texel per output pixel, same resolution as the demosaic FBO):
 *  statsH : horizontal 5-tap box of Y (and Y²) of the (possibly β-blended)
 *           iteration input.
 *  statsV : vertical 5-tap box of the statsH output → full 5x5 box stats.
 *  main   : YC1C2 → per-pixel ε = scale·(σ̂² + σ_dm²), pick the min-variance side
 *           window (sampling statsV at 8 window centres), filter Y and C1/C2
 *           with luma-driven coefficients, invert back to RGB.
 *
 * ε for luma (Y) uses a small scale so texture is preserved; ε for chroma
 * (C1/C2) uses an independent large scale (plan: ×6~16) because C1/C2
 * variance is compressed but strongly spatially correlated → chroma bends to
 * the dense chroma box at the pixel to kill low-frequency chromatic blotches.
 */
class OutNrShaderProgram {

    private var statsHProgramId = 0
    private var statsVProgramId = 0
    private var mainProgramId = 0
    private var ready = false

    private var hInTexLoc = 0
    private var hBaseTexLoc = 0
    private var hBetaLoc = 0
    private var hYWeightsLoc = 0
    private var hWinScaleLoc = 0

    private var vStatsTexLoc = 0
    private var vWinScaleLoc = 0

    private var mInTexLoc = 0
    private var mBaseTexLoc = 0
    private var mStatsTexLoc = 0
    private var mSigmaTexLoc = 0
    private var mSigmaSizeLoc = 0
    private var mOutSizeLoc = 0
    private var mBetaLoc = 0
    private var mLumaEpsScaleLoc = 0
    private var mChromaEpsScaleLoc = 0
    private var mSigmaDm2Loc = 0
    private var mInverseRange2Loc = 0
    private var mEpsScaleLoc = 0
    private var mUseIsoSigmaLoc = 0
    private var mIsoModelALoc = 0
    private var mIsoModelBLoc = 0
    private var mWinScaleLoc = 0
    private var mEpsBoostLoc = 0

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create() {
        statsHProgramId = createProgram(VERTEX_SHADER, STATS_H_FRAGMENT)
        statsVProgramId = createProgram(VERTEX_SHADER, STATS_V_FRAGMENT)
        mainProgramId = createProgram(VERTEX_SHADER, MAIN_FRAGMENT)
        if (statsHProgramId == 0 || statsVProgramId == 0 || mainProgramId == 0) {
            Log.e(TAG, "Failed to create Stage-5 denoise programs")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create Stage-5 denoise programs")
            return
        }

        hInTexLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_inTex")
        hBaseTexLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_baseTex")
        hBetaLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_beta")
        hYWeightsLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_yWeights")
        hWinScaleLoc = GLES20.glGetUniformLocation(statsHProgramId, "u_win_scale")

        vStatsTexLoc = GLES20.glGetUniformLocation(statsVProgramId, "u_statsTex")
        vWinScaleLoc = GLES20.glGetUniformLocation(statsVProgramId, "u_win_scale")

        mInTexLoc = GLES20.glGetUniformLocation(mainProgramId, "u_inTex")
        mBaseTexLoc = GLES20.glGetUniformLocation(mainProgramId, "u_baseTex")
        mStatsTexLoc = GLES20.glGetUniformLocation(mainProgramId, "u_statsTex")
        mSigmaTexLoc = GLES20.glGetUniformLocation(mainProgramId, "u_sigmaTex")
        mSigmaSizeLoc = GLES20.glGetUniformLocation(mainProgramId, "u_sigmaSize")
        mOutSizeLoc = GLES20.glGetUniformLocation(mainProgramId, "u_outSize")
        mBetaLoc = GLES20.glGetUniformLocation(mainProgramId, "u_beta")
        mLumaEpsScaleLoc = GLES20.glGetUniformLocation(mainProgramId, "u_lumaEpsScale")
        mChromaEpsScaleLoc = GLES20.glGetUniformLocation(mainProgramId, "u_chromaEpsScale")
        mSigmaDm2Loc = GLES20.glGetUniformLocation(mainProgramId, "u_sigma_dm2")
        mInverseRange2Loc = GLES20.glGetUniformLocation(mainProgramId, "u_inverse_range2")
        mEpsScaleLoc = GLES20.glGetUniformLocation(mainProgramId, "u_epsScale")
        mUseIsoSigmaLoc = GLES20.glGetUniformLocation(mainProgramId, "u_use_iso_sigma")
        mIsoModelALoc = GLES20.glGetUniformLocation(mainProgramId, "u_iso_model_a")
        mIsoModelBLoc = GLES20.glGetUniformLocation(mainProgramId, "u_iso_model_b")
        mWinScaleLoc = GLES20.glGetUniformLocation(mainProgramId, "u_win_scale")
        mEpsBoostLoc = GLES20.glGetUniformLocation(mainProgramId, "u_eps_boost")

        ready = true
        Log.d(TAG, "Stage-5 denoise programs created: h=$statsHProgramId v=$statsVProgramId m=$mainProgramId")
        com.agx.camera.CrashLogger.log(TAG, "Program created: outNr=($statsHProgramId,$statsVProgramId,$mainProgramId)")
    }

    private fun drawQuad(programId: Int) {
        val posHandle = GLES20.glGetAttribLocation(programId, "a_position")
        val texHandle = GLES20.glGetAttribLocation(programId, "a_texCoord")
        if (posHandle >= 0) {
            GLES20.glEnableVertexAttribArray(posHandle)
            GLES20.glVertexAttribPointer(posHandle, 2, GLES20.GL_FLOAT, false, 0, quadVertices)
        }
        if (texHandle >= 0) {
            GLES20.glEnableVertexAttribArray(texHandle)
            GLES20.glVertexAttribPointer(texHandle, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)
        }
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        if (posHandle >= 0) GLES20.glDisableVertexAttribArray(posHandle)
        if (texHandle >= 0) GLES20.glDisableVertexAttribArray(texHandle)
    }

    fun drawStatsH(inTex: Int, baseTex: Int, beta: Float, yWeights: FloatArray, winScale: Float = 1f) {
        if (statsHProgramId == 0) return
        GLES20.glUseProgram(statsHProgramId)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inTex)
        GLES20.glUniform1i(hInTexLoc, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, baseTex)
        GLES20.glUniform1i(hBaseTexLoc, 1)
        GLES20.glUniform1f(hBetaLoc, beta)
        GLES20.glUniform3f(hYWeightsLoc, yWeights[0], yWeights[1], yWeights[2])
        GLES20.glUniform1f(hWinScaleLoc, winScale)
        drawQuad(statsHProgramId)
    }

    fun drawStatsV(statsTex: Int, winScale: Float = 1f) {
        if (statsVProgramId == 0) return
        GLES20.glUseProgram(statsVProgramId)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, statsTex)
        GLES20.glUniform1i(vStatsTexLoc, 0)
        GLES20.glUniform1f(vWinScaleLoc, winScale)
        drawQuad(statsVProgramId)
    }

    fun drawMain(
        inTex: Int, baseTex: Int, statsTex: Int, sigmaTex: Int,
        sigmaW: Float, sigmaH: Float, outW: Float, outH: Float,
        beta: Float, lumaEpsScale: Float, chromaEpsScale: Float, sigmaDm2: Float,
        inverseRange2: Float, epsScale: Float,
        useIsoSigma: Boolean = false,
        isoModelA: Float = 0f, isoModelB: Float = 0f,
        winScale: Float = 1f, epsBoost: Float = 1f
    ) {
        if (mainProgramId == 0) return
        GLES20.glUseProgram(mainProgramId)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inTex)
        GLES20.glUniform1i(mInTexLoc, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, baseTex)
        GLES20.glUniform1i(mBaseTexLoc, 1)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, statsTex)
        GLES20.glUniform1i(mStatsTexLoc, 2)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE3)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, sigmaTex)
        GLES20.glUniform1i(mSigmaTexLoc, 3)
        GLES20.glUniform2f(mSigmaSizeLoc, sigmaW, sigmaH)
        GLES20.glUniform2f(mOutSizeLoc, outW, outH)
        GLES20.glUniform1f(mBetaLoc, beta)
        GLES20.glUniform1f(mLumaEpsScaleLoc, lumaEpsScale)
        GLES20.glUniform1f(mChromaEpsScaleLoc, chromaEpsScale)
        GLES20.glUniform1f(mSigmaDm2Loc, sigmaDm2)
        GLES20.glUniform1f(mInverseRange2Loc, inverseRange2)
        GLES20.glUniform1f(mEpsScaleLoc, epsScale)
        GLES20.glUniform1f(mUseIsoSigmaLoc, if (useIsoSigma) 1f else 0f)
        GLES20.glUniform1f(mIsoModelALoc, isoModelA)
        GLES20.glUniform1f(mIsoModelBLoc, isoModelB)
        GLES20.glUniform1f(mWinScaleLoc, winScale)
        GLES20.glUniform1f(mEpsBoostLoc, epsBoost)
        drawQuad(mainProgramId)
    }

    fun isReady(): Boolean = ready && statsHProgramId != 0 && statsVProgramId != 0 && mainProgramId != 0

    fun destroy() {
        val ids = intArrayOf(statsHProgramId, statsVProgramId, mainProgramId)
        GLES20.glDeleteProgram(statsHProgramId)
        GLES20.glDeleteProgram(statsVProgramId)
        GLES20.glDeleteProgram(mainProgramId)
        ready = false
        statsHProgramId = 0
        statsVProgramId = 0
        mainProgramId = 0
    }

    companion object {
        private const val TAG = "OutNrShaderProgram"

        private val QUAD_COORDS = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        private val QUAD_TEX_COORDS = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f)

        /** 8 SWGF side-window centres (in output pixels), r = 2. */
        val WINDOW_CENTERS = arrayOf(
            intArrayOf(-2, 0), intArrayOf(2, 0), intArrayOf(0, -2), intArrayOf(0, 2),
            intArrayOf(-2, -2), intArrayOf(2, -2), intArrayOf(-2, 2), intArrayOf(2, 2)
        )

        private const val VERTEX_SHADER = """
#version 300 es
in vec2 a_position;
in vec2 a_texCoord;
out vec2 v_texCoord;
uniform mat4 u_transformMatrix;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = (u_transformMatrix * vec4(a_texCoord, 0.0, 1.0)).xy;
}
"""

        private const val STATS_H_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_inTex;
uniform sampler2D u_baseTex;
uniform float u_beta;
uniform vec3 u_yWeights;
// Scaling the stats box alongside the window centres keeps the 8 SWGF
// candidate windows self-similar at any resolution: preview (winScale=1)
// keeps the exact 5-tap box (R=2, bit-identical), while the capture path
// (winScale>1) widens H and V identically so the boxes tile the plane as
// densely as the preview.  Without this the capture windows drift apart and
// the min-variance/distance selection aliases into blocky patches.
uniform float u_win_scale;

float lumaOf(vec3 rgb) { return dot(u_yWeights, rgb); }

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);
    int R = max(2, int(round(2.0 * u_win_scale)));
    float sum = 0.0;
    float sumSq = 0.0;
    for (int dx = -R; dx <= R; dx++) {
        ivec2 t = base + ivec2(dx, 0);
        vec3 a = texelFetch(u_inTex, t, 0).rgb;
        vec3 b = texelFetch(u_baseTex, t, 0).rgb;
        vec3 rgb = mix(a, b, u_beta);
        float y = lumaOf(rgb);
        sum += y;
        sumSq += y * y;
    }
    outColor = vec4(sum / float(2 * R + 1), sumSq / float(2 * R + 1), 0.0, 0.0);
}
"""

        private const val STATS_V_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_statsTex;
uniform float u_win_scale;

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);
    int R = max(2, int(round(2.0 * u_win_scale)));
    float sum = 0.0;
    float sumSq = 0.0;
    for (int dy = -R; dy <= R; dy++) {
        vec2 s = texelFetch(u_statsTex, base + ivec2(0, dy), 0).rg;
        sum += s.r;
        sumSq += s.g;
    }
    outColor = vec4(sum / float(2 * R + 1), sumSq / float(2 * R + 1), 0.0, 0.0);
}
"""

        private const val MAIN_FRAGMENT = """
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_inTex;
uniform sampler2D u_baseTex;
uniform sampler2D u_statsTex;
uniform sampler2D u_sigmaTex;
uniform vec2 u_sigmaSize;
uniform vec2 u_outSize;
uniform float u_beta;
uniform float u_lumaEpsScale;
uniform float u_chromaEpsScale;
uniform float u_sigma_dm2;
uniform float u_inverse_range2;
uniform float u_epsScale;
// Resolution alignment: S5 geometry lives in output-pixel space, so at
// capture resolution (up to 3-4x the preview FBO) the fixed ±2 px windows
// cover only a 3-4x smaller image footprint than on the preview, weakening
// the low-frequency chroma collapse.  u_win_scale scales both the 8 SWGF
// window centres and the dense chroma-mean half-width so the box covers the
// same relative image region at any resolution (preview R=2 → the exact 5x5
// box).  u_eps_boost is the capture-only epsilon multiplier: the preview
// sees the S3 + 4x4 boxAA-denoised residual (σ̂²/32), but a 1:1 capture has
// no boxAA, so its per-pixel residual is ~boxAA² ≈ 16x larger; boosting ε
// restores aC/aY parity so the still collapses blotches as hard as the
// preview does.
uniform float u_win_scale;
uniform float u_eps_boost;
uniform float u_use_iso_sigma;
uniform float u_iso_model_a;
uniform float u_iso_model_b;

vec3 yccOf(vec3 rgb) {
    float y = 0.25 * rgb.r + 0.5  * rgb.g + 0.25 * rgb.b;
    float c1 = rgb.r - rgb.b;
    float c2 = 0.5 * (rgb.r + rgb.b) - rgb.g;
    return vec3(y, c1, c2);
}

vec3 rgbOf(vec3 ycc) {
    float y = ycc.x; float c1 = ycc.y; float c2 = ycc.z;
    return vec3(y + 0.5*c1 + 0.5*c2, y - 0.5*c2, y - 0.5*c1 + 0.5*c2);
}

// stats at a window centre in output-pixel coordinates
vec2 statsAt(ivec2 base, ivec2 winCentre) {
    return texelFetch(u_statsTex, base + winCentre, 0).rg;
}

// mean chroma over a DENSE box of half-width R = max(2, round(2*u_win_scale))
// (preview R=2 → 5x5) centred at base+winCentre, from the composed input.
// The earlier strided 5x5 lattice (offsets ∓2·step at u_chroma_step=3, 13x13
// support) quantized the mean into stair blocks wherever the window crossed
// the highlight rim or the demosaic's boxAA moiré; the synthetic-Bayer repro
// projects exactly those stairs as blocky seams along with amplification
// (14 big seams / RMS 0.00437 vs 6 / 0.00300 with the plain argmin).  A
// dense box varies continuously with position while still collapsing
// low-frequency chroma noise, and base-centric (non-winner) placement makes
// it independent of the per-pixel window switches: combined they cut the
// fabricated-disc artifact completely (RMS 0.00183, 0 big seams vs 0.0080/3)
// and the bayer-chain average jump to the lowest of any variant (0.00131).
// R scales with u_win_scale so the capture path keeps the same relative
// footprint without re-introducing stride quantization.
// NOTE: deliberately NOT clip-filtered at the tap level — excluding near-clip
// taps from the box starves the C-mean right around a clipped region and
// turns it per-pixel noisy (blocky water-stain patches radiating from
// highlights).  The active-clip handling lives in the demosaic's
// clip-attenuator and the S3 clipped-centre guard instead.
vec2 chromaMean(ivec2 base, ivec2 winCentre) {
    int R = min(max(2, int(round(2.0 * u_win_scale))), 8);
    vec2 row[17];
    for (int dy = -R; dy <= R; dy++) {
        ivec2 rr = base + winCentre + ivec2(0, dy);
        float h1 = 0.0;
        float h2 = 0.0;
        for (int dx = -R; dx <= R; dx++) {
            ivec2 t = rr + ivec2(dx, 0);
            vec3 a = texelFetch(u_inTex, t, 0).rgb;
            vec3 b = texelFetch(u_baseTex, t, 0).rgb;
            vec3 ycc = yccOf(mix(a, b, u_beta));
            h1 += ycc.y;
            h2 += ycc.z;
        }
        row[dy + R] = vec2(h1, h2);
    }
    vec2 s = vec2(0.0);
    for (int dy = -R; dy <= R; dy++) s += row[dy + R];
    float n = float(2 * R + 1);
    return s / (n * n);
}

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);

    vec3 a = texelFetch(u_inTex, base, 0).rgb;
    vec3 b = texelFetch(u_baseTex, base, 0).rgb;
    vec3 rgbIn = mix(a, b, u_beta);
    vec3 yccIn = yccOf(rgbIn);

    // u_sigmaTex R holds σ̂² in raw-DN² (10-bit sensor units). Stage-5 input
    // (demosaic FBO) is normalized to 0..1 by /(whiteLevel-blackLevel), so the
    // variance must be converted to the same domain: multiply BOTH σ̂² and the
    // demosaic residual variance by 1/(whiteLevel-blackLevel)² before feeding ε.
    // u_epsScale then rescales the raw-DN² noise floor to the actual S3+boxAA
    // residual that this pass sees (S5D: measured var ≈2-6e-6 image units vs
    // σ̂²≈113 DN² → ε was ~30-100× the true residual → aY≈0.02 → output collapsed
    // to the window mean, brightening + flat 5x5 blocks. Scale brings ε to the
    // real residual so aY lands in a denoising-friendly range, not a mean-sink.)
    //
    // LIVE-PREVIEW path: when the S2 σ̂ grid is gated off (Stage-1/2/3 are
    // capture-only), u_use_iso_sigma=1 substitutes the Stage-0 ISO model
    // σ̂² = a·signal + b with signal sampled at the owning output texel's luma.
    // The luma here is yccIn.x normalized to 0..1 (relative to whiteRange), so
    // signal is luma·whiteRange and whiteRange = 1/sqrt(u_inverse_range2).
    float sigma2 = 0.0;
    if (u_use_iso_sigma > 0.5) {
        float whiteRangeI = 1.0 / sqrt(max(u_inverse_range2, 1e-6));
        float signalDN = max(yccIn.x * whiteRangeI, 0.0);
        sigma2 = max(u_iso_model_a * signalDN + u_iso_model_b, 0.0);
    } else {
        ivec2 sb = ivec2(floor((vec2(base) + 0.5) * u_sigmaSize / u_outSize));
        sb = clamp(sb, ivec2(0), ivec2(u_sigmaSize) - ivec2(1));
        sigma2 = max(texelFetch(u_sigmaTex, sb, 0).r, 0.0);
    }
    // Luma keeps its texture: ε_Y uses a moderate scale (≈1.0-1.4) so aY stays
    // high and luminance is only lightly smoothed.  Chroma gets a very large
    // independent scale (≈16-64) so aC → 0: C1/C2 are pulled to the dense
    // chroma box at the pixel, collapsing the dark-region chromatic blotches.
    float epsY = u_lumaEpsScale * (sigma2 + u_sigma_dm2) * u_inverse_range2 * u_epsScale * u_eps_boost;
    float epsC = u_chromaEpsScale * (sigma2 + u_sigma_dm2) * u_inverse_range2 * u_epsScale * u_eps_boost;

    // SWGF (Yin 2019 Alg.1) window selection, softened.  A hard argmin among
    // the 8 discrete side windows maps every output pixel to exactly one mean;
    // at a sharp luma edge (clipped highlight rim) the winner map becomes an
    // 8-ray starburst — adjacent pixels whose best window differs land on
    // different window means and the seams read as blocky stain patches
    // radiating from the bright region.  Instead fuse the two closest windows
    // with weights ∝ 1/score², so the mixture is continuous across seams.  The
    // distance-based weight is unchanged in spirit: a dark pixel still keeps
    // out of bright windows (their 1/score² weight is ~0), no brighten/collapse.
    float bestScore = 1.0e30;
    float sndScore = 1.0e30;
    vec2 bestStat = vec2(0.0);
    vec2 sndStat = vec2(0.0);
    float bestVar = 0.0;
    float sndVar = 0.0;
    for (int k = 0; k < 8; k++) {
        ivec2 c = ivec2(vec2(SW_WIN[k]) * u_win_scale);
        vec2 st = statsAt(base, c);
        float v = max(st.g - st.r * st.r, 0.0);
        float score = abs(st.r - yccIn.x) / (v + epsY);
        if (score < bestScore) {
            sndScore = bestScore;
            sndVar = bestVar;
            sndStat = bestStat;
            bestScore = score;
            bestVar = v;
            bestStat = st;
        } else if (score < sndScore) {
            sndScore = score;
            sndVar = v;
            sndStat = st;
        }
    }
    float wBest = 1.0 / (bestScore * bestScore + 1e-12);
    float wSnd = 1.0 / (sndScore * sndScore + 1e-12);
    float wTot = wBest + wSnd;
    float pBest = wBest / wTot;
    float pSnd = wSnd / wTot;
    float meanY = pBest * bestStat.r + pSnd * sndStat.r;
    float varY = pBest * bestVar + pSnd * sndVar;

    float aY = varY / (varY + epsY);
    float outY = aY * yccIn.x + (1.0 - aY) * meanY;

    // Chroma is pulled to the dense chroma box at the PIXEL, not the
    // best-window centre.  Sampling at bestCentre made the collapsible chroma
    // copy the winning window's (discrete, 8-way) box mean, and those
    // per-pixel winner jumps were the blocky stain seams radiating from bright
    // clipped regions (repro: RMS 0.0080/3 big seams -> 0.0053/0 with base).
    // The box at base keeps the low-frequency collapse for the dark blotches
    // but varies continuously — no winner-dependent seams, no stride steps.
    vec2 cm = chromaMean(base, ivec2(0, 0));
    float aC = varY / (varY + epsC);
    float outC1 = aC * yccIn.y + (1.0 - aC) * cm.x;
    float outC2 = aC * yccIn.z + (1.0 - aC) * cm.y;

    outColor = vec4(rgbOf(vec3(outY, outC1, outC2)), 1.0);
}
"""

        private fun createProgram(vertexSource: String, fragmentSource: String): Int {
            // Inject the window-centre constant array into the main shader.
            val frag = if (fragmentSource == MAIN_FRAGMENT) {
                val sb = StringBuilder()
                sb.append("#version 300 es\n")
                sb.append("const ivec2 SW_WIN[8] = ivec2[8](")
                for ((i, c) in WINDOW_CENTERS.withIndex()) {
                    if (i > 0) sb.append(", ")
                    sb.append("ivec2(${c[0]}, ${c[1]})")
                }
                sb.append(");\n")
                sb.append(fragmentSource)
                sb.toString()
            } else {
                fragmentSource
            }
            val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource)
            val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, frag)
            if (vertexShader == 0 || fragmentShader == 0) return 0

            val program = GLES20.glCreateProgram()
            GLES20.glAttachShader(program, vertexShader)
            GLES20.glAttachShader(program, fragmentShader)
            GLES20.glLinkProgram(program)

            val linkStatus = IntArray(1)
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0)
            if (linkStatus[0] != GLES20.GL_TRUE) {
                val info = "Program link failed: ${GLES20.glGetProgramInfoLog(program)}"
                Log.e(TAG, info)
                com.agx.camera.CrashLogger.log(TAG, info)
                GLES20.glDeleteProgram(program)
                return 0
            }
            GLES20.glDeleteShader(vertexShader)
            GLES20.glDeleteShader(fragmentShader)
            return program
        }

        private fun loadShader(type: Int, source: String): Int {
            val shader = GLES20.glCreateShader(type)
            GLES20.glShaderSource(shader, source)
            GLES20.glCompileShader(shader)
            val compiled = IntArray(1)
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0)
            if (compiled[0] != GLES20.GL_TRUE) {
                val info = "Shader compile failed: ${GLES20.glGetShaderInfoLog(shader)}"
                Log.e(TAG, info)
                com.agx.camera.CrashLogger.log(TAG, info)
                GLES20.glDeleteShader(shader)
                return 0
            }
            return shader
        }
    }
}