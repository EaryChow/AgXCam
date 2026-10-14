package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Stage 5-C - post-demosaic chroma denoise, split from the luma branch.
 *
 * S5-L (OutNrShaderProgram) filters luma and passes chroma through bit-exactly.
 * This program consumes that result: it reads the S5-L luma as its guide and
 * the demosaic RGB as its chroma source, then filters the two chroma
 * coordinates with a guided soft-weight weighted-mean stage followed by a
 * per-cell chroma mottle stage. The luma coordinate is never written, so the
 * two branches cannot fight over it.
 *
 * Transform T (used throughout, and its own inverse):
 *   Y  = 0.25 R + 0.5 G + 0.25 B
 *   C1 = (B - R) / 4
 *   C2 = (R - 2 G + B) / 4
 *   R = Y - 2 C1 + C2,  G = Y - C2,  B = Y + 2 C1 + C2
 *
 * Passes, each one texel per output pixel except CELL and the sigma source:
 *   sigma  : per-channel axis-min sigma_hat^2 in image units, clamped to the
 *            Stage-0 ISO model floor, then propagated FULLY (T Sigma T^t with
 *            the measured cross-correlation tables). Output
 *            (sigY2, sigC1_2, sigC2_2).
 *   main   : the guided chroma stage. Soft-weight weighted mean over the dense
 *            (2*r_support+1)^2 support window with a luma pre-cull (a tap whose
 *            luma-only weight cannot reach the cull threshold skips the chroma
 *            fetch and the chroma distance), then pulls the pixel's chroma
 *            toward it by a guide variance D^2 ratio. Output is the full RGB
 *            image rebuilt with the guide luma.
 *   cell   : one texel per 8x8 cell of the stage-1 image. Reports that cell's
 *            vector-median chroma and the mottle blend weight per chroma
 *            coordinate; a variance guard disables a cell that sits on a
 *            gradient.
 *   blend  : one texel per output pixel. Blends the stage-1 chroma toward its
 *            owning cell's median by the cell weights, and rebuilds RGB with
 *            the guide luma. Pixels outside the cell grid pass through.
 *
 * The host drives all four in order (PreviewRenderer.runStage5C); a strength of
 * 0 makes lambda zero at every stage, so the chain is a bit-exact identity and
 * the caller skips it entirely.
 */
class S5CShaderProgram {

    private var sigmaProgramId = 0
    private var mainProgramId = 0
    private var cellProgramId = 0
    private var blendProgramId = 0
    private var ready = false

    /**
     * The filter program only. The sigma pass is charged elsewhere and the cell
     * pass runs at 1/64 the texel count, so the budget counts this pass.
     */
    fun budgetProgramId(): Int = mainProgramId

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create() {
        sigmaProgramId = createProgram(VERTEX_SHADER, SIGMA_FRAGMENT)
        mainProgramId = createProgram(VERTEX_SHADER, MAIN_FRAGMENT)
        cellProgramId = createProgram(VERTEX_SHADER, CELL_FRAGMENT)
        blendProgramId = createProgram(VERTEX_SHADER, BLEND_FRAGMENT)
        if (sigmaProgramId == 0 || mainProgramId == 0 ||
            cellProgramId == 0 || blendProgramId == 0
        ) {
            Log.e(TAG, "Failed to create Stage-5C chroma programs")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create Stage-5C chroma programs")
            return
        }
        ready = true
        Log.d(TAG, "Stage-5C programs created: s=$sigmaProgramId m=$mainProgramId c=$cellProgramId b=$blendProgramId")
        com.agx.camera.CrashLogger.log(
            TAG, "Program created: s5c=($sigmaProgramId,$mainProgramId,$cellProgramId,$blendProgramId)"
        )
    }

    fun isReady(): Boolean = ready && mainProgramId != 0

    fun destroy() {
        GLES20.glDeleteProgram(sigmaProgramId)
        GLES20.glDeleteProgram(mainProgramId)
        GLES20.glDeleteProgram(cellProgramId)
        GLES20.glDeleteProgram(blendProgramId)
        ready = false
        sigmaProgramId = 0
        mainProgramId = 0
        cellProgramId = 0
        blendProgramId = 0
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

    private fun setF(programId: Int, name: String, value: Float) {
        GLES20.glUniform1f(GLES20.glGetUniformLocation(programId, name), value)
    }

    private fun set2F(programId: Int, name: String, x: Float, y: Float) {
        GLES20.glUniform2f(GLES20.glGetUniformLocation(programId, name), x, y)
    }

    private fun setTex(programId: Int, name: String, unit: Int, tex: Int) {
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0 + unit)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(programId, name), unit)
    }

    /**
     * Sigma pass. Reads the demosaic RGB directly and writes per-channel
     * axis-min sigma_hat^2 (image units) propagated through T into
     * (sigY2, sigC1_2, sigC2_2). The ISO floor is evaluated per channel from
     * the centre luma, matching the S5-C reference.
     */
    fun drawSigma(
        inTex: Int, viewW: Float, viewH: Float,
        isoModelA: Float, isoModelB: Float, inverseRange2: Float
    ) {
        if (sigmaProgramId == 0) return
        GLES20.glUseProgram(sigmaProgramId)
        setTex(sigmaProgramId, "u_inTex", 0, inTex)
        set2F(sigmaProgramId, "u_viewSize", viewW, viewH)
        setF(sigmaProgramId, "u_iso_model_a", isoModelA)
        setF(sigmaProgramId, "u_iso_model_b", isoModelB)
        setF(sigmaProgramId, "u_inverse_range2", inverseRange2)
        drawQuad(sigmaProgramId)
    }

    /**
     * Stage-1 chroma filter. [guideTex] is the S5-L output (or the demosaic
     * input itself when S5-L is off); its luma is the guide and is what the
     * rebuilt RGB carries. [rSupport] is the dense support-window radius and
     * [rGuide] the dense guide-window radius, both from the
     * window-geometry resolution law.
     */
    fun drawStage1(
        inTex: Int, guideTex: Int, sigmaTex: Int, strength: Float,
        s: Float, sW: Float, gC: Float, gLum: Float, rRetain: Float, mCx: Float, mCy: Float,
        rSupport: Float, rGuide: Float
    ) {
        if (mainProgramId == 0) return
        GLES20.glUseProgram(mainProgramId)
        setTex(mainProgramId, "u_inTex", 0, inTex)
        setTex(mainProgramId, "u_guideTex", 1, guideTex)
        setTex(mainProgramId, "u_sigmaTex", 2, sigmaTex)
        setF(mainProgramId, "u_strength", strength)
        setF(mainProgramId, "u_s", s)
        setF(mainProgramId, "u_s_w", sW)
        setF(mainProgramId, "u_g_c", gC)
        setF(mainProgramId, "u_g_lum", gLum)
        setF(mainProgramId, "u_r_retain", rRetain)
        set2F(mainProgramId, "u_m_c", mCx, mCy)
        setF(mainProgramId, "u_r_support", rSupport)
        setF(mainProgramId, "u_r_guide", rGuide)
        drawQuad(mainProgramId)
    }

    /** Cell mottle pass, one texel per 8x8 cell of the stage-1 image. */
    fun drawCell(stage1Tex: Int, sigmaTex: Int, strength: Float, sM: Float, kGrad: Float) {
        if (cellProgramId == 0) return
        GLES20.glUseProgram(cellProgramId)
        setTex(cellProgramId, "u_stage1Tex", 0, stage1Tex)
        setTex(cellProgramId, "u_sigmaTex", 1, sigmaTex)
        setF(cellProgramId, "u_strength", strength)
        setF(cellProgramId, "u_s_m", sM)
        setF(cellProgramId, "u_k_grad", kGrad)
        drawQuad(cellProgramId)
    }

    /** Final chroma blend against the owning cell, one texel per output pixel. */
    fun drawBlend(stage1Tex: Int, cellTex: Int) {
        if (blendProgramId == 0) return
        GLES20.glUseProgram(blendProgramId)
        setTex(blendProgramId, "u_stage1Tex", 0, stage1Tex)
        setTex(blendProgramId, "u_cellTex", 1, cellTex)
        drawQuad(blendProgramId)
    }

    companion object {
        private const val TAG = "S5CShaderProgram"

        private val QUAD_COORDS = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        private val QUAD_TEX_COORDS = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f)

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

        // 8 neighbours for the axis-min estimator, shared with the sigma
        // programs so the split measures the same estimator S5 consumes.
        private const val SIGMA_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_inTex;
uniform vec2 u_viewSize;
uniform float u_iso_model_a;
uniform float u_iso_model_b;
uniform float u_inverse_range2;

const int NOX[8] = int[8]( 1, -1,  0,  0,  1, -1,  1, -1);
const int NOY[8] = int[8]( 0,  0,  1, -1,  1,  1, -1, -1);

float channelOf(vec4 c, int p) {
    if (p == 0) return c.r;
    if (p == 1) return c.g;
    return c.b;
}

// Minimum squared 2-sample axis difference over the four neighbour axes, times
// the 2-sample unbiased factor 0.5 times 2.1981 (the MAD-consistency constant).
float axisMinSig2(ivec2 base, int p) {
    float vals[8];
    for (int k = 0; k < 8; k++) {
        ivec2 t = clamp(base + ivec2(NOX[k], NOY[k]), ivec2(0), ivec2(u_viewSize) - ivec2(1));
        vals[k] = channelOf(texelFetch(u_inTex, t, 0), p);
    }
    float a0 = vals[0] - vals[1];
    float a1 = vals[2] - vals[3];
    float a2 = vals[4] - vals[6];
    float a3 = vals[5] - vals[7];
    float m = min(min(a0 * a0, a1 * a1), min(a2 * a2, a3 * a3));
    return 2.1981 * 0.5 * m;
}

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);
    vec4 center = texelFetch(u_inTex, base, 0);

    // ISO-model floor in image units: sigma^2 = a*signalDN + b evaluated at the
    // centre luma. signalDN = luma / sqrt(inverseRange2) recovers the raw-DN
    // signal the model is anchored on, then the result is divided back to image
    // units so every quantity below shares one domain.
    float luma = 0.25 * center.r + 0.5 * center.g + 0.25 * center.b;
    float whiteRangeI = 1.0 / sqrt(max(u_inverse_range2, 1.0e-12));
    float signalDN = max(luma * whiteRangeI, 0.0);
    float floor2 = max(u_iso_model_a * signalDN + u_iso_model_b, 0.0) * u_inverse_range2;

    float sigR2 = max(axisMinSig2(base, 0), floor2);
    float sigG2 = max(axisMinSig2(base, 1), floor2);
    float sigB2 = max(axisMinSig2(base, 2), floor2);

    // Full T Sigma T^t propagation of the per-channel variances through the
    // luma/chroma transform. The cross terms use the constants below: after the
    // order move S5-C runs on the Bradford-adapted signal, so the S5-input
    // residual cross-correlation is constant across signal level and WB state.
    // The old luma-indexed linear-sRGB tables were a matrix-rotation artifact
    // and no longer apply; these constants are the Bradford rotation of the
    // per-phase covariance. sigma_Y^2 and sigma_C2^2 differ by the cross terms,
    // and both consumers need the true propagated luma variance: the d^2 guide
    // term (sigma_Y,retain) and the stage-2 guide guard. rho_RG +0.598,
    // rho_RB +0.221, rho_GB -0.122 are the camera-native values rounded to
    // 3 dp; the lowest-signal anchor (signal <= 0.05) reads +0.220 / -0.126 and
    // is dropped because the difference is O(0.004) and only at the lowest
    // signal levels. The [0,1,0] clip neutralization now runs after S5, so it
    // no longer enters this propagation.
    float pRg =  0.598;
    float pRb =  0.221;
    float pGb = -0.122;
    float sR = sqrt(max(sigR2, 0.0));
    float sG = sqrt(max(sigG2, 0.0));
    float sB = sqrt(max(sigB2, 0.0));
    float covRg = pRg * sR * sG;
    float covRb = pRb * sR * sB;
    float covGb = pGb * sG * sB;
    float sigY2 = (sigR2 + 4.0 * sigG2 + sigB2 + 4.0 * covRg + 2.0 * covRb + 4.0 * covGb) * (1.0 / 16.0);
    float sigC1_2 = (sigR2 + sigB2 - 2.0 * covRb) * (1.0 / 16.0);
    float sigC2_2 = (sigR2 + 4.0 * sigG2 + sigB2 - 4.0 * covRg + 2.0 * covRb - 4.0 * covGb) * (1.0 / 16.0);

    outColor = vec4(sigY2, sigC1_2, sigC2_2, 0.0);
}
"""

        private const val MAIN_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_inTex;
uniform sampler2D u_guideTex;
uniform sampler2D u_sigmaTex;
uniform float u_strength;
uniform float u_s;
uniform float u_s_w;
uniform float u_g_c;
uniform float u_g_lum;
uniform float u_r_retain;
uniform vec2 u_m_c;
// Window-geometry resolution law: the support is a dense
// (2*r_support+1)^2 window and the D^2 guide a dense (2*r_guide+1)^2 window,
// both keyed on winScale = width / 640 (the same idiom as the S5-L radius
// law). The 9x9/5x5 origin (winScale 1.5) and the 35x35/19x19 1:1
// (winScale 6.4) are certified.
uniform float u_r_support;
uniform float u_r_guide;

float luma(vec3 rgb) { return 0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b; }
vec2 chroma(vec3 rgb) { return vec2((rgb.b - rgb.r) * 0.25, (rgb.r - 2.0 * rgb.g + rgb.b) * 0.25); }
vec3 rgbOf(float y, vec2 c) { return vec3(y - 2.0 * c.x + c.y, y - c.y, y + 2.0 * c.x + c.y); }

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);
    int rs = max(1, int(u_r_support));
    ivec2 sz = textureSize(u_inTex, 0);
    ivec2 maxT = sz - ivec2(1);

    vec4 sig = texelFetch(u_sigmaTex, base, 0);
    float sigY2 = max(sig.r, 1.0e-12);
    float sigC1_2 = max(sig.g, 1.0e-12);
    float sigC2_2 = max(sig.b, 1.0e-12);
    float sy2 = max(u_r_retain * u_r_retain * sigY2, 1.0e-12);

    float yhat = luma(texelFetch(u_guideTex, base, 0).rgb);
    vec2 cin = chroma(texelFetch(u_inTex, base, 0).rgb) - u_m_c;

    // Stage-1 estimate and guide variance accumulate in one sweep over the
    // dense (2*rs+1)^2 support window: the (2*rg+1)^2 guide window sits inside
    // it at every resolution (the guide radius law has half the support
    // slope), so the guide reuses the same taps instead of a second pass. A
    // tap whose luma-only weight cannot reach CULL_W carries a full weight
    // below CULL_W (the chroma terms only ever shrink the weight); outside
    // the guide window its chroma fetch is skipped entirely, inside the guide
    // window the cull only skips the chat accumulation, because the guide
    // weights are unculled. Chroma-disparate taps reach chat only through the
    // soft weight, which is what kills the unweighted box mean's chroma halo;
    // the support floor stays omitted because the weighted mean is defined at
    // every pixel. The edge clamp reproduces the reference edge-replicated
    // padding.
    const float CULL_W = 0.05;
    int rg = max(1, int(u_r_guide));
    vec2 sumWc = vec2(0.0);
    float sumW = 0.0;
    float acc = 0.0;
    float wsum = 0.0;
    for (int dy = -rs; dy <= rs; dy++) {
        for (int dx = -rs; dx <= rs; dx++) {
            ivec2 t = clamp(base + ivec2(dx, dy), ivec2(0), maxT);
            float yv = luma(texelFetch(u_guideTex, t, 0).rgb);
            float dyv = yv - yhat;
            float lterm = dyv * dyv / sy2;
            float wL = u_s_w / (lterm + u_s_w);
            bool inGuide = abs(dy) <= rg && abs(dx) <= rg;
            if (wL < CULL_W && !inGuide) continue;
            vec2 cv = chroma(texelFetch(u_inTex, t, 0).rgb) - u_m_c;
            float dc1 = cv.x - cin.x;
            float dc2 = cv.y - cin.y;
            float d2 = lterm + dc1 * dc1 / sigC1_2 + dc2 * dc2 / sigC2_2;
            float w = u_s_w / (d2 + u_s_w);
            if (wL >= CULL_W) {
                sumWc += w * cv;
                sumW += w;
            }
            if (inGuide) {
                acc += w * dyv * dyv;
                wsum += w;
            }
        }
    }
    vec2 chat = sumWc / max(sumW, 1.0e-12);
    float d2guide = acc / max(wsum, 1.0e-12);

    float g = u_s * u_g_c * u_strength * u_g_lum;
    float lam1 = g * sigC1_2 / (d2guide + g * sigC1_2);
    float lam2 = g * sigC2_2 / (d2guide + g * sigC2_2);
    vec2 cOut = vec2(
        (1.0 - lam1) * cin.x + lam1 * chat.x,
        (1.0 - lam2) * cin.y + lam2 * chat.y
    );
    outColor = vec4(rgbOf(yhat, cOut), 1.0);
}
"""

        private const val CELL_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_stage1Tex;
uniform sampler2D u_sigmaTex;
uniform float u_strength;
uniform float u_s_m;
uniform float u_k_grad;

const int CELL = 8;
const int N = CELL * CELL;
const int ITERS = 8;

float luma(vec3 rgb) { return 0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b; }
vec2 chroma(vec3 rgb) { return vec2((rgb.b - rgb.r) * 0.25, (rgb.r - 2.0 * rgb.g + rgb.b) * 0.25); }

void main() {
    ivec2 cell = ivec2(gl_FragCoord.xy);
    ivec2 base = cell * CELL;
    ivec2 size = textureSize(u_stage1Tex, 0);

    vec2 C[N];
    float ysum = 0.0;
    float ysum2 = 0.0;
    float ssum = 0.0;
    float sc1 = 0.0;
    float sc2 = 0.0;
    int idx = 0;
    for (int yy = 0; yy < CELL; yy++) {
        for (int xx = 0; xx < CELL; xx++) {
            ivec2 t = clamp(base + ivec2(xx, yy), ivec2(0), size - ivec2(1));
            vec3 rgb = texelFetch(u_stage1Tex, t, 0).rgb;
            float y = luma(rgb);
            ysum += y;
            ysum2 += y * y;
            vec4 sg = texelFetch(u_sigmaTex, t, 0);
            ssum += sg.r;
            sc1 += sg.g;
            sc2 += sg.b;
            C[idx] = chroma(rgb);
            idx++;
        }
    }
    float n = float(N);
    float meanY = ysum / n;
    float yvar = max(ysum2 / n - meanY * meanY, 0.0);
    float meanSigY = ssum / n;
    float meanSc1 = sc1 / n;
    float meanSc2 = sc2 / n;

    // A cell whose luma variance exceeds the noise level carries structure, so
    // its chroma is left alone. This is the one gate that keeps the mottle
    // stage off coloured edges.
    bool disabled = yvar > u_k_grad * meanSigY;

    // The certified chain tiles only the floor(h/8) x floor(w/8) interior; the
    // right/bottom border keeps the stage-1 result. A partial cell has no full
    // 8x8 sample set, so leave it alone (lambda 0).
    bool partial = (base.x + CELL > size.x) || (base.y + CELL > size.y);

    // Uniform-weight Weiszfeld over the cell's chroma samples.
    vec2 c = vec2(0.0);
    for (int k = 0; k < N; k++) c += C[k];
    c /= n;
    for (int it = 0; it < ITERS; it++) {
        vec2 num = vec2(0.0);
        float den = 0.0;
        for (int k = 0; k < N; k++) {
            float dist = max(length(C[k] - c), 1.0e-12);
            num += C[k] / dist;
            den += 1.0 / dist;
        }
        c = num / max(den, 1.0e-12);
    }

    float g = u_s_m * u_strength;
    float lam1 = 0.0;
    float lam2 = 0.0;
    if (!disabled && !partial) {
        lam1 = g * meanSc1 / (yvar + g * meanSc1);
        lam2 = g * meanSc2 / (yvar + g * meanSc2);
    }
    outColor = vec4(c.x, c.y, lam1, lam2);
}
"""

        private const val BLEND_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_stage1Tex;
uniform sampler2D u_cellTex;

const int CELL = 8;

float luma(vec3 rgb) { return 0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b; }
vec2 chroma(vec3 rgb) { return vec2((rgb.b - rgb.r) * 0.25, (rgb.r - 2.0 * rgb.g + rgb.b) * 0.25); }
vec3 rgbOf(float y, vec2 c) { return vec3(y - 2.0 * c.x + c.y, y - c.y, y + 2.0 * c.x + c.y); }

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 rgb = texelFetch(u_stage1Tex, p, 0).rgb;
    ivec2 csize = textureSize(u_cellTex, 0);

    // Partial cells past the right/bottom edge have no median to borrow.
    if (p.x >= csize.x * CELL || p.y >= csize.y * CELL) {
        outColor = vec4(rgb, 1.0);
        return;
    }

    vec4 cellv = texelFetch(u_cellTex, p / CELL, 0);
    float y = luma(rgb);
    vec2 cc = chroma(rgb);
    vec2 outC = vec2(
        (1.0 - cellv.z) * cc.x + cellv.z * cellv.x,
        (1.0 - cellv.w) * cc.y + cellv.w * cellv.y
    );
    outColor = vec4(rgbOf(y, outC), 1.0);
}
"""

        private fun createProgram(vertexSource: String, fragmentSource: String): Int {
            val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource)
            val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource)
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
