package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Stage 2 - robust per-texel sigma-hat re-estimation (MAD).
 *
 * Reads the sparse Bayer grid (RGBA32F, one texel per 2x2 sensor cell, black
 * level subtracted and clamped >= 0 - the same C3 indexing contract the whole
 * pipeline uses) and estimates sigma_hat per texel with median-absolute-deviation
 * statistics on the 8 same-CFA-phase neighbours (+-1 grid texel), so
 * residual hot pixels / missed DPC defects cannot inflate the estimate;
 * the MAD over same-phase neighbours ignores such sparse outlier taps.
 *
 * Output RGBA32F:
 *   R = sigma_hat^2  (MAD-based variance, clamped to the Stage-0 ISO model floor)
 *   G = sigma_hat
 *   B = sigma_hat^2 of phase 0 (debug)
 *   A = sigma_hat^2 of phase 3 (debug)
 *
 * The ISO model floor (clamp lower bound) keeps AgX from "developing"
 * dark-margin noise: in flat dark patches MAD collapses to ~0, so we never
 * report a sigma_hat below what the Stage-0 noise model says (Stage 6 note).
 *
 * Two input domains:
 *  - rgbMode=false: the sparse Bayer grid (u_domainScale = 1, raw DN); the
 *    8-neighbour channel-wise MAD (robust to hot pixels / missed DPC).
 *  - rgbMode=true:  a demosaiced RGB texture in 0..1 pixel units
 *    (u_domainScale = whiteLevel-blackLevel) - the capture path's post-RAW
 *    re-estimation point; alpha is skipped (3 channels).  RGB content uses the
 *    direction-aware sigma_hat^2 (min of the 4 axis pair-differences) instead of the
 *    MAD: printed/demic pattern straddling +-1 neighbours inflates the MAD to
 *    sigma_hat^2~10-20k DN^2 at ink edges, which the S5 contract would read as noise and
 *    pit the letters (close-inspection residual).  Flats stay floor-anchored.
 */
class SigmaHatShaderProgram {

    private var programId = 0

    private var uSparseTexLoc = 0
    private var uTransformMatrixLoc = 0
    private var uSensorSizeLoc = 0
    private var uCropOriginLoc = 0
    private var uCropSizeLoc = 0
    private var uViewSizeLoc = 0
    private var uBlackLevelPatternLoc = 0
    private var uIsoModelA = 0
    private var uIsoModelB = 0
    private var uDomainScaleLoc = 0
    private var uRgbModeLoc = 0

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create() {
        programId = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        if (programId == 0) {
            Log.e(TAG, "Failed to create sigma-hat shader program")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create sigma-hat shader program")
            return
        }
        uSparseTexLoc = GLES20.glGetUniformLocation(programId, "u_sparseTex")
        uTransformMatrixLoc = GLES20.glGetUniformLocation(programId, "u_transformMatrix")
        uSensorSizeLoc = GLES20.glGetUniformLocation(programId, "u_sensorSize")
        uCropOriginLoc = GLES20.glGetUniformLocation(programId, "u_cropOrigin")
        uCropSizeLoc = GLES20.glGetUniformLocation(programId, "u_cropSize")
        uViewSizeLoc = GLES20.glGetUniformLocation(programId, "u_viewSize")
        uBlackLevelPatternLoc = GLES20.glGetUniformLocation(programId, "u_black_level_pattern")
        uIsoModelA = GLES20.glGetUniformLocation(programId, "u_iso_model_a")
        uIsoModelB = GLES20.glGetUniformLocation(programId, "u_iso_model_b")
        uDomainScaleLoc = GLES20.glGetUniformLocation(programId, "u_domain_scale")
        uRgbModeLoc = GLES20.glGetUniformLocation(programId, "u_rgb_mode")
        Log.d(TAG, "Sigma-hat shader program created: $programId")
        com.agx.camera.CrashLogger.log(TAG, "Program created: sigmaHat=$programId")
    }

    fun draw(
        transformMatrix: FloatArray,
        cropOriginX: Float, cropOriginY: Float,
        cropSizeX: Float, cropSizeY: Float,
        viewWidth: Float, viewHeight: Float,
        sensorWidth: Float, sensorHeight: Float,
        sparseTex: Int,
        blackLevelPattern: IntArray,
        isoModelA: Float, isoModelB: Float,
        domainScale: Float = 1f,
        rgbMode: Boolean = false
    ) {
        if (programId == 0) return
        GLES20.glUseProgram(programId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, sparseTex)
        GLES20.glUniform1i(uSparseTexLoc, 0)

        GLES20.glUniformMatrix4fv(uTransformMatrixLoc, 1, false, transformMatrix, 0)
        GLES20.glUniform2f(uSensorSizeLoc, sensorWidth, sensorHeight)
        GLES20.glUniform2f(uCropOriginLoc, cropOriginX, cropOriginY)
        GLES20.glUniform2f(uCropSizeLoc, cropSizeX, cropSizeY)
        GLES20.glUniform2f(uViewSizeLoc, viewWidth, viewHeight)
        GLES20.glUniform4i(uBlackLevelPatternLoc,
            blackLevelPattern[0], blackLevelPattern[1],
            blackLevelPattern[2], blackLevelPattern[3])
        GLES20.glUniform1f(uIsoModelA, isoModelA)
        GLES20.glUniform1f(uIsoModelB, isoModelB)
        GLES20.glUniform1f(uDomainScaleLoc, domainScale)
        GLES20.glUniform1f(uRgbModeLoc, if (rgbMode) 1f else 0f)

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

    fun isReady(): Boolean = programId != 0

    fun destroy() {
        if (programId != 0) {
            GLES20.glDeleteProgram(programId)
            programId = 0
        }
    }

    companion object {
        private const val TAG = "SigmaHatShaderProgram"

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

        private const val FRAGMENT_SHADER = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_sparseTex;
uniform vec2 u_sensorSize;
uniform vec2 u_cropOrigin;
uniform vec2 u_cropSize;
uniform vec2 u_viewSize;
uniform ivec4 u_black_level_pattern;
uniform float u_iso_model_a;
uniform float u_iso_model_b;
// Domain of the input texels: u_domainScale = 1 for the sparse Bayer grid
// (already black-subtracted raw DN); = whiteLevel-blackLevel for a demosaiced
// RGBA input in 0..1 pixel units, so the MAD variance is scaled back to raw
// DN^2 (x scale^2) and the ISO-model floor is evaluated at DN signal (x scale).
uniform float u_domain_scale;
// 0 = sparse grid (4 CFA phases incl. alpha), 1 = RGB demosaic (3 channels).
uniform float u_rgb_mode;

${DenoiseGlsl.MOSAIC_HELPERS}

// 8 same-CFA-phase neighbours in grid units (+-1 output texel).
const int NOX[8] = int[8]( 1, -1,  0,  0,  1, -1,  1, -1);
const int NOY[8] = int[8]( 0,  0,  1, -1,  1,  1, -1, -1);

float channelOf(vec4 c, int p) {
    if (p == 0) return c.r;
    if (p == 1) return c.g;
    if (p == 2) return c.b;
    return c.a;
}

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);
    float sig2Total = 0.0;
    vec4 sig2ByPhase = vec4(0.0);
    int phaseCount = 4;
    if (u_rgb_mode > 0.5) phaseCount = 3;

    for (int p = 0; p < 4; p++) {
        if (p >= phaseCount) break;
        float vals[8];
        for (int k = 0; k < 8; k++) {
            ivec2 t = clamp(base + ivec2(NOX[k], NOY[k]), ivec2(0), ivec2(u_viewSize) - ivec2(1));
            vals[k] = channelOf(texelFetch(u_sparseTex, t, 0), p) * u_domain_scale;
        }
        // sigma_hat^2.  Preview (sparse Bayer grid) -> median-absolute-deviation over the
        // 8 same-CFA-phase neighbours (robust to hot pixels / missed DPC);
        // capture (demosaic RGB) -> a printed/demic pattern (backlit sign text,
        // halftone) straddles the +-1 neighbours and inflates the MAD to
        // sigma_hat^2~10-20k DN^2, which the S5 epsilon contract reads as NOISE -> aY drops ->
        // letter pixels drift to the window mean (residual pitting on close
        // inspection, device-verified 6 pit pixels, all at ink edges with
        // sigma_hat^2 10-22k vs 1.4k interior).  For RGB content take the MINIMUM
        // squared pair-difference over the 4 axes (E/W, N/S, diag1, diag2):
        // along a stroke edge the along-edge axis stays at the REGION's own
        // noise, so ink-edge sigma_hat^2 returns to the design scale and aY stays high.
        // The 2-sample axis variance is unbiased for pure noise; the min-of-4
        // selection bias only lowers sigma_hat^2 below the ISO-model floor, which the
        // max() clamp re-anchors - flats are floor-dominated, so
        // noise calibration is unchanged.  (textured letters): 641->35 pits,
        // edge sigma_hat^2 20.4k->57.
        float sig2;
        if (u_rgb_mode > 0.5) {
            float a0 = vals[0] - vals[1];
            float a1 = vals[2] - vals[3];
            float a2 = vals[4] - vals[6];
            float a3 = vals[5] - vals[7];
            float m = min(min(a0 * a0, a1 * a1), min(a2 * a2, a3 * a3));
            sig2 = 2.1981 * 0.5 * m;
        } else {
            // ascending sort (8 elements, insertion)
            for (int i = 1; i < 8; i++) {
                float v = vals[i];
                int j = i - 1;
                while (j >= 0 && vals[j] > v) {
                    vals[j + 1] = vals[j];
                    j--;
                }
                vals[j + 1] = v;
            }
            float med = (vals[3] + vals[4]) * 0.5;

            float devs[8];
            for (int k = 0; k < 8; k++) devs[k] = abs(vals[k] - med);
            for (int i = 1; i < 8; i++) {
                float v = devs[i];
                int j = i - 1;
                while (j >= 0 && devs[j] > v) {
                    devs[j + 1] = devs[j];
                    j--;
                }
                devs[j + 1] = v;
            }
            float mad = (devs[3] + devs[4]) * 0.5;
            // sigma_hat^2 from MAD^2. vals[] are already scaled into the sigma_hat domain by
            // u_domain_scale above (preview: domainScale=1 -> 0..1 image units;
            // capture rgbMode: domainScale=whiteRange -> raw-DN).  Squaring the
            // scale AGAIN here over-inflated the capture texture to sigma_hat^2*WR^2: the
            // S5 main divides by 1/WR^2 once, leaving capture epsilon ~ 0.7*sigma_hat^2 (~WR^2 x
            // the designed 0.7*sigma_hat^2/WR^2) -> aY~0 -> thin bright strokes collapse to
            // the window mean (backlit-letter pitting/halo, device-verified).
            // Keep a single power so the texture honours the S5 contract: sigma_hat^2 in
            // raw-DN^2 on capture, image-DN^2/unit^2 on preview (x1 unchanged).
            sig2 = 2.1981 * mad * mad;
        }
        if (p == 0) sig2ByPhase.x = sig2;
        else if (p == 1) sig2ByPhase.y = sig2;
        else if (p == 2) sig2ByPhase.z = sig2;
        else sig2ByPhase.w = sig2;
        sig2Total += sig2;
    }

    float sig2Mean = sig2Total / float(phaseCount);

    vec4 center = texelFetch(u_sparseTex, base, 0);
    float meanSignal = 0.0;
    if (u_rgb_mode > 0.5) {
        meanSignal = (0.25 * center.r + 0.5 * center.g + 0.25 * center.b) * u_domain_scale;
    } else {
        meanSignal = (center.r + center.g + center.b + center.a) * 0.25 * u_domain_scale;
    }
    float floor2 = isoModelSigmaSq(meanSignal);

    float out2 = max(sig2Mean, floor2);
    outColor = vec4(out2, sqrt(max(out2, 1.0e-6)), sig2ByPhase.x, sig2ByPhase.w);
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