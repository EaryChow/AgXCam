package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Stage 2 — robust per-texel sigma-hat re-estimation (MAD).
 *
 * Reads the sparse Bayer grid (RGBA32F, one texel per 2x2 sensor cell, black
 * level subtracted and clamped >= 0 — the same C3 indexing contract the whole
 * pipeline uses) and estimates σ̂ per texel with median-absolute-deviation
 * statistics on the 8 same-CFA-phase neighbours (±1 grid texel), so
 * residual hot pixels / missed DPC defects cannot inflate the estimate
 * (plan §3 Stage 2, DR-8).
 *
 * Output RGBA32F:
 *   R = σ̂²  (MAD-based variance, clamped to the Stage-0 ISO model floor)
 *   G = σ̂
 *   B = σ̂² of phase 0 (debug)
 *   A = σ̂² of phase 3 (debug)
 *
 * The ISO model floor (clamp lower bound) keeps AgX from "developing"
 * dark-margin noise: in flat dark patches MAD collapses to ~0, so we never
 * report a σ̂ below what the Stage-0 noise model says (DR-4 / Stage 6 note).
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
        isoModelA: Float, isoModelB: Float
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

${DenoiseGlsl.MOSAIC_HELPERS}

// 8 same-CFA-phase neighbours in grid units (±1 output texel).
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

    for (int p = 0; p < 4; p++) {
        float vals[8];
        for (int k = 0; k < 8; k++) {
            ivec2 t = clamp(base + ivec2(NOX[k], NOY[k]), ivec2(0), ivec2(u_viewSize) - ivec2(1));
            vals[k] = channelOf(texelFetch(u_sparseTex, t, 0), p);
        }
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
        float sig2 = 2.1981 * mad * mad; // (1.4826^2)
        if (p == 0) sig2ByPhase.x = sig2;
        else if (p == 1) sig2ByPhase.y = sig2;
        else if (p == 2) sig2ByPhase.z = sig2;
        else sig2ByPhase.w = sig2;
        sig2Total += sig2;
    }

    float sig2Mean = sig2Total * 0.25;

    vec4 center = texelFetch(u_sparseTex, base, 0);
    float meanSignal = (center.r + center.g + center.b + center.a) * 0.25;
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