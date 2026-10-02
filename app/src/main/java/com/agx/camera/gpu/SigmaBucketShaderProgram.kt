package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Axis-min bucket split (diagnostic only).
 *
 * The second sigma_hat re-estimate point currently averages the three
 * axis-min variances into one number, which hides the per-channel spread.
 * This pass writes the split so the spread can be measured:
 *
 *   R = sigma_hat^2 for the R channel's axis-min
 *   G = sigma_hat^2 for the G channel's axis-min
 *   B = sigma_hat^2 for the B channel's axis-min
 *   A = sigma_hat^2 of the 3-channel mean (what S5 consumes today)
 *
 * It renders into its OWN FBO and nothing downstream reads it: S5 keeps
 * consuming the untouched SigmaHatShaderProgram output, so the S5 result is
 * bit-identical whether this pass ran or not. That is the whole point of the
 * measurement-first discipline - the split is observed before it is allowed
 * to be consumed.
 */
class SigmaBucketShaderProgram {

    private var programId = 0

    /** Exposed only for the shader budget table in the measurement report. */
    fun budgetProgramId(): Int = programId

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

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create() {
        programId = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        if (programId == 0) {
            Log.e(TAG, "Failed to create sigma bucket shader program")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create sigma bucket shader program")
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
        Log.d(TAG, "Sigma bucket shader program created: $programId")
        com.agx.camera.CrashLogger.log(TAG, "Program created: sigmaBucket=$programId")
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
        domainScale: Float
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
        private const val TAG = "SigmaBucketShaderProgram"

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

        // Same estimator as SigmaHatShaderProgram's rgbMode branch: the
        // minimum squared 2-sample axis difference over the four neighbour
        // axes, times the 2-sample unbiased factor 0.5 times 2.1981 (the
        // MAD-consistency constant). Kept in lockstep with that shader on
        // purpose: if the two ever diverge the split stops measuring the same
        // estimator S5 consumes, and the bucket report would be a fiction.
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
uniform float u_domain_scale;

${DenoiseGlsl.MOSAIC_HELPERS}

const int NOX[8] = int[8]( 1, -1,  0,  0,  1, -1,  1, -1);
const int NOY[8] = int[8]( 0,  0,  1, -1,  1,  1, -1, -1);

float channelOf(vec4 c, int p) {
    if (p == 0) return c.r;
    if (p == 1) return c.g;
    if (p == 2) return c.b;
    return c.a;
}

float axisMinSig2(ivec2 base, int p) {
    float vals[8];
    for (int k = 0; k < 8; k++) {
        ivec2 t = clamp(base + ivec2(NOX[k], NOY[k]), ivec2(0), ivec2(u_viewSize) - ivec2(1));
        vals[k] = channelOf(texelFetch(u_sparseTex, t, 0), p) * u_domain_scale;
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
    float sig2R = axisMinSig2(base, 0);
    float sig2G = axisMinSig2(base, 1);
    float sig2B = axisMinSig2(base, 2);

    vec4 center = texelFetch(u_sparseTex, base, 0);
    float meanSignal = (0.25 * center.r + 0.5 * center.g + 0.25 * center.b) * u_domain_scale;
    float floor2 = isoModelSigmaSq(meanSignal);

    // Same order S5 uses: floor the channel mean, not the channels. Flooring
    // each channel first would lift the mean and column A would stop matching
    // the value S5 reads.
    float meanFloored = max((sig2R + sig2G + sig2B) / 3.0, floor2);

    // R/G/B stay unfloored so they remain the raw per-channel split.
    outColor = vec4(sig2R, sig2G, sig2B, meanFloored);
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