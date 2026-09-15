package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Stage 3 — RAW-domain pre-denoise (ISO-driven), green-guided guided filter on
 * the sparse Bayer grid (plan §3 Stage 3). Replaces the placeholder Gaussian
 * kernel; retains the placeholder's shell + C3 indexing contract (one output
 * texel per 2x2 sensor cell, four phases in RGBA, RGBA32F).
 *
 * Input:  sparse grid RGBA32F (black level already subtracted, clamped >= 0).
 * Output: same-layout sparse grid, blended by u_alpha:
 *     out_p = alpha * u_p + (1 - alpha) * GF_G(u)_p
 * with alpha = effective keep-raw blend (1 = fully raw / off, 0.3 = strong),
 * computed on the host from ISO via NoiseModel.alphaRaw(iso) and the S3 slider.
 *
 * The guide is the green channel (highest sampling / SNR): guide value at every
 * texel = mean of the two green phases of that texel. A guided-filter window
 * (5x5 same-CFA lattice, radius 2) per phase produces coefficients
 * a = cov(I,G)/(var(G)+ε), b = meanI − a·meanG, output = a·g0 + b with g0 the
 * guide at the centre.
 */
class RawDenoiseShaderProgram {

    private var programId = 0

    private var uGridTexLoc = 0
    private var uTransformMatrixLoc = 0
    private var uSensorSizeLoc = 0
    private var uCropOriginLoc = 0
    private var uCropSizeLoc = 0
    private var uViewSizeLoc = 0
    private var uAlphaLoc = 0
    private var uEpsLoc = 0

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create() {
        programId = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        if (programId == 0) {
            Log.e(TAG, "Failed to create RAW denoise shader program")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create RAW denoise shader program")
            return
        }
        uGridTexLoc = GLES20.glGetUniformLocation(programId, "u_gridTex")
        uTransformMatrixLoc = GLES20.glGetUniformLocation(programId, "u_transformMatrix")
        uSensorSizeLoc = GLES20.glGetUniformLocation(programId, "u_sensorSize")
        uCropOriginLoc = GLES20.glGetUniformLocation(programId, "u_cropOrigin")
        uCropSizeLoc = GLES20.glGetUniformLocation(programId, "u_cropSize")
        uViewSizeLoc = GLES20.glGetUniformLocation(programId, "u_viewSize")
        uAlphaLoc = GLES20.glGetUniformLocation(programId, "u_alpha")
        uEpsLoc = GLES20.glGetUniformLocation(programId, "u_eps")
        Log.d(TAG, "RAW denoise shader program created: $programId")
        com.agx.camera.CrashLogger.log(TAG, "Program created: rawDenoise=$programId")
    }

    fun draw(
        transformMatrix: FloatArray,
        cropOriginX: Float, cropOriginY: Float,
        cropSizeX: Float, cropSizeY: Float,
        viewWidth: Float, viewHeight: Float,
        sensorWidth: Float, sensorHeight: Float,
        gridTex: Int,
        alpha: Float,
        eps: Float
    ) {
        if (programId == 0) return
        GLES20.glUseProgram(programId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, gridTex)
        GLES20.glUniform1i(uGridTexLoc, 0)

        GLES20.glUniformMatrix4fv(uTransformMatrixLoc, 1, false, transformMatrix, 0)
        GLES20.glUniform2f(uSensorSizeLoc, sensorWidth, sensorHeight)
        GLES20.glUniform2f(uCropOriginLoc, cropOriginX, cropOriginY)
        GLES20.glUniform2f(uCropSizeLoc, cropSizeX, cropSizeY)
        GLES20.glUniform2f(uViewSizeLoc, viewWidth, viewHeight)
        GLES20.glUniform1f(uAlphaLoc, alpha)
        GLES20.glUniform1f(uEpsLoc, eps)

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
        private const val TAG = "RawDenoiseShaderProgram"

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

uniform sampler2D u_gridTex;
uniform vec2 u_sensorSize;
uniform vec2 u_cropOrigin;
uniform vec2 u_cropSize;
uniform vec2 u_viewSize;
uniform float u_alpha;
uniform float u_eps;

// Same-CFA neighbourhood in grid units (radius 2 lattice)
const int WX[25] = int[25](
    -2, -2, -2, -2, -2,
    -1, -1, -1, -1, -1,
     0,  0,  0,  0,  0,
     1,  1,  1,  1,  1,
     2,  2,  2,  2,  2);
const int WY[25] = int[25](
    -2, -1,  0,  1,  2,
    -2, -1,  0,  1,  2,
    -2, -1,  0,  1,  2,
    -2, -1,  0,  1,  2,
    -2, -1,  0,  1,  2);

float channelOf(vec4 c, int p) {
    if (p == 0) return c.r;
    if (p == 1) return c.g;
    if (p == 2) return c.b;
    return c.a;
}

// Green guide value at texel t = mean of the two green phases (high SNR).
float guideAt(ivec2 t) {
    vec4 c = texelFetch(u_gridTex, t, 0);
    return (c.g + c.b) * 0.5;
}

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);
    vec4 center = texelFetch(u_gridTex, base, 0);
    vec4 result = vec4(0.0);

    float g0 = guideAt(base);

    for (int p = 0; p < 4; p++) {
        float y0 = channelOf(center, p);

        float meanI = 0.0;
        float meanG = 0.0;
        float sumII = 0.0;
        float sumGG = 0.0;
        float sumIG = 0.0;
        for (int k = 0; k < 25; k++) {
            ivec2 t = clamp(base + ivec2(WX[k], WY[k]), ivec2(0), ivec2(u_viewSize) - ivec2(1));
            vec4 c = texelFetch(u_gridTex, t, 0);
            float i = channelOf(c, p);
            float g = (c.g + c.b) * 0.5;    // green guide field for every phase
            meanI += i;
            meanG += g;
            sumII += i * i;
            sumGG += g * g;
            sumIG += i * g;
        }
        float invN = 1.0 / 25.0;
        meanI *= invN;
        meanG *= invN;
        float varG = max(sumGG * invN - meanG * meanG, 0.0);
        float covIG = sumIG * invN - meanI * meanG;

        float a = covIG / (varG + u_eps);
        float b = meanI - a * meanG;
        float gfOut = a * g0 + b;
        float outVal = mix(y0, gfOut, 1.0 - u_alpha);
        outVal = max(outVal, 0.0);

        if (p == 0) result.r = outVal;
        else if (p == 1) result.g = outVal;
        else if (p == 2) result.b = outVal;
        else result.a = outVal;
    }
    outColor = result;
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