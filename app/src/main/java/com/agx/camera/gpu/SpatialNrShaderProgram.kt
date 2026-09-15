package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Legacy Bayer-domain spatial noise reduction (GLES 3.0), used as a fallback
 * when the DPC + green-guided-filter chain is unavailable.
 *
 * Per output texel, computes a Gaussian-weighted average of same-CFA-phase
 * neighbors in a 5x5 lattice window (+-4 sensor pixels, 25 samples per phase),
 * then blends toward the raw center sample by the strength slider (0 = exactly
 * raw, 1 = full average). Each output texel carries all four mosaic phases
 * (R, G1, G2, B) for the demosaic pass -- critical for avoiding magenta casts
 * and mosaic artefacts.
 *
 * Single RGBA32F output, O(output pixels), no history, no MRT.
 */
class SpatialNrShaderProgram {

    private var programId = 0

    private var uBayerTexLoc = 0
    private var uTransformMatrixLoc = 0
    private var uSensorSizeLoc = 0
    private var uCropOriginLoc = 0
    private var uCropSizeLoc = 0
    private var uViewSizeLoc = 0
    private var uBlackLevelPatternLoc = 0
    private var uBitDepthLoc = 0
    private var uNrStrengthLoc = 0

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create() {
        programId = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        if (programId == 0) {
            Log.e(TAG, "Failed to create Spatial NR shader program")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create Spatial NR shader program")
            return
        }

        uBayerTexLoc = GLES20.glGetUniformLocation(programId, "u_bayerTex")
        uTransformMatrixLoc = GLES20.glGetUniformLocation(programId, "u_transformMatrix")
        uSensorSizeLoc = GLES20.glGetUniformLocation(programId, "u_sensorSize")
        uCropOriginLoc = GLES20.glGetUniformLocation(programId, "u_cropOrigin")
        uCropSizeLoc = GLES20.glGetUniformLocation(programId, "u_cropSize")
        uViewSizeLoc = GLES20.glGetUniformLocation(programId, "u_viewSize")
        uBlackLevelPatternLoc = GLES20.glGetUniformLocation(programId, "u_black_level_pattern")
        uBitDepthLoc = GLES20.glGetUniformLocation(programId, "u_bit_depth")
        uNrStrengthLoc = GLES20.glGetUniformLocation(programId, "u_nr_strength")

        Log.d(TAG, "Spatial NR shader program created: $programId")
        com.agx.camera.CrashLogger.log(TAG, "Program created: spatialNr=$programId")
    }

    fun draw(
        transformMatrix: FloatArray,
        cropOriginX: Float, cropOriginY: Float,
        cropSizeX: Float, cropSizeY: Float,
        viewWidth: Float, viewHeight: Float,
        sensorWidth: Float, sensorHeight: Float,
        bayerTex: Int,
        blackLevelPattern: IntArray,
        bitDepth: Int,
        nrStrength: Float
    ) {
        if (programId == 0) return
        GLES20.glUseProgram(programId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, bayerTex)
        GLES20.glUniform1i(uBayerTexLoc, 0)

        GLES20.glUniformMatrix4fv(uTransformMatrixLoc, 1, false, transformMatrix, 0)
        GLES20.glUniform2f(uSensorSizeLoc, sensorWidth, sensorHeight)
        GLES20.glUniform2f(uCropOriginLoc, cropOriginX, cropOriginY)
        GLES20.glUniform2f(uCropSizeLoc, cropSizeX, cropSizeY)
        GLES20.glUniform2f(uViewSizeLoc, viewWidth, viewHeight)
        GLES20.glUniform4i(uBlackLevelPatternLoc,
            blackLevelPattern[0], blackLevelPattern[1],
            blackLevelPattern[2], blackLevelPattern[3])
        GLES20.glUniform1i(uBitDepthLoc, bitDepth)
        GLES20.glUniform1f(uNrStrengthLoc, nrStrength)

        val posHandle = GLES20.glGetAttribLocation(programId, "a_position")
        val texHandle = GLES20.glGetAttribLocation(programId, "a_texCoord")

        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 2, GLES20.GL_FLOAT, false, 0, quadVertices)

        GLES20.glEnableVertexAttribArray(texHandle)
        GLES20.glVertexAttribPointer(texHandle, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(posHandle)
        GLES20.glDisableVertexAttribArray(texHandle)
    }

    fun isReady(): Boolean = programId != 0

    fun destroy() {
        if (programId != 0) {
            GLES20.glDeleteProgram(programId)
            programId = 0
        }
    }

    companion object {
        private const val TAG = "SpatialNrShaderProgram"

        private val QUAD_COORDS = floatArrayOf(
            -1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f
        )
        private val QUAD_TEX_COORDS = floatArrayOf(
            0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f
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

        private const val FRAGMENT_SHADER = """
#version 300 es
precision highp float;
precision highp usampler2D;
precision highp int;

in vec2 v_texCoord;
out vec4 outDenoised;

uniform usampler2D u_bayerTex;
uniform vec2 u_sensorSize;
uniform vec2 u_cropOrigin;
uniform vec2 u_cropSize;
uniform vec2 u_viewSize;
uniform ivec4 u_black_level_pattern;
uniform int u_bit_depth;
uniform float u_nr_strength;

float unpackRaw(uint rawPacked) {
    uint mask;
    if (u_bit_depth <= 10) {
        mask = 0x3FFu;
    } else if (u_bit_depth <= 12) {
        mask = 0xFFFu;
    } else if (u_bit_depth <= 14) {
        mask = 0x3FFFu;
    } else {
        mask = 0xFFFFu;
    }
    return float(rawPacked & mask);
}

void main() {
    vec2 sensorUV = u_cropOrigin + v_texCoord * u_cropSize;
    sensorUV = clamp(sensorUV, vec2(0.0), u_sensorSize - vec2(1.0));
    ivec2 sc = ivec2(floor(sensorUV));

    int parityX = abs(sc.x % 2);
    int parityY = abs(sc.y % 2);

    // A downscaled 1:1.5 sensor footprint per output texel.  The preview
    // demosaic box-AA (u_box_aa=4) averages a 4x4 sensor window — so with
    // stronger downscale that window collapses onto this one texel, and the
    // box-AA's noise reduction is lost unless this texel carries a per-phase
    // average over that same 4x4 window.  With weaker downscale the box-AA
    // exercises many texels and the raw single-cell anchor is correct.
    float extent = max(u_cropSize.x / u_viewSize.x, u_cropSize.y / u_viewSize.y);
    float alpha = clamp((extent - 1.5) / 1.5, 0.0, 1.0);

    // Per-phase average over the demosaic box-AA window [sc-1, sc+2]^2.
    vec4 boxSum = vec4(0.0);
    ivec4 boxCnt = ivec4(0);
    for (int dy = -1; dy <= 2; dy++) {
        for (int dx = -1; dx <= 2; dx++) {
            ivec2 coord = sc + ivec2(dx, dy);
            coord = clamp(coord, ivec2(0), ivec2(u_sensorSize) - ivec2(1));
            int ph = abs(coord.x % 2) + abs(coord.y % 2) * 2;
            float val = unpackRaw(texelFetch(u_bayerTex, coord, 0).r);
            val -= float(u_black_level_pattern[ph]);
            val = max(val, 0.0);
            if (ph == 0) { boxSum.x += val; boxCnt.x++; }
            else if (ph == 1) { boxSum.y += val; boxCnt.y++; }
            else if (ph == 2) { boxSum.z += val; boxCnt.z++; }
            else { boxSum.w += val; boxCnt.w++; }
        }
    }
    vec4 boxAvg = vec4(0.0);
    boxAvg.x = (boxCnt.x > 0) ? boxSum.x / float(boxCnt.x) : 0.0;
    boxAvg.y = (boxCnt.y > 0) ? boxSum.y / float(boxCnt.y) : 0.0;
    boxAvg.z = (boxCnt.z > 0) ? boxSum.z / float(boxCnt.z) : 0.0;
    boxAvg.w = (boxCnt.w > 0) ? boxSum.w / float(boxCnt.w) : 0.0;

    vec4 result = vec4(0.0);
    for (int p = 0; p < 4; p++) {
        int phaseX = p & 1;
        int phaseY = p >> 1;

        // Center pixel for this phase: the sensor pixel in the texel's 2x2
        // cell whose CFA phase matches p.  parityX^(p&1) gives 0 when the
        // cell origin already has the right x-parity, 1 otherwise.
        ivec2 cc = sc + ivec2(parityX ^ phaseX, parityY ^ phaseY);

        // Gaussian-weighted average of same-phase neighbors in a 5x5 lattice
        // (±4 sensor pixels, 25 samples).  σ² ≈ 4 gives gentle smoothing.
        float sum = 0.0;
        float wsum = 0.0;
        for (int dy = -4; dy <= 4; dy += 2) {
            for (int dx = -4; dx <= 4; dx += 2) {
                ivec2 coord = cc + ivec2(dx, dy);
                coord = clamp(coord, ivec2(0), ivec2(u_sensorSize) - ivec2(1));
                // Only accept same-CFA-phase pixels (absolute parity check).
                if (abs(coord.x % 2) != phaseX || abs(coord.y % 2) != phaseY) {
                    continue;
                }
                float val = unpackRaw(texelFetch(u_bayerTex, coord, 0).r);
                int ph = abs(coord.x % 2) + abs(coord.y % 2) * 2;
                val -= float(u_black_level_pattern[ph]);
                val = max(val, 0.0);
                float d2 = float(dx * dx + dy * dy);
                float w = exp(-d2 / 8.0);
                sum += val * w;
                wsum += w;
            }
        }

        // Center sample for this phase.
        float centerVal = unpackRaw(texelFetch(u_bayerTex, cc, 0).r);
        centerVal -= float(u_black_level_pattern[p]);
        centerVal = max(centerVal, 0.0);

        float gaussAvg = (wsum > 1e-12) ? sum / wsum : centerVal;

        // Anchor: ramp from the single-cell sample (near 1:1) to the box-AA
        // window average (zoomed out) so the strength-0 result never throws
        // away the demosaic box-AA's noise reduction.
        float boxVal = (p == 0) ? boxAvg.x : (p == 1) ? boxAvg.y : (p == 2) ? boxAvg.z : boxAvg.w;
        float anchor = mix(centerVal, boxVal, alpha);

        result[p] = mix(anchor, gaussAvg, u_nr_strength);
    }

    outDenoised = result;
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