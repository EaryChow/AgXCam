package com.agx.camera.gpu

import android.opengl.GLES20
import android.opengl.GLES30
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

private const val GL_HALF_FLOAT = 0x140B

class BayerShaderProgram {

    private var programId = 0
    private var demosaicProgramId = 0
    private var bayerTextureId = 0
    private var lensShadingTextureId = 0

    private var uBayerTexLoc = 0
    private var uLensShadingMapLoc = 0
    private var uOutputResolutionLoc = 0
    private var uTransformMatrixLoc = 0
    private var uSensorSizeLoc = 0

    private var uSceneLinearTo709Loc = 0
    private var uInsetmatLoc = 0
    private var uOutsetmatLoc = 0
    private var u709To2020Loc = 0
    private var uWhiteLevelLoc = 0
    private var uBlackLevelLoc = 0
    private var uLogMinLoc = 0
    private var uLogMaxLoc = 0
    private var uLogMidgrayLoc = 0
    private var uDisplayMidgrayLoc = 0
    private var uContrastLoc = 0
    private var uToeLoc = 0
    private var uShoulderLoc = 0

    private var uBlackLevelPatternLoc = 0
    private var uBayerColorMapLoc = 0
    private var uBitDepthLoc = 0
    private var uNrStrengthLoc = 0

    private var dUTextureLoc = 0
    private var dULensShadingMapLoc = 0
    private var dUOutputResolutionLoc = 0
    private var dUTransformMatrixLoc = 0
    private var dUSensorSizeLoc = 0
    private var dUBlackLevelPatternLoc = 0
    private var dUBayerColorMapLoc = 0
    private var dUBitDepthLoc = 0

    private var sensorWidth = 0
    private var sensorHeight = 0

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create(sensorW: Int, sensorH: Int) {
        sensorWidth = sensorW
        sensorHeight = sensorH

        programId = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        if (programId == 0) {
            Log.e(TAG, "Failed to create Bayer shader program")
            return
        }

        demosaicProgramId = createProgram(VERTEX_SHADER, DEMOSAIC_FRAGMENT_SHADER)
        if (demosaicProgramId == 0) {
            Log.e(TAG, "Failed to create demosaic shader program")
            return
        }

        uBayerTexLoc = GLES20.glGetUniformLocation(programId, "u_bayerTex")
        uLensShadingMapLoc = GLES20.glGetUniformLocation(programId, "u_lens_shading_map")
        uOutputResolutionLoc = GLES20.glGetUniformLocation(programId, "u_outputResolution")
        uTransformMatrixLoc = GLES20.glGetUniformLocation(programId, "u_transformMatrix")
        uSensorSizeLoc = GLES20.glGetUniformLocation(programId, "u_sensorSize")

        uSceneLinearTo709Loc = GLES20.glGetUniformLocation(programId, "u_scene_linear_to_709")
        uInsetmatLoc = GLES20.glGetUniformLocation(programId, "u_insetmat")
        uOutsetmatLoc = GLES20.glGetUniformLocation(programId, "u_outsetmat")
        u709To2020Loc = GLES20.glGetUniformLocation(programId, "u_709_to_2020")
        uWhiteLevelLoc = GLES20.glGetUniformLocation(programId, "u_white_level")
        uBlackLevelLoc = GLES20.glGetUniformLocation(programId, "u_black_level")
        uLogMinLoc = GLES20.glGetUniformLocation(programId, "u_log_min")
        uLogMaxLoc = GLES20.glGetUniformLocation(programId, "u_log_max")
        uLogMidgrayLoc = GLES20.glGetUniformLocation(programId, "u_log_midgray")
        uDisplayMidgrayLoc = GLES20.glGetUniformLocation(programId, "u_display_midgray")
        uContrastLoc = GLES20.glGetUniformLocation(programId, "u_contrast")
        uToeLoc = GLES20.glGetUniformLocation(programId, "u_toe")
        uShoulderLoc = GLES20.glGetUniformLocation(programId, "u_shoulder")

        uBlackLevelPatternLoc = GLES20.glGetUniformLocation(programId, "u_black_level_pattern")
        uBayerColorMapLoc = GLES20.glGetUniformLocation(programId, "u_bayer_color_map")
        uBitDepthLoc = GLES20.glGetUniformLocation(programId, "u_bit_depth")
        uNrStrengthLoc = GLES20.glGetUniformLocation(programId, "u_nr_strength")

        dUTextureLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_bayerTex")
        dULensShadingMapLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_lens_shading_map")
        dUOutputResolutionLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_outputResolution")
        dUTransformMatrixLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_transformMatrix")
        dUSensorSizeLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_sensorSize")
        dUBlackLevelPatternLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_black_level_pattern")
        dUBayerColorMapLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_bayer_color_map")
        dUBitDepthLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_bit_depth")

        val textures = IntArray(2)
        GLES20.glGenTextures(2, textures, 0)
        bayerTextureId = textures[0]
        lensShadingTextureId = textures[1]

        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, bayerTextureId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, lensShadingTextureId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        Log.d(TAG, "Bayer shader program created: $programId, sensor: ${sensorWidth}x${sensorHeight}")
    }

    fun uploadBayer(buffer: ByteBuffer, width: Int, height: Int, stridePixels: Int) {
        GLES30.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, bayerTextureId)

        if (stridePixels > width && stridePixels > 0) {
            GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, stridePixels)
        }

        buffer.position(0)
        GLES30.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_R16UI,
            width, height, 0,
            GLES30.GL_RED_INTEGER, GLES20.GL_UNSIGNED_SHORT, buffer
        )

        if (stridePixels > width && stridePixels > 0) {
            GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, 0)
        }
    }

    fun uploadLensShadingMap(data: ShortArray, width: Int, height: Int) {
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, lensShadingTextureId)

        val buf = ByteBuffer.allocateDirect(data.size * 2)
            .order(ByteOrder.nativeOrder())
        buf.asShortBuffer().put(data)
        buf.position(0)

        GLES30.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F,
            width, height, 0,
            GLES30.GL_RGBA, GL_HALF_FLOAT, buf
        )
    }

    fun uploadIdentityLensShading() {
        val identity = ShortArray(4) { floatToHalf(1.0f) }
        uploadLensShadingMap(identity, 1, 1)
    }

    fun draw(
        outputWidth: Int, outputHeight: Int,
        transformMatrix: FloatArray,
        sceneLinearTo709: FloatArray,
        insetMat: FloatArray,
        outsetMat: FloatArray,
        toRec2020: FloatArray,
        whiteLevel: Float, blackLevel: Float,
        logMin: Float, logMax: Float,
        logMidgray: Float, displayMidgray: Float,
        contrast: Float, toe: Float, shoulder: Float,
        blackLevelPattern: IntArray,
        bayerColorMap: IntArray,
        bitDepth: Int,
        nrStrength: Float
    ) {
        GLES20.glUseProgram(programId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, bayerTextureId)
        GLES20.glUniform1i(uBayerTexLoc, 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, lensShadingTextureId)
        GLES20.glUniform1i(uLensShadingMapLoc, 1)

        GLES20.glUniform2f(uOutputResolutionLoc, outputWidth.toFloat(), outputHeight.toFloat())
        GLES20.glUniformMatrix4fv(uTransformMatrixLoc, 1, false, transformMatrix, 0)
        GLES20.glUniform2f(uSensorSizeLoc, sensorWidth.toFloat(), sensorHeight.toFloat())

        GLES20.glUniformMatrix3fv(uSceneLinearTo709Loc, 1, true, sceneLinearTo709, 0)
        GLES20.glUniformMatrix3fv(uInsetmatLoc, 1, true, insetMat, 0)
        GLES20.glUniformMatrix3fv(uOutsetmatLoc, 1, true, outsetMat, 0)
        GLES20.glUniformMatrix3fv(u709To2020Loc, 1, true, toRec2020, 0)
        GLES20.glUniform1f(uWhiteLevelLoc, whiteLevel)
        GLES20.glUniform1f(uBlackLevelLoc, blackLevel)
        GLES20.glUniform1f(uLogMinLoc, logMin)
        GLES20.glUniform1f(uLogMaxLoc, logMax)
        GLES20.glUniform1f(uLogMidgrayLoc, logMidgray)
        GLES20.glUniform1f(uDisplayMidgrayLoc, displayMidgray)
        GLES20.glUniform1f(uContrastLoc, contrast)
        GLES20.glUniform1f(uToeLoc, toe)
        GLES20.glUniform1f(uShoulderLoc, shoulder)

        GLES20.glUniform4i(uBlackLevelPatternLoc,
            blackLevelPattern[0], blackLevelPattern[1],
            blackLevelPattern[2], blackLevelPattern[3])
        GLES20.glUniform4i(uBayerColorMapLoc,
            bayerColorMap[0], bayerColorMap[1],
            bayerColorMap[2], bayerColorMap[3])
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

    fun drawDemosaic(
        outputWidth: Int, outputHeight: Int,
        transformMatrix: FloatArray,
        blackLevelPattern: IntArray,
        bayerColorMap: IntArray,
        bitDepth: Int
    ) {
        GLES20.glUseProgram(demosaicProgramId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, bayerTextureId)
        GLES20.glUniform1i(dUTextureLoc, 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, lensShadingTextureId)
        GLES20.glUniform1i(dULensShadingMapLoc, 1)

        GLES20.glUniform2f(dUOutputResolutionLoc, outputWidth.toFloat(), outputHeight.toFloat())
        GLES20.glUniformMatrix4fv(dUTransformMatrixLoc, 1, false, transformMatrix, 0)
        GLES20.glUniform2f(dUSensorSizeLoc, sensorWidth.toFloat(), sensorHeight.toFloat())

        GLES20.glUniform4i(dUBlackLevelPatternLoc,
            blackLevelPattern[0], blackLevelPattern[1],
            blackLevelPattern[2], blackLevelPattern[3])
        GLES20.glUniform4i(dUBayerColorMapLoc,
            bayerColorMap[0], bayerColorMap[1],
            bayerColorMap[2], bayerColorMap[3])
        GLES20.glUniform1i(dUBitDepthLoc, bitDepth)

        val posHandle = GLES20.glGetAttribLocation(demosaicProgramId, "a_position")
        val texHandle = GLES20.glGetAttribLocation(demosaicProgramId, "a_texCoord")

        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 2, GLES20.GL_FLOAT, false, 0, quadVertices)

        GLES20.glEnableVertexAttribArray(texHandle)
        GLES20.glVertexAttribPointer(texHandle, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(posHandle)
        GLES20.glDisableVertexAttribArray(texHandle)
    }

    fun destroy() {
        if (programId != 0) {
            GLES20.glDeleteProgram(programId)
            programId = 0
        }
        if (demosaicProgramId != 0) {
            GLES20.glDeleteProgram(demosaicProgramId)
            demosaicProgramId = 0
        }
        val textures = intArrayOf(bayerTextureId, lensShadingTextureId)
        GLES20.glDeleteTextures(2, textures, 0)
        bayerTextureId = 0
        lensShadingTextureId = 0
    }

    companion object {
        private const val TAG = "BayerShaderProgram"

        private val QUAD_COORDS = floatArrayOf(
            -1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f
        )
        private val QUAD_TEX_COORDS = floatArrayOf(
            0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f
        )

        private const val VERTEX_SHADER = """
attribute vec2 a_position;
attribute vec2 a_texCoord;
varying vec2 v_texCoord;
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
out vec4 fragColor;

uniform usampler2D u_bayerTex;
uniform sampler2D u_lens_shading_map;
uniform vec2 u_outputResolution;
uniform vec2 u_sensorSize;

uniform mat3 u_scene_linear_to_709;
uniform mat3 u_insetmat;
uniform mat3 u_outsetmat;
uniform mat3 u_709_to_2020;
uniform float u_white_level;
uniform float u_black_level;
uniform float u_log_min;
uniform float u_log_max;
uniform float u_log_midgray;
uniform float u_display_midgray;
uniform float u_contrast;
uniform float u_toe;
uniform float u_shoulder;

uniform ivec4 u_black_level_pattern;
uniform ivec4 u_bayer_color_map;
uniform int u_bit_depth;
uniform float u_nr_strength;

${AgxCoreGlsl.CORE_HELPERS}

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

float sampleBayerRaw(ivec2 coord) {
    ivec2 clamped = clamp(coord, ivec2(0), ivec2(u_sensorSize) - ivec2(1));
    uint raw = texelFetch(u_bayerTex, clamped, 0).r;
    float val = unpackRaw(raw);
    int phase = abs(clamped.x % 2) + abs(clamped.y % 2) * 2;
    val -= float(u_black_level_pattern[phase]);
    return max(val, 0.0);
}

int safePhase(int x, int y) {
    return abs(x % 2) + abs(y % 2) * 2;
}

vec2 clampSensor(vec2 v) {
    return clamp(v, vec2(0.0), u_sensorSize - vec2(1.0));
}

float lsGain(vec2 lsNeighborUV) {
    ivec2 c = ivec2(clampSensor(lsNeighborUV));
    return texture(u_lens_shading_map, vec2(c) / u_sensorSize)[safePhase(c.x, c.y)];
}

vec3 demosaicBilinear(usampler2D tex, vec2 sensorUV, vec2 lsSensorUV) {
    ivec2 sensorCoord = ivec2(sensorUV);
    int phase = safePhase(sensorCoord.x, sensorCoord.y);
    int color = u_bayer_color_map[phase];

    float gain = texture(u_lens_shading_map, lsSensorUV / u_sensorSize)[phase];

    ivec2 nN_coord = sensorCoord + ivec2( 0, -1);
    ivec2 nS_coord = sensorCoord + ivec2( 0,  1);
    ivec2 nW_coord = sensorCoord + ivec2(-1,  0);
    ivec2 nE_coord = sensorCoord + ivec2( 1,  0);
    ivec2 nNW_coord = sensorCoord + ivec2(-1, -1);
    ivec2 nNE_coord = sensorCoord + ivec2( 1, -1);
    ivec2 nSW_coord = sensorCoord + ivec2(-1,  1);
    ivec2 nSE_coord = sensorCoord + ivec2( 1,  1);

    vec2 lsN  = lsSensorUV + vec2( 0.0, -1.0);
    vec2 lsS  = lsSensorUV + vec2( 0.0,  1.0);
    vec2 lsW  = lsSensorUV + vec2(-1.0,  0.0);
    vec2 lsE  = lsSensorUV + vec2( 1.0,  0.0);
    vec2 lsNW = lsSensorUV + vec2(-1.0, -1.0);
    vec2 lsNE = lsSensorUV + vec2( 1.0, -1.0);
    vec2 lsSW = lsSensorUV + vec2(-1.0,  1.0);
    vec2 lsSE = lsSensorUV + vec2( 1.0,  1.0);

    float nN  = sampleBayerRaw(nN_coord)  * lsGain(lsN);
    float nS  = sampleBayerRaw(nS_coord)  * lsGain(lsS);
    float nW  = sampleBayerRaw(nW_coord)  * lsGain(lsW);
    float nE  = sampleBayerRaw(nE_coord)  * lsGain(lsE);

    float nNW = sampleBayerRaw(nNW_coord) * lsGain(lsNW);
    float nNE = sampleBayerRaw(nNE_coord) * lsGain(lsNE);
    float nSW = sampleBayerRaw(nSW_coord) * lsGain(lsSW);
    float nSE = sampleBayerRaw(nSE_coord) * lsGain(lsSE);

    float gradH = abs(nW - nE);
    float gradV = abs(nN - nS);

    float center = sampleBayerRaw(sensorCoord) * gain;
    float r = 0.0, g = 0.0, b = 0.0;

    if (color == 0) {
        r = center;
        float diag = (nNW + nNE + nSW + nSE) * 0.25;
        g = (gradH > gradV) ? (nW + nE) * 0.5 : diag;
        b = diag;
    } else if (color == 2) {
        b = center;
        float diag = (nNW + nNE + nSW + nSE) * 0.25;
        g = (gradH > gradV) ? (nW + nE) * 0.5 : diag;
        r = diag;
    } else {
        g = center;
        int colorNS = u_bayer_color_map[safePhase(sensorCoord.x, sensorCoord.y - 1)];
        if (colorNS == 0) {
            r = (nN + nS) * 0.5;
            b = (nW + nE) * 0.5;
        } else {
            r = (nW + nE) * 0.5;
            b = (nN + nS) * 0.5;
        }
    }

    return vec3(r, g, b);
}

vec3 previewNR(usampler2D tex, vec2 uv, vec2 lsUV, float strength) {
    if (strength <= 0.0) {
        return demosaicBilinear(tex, uv, lsUV);
    }
    vec3 center = demosaicBilinear(tex, uv, lsUV);
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 pixelOff = vec2(float(x), float(y));
            vec3 s = demosaicBilinear(tex, uv + pixelOff, lsUV + pixelOff);
            float w = 1.0 / (1.0 + length(s - center));
            sum += s * w;
            wsum += w;
        }
    }
    vec3 filtered = sum / wsum;
    return mix(center, filtered, strength);
}

${AgxCoreGlsl.AGX_FORMATION}

void main() {
    vec2 uv = v_texCoord;
    vec2 lsSensorUV = uv * u_sensorSize;

    vec2 sensorUV = uv * u_sensorSize;
    sensorUV = clamp(sensorUV, vec2(0.0), u_sensorSize - vec2(1.0));
    lsSensorUV = clamp(lsSensorUV, vec2(0.0), u_sensorSize - vec2(1.0));

    vec3 linearRGB = previewNR(u_bayerTex, sensorUV, lsSensorUV, u_nr_strength);

    vec3 formed = agxFormation(linearRGB);
    fragColor = vec4(formed, 1.0);
}
"""

        private const val DEMOSAIC_FRAGMENT_SHADER = """
#version 300 es
precision highp float;
precision highp usampler2D;
precision highp int;

in vec2 v_texCoord;
out vec4 fragColor;

uniform usampler2D u_bayerTex;
uniform sampler2D u_lens_shading_map;
uniform vec2 u_outputResolution;
uniform vec2 u_sensorSize;

uniform ivec4 u_black_level_pattern;
uniform ivec4 u_bayer_color_map;
uniform int u_bit_depth;

${AgxCoreGlsl.CORE_HELPERS}

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

float sampleBayerRaw(ivec2 coord) {
    ivec2 clamped = clamp(coord, ivec2(0), ivec2(u_sensorSize) - ivec2(1));
    uint raw = texelFetch(u_bayerTex, clamped, 0).r;
    float val = unpackRaw(raw);
    int phase = abs(clamped.x % 2) + abs(clamped.y % 2) * 2;
    val -= float(u_black_level_pattern[phase]);
    return max(val, 0.0);
}

int safePhase(int x, int y) {
    return abs(x % 2) + abs(y % 2) * 2;
}

vec2 clampSensor(vec2 v) {
    return clamp(v, vec2(0.0), u_sensorSize - vec2(1.0));
}

float lsGain(vec2 lsNeighborUV) {
    ivec2 c = ivec2(clampSensor(lsNeighborUV));
    return texture(u_lens_shading_map, vec2(c) / u_sensorSize)[safePhase(c.x, c.y)];
}

vec3 demosaicBilinear(usampler2D tex, vec2 sensorUV, vec2 lsSensorUV) {
    ivec2 sensorCoord = ivec2(sensorUV);
    int phase = safePhase(sensorCoord.x, sensorCoord.y);
    int color = u_bayer_color_map[phase];

    float gain = texture(u_lens_shading_map, lsSensorUV / u_sensorSize)[phase];

    ivec2 nN_coord = sensorCoord + ivec2( 0, -1);
    ivec2 nS_coord = sensorCoord + ivec2( 0,  1);
    ivec2 nW_coord = sensorCoord + ivec2(-1,  0);
    ivec2 nE_coord = sensorCoord + ivec2( 1,  0);
    ivec2 nNW_coord = sensorCoord + ivec2(-1, -1);
    ivec2 nNE_coord = sensorCoord + ivec2( 1, -1);
    ivec2 nSW_coord = sensorCoord + ivec2(-1,  1);
    ivec2 nSE_coord = sensorCoord + ivec2( 1,  1);

    vec2 lsN  = lsSensorUV + vec2( 0.0, -1.0);
    vec2 lsS  = lsSensorUV + vec2( 0.0,  1.0);
    vec2 lsW  = lsSensorUV + vec2(-1.0,  0.0);
    vec2 lsE  = lsSensorUV + vec2( 1.0,  0.0);
    vec2 lsNW = lsSensorUV + vec2(-1.0, -1.0);
    vec2 lsNE = lsSensorUV + vec2( 1.0, -1.0);
    vec2 lsSW = lsSensorUV + vec2(-1.0,  1.0);
    vec2 lsSE = lsSensorUV + vec2( 1.0,  1.0);

    float nN  = sampleBayerRaw(nN_coord)  * lsGain(lsN);
    float nS  = sampleBayerRaw(nS_coord)  * lsGain(lsS);
    float nW  = sampleBayerRaw(nW_coord)  * lsGain(lsW);
    float nE  = sampleBayerRaw(nE_coord)  * lsGain(lsE);

    float nNW = sampleBayerRaw(nNW_coord) * lsGain(lsNW);
    float nNE = sampleBayerRaw(nNE_coord) * lsGain(lsNE);
    float nSW = sampleBayerRaw(nSW_coord) * lsGain(lsSW);
    float nSE = sampleBayerRaw(nSE_coord) * lsGain(lsSE);

    float gradH = abs(nW - nE);
    float gradV = abs(nN - nS);

    float center = sampleBayerRaw(sensorCoord) * gain;
    float r = 0.0, g = 0.0, b = 0.0;

    if (color == 0) {
        r = center;
        float diag = (nNW + nNE + nSW + nSE) * 0.25;
        g = (gradH > gradV) ? (nW + nE) * 0.5 : diag;
        b = diag;
    } else if (color == 2) {
        b = center;
        float diag = (nNW + nNE + nSW + nSE) * 0.25;
        g = (gradH > gradV) ? (nW + nE) * 0.5 : diag;
        r = diag;
    } else {
        g = center;
        int colorNS = u_bayer_color_map[safePhase(sensorCoord.x, sensorCoord.y - 1)];
        if (colorNS == 0) {
            r = (nN + nS) * 0.5;
            b = (nW + nE) * 0.5;
        } else {
            r = (nW + nE) * 0.5;
            b = (nN + nS) * 0.5;
        }
    }

    return vec3(r, g, b);
}

void main() {
    vec2 uv = v_texCoord;
    vec2 lsSensorUV = uv * u_sensorSize;

    vec2 sensorUV = uv * u_sensorSize;
    sensorUV = clamp(sensorUV, vec2(0.0), u_sensorSize - vec2(1.0));
    lsSensorUV = clamp(lsSensorUV, vec2(0.0), u_sensorSize - vec2(1.0));

    vec3 linearRGB = demosaicBilinear(u_bayerTex, sensorUV, lsSensorUV);
    fragColor = vec4(linearRGB, 1.0);
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
                Log.e(TAG, "Program link failed: ${GLES20.glGetProgramInfoLog(program)}")
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
                Log.e(TAG, "Shader compile failed: ${GLES20.glGetShaderInfoLog(shader)}")
                GLES20.glDeleteShader(shader)
                return 0
            }
            return shader
        }

        private fun floatToHalf(f: Float): Short {
            val bits = java.lang.Float.floatToRawIntBits(f)
            val sign = (bits ushr 16) and 0x8000
            var exp = ((bits ushr 23) and 0xFF) - 127 + 15
            var mantissa = bits and 0x7FFFFF
            if (exp <= 0) {
                exp = 0
                mantissa = 0
            } else if (exp >= 31) {
                exp = 31
                mantissa = 0
            } else {
                mantissa = mantissa shr 13
            }
            return (sign or (exp shl 10) or mantissa).toShort()
        }
    }
}
