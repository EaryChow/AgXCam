package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class YuvShaderProgram {

    private var programId = 0
    private var yTextureId = 0
    private var uTextureId = 0
    private var vTextureId = 0

    private var uYTexLoc = 0
    private var uUTexLoc = 0
    private var vVTexLoc = 0
    private var uZoomFactorLoc = 0
    private var uZoomCenterLoc = 0
    private var uOutputResolutionLoc = 0
    private var uSensorOrientationLoc = 0
    private var uFlipXLoc = 0

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

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create() {
        programId = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        if (programId == 0) {
            Log.e(TAG, "Failed to create YUV shader program")
            return
        }

        uYTexLoc = GLES20.glGetUniformLocation(programId, "u_yTex")
        uUTexLoc = GLES20.glGetUniformLocation(programId, "u_uTex")
        vVTexLoc = GLES20.glGetUniformLocation(programId, "u_vTex")
        uZoomFactorLoc = GLES20.glGetUniformLocation(programId, "u_zoom_factor")
        uZoomCenterLoc = GLES20.glGetUniformLocation(programId, "u_zoom_center")
        uOutputResolutionLoc = GLES20.glGetUniformLocation(programId, "u_outputResolution")
        uSensorOrientationLoc = GLES20.glGetUniformLocation(programId, "u_sensorOrientation")
        uFlipXLoc = GLES20.glGetUniformLocation(programId, "u_flipX")

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

        val textures = IntArray(3)
        GLES20.glGenTextures(3, textures, 0)
        yTextureId = textures[0]
        uTextureId = textures[1]
        vTextureId = textures[2]

        for (texId in textures) {
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texId)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        }

        Log.d(TAG, "YUV shader program created: $programId")
    }

    fun uploadY(plane: ByteBuffer, width: Int, height: Int) {
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, yTextureId)
        plane.position(0)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE,
            width, height, 0,
            GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, plane
        )
    }

    fun uploadU(plane: ByteBuffer, width: Int, height: Int) {
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, uTextureId)
        plane.position(0)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE,
            width, height, 0,
            GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, plane
        )
    }

    fun uploadV(plane: ByteBuffer, width: Int, height: Int) {
        GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, vTextureId)
        plane.position(0)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE,
            width, height, 0,
            GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, plane
        )
    }

    fun draw(
        outputWidth: Int, outputHeight: Int,
        zoomFactor: Float, zoomCenterX: Float, zoomCenterY: Float,
        sensorOrientation: Int, flipX: Boolean,
        sceneLinearTo709: FloatArray,
        insetMat: FloatArray,
        outsetMat: FloatArray,
        toRec2020: FloatArray,
        whiteLevel: Float, blackLevel: Float,
        logMin: Float, logMax: Float,
        logMidgray: Float, displayMidgray: Float,
        contrast: Float, toe: Float, shoulder: Float
    ) {
        GLES20.glUseProgram(programId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, yTextureId)
        GLES20.glUniform1i(uYTexLoc, 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, uTextureId)
        GLES20.glUniform1i(uUTexLoc, 1)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, vTextureId)
        GLES20.glUniform1i(vVTexLoc, 2)

        GLES20.glUniform1f(uZoomFactorLoc, zoomFactor)
        GLES20.glUniform2f(uZoomCenterLoc, zoomCenterX, zoomCenterY)
        GLES20.glUniform2f(uOutputResolutionLoc, outputWidth.toFloat(), outputHeight.toFloat())
        GLES20.glUniform1f(uSensorOrientationLoc, sensorOrientation.toFloat())
        GLES20.glUniform1f(uFlipXLoc, if (flipX) 1.0f else 0.0f)

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

    fun destroy() {
        if (programId != 0) {
            GLES20.glDeleteProgram(programId)
            programId = 0
        }
        val textures = intArrayOf(yTextureId, uTextureId, vTextureId)
        GLES20.glDeleteTextures(3, textures, 0)
        yTextureId = 0
        uTextureId = 0
        vTextureId = 0
    }

    companion object {
        private const val TAG = "YuvShaderProgram"

        private val QUAD_COORDS = floatArrayOf(
            -1f, -1f,  1f, -1f,  -1f, 1f,  1f, 1f
        )
        private val QUAD_TEX_COORDS = floatArrayOf(
            0f, 0f,  1f, 0f,  0f, 1f,  1f, 1f
        )

        private const val VERTEX_SHADER = """
attribute vec2 a_position;
attribute vec2 a_texCoord;
varying vec2 v_texCoord;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
}
"""

        private const val FRAGMENT_SHADER = """
precision mediump float;
varying vec2 v_texCoord;
uniform sampler2D u_yTex;
uniform sampler2D u_uTex;
uniform sampler2D u_vTex;
uniform float u_zoom_factor;
uniform vec2 u_zoom_center;
uniform vec2 u_outputResolution;
uniform float u_sensorOrientation;
uniform float u_flipX;

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

${AgxCoreGlsl.CORE_HELPERS}

vec3 agxFormationYuv(vec3 rgb) {
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

void main() {
    vec2 uv = v_texCoord;
    uv = (uv - u_zoom_center) / u_zoom_factor + u_zoom_center;
    float angle = u_sensorOrientation;
    if (angle == 90.0) {
        uv = vec2(uv.y, 1.0 - uv.x);
    } else if (angle == 180.0) {
        uv = 1.0 - uv;
    } else if (angle == 270.0) {
        uv = vec2(1.0 - uv.y, uv.x);
    }
    if (u_flipX > 0.5) {
        uv.x = 1.0 - uv.x;
    }

    float y = texture2D(u_yTex, uv).r;
    float u = texture2D(u_uTex, uv).r - 0.5;
    float v = texture2D(u_vTex, uv).r - 0.5;
    vec3 rgb = vec3(
        y + 1.402 * v,
        y - 0.344136 * u - 0.714136 * v,
        y + 1.772 * u
    );

    rgb = agxFormationYuv(rgb);
    gl_FragColor = vec4(rgb, 1.0);
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
    }
}
