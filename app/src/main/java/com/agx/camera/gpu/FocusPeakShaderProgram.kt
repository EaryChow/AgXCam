package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Focus-peak composite pass.
 *
 * Takes the fully tone-mapped preview (which already contains the zoom crop) and
 * tints the sharp edges green. Two contrast measures are combined:
 *
 *  - a screen-pixel ring at [radius] (in viewport px) — this is the band detector,
 *    it makes the highlight a fixed width in screen pixels and is unaffected by
 *    zoom/thickness;
 *  - a source-anchored ring at [sharpOffsetX]/[sharpOffsetY] (in UV units that
 *    correspond to a fixed distance in SOURCE pixels, so it scales inversely to
 *    the zoom). This keeps detecting truly sharp edges even after magnification
 *    dilutes their per-screen-pixel gradient.
 *
 * The max of the two passes the threshold, so zooming or switching to a tele lens
 * can no longer make in-focus edges drop out of the highlight. Everything else is
 * passed through untouched. Runs on the displayed FBO content, so it also tracks
 * crop and aspect letterbox.
 */
class FocusPeakShaderProgram {

    private var programId = 0
    private var uTextureLoc = 0
    private var uTexelSizeLoc = 0
    private var uRadiusLoc = 0
    private var uSharpOffsetLoc = 0
    private var uThresholdLoc = 0
    private var uStrengthLoc = 0
    private var uColorLoc = 0

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    /** Highlight colour tint. */
    var color = floatArrayOf(0.1f, 0.95f, 0.15f)

    fun create() {
        programId = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        if (programId == 0) {
            Log.e(TAG, "Failed to create focus peak shader program")
            return
        }
        uTextureLoc = GLES20.glGetUniformLocation(programId, "u_texture")
        uTexelSizeLoc = GLES20.glGetUniformLocation(programId, "u_texelSize")
        uRadiusLoc = GLES20.glGetUniformLocation(programId, "u_radius")
        uSharpOffsetLoc = GLES20.glGetUniformLocation(programId, "u_sharpOffset")
        uThresholdLoc = GLES20.glGetUniformLocation(programId, "u_threshold")
        uStrengthLoc = GLES20.glGetUniformLocation(programId, "u_strength")
        uColorLoc = GLES20.glGetUniformLocation(programId, "u_color")
        Log.d(TAG, "Focus peak shader program created: $programId")
    }

    /**
     * Composite the preview [textureId] with the green focus-peak overlay.
     * [screenWidth]/[screenHeight] are the viewport pixels the FBO is displayed in
     * (texel size = 1/screen), so [radius] is a screen-pixel radius and the line
     * stays a fixed pixel thickness whatever the zoom.
     * [sharpOffsetX]/[sharpOffsetY] are UV steps (in the FBO texture) that span a
     * fixed distance in SOURCE pixels, inverse-scaled by zoom, so the sharpness
     * criterion survives magnification.
     */
    fun draw(
        textureId: Int,
        screenWidth: Int,
        screenHeight: Int,
        radius: Float,
        sharpOffsetX: Float,
        sharpOffsetY: Float,
        threshold: Float,
        strength: Float
    ) {
        if (programId == 0) return
        GLES20.glUseProgram(programId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
        GLES20.glUniform1i(uTextureLoc, 0)
        GLES20.glUniform2f(
            uTexelSizeLoc,
            if (screenWidth > 0) 1f / screenWidth else 1f,
            if (screenHeight > 0) 1f / screenHeight else 1f
        )
        GLES20.glUniform1f(uRadiusLoc, radius)
        GLES20.glUniform2f(uSharpOffsetLoc, sharpOffsetX, sharpOffsetY)
        GLES20.glUniform1f(uRadiusLoc, radius)
        GLES20.glUniform1f(uThresholdLoc, threshold)
        GLES20.glUniform1f(uStrengthLoc, strength)
        GLES20.glUniform3f(uColorLoc, color[0], color[1], color[2])

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
    }

    companion object {
        private const val TAG = "FocusPeakShaderProgram"

        private val QUAD_COORDS = floatArrayOf(
            -1f, -1f,
             1f, -1f,
            -1f,  1f,
             1f,  1f
        )
        private val QUAD_TEX_COORDS = floatArrayOf(
            0f, 0f,
            1f, 0f,
            0f, 1f,
            1f, 1f
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
uniform sampler2D u_texture;
uniform vec2 u_texelSize;
uniform float u_radius;
uniform vec2 u_sharpOffset;
uniform float u_threshold;
uniform float u_strength;
uniform vec3 u_color;

float lum(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

float ringEdge(vec2 uv, vec2 step, out float sumLum) {
    float e = 0.0;
    float s = 0.0;
    float v;
    vec2 off;
    off = vec2(-1.0, -1.0);
    v = lum(texture2D(u_texture, uv + off * step).rgb);
    e = max(e, abs(lum(texture2D(u_texture, uv).rgb) - v));
    s += v;
    off = vec2( 0.0, -1.0);
    v = lum(texture2D(u_texture, uv + off * step).rgb);
    e = max(e, abs(lum(texture2D(u_texture, uv).rgb) - v));
    s += v;
    off = vec2( 1.0, -1.0);
    v = lum(texture2D(u_texture, uv + off * step).rgb);
    e = max(e, abs(lum(texture2D(u_texture, uv).rgb) - v));
    s += v;
    off = vec2(-1.0,  0.0);
    v = lum(texture2D(u_texture, uv + off * step).rgb);
    e = max(e, abs(lum(texture2D(u_texture, uv).rgb) - v));
    s += v;
    off = vec2( 1.0,  0.0);
    v = lum(texture2D(u_texture, uv + off * step).rgb);
    e = max(e, abs(lum(texture2D(u_texture, uv).rgb) - v));
    s += v;
    off = vec2(-1.0,  1.0);
    v = lum(texture2D(u_texture, uv + off * step).rgb);
    e = max(e, abs(lum(texture2D(u_texture, uv).rgb) - v));
    s += v;
    off = vec2( 0.0,  1.0);
    v = lum(texture2D(u_texture, uv + off * step).rgb);
    e = max(e, abs(lum(texture2D(u_texture, uv).rgb) - v));
    s += v;
    off = vec2( 1.0,  1.0);
    v = lum(texture2D(u_texture, uv + off * step).rgb);
    e = max(e, abs(lum(texture2D(u_texture, uv).rgb) - v));
    s += v;
    sumLum = s;
    return e;
}

void main() {
    vec4 color = texture2D(u_texture, v_texCoord);
    // Screen-anchored band: constant pixel thickness, weaker under zoom.
    float sumLong;
    float edgeLong = ringEdge(v_texCoord, u_radius * u_texelSize, sumLong);
    // Source-anchored sharpness: a fixed distance in source pixels, so it keeps
    // firing while the screen gradient is diluted by the zoom.
    float sumShort;
    float edgeShort = ringEdge(v_texCoord, u_sharpOffset, sumShort);
    float relLong = edgeLong / max((lum(texture2D(u_texture, v_texCoord).rgb) + sumLong) / 9.0, 0.08);
    float relShort = edgeShort / max((lum(texture2D(u_texture, v_texCoord).rgb) + sumShort) / 9.0, 0.08);
    // Pure relative contrast: exposure-invariant, blur rejects at ~10-15% contrast,
    // sharp edges fire at ~30%+ regardless of overall brightness.
    float score = max(relLong, relShort);
    float boost = smoothstep(u_threshold * 0.97, u_threshold * 1.03, score);
    color.rgb = mix(color.rgb, u_color, boost * u_strength);
    gl_FragColor = color;
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