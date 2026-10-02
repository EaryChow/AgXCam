package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Shared structure map, first GPU version.
 *
 * Runs after S5 and before the AgX pass, on the S5 output. It is a diagnostic
 * pass in this phase: nothing consumes its output, so the main path is
 * bit-identical whether or not it runs. What it produces per texel:
 *
 *   R = SNR ratio, residual energy / (sigma^2 * r_retain)
 *   G = a_lum, the luminance adaptation factor from formed-picture luminance
 *   B = class index (0 PLAIN / 1 TEXTURE / 2 EDGE)
 *   A = gradient direction coherence in 0..1
 *
 * Formed-picture luminance comes from the 1D distilled AgX curve, passed in
 * as a small uniform table: running full AgX per texel to obtain the toe /
 * mid-pivot / shoulder band would cost more than the pass itself, and the
 * band only has to be approximately right.
 */
class StructureMapShaderProgram {

    private var programId = 0

    /** Exposed only for the shader budget table in the measurement report. */
    fun budgetProgramId(): Int = programId

    private var uInputTexLoc = 0
    private var uSigmaTexLoc = 0
    private var uSigmaSizeLoc = 0
    private var uOutputSizeLoc = 0
    private var uRRetainLoc = 0
    private var uPlainMaxLoc = 0
    private var uTextureMinLoc = 0
    private var uEdgeCoherenceMinLoc = 0
    private var uOrganizationMinLoc = 0
    private var uLutLoc = 0
    private var uLutDomainLoc = 0
    private var uLutEntriesLoc = 0
    private var uMidPivotLoc = 0
    private var uWindowRadiusLoc = 0

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create() {
        programId = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        if (programId == 0) {
            Log.e(TAG, "Failed to create structure map shader program")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create structure map shader program")
            return
        }
        uInputTexLoc = GLES20.glGetUniformLocation(programId, "u_input_tex")
        uSigmaTexLoc = GLES20.glGetUniformLocation(programId, "u_sigma_tex")
        uSigmaSizeLoc = GLES20.glGetUniformLocation(programId, "u_sigma_size")
        uOutputSizeLoc = GLES20.glGetUniformLocation(programId, "u_output_size")
        uRRetainLoc = GLES20.glGetUniformLocation(programId, "u_r_retain")
        uPlainMaxLoc = GLES20.glGetUniformLocation(programId, "u_plain_max")
        uTextureMinLoc = GLES20.glGetUniformLocation(programId, "u_texture_min")
        uEdgeCoherenceMinLoc = GLES20.glGetUniformLocation(programId, "u_edge_coherence_min")
        uOrganizationMinLoc = GLES20.glGetUniformLocation(programId, "u_organization_min")
        uLutLoc = GLES20.glGetUniformLocation(programId, "u_formed_lut")
        uLutDomainLoc = GLES20.glGetUniformLocation(programId, "u_lut_domain")
        uLutEntriesLoc = GLES20.glGetUniformLocation(programId, "u_lut_entries")
        uMidPivotLoc = GLES20.glGetUniformLocation(programId, "u_mid_pivot")
        uWindowRadiusLoc = GLES20.glGetUniformLocation(programId, "u_window_radius")
        Log.d(TAG, "Structure map shader program created: $programId")
        com.agx.camera.CrashLogger.log(TAG, "Program created: structureMap=$programId")
    }

    /**
     * `formedLut` is the distilled AgX table (see FormedPictureLut); it is
     * uploaded as a 1D strip so the shader can look it up by log2 of the
     * linear input.
     */
    fun draw(
        inputTex: Int,
        sigmaTex: Int,
        sigmaW: Int, sigmaH: Int,
        outW: Int, outH: Int,
        rRetain: Float,
        plainMax: Float,
        textureMin: Float,
        edgeCoherenceMin: Float,
        formedLut: FloatArray,
        lutDomainMinStop: Float,
        lutDomainMaxStop: Float,
        midPivot: Float = 0.45f,
        windowRadius: Int = 1
    ) {
        if (programId == 0) return
        if (formedLut.size > FormedPictureLut.MAX_ENTRIES) {
            // Uploading more entries than the uniform was declared with is a
            // GL error and would leave the tail of the table undefined on the
            // GPU, so it is refused rather than truncated.
            val msg = "formed LUT has ${formedLut.size} entries, shader allows ${FormedPictureLut.MAX_ENTRIES}"
            Log.e(TAG, msg)
            com.agx.camera.CrashLogger.log(TAG, msg)
            return
        }
        GLES20.glUseProgram(programId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inputTex)
        GLES20.glUniform1i(uInputTexLoc, 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, sigmaTex)
        GLES20.glUniform1i(uSigmaTexLoc, 1)

        GLES20.glUniform2f(uSigmaSizeLoc, sigmaW.toFloat(), sigmaH.toFloat())
        GLES20.glUniform2f(uOutputSizeLoc, outW.toFloat(), outH.toFloat())
        GLES20.glUniform1f(uRRetainLoc, rRetain)
        GLES20.glUniform1f(uPlainMaxLoc, plainMax)
        GLES20.glUniform1f(uTextureMinLoc, textureMin)
        GLES20.glUniform1f(uEdgeCoherenceMinLoc, edgeCoherenceMin)
        GLES20.glUniform1f(
            uOrganizationMinLoc,
            StructureMapClassifier.ORGANIZATION_MIN
        )
        GLES20.glUniform1f(uMidPivotLoc, midPivot)
        GLES20.glUniform1i(uWindowRadiusLoc, windowRadius)

        val lutBuf = ByteBuffer.allocateDirect(formedLut.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer().put(formedLut)
        lutBuf.position(0)
        GLES20.glUniform1fv(uLutLoc, formedLut.size, lutBuf)
        GLES20.glUniform2f(uLutDomainLoc, lutDomainMinStop, lutDomainMaxStop)
        GLES20.glUniform1i(uLutEntriesLoc, formedLut.size)

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

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
    }

    fun isReady(): Boolean = programId != 0

    fun destroy() {
        if (programId != 0) {
            GLES20.glDeleteProgram(programId)
            programId = 0
        }
    }

    companion object {
        private const val TAG = "StructureMapShaderProgram"

        private val QUAD_COORDS = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        private val QUAD_TEX_COORDS = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f)

        private const val VERTEX_SHADER = """
#version 300 es
in vec2 a_position;
in vec2 a_texCoord;
out vec2 v_texCoord;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
}
"""

        private const val FRAGMENT_SHADER = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
// outColor = (ratio, lumAdapt, class, organization)
out vec4 outColor;

uniform sampler2D u_input_tex;
uniform sampler2D u_sigma_tex;
uniform vec2 u_sigma_size;
uniform vec2 u_output_size;
uniform float u_r_retain;
uniform float u_plain_max;
uniform float u_texture_min;
uniform float u_edge_coherence_min;
uniform float u_organization_min;
uniform float u_formed_lut[${FormedPictureLut.MAX_ENTRIES}];
uniform vec2 u_lut_domain;
uniform int u_lut_entries;
uniform float u_mid_pivot;
uniform int u_window_radius;

const vec3 LUMA = vec3(0.25, 0.5, 0.25);

// Distilled AgX curve lookup: the S5 output is linear camera-native, and the
// luminance band has to be judged on the formed picture, so the input is
// mapped through log2 and looked up in the precomputed 1D table.
float formedLuminance(vec3 linear) {
    float y = max(dot(linear, LUMA), 1.0e-6);
    float stop = log2(y);
    if (stop <= u_lut_domain.x) return u_formed_lut[0];
    if (stop >= u_lut_domain.y) return u_formed_lut[u_lut_entries - 1];
    float t = (stop - u_lut_domain.x) / (u_lut_domain.y - u_lut_domain.x);
    float pos = t * float(u_lut_entries - 1);
    int i0 = int(floor(pos));
    int i1 = min(i0 + 1, u_lut_entries - 1);
    return mix(u_formed_lut[i0], u_formed_lut[i1], pos - float(i0));
}

// Zhang luminance adaptation, four constants pending recalibration for this
// AgX curve. Mirrors StructureMapClassifier.aLum.
float aLum(float c) {
    float gn = u_mid_pivot;
    if (c <= gn * 0.5) {
        float t = max(1.0 - 2.0 * c / gn, 0.0);
        return 1.0 + 2.0 * pow(t, 3.0);
    }
    float t = max(2.0 * c / gn - 1.0, 0.0);
    return 1.0 + 0.8 * pow(t, 2.0);
}

vec3 lumaSpace(vec3 c) {
    return vec3(dot(c, LUMA), c.r - dot(c, LUMA), c.b - dot(c, LUMA));
}

vec2 gradDir(vec2 p) {
    float l = lumaSpace(texture(u_input_tex, (p + vec2(-1.0, 0.0)) / u_output_size).rgb).x;
    float r = lumaSpace(texture(u_input_tex, (p + vec2(1.0, 0.0)) / u_output_size).rgb).x;
    float d = lumaSpace(texture(u_input_tex, (p + vec2(0.0, -1.0)) / u_output_size).rgb).x;
    float u = lumaSpace(texture(u_input_tex, (p + vec2(0.0, 1.0)) / u_output_size).rgb).x;
    return vec2(r - l, u - d);
}

// Mean residual over the window at p, divided by the sigma prediction there.
// Extracted so the organization measure below can evaluate a neighbour's ratio
// the same way this texel's ratio is evaluated.
float localRatio(vec2 p) {
    vec3 c = texture(u_input_tex, p / u_output_size).rgb;
    float centerL = lumaSpace(c).x;
    float residual = 0.0;
    int n = 0;
    int r = max(u_window_radius, 1);
    for (int dy = -1; dy <= 1; dy++) {
        if (dy < -r || dy > r) continue;
        for (int dx = -1; dx <= 1; dx++) {
            if (dx < -r || dx > r) continue;
            vec2 q = clamp(p + vec2(float(dx), float(dy)), vec2(0.0), u_output_size - vec2(1.0));
            float l = lumaSpace(texture(u_input_tex, q / u_output_size).rgb).x;
            float diff = l - centerL;
            residual += diff * diff;
            n++;
        }
    }
    float meanResidual = n > 0 ? residual / float(n) : 0.0;
    vec2 gridUv = (p + 0.5) / u_sigma_size;
    float sigma2 = texture(u_sigma_tex, gridUv).r;
    return meanResidual / max(sigma2 * u_r_retain, 1.0e-9);
}

// Spatial continuity of the above-threshold neighbours in 0..1. Mirrors what
// StructureMapClassifier.classify expects for `organization`: scattered
// single-texel spikes read near 0, a connected textured patch reads near 1.
float organizationAt(vec2 p) {
    int above = 0;
    int m = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            vec2 q = clamp(p + vec2(float(dx), float(dy)), vec2(0.0), u_output_size - vec2(1.0));
            if (localRatio(q) >= u_texture_min) above++;
            m++;
        }
    }
    return m > 0 ? float(above) / float(m) : 0.0;
}

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);
    vec2 px = vec2(base);
    vec3 center = texture(u_input_tex, v_texCoord).rgb;
    float centerL = lumaSpace(center).x;

    // Residual energy over a small window, in the same linear domain the S5
    // epsilon works in.
    float ratio = localRatio(px);

    // Gradient direction coherence: how tightly the neighbouring gradients
    // line up. Pure noise gives an incoherent average; a stroke does not.
    vec2 g = gradDir(px);
    float gmag = length(g);
    float coherence = 0.0;
    if (gmag > 1.0e-8) {
        vec2 dir = g / gmag;
        float acc = 0.0;
        int m = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                vec2 q = clamp(px + vec2(float(dx), float(dy)), vec2(0.0), u_output_size - vec2(1.0));
                vec2 ng = gradDir(q);
                float nm = length(ng);
                if (nm <= 1.0e-8) continue;
                acc += dot(dir, ng / nm);
                m++;
            }
        }
        coherence = m > 0 ? max(acc / float(m), 0.0) : 0.0;
    }

    float formed = formedLuminance(center);
    float lumAdapt = aLum(formed);

    int cls;
    float organization = organizationAt(px);
    if (coherence >= u_edge_coherence_min) {
        cls = 2;
    } else if (ratio >= u_texture_min && organization >= u_organization_min) {
        cls = 1;
    } else if (ratio <= u_plain_max) {
        cls = 0;
    } else {
        // Between the bands the pass has no history to lean on (temporal
        // hysteresis lives on the host, in StructureMapStabilizer), so the
        // conservative reading is PLAIN.
        cls = 0;
    }

    outColor = vec4(ratio, lumAdapt, float(cls), organization);
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