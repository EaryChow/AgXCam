package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Stage 5-W - octave-band chroma noise-floor write, run on the S5-C output.
 *
 * S5-C leaves a residual low-frequency chroma mottle shaped by the chain's
 * own noise transfer. This stage writes that floor down without any
 * classification and without hard regions: every kernel is luma-bilateral,
 * so a tap spanning a real luma edge lands at ~zero weight and no chroma
 * bleeds across a clipped or higher-exposure-range area, and every per-band gain is
 * continuous in [G_MIN, 1], so the write strength never steps.
 *
 * Transform T is the S5-C basis, never mixed with the S5-L one:
 *   Y  = 0.25 R + 0.5 G + 0.25 B
 *   C1 = (B - R) / 4
 *   C2 = (R - 2 G + B) / 4
 *   R = Y - 2 C1 + C2,  G = Y - C2,  B = Y + 2 C1 + C2
 *
 * Passes, each one texel per output pixel except GAIN:
 *   level : one sweep, written to both level planes at once (MRT locations 0
 *           and 1). The tap grid is stride-2 - every 2nd tap of the outer
 *           box - accumulating four nested luma-gated weighted means of the
 *           chroma, one per octave radius r0..r3 at half tap density; the
 *           covered output-pixel span is unchanged. Plane 0 carries (L0, L1)
 *           and plane 1 carries (L2, L3), two chroma coordinates each. A
 *           level whose gate left nothing to average keeps the centre
 *           chroma.
 *   gain  : one texel per 8x8 cell. Each band's energy E[k] is the cell mean
 *           of 0.5*|band|^2; its noise level is the propagated chroma sigma
 *           at the cell centre times the band shape fraction, the chain
 *           total-gain calibration and the squared EV PP gain (1.0 at the
 *           +1.5 EV default), and the gain is 1 - (sn/E)^Q clamped to
 *           [G_MIN, 1], exactly 1 where the cell carries no band energy.
 *   blend : rebuilds chroma as L3 plus the four gain-weighted bands and
 *           rebuilds RGB with the untouched centre luma. The gain plane is
 *           interpolated by hand: RGBA32F allocates NEAREST on this context
 *           and ES 3.0 float textures do not filter.
 *
 * The host drives all three (PreviewRenderer.runStage5W); a strength of 0
 * makes the blend a bit-exact identity and the caller skips the stage.
 */
class S5WShaderProgram {

    private var levelProgramId = 0
    private var gainProgramId = 0
    private var blendProgramId = 0
    private var ready = false

    /**
     * The level pass is the expensive one; the gain pass runs at 1/64 the
     * texel count and the blend is a handful of taps, so the budget counts
     * this pass.
     */
    fun budgetProgramId(): Int = levelProgramId

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create() {
        levelProgramId = createProgram(VERTEX_SHADER, LEVEL_FRAGMENT)
        gainProgramId = createProgram(VERTEX_SHADER, GAIN_FRAGMENT)
        blendProgramId = createProgram(VERTEX_SHADER, BLEND_FRAGMENT)
        if (levelProgramId == 0 || gainProgramId == 0 || blendProgramId == 0) {
            Log.e(TAG, "Failed to create Stage-5W chroma floor programs")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create Stage-5W chroma floor programs")
            return
        }
        ready = true
        Log.d(TAG, "Stage-5W programs created: l=$levelProgramId g=$gainProgramId b=$blendProgramId")
        com.agx.camera.CrashLogger.log(
            TAG, "Program created: s5w=($levelProgramId,$gainProgramId,$blendProgramId)"
        )
    }

    fun isReady(): Boolean = ready && levelProgramId != 0

    fun destroy() {
        GLES20.glDeleteProgram(levelProgramId)
        GLES20.glDeleteProgram(gainProgramId)
        GLES20.glDeleteProgram(blendProgramId)
        ready = false
        levelProgramId = 0
        gainProgramId = 0
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
     * Level pass, one sweep into both level planes (MRT). The radii arrive
     * from the window-geometry resolution law in output pixels; the sweep
     * samples every STRIDE-th tap, so the radii are converted to sampled
     * units inside the shader.
     */
    fun drawLevel(
        inTex: Int, sigmaTex: Int, resW: Float, resH: Float,
        r0: Float, r1: Float, r2: Float, r3: Float
    ) {
        if (levelProgramId == 0) return
        GLES20.glUseProgram(levelProgramId)
        setTex(levelProgramId, "u_input_tex", 0, inTex)
        setTex(levelProgramId, "u_sigma_tex", 1, sigmaTex)
        set2F(levelProgramId, "u_res", resW, resH)
        setF(levelProgramId, "u_r0", r0)
        setF(levelProgramId, "u_r1", r1)
        setF(levelProgramId, "u_r2", r2)
        setF(levelProgramId, "u_r3", r3)
        drawQuad(levelProgramId)
    }

    /** Gain pass, one texel per 8x8 cell of the level planes. [evGain2] is
     *  the squared EV PP gain folded into every band's noise target (1.0 at
     *  the +1.5 EV default), the same fold runStage5 applies to its ISO
     *  model. */
    fun drawGain(
        inTex: Int, level01Tex: Int, level23Tex: Int, sigmaTex: Int,
        evGain2: Float, mottleFix: Float = 1f
    ) {
        if (gainProgramId == 0) return
        GLES20.glUseProgram(gainProgramId)
        setTex(gainProgramId, "u_input_tex", 0, inTex)
        setTex(gainProgramId, "u_level01_tex", 1, level01Tex)
        setTex(gainProgramId, "u_level23_tex", 2, level23Tex)
        setTex(gainProgramId, "u_sigma_tex", 3, sigmaTex)
        setF(gainProgramId, "u_ev_gain2", evGain2)
        setF(gainProgramId, "u_mottle_fix", mottleFix)
        drawQuad(gainProgramId)
    }

    /** Final band blend, one texel per output pixel. */
    fun drawBlend(
        inTex: Int, level01Tex: Int, level23Tex: Int, gainTex: Int,
        resW: Float, resH: Float, strength: Float
    ) {
        if (blendProgramId == 0) return
        GLES20.glUseProgram(blendProgramId)
        setTex(blendProgramId, "u_input_tex", 0, inTex)
        setTex(blendProgramId, "u_level01_tex", 1, level01Tex)
        setTex(blendProgramId, "u_level23_tex", 2, level23Tex)
        setTex(blendProgramId, "u_gain_tex", 3, gainTex)
        set2F(blendProgramId, "u_res", resW, resH)
        setF(blendProgramId, "u_strength", strength)
        drawQuad(blendProgramId)
    }

    companion object {
        private const val TAG = "S5WShaderProgram"

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

        private const val LEVEL_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
// MRT pair: location 0 carries (L0, L1), location 1 carries (L2, L3), so a
// single sweep feeds both level planes.
layout(location = 0) out vec4 outLevel01;
layout(location = 1) out vec4 outLevel23;

uniform sampler2D u_input_tex;
uniform sampler2D u_sigma_tex;
uniform vec2 u_res;
// Window-geometry resolution law: four nested octave radii keyed on winScale
// = width / 640 (the same idiom as the S5-C radius law), in output pixels.
// The sweep samples every STRIDE-th tap, so the per-level boxes use ceil(r/2)
// in sampled units and cover the same output-pixel span at half tap density;
// a sampled tap costs one luma fetch and four gated accumulates.
uniform float u_r0;
uniform float u_r1;
uniform float u_r2;
uniform float u_r3;

const int STRIDE = 2;

// Gate soft-weight floor, and the squared luma retention from the S5-C guide
// law (RETAIN^2 = 0.28^2).
const float GATE_SW = 2.0;
const float RETAIN2 = 0.0784;

float luma(vec3 rgb) { return 0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b; }
vec2 chroma(vec3 rgb) { return vec2((rgb.b - rgb.r) * 0.25, (rgb.r - 2.0 * rgb.g + rgb.b) * 0.25); }

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy);
    ivec2 maxT = ivec2(u_res) - ivec2(1);

    vec3 rgbC = texelFetch(u_input_tex, base, 0).rgb;
    float y = luma(rgbC);
    vec2 cc = chroma(rgbC);
    float sigY2 = max(texelFetch(u_sigma_tex, base, 0).r, 0.0);
    float syRet = RETAIN2 * sigY2 + 1.0e-7;

    int ir0 = max(1, int(ceil(u_r0 * 0.5)));
    int ir1 = max(1, int(ceil(u_r1 * 0.5)));
    int ir2 = max(1, int(ceil(u_r2 * 0.5)));
    int R = max(1, int(ceil(u_r3 * 0.5)));

    vec2 acc1 = vec2(0.0);
    vec2 acc2 = vec2(0.0);
    vec2 acc3 = vec2(0.0);
    vec2 acc4 = vec2(0.0);
    float wsum1 = 0.0;
    float wsum2 = 0.0;
    float wsum3 = 0.0;
    float wsum4 = 0.0;
    for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
            ivec2 t = clamp(base + ivec2(dx * STRIDE, dy * STRIDE), ivec2(0), maxT);
            vec3 rgbT = texelFetch(u_input_tex, t, 0).rgb;
            float yT = luma(rgbT);
            // Luma-bilateral gate: a tap on the far side of a real luma edge
            // has (yT - y)^2 >> syRet, so its weight collapses toward zero
            // and no chroma crosses the edge.
            float dyl = yT - y;
            float wg = GATE_SW / (dyl * dyl / syRet + GATE_SW);
            vec2 cT = chroma(rgbT);
            int amax = max(abs(dy), abs(dx));
            if (amax <= ir0) { acc1 += wg * cT; wsum1 += wg; }
            if (amax <= ir1) { acc2 += wg * cT; wsum2 += wg; }
            if (amax <= ir2) { acc3 += wg * cT; wsum3 += wg; }
            if (amax <= R)   { acc4 += wg * cT; wsum4 += wg; }
        }
    }
    // A level with nothing behind the gate keeps the centre chroma, so the
    // band sitting on it is exactly zero.
    vec2 L0 = wsum1 > 1.0e-6 ? acc1 / wsum1 : cc;
    vec2 L1 = wsum2 > 1.0e-6 ? acc2 / wsum2 : cc;
    vec2 L2 = wsum3 > 1.0e-6 ? acc3 / wsum3 : cc;
    vec2 L3 = wsum4 > 1.0e-6 ? acc4 / wsum4 : cc;

    outLevel01 = vec4(L0.x, L0.y, L1.x, L1.y);
    outLevel23 = vec4(L2.x, L2.y, L3.x, L3.y);
}
"""

        private const val GAIN_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_input_tex;
uniform sampler2D u_level01_tex;
uniform sampler2D u_level23_tex;
uniform sampler2D u_sigma_tex;
// EV fold: the picture formation after Stage 5 multiplies the linear scene
// by 2^EV, so the noise target is scaled by the squared gain, anchored at
// the +1.5 EV default (1.0 there) and clamped on the host exactly like the
// S5-L ISO-model fold.
uniform float u_ev_gain2;
// Shaped dark-noise boost on the noise target (the sim's
// shaped_dark_boost); mirrors the S5-C stage-1 weight-variances scale.
uniform float u_mottle_fix;

const int CELL = 8;
const int N = CELL * CELL;

// Per-band noise shape fractions of the propagated chroma marginal; the four
// sum to one. CAL is the chain noise total-gain against that marginal, Q
// shapes the gain curve and G_MIN is the floor that always keeps a share of
// the band.
const float K_BAND0 = 0.8526;
const float K_BAND1 = 0.0918;
const float K_BAND2 = 0.0414;
const float K_BAND3 = 0.0142;
const float CAL = 1.935;
const float Q = 1.5;
const float G_MIN = 0.05;

vec2 chroma(vec3 rgb) { return vec2((rgb.b - rgb.r) * 0.25, (rgb.r - 2.0 * rgb.g + rgb.b) * 0.25); }

void main() {
    ivec2 cell = ivec2(gl_FragCoord.xy);
    ivec2 base = cell * CELL;
    ivec2 size = textureSize(u_input_tex, 0);

    float e0 = 0.0;
    float e1 = 0.0;
    float e2 = 0.0;
    float e3 = 0.0;
    for (int yy = 0; yy < CELL; yy++) {
        for (int xx = 0; xx < CELL; xx++) {
            ivec2 t = clamp(base + ivec2(xx, yy), ivec2(0), size - ivec2(1));
            vec2 c = chroma(texelFetch(u_input_tex, t, 0).rgb);
            vec4 l01 = texelFetch(u_level01_tex, t, 0);
            vec4 l23 = texelFetch(u_level23_tex, t, 0);
            vec2 b0 = c - l01.xy;
            vec2 b1 = l01.xy - l01.zw;
            vec2 b2 = l01.zw - l23.xy;
            vec2 b3 = l23.xy - l23.zw;
            e0 += 0.5 * (b0.x * b0.x + b0.y * b0.y);
            e1 += 0.5 * (b1.x * b1.x + b1.y * b1.y);
            e2 += 0.5 * (b2.x * b2.x + b2.y * b2.y);
            e3 += 0.5 * (b3.x * b3.x + b3.y * b3.y);
        }
    }
    float n = float(N);
    e0 /= n;
    e1 /= n;
    e2 /= n;
    e3 /= n;

    // Noise level per band from the propagated chroma variance at the cell
    // centre; the sigma texture is full res, so its centre texel anchors it.
    ivec2 cbase = clamp(base + ivec2(CELL / 2), ivec2(0), size - ivec2(1));
    vec4 sg = texelFetch(u_sigma_tex, cbase, 0);
    float sigC = max(sg.g + sg.b, 0.0);
    if (u_mottle_fix > 0.5) {
        vec3 px = texelFetch(u_input_tex, cbase, 0).rgb;
        float yy = max(0.25 * px.r + 0.5 * px.g + 0.25 * px.b, 0.0);
        float qq = (max(px.r, max(px.g, px.b)) - min(px.r, min(px.g, px.b)))
            / max(yy, 1.0e-6);
        float gate = 1.0 / (1.0 + pow(qq / 0.6, 4.0));
        sigC *= 1.0 + 11.0 * exp(-yy / 0.2) * gate;
    }
    float sn0 = sigC * K_BAND0 * CAL * u_ev_gain2;
    float sn1 = sigC * K_BAND1 * CAL * u_ev_gain2;
    float sn2 = sigC * K_BAND2 * CAL * u_ev_gain2;
    float sn3 = sigC * K_BAND3 * CAL * u_ev_gain2;

    float g0 = e0 > 1.0e-12 ? clamp(1.0 - pow(sn0 / e0, Q), G_MIN, 1.0) : 1.0;
    float g1 = e1 > 1.0e-12 ? clamp(1.0 - pow(sn1 / e1, Q), G_MIN, 1.0) : 1.0;
    float g2 = e2 > 1.0e-12 ? clamp(1.0 - pow(sn2 / e2, Q), G_MIN, 1.0) : 1.0;
    float g3 = e3 > 1.0e-12 ? clamp(1.0 - pow(sn3 / e3, Q), G_MIN, 1.0) : 1.0;
    outColor = vec4(g0, g1, g2, g3);
}
"""

        private const val BLEND_FRAGMENT = """
#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_input_tex;
uniform sampler2D u_level01_tex;
uniform sampler2D u_level23_tex;
uniform sampler2D u_gain_tex;
uniform vec2 u_res;
uniform float u_strength;

float luma(vec3 rgb) { return 0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b; }
vec2 chroma(vec3 rgb) { return vec2((rgb.b - rgb.r) * 0.25, (rgb.r - 2.0 * rgb.g + rgb.b) * 0.25); }
vec3 rgbOf(float y, vec2 c) { return vec3(y - 2.0 * c.x + c.y, y - c.y, y + 2.0 * c.x + c.y); }

// The gain plane is RGBA32F and this context leaves it NEAREST (ES 3.0 float
// textures do not filter), so the blend interpolates it by hand: four
// texelFetches around the pixel centre with linear weights, edge-clamped the
// way CLAMP_TO_EDGE would land.
vec4 gainAt(vec2 uv) {
    ivec2 gs = textureSize(u_gain_tex, 0);
    vec2 pos = uv * vec2(gs) - 0.5;
    vec2 f = fract(pos);
    ivec2 i0 = ivec2(floor(pos));
    ivec2 ic = clamp(i0, ivec2(0), gs - ivec2(1));
    ivec2 i1 = clamp(i0 + ivec2(1), ivec2(0), gs - ivec2(1));
    vec4 g00 = texelFetch(u_gain_tex, ivec2(ic.x, ic.y), 0);
    vec4 g10 = texelFetch(u_gain_tex, ivec2(i1.x, ic.y), 0);
    vec4 g01 = texelFetch(u_gain_tex, ivec2(ic.x, i1.y), 0);
    vec4 g11 = texelFetch(u_gain_tex, ivec2(i1.x, i1.y), 0);
    return mix(mix(g00, g10, f.x), mix(g01, g11, f.x), f.y);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 rgb = texelFetch(u_input_tex, p, 0).rgb;
    float y = luma(rgb);
    vec2 c = chroma(rgb);
    vec4 l01 = texelFetch(u_level01_tex, p, 0);
    vec4 l23 = texelFetch(u_level23_tex, p, 0);
    vec4 g = gainAt(gl_FragCoord.xy / u_res);
    // Rebuild from the coarsest level up: L3 plus each band kept by its gain.
    vec2 cout = l23.zw
        + g.x * (c - l01.xy)
        + g.y * (l01.xy - l01.zw)
        + g.z * (l01.zw - l23.xy)
        + g.w * (l23.xy - l23.zw);
    cout = mix(c, cout, u_strength);
    outColor = vec4(rgbOf(y, cout), 1.0);
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
