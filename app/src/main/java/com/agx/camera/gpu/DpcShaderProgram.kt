package com.agx.camera.gpu

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Stage 1 — Defect pixel correction (DPC) on the sparse Bayer grid.
 *
 * Four passes, all at demosaic/output resolution (one texel per 2x2 sensor cell,
 * four CFA phases in RGBA — the placeholder's indexing contract, C3):
 *
 *  Pass 0 (AVG)     : α-trimmed mean of the 8 same-color neighbours (5x5 same
 *                     phase lattice, drop 1 largest & 1 smallest) per phase.
 *  Pass 1 (DETECT)  : Condition A (|I−Iavg| > band, band = max(M1·Iavg, θ·σ̂_ISO))
 *                     AND Condition B (center deviation isolated: > M2 × max
 *                     neighbour deviation) → defect flag map (1 hot / −1 cold / 0).
 *  Pass 2 (CORRECT) : flagged texels replaced by feature-direction estimate I_D
 *                     (smooth direction pair), M3=M1 re-check, fallback to
 *                     non-directional I_ND (hot: 2nd largest / cold: 2nd smallest).
 *  Pass 3 (COUPLET) : second couplet-reinforcement pass — only texels whose 8
 *                     neighbour cells contain a flagged pixel re-run correction,
 *                     with flagged neighbours excluded from the candidate set.
 *
 * σ̂ for the absolute threshold comes from the Stage 0 ISO-calibrated model
 * (uniforms u_iso_model_a/b), NOT the Stage 2 patch estimate (plan R2).
 *
 * Bypass: u_dpc_enabled = 0 → pass-through copy of raw (black-subtracted,
 * clamped ≥ 0) → output is byte-identical to the placeholder's strength-0 anchor.
 */
class DpcShaderProgram {

    private var programId = 0

    private var uBayerTexLoc = 0
    private var uAvgTexLoc = 0
    private var uFlagTexLoc = 0
    private var uTransformMatrixLoc = 0
    private var uSensorSizeLoc = 0
    private var uCropOriginLoc = 0
    private var uCropSizeLoc = 0
    private var uViewSizeLoc = 0
    private var uBlackLevelPatternLoc = 0
    private var uBitDepthLoc = 0
    private var uPassLoc = 0
    private var uDpcEnabledLoc = 0
    private var uM1Loc = 0
    private var uM2Loc = 0
    private var uThetaLoc = 0
    private var uIsoModelA = 0
    private var uIsoModelB = 0
    private var uCorrStrengthLoc = 0

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create() {
        programId = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        if (programId == 0) {
            Log.e(TAG, "Failed to create DPC shader program")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create DPC shader program")
            return
        }
        uBayerTexLoc = GLES20.glGetUniformLocation(programId, "u_bayerTex")
        uAvgTexLoc = GLES20.glGetUniformLocation(programId, "u_avgTex")
        uFlagTexLoc = GLES20.glGetUniformLocation(programId, "u_flagTex")
        uTransformMatrixLoc = GLES20.glGetUniformLocation(programId, "u_transformMatrix")
        uSensorSizeLoc = GLES20.glGetUniformLocation(programId, "u_sensorSize")
        uCropOriginLoc = GLES20.glGetUniformLocation(programId, "u_cropOrigin")
        uCropSizeLoc = GLES20.glGetUniformLocation(programId, "u_cropSize")
        uViewSizeLoc = GLES20.glGetUniformLocation(programId, "u_viewSize")
        uBlackLevelPatternLoc = GLES20.glGetUniformLocation(programId, "u_black_level_pattern")
        uBitDepthLoc = GLES20.glGetUniformLocation(programId, "u_bit_depth")
        uPassLoc = GLES20.glGetUniformLocation(programId, "u_pass")
        uDpcEnabledLoc = GLES20.glGetUniformLocation(programId, "u_dpc_enabled")
        uM1Loc = GLES20.glGetUniformLocation(programId, "u_m1")
        uM2Loc = GLES20.glGetUniformLocation(programId, "u_m2")
        uThetaLoc = GLES20.glGetUniformLocation(programId, "u_theta")
        uIsoModelA = GLES20.glGetUniformLocation(programId, "u_iso_model_a")
        uIsoModelB = GLES20.glGetUniformLocation(programId, "u_iso_model_b")
        uCorrStrengthLoc = GLES20.glGetUniformLocation(programId, "u_corr_strength")
        Log.d(TAG, "DPC shader program created: $programId")
        com.agx.camera.CrashLogger.log(TAG, "Program created: dpc=$programId")
    }

    fun draw(
        pass: Int,
        transformMatrix: FloatArray,
        cropOriginX: Float, cropOriginY: Float,
        cropSizeX: Float, cropSizeY: Float,
        viewWidth: Float, viewHeight: Float,
        sensorWidth: Float, sensorHeight: Float,
        bayerTex: Int,
        avgTex: Int,
        flagTex: Int,
        fallbackFloatTex: Int,
        blackLevelPattern: IntArray,
        bitDepth: Int,
        dpcEnabled: Boolean,
        m1: Float, m2: Float, theta: Float,
        isoModelA: Float, isoModelB: Float,
        corrStrength: Float
    ) {
        if (programId == 0) return
        GLES20.glUseProgram(programId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, bayerTex)
        GLES20.glUniform1i(uBayerTexLoc, 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, if (pass == PASS_AVG) fallbackFloatTex else avgTex)
        GLES20.glUniform1i(uAvgTexLoc, 1)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, if (pass != PASS_COUPLET) fallbackFloatTex else flagTex)
        GLES20.glUniform1i(uFlagTexLoc, 2)

        GLES20.glUniformMatrix4fv(uTransformMatrixLoc, 1, false, transformMatrix, 0)
        GLES20.glUniform2f(uSensorSizeLoc, sensorWidth, sensorHeight)
        GLES20.glUniform2f(uCropOriginLoc, cropOriginX, cropOriginY)
        GLES20.glUniform2f(uCropSizeLoc, cropSizeX, cropSizeY)
        GLES20.glUniform2f(uViewSizeLoc, viewWidth, viewHeight)
        GLES20.glUniform4i(uBlackLevelPatternLoc,
            blackLevelPattern[0], blackLevelPattern[1],
            blackLevelPattern[2], blackLevelPattern[3])
        GLES20.glUniform1i(uBitDepthLoc, bitDepth)
        GLES20.glUniform1i(uPassLoc, pass)
        GLES20.glUniform1f(uDpcEnabledLoc, if (dpcEnabled) 1f else 0f)
        GLES20.glUniform1f(uM1Loc, m1)
        GLES20.glUniform1f(uM2Loc, m2)
        GLES20.glUniform1f(uThetaLoc, theta)
        GLES20.glUniform1f(uIsoModelA, isoModelA)
        GLES20.glUniform1f(uIsoModelB, isoModelB)
        GLES20.glUniform1f(uCorrStrengthLoc, corrStrength)

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

    fun rawProgramId(): Int = programId

    fun destroy() {
        if (programId != 0) {
            GLES20.glDeleteProgram(programId)
            programId = 0
        }
    }

    companion object {
        const val PASS_AVG = 0
        const val PASS_DETECT = 1
        const val PASS_CORRECT = 2
        const val PASS_COUPLET = 3

        private const val TAG = "DpcShaderProgram"

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
precision highp usampler2D;
precision highp int;

in vec2 v_texCoord;
out vec4 outColor;

uniform usampler2D u_bayerTex;
uniform sampler2D u_avgTex;
uniform sampler2D u_flagTex;
uniform vec2 u_sensorSize;
uniform vec2 u_cropOrigin;
uniform vec2 u_cropSize;
uniform vec2 u_viewSize;
uniform ivec4 u_black_level_pattern;
uniform int u_bit_depth;
uniform int u_pass;
uniform float u_dpc_enabled;
uniform float u_m1;
uniform float u_m2;
uniform float u_theta;
uniform float u_iso_model_a;
uniform float u_iso_model_b;
uniform float u_corr_strength;

${DenoiseGlsl.MOSAIC_HELPERS}

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

// black-subtracted signed sample at an arbitrary sensor position
float bayerAt(ivec2 coord) {
    ivec2 clamped = clamp(coord, ivec2(0), ivec2(u_sensorSize) - ivec2(1));
    uint raw = texelFetch(u_bayerTex, clamped, 0).r;
    float val = unpackRaw(raw);
    int ph = abs(clamped.x % 2) + abs(clamped.y % 2) * 2;
    return val - float(u_black_level_pattern[ph]);
}

// Iavg of a *cell* covering sensor position j, phase p
float avgAt(ivec2 sensorPos, int p) {
    ivec2 t = cellTexelFor(sensorPos);
    vec4 av = texelFetch(u_avgTex, t, 0);
    if (p == 0) return av.r;
    if (p == 1) return av.g;
    if (p == 2) return av.b;
    return av.a;
}

float flagAt(ivec2 sensorPos, int p) {
    ivec2 t = cellTexelFor(sensorPos);
    vec4 f = texelFetch(u_flagTex, t, 0);
    if (p == 0) return f.r;
    if (p == 1) return f.g;
    if (p == 2) return f.b;
    return f.a;
}

// 8 same-color neighbour offsets (5x5 lattice around a same-phase pixel)
const int NDX[8] = int[8]( 2, -2,  0,  0,  2, -2,  2, -2);
const int NDY[8] = int[8]( 0,  0,  2, -2,  2,  2, -2, -2);

// Condition B: the center deviation must exceed M2 × every neighbour's deviation
bool codeCheck(float devMag, ivec2 cc, int p) {
    float maxNb = 0.0;
    for (int k = 0; k < 8; k++) {
        ivec2 n = cc + ivec2(NDX[k], NDY[k]);
        float nbI = bayerAt(n);
        float nbAvg = avgAt(n, p);
        maxNb = max(maxNb, abs(nbI - nbAvg));
    }
    return devMag > u_m2 * maxNb;
}

void main() {
    ivec2 sc = cellOrigin();
    int parityX = abs(sc.x % 2);
    int parityY = abs(sc.y % 2);

    if (u_pass == 0) {
        // Pass A: α-trimmed mean of the 8 same-colour neighbours (drop 1 max, 1 min)
        vec4 result = vec4(0.0);
        for (int p = 0; p < 4; p++) {
            int phaseX = p & 1;
            int phaseY = p >> 1;
            ivec2 cc = sc + ivec2(parityX ^ phaseX, parityY ^ phaseY);
            float vals[8];
            for (int k = 0; k < 8; k++) {
                vals[k] = bayerAt(cc + ivec2(NDX[k], NDY[k]));
            }
            // sort ascending (8 elements, insertion)
            for (int i = 1; i < 8; i++) {
                float v = vals[i];
                int j = i - 1;
                while (j >= 0 && vals[j] > v) {
                    vals[j + 1] = vals[j];
                    j--;
                }
                vals[j + 1] = v;
            }
            float sum = 0.0;
            for (int k = 1; k < 7; k++) {
                sum += vals[k];
            }
            if (p == 0) result.r = sum / 6.0;
            else if (p == 1) result.g = sum / 6.0;
            else if (p == 2) result.b = sum / 6.0;
            else result.a = sum / 6.0;
        }
        outColor = result;
        return;
    }

    if (u_pass == 1) {
        // Pass B: detection → defect flags
        vec4 flags = vec4(0.0);
        if (u_dpc_enabled > 0.5) {
            for (int p = 0; p < 4; p++) {
                int phaseX = p & 1;
                int phaseY = p >> 1;
                ivec2 cc = sc + ivec2(parityX ^ phaseX, parityY ^ phaseY);
                float I = bayerAt(cc);
                float Iavg = avgAt(cc, p);
                float sigma = isoModelSigma(Iavg);
                float band = max(u_m1 * max(Iavg, 0.0), u_theta * sigma);

                float devC = I - Iavg;
                if (abs(devC) > band) {
                    // isolation check: center deviation must be M2× every neighbour deviation
                    float maxNb = 0.0;
                    for (int k = 0; k < 8; k++) {
                        ivec2 n = cc + ivec2(NDX[k], NDY[k]);
                        float nbI = bayerAt(n);
                        float nbAvg = avgAt(n, p);
                        maxNb = max(maxNb, abs(nbI - nbAvg));
                    }
                    if (abs(devC) > u_m2 * maxNb) {
                        float flag = (devC > 0.0) ? 1.0 : -1.0;
                        if (p == 0) flags.r = flag;
                        else if (p == 1) flags.g = flag;
                        else if (p == 2) flags.b = flag;
                        else flags.a = flag;
                    }
                }
            }
        }
        outColor = flags;
        return;
    }

    // Pass C (correct) / Pass 3 (couplet reinforce)
    vec4 result = vec4(0.0);
    bool couplet = (u_pass == 3);
    bool anyNeighbourFlagged = false;
    if (couplet) {
        for (int p = 0; p < 4 && !anyNeighbourFlagged; p++) {
            int phaseX = p & 1;
            int phaseY = p >> 1;
            ivec2 cc = sc + ivec2(parityX ^ phaseX, parityY ^ phaseY);
            for (int k = 0; k < 8; k++) {
                if (flagAt(cc + ivec2(NDX[k], NDY[k]), p) != 0.0) {
                    anyNeighbourFlagged = true;
                    break;
                }
            }
        }
    }

    for (int p = 0; p < 4; p++) {
        int phaseX = p & 1;
        int phaseY = p >> 1;
        ivec2 cc = sc + ivec2(parityX ^ phaseX, parityY ^ phaseY);
        float I = bayerAt(cc);
        float Iavg = avgAt(cc, p);
        float sigma = isoModelSigma(Iavg);
        float band = max(u_m1 * max(Iavg, 0.0), u_theta * sigma);

        // candidate neighbour values for this phase
        float cands[8];
        for (int k = 0; k < 8; k++) {
            cands[k] = bayerAt(cc + ivec2(NDX[k], NDY[k]));
        }
        if (couplet) {
            // exclude marked neighbours from the candidate set
            for (int k = 0; k < 8; k++) {
                float f = flagAt(cc + ivec2(NDX[k], NDY[k]), p);
                if (f != 0.0) cands[k] = -1.0e9; // poison
            }
        }

        float outv = max(I, 0.0);
        bool needPassThrough = !(u_dpc_enabled > 0.5) ||
            (couplet && !anyNeighbourFlagged);

        if (needPassThrough) {
            // raw pass-through (also the couplet no-op path)
        } else {
            float devC = I - Iavg;
            bool hot = devC > 0.0 && codeCheck(devC, cc, p);
            bool cold = devC < 0.0 && codeCheck(-devC, cc, p);
            if (hot || cold) {
                // direction pair estimates (E,W),(N,S),(NE,SW),(NW,SE)
                float best = 1.0e30;
                float bestAvg = 0.0;
                // use index sums to pair up the NDX/NDY tables:
                // pairA(k) = k, pairB(k) = 8 - 1 - k  → (0,7)=(E,NW) no.
                // explicit pairing below.
                float e = cands[0]; float w = cands[1];
                float n = cands[2]; float s = cands[3];
                float ne = cands[4]; float sw = cands[5];
                float se = cands[6]; float nw = cands[7];

                float pairs[8];
                pairs[0] = e; pairs[1] = w; pairs[2] = n; pairs[3] = s;
                pairs[4] = ne; pairs[5] = sw; pairs[6] = se; pairs[7] = nw;
                // average per direction
                float aH = (e + w) * 0.5;
                float aV = (n + s) * 0.5;
                float a45 = (ne + sw) * 0.5;
                float a135 = (se + nw) * 0.5;
                float dH = abs(e - w);
                float dV = abs(n - s);
                float d45 = abs(ne - sw);
                float d135 = abs(se - nw);

                // feature direction: smoothest pair (min difference), then fallback
                float id = aH;
                if (dV < dH && dV <= d45 && dV <= d135) id = aV;
                else if (d45 < dH && d45 <= dV && d45 <= d135) id = a45;
                else if (d135 < dH && d135 <= dV && d135 <= d45) id = a135;

                // if the direction value is out of range (poisoned partner), fall back
                if (couplet) {
                    bool poisoned = (abs(e) > 1.0e8) || (abs(w) > 1.0e8) ||
                        (abs(n) > 1.0e8) || (abs(s) > 1.0e8) ||
                        (abs(ne) > 1.0e8) || (abs(sw) > 1.0e8) ||
                        (abs(se) > 1.0e8) || (abs(nw) > 1.0e8);
                    if (poisoned) {
                        // rebuild from clean candidates
                        float c2nd = 0.0;
                        int cnt = 0;
                        float lo1 = 1.0e30, lo2 = 1.0e30;
                        float hi1 = -1.0e30, hi2 = -1.0e30;
                        for (int k = 0; k < 8; k++) {
                            if (abs(cands[k]) > 1.0e8) continue;
                            cnt++;
                            if (cands[k] < lo1) { lo2 = lo1; lo1 = cands[k]; }
                            else if (cands[k] < lo2) { lo2 = cands[k]; }
                            if (cands[k] > hi1) { hi2 = hi1; hi1 = cands[k]; }
                            else if (cands[k] > hi2) { hi2 = cands[k]; }
                        }
                        if (cnt >= 6) {
                            c2nd = hot ? hi2 : lo2;
                            outv = max(c2nd, 0.0);
                            if (p == 0) result.r = outv;
                            else if (p == 1) result.g = outv;
                            else if (p == 2) result.b = outv;
                            else result.a = outv;
                            continue;
                        }
                    }
                }

                // M3 = M1 re-check on the direction estimate
                float m3 = u_m1;
                if (abs(id - Iavg) <= max(m3 * max(Iavg, 0.0), u_theta * sigma)) {
                    outv = max(id, 0.0);
                } else {
                    // fallback I_ND: hot → 2nd largest clean candidate, cold → 2nd smallest
                    float lo1 = 1.0e30, lo2 = 1.0e30;
                    float hi1 = -1.0e30, hi2 = -1.0e30;
                    for (int k = 0; k < 8; k++) {
                        if (abs(cands[k]) > 1.0e8) continue;
                        if (cands[k] < lo1) { lo2 = lo1; lo1 = cands[k]; }
                        else if (cands[k] < lo2) { lo2 = cands[k]; }
                        if (cands[k] > hi1) { hi2 = hi1; hi1 = cands[k]; }
                        else if (cands[k] > hi2) { hi2 = cands[k]; }
                    }
                    outv = hot ? hi2 : lo2;
                    outv = max(outv, 0.0);
                }
            }
        }

        // strength blend (0 at bypass → raw)
        if (p == 0) result.r = mix(max(I, 0.0), outv, u_corr_strength);
        else if (p == 1) result.g = mix(max(I, 0.0), outv, u_corr_strength);
        else if (p == 2) result.b = mix(max(I, 0.0), outv, u_corr_strength);
        else result.a = mix(max(I, 0.0), outv, u_corr_strength);
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