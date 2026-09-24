package com.agx.camera.gpu

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.PI
import kotlin.random.Random
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

// Mirrored GLSL + host constants (OutNrShaderProgram / PreviewRenderer.runStage5).
private val WINDOW_CENTERS = arrayOf(
    intArrayOf(-2, 0), intArrayOf(2, 0), intArrayOf(0, -2), intArrayOf(0, 2),
    intArrayOf(-2, -2), intArrayOf(2, -2), intArrayOf(-2, 2), intArrayOf(2, 2)
)
private const val WY = 0.25f
private const val WG = 0.5f
private const val WB = 0.25f

// Host defaults (PreviewRenderer.runStage5) - the calibration targets.
private const val LUMA_EPS_SCALE = 1.4f
private const val CHROMA_EPS_SCALE = 64.0f
private const val BETA = 0.3f
private const val SIGMA_DM2 = 10f
private const val SIGMA_SCALE = 1f / 32f
private const val ROUND2_EPS_MULT = 1.96f // (kappa x 1.4)^2, epsilon proportional to sigma_hat^2

/**
 * Pure-noise-block anchor calibration for the Stage-5 double pass
 * plus the double-pass regression guarantees (requirements).
 *
 * This is a headless JVM mirror of the Stage-5 GLSL (OutNrShaderProgram
 * STATS_H/STATS_V/MAIN_FRAGMENT) - the same approach as S5ReproTest - extended
 * to (a) run BOTH SWGF iterations with beta noise-return, (b) control every host
 * parameter (lumaEpsScale/chromaEpsScale, round-2 epsilon multiplier, beta, winScale,
 * epsBoost, evGain2 fold, ISO model A/B), and (c) read out noise attenuation on
 * flat pure-noise blocks.
 *
 * Two measurement families (Stage 5 and the double pass):
 *  1. ANCHOR identity: on a flat block whose variance sigma^2 is known exactly,
 *     set epsilon = kappa^2*sigma^2 and check measured single-pass attenuation against the
 *     theory table  a = 1/(1+kappa^2), att = sigma*sqrt(a^2+(1-a^2)/|omega|), |omega|=25 -
 *     at kappa = 1.0 / 1.2 / 1.5, plus the round-2 kappa x 1.4 tier and the two-round
 *     cascade product (the theoretical ideal lower bound), with and without the
 *     beta=0.3 noise return.
 *  2. PIPELINE config: real host epsilon math (ISO model * evGain2 *
 *     sigmaScale * epsBoost * (sigma_hat^2+sigma_dm^2)/whiteRange^2) run on flat blocks at
 *     representative sigma truth tiers, across ISO x signal x path (preview,
 *     capture 1:1 boxAA-off, capture reduced boxAA-on, EV-comp low end).
 *     Reports the effective kappa = sqrt(epsilon/sigma^2_truth) and the measured cascade
 *     attenuation so the model->truth ratio is visible per tier.
 *
 * Requirements:
 *  (a) highlight-texture/fine-edge preservation is not over-smoothed by the
 *      second pass (grating-modulation retention + edge rise-width floors);
 *  (b) the two-round cascade attenuation is measured and tabled by
 *      kappa = 1.0/1.2/1.5 including the beta-return's actual effect;
 *  (c) slider strength 0 stays bit-exact bypass (host returns the input);
 *  (d) S5ReproTest's bigSeams==0 guarantee carries over to the double pass.
 *
 * Note on "measured": all numbers here are computed with the faithful JVM
 * mirror (deterministic fixed seeds), i.e. the offline digital-calibration
 * leg; on-device flat-field confirmation is follow-up work.
 */
class S5NoiseAnchorCalibTest {

    // ------------------------------------------------------------------
    // Mirrored GLSL constants (OutNrShaderProgram).
    // ------------------------------------------------------------------

    private data class Pln(val w: Int, val h: Int, val c: Array<FloatArray>) {
        fun at(x: Int, y: Int, ch: Int): Float {
            val cx = x.coerceIn(0, w - 1)
            val cy = y.coerceIn(0, h - 1)
            return c[cy][cx * 3 + ch]
        }
    }

    /** One Stage-5 pipeline configuration (defaults = live preview, ISO 800, EV+1.5). */
    private data class Cfg(
        val beta: Float = BETA,
        val lumaEpsScale: Float = LUMA_EPS_SCALE,
        val chromaEpsScale: Float = CHROMA_EPS_SCALE,
        val sigmaDm2: Float = SIGMA_DM2,
        val sigmaScale: Float = SIGMA_SCALE,
        val whiteRange: Float = 959f, // preview effWhite-effBlack (1023-64)
        val winScale: Float = 1f,
        val epsBoost: Float = 1f,
        val iso: Int = 800,
        val evGain2: Float = 1f,
        val round2EpsMult: Float = ROUND2_EPS_MULT,
        val iterations: Int = 2,
        val strength: Float = 1f
    ) {
        val inverseRange2: Float = 1f / (whiteRange * whiteRange)
        val isoModelA: Float = 0.0067f * iso.coerceAtLeast(1) / 100f
        val isoModelB: Float = {
            val g = 0.33f * iso.coerceAtLeast(1) / 100f
            g * g
        }()
    }

    private fun segmentOf(iso: Int): String = when {
        iso <= 1600 -> "measured"
        iso <= 6400 -> "interpolated"
        else -> "assumed"
    }

    private fun lumaOf(r: Float, g: Float, b: Float): Float = WY * r + WG * g + WB * b
    private fun mix(a: Float, b: Float, t: Float): Float = a + (b - a) * t

    private fun yccOf(r: Float, g: Float, b: Float): FloatArray =
        floatArrayOf(lumaOf(r, g, b), r - b, 0.5f * (r + b) - g)

    private fun rgbOf(y: Float, c1: Float, c2: Float): FloatArray =
        floatArrayOf(y + 0.5f * c1 + 0.5f * c2, y - 0.5f * c2, y - 0.5f * c1 + 0.5f * c2)

    // ------------------------------------------------------------------
    // Scene generators.
    // ------------------------------------------------------------------

    private fun gauss(rnd: Random): Float {
        val u1 = rnd.nextFloat().coerceAtLeast(1e-7f)
        val u2 = rnd.nextFloat()
        return sqrt(-2f * kotlin.math.ln(u1)) * cos(2f * PI.toFloat() * u2)
    }

    /** Flat-field block, neutral RGB, per-channel IID Gaussian noise sigma_px. */
    private fun noiseBlock(size: Int, signal: Float, sigmaPx: Float, seed: Long): Pln {
        val rnd = Random(seed)
        val c = Array(size) { FloatArray(size * 3) }
        for (y in 0 until size) {
            for (x in 0 until size) {
                val n0 = gauss(rnd) * sigmaPx
                val n1 = gauss(rnd) * sigmaPx
                val n2 = gauss(rnd) * sigmaPx
                c[y][x * 3 + 0] = signal + n0
                c[y][x * 3 + 1] = signal + n1
                c[y][x * 3 + 2] = signal + n2
            }
        }
        return Pln(size, size, c)
    }

    /**
     * Scene for the texture-preservation regression: three vertical bands
     * (shadow / mid / highlight). Within mid and highlight, a coarse grating
     * (period Pc=16, amp Ac) and a fine grating (Pf=6, amp Af) region, plus a
     * hard step edge between adjacent bands. Gaussian noise added per channel.
     */
    private fun textureScene(
        size: Int, bandXs: IntArray, sigmaPx: Float,
        pc: Int, ac: Float, pf: Int, af: Float, seed: Long
    ): Pln {
        val rnd = Random(seed)
        val c = Array(size) { FloatArray(size * 3) }
        for (y in 0 until size) {
            for (x in 0 until size) {
                var signal: Float
                val band = bandXs.indexOfLast { x >= it }
                when {
                    band <= 0 -> signal = 0.2f
                    band == 1 -> {
                        signal = 0.5f
                        if (y < size / 2) signal += ac * sin(2f * PI.toFloat() * x / pc)
                        else signal += af * sin(2f * PI.toFloat() * x / pf)
                    }
                    else -> {
                        signal = 0.83f
                        if (y < size / 2) signal += ac * sin(2f * PI.toFloat() * x / pc)
                        else signal += af * sin(2f * PI.toFloat() * x / pf)
                    }
                }
                val n0 = gauss(rnd) * sigmaPx
                val n1 = gauss(rnd) * sigmaPx
                val n2 = gauss(rnd) * sigmaPx
                c[y][x * 3 + 0] = signal + n0
                c[y][x * 3 + 1] = signal + n1
                c[y][x * 3 + 2] = signal + n2
            }
        }
        return Pln(size, size, c)
    }

    /** Disc-on-field scene (S5ReproTest style) for the double-pass seam check. */
    private fun discScene(size: Int, sigmaPx: Float, seed: Long): Triple<Pln, Int, Int> {
        val rnd = Random(seed)
        val cx = size / 2
        val cy = size / 2
        val cr = (size * 0.22f).toInt()
        val c = Array(size) { FloatArray(size * 3) }
        for (y in 0 until size) {
            for (x in 0 until size) {
                val dx = x - cx
                val dy = y - cy
                val inside = dx * dx + dy * dy <= cr * cr
                val base = if (inside) floatArrayOf(0.98f, 0.94f, 0.88f)
                    else floatArrayOf(0.25f, 0.25f, 0.27f)
                val n0 = gauss(rnd) * sigmaPx
                val n1 = gauss(rnd) * sigmaPx
                val n2 = gauss(rnd) * sigmaPx
                c[y][x * 3 + 0] = base[0] + n0
                c[y][x * 3 + 1] = base[1] + n1
                c[y][x * 3 + 2] = base[2] + n2
            }
        }
        return Triple(Pln(size, size, c), cx, cy)
    }

    // ------------------------------------------------------------------
    // Faithful SWGF mirror (one full iteration).
    // ------------------------------------------------------------------

    /** Separable 5-tap (win-scaled) box stats of the beta-composed luma: (mean, 2nd moment). */
    private fun statsOf(inImg: Pln, baseImg: Pln, beta: Float, winScale: Float): Pair<Array<FloatArray>, Array<FloatArray>> {
        val w = inImg.w
        val h = inImg.h
        val r = max(2, (2.0 * winScale).roundToInt())
        val hMean = Array(h) { FloatArray(w) }
        val hSq = Array(h) { FloatArray(w) }
        for (y in 0 until h) {
            for (x in 0 until w) {
                var s = 0f
                var s2 = 0f
                for (dx in -r..r) {
                    val ar = inImg.at(x + dx, y, 0); val ag = inImg.at(x + dx, y, 1); val ab = inImg.at(x + dx, y, 2)
                    val br = baseImg.at(x + dx, y, 0); val bg = baseImg.at(x + dx, y, 1); val bb = baseImg.at(x + dx, y, 2)
                    val yv = lumaOf(mix(ar, br, beta), mix(ag, bg, beta), mix(ab, bb, beta))
                    s += yv
                    s2 += yv * yv
                }
                hMean[y][x] = s / (2 * r + 1)
                hSq[y][x] = s2 / (2 * r + 1)
            }
        }
        val mean = Array(h) { FloatArray(w) }
        val sq = Array(h) { FloatArray(w) }
        for (y in 0 until h) {
            for (x in 0 until w) {
                var s = 0f
                var s2 = 0f
                for (dy in -r..r) {
                    val yy = (y + dy).coerceIn(0, h - 1)
                    s += hMean[yy][x]
                    s2 += hSq[yy][x]
                }
                mean[y][x] = s / (2 * r + 1)
                sq[y][x] = s2 / (2 * r + 1)
            }
        }
        return mean to sq
    }

    /** Dense chroma box mean of the beta-composed input at the pixel (base-centric). */
    private fun chromaMeanAt(inImg: Pln, baseImg: Pln, beta: Float, x: Int, y: Int, r: Int): FloatArray {
        var s1 = 0f
        var s2 = 0f
        for (dy in -r..r) {
            for (dx in -r..r) {
                val ar = inImg.at(x + dx, y + dy, 0); val ag = inImg.at(x + dx, y + dy, 1); val ab = inImg.at(x + dx, y + dy, 2)
                val br = baseImg.at(x + dx, y + dy, 0); val bg = baseImg.at(x + dx, y + dy, 1); val bb = baseImg.at(x + dx, y + dy, 2)
                s1 += mix(ar, br, beta) - mix(ab, bb, beta)
                s2 += 0.5f * (mix(ar, br, beta) + mix(ab, bb, beta)) - mix(ag, bg, beta)
            }
        }
        val n = (2 * r + 1) * (2 * r + 1)
        return floatArrayOf(s1 / n, s2 / n)
    }

    /** Host-math epsilon in image units (PreviewRenderer.runStage5 + MAIN_FRAGMENT). */
    private fun epsBaseImageUnits(cfg: Cfg, yccY: Float): Float {
        val whiteRangeI = 1f / sqrt(cfg.inverseRange2)
        val signalDN = max(yccY * whiteRangeI, 0f)
        val sigma2 = max(cfg.isoModelA * signalDN + cfg.isoModelB, 0f) * cfg.evGain2
        return (sigma2 + cfg.sigmaDm2) * cfg.inverseRange2 * cfg.sigmaScale * cfg.epsBoost
    }

    private fun cfgEpsY(cfg: Cfg, yccY: Float): Float = cfg.lumaEpsScale * epsBaseImageUnits(cfg, yccY)
    private fun cfgEpsC(cfg: Cfg, yccY: Float): Float = cfg.chromaEpsScale * epsBaseImageUnits(cfg, yccY)

/**
 * MAD sigma_hat mirror of SigmaHatShaderProgram (rgbMode=true domain): per-pixel
 * sigma_hat^2 as the mean over the R/G/B channels of 2.1981*MAD^2 of the 8 spatial
 * neighbours (+-1 texel), scaled by whiteRange^2 to raw-DN^2, then clamped up
 * to the Stage-0 ISO model floor at the centre luma - i.e. exactly what
 * Stage 5 consumes as the sigma_hat texture when useIsoSigma=false.
 */
    private fun madSigma2Dn(img: Pln, cfg: Cfg): Array<FloatArray> {
        val w = img.w
        val h = img.h
        val whiteRange = 1f / sqrt(cfg.inverseRange2)
        val nox = intArrayOf(1, -1, 0, 0, 1, -1, 1, -1)
        val noy = intArrayOf(0, 0, 1, -1, 1, 1, -1, -1)
        val out = Array(h) { FloatArray(w) }
        for (y in 0 until h) {
            for (x in 0 until w) {
                var sig2Total = 0f
                for (p in 0 until 3) {
                    val vals = FloatArray(8)
                    for (k in 0 until 8) {
                        val nx = (x + nox[k]).coerceIn(0, w - 1)
                        val ny = (y + noy[k]).coerceIn(0, h - 1)
vals[k] = img.at(nx, ny, p) * whiteRange
                }
                vals.sort()
                val med = (vals[3] + vals[4]) * 0.5f
                val devs = FloatArray(8) { abs(vals[it] - med) }
                devs.sort()
                val mad = (devs[3] + devs[4]) * 0.5f
                sig2Total += 2.1981f * mad * mad
                }
                val sig2Mean = sig2Total / 3f
                val luma = lumaOf(img.at(x, y, 0), img.at(x, y, 1), img.at(x, y, 2))
                val floor2 = max(cfg.isoModelA * max(luma * whiteRange, 0f) + cfg.isoModelB, 0f)
                out[y][x] = max(sig2Mean, floor2)
            }
        }
        return out
    }

/**
 * Direction-aware MAD variant (residual-pit candidate, heads-off of the S5
 * formula): instead of the median-abs-dev over all 8 neighbours (which the
 * edge straddling a thin bright stroke inflates to sigma_hat^2 ~ 10-20k DN^2, which raised
 * id-epsilon so SWGF pulls the edge pixel toward the mixed window mean), take the
 * MINIMUM squared pair-difference over the 4 axes (E/W, N/S, diag1, diag2).
 * At a straight edge the along-edge axis stays at the REGION's own noise,
 * so the sigma_hat^2 stays small at the ink edge and aY stays high.  On pure noise
 * each axis is an unbiased sigma^2 estimator (E[(n1-n2)^2/2]=sigma^2); the min of 4
 * is biased low, which the ISO-model floor (same as madSigma2Dn) anchors
 * wherever the region is quiet.  Same channels->mean, same 2.1981 constant
 * and floor so the calibration domain is unchanged.
 * mode="min" | "med" selects the axis reduction (measure both).
 */
    private fun madSigma2DnAxis(img: Pln, cfg: Cfg, mode: String): Array<FloatArray> {
        val w = img.w
        val h = img.h
        val whiteRange = 1f / sqrt(cfg.inverseRange2)
        // GLSL neighbour order (NOX, NOY): k0 E, k1 W, k2 N, k3 S,
        // k4 SE(1,1), k5 SW(1,-1), k6 NW(-1,-1), k7 NE(-1,1)
        val nox = intArrayOf(1, -1, 0, 0, 1, -1, 1, -1)
        val noy = intArrayOf(0, 0, 1, -1, 1, 1, -1, -1)
        val ax = arrayOf(intArrayOf(0, 1), intArrayOf(2, 3), intArrayOf(4, 6), intArrayOf(5, 7))
        val out = Array(h) { FloatArray(w) }
        for (y in 0 until h) {
            for (x in 0 until w) {
                var sig2Total = 0f
                for (p in 0 until 3) {
                    val vals = FloatArray(8)
                    for (k in 0 until 8) {
                        val nx = (x + nox[k]).coerceIn(0, w - 1)
                        val ny = (y + noy[k]).coerceIn(0, h - 1)
                        vals[k] = img.at(nx, ny, p) * whiteRange
                    }
                    val axisVars = FloatArray(4)
                    for (a in 0 until 4) {
                        val d = vals[ax[a][0]] - vals[ax[a][1]]
                        axisVars[a] = 0.5f * d * d
                    }
                    axisVars.sort()
                    val vSel = if (mode == "min") axisVars[0] else axisVars[1] // min / lower-median
                    sig2Total += 2.1981f * vSel
                }
                val sig2Mean = sig2Total / 3f
                val luma = lumaOf(img.at(x, y, 0), img.at(x, y, 1), img.at(x, y, 2))
                val floor2 = max(cfg.isoModelA * max(luma * whiteRange, 0f) + cfg.isoModelB, 0f)
                out[y][x] = max(sig2Mean, floor2)
            }
        }
        return out
    }

    // GLSL round() is half-away-from-zero; Kotlin roundToInt() is half-up.
    private fun glslRound(v: Float): Int = if (v >= 0f) (v + 0.5f).toInt() else (v - 0.5f).toInt()

/**
 * One SWGF iteration mirroring MAIN_FRAGMENT:
 *  - epsY/epsC taken from the host math unless anchor overrides are given
 *    (anchor mode: epsilon = kappa^2*sigma^2 directly, the attenuation identity test);
 *  - epsMult multiplies both epsilon (round-2 (kappa x 1.4)^2 fold);
 *  - beta composes in/out of stats, chroma box and final strength mix.
 */
    private fun s5Pass(
        inImg: Pln, baseImg: Pln, cfg: Cfg, epsMult: Float = 1f,
        epsYAnchor: Float? = null, epsCAnchor: Float? = null,
        sigma2Dn: Array<FloatArray>? = null, dbgXY: IntArray? = null,
        sigmaDoubleDomain: Boolean = false
    ): Pln {
        val w = inImg.w
        val h = inImg.h
        val rWin = max(2, glslRound(2.0f * cfg.winScale))
        val rChroma = min(max(2, glslRound(2.0f * cfg.winScale)), 8)
        val (statsMean, statsSq) = statsOf(inImg, baseImg, cfg.beta, cfg.winScale)
        val out = Array(h) { FloatArray(w * 3) }
        for (y in 0 until h) {
            for (x in 0 until w) {
                val ar = inImg.at(x, y, 0); val ag = inImg.at(x, y, 1); val ab = inImg.at(x, y, 2)
                val br = baseImg.at(x, y, 0); val bg = baseImg.at(x, y, 1); val bb = baseImg.at(x, y, 2)
                val rgbIn = floatArrayOf(mix(ar, br, cfg.beta), mix(ag, bg, cfg.beta), mix(ab, bb, cfg.beta))
                val yccIn = yccOf(rgbIn[0], rgbIn[1], rgbIn[2])

                val epsY: Float
                val epsC: Float
                if (epsYAnchor != null && epsCAnchor != null) {
                    epsY = epsYAnchor * epsMult
                    epsC = epsCAnchor * epsMult
                } else if (sigma2Dn != null) {
                    // Texture-driven sigma_hat (useIsoSigma=false): epsilon from the per-pixel
                    // MAD sigma_hat^2 (DN^2), same sigma_dm^2 base as the formula branch.
                    // sigmaDoubleDomain reproduces the capture sigma-hat bug: the
                    // texture holds sigma_hat^2*WR^2 while the main divides by 1/WR^2 once,
                    // leaving epsilon ~ 0.7*sigma_hat^2 instead of the designed 0.7*sigma_hat^2/WR^2.
                    val s2eff = if (sigmaDoubleDomain) sigma2Dn[y][x] * (1f / cfg.inverseRange2) else sigma2Dn[y][x]
                    val base = (s2eff + cfg.sigmaDm2) * cfg.inverseRange2 * cfg.sigmaScale * cfg.epsBoost
                    epsY = cfg.lumaEpsScale * base * epsMult
                    epsC = cfg.chromaEpsScale * base * epsMult
                } else {
                    val base = epsBaseImageUnits(cfg, yccIn[0])
                    epsY = cfg.lumaEpsScale * base * epsMult
                    epsC = cfg.chromaEpsScale * base * epsMult
                }

                // 8 SWGF side windows, distance proportional to |mean-y|/(var+epsilon), soft-2 fusion.
                var bestScore = 1.0e30f
                var sndScore = 1.0e30f
                var bestMean = 0f
                var bestVar = 0f
                var sndMean = 0f
                var sndVar = 0f
                for (k in 0..7) {
                    val sx = (x + WINDOW_CENTERS[k][0] * cfg.winScale).coerceIn(0f, (w - 1).toFloat()).toInt()
                    val sy = (y + WINDOW_CENTERS[k][1] * cfg.winScale).coerceIn(0f, (h - 1).toFloat()).toInt()
                    val m = statsMean[sy][sx]
                    val v = max(statsSq[sy][sx] - m * m, 0f)
                    val score = abs(m - yccIn[0]) / (v + epsY)
                    if (score < bestScore) {
                        sndScore = bestScore; sndMean = bestMean; sndVar = bestVar
                        bestScore = score; bestMean = m; bestVar = v
                    } else if (score < sndScore) {
                        sndScore = score; sndMean = m; sndVar = v
                    }
                }
                val wB = 1f / (bestScore * bestScore + 1e-12f)
                val wS = 1f / (sndScore * sndScore + 1e-12f)
                val pB = wB / (wB + wS)
                val pS = wS / (wB + wS)
                val meanY = pB * bestMean + pS * sndMean
                val varY = pB * bestVar + pS * sndVar

                val aY = varY / (varY + epsY)
                val outY = aY * yccIn[0] + (1f - aY) * meanY

                val cm = chromaMeanAt(inImg, baseImg, cfg.beta, x, y, rChroma)
                val aC = varY / (varY + epsC)
                val outC1 = aC * yccIn[1] + (1f - aC) * cm[0]
                val outC2 = aC * yccIn[2] + (1f - aC) * cm[1]

                val rgbF = rgbOf(outY, outC1, outC2)
                val st = cfg.strength
                if (dbgXY != null && x == dbgXY[0] && y == dbgXY[1]) {
                    println(String.format(
                        "  [s5 dbg(%d,%d) epsMult=%.2f] in=%.3f yccIn=(%.3f,%.3f,%.3f) epsY=%.2e epsC=%.2e " +
                            "bestM=%.3f bestV=%.4f sndM=%.3f sndV=%.4f meanY=%.3f varY=%.4f aY=%.3f aC=%.3f outY=%.3f rgbF=(%.3f,%.3f,%.3f) rgbIn=(%.3f,%.3f,%.3f)\n",
                        x, y, epsMult, inImg.at(x, y, 0), yccIn[0], yccIn[1], yccIn[2], epsY, epsC,
                        bestMean, bestVar, sndMean, sndVar, meanY, varY, aY, varY / (varY + epsC), outY,
                        rgbF[0], rgbF[1], rgbF[2], rgbIn[0], rgbIn[1], rgbIn[2]
                    ))
                }
                out[y][x * 3 + 0] = mix(rgbIn[0], rgbF[0], st)
                out[y][x * 3 + 1] = mix(rgbIn[1], rgbF[1], st)
                out[y][x * 3 + 2] = mix(rgbIn[2], rgbF[2], st)
            }
        }
        return Pln(w, h, out)
    }

    /** Run the double pass as the host does (host skips at strength 0 -> bit-exact). */
    private fun runS5(img: Pln, cfg: Cfg, epsYAnchor: Float? = null, epsCAnchor: Float? = null,
        sigma2Dn: Array<FloatArray>? = null, dbgXY: IntArray? = null,
        sigmaDoubleDomain: Boolean = false): Pln {
        if (cfg.strength <= 0f) return img
        val r1 = s5Pass(img, img, cfg, 1f, epsYAnchor, epsCAnchor, sigma2Dn, dbgXY, sigmaDoubleDomain)
        if (cfg.iterations < 2) return r1
        return s5Pass(r1, img, cfg, cfg.round2EpsMult, epsYAnchor, epsCAnchor, sigma2Dn, dbgXY, sigmaDoubleDomain)
    }

    // ------------------------------------------------------------------
    // Measurement helpers.
    // ------------------------------------------------------------------

    private fun lumaStd(p: Pln, lo: Int, hi: Int): Float {
        var s = 0.0
        var s2 = 0.0
        var n = 0.0
        for (y in lo until hi) {
            for (x in lo until hi) {
                val l = lumaOf(p.at(x, y, 0), p.at(x, y, 1), p.at(x, y, 2)).toDouble()
                s += l; s2 += l * l; n += 1.0
            }
        }
        val m = s / n
        return sqrt(max(s2 / n - m * m, 0.0)).toFloat()
    }

    private fun chanStd(p: Pln, ch: Int, lo: Int, hi: Int): Float {
        var s = 0.0
        var s2 = 0.0
        var n = 0.0
        for (y in lo until hi) {
            for (x in lo until hi) {
                val v = p.at(x, y, ch).toDouble()
                s += v; s2 += v * v; n += 1.0
            }
        }
        val m = s / n
        return sqrt(max(s2 / n - m * m, 0.0)).toFloat()
    }

    /** Attenuation identity: a = 1/(1+kappa^2); att = sqrt(a^2 + (1-a^2)/|omega|), |omega|=25. */
    private fun theoryAtt(k: Double): Double {
        val a = 1.0 / (1.0 + k * k)
        return sqrt(a * a + (1.0 - a * a) / 25.0)
    }

    /** Single-sided grating amplitude via DFT bin at period P over a horizontal profile. */
    private fun gratingAmp(p: Pln, y0: Int, y1: Int, x0: Int, x1: Int, period: Int): Float {
        var re = 0f
        var im = 0f
        var n = 0f
        val rows = y0 until y1
        // Average the profile over rows to suppress per-pixel noise.
        val profile = FloatArray(x1 - x0)
        for (x in x0 until x1) {
            var acc = 0f
            for (y in rows) acc += lumaOf(p.at(x, y, 0), p.at(x, y, 1), p.at(x, y, 2))
            profile[x - x0] = acc / (y1 - y0)
        }
        for ((i, v) in profile.withIndex()) {
            val t = 2f * PI.toFloat() * i / period
            re += v * cos(t)
            im += v * sin(t)
            n++
        }
        return 2f * sqrt(re * re + im * im) / n
    }

    /** Edge rise-width: number of columns over which a 10%->90% crossing takes place. */
    private fun edgeRiseWidth(p: Pln, y: Int, x0: Int, x1: Int, lo: Float, hi: Float): Int {
        val raw = (x0..x1).map { x -> lumaOf(p.at(x, y, 0), p.at(x, y, 1), p.at(x, y, 2)) }
        // Box-smooth radius 1 to suppress per-pixel noise in the level crossing.
        val prof = FloatArray(raw.size)
        for (i in prof.indices) {
            prof[i] = (raw[max(i - 1, 0)] + raw[i] + raw[min(i + 1, raw.lastIndex)]) / 3f
        }
        // lo/hi are the band LEVELS on the two sides; the transition is the
        // span of columns inside the 25-75% of the lo->hi step. Grating swings
        // (+-0.04) stay outside this band by construction.
        val tLo = lo + 0.25f * (hi - lo)
        val tHi = hi - 0.25f * (hi - lo)
        val a = prof.indexOfFirst { it >= tLo }
        val b = prof.indexOfLast { it <= tHi }
        if (a < 0 || b < 0) return x1 - x0
        return max(b - a + 1, 1)
    }

    private fun f4(v: Double): String = String.format("%.4f", v)
    private fun f3(v: Double): String = String.format("%.3f", v)
    private fun f2(v: Double): String = String.format("%.2f", v)

    // ------------------------------------------------------------------
    // 1. ANCHOR identity + cascade (kappa = 1.0 / 1.2 / 1.5).
    // ------------------------------------------------------------------

    @Test
    fun anchorIdentityAndCascade() {
        val sign = 0.5f
        val size = 160
        val lo = 8
        val hi = size - 8
        val sigmaChPx = 8f / 959f                 // 8 DN noise -> image units
        val sigmaLumaInj = 0.61237244f * sigmaChPx // luma noise of neutral RGB
        val my = sigmaLumaInj.toDouble()
        val sb = StringBuilder()
        sb.append("=== S5 double-pass pure-noise anchor (JVM mirror, midpoint signal=${sign}) ===\n\n")
        sb.append(String.format("%-5s %-7s %-7s %-8s %-8s %-8s %-8s %-8s %-8s\n",
            "kappa", "a", "LB1", "meas1", "LB2(kappa>=1.4)", "meas2an", "LBcas", "meas-cas(beta0)", "meas-cas(beta.3)"))
        var allOk = true

        for (k in doubleArrayOf(1.0, 1.2, 1.5)) {
            val k2 = k * 1.4
            val lb1 = theoryAtt(k)
            val lb2 = theoryAtt(k2)
            val lbCas = lb1 * lb2

            // epsilon anchors in image units: epsilon = kappa^2*sigma^2.
            val epsY = (k * k * my * my).toFloat()
            val epsC = epsY * (CHROMA_EPS_SCALE / LUMA_EPS_SCALE) // keep Y:C design ratio

            val img = noiseBlock(size, sign, sigmaChPx, seed = 1001 + (k * 100).toLong())
            val sigLumaIn = lumaStd(img, lo, hi).toDouble()

            // Round 1 single pass.
            val r1 = s5Pass(img, img, Cfg(beta = BETA, lumaEpsScale = 1.4f, chromaEpsScale = 64f),
                1f, epsY, epsC)
            val meas1 = lumaStd(r1, lo, hi).toDouble() / sigLumaIn

            // Round 2 anchor on a FRESH white block at kappa2 (single-pass identity).
            val fresh = noiseBlock(size, sign, sigmaChPx, seed = 2001 + (k * 100).toLong())
            val r2fresh = s5Pass(fresh, fresh, Cfg(beta = 0f), 1f,
                epsY * ROUND2_EPS_MULT, epsC * ROUND2_EPS_MULT)
            val meas2an = lumaStd(r2fresh, lo, hi).toDouble() / lumaStd(fresh, lo, hi).toDouble()

            // Cascade without beta (round 2 blends round-1 with itself, epsilon x 1.96).
            val cas0 = runS5(img, Cfg(beta = 0f, lumaEpsScale = 1.4f, chromaEpsScale = 64f),
                epsY, epsC)
            val measCas0 = lumaStd(cas0, lo, hi).toDouble() / sigLumaIn

            // Cascade with beta=0.3 noise return (shipped flavor).
            val casB = runS5(img, Cfg(beta = BETA, lumaEpsScale = 1.4f, chromaEpsScale = 64f),
                epsY, epsC)
            val measCasB = lumaStd(casB, lo, hi).toDouble() / sigLumaIn

            sb.append(String.format("%-5s %-7s %-7s %-8s %-8s %-8s %-8s %-8s %-8s\n",
                f3(k), f3(1.0 / (1.0 + k * k)), f4(lb1), f4(meas1), f4(lb2),
                f4(meas2an), f4(lbCas), f4(measCas0), f4(measCasB)))

            // Result: measured >= theory lower bound; beta raises the
            // residual vs the no-beta cascade (noise return), so the beta gap matches
            // the re-injection rather than exceeding the bounds.
            assertTrue("kappa=$k round-1 measured $meas1 must stay at/above LB $lb1 (mirror tol)",
                meas1 >= lb1 * 0.92)
            assertTrue("kappa=$k round-2(x1.4) fresh-block measured $meas2an must stay at/above LB $lb2",
                meas2an >= lb2 * 0.88)
            assertTrue("kappa=$k no-beta cascade measured $measCas0 must stay at/above LB $lbCas",
                measCas0 >= lbCas * 0.85)
            assertTrue("kappa=$k beta cascade measured $measCasB must stay at/above LB $lbCas",
                measCasB >= lbCas * 0.85)
            // beta=0.3 must not DOUBLE-COUNT: the cascade with beta must not be much
            // below the no-beta one (the return re-injects noise rather than
            // adding attenuation) - clamp sanity band instead of a hard sign.
            assertTrue("kappa=$k beta=0.3 cascade $measCasB must not drop far below no-beta $measCas0",
                measCasB >= measCas0 - 0.03)
            if (meas1 < lb1 * 0.92 || meas2an < lb2 * 0.88 || measCas0 < lbCas * 0.85 ||
                measCasB < lbCas * 0.85 || measCasB < measCas0 - 0.03
            ) allOk = false
        }
        sb.append("\nAnchor result: all kappa tiers measured >= theory LB (tolerances above) -> ")
        sb.append(if (allOk) "PASS\n\n" else "FAIL\n\n")

        java.io.File("build/s5noise_anchor.txt").let { f ->
            val current = if (f.exists()) f.readText() else ""
            f.writeText(current + sb.toString())
        }
        assertTrue("noise-anchor identity: all tiers within tolerance of the theory lower bounds", allOk)
    }

    // ------------------------------------------------------------------
    // 3. MAD sigma_hat texture vs ISO-formula sigma_hat (re-calibration check).
    //    Re-runs the three anchor tiers with epsilon built from the MAD sigma_hat^2 texture
    //    instead of epsilon=kappa^2*sigma^2; deviation vs the formula-driven
    //    attenuation must stay within +-10% on flat noise, else only the
    //    sigmaScale term may be retuned (as opposed to touching S5 math).
    // ------------------------------------------------------------------

    @Test
    fun madSigmaTextureDrivenMatchesFormulaWithinTenPct() {
        val sign = 0.5f
        val size = 128
        val lo = 10
        val hi = size - 10
        val sb = StringBuilder()
        sb.append("=== MAD sigma_hat texture vs ISO-formula sigma_hat, three anchor tiers (mid signal=0.5) ===\n\n")
        sb.append(String.format("%-8s %-8s %-9s %-10s %-11s %-14s %-14s %-8s\n",
            "ISO", "sigmaDN", "att_form", "att_mad", "rel-diff", "MAD SigmaHat2/DN^2", "model SigmaHat2/DN^2", "pass"))
        var allOk = true
        var seed = 7001L
        for (iso in intArrayOf(800, 3200, 12800)) {
            val cfg = Cfg(iso = iso, evGain2 = 1f)
            val signalDN = sign * (1f / sqrt(cfg.inverseRange2))
            val sigma2Ref = cfg.isoModelA * signalDN + cfg.isoModelB
            val sigmaChPx = sqrt(sigma2Ref) / (1f / sqrt(cfg.inverseRange2))
            val img = noiseBlock(size, sign, sigmaChPx, seed = seed)
            val sigLumaIn = lumaStd(img, lo, hi)

            // Formula-driven (useIsoSigma=true semantics - the theory anchor).
            val outF = runS5(img, cfg)
            val attF = lumaStd(outF, lo, hi) / sigLumaIn

            // Texture-driven (useIsoSigma=false): epsilon from the sigma_hat^2 map.  Capture
            // rgbMode now uses the direction-aware (axis-min) sigma_hat generator (shipped
            // GLSL); on FLAT noise it is floor-anchored and equals the
            // ISO model up to the measurement, so this invariant must hold here.
            val madMap = madSigma2DnAxis(img, cfg, "min")
            val outT = runS5(img, cfg, sigma2Dn = madMap)
            val attT = lumaStd(outT, lo, hi) / sigLumaIn

            // Sampled texel MAD vs model for the acceptance-(a) table.
            val mid = size / 2
            val madMid = madMap[mid][mid]
            val rel = (abs(attT - attF) / attF)

            // Acceptance (b): no visible regression vs the measured tolerance values.
            val inTol = rel <= 0.10f
            sb.append(String.format("%-8s %-8s %-9s %-10s %-11s %-12s %-10s %-10s\n",
                "$iso", f3(sqrt(sigma2Ref).toDouble()), f4(attF.toDouble()), f4(attT.toDouble()),
                f4(rel.toDouble()), f2(madMid.toDouble()), f2(sigma2Ref.toDouble()),
                if (inTol) "PASS" else "FAIL"))
            if (!inTol) allOk = false
            seed += 17
        }
        sb.append("\nSigma-scale check (within +-10% -> keep the current sigmaScale): " +
            if (allOk) "PASS\n\n" else "FAIL (retune sigmaScale only)\n\n")

        java.io.File("build/s5noise_anchor.txt").let { f ->
            val current = if (f.exists()) f.readText() else ""
            f.writeText(current + sb.toString())
        }
        assertTrue("MAD sigma_hat-driven S5 must stay within +-10% of the ISO-formula attenuation on flat noise", allOk)
    }

    // ------------------------------------------------------------------
    // 2. PIPELINE-config cascade attenuation: ISO x signal x path.
    // ------------------------------------------------------------------

    @Test
    fun pipelineConfigCascade() {
        val sb = StringBuilder()
        sb.append("=== Pipeline-config double-pass cascade (epsilon from ISO model, host math) ===\n\n")

        data class Path(val name: String, val winScale: Float, val epsBoost: Float, val evGain2: Float, val size: Int, val lo: Int)

        val preview = Path("preview", 1f, 1f, 1f, 152, 8)
        val evLow = Path("EV-low(1/8)", 1f, 1f, 0.125f, 152, 8)
        val cap1_1 = Path("capture-1:1", 3f, 16f, 1f, 208, 18)
        val capBoxAa = Path("capture-boxAA", 1.5f, 1f, 1f, 176, 12)

        var allOk = true
        val rows = StringBuilder()
        rows.append(String.format("%-9s %-7s %-5s %-6s %-8s %-8s %-8s %-8s %-9s %-9s %-9s\n",
            "path", "ISO(seg)", "sig", "sigmaDN", "kappa_eff", "att1", "att2", "casbeta.3", "cas(beta0)", "C1att", "C2att"))

        val sigmaDnTiers = doubleArrayOf(2.0, 4.0, 8.0, 16.0, 24.0)
        val signals = doubleArrayOf(0.16, 0.5, 0.83)
        val paths = listOf(preview, capBoxAa, cap1_1, evLow)

        // Full grid on preview + EV-low at all ISO; capture paths at
        // measured (800), collision (3200) and assumed (12800) ISO; 6400 is the
        // interpolated boundary.
        for (path in paths) {
            val isos = if (path.name.startsWith("preview") || path.name.startsWith("EV")) intArrayOf(800, 3200, 6400, 12800)
            else intArrayOf(800, 3200, 12800)
            for (iso in isos) {
                for (sig in signals) {
                    // The pipeline sigma_hat model predicts variance on this signal; the
                    // block injects sigma truth = one representative tier (8 DN).
                    val sigLumaPxInj = 0.61237244f * (8f / 959f)
                    val img = noiseBlock(path.size, sig.toFloat(), 8f / 959f, seed = 3000L + iso.toLong())
                    val cfg = Cfg(
                        beta = BETA, iso = iso, winScale = path.winScale, epsBoost = path.epsBoost,
                        evGain2 = path.evGain2
                    )
                    val r1 = s5Pass(img, img, cfg)
                    val r2 = s5Pass(r1, img, cfg, cfg.round2EpsMult)
                    val casB = lumaStd(r2, path.lo, path.size - path.lo).toDouble() / lumaStd(img, path.lo, path.size - path.lo).toDouble()
                    val cas0 = runS5(img, cfg.copy(beta = 0f)).let {
                        lumaStd(it, path.lo, path.size - path.lo).toDouble() / lumaStd(img, path.lo, path.size - path.lo).toDouble()
                    }
                    val c1 = lumaStd(img, path.lo, path.size - path.lo)
                    val att1 = lumaStd(r1, path.lo, path.size - path.lo).toDouble() / c1.toDouble()
                    val att2 = lumaStd(r2, path.lo, path.size - path.lo).toDouble() / lumaStd(r1, path.lo, path.size - path.lo).toDouble()
                    val epsYi = cfgEpsY(cfg, sig.toFloat())
                    val kappa = sqrt((epsYi / (sigLumaPxInj * sigLumaPxInj)).toDouble())
                    val c1att = chanStd(r2, 0, path.lo, path.size - path.lo) / chanStd(img, 0, path.lo, path.size - path.lo)
                    val c2att = chanStd(r2, 2, path.lo, path.size - path.lo) / chanStd(img, 2, path.lo, path.size - path.lo)
                    rows.append(String.format("%-9s %-7s %-5s %-6s %-8s %-8s %-8s %-8s %-9s %-9s %-9s\n",
                        path.name, "${iso}(${segmentOf(iso)})", f2(sig), "8",
                        f2(kappa), f4(att1), f4(att2), f4(casB), f4(cas0), f4(c1att.toDouble()), f4(c2att.toDouble())))
                    // The filter must always DENOISE (att <= 1) and never collapse
                    // below the residual floor (att not << 0.05).
                    assertTrue("${path.name} ISO=$iso sig=$sig cascade beta attenuates (<=1) got $casB", casB <= 1.001f)
                    assertTrue("${path.name} ISO=$iso sig=$sig cascade beta not degenerate ($casB)", casB >= 0.04f)
                    if (path.name == "capture-1:1" || path.name == "capture-boxAA") {
                        allOk = allOk && casB <= 1.001f && casB >= 0.04f
                    }
                }
            }
        }

        // Detailed sigma-tier scan at the collision ISO (3200, mid signal):
        // how kappa_eff and the cascade vary as the truth dictates.
        rows.append("\n-- ISO 3200 mid-signal sigma-tier scan (preview) --\n")
        rows.append(String.format("%-6s %-8s %-8s %-8s %-9s\n", "sigmaDN", "kappa_eff", "att1", "att2", "casbeta.3"))
        for (sigD in sigmaDnTiers) {
            val img = noiseBlock(152, 0.5f, (sigD / 959f).toFloat(), seed = 9000L + (sigD * 10).toLong())
            val cfg = Cfg(iso = 3200, beta = BETA)
            val sigL = 0.61237244f * sigD / 959f
            val r1 = s5Pass(img, img, cfg)
            val r2 = s5Pass(r1, img, cfg, cfg.round2EpsMult)
            val epsYi = cfgEpsY(cfg, 0.5f)
            val kappa = sqrt((epsYi / (sigL * sigL)).toDouble())
            val att1 = lumaStd(r1, 8, 144).toDouble() / lumaStd(img, 8, 144).toDouble()
            val att2 = lumaStd(r2, 8, 144).toDouble() / lumaStd(r1, 8, 144).toDouble()
            val cas = lumaStd(r2, 8, 144).toDouble() / lumaStd(img, 8, 144).toDouble()
            rows.append(String.format("%-6s %-8s %-8s %-8s %-9s\n", f1(sigD), f2(kappa), f4(att1), f4(att2), f4(cas)))
            assertTrue("ISO3200 mid sigma=${sigD.toInt()} cascade must remain denoising (cas=$cas<=1)", cas <= 1.001f)
            assertTrue("ISO3200 mid sigma=${sigD.toInt()} must not collapse (cas=$cas>=0.04)", cas >= 0.04f)
        }

        sb.append(rows)
        // The collision case: the S5D evidence showed the post-S3+boxAA residual is
        // sigma_hat/sqrt(32) (model sigma_hat ~ 32x residual variance). Inject that residual truth
        // and require kappa_eff to land in an engaged-but-bounded window. Note the
        // measured kappa sits above the nominal lumaEpsScale=1.4 coefficient by
        // the luma-mix fold sqrt(1/0.375)=1.63 (Y=0.25R+0.5G+0.25B -> var_Y=
        // 0.375*var_ch); the calibration table records this realization as-is.
        val cfg3200 = Cfg(iso = 3200)
        val sigHatDn = sqrt((cfg3200.isoModelA * 0.5f * 959f + cfg3200.isoModelB).toDouble())
        val sigmaResDn = sigHatDn / sqrt(32.0)
        val img3200 = noiseBlock(152, 0.5f, (sigmaResDn / 959f).toFloat(), seed = 777L)
        val sigL = 0.61237244f * (sigmaResDn / 959f).toFloat()
        val epsYi3200 = cfgEpsY(cfg3200, 0.5f)
        val kappa3200 = sqrt((epsYi3200 / (sigL * sigL)).toDouble())
        val r1c = s5Pass(img3200, img3200, cfg3200)
        val r2c = s5Pass(r1c, img3200, cfg3200, cfg3200.round2EpsMult)
        val cas3200 = lumaStd(r2c, 8, 144).toDouble() / lumaStd(img3200, 8, 144).toDouble()
        sb.append("ISO3200 mid-signal sigma_res=sigma_hat/sqrt(32)=${f2(sigmaResDn)}DN  kappa_eff=${f2(kappa3200)}  casbeta.3=${f4(cas3200)}\n")
        sb.append("(coefficient sqrt(lumaEpsScale)=${f2(sqrt(0.375) * kappa3200)}; Y-mix fold sqrt(1/0.375)=${f2(sqrt(1.0 / 0.375))}x)\n")
        allOk = allOk && kappa3200 in 1.0..2.5 && cas3200 in 0.03..0.75
        assertTrue("ISO3200 mid-signal sigma_res=${f2(sigmaResDn)}DN kappa_eff=$kappa3200 must be engaged in [1.0,2.5] (collision anchor)",
            kappa3200 in 1.0..2.5)
        assertTrue("ISO3200 mid-signal collision cascade $cas3200 must denoise (0.03..0.75)",
            cas3200 in 0.03..0.75)

        sb.append("\nPIPELINE RESULT: $allOk\n")
        java.io.File("build/s5noise_anchor.txt").let { f ->
            val current = if (f.exists()) f.readText() else ""
            f.writeText(current + sb.toString() + "\n")
        }
        assertTrue("pipeline-config denoise sanity across ISO/signal/path grid", allOk)
    }

    private fun f1(v: Double): String = String.format("%.1f", v)

    // ------------------------------------------------------------------
    // 3. Acceptance (a): texture preservation through the double pass.
    // ------------------------------------------------------------------

    @Test
    fun texturePreservationDoublePass() {
        val size = 176
        val bandXs = intArrayOf(0, 58, 116, size) // shadow | mid | highlight
        val cfg = Cfg(iso = 3200) // double pass, preview, beta=0.3
        // sigma tiers: 10 DN = general mid-scale scene noise; 3 DN ~ sigma_hat/sqrt(32) =
        // the post-S3 collision residual (highest realized kappa_eff).
        data class Patch(val y0: Int, val y1: Int, val x0: Int, val x1: Int, val period: Int)
        val patches = listOf(
            Patch(8, 20, 62, 112, 16),
            Patch(100, 112, 62, 112, 6),
            Patch(8, 20, 120, 170, 16),
            Patch(100, 112, 120, 170, 6)
        )
        val sb = StringBuilder()
        sb.append("=== Texture preservation (ISO 3200, double pass, beta=0.3) ===\n")

        for (sigmaDN in intArrayOf(10, 3)) {
            val sigmaPx = sigmaDN.toFloat() / 959f
            val img = textureScene(size, bandXs, sigmaPx, pc = 16, ac = 0.03f, pf = 6, af = 0.04f, seed = 4242L)
            val r1 = s5Pass(img, img, cfg)
            val r2 = s5Pass(r1, img, cfg, cfg.round2EpsMult)
            sb.append(String.format("-- sigma=%d DN --\n", sigmaDN))
            sb.append(String.format("%-24s %-8s %-8s\n", "patch(period)", "ret-r1", "ret-r2"))

            for (p in patches) {
                val (y0, y1, x0, x1, period) = p
                val sin = gratingAmp(img, y0, y1, x0, x1, period)
                val s1 = gratingAmp(r1, y0, y1, x0, x1, period)
                val s2 = gratingAmp(r2, y0, y1, x0, x1, period)
                val band = if (x0 >= 116) "highlight" else "mid"
                val kind = if (period == 16) "coarse-P16" else "fine-P6"
                val ret1 = s1 / sin
                val ret2 = s2 / sin
                sb.append(String.format("%-24s %-8s %-8s\n", "$band $kind",
                    f3(ret1.toDouble()), f3(ret2.toDouble())))
                if (band == "highlight") {
                    val floor = if (period == 16) 0.55f else 0.4f
                    assertTrue("sigma=$sigmaDN highlight $kind: round-2 retention $ret2 must stay >= $floor",
                        ret2 >= floor)
                }
            }
            // Edge sharpness: step edges between bands must not widen.
            val edgeY = size / 2
            sb.append(String.format("%-18s %-14s %-6s %-6s\n", "edge", "input", "r1", "r2"))
            data class EdgeDef(val x: Int, val label: String, val lo: Float, val hi: Float)
            val edges = listOf(
                EdgeDef(58, "shadow|mid", 0.2f, 0.5f),
                EdgeDef(116, "mid|highlight", 0.5f, 0.83f)
            )
            for (e in edges) {
                val inW = edgeRiseWidth(img, edgeY, e.x - 8, e.x + 8, e.lo, e.hi)
                val r1W = edgeRiseWidth(r1, edgeY, e.x - 8, e.x + 8, e.lo, e.hi)
                val r2W = edgeRiseWidth(r2, edgeY, e.x - 8, e.x + 8, e.lo, e.hi)
                sb.append(String.format("sigma=%d %-12s %-14s %-6s %-6s\n", sigmaDN, e.label, "$inW px", "$r1W px", "$r2W px"))
                assertTrue("sigma=$sigmaDN ${e.label} edge: r1 must stay sharp (r1=$r1W px)", r1W <= 4)
                assertTrue("sigma=$sigmaDN ${e.label} edge: r2 must stay sharp (r2=$r2W px)", r2W <= 4)
            }

            // Cross-check noise: the flat shadow band (no grating) must be
            // denoised. At the 10 DN tier the realized kappa is light (~0.5), so
            // only no-amplification is required; at the collision-residual
            // tier (sigma ~ sigma_hat/sqrt(32) -> kappa_eff ~ 1.9) the cascade must actually clean.
            val flatIn2 = lumaStd(img, 88, 152, 6, 52)
            val flatR2 = lumaStd(r2, 88, 152, 6, 52)
            sb.append(String.format("sigma=%d flat-shadow sigma: in %.6f -> r2 %.6f (att %.3f)\n",
                sigmaDN, flatIn2, flatR2, flatR2 / flatIn2))
            if (sigmaDN == 10) {
                assertTrue("sigma=$sigmaDN flat shadow must not amplify (att=${flatR2 / flatIn2})",
                    flatR2 <= flatIn2 * 1.02f)
            } else {
                assertTrue("sigma=$sigmaDN flat shadow must be denoised by double pass (att=${flatR2 / flatIn2})",
                    flatR2 < flatIn2 * 0.85f)
            }
        }

        java.io.File("build/s5noise_anchor.txt").let { f ->
            val current = if (f.exists()) f.readText() else ""
            f.writeText(current + sb.toString() + "\n")
        }
        // (Assertions already fired above; reaching here = pass.)
    }

    private fun lumaStd(p: Pln, y0: Int, y1: Int, loX: Int, hiX: Int): Float {
        var s = 0.0
        var s2 = 0.0
        var n = 0.0
        for (y in y0 until y1) {
            for (x in loX until hiX) {
                val l = lumaOf(p.at(x, y, 0), p.at(x, y, 1), p.at(x, y, 2)).toDouble()
                s += l; s2 += l * l; n += 1.0
            }
        }
        val m = s / n
        return sqrt(max(s2 / n - m * m, 0.0)).toFloat()
    }

    // ------------------------------------------------------------------
    // 4. Acceptance (c): slider 0 = bit-exact bypass through the double pass.
    // ------------------------------------------------------------------

    @Test
    fun strengthZeroBitExactBypass() {
        val size = 96
        val img = textureScene(size, intArrayOf(0, 32, 64, size), 8f / 959f, 12, 0.02f, 6, 0.03f, seed = 11L)
        val cfg = Cfg(iterations = 2, beta = BETA, strength = 0f)
        val out = runS5(img, cfg)
        // runS5 short-circuits at strength 0 -> out IS img (bit-exact); also
        // assert planes are reference-identical.
        assertTrue("strength=0 must return the input Pln instance (full bypass)", out === img)
        for (y in 0 until size) {
            for (x in 0 until size) {
                val a = img.at(x, y, 0)
                val b = out.at(x, y, 0)
                if (a != b) {
                    assertTrue("bit-exact bypass violated at ($x,$y)", false)
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 5. Acceptance (d): double pass must not reintroduce winner-jump seams.
    // ------------------------------------------------------------------

    @Test
    fun doublePassNoSeams() {
        val size = 129
        val sigmaPx = 3f / 959f
        val (img, cx, cy) = discScene(size, sigmaPx, seed = 7L)
        val cfg = Cfg(iso = 800, beta = BETA) // full double pass, preview
        val r2 = runS5(img, cfg)

        // C1-delta neighbor-jump scan on the annulus rows (S5ReproTest metric:
        // |delta delta C1| > 0.03 counts as a big seam).
        var bigSeams = 0
        var maxJump = 0f
        for (y in cy - 3..cy + 3) {
            for (x in cx + 1..cx + 40) {
                val d0 = (r2.at(x, y, 0) - r2.at(x, y, 2)) - (img.at(x, y, 0) - img.at(x, y, 2))
                val d1 = (r2.at(x - 1, y, 0) - r2.at(x - 1, y, 2)) - (img.at(x - 1, y, 0) - img.at(x - 1, y, 2))
                val j = abs(d0 - d1)
                if (j > 0.03f) bigSeams++
                maxJump = max(maxJump, j)
            }
        }
        assertTrue(
            "double pass must keep the annulus seam-free: bigSeams=$bigSeams (want 0), maxJump=$maxJump",
            bigSeams == 0
        )
    }

    // ------------------------------------------------------------------
    // 6. H1 probe: does S5 alone (no S1) dissolve a thin bright stroke on a
    // near-black field? Peak-contrast retention vs slider, orientation, width.
    // ------------------------------------------------------------------

    private fun onSceneStroke(x: Int, y: Int, size: Int, w: Int, angle: Int): Boolean {
        val cx = size / 2
        val cy = size / 2
        return when (angle) {
            0 -> x >= cx - w / 2 && x < cx - w / 2 + w
            90 -> y >= cy - w / 2 && y < cy - w / 2 + w
            45 -> (x - y) >= (cx - cy) && (x - y) < (cx - cy) + w
            135 -> (x + y) >= (cx + cy) && (x + y) < (cx + cy) + w
            else -> (2 * (x - cx) - (y - cy)) >= 0 && (2 * (x - cx) - (y - cy)) < 2 * w
        }
    }

    private fun thinStrokeScene(
        size: Int, w: Int, angle: Int, sigmaPx: Float, seed: Long,
        bg: Float = 0.06f, fg: Float = 0.95f
    ): Pln {
        val rnd = Random(seed)
        val c = Array(size) { FloatArray(size * 3) }
        for (y in 0 until size) {
            for (x in 0 until size) {
                val s = if (onSceneStroke(x, y, size, w, angle)) fg else bg
                c[y][x * 3 + 0] = s + gauss(rnd) * sigmaPx
                c[y][x * 3 + 1] = s + gauss(rnd) * sigmaPx
                c[y][x * 3 + 2] = s + gauss(rnd) * sigmaPx
            }
        }
        return Pln(size, size, c)
    }

    /**
     * Peak-contrast retention of a stroke: scan perpendicular to it over a
     * band around the centre and compare (peak - local bg) to the input value.
     * Horizontal strokes are scanned vertically; all others horizontally.
     */
    private fun strokeRetention(p: Pln, size: Int, w: Int, angle: Int, bg: Float = 0.06f, fg: Float = 0.95f): Float {
        val cx = size / 2
        val cy = size / 2
        fun lum(x: Int, y: Int) = lumaOf(p.at(x, y, 0), p.at(x, y, 1), p.at(x, y, 2))
        var peak = 0f
        var base = 0f
        var n = 0
        if (angle == 90) {
            for (x in cx - 10..cx + 10) {
                for (yy in cy - 8..cy + 8) peak = max(peak, lum(x, yy))
                base += 0.5f * (lum(x, cy - 14) + lum(x, cy + 14))
                n++
            }
        } else {
            for (y in cy - 10..cy + 10) {
                for (xx in cx - 8..cx + 8) peak = max(peak, lum(xx, y))
                base += 0.5f * (lum(cx - 14, y) + lum(cx + 14, y))
                n++
            }
        }
        base /= max(n, 1)
        return (peak - base) / (fg - bg)
    }

    @Test
    fun thinStrokeS5DissolutionProbe() {
        val size = 160
        val sigmaPx = 3f / 959f
        val bg = 0.06f
        val fg = 0.95f
        val sb = StringBuilder()
        sb.append("=== H1: S5-only thin bright stroke retention (fg=0.95 over bg=0.06, sigma=3 DN) ===\n")
        sb.append("retention = (peak-bg)/(fg-bg); input=1.0\n")
        sb.append(String.format("%-7s %-4s", "orient", "w"))
        for (s in floatArrayOf(0f, 0.25f, 0.5f, 0.75f, 1f)) sb.append(String.format("  ret@%-4.2f", s))
        sb.append('\n')
        val labels = mapOf(0 to "V", 90 to "H", 45 to "45", 135 to "135", 99 to "26deg")
        for (angle in intArrayOf(0, 90, 45, 135, 99)) {
            for (w in intArrayOf(1, 2, 3)) {
                if (angle == 99 && w != 1) continue
                val scene = thinStrokeScene(size, w, angle, sigmaPx, 31337L, bg, fg)
                sb.append(String.format("%-7s %-4d", labels[angle], w))
                for (s in floatArrayOf(0f, 0.25f, 0.5f, 0.75f, 1f)) {
                    val out = runS5(scene, Cfg(iso = 800, strength = s))
                    sb.append(String.format("  %8.3f", strokeRetention(out, size, w, angle, bg, fg)))
                }
                sb.append('\n')
            }
        }
        System.out.println(sb)
        java.io.File("build/s5_thin_stroke_h1.txt").writeText(sb.toString())
        // Harness sanity: strength 0 is the bit-exact input.
        val scene = thinStrokeScene(size, 1, 0, sigmaPx, 31337L, bg, fg)
        assertTrue("strength 0 stroke must be present",
            strokeRetention(runS5(scene, Cfg(iso = 800, strength = 0f)), size, 1, 0, bg, fg) > 0.5f)
    }

    // ------------------------------------------------------------------
    // 7. Capture-path: winScale>1 (R=8, +-8 offsets) + epsBoost=16 +
    // texture-driven MAD sigma-hat. Spatial metrics blind to peak contrast:
    // interior pitting, halo bleed, edge raggedness. Dose response in strength.
    // ------------------------------------------------------------------

    private fun captureStrokeScene(
        size: Int, w: Int, sigmaPx: Float, seed: Long, bg: Float = 0.06f, fg: Float = 0.95f
    ): Pln {
        val rnd = Random(seed)
        val c = Array(size) { FloatArray(size * 3) }
        val ctr = size / 2f
        for (y in 0 until size) {
            for (x in 0 until size) {
                val d = abs(x + 0.5f - ctr)
                val half = w * 0.5f
                val s = when {
                    d <= half -> fg
                    d <= half + 1f -> bg + (fg - bg) * (half + 1f - d)
                    else -> bg
                }
                c[y][x * 3 + 0] = s + gauss(rnd) * sigmaPx
                c[y][x * 3 + 1] = s + gauss(rnd) * sigmaPx
                c[y][x * 3 + 2] = s + gauss(rnd) * sigmaPx
            }
        }
        return Pln(size, size, c)
    }

    /**
     * [0] interior pitting rate: fraction of stroke-interior luma below
     *     bg + 0.5*(fg-bg); [1] halo: mean luma excess in the +2..+6 px ring
     *     outside the stroke / (fg-bg); [2] edge raggedness: y-std of the
     *     per-row 50% crossing column (px).
     */
    private fun captureStrokeMetrics(p: Pln, size: Int, w: Int, bg: Float, fg: Float): FloatArray {
        val ctr = size / 2
        val delta = fg - bg
        val y0 = size / 4
        val y1 = 3 * size / 4
        var pits = 0
        var interior = 0
        var haloSum = 0f
        var haloN = 0
        val crossings = ArrayList<Float>()
        val half = w / 2
        for (y in y0 until y1) {
            val prof = FloatArray(size) { lumaOf(p.at(it, y, 0), p.at(it, y, 1), p.at(it, y, 2)) }
            // Nominal flat-fg columns only (|x+0.5-ctr| <= half-0.5): pitting is
            // the dark dip the window-mean injection carves into the stroke,
            // concentrated just inside the edge, so it must be in-mask.
            for (x in 0 until size) {
                val dIn = abs(x + 0.5f - ctr)
                if (dIn > half - 0.5f) continue
                interior++
                if (prof[x] < bg + 0.5f * delta) pits++
            }
            var c0 = -1f
            for (x in ctr - 24..ctr) if (prof[x] >= bg + 0.5f * delta) { c0 = x.toFloat(); break }
            if (c0 >= 0f) crossings.add(c0)
            val lo = ctr + half + 1
            val hi = ctr + half + 5
            for (x in lo..hi) {
                haloSum += max(prof[x] - bg, 0f) / delta
                haloN++
            }
        }
        var halo = 0f
        if (haloN > 0) halo = haloSum / haloN
        var rag = 0f
        if (crossings.size > 1) {
            val m = crossings.average().toFloat()
            var v = 0f
            for (c in crossings) v += (c - m) * (c - m)
            rag = sqrt(v / crossings.size)
        }
        return floatArrayOf(pits.toFloat() / max(interior, 1), halo, rag)
    }

    @Test
    fun captureStrokeS5PitHaloProbe() {
        val size = 192
        val sigmaPx = 3f / 959f
        val bg = 0.06f
        val fg = 0.95f
        val sb = StringBuilder()
        sb.append("=== capture-path S5: pits / halo / raggedness (winScale=4, epsBoost=16, axis-min sigma_hat) ===\n")
        sb.append(String.format("%-5s %-6s %-10s %-9s %-9s %-9s\n", "w", "strength", "sigmaMode", "pitRate", "halo", "ragStd"))
        for (w in intArrayOf(4, 8, 16, 32)) {
            val scene = captureStrokeScene(size, w, sigmaPx, 909L, bg, fg)
            val cfgBase = Cfg(iso = 800, winScale = 4f, epsBoost = 16f)
            // Shipped capture GLSL: axis-min sigma_hat^2 (F) - the MAD-8 rows below
            // (sigProd = MAD*WR^2) only reproduce the historical double-domain.
            val sig = madSigma2DnAxis(scene, cfgBase, "min")
            // Production rgbMode double-scales sig2 by u_domain_scale^2
            // (vals are already x domainScale at line 201): sig2_prod = sig * whiteRange^2.
            val wR2 = 1f / cfgBase.inverseRange2
            val sigFull = madSigma2Dn(scene, cfgBase)
            val sigProd = Array(sig.size) { i -> FloatArray(sig[i].size) { j -> sigFull[i][j] * wR2 } }
            for (s in floatArrayOf(0f, 0.5f, 1f)) {
                val cfg = Cfg(iso = 800, winScale = 4f, epsBoost = 16f, strength = s)
                val outMirror = runS5(scene, cfg, sigma2Dn = sig)
                val mM = captureStrokeMetrics(outMirror, size, w, bg, fg)
                sb.append(String.format("%-5d %-6.2f %-10s %-9.4f %-9.4f %-9.3f\n",
                    w, s, "mirror", mM[0], mM[1], mM[2]))
                if (s > 0f) {
                    val outProd = runS5(scene, cfg, sigma2Dn = sigProd)
                    val mP = captureStrokeMetrics(outProd, size, w, bg, fg)
                    sb.append(String.format("%-5d %-6.2f %-10s %-9.4f %-9.4f %-9.3f\n",
                        w, s, "prod(xWR2)", mP[0], mP[1], mP[2]))
                }
            }
        }
        System.out.println(sb)
        java.io.File("build/s5_capture_pit_halo.txt").writeText(sb.toString())
        // Diagnostic: sigma_hat and output profile across a w=32 stroke at s=1.
        run {
            val sc = captureStrokeScene(size, 32, sigmaPx, 909L, bg, fg)
            val cb = Cfg(iso = 800, winScale = 4f, epsBoost = 16f)
            val sg = madSigma2DnAxis(sc, cb, "min")
            val wR2 = 1f / cb.inverseRange2
            val sgFull = madSigma2Dn(sc, cb)
            val sgP = Array(sg.size) { i -> FloatArray(sg[i].size) { j -> sgFull[i][j] * wR2 } }
            val cfg = Cfg(iso = 800, winScale = 4f, epsBoost = 16f, strength = 1f)
            val om = runS5(sc, cfg, sigma2Dn = sg)
            val op = runS5(sc, cfg, sigma2Dn = sgP)
            val d = StringBuilder("=== profile w=32 s=1 y=96 (x 78..114) ===\n")
            d.append(String.format("%-4s %-8s %-8s %-8s %-8s\n", "x", "scene", "sigMirror", "outMirror", "outProd"))
            for (x in 78..114) {
                val scn = lumaOf(sc.at(x, 96, 0), sc.at(x, 96, 1), sc.at(x, 96, 2))
                val omv = lumaOf(om.at(x, 96, 0), om.at(x, 96, 1), om.at(x, 96, 2))
                val opv = lumaOf(op.at(x, 96, 0), op.at(x, 96, 1), op.at(x, 96, 2))
                d.append(String.format("%-4d %-8.4f %-8.1f %-8.4f %-8.4f\n", x, scn, sg[96][x], omv, opv))
            }
            System.out.println(d)
            java.io.File("build/s5_capture_profile.txt").writeText(d.toString())
        }
        // Sanity: strength 0 must be the untouched input (no pits, no halo).
        val scene = captureStrokeScene(size, 8, sigmaPx, 909L, bg, fg)
        val clean = captureStrokeMetrics(scene, size, 8, bg, fg)
        assertTrue("strength 0 must be pit-free", clean[0] < 0.02f)
        assertTrue("strength 0 must be halo-free", clean[1] < 0.05f)
    }

    // ------------------------------------------------------------------
    // 8. Device-dump replay + metrics (cap1789706340051).
    //    Replays the real capture through the JVM mirror with the saved MAD
    //    sigma-hat, validates the mirror against the device, then reports
    //    the spatial artifacts (pits/halo) and the sigma magnitudes - the
    //    capture-path ground truth for judging H1/H2.
    // ------------------------------------------------------------------

    private val dumpW = 1024
    private val dumpH = 1024

    private fun dumpDir(): File {
        System.getProperty("capdump.dir")?.let { p -> val f = File(p); if (f.isDirectory) return f }
        // No baked-in default: the device dump is machine-local (gitignored),
        // so without -Dcapdump.dir the replay tests are skipped by
        // requireDeviceDump.
        return File("")
    }

    /** Device-dump gate: skip (not fail) when the gitignored dump is absent. */
    private fun requireDeviceDump(required: Array<String>): File {
        val dir = dumpDir()
        val missing = required.filter { !File(dir, it).exists() }
        assumeTrue(
            "device capture dump missing - skipping (clone/produce it or set -Dcapdump.dir): " +
                "$dir ${missing.joinToString { "[$it]" }}",
            missing.isEmpty()
        )
        return dir
    }

    private fun readRgbaF32(dir: File, name: String): FloatArray {
        val bytes = File(dir, name).readBytes()
        val bbuf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(bytes.size / 4)
        bbuf.asFloatBuffer().get(out)
        return out
    }

    private fun lumaOnDev(a: FloatArray, x: Int, y: Int): Float =
        lumaOf(a[(y * dumpW + x) * 4], a[(y * dumpW + x) * 4 + 1], a[(y * dumpW + x) * 4 + 2])

    private fun pctSorted(vals: List<Float>, p: Double): Float {
        val c = FloatArray(vals.size)
        for (i in c.indices) c[i] = vals[i]
        c.sort()
        val idx = ((c.size - 1) * p).toInt()
        return c[idx]
    }

    @Test
    fun deviceCaptureReplayAndMetrics() {
        val dir = requireDeviceDump(arrayOf("s5_in.f32", "s5_out.f32", "s5_sigma.f32"))
        val inF = readRgbaF32(dir, "s5_in.f32")
        val outF = readRgbaF32(dir, "s5_out.f32")
        val sigF = readRgbaF32(dir, "s5_sigma.f32")
        val w = dumpW; val h = dumpH
        val margin = 48
        val sb = StringBuilder()
        val cfgDev = Cfg(iso = 1029, winScale = 4.266667f, epsBoost = 16f, strength = 1f)
        sb.append("=== device S5 capture replay (iso=1029 ws=4.27 eb=16 st=1, sigmaTex) ===\n")

        // sigma_hat^2 (R) magnitude split by signal.
        val darkSig = ArrayList<Float>()
        val brightSig = ArrayList<Float>()
        val allSig = ArrayList<Float>()
        for (y in margin until h - margin) {
            for (x in margin until w - margin) {
                val L = lumaOnDev(inF, x, y)
                val s = sigF[(y * w + x) * 4]
                allSig.add(s)
                if (L < 0.3f) darkSig.add(s)
                else if (L > 0.6f) brightSig.add(s)
            }
        }
        sb.append(String.format(
            "sigma2 deviceR (=sig2hat x WR^2): dark p50=%.0f p90=%.0f | bright p50=%.0f p90=%.0f | p99.9=%.0f max=%.0f\n",
            pctSorted(darkSig, 0.5), pctSorted(darkSig, 0.9),
            pctSorted(brightSig, 0.5), pctSorted(brightSig, 0.9),
            pctSorted(allSig, 0.999), pctSorted(allSig, 1.0)
        ))

        // Build Pln inputs + mirror sigma for replay (and mirror-sigma comparison).
        // RGBA is pixel-major (r,g,b,a per pixel); deinterleave skipping alpha.
        val inP = Pln(w, h, Array(h) { y -> FloatArray(w * 3) { i ->
            val px = i / 3
            inF[(y * w + px) * 4 + (i - px * 3)]
        } })
        val mirrorSig = madSigma2Dn(inP, cfgDev)
        val darkMir = ArrayList<Float>()
        val brightMir = ArrayList<Float>()
        for (y in margin until h - margin) {
            for (x in margin until w - margin) {
                val L = lumaOnDev(inF, x, y)
                val m = mirrorSig[y][x]
                if (L < 0.3f) darkMir.add(m) else if (L > 0.6f) brightMir.add(m)
            }
        }
        val devDark = pctSorted(darkSig, 0.5)
        val devBright = pctSorted(brightSig, 0.5)
        val mirDark = pctSorted(darkMir, 0.5)
        val mirBright = pctSorted(brightMir, 0.5)
        // NOTE: GLSL SigmaHat applies u_domain_scale a second time in the mad term
        // (:201 vals*scale, :227 sig2=mad^2*scale^2), so R holds sigma_hat^2*WR^2; the S5 main
        // then divides by inverse_range2 (=1/WR^2), i.e. the extra scale cancels
        // except sigma_dm^2 -> ~0. Replay must feed R/WR^2 (a DN^2, which the mirror expects).
        val whiteRangeDev = 959f
        val devWr2 = whiteRangeDev * whiteRangeDev
        // Domain auto-detect: pre-fix sigma buffer holds sigma_hat^2*WR^2 (the GLSL
        // SigmaHat :227 double-scaled mad); post-fix it holds sigma_hat^2 raw-DN^2.
        // Pick the effective cell that reproduces the mirror medians (x~1.00).
        val devDarkR = devDark / devWr2; val devBrightR = devBright / devWr2
        val mDark = max(mirDark, 1e-6f); val mBright = max(mirBright, 1e-6f)
        val rDark = max(devDarkR, 1e-6f); val rBright = max(devBrightR, 1e-6f)
        val ratioDouble = max(devDarkR / mDark, mDark / rDark) + max(devBrightR / mBright, mBright / rBright)
        val ratioRaw = max(devDark / mDark, mDark / rDark) + max(devBright / mBright, mBright / rBright)
        val devScaleEffective = if (ratioDouble <= ratioRaw) devWr2 else 1f
        sb.append(String.format(
            "sigma-domain: %s (device sigma_hat^2 = R/%s)\n",
            if (devScaleEffective > 1f) "DOUBLE (pre-fix sigma_hat^2*WR^2)" else "SINGLE (post-fix raw-DN^2)",
            if (devScaleEffective > 1f) "WR^2" else "1"
        ))
        sb.append(String.format(
            "sigma2 p50  device(R/<scale>) vs mirror(DN2): dark %.1f/%.1f (x%.2f)  bright %.1f/%.1f (x%.2f)\n",
            devDark / devScaleEffective, mirDark, (devDark / devScaleEffective) / mDark,
            devBright / devScaleEffective, mirBright, (devBright / devScaleEffective) / mBright
        ))

        // sigma2Dn array from the device R channel (normalized to DN^2) -> replay.
        val sigma2Dev = Array(h) { y -> FloatArray(w) { x -> sigF[(y * w + x) * 4] / devScaleEffective } }
        val devOut = Pln(w, h, Array(h) { y -> FloatArray(w * 3) { i ->
            val px = i / 3
            outF[(y * w + px) * 4 + (i - px * 3)]
        } })

        fun bandMetrics(a: Pln, b: Pln): FloatArray {
            // [0] pitRate (bright, out<-in-0.08)  [1] haloRate (dark, out>in+0.08)
            // [2] meanPitDepth [3] meanHaloExcess
            var pitN = 0; var haloN = 0; var brightN = 0; var darkN = 0; var pitSum = 0f; var haloSum = 0f
            for (y in margin until h - margin) {
                for (x in margin until w - margin) {
                    val Li = lumaOf(a.at(x, y, 0), a.at(x, y, 1), a.at(x, y, 2))
                    val Lo = lumaOf(b.at(x, y, 0), b.at(x, y, 1), b.at(x, y, 2))
                    val d = Lo - Li
                    if (Li > 0.6f) { brightN++; if (d < -0.08f) { pitN++; pitSum += -d } }
                    else if (Li < 0.3f) { darkN++; if (d > 0.08f) { haloN++; haloSum += d } }
                }
            }
            return floatArrayOf(
                pitN.toFloat() / max(brightN, 1), haloN.toFloat() / max(darkN, 1),
                pitSum / max(pitN, 1), haloSum / max(haloN, 1)
            )
        }

        // Replay the full pipeline with the device sigma and compare.
        val replay = runS5(inP, cfgDev, sigma2Dn = sigma2Dev)
        var rms = 0.0; var n = 0
        for (y in margin until h - margin) {
            for (x in margin until w - margin) {
                val d = lumaOf(replay.at(x, y, 0), replay.at(x, y, 1), replay.at(x, y, 2)) -
                    lumaOf(devOut.at(x, y, 0), devOut.at(x, y, 1), devOut.at(x, y, 2))
                rms += d * d; n++
            }
        }
        rms = sqrt(rms / max(n, 1))
        sb.append(String.format("replay vs device out: luma RMS=%.5f\n", rms))

        // Device artifact metrics.
        val mA = bandMetrics(inP, devOut)
        sb.append(String.format(
            "device s5 (st=1): pitRate=%.4f haloRate=%.4f meanPit=%.4f meanHalo=%.4f\n",
            mA[0], mA[1], mA[2], mA[3]
        ))
        // Mirror dose response on the real capture input + real sigma.
        for (s in floatArrayOf(0.25f, 0.5f, 1f)) {
            val o = runS5(inP, Cfg(iso = 1029, winScale = 4.266667f, epsBoost = 16f, strength = s),
                sigma2Dn = sigma2Dev)
            val m = bandMetrics(inP, o)
            sb.append(String.format(
                "mirror st=%.2f: pitRate=%.4f haloRate=%.4f meanPit=%.4f meanHalo=%.4f\n",
                s, m[0], m[1], m[2], m[3]
            ))
        }
        // Contrast retention (peak 95% - dark 5%) in/out.
        val inL = ArrayList<Float>(); val outL = ArrayList<Float>()
        for (y in margin until h - margin) for (x in margin until w - margin) {
            inL.add(lumaOnDev(inF, x, y)); outL.add(lumaOf(devOut.at(x, y, 0), devOut.at(x, y, 1), devOut.at(x, y, 2)))
        }
        val cin = pctSorted(inL, 0.95) - pctSorted(inL, 0.05)
        val cout = pctSorted(outL, 0.95) - pctSorted(outL, 0.05)
        sb.append(String.format("contrast(p95-p5): in=%.3f out=%.3f retention=%.3f\n", cin, cout, cout / max(cin, 1e-6f)))

        // Coarse 64x64 luma maps so the ROI content is visible without a viewer.
        val nC = 64
        val inMap = FloatArray(nC * nC)
        val outMap = FloatArray(nC * nC)
        for (cy in 0 until nC) {
            for (cx in 0 until nC) {
                var si = 0f; var so = 0f; var c = 0
                val y0b = margin + ((h - 2 * margin) * cy) / nC
                val y1b = margin + ((h - 2 * margin) * (cy + 1)) / nC
                val x0b = margin + ((w - 2 * margin) * cx) / nC
                val x1b = margin + ((w - 2 * margin) * (cx + 1)) / nC
                for (y in y0b until y1b) for (x in x0b until x1b) {
                    si += lumaOnDev(inF, x, y)
                    so += lumaOf(devOut.at(x, y, 0), devOut.at(x, y, 1), devOut.at(x, y, 2))
                    c++
                }
                inMap[cy * nC + cx] = si / max(c, 1)
                outMap[cy * nC + cx] = so / max(c, 1)
            }
        }
        fun charOf(v: Float): Char = when {
            v < 0.02f -> '.'
            v < 0.05f -> ':'
            v < 0.15f -> 'o'
            v < 0.4f -> 'O'
            v < 0.7f -> '#'
            else -> '@'
        }
        fun mapLines(m: FloatArray): String = buildString {
            for (cy in 0 until nC) {
                for (cx in 0 until nC) append(charOf(m[cy * nC + cx]))
                append('\n')
            }
        }
        sb.append("\n-- input luma map (64x64, '-'=dark '@'=bright) --\n").append(mapLines(inMap))
            .append("\n-- output luma map --\n").append(mapLines(outMap))

        // Replay (mirror) output map, same grid, to compare with the device.
        val replayMap = FloatArray(nC * nC)
        for (cy in 0 until nC) {
            for (cx in 0 until nC) {
                var so = 0f; var c = 0
                val y0b = margin + ((h - 2 * margin) * cy) / nC
                val y1b = margin + ((h - 2 * margin) * (cy + 1)) / nC
                val x0b = margin + ((w - 2 * margin) * cx) / nC
                val x1b = margin + ((w - 2 * margin) * (cx + 1)) / nC
                for (y in y0b until y1b) for (x in x0b until x1b) {
                    so += lumaOf(replay.at(x, y, 0), replay.at(x, y, 1), replay.at(x, y, 2))
                    c++
                }
                replayMap[cy * nC + cx] = so / max(c, 1)
            }
        }
        sb.append("\n-- replay (mirror st=1) luma map --\n").append(mapLines(replayMap))
        // Pixel probes: input / device-sh / mirror-sh / replay-out / device-out.
        val probes = intArrayOf(80, 80, 300, 500, 512, 700, 900, 400, 100, 900)
        for (i in 0 until probes.size / 2) {
            val px = probes[i * 2]; val py = probes[i * 2 + 1]
            val li = lumaOnDev(inF, px, py)
            val ds = sigF[(py * w + px) * 4] / devScaleEffective
            val ms = mirrorSig[py][px]
            val lr = lumaOf(replay.at(px, py, 0), replay.at(px, py, 1), replay.at(px, py, 2))
            val lo = lumaOf(devOut.at(px, py, 0), devOut.at(px, py, 1), devOut.at(px, py, 2))
            val pro = floatArrayOf(
                devOut.at(px, py, 0), devOut.at(px, py, 1), devOut.at(px, py, 2), devOut.at(px, py, 3)
            )
            sb.append(String.format(
                "probe(%d,%d): in=%.3f devSig2=%.1f mirSig2=%.1f replay=%.3f devOut=%.3f devRgba=(%.3f,%.3f,%.3f,%.3f)\n",
                px, py, li, ds, ms, lr, lo, pro[0], pro[1], pro[2], pro[3]
            ))
        }
        val maxIn = pctSorted(inL, 1.0); val maxOut = pctSorted(outL, 1.0)
        val brightIn = inL.count { it > 0.3f }; val brightOut = outL.count { it > 0.3f }
        // Luma histograms (in/out) to see the real output distribution.
        val bins = doubleArrayOf(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)
        fun hist(vals: List<Float>): IntArray {
            val c = IntArray(bins.size)
            for (v in vals) {
                var i = 0
                while (i < bins.size && v >= bins[i]) i++
                if (i < bins.size) c[i]++
            }
            return c
        }
        val hi = hist(inL); val ho = hist(outL)
        sb.append("hist in :").append(StringBuilder().also { b ->
            for (i in bins.indices) b.append(" <%.1f:%d".format(bins[i], hi[i]))
        }.toString()).append('\n')
        sb.append("hist out:").append(StringBuilder().also { b ->
            for (i in bins.indices) b.append(" <%.1f:%d".format(bins[i], ho[i]))
        }.toString()).append('\n')
        // Raw 12x12 luma block of out at (400,400).
        sb.append("-- devOut luma block at (392..403,392..403) --\n")
        for (dy in 0 until 12) {
            var row = ""
            for (dx in 0 until 12) row += "%.3f ".format(
                lumaOf(devOut.at(392 + dx, 392 + dy, 0), devOut.at(392 + dx, 392 + dy, 1), devOut.at(392 + dx, 392 + dy, 2))
            )
            sb.append(row.trimEnd()).append('\n')
        }
        sb.append("\n-- devIn luma block at (392..403,392..403) --\n")
        for (dy in 0 until 12) {
            var row = ""
            for (dx in 0 until 12) row += "%.3f ".format(lumaOnDev(inF, 392 + dx, 392 + dy))
            sb.append(row.trimEnd()).append('\n')
        }
        // Mechanism probe: unwind one SWGF step at a few pixels (round 1).
        val (sMean, sSq) = statsOf(inP, inP, cfgDev.beta, cfgDev.winScale)
        sb.append("mechanism (s1, texture sigma2, round1):\n")
        // Brute-force sanity on the stats at (392,392).
        var boxSum = 0f; var boxN = 0f; var hSum = 0.0; var hN = 0
        for (dy in -9..9) for (dx in -9..9) {
            boxSum += lumaOf(inP.at(392 + dx, 392 + dy, 0), inP.at(392 + dx, 392 + dy, 1), inP.at(392 + dx, 392 + dy, 2))
            boxN += 1f
        }
        for (dx in -9..9) {
            hSum += lumaOf(inP.at(392 + dx, 392, 0), inP.at(392 + dx, 392, 1), inP.at(392 + dx, 392, 2))
            hN++
        }
sb.append(String.format(
            "  statsCheck(392,392): mean=%s var=%.5f | box19x19=%.4f horiz19=%.3f\n",
            sMean[392][392], sSq[392][392] - sMean[392][392] * sMean[392][392],
            boxSum / max(boxN, 1f), hSum / max(hN, 1)
        ))

        // Round-2 unwind on the actual replay pipeline.
        val r1P = runS5(inP, cfgDev, sigma2Dn = sigma2Dev, dbgXY = intArrayOf(393, 392)) // same as replay (st=1)
        val (sMean2, sSq2) = statsOf(r1P, inP, cfgDev.beta, cfgDev.winScale)
        for (pn in intArrayOf(392, 393)) {
            val x = pn; val y = 392
            val s2 = sigF[(y * w + x) * 4] / devScaleEffective
            val base = (s2 + cfgDev.sigmaDm2) * cfgDev.inverseRange2 * cfgDev.sigmaScale * cfgDev.epsBoost
            val epsY = cfgDev.lumaEpsScale * base * cfgDev.round2EpsMult
            sb.append(String.format(
                "  r2(%d,%d): in=%.3f sig2=%.1f epsY=%.2e sMean2=%.3f var2=%.5f | replay=%.3f devOut=%.3f\n",
                x, y, lumaOnDev(inF, x, y), s2, epsY,
                sMean2[y][x], sSq2[y][x] - sMean2[y][x] * sMean2[y][x],
                lumaOf(replay.at(x, y, 0), replay.at(x, y, 1), replay.at(x, y, 2)),
                lumaOf(devOut.at(x, y, 0), devOut.at(x, y, 1), devOut.at(x, y, 2))
            ))
        }
        sb.append(String.format("  replay(392..395,392): %s\n",
            (0 until 4).joinToString(" ") { i ->
                "%.3f".format(lumaOf(replay.at(392 + i, 392, 0), replay.at(392 + i, 392, 1), replay.at(392 + i, 392, 2)))
            }))

        // Full-channel round-2 unwind at (393,392): luma + chroma.
        fun yccInline(v: FloatArray): FloatArray { // matches mirror yccOf
            return floatArrayOf(lumaOf(v[0], v[1], v[2]), v[0] - v[2], 0.5f * (v[0] + v[2]) - v[1])
        }
        fun rgbInline(yc: FloatArray): FloatArray { // matches mirror rgbOf
            return floatArrayOf(
                yc[0] + 0.5f * yc[1] + 0.5f * yc[2],
                yc[0] - 0.5f * yc[2],
                yc[0] - 0.5f * yc[1] + 0.5f * yc[2]
            )
        }
        val x3 = 393; val y3 = 392
        val rgbIn3 = floatArrayOf(
            mix(r1P.at(x3, y3, 0), inP.at(x3, y3, 0), cfgDev.beta),
            mix(r1P.at(x3, y3, 1), inP.at(x3, y3, 1), cfgDev.beta),
            mix(r1P.at(x3, y3, 2), inP.at(x3, y3, 2), cfgDev.beta)
        )
        val ycc3 = yccInline(rgbIn3)
        val s2_3 = sigF[(y3 * w + x3) * 4] / devScaleEffective
        val base3 = (s2_3 + cfgDev.sigmaDm2) * cfgDev.inverseRange2 * cfgDev.sigmaScale * cfgDev.epsBoost
        val epsY3 = cfgDev.lumaEpsScale * base3 * cfgDev.round2EpsMult
        val epsC3 = cfgDev.chromaEpsScale * base3 * cfgDev.round2EpsMult
        val rChroma3 = min(max(2, glslRound(2.0f * cfgDev.winScale)), 8)
        val cm3 = chromaMeanAt(r1P, inP, cfgDev.beta, x3, y3, rChroma3)
        var best3 = 1.0e30f; var snd3 = 1.0e30f
        var bestM3 = 0f; var bestV3 = 0f; var sndM3 = 0f; var sndV3 = 0f
        for (k in 0..7) {
            val sx = (x3 + WINDOW_CENTERS[k][0] * cfgDev.winScale).coerceIn(0f, (w - 1).toFloat()).toInt()
            val sy = (y3 + WINDOW_CENTERS[k][1] * cfgDev.winScale).coerceIn(0f, (h - 1).toFloat()).toInt()
            val m = sMean2[sy][sx]
            val v = max(sSq2[sy][sx] - m * m, 0f)
            val sc = abs(m - ycc3[0]) / (v + epsY3)
            if (sc < best3) { snd3 = best3; sndM3 = bestM3; sndV3 = bestV3; best3 = sc; bestM3 = m; bestV3 = v }
            else if (sc < snd3) { snd3 = sc; sndM3 = m; sndV3 = v }
        }
        val wB3 = 1f / (best3 * best3 + 1e-12f); val wS3 = 1f / (snd3 * snd3 + 1e-12f)
        val pB3 = wB3 / (wB3 + wS3); val pS3 = wS3 / (wB3 + wS3)
        val meanY3 = pB3 * bestM3 + pS3 * sndM3
        val varY3 = pB3 * bestV3 + pS3 * sndV3
        val aY3 = varY3 / (varY3 + epsY3)
        val aC3 = varY3 / (varY3 + epsC3)
        val outY3 = aY3 * ycc3[0] + (1f - aY3) * meanY3
        val outC3a = aC3 * ycc3[1] + (1f - aC3) * cm3[0]
        val outC3b = aC3 * ycc3[2] + (1f - aC3) * cm3[1]
        val rgbF3 = rgbInline(floatArrayOf(outY3, outC3a, outC3b))
        sb.append(String.format(
            "  fullr2(%d,%d): yccIn=(%.3f,%.3f,%.3f) cm=(%.3f,%.3f) epsY=%.2e epsC=%.2e\n" +
                "    bestM=%.3f bestV=%.4f meanY=%.3f varY=%.4f aY=%.3f aC=%.3f\n" +
                "    rgbIn=(%.3f,%.3f,%.3f) rgbF=(%.3f,%.3f,%.3f) luma=%.3f replay=%.3f devOut=%.3f\n",
            x3, y3, ycc3[0], ycc3[1], ycc3[2], cm3[0], cm3[1], epsY3, epsC3,
            bestM3, bestV3, meanY3, varY3, aY3, aC3,
            rgbIn3[0], rgbIn3[1], rgbIn3[2], rgbF3[0], rgbF3[1], rgbF3[2],
            lumaOf(rgbF3[0], rgbF3[1], rgbF3[2]),
            lumaOf(replay.at(x3, y3, 0), replay.at(x3, y3, 1), replay.at(x3, y3, 2)),
            lumaOf(devOut.at(x3, y3, 0), devOut.at(x3, y3, 1), devOut.at(x3, y3, 2))
        ))
        for (pn in intArrayOf(392, 393, 394, 400)) {
            val y = 392
            val x = pn
            val s2 = sigF[(y * w + x) * 4] / devScaleEffective
            val base = (s2 + cfgDev.sigmaDm2) * cfgDev.inverseRange2 * cfgDev.sigmaScale * cfgDev.epsBoost
            val epsY = cfgDev.lumaEpsScale * base
            val li = lumaOnDev(inF, x, y)
            var best = 1.0e30f; var snd = 1.0e30f
            var bestM = 0f; var bestV = 0f; var sndM = 0f; var sndV = 0f
            for (k in 0..7) {
                val sx = (x + WINDOW_CENTERS[k][0] * cfgDev.winScale).coerceIn(0f, (w - 1).toFloat()).toInt()
                val sy = (y + WINDOW_CENTERS[k][1] * cfgDev.winScale).coerceIn(0f, (h - 1).toFloat()).toInt()
                val m = sMean[sy][sx]
                val v = max(sSq[sy][sx] - m * m, 0f)
                val sc = abs(m - li) / (v + epsY)
                if (sc < best) { snd = best; sndM = bestM; sndV = bestV; best = sc; bestM = m; bestV = v }
                else if (sc < snd) { snd = sc; sndM = m; sndV = v }
            }
            val wB = 1f / (best * best + 1e-12f); val wS = 1f / (snd * snd + 1e-12f)
            val pB = wB / (wB + wS); val pS = wS / (wB + wS)
            val meanY = pB * bestM + pS * sndM
            val varY = pB * bestV + pS * sndV
            val aY = varY / (varY + epsY)
            val outY = aY * li + (1f - aY) * meanY
            sb.append(String.format(
                "  (%d,%d): in=%.3f sig2=%.1f epsY=%.2e varY=%.3e aY=%.3f bestM=%.3f sndM=%.3f meanY=%.3f out=%.3f\n",
                x, y, li, s2, epsY, varY, aY, bestM, sndM, meanY, outY
            ))
        }
        sb.append(String.format(
            "deciles in : %s\n", List(9) { i -> "%6.3f".format(pctSorted(inL, (i + 1) / 10.0)) }.joinToString("")
        ))
        sb.append(String.format(
            "deciles out: %s\n", List(9) { i -> "%6.3f".format(pctSorted(outL, (i + 1) / 10.0)) }.joinToString("")
        ))
        sb.append(String.format(
            "max in=%.3f out=%.3f | pixels >0.3: in=%d out=%d (n=%.0f)\n",
            maxIn, maxOut, brightIn, brightOut, (inL.size).toFloat()
        ))

        // Find divergent bright pixels (device vs replay) and unwind one.
        data class DiffPx(val x: Int, val y: Int, val li: Float, val dr: Float, val rr: Float, val dd: Float)
        val difs = ArrayList<DiffPx>()
        for (y in margin until h - margin) {
            for (x in margin until w - margin) {
                val li = lumaOnDev(inF, x, y)
                if (li < 0.3f) continue
                val dr = lumaOf(devOut.at(x, y, 0), devOut.at(x, y, 1), devOut.at(x, y, 2))
                val rr = lumaOf(replay.at(x, y, 0), replay.at(x, y, 1), replay.at(x, y, 2))
                difs.add(DiffPx(x, y, li, dr, rr, dr - rr))
            }
        }
        val big = difs.filter { abs(it.dd) > 0.1f }
        sb.append(String.format(
            "bright px: %d | replay-devOut |diff|>0.1: %d of them | mean|dd|=%.3f\n",
            difs.size, big.size,
            if (difs.isEmpty()) 0f else difs.sumOf { abs(it.dd).toDouble() }.toFloat() / difs.size
        ))
        big.sortedByDescending { abs(it.dd) }.take(6).forEach {
            sb.append(String.format(
                "  diff(%d,%d): in=%.3f dev=%.3f replay=%.3f (d=%.3f)\n",
                it.x, it.y, it.li, it.dr, it.rr, it.dd
            ))
        }
        // Unwind the single largest-divergence letter pixel (round 1 + 2).
        val unw = difs.maxByOrNull { abs(it.dd) }
        if (unw != null && abs(unw.dd) > 0.1f) {
            val xu = unw.x; val yu = unw.y
            // Dump the 8 device window stats at this pixel to find which one the
            // device output (~ its mean) implies it selected.
            val (sm1u, sq1u) = statsOf(inP, inP, cfgDev.beta, cfgDev.winScale)
            val s2u = sigF[(yu * w + xu) * 4] / devScaleEffective
            val baseU = (s2u + cfgDev.sigmaDm2) * cfgDev.inverseRange2 * cfgDev.sigmaScale * cfgDev.epsBoost
            val epsY1u = cfgDev.lumaEpsScale * baseU
            val lU = lumaOnDev(inF, xu, yu)
            for (k in 0..7) {
                val sx = (xu + WINDOW_CENTERS[k][0] * cfgDev.winScale).coerceIn(0f, (w - 1).toFloat()).toInt()
                val sy = (yu + WINDOW_CENTERS[k][1] * cfgDev.winScale).coerceIn(0f, (h - 1).toFloat()).toInt()
                val m = sm1u[sy][sx]
                val v = max(sq1u[sy][sx] - m * m, 0f)
                val sc = abs(m - lU) / (v + epsY1u)
                sb.append(String.format(
                    "    w%d at (%d,%d): off=(%+d,%+d) m=%.4f v=%.5f score=%.2f%n",
                    k, sx, sy, WINDOW_CENTERS[k][0] * cfgDev.winScale.toInt(), WINDOW_CENTERS[k][1] * cfgDev.winScale.toInt(),
                    m, v, sc
                ))
            }
            sb.append(String.format(
                "    devOut=%.3f devRgba=(%.3f,%.3f,%.3f) (any window m near devOut?)\n",
                lumaOf(unw.dr, 0f, 0f).let { unw.dr },
                devOut.at(xu, yu, 0), devOut.at(xu, yu, 1), devOut.at(xu, yu, 2)
            ))
            // 19x19 luma neighbourhood (mirror's stats box R=9 at the pixel)
            val rrC = 9
            for (dyO in -rrC..rrC) {
                var rowS = ""
                for (dxO in -rrC..rrC) {
                    rowS += if (lumaOnDev(inF, xu + dxO, yu + dyO) > 0.4f) "#" else "."
                }
                sb.append("    b[" + (dyO + rrC) + "]: " + rowS + (if (dyO == 0) "  <- y=yu" else "") + "\n")
            }
            for (round in intArrayOf(1, 2)) {
                val epsMult = if (round == 1) 1f else cfgDev.round2EpsMult
                val (sm, sq) = if (round == 1) sm1u to sq1u else statsOf(replay, inP, cfgDev.beta, cfgDev.winScale)
                val s2 = sigF[(yu * w + xu) * 4] / devScaleEffective
                val base = (s2 + cfgDev.sigmaDm2) * cfgDev.inverseRange2 * cfgDev.sigmaScale * cfgDev.epsBoost
                val epsY = cfgDev.lumaEpsScale * base * epsMult
                val yc = yccOf(
                    mix(inP.at(xu, yu, 0), inP.at(xu, yu, 0), cfgDev.beta),
                    mix(inP.at(xu, yu, 1), inP.at(xu, yu, 1), cfgDev.beta),
                    mix(inP.at(xu, yu, 2), inP.at(xu, yu, 2), cfgDev.beta)
                )
                var bS = 1.0e30f; var sS = 1.0e30f; var bM = 0f; var bV = 0f; var sM = 0f; var sV = 0f
                for (k in 0..7) {
                    val sx = (xu + WINDOW_CENTERS[k][0] * cfgDev.winScale).coerceIn(0f, (w - 1).toFloat()).toInt()
                    val sy = (yu + WINDOW_CENTERS[k][1] * cfgDev.winScale).coerceIn(0f, (h - 1).toFloat()).toInt()
                    val m = sm[sy][sx]
                    val v = max(sq[sy][sx] - m * m, 0f)
                    val sc = abs(m - yc[0]) / (v + epsY)
                    if (sc < bS) { sS = bS; sM = bM; sV = bV; bS = sc; bM = m; bV = v }
                    else if (sc < sS) { sS = sc; sM = m; sV = v }
                }
                val wB = 1f / (bS * bS + 1e-12f); val wS = 1f / (sS * sS + 1e-12f)
                val pB = wB / (wB + wS); val pS = wS / (wB + wS)
                val meanY = pB * bM + pS * sM
                val varY = pB * bV + pS * sV
                val aY = varY / (varY + epsY)
                sb.append(String.format(
                    "  unw(%d,%d) r%d: in=%.3f inRgba=(%.3f,%.3f,%.3f) sig2=%.0f epsY=%.2e bestM=%.3f bestV=%.4f meanY=%.3f varY=%.4f aY=%.3f | replay=%.3f dev=%.3f\n",
                    xu, yu, round, lumaOnDev(inF, xu, yu), inP.at(xu, yu, 0), inP.at(xu, yu, 1), inP.at(xu, yu, 2),
                    s2, epsY, bM, bV, meanY, varY, aY,
                    lumaOf(replay.at(xu, yu, 0), replay.at(xu, yu, 1), replay.at(xu, yu, 2)),
                    lumaOf(devOut.at(xu, yu, 0), devOut.at(xu, yu, 1), devOut.at(xu, yu, 2))
                ))
            }
        }

        System.out.println(sb)
        java.io.File("build/s5_device_replay.txt").writeText(sb.toString())
    }

    // ------------------------------------------------------------------
    // repro: capture sigma-hat double-domain (sigma_hat^2*WR^2 in the texture vs the
    // main's 1/WR^2) inflates the capture epsilon by ~ WR^2, collapsing thin bright
    // strokes to the SWGF window mean (the photographed letter artifact -
    // validated against the device dump: (49,954) in=0.705 devOut=0.294).
    // With the domain fixed (single-domain sigma_hat^2) the strokes survive.
    // Guard: no S5 formula/iteration/DPC/S3/S4 changes - this only pins the
    // input-domain bug (SigmaHatShaderProgram :227) that the fix removes.
    // ------------------------------------------------------------------

    @Test
    fun captureDoubleDomainSigmaLetterCollapse() {
        val sb = StringBuilder()
        sb.append("=== capture sigma_hat double-domain epsilon collapse (letters) ===\n\n")

        // Dark near-black background + thin bright letter strokes, like the
        // real backlit-sign capture (bg~0.02, letters~0.68, WR=959, iso~1029).
        val size = 256
        val cfg = Cfg(winScale = 4.0f, epsBoost = 16f, whiteRange = 959f, iso = 1029, evGain2 = 1f)
        val sigmaPx = 1.2f / 959f
        val seed = 4242L
        val rnd = Random(seed)
        // Stroke mask: blocky letters ("IS-OP" style) of varying thickness.
        val lx0 = 64
        val ly0 = 88
        val mask = Array(size) { BooleanArray(size) }
        fun stroke(x: Int, y: Int, wd: Int, ht: Int) {
            for (dy in y until y + ht) for (dx in x until x + wd) {
                if (dx in 0 until size && dy in 0 until size) mask[dy][dx] = true
            }
        }
        // Column strokes (narrow + medium): [x, y, w, h] - thin like real letters
        // so even +-1-neighbour MAD straddles the ink (sigma_hat^2 ~ 100s of DN^2).
        val bars = arrayOf(
            intArrayOf(lx0, ly0, 4, 96), intArrayOf(lx0 + 16, ly0, 6, 96),
            intArrayOf(lx0 + 34, ly0, 8, 96), intArrayOf(lx0 + 54, ly0, 5, 96),
            intArrayOf(lx0 + 110, ly0, 34, 96) // wide bar: should survive both
        )
        for (b in bars) stroke(b[0], b[1], b[2], b[3])
    // Horizontal connectors make the strokes letter-like (thin arms).
        stroke(lx0, ly0 + 88, 64, 4)
        stroke(lx0 + 16, ly0 + 16, 4, 56) // mid bar of "E"
        stroke(lx0 + 34, ly0 + 16, 4, 56)
        stroke(lx0, ly0 + 16, 4, 56)

        val c = Array(size) { FloatArray(size * 3) }
        for (y in 0 until size) {
            for (x in 0 until size) {
                val bg = 0.02f
                val sig = if (mask[y][x]) 0.68f else bg
                for (ch in 0 until 3) c[y][x * 3 + ch] = sig + gauss(rnd) * sigmaPx
            }
        }
        val scene = Pln(size, size, c)
        fun lumaIn(x: Int, y: Int): Float =
            lumaOf(c[y][x * 3], c[y][x * 3 + 1], c[y][x * 3 + 2])
        // Shipped capture sigma_hat generator (axis-min, rgbMode); the double-domain
        // variant below emulates the historical sigma_hat^2*WR^2 capture texture.
        val madMap = madSigma2DnAxis(scene, cfg, "min")

        // All bright pixels (rims included - the real capture's letters are all
        // thin, so every letter pixel sits within a window's straddle reach).
        val lo = 6
        val hi = size - 6
        val bright = ArrayList<IntArray>()
        for (y in lo until hi) {
            for (x in lo until hi) {
                if (lumaIn(x, y) < 0.5f) continue
                bright.add(intArrayOf(x, y))
            }
        }

        // epsY actually seen at bright pixels (sigma_hat^2 in the sigma-hat texture):
        val s2Bright = bright.map { madMap[it[1]][it[0]].toDouble() }.average()
        val baseDev = (s2Bright + cfg.sigmaDm2).toFloat() * (1f / cfg.inverseRange2) * cfg.inverseRange2 * cfg.sigmaScale * cfg.epsBoost
        val epsYDev = cfg.lumaEpsScale * baseDev
        val baseFix = (s2Bright + cfg.sigmaDm2).toFloat() * cfg.inverseRange2 * cfg.sigmaScale * cfg.epsBoost
        val epsYFix = cfg.lumaEpsScale * baseFix
        sb.append(String.format("bright sigma_hat^2=%.0f DN^2 (mean) -> epsY device=%.2f (double-domain, >>var -> aY~0) vs fixed=%.2e (design)\n",
            s2Bright, epsYDev.toDouble(), epsYFix.toDouble()))

        fun retention(p: Pln): Double {
            var s = 0.0
            var n = 0.0
            for ((x, y) in bright) {
                if (x >= lx0 + 110) continue // thin strokes only (wide bar excluded)
                val li = lumaIn(x, y).toDouble()
                val ol = lumaOf(p.at(x, y, 0), p.at(x, y, 1), p.at(x, y, 2)).toDouble()
                if (li > 0.001) { s += ol / li; n++ }
            }
            return s / n
        }

        val outFix = runS5(scene, cfg, sigma2Dn = madMap)
        val outDev = runS5(scene, cfg, sigma2Dn = madMap, sigmaDoubleDomain = true)
        val retFix = retention(outFix)
        val retDev = retention(outDev)
        val wideFix = bright.filter { it[0] >= lx0 + 110 && it[0] <= lx0 + 143 }
        val wideDev = wideFix.map { val li = lumaIn(it[0], it[1]).toDouble()
            lumaOf(outDev.at(it[0], it[1], 0), outDev.at(it[0], it[1], 1), outDev.at(it[0], it[1], 2)).toDouble() / li }
            .average()
        sb.append(String.format(
            "retention (letter luma out/in): fixed=%.3f device-domain=%.3f | wide-bar(34px) under device-domain=%.3f (should stay high)\n",
            retFix, retDev, wideDev
        ))

        java.io.File("build/s5noise_anchor.txt").let { f ->
            val current = if (f.exists()) f.readText() else ""
            f.writeText(current + sb.toString())
        }

        // The bug under test is exactly the device's epsilon-inflation; the fixed
        // domain must keep the letters and the inflated domain must crush them.
        assertTrue(
            "fixed sigma_hat domain must keep letter luma (retention=$retFix, want >=0.85)",
            retFix >= 0.85
        )
        assertTrue(
            "device double-domain must collapse letter luma (retention=$retDev, want <=0.65)",
            retDev <= 0.65
        )
        assertTrue(
            "double-domain must Crush letters at least 25% harder than fixed (dev=$retDev fix=$retFix)",
            retDev <= retFix - 0.25
        )
        assertTrue(
            "wide stroke must survive both modes (wide-dev=$wideDev, want >=0.9)",
            wideDev >= 0.9
        )
    }

    // ------------------------------------------------------------------
    // Residual (post-fix, design operating point): the fixed-domain replay
    // still pits a fraction of thin bright pixels (device "visible on close
    // inspection"). Identify WHICH pixels pit under the design epsilon and WHY:
    // per-pixel sigma_hat^2/epsY/varY/aY + window choice, then the same on a thin-stroke
    // synthetic (parity). Diagnostic only; the S5 math is unchanged.
    // ------------------------------------------------------------------

    @Test
    fun residualPitMechanism() {
        val dir = requireDeviceDump(arrayOf("s5_in.f32", "s5_sigma.f32"))
        val inF = readRgbaF32(dir, "s5_in.f32")
        val sigF = readRgbaF32(dir, "s5_sigma.f32")
        val w = dumpW; val h = dumpH
        val margin = 48
        val sb = StringBuilder()
        sb.append("=== residual pit mechanism (fixed domain, design epsilon, PRE-axis-min dump) ===\n")
        val cfgDev = Cfg(iso = 1029, winScale = 4.266667f, epsBoost = 16f, strength = 1f)

        val inP = Pln(w, h, Array(h) { y -> FloatArray(w * 3) { i ->
            val px = i / 3
            inF[(y * w + px) * 4 + (i - px * 3)]
        } })
        val mirrorSig = madSigma2Dn(inP, cfgDev)
        // Auto-detect the sigma-hat domain (pre-fix dump = sigma_hat^2*WR^2, post-fix = raw).
        val wr2 = 959f * 959f
        val darkMir = ArrayList<Float>()
        for (y in margin until h - margin) for (x in margin until w - margin) {
            if (lumaOnDev(inF, x, y) < 0.3f) darkMir.add(mirrorSig[y][x])
        }
        val mirDk = pctSorted(darkMir, 0.5)
        val darkSig = ArrayList<Float>()
        for (y in margin until h - margin) for (x in margin until w - margin) {
            if (lumaOnDev(inF, x, y) < 0.3f) darkSig.add(sigF[(y * w + x) * 4])
        }
        val rD = pctSorted(darkSig, 0.5) / wr2
        val scaleDev = if (abs(rD - mirDk) < 0.05f * mirDk) wr2 else 1f
        sb.append(String.format(
            "sigma-hat domain: %s (device sigma_hat^2 = R%s)\n",
            if (scaleDev > 1f) "DOUBLE (pre-fix)" else "SINGLE (post-fix)",
            if (scaleDev > 1f) "/WR^2" else ""
        ))

        val sigma2Dev = Array(h) { y -> FloatArray(w) { x -> sigF[(y * w + x) * 4] / scaleDev } }
        val replay = runS5(inP, cfgDev, sigma2Dn = sigma2Dev)

        // Build the pit set (bright pixels darkened >8% by the DESIGN filter).
        data class Pit(val x: Int, val y: Int, val li: Float, val lp: Float, val s2: Float)
        val pits = ArrayList<Pit>()
        val kept = ArrayList<Pit>()
        for (y in margin until h - margin) for (x in margin until w - margin) {
            val li = lumaOnDev(inF, x, y)
            if (li < 0.6f) continue
            val lp = lumaOf(replay.at(x, y, 0), replay.at(x, y, 1), replay.at(x, y, 2))
            val s2 = sigma2Dev[y][x]
            (if (lp < li - 0.08f) pits else kept).add(Pit(x, y, li, lp, s2))
        }
        sb.append(String.format(
            "bright=%d kept=%d pit=%d (rate=%.3f) meanPitDepth=%.3f\n",
            pits.size + kept.size, kept.size, pits.size,
            pits.size.toFloat() / max(pits.size + kept.size, 1),
            if (pits.isEmpty()) 0f else pits.sumOf { (it.li - it.lp).toDouble() }.toFloat() / pits.size
        ))
        // Group diagnostics: pit vs kept, sigma_hat^2 + epsY + in-luma.
        fun s2p50(v: List<Pit>) = pctSorted(v.map { it.s2 }, 0.5)
        fun epsAt(s2: Float) = cfgDev.lumaEpsScale * (s2 + cfgDev.sigmaDm2) * cfgDev.inverseRange2 * cfgDev.sigmaScale * cfgDev.epsBoost
        sb.append(String.format(
            "pit : n=%4d SigmaHat2p50=%7.0f epsYp50=%.2e in-mean=%.3f in-p50=%.3f\n",
            pits.size, s2p50(pits).toDouble(), epsAt(s2p50(pits)).toDouble(),
            if (pits.isEmpty()) -1f else pits.sumOf { it.li.toDouble() }.toFloat() / pits.size, pits.map { it.li }.let { pctSorted(it, 0.5) }
        ))
        sb.append(String.format(
            "kept: n=%4d SigmaHat2p50=%7.0f epsYp50=%.2e in-mean=%.3f in-p50=%.3f\n",
            kept.size, s2p50(kept).toDouble(), epsAt(s2p50(kept)).toDouble(),
            if (kept.isEmpty()) -1f else kept.sumOf { it.li.toDouble() }.toFloat() / kept.size, kept.map { it.li }.let { pctSorted(it, 0.5) }
        ))

        // Unwind top 10 pit pixels: every window's (m, v, score) + the fused aY.
        val (sm1u, sq1u) = statsOf(inP, inP, cfgDev.beta, cfgDev.winScale)
        pits.sortedByDescending { it.li - it.lp }.take(10).forEach { p ->
            val epsY = epsAt(p.s2)
            val wB = FloatArray(8); val wS = FloatArray(8); val wM = FloatArray(8); val wV = FloatArray(8)
            for (k in 0..7) {
                val sx = (p.x + WINDOW_CENTERS[k][0] * cfgDev.winScale).coerceIn(0f, (w - 1).toFloat()).toInt()
                val sy = (p.y + WINDOW_CENTERS[k][1] * cfgDev.winScale).coerceIn(0f, (h - 1).toFloat()).toInt()
                wM[k] = sm1u[sy][sx]; wV[k] = max(sq1u[sy][sx] - wM[k] * wM[k], 0f)
                wS[k] = abs(wM[k] - p.li) / (wV[k] + epsY)
            }
            val order = (0..7).sortedBy { wS[it] }
            val b1 = order[0]; val b2 = order[1]
            val v1 = wV[b1]; val v2 = wV[b2]
            val aY = if (p.li < 1e-4f) 1f else {
                val wwB = 1f / (wS[b1] * wS[b1] + 1e-12f); val wwS = 1f / (wS[b2] * wS[b2] + 1e-12f)
                val pB = wwB / (wwB + wwS); val pS = wwS / (wwB + wwS)
                val varY = pB * v1 + pS * v2
                val meanY = pB * wM[b1] + pS * wM[b2]
                sb.append(String.format(
                    "  pit(%d,%d) in=%.3f->%.3f sig2=%.0f epsY=%.2e | win1[%d] %s m=%.3f v=%.4f s=%.2f  win2[%d] %s m=%.3f v=%.4f s=%.2f  fusedM=%.3f fusedV=%.4f\n",
                    p.x, p.y, p.li, p.lp, p.s2, epsY,
                    b1, "off=(" + (WINDOW_CENTERS[b1][0] * cfgDev.winScale.toInt()) + "," + (WINDOW_CENTERS[b1][1] * cfgDev.winScale.toInt()) + ")",
                    wM[b1], v1, wS[b1],
                    b2, "off=(" + (WINDOW_CENTERS[b2][0] * cfgDev.winScale.toInt()) + "," + (WINDOW_CENTERS[b2][1] * cfgDev.winScale.toInt()) + ")",
                    wM[b2], v2, wS[b2],
                    meanY, varY
                ))
                varY / (varY + epsY)
            }
            sb.append(String.format("  -> aY=%.3f\n", aY))
        }

        java.io.File("build/s5noise_anchor.txt").let { f ->
            val current = if (f.exists()) f.readText() else ""
            f.writeText(current + sb.toString())
        }
        System.out.println(sb)
    }

    // ------------------------------------------------------------------
    // Residual-pit candidates: the pit pixels are driven by edge-inflated
    // sigma_hat^2 (10-22k DN^2 vs 1.4k interior) raising the design epsilonY.  The epsilon/kappa/
    // iteration/DPC/S3/S4 levers are frozen, so the only admissible lever is
    // the sigma_hat GENERATOR feeding epsilon.  Compare the shipped MAD-8 sigma_hat against two
    // direction-aware variants (min / lower-median over the 4 axes) on the
    // device capture AND on a synthetic letter scene (parity), measuring
    // pit count/depth, kept-region stability and thin-stroke retention.
    // ------------------------------------------------------------------

    private fun replayPitStats(
        img: Pln, sig: Array<FloatArray>, cfg: Cfg, margin: Int,
        thinExclude: ((Int, Int) -> Boolean)?, lumaIn: (Int, Int) -> Float
    ): Triple<IntArray, Float, Float> {
        val out = runS5(img, cfg, sigma2Dn = sig)
        var nPit = 0; var depth = 0.0; var nBright = 0
        for (y in margin until img.h - margin) for (x in margin until img.w - margin) {
            val li = lumaIn(x, y)
            if (li < 0.6f) continue
            if (thinExclude != null && thinExclude(x, y)) continue
            nBright++
            val lp = lumaOf(out.at(x, y, 0), out.at(x, y, 1), out.at(x, y, 2))
            if (lp < li - 0.08f) { nPit++; depth += li - lp }
        }
        return Triple(intArrayOf(nPit, nBright), if (nPit == 0) 0f else (depth / nPit).toFloat(), 0f)
    }

    @Test
    fun residualPitCandidates() {
        val dir = requireDeviceDump(arrayOf("s5_in.f32", "s5_sigma.f32"))
        val inF = readRgbaF32(dir, "s5_in.f32")
        val sigF = readRgbaF32(dir, "s5_sigma.f32")
        val w = dumpW; val h = dumpH
        val margin = 48
        val sb = StringBuilder()
        sb.append("=== residual pit candidates (sigma_hat generator variants; dump predates the axis-min GLSL) ===\n")
        val cfgDev = Cfg(iso = 1029, winScale = 4.266667f, epsBoost = 16f, strength = 1f)

        val inP = Pln(w, h, Array(h) { y -> FloatArray(w * 3) { i ->
            val px = i / 3
            inF[(y * w + px) * 4 + (i - px * 3)]
        } })
        fun lumaInDev(x: Int, y: Int) = lumaOnDev(inF, x, y)

        // Device dump is pre-fix (R = sigma_hat^2*WR^2); align to single domain.
        val wr2 = 959f * 959f
        val devSigma = Array(h) { y -> FloatArray(w) { x -> sigF[(y * w + x) * 4] / wr2 } }

        // Former pit pixels (from residualPitMechanism) to track sigma_hat^2/epsY/out under each variant.
        val formerPits = arrayOf(
            intArrayOf(49, 955), intArrayOf(51, 935), intArrayOf(48, 934),
            intArrayOf(50, 935), intArrayOf(51, 954), intArrayOf(52, 935)
        )
        fun epsAt(s2: Float) = cfgDev.lumaEpsScale * (s2 + cfgDev.sigmaDm2) * cfgDev.inverseRange2 * cfgDev.sigmaScale * cfgDev.epsBoost

        val variants = linkedMapOf<String, Array<FloatArray>>(
            "device(fixed)" to devSigma,
            "mirror-full" to madSigma2Dn(inP, cfgDev),
            "axis-min" to madSigma2DnAxis(inP, cfgDev, "min"),
            "axis-med" to madSigma2DnAxis(inP, cfgDev, "med")
        )
        sb.append(String.format("variant            pits/bright depth  pitSigmaHat2p50  pit-epsY   formerPit out [+] formerPit SigmaHat2 (device dump = pre-axis-min GLSL)\n"))
        for ((name, sig) in variants) {
            val (pc, depth, _) = replayPitStats(inP, sig, cfgDev, margin, null) { x, y -> lumaInDev(x, y) } as Triple<IntArray, Float, Float>
            val out = runS5(inP, cfgDev, sigma2Dn = sig)
            val pitSig = ArrayList<Float>()
            for (y in margin until h - margin) for (x in margin until w - margin) {
                if (lumaInDev(x, y) < 0.6f) continue
                if (lumaOf(out.at(x, y, 0), out.at(x, y, 1), out.at(x, y, 2)) < lumaInDev(x, y) - 0.08f) pitSig.add(sig[y][x])
            }
            val pitP50 = if (pitSig.isEmpty()) -1f else pctSorted(pitSig, 0.5)
            val epsP50 = if (pitSig.isEmpty()) -1f else epsAt(pitP50)
            val fpOut = formerPits.map { lumaOf(out.at(it[0], it[1], 0), out.at(it[0], it[1], 1), out.at(it[0], it[1], 2)) }
            val fpSig = formerPits.map { sig[it[1]][it[0]] }
            sb.append(String.format(
                "%-16s %3d/%-6d %.3f  %7.0f  %.2e  [%s]  [%s]\n",
                name, pc[0], pc[1], depth,
                pitP50.toDouble(), epsP50.toDouble(),
                fpOut.joinToString { String.format("%.2f", it) },
                fpSig.joinToString { String.format("%.0f", it) }
            ))
        }

        // ---- Synthetic letter scene (parity) ----
        // Flat fills give sigma_hat^2 ~ 56 at the rim only: the REAL printed letters carry
        // pattern/halftone (sigma_hat^2 10-22k DN^2 on the capture); a faithful headless
        // repro needs textured ink. Two texture levels: "tex" (grain+2px checker,
        // tuned so edge sigma_hat^2 ~ 10k) reproduces the rounded dot; "flat" shows the
        // texture-free baseline.
        val size = 256
        val cfgSyn = Cfg(winScale = 4.0f, epsBoost = 16f, whiteRange = 959f, iso = 1029, evGain2 = 1f)
        val sigmaPx = 1.2f / 959f
        val lx0 = 64; val ly0 = 88
        fun buildLetterScene(texGrain: Float, texCheck: Float): Pair<Pln, Array<IntArray>> {
            val rnd = Random(4242L)
            val mask = Array(size) { BooleanArray(size) }
            fun stroke(x: Int, y: Int, wd: Int, ht: Int) {
                for (dy in y until y + ht) for (dx in x until x + wd) if (dx in 0 until size && dy in 0 until size) mask[dy][dx] = true
            }
            val bars = arrayOf(
                intArrayOf(lx0, ly0, 4, 96), intArrayOf(lx0 + 16, ly0, 6, 96),
                intArrayOf(lx0 + 34, ly0, 8, 96), intArrayOf(lx0 + 54, ly0, 5, 96),
                intArrayOf(lx0 + 110, ly0, 34, 96)
            )
            for (b in bars) stroke(b[0], b[1], b[2], b[3])
            stroke(lx0, ly0 + 88, 64, 4)
            val c = Array(size) { FloatArray(size * 3) }
            for (y in 0 until size) for (x in 0 until size) {
                val fg = mask[y][x]
                var v = if (fg) 0.68f else 0.02f + gauss(rnd) * sigmaPx
                if (fg && texGrain > 0f) {
                    v += gauss(rnd) * texGrain
                    if (((x ushr 1) + (y ushr 1)) and 1 == 0) v += texCheck else v -= texCheck
                }
                for (ch in 0 until 3) c[y][x * 3 + ch] = v
            }
            val scene = Pln(size, size, c)
            val bright = ArrayList<IntArray>()
            for (y in 6 until size - 6) for (x in 6 until size - 6) {
                if (lumaOf(c[y][x * 3], c[y][x * 3 + 1], c[y][x * 3 + 2]) > 0.5f) bright.add(intArrayOf(x, y))
            }
            return scene to bright.toTypedArray()
        }
        fun letterMetrics(scene: Pln, bright: Array<IntArray>, sig: Array<FloatArray>,
                          name: String, sb: StringBuilder) {
            val out = runS5(scene, cfgSyn, sigma2Dn = sig)
            var tPit = 0; var tDepth = 0.0; var tN = 0
            var wideSum = 0.0; var wideN = 0
            var bgDrift = 0.0; var bgN = 0
            val thinSig = ArrayList<Float>(); val edgeSig = ArrayList<Float>()
            val lo = 6; val hi = scene.h - 6
            for (y in lo until hi) for (x in lo until hi) {
                val li = lumaOf(scene.at(x, y, 0), scene.at(x, y, 1), scene.at(x, y, 2))
                val lp = lumaOf(out.at(x, y, 0), out.at(x, y, 1), out.at(x, y, 2))
                val inWide = x >= lx0 + 110 && x <= lx0 + 143 && y in ly0 until ly0 + 96
                if (li > 0.5f) {
                    if (inWide) { wideSum += lp / li; wideN++ }
                    else {
                        tN++
                        if (lp < li - 0.08f) { tPit++; tDepth += li - lp }
                        thinSig.add(sig[y][x])
                        val rim = if (x - 1 >= 0 && x + 1 < scene.w)
                            (scene.at(x - 1, y, 0) > 0.5f) xor (scene.at(x + 1, y, 0) > 0.5f) else false
                        if (rim) edgeSig.add(sig[y][x])
                    }
                } else if (lp > li) { bgDrift += lp - li; bgN++ }
            }
            sb.append(String.format(
                "%-13s thinPit/bright=%3d/%-6d depth=%.3f wideRet=%.3f bgDrift=%.1e thinSigmaHat2p50=%7.0f edgeSigmaHat2p50=%7.0f keptBright=%.4f\n",
                name, tPit, tN, if (tPit == 0) 0f else (tDepth / tPit).toFloat(),
                if (wideN == 0) -1f else (wideSum / wideN).toFloat(),
                if (bgN == 0) -1f else (bgDrift / bgN).toFloat(),
                if (thinSig.isEmpty()) -1f else pctSorted(thinSig, 0.5).toDouble(),
                if (edgeSig.isEmpty()) -1f else pctSorted(edgeSig, 0.5).toDouble(),
                lumaOf(out.at(0, 0, 0), out.at(0, 0, 1), out.at(0, 0, 2))
            ))
        }
        sb.append(String.format("\nsynthetic letter scene (parity; texture reproduces capture-like edge sigma_hat^2):\n"))
        sb.append(String.format("scene          sigmaHat-gen   thinPit/bright  depth   wideRet  bgDrift   thinSigmaHat2p50 edgeSigmaHat2p50   keptBright(0,0)\n"))
        for (fgTex in floatArrayOf(0.05f, 0.0f)) {
            val (scene, bright) = buildLetterScene(fgTex, 0.035f)
            val label = if (fgTex > 0f) "tex (grain .05+.035ck)" else "flat (no texture)  "
            val full = madSigma2Dn(scene, cfgSyn)
            val axMin = madSigma2DnAxis(scene, cfgSyn, "min")
            letterMetrics(scene, bright, full, label + " | full    ", sb)
            letterMetrics(scene, bright, axMin, label + " | axis-min", sb)
            sb.append("\n")
        }

        java.io.File("build/s5noise_anchor.txt").let { f ->
            val current = if (f.exists()) f.readText() else ""
            f.writeText(current + sb.toString())
        }
        System.out.println(sb)
    }

    // ------------------------------------------------------------------
    // Requirements for the capture sigma_hat generator change (axis-min, rgbMode only):
    //  (a) repro - textured letter strokes pit >=300 pixels under the MAD-8
    //      generator (the close-inspection residual), and axis-min drops that to
    //      15% or fewer (device headless-equivalent: 6->2 on the real dump).
    //  (b) no calibration regression - on FLAT noise axis-min is floor-anchored
    //      and must stay within +-10% of the MAD-8 measured sigma_hat (the
    //      ISO-model floor dominates both on flats).
    //  (c) wide strokes and dark background must not drift either way.
    // ------------------------------------------------------------------

    private fun texturedLetterScene(size: Int, texGrain: Float, texCheck: Float): Pair<Pln, Array<IntArray>> {
        val rnd = Random(4242L)
        val lx0 = 64; val ly0 = 88
        val mask = Array(size) { BooleanArray(size) }
        fun stroke(x: Int, y: Int, wd: Int, ht: Int) {
            for (dy in y until y + ht) for (dx in x until x + wd) if (dx in 0 until size && dy in 0 until size) mask[dy][dx] = true
        }
        val bars = arrayOf(
            intArrayOf(lx0, ly0, 4, 96), intArrayOf(lx0 + 16, ly0, 6, 96),
            intArrayOf(lx0 + 34, ly0, 8, 96), intArrayOf(lx0 + 54, ly0, 5, 96),
            intArrayOf(lx0 + 110, ly0, 34, 96)
        )
        for (b in bars) stroke(b[0], b[1], b[2], b[3])
        stroke(lx0, ly0 + 88, 64, 4)
        val c = Array(size) { FloatArray(size * 3) }
        for (y in 0 until size) for (x in 0 until size) {
            var v = if (mask[y][x]) 0.68f else 0.02f + gauss(rnd) * (1.2f / 959f)
            if (mask[y][x] && texGrain > 0f) {
                v += gauss(rnd) * texGrain
                if (((x ushr 1) + (y ushr 1)) and 1 == 0) v += texCheck else v -= texCheck
            }
            for (ch in 0 until 3) c[y][x * 3 + ch] = v
        }
        val scene = Pln(size, size, c)
        val bright = ArrayList<IntArray>()
        for (y in 6 until size - 6) for (x in 6 until size - 6) {
            if (lumaOf(c[y][x * 3], c[y][x * 3 + 1], c[y][x * 3 + 2]) > 0.5f) bright.add(intArrayOf(x, y))
        }
        return scene to bright.toTypedArray()
    }

    private fun letterPitAndWide(
        scene: Pln, sig: Array<FloatArray>, cfg: Cfg, bright: Array<IntArray>, lx0: Int, ly0: Int
    ): FloatArray {
        // [0] thin pit count, [1] thin bright count, [2] wide retention, [3] bg drift max
        val out = runS5(scene, cfg, sigma2Dn = sig)
        var tPit = 0; var tN = 0; var wideSum = 0.0; var wideN = 0; var bgMax = 0f
        for (y in 6 until scene.h - 6) for (x in 6 until scene.w - 6) {
            val li = lumaOf(scene.at(x, y, 0), scene.at(x, y, 1), scene.at(x, y, 2))
            val lp = lumaOf(out.at(x, y, 0), out.at(x, y, 1), out.at(x, y, 2))
            val inWide = x >= lx0 + 110 && x <= lx0 + 143 && y in ly0 until ly0 + 96
            if (li > 0.5f) {
                if (inWide) { wideSum += lp / li; wideN++ } else { tN++; if (lp < li - 0.08f) tPit++ }
            } else if (lp > li && lp - li > bgMax) bgMax = lp - li
        }
        return floatArrayOf(tPit.toFloat(), tN.toFloat(),
            if (wideN == 0) 1f else (wideSum / wideN).toFloat(), bgMax)
    }

    @Test
    fun sigmaHatAxisMinResidualL1() {
        val size = 256
        val cfg = Cfg(winScale = 4.0f, epsBoost = 16f, whiteRange = 959f, iso = 1029, evGain2 = 1f)
        val (scene, bright) = texturedLetterScene(size, 0.05f, 0.035f)
        val sb = StringBuilder()
        sb.append("=== sigmaHat axis-min: residual + calibration guard ===\n")
        val sigMad = madSigma2Dn(scene, cfg)
        val sigAx = madSigma2DnAxis(scene, cfg, "min")
        val m = letterPitAndWide(scene, sigMad, cfg, bright, 64, 88)
        val a = letterPitAndWide(scene, sigAx, cfg, bright, 64, 88)
        sb.append(String.format(
            "textured letters: MAD-8 pits=%d/%d wideRet=%.3f bgMax=%.4f | axis-min pits=%d/%d wideRet=%.3f bgMax=%.4f\n",
            m[0].toInt(), m[1].toInt(), m[2], m[3], a[0].toInt(), a[1].toInt(), a[2], a[3]
        ))

        // (a) reproduction must be real, and axis-min must crush it.
        assertTrue("repro needs >=300 pits under MAD-8 (got ${m[0].toInt()})", m[0] >= 300f)
        assertTrue(
            "axis-min must cut pits to <=15% of MAD-8 (mad=${m[0].toInt()} ax=${a[0].toInt()})",
            a[0] <= m[0] * 0.15f
        )
        assertTrue(
            "axis-min wide strokes must survive (wideRet=${a[2]})",
            a[2] >= 0.9f
        )
        assertTrue(
            "axis-min must not darken background more than MAD-8 (madBg=${m[3]} axBg=${a[3]})",
            a[3] <= m[3] + 0.01f
        )

        // (b) flat-noise calibration: axis-min vs MAD-8 within +-10%.
        val flatCfg = Cfg(iso = 3200, evGain2 = 1f)
        val sign = 0.5f
        val signalDN = sign * (1f / sqrt(flatCfg.inverseRange2))
        val sigma2Ref = flatCfg.isoModelA * signalDN + flatCfg.isoModelB
        val sigmaChPx = sqrt(sigma2Ref) / (1f / sqrt(flatCfg.inverseRange2))
        val flat = noiseBlock(128, sign, sigmaChPx, seed = 7001L)
        val sMad = madSigma2Dn(flat, flatCfg)
        val sAx = madSigma2DnAxis(flat, flatCfg, "min")
        val lo = 10; val hi = 118
        val madVals = ArrayList<Float>(); val axVals = ArrayList<Float>()
        for (y in lo until hi) for (x in lo until hi) { madVals.add(sMad[y][x]); axVals.add(sAx[y][x]) }
        val madP50 = pctSorted(madVals, 0.5); val axP50 = pctSorted(axVals, 0.5)
        val rel = abs(axP50 - madP50) / max(madP50, 1e-6f)
        sb.append(String.format(
            "flat noise (iso=3200): MAD SigmaHat2p50=%.1f axis-min SigmaHat2p50=%.1f rel=%.3f (model=%.1f)\n",
            madP50.toDouble(), axP50.toDouble(), rel.toDouble(), sigma2Ref.toDouble()
        ))
        assertTrue("flat-noise SigmaHat2 must stay within +-10% of MAD-8 (rel=$rel)", rel <= 0.10f)

        java.io.File("build/s5noise_anchor.txt").let { f ->
            val current = if (f.exists()) f.readText() else ""
            f.writeText(current + sb.toString() + "\n")
        }
        System.out.println(sb)
    }
}