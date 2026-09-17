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
import org.junit.Test

// Mirrored GLSL + host constants (OutNrShaderProgram / PreviewRenderer.runStage5).
private val WINDOW_CENTERS = arrayOf(
    intArrayOf(-2, 0), intArrayOf(2, 0), intArrayOf(0, -2), intArrayOf(0, 2),
    intArrayOf(-2, -2), intArrayOf(2, -2), intArrayOf(-2, 2), intArrayOf(2, 2)
)
private const val WY = 0.25f
private const val WG = 0.5f
private const val WB = 0.25f

// Host defaults (PreviewRenderer.runStage5) — the calibration targets.
private const val LUMA_EPS_SCALE = 1.4f
private const val CHROMA_EPS_SCALE = 64.0f
private const val BETA = 0.3f
private const val SIGMA_DM2 = 10f
private const val SIGMA_SCALE = 1f / 32f
private const val ROUND2_EPS_MULT = 1.96f // (κ×1.4)², ε ∝ σ̂²

/**
 * Pure-noise-block anchor calibration for the Stage-5 double pass (plan §5 T1)
 * plus the double-pass regression guarantees (§4 acceptance).
 *
 * This is a headless JVM mirror of the Stage-5 GLSL (OutNrShaderProgram
 * STATS_H/STATS_V/MAIN_FRAGMENT) — the same approach as S5ReproTest — extended
 * to (a) run BOTH SWGF iterations with β noise-return, (b) control every host
 * parameter (lumaEpsScale/chromaEpsScale, round-2 ε multiplier, β, winScale,
 * epsBoost, evGain2 fold, ISO model A/B), and (c) read out noise attenuation on
 * flat pure-noise blocks.
 *
 * Two measurement families (mirroring doc §3 Stage 5 + §5 T1):
 *  1. ANCHOR identity: on a flat block whose variance σ² is known exactly,
 *     set ε = κ²·σ² and check measured single-pass attenuation against the
 *     theory table  a = 1/(1+κ²), att = σ√(a²+(1−a²)/|ω|), |ω|=25 —
 *     at κ = 1.0 / 1.2 / 1.5, plus the round-2 κ×1.4 tier and the two-round
 *     cascade product (the doc's ideal lower bound), with and without the
 *     β=0.3 noise return.
 *  2. PIPELINE config: real host epsilon math (ISO model · evGain2 ·
 *     sigmaScale · epsBoost · (σ̂²+σ_dm²)/whiteRange²) run on flat blocks at
 *     representative σ truth tiers, across ISO × signal × path (preview,
 *     capture 1:1 boxAA-off, capture reduced boxAA-on, EV-comp low end).
 *     Reports the effective κ = √(ε/σ²_truth) and the measured cascade
 *     attenuation so the model→truth ratio is visible per tier.
 *
 * Acceptance (plan §4 Phase A + this task):
 *  (a) highlight-texture/fine-edge preservation is not over-smoothed by the
 *      second pass (grating-modulation retention + edge rise-width floors);
 *  (b) the two-round cascade attenuation is measured and tabled by
 *      κ = 1.0/1.2/1.5 including the β-return's actual effect;
 *  (c) slider strength 0 stays bit-exact bypass (host returns the input);
 *  (d) S5ReproTest's bigSeams==0 guarantee carries over to the double pass.
 *
 * Note on "measured": all numbers here are computed with the faithful JVM
 * mirror (deterministic fixed seeds), i.e. the offline digital-calibration
 * leg of T1; on-device flat-field confirmation is T6/T1-device follow-up.
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

    /** Flat-field block, neutral RGB, per-channel IID Gaussian noise σ_px. */
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

    /** Separable 5-tap (win-scaled) box stats of the β-composed luma: (mean, 2nd moment). */
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

    /** Dense chroma box mean of the β-composed input at the pixel (base-centric). */
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

    /** Host-math ε in image units (PreviewRenderer.runStage5 + MAIN_FRAGMENT). */
    private fun epsBaseImageUnits(cfg: Cfg, yccY: Float): Float {
        val whiteRangeI = 1f / sqrt(cfg.inverseRange2)
        val signalDN = max(yccY * whiteRangeI, 0f)
        val sigma2 = max(cfg.isoModelA * signalDN + cfg.isoModelB, 0f) * cfg.evGain2
        return (sigma2 + cfg.sigmaDm2) * cfg.inverseRange2 * cfg.sigmaScale * cfg.epsBoost
    }

    private fun cfgEpsY(cfg: Cfg, yccY: Float): Float = cfg.lumaEpsScale * epsBaseImageUnits(cfg, yccY)
    private fun cfgEpsC(cfg: Cfg, yccY: Float): Float = cfg.chromaEpsScale * epsBaseImageUnits(cfg, yccY)

    // GLSL round() is half-away-from-zero; Kotlin roundToInt() is half-up.
    private fun glslRound(v: Float): Int = if (v >= 0f) (v + 0.5f).toInt() else (v - 0.5f).toInt()

    /**
     * One SWGF iteration mirroring MAIN_FRAGMENT:
     *  - epsY/epsC taken from the host math unless anchor overrides are given
     *    (anchor mode: ε = κ²·σ² directly, the §3 attenuation identity test);
     *  - epsMult multiplies both ε (round-2 (κ×1.4)² fold);
     *  - β composes in/out of stats, chroma box and final strength mix.
     */
    private fun s5Pass(
        inImg: Pln, baseImg: Pln, cfg: Cfg, epsMult: Float = 1f,
        epsYAnchor: Float? = null, epsCAnchor: Float? = null
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
                } else {
                    val base = epsBaseImageUnits(cfg, yccIn[0])
                    epsY = cfg.lumaEpsScale * base * epsMult
                    epsC = cfg.chromaEpsScale * base * epsMult
                }

                // 8 SWGF side windows, distance ∝ |mean−y|/(var+ε), soft-2 fusion.
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
                out[y][x * 3 + 0] = mix(rgbIn[0], rgbF[0], st)
                out[y][x * 3 + 1] = mix(rgbIn[1], rgbF[1], st)
                out[y][x * 3 + 2] = mix(rgbIn[2], rgbF[2], st)
            }
        }
        return Pln(w, h, out)
    }

    /** Run the double pass as the host does (host skips at strength 0 → bit-exact). */
    private fun runS5(img: Pln, cfg: Cfg, epsYAnchor: Float? = null, epsCAnchor: Float? = null): Pln {
        if (cfg.strength <= 0f) return img
        val r1 = s5Pass(img, img, cfg, 1f, epsYAnchor, epsCAnchor)
        if (cfg.iterations < 2) return r1
        return s5Pass(r1, img, cfg, cfg.round2EpsMult, epsYAnchor, epsCAnchor)
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

    /** Doc §3 attenuation identity: a = 1/(1+κ²); att = √(a² + (1−a²)/|ω|), |ω|=25. */
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

    /** Edge rise-width: number of columns over which a 10%→90% crossing takes place. */
    private fun edgeRiseWidth(p: Pln, y: Int, x0: Int, x1: Int, lo: Float, hi: Float): Int {
        val raw = (x0..x1).map { x -> lumaOf(p.at(x, y, 0), p.at(x, y, 1), p.at(x, y, 2)) }
        // Box-smooth radius 1 to suppress per-pixel noise in the level crossing.
        val prof = FloatArray(raw.size)
        for (i in prof.indices) {
            prof[i] = (raw[max(i - 1, 0)] + raw[i] + raw[min(i + 1, raw.lastIndex)]) / 3f
        }
        // lo/hi are the band LEVELS on the two sides; the transition is the
        // span of columns inside the 25–75% of the lo→hi step. Grating swings
        // (±0.04) stay outside this band by construction.
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
    // 1. ANCHOR identity + cascade (κ = 1.0 / 1.2 / 1.5).
    // ------------------------------------------------------------------

    @Test
    fun anchorIdentityAndCascade() {
        val sign = 0.5f
        val size = 160
        val lo = 8
        val hi = size - 8
        val sigmaChPx = 8f / 959f                 // 8 DN noise → image units
        val sigmaLumaInj = 0.61237244f * sigmaChPx // luma noise of neutral RGB
        val my = sigmaLumaInj.toDouble()
        val sb = StringBuilder()
        sb.append("=== S5 double-pass pure-noise anchor (JVM mirror, midpoint signal=${sign}) ===\n\n")
        sb.append(String.format("%-5s %-7s %-7s %-8s %-8s %-8s %-8s %-8s %-8s\n",
            "κ", "a", "LB1", "meas1", "LB2(κ≥1.4)", "meas2an", "LBcas", "meas-cas(β0)", "meas-cas(β.3)"))
        var allOk = true

        for (k in doubleArrayOf(1.0, 1.2, 1.5)) {
            val k2 = k * 1.4
            val lb1 = theoryAtt(k)
            val lb2 = theoryAtt(k2)
            val lbCas = lb1 * lb2

            // ε anchors in image units: ε = κ²·σ².
            val epsY = (k * k * my * my).toFloat()
            val epsC = epsY * (CHROMA_EPS_SCALE / LUMA_EPS_SCALE) // keep Y:C design ratio

            val img = noiseBlock(size, sign, sigmaChPx, seed = 1001 + (k * 100).toLong())
            val sigLumaIn = lumaStd(img, lo, hi).toDouble()

            // Round 1 single pass.
            val r1 = s5Pass(img, img, Cfg(beta = BETA, lumaEpsScale = 1.4f, chromaEpsScale = 64f),
                1f, epsY, epsC)
            val meas1 = lumaStd(r1, lo, hi).toDouble() / sigLumaIn

            // Round 2 anchor on a FRESH white block at κ2 (single-pass identity).
            val fresh = noiseBlock(size, sign, sigmaChPx, seed = 2001 + (k * 100).toLong())
            val r2fresh = s5Pass(fresh, fresh, Cfg(beta = 0f), 1f,
                epsY * ROUND2_EPS_MULT, epsC * ROUND2_EPS_MULT)
            val meas2an = lumaStd(r2fresh, lo, hi).toDouble() / lumaStd(fresh, lo, hi).toDouble()

            // Cascade without β (round 2 blends round-1 with itself, ε×1.96).
            val cas0 = runS5(img, Cfg(beta = 0f, lumaEpsScale = 1.4f, chromaEpsScale = 64f),
                epsY, epsC)
            val measCas0 = lumaStd(cas0, lo, hi).toDouble() / sigLumaIn

            // Cascade with β=0.3 noise return (shipped flavor).
            val casB = runS5(img, Cfg(beta = BETA, lumaEpsScale = 1.4f, chromaEpsScale = 64f),
                epsY, epsC)
            val measCasB = lumaStd(casB, lo, hi).toDouble() / sigLumaIn

            sb.append(String.format("%-5s %-7s %-7s %-8s %-8s %-8s %-8s %-8s %-8s\n",
                f3(k), f3(1.0 / (1.0 + k * k)), f4(lb1), f4(meas1), f4(lb2),
                f4(meas2an), f4(lbCas), f4(measCas0), f4(measCasB)))

            // Doc §5 T1 verdict: measured ≥ theory lower bound; β raises the
            // residual vs the no-β cascade (noise return), so the β gap matches
            // the re-injection rather than exceeding the bounds.
            assertTrue("κ=$k round-1 measured $meas1 must stay at/above LB $lb1 (mirror tol)",
                meas1 >= lb1 * 0.92)
            assertTrue("κ=$k round-2(×1.4) fresh-block measured $meas2an must stay at/above LB $lb2",
                meas2an >= lb2 * 0.88)
            assertTrue("κ=$k no-β cascade measured $measCas0 must stay at/above LB $lbCas",
                measCas0 >= lbCas * 0.85)
            assertTrue("κ=$k β cascade measured $measCasB must stay at/above LB $lbCas",
                measCasB >= lbCas * 0.85)
            // β=0.3 must not DOUBLE-COUNT: the cascade with β must not be much
            // below the no-β one (the return re-injects noise rather than
            // adding attenuation) — clamp sanity band instead of a hard sign.
            assertTrue("κ=$k β=0.3 cascade $measCasB must not drop far below no-β $measCas0",
                measCasB >= measCas0 - 0.03)
            if (meas1 < lb1 * 0.92 || meas2an < lb2 * 0.88 || measCas0 < lbCas * 0.85 ||
                measCasB < lbCas * 0.85 || measCasB < measCas0 - 0.03
            ) allOk = false
        }
        sb.append("\nANCHOR VERDICT: all κ tiers measured ≥ theory LB (tolerances above) → ")
        sb.append(if (allOk) "PASS\n\n" else "FAIL\n\n")

        java.io.File("build/s5noise_anchor.txt").let { f ->
            val current = if (f.exists()) f.readText() else ""
            f.writeText(current + sb.toString())
        }
        assertTrue("anchor identity: all tiers within tolerance of the doc theory lower bounds", allOk)
    }

    // ------------------------------------------------------------------
    // 2. PIPELINE-config cascade attenuation: ISO × signal × path.
    // ------------------------------------------------------------------

    @Test
    fun pipelineConfigCascade() {
        val sb = StringBuilder()
        sb.append("=== Pipeline-config double-pass cascade (ε from ISO model, host math) ===\n\n")

        data class Path(val name: String, val winScale: Float, val epsBoost: Float, val evGain2: Float, val size: Int, val lo: Int)

        val preview = Path("preview", 1f, 1f, 1f, 152, 8)
        val evLow = Path("EV-low(1/8)", 1f, 1f, 0.125f, 152, 8)
        val cap1_1 = Path("capture-1:1", 3f, 16f, 1f, 208, 18)
        val capBoxAa = Path("capture-boxAA", 1.5f, 1f, 1f, 176, 12)

        var allOk = true
        val rows = StringBuilder()
        rows.append(String.format("%-9s %-7s %-5s %-6s %-8s %-8s %-8s %-8s %-9s %-9s %-9s\n",
            "path", "ISO(seg)", "sig", "σDN", "κ_eff", "att1", "att2", "casβ.3", "cas(β0)", "C1att", "C2att"))

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
                    // The pipeline σ̂ model predicts variance on this signal; the
                    // block injects σ truth = one representative tier (8 DN).
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
                    // The filter must always DENOISE (att ≤ 1) and never collapse
                    // below the residual floor (att not ≪ 0.05).
                    assertTrue("${path.name} ISO=$iso sig=$sig cascade β attenuates (≤1) got $casB", casB <= 1.001f)
                    assertTrue("${path.name} ISO=$iso sig=$sig cascade β not degenerate ($casB)", casB >= 0.04f)
                    if (path.name == "capture-1:1" || path.name == "capture-boxAA") {
                        allOk = allOk && casB <= 1.001f && casB >= 0.04f
                    }
                }
            }
        }

        // Detailed σ-tier scan at the doc collision ISO (3200, mid signal):
        // how κ_eff and the cascade vary as the truth dictates.
        rows.append("\n-- ISO 3200 mid-signal σ-tier scan (preview) --\n")
        rows.append(String.format("%-6s %-8s %-8s %-8s %-9s\n", "σDN", "κ_eff", "att1", "att2", "casβ.3"))
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
            assertTrue("ISO3200 mid σ=${sigD.toInt()} cascade must remain denoising (cas=$cas≤1)", cas <= 1.001f)
            assertTrue("ISO3200 mid σ=${sigD.toInt()} must not collapse (cas=$cas≥0.04)", cas >= 0.04f)
        }

        sb.append(rows)
        // The collision case: the S5D evidence showed the post-S3+boxAA residual is
        // σ̂/√32 (model σ̂ ≈ 32× residual variance). Inject that residual truth
        // and require κ_eff to land in an engaged-but-bounded window. Note the
        // measured κ sits above the nominal lumaEpsScale=1.4 coefficient by
        // the luma-mix fold √(1/0.375)=1.63 (Y=0.25R+0.5G+0.25B → var_Y=
        // 0.375·var_ch); the appendix-C table records this realization as-is.
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
        sb.append("ISO3200 mid-signal σ_res=σ̂/√32=${f2(sigmaResDn)}DN  κ_eff=${f2(kappa3200)}  casβ.3=${f4(cas3200)}\n")
        sb.append("(coefficient √lumaEpsScale=${f2(sqrt(0.375) * kappa3200)}; Y-mix fold √(1/0.375)=${f2(sqrt(1.0 / 0.375))}×)\n")
        allOk = allOk && kappa3200 in 1.0..2.5 && cas3200 in 0.03..0.75
        assertTrue("ISO3200 mid-signal σ_res=${f2(sigmaResDn)}DN κ_eff=$kappa3200 must be engaged in [1.0,2.5] (collision anchor)",
            kappa3200 in 1.0..2.5)
        assertTrue("ISO3200 mid-signal collision cascade $cas3200 must denoise (0.03..0.75)",
            cas3200 in 0.03..0.75)

        sb.append("\nPIPELINE VERDICT: $allOk\n")
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
        val cfg = Cfg(iso = 3200) // double pass, preview, β=0.3
        // σ tiers: 10 DN = general mid-scale scene noise; 3 DN ≈ σ̂/√32 =
        // the post-S3 collision residual (highest realized κ_eff).
        data class Patch(val y0: Int, val y1: Int, val x0: Int, val x1: Int, val period: Int)
        val patches = listOf(
            Patch(8, 20, 62, 112, 16),
            Patch(100, 112, 62, 112, 6),
            Patch(8, 20, 120, 170, 16),
            Patch(100, 112, 120, 170, 6)
        )
        val sb = StringBuilder()
        sb.append("=== Texture preservation (ISO 3200, double pass, β=0.3) ===\n")

        for (sigmaDN in intArrayOf(10, 3)) {
            val sigmaPx = sigmaDN.toFloat() / 959f
            val img = textureScene(size, bandXs, sigmaPx, pc = 16, ac = 0.03f, pf = 6, af = 0.04f, seed = 4242L)
            val r1 = s5Pass(img, img, cfg)
            val r2 = s5Pass(r1, img, cfg, cfg.round2EpsMult)
            sb.append(String.format("-- σ=%d DN --\n", sigmaDN))
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
                    assertTrue("σ=$sigmaDN highlight $kind: round-2 retention $ret2 must stay ≥ $floor",
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
                sb.append(String.format("σ=%d %-12s %-14s %-6s %-6s\n", sigmaDN, e.label, "$inW px", "$r1W px", "$r2W px"))
                assertTrue("σ=$sigmaDN ${e.label} edge: r1 must stay sharp (r1=$r1W px)", r1W <= 4)
                assertTrue("σ=$sigmaDN ${e.label} edge: r2 must stay sharp (r2=$r2W px)", r2W <= 4)
            }

            // Cross-check noise: the flat shadow band (no grating) must be
            // denoised. At the 10 DN tier the realized κ is light (≈0.5), so
            // only no-amplification is required; at the collision-residual
            // tier (σ ≈ σ̂/√32 → κ_eff ≈ 1.9) the cascade must actually clean.
            val flatIn2 = lumaStd(img, 88, 152, 6, 52)
            val flatR2 = lumaStd(r2, 88, 152, 6, 52)
            sb.append(String.format("σ=%d flat-shadow σ: in %.6f → r2 %.6f (att %.3f)\n",
                sigmaDN, flatIn2, flatR2, flatR2 / flatIn2))
            if (sigmaDN == 10) {
                assertTrue("σ=$sigmaDN flat shadow must not amplify (att=${flatR2 / flatIn2})",
                    flatR2 <= flatIn2 * 1.02f)
            } else {
                assertTrue("σ=$sigmaDN flat shadow must be denoised by double pass (att=${flatR2 / flatIn2})",
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
        // runS5 short-circuits at strength 0 → out IS img (bit-exact); also
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
        // |ΔΔC1| > 0.03 counts as a big seam).
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
}