package com.agx.camera.gpu

import kotlin.math.abs
import kotlin.math.PI
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random
import org.junit.Assert.assertTrue
import org.junit.Test

private val SENSOR_W2 = 96
private val SENSOR_H2 = 96

/**
 * Faithful headless mirror of the DEPLOYED preview S1/S3 path
 * (BayerShaderProgram DEMOSAIC_FRAGMENT_SHADER + S3_PACK_FRAGMENT_SHADER):
 *   - view transform T applied to a_texCoord (mirror/flip) by the vertex shader,
 *     while gl_FragCoord stays an untransformed output-grid index;
 *   - transform-aware S3_PACK boxed/anchored write (min/max sorted footprint);
 *   - the exact denoisedSampleRaw read (tc = T^-1*(cell-origin)/crop * G + 0.5
 *     with CLAMP_TO_EDGE GL_LINEAR weighting, channel = safePhase(cell));
 *   - gating per the LIVE preview (PreviewRenderer's previewZoomK gate): the
 *     sparse mosaic feeds the demosaic only at k<=2.  Within that range the
 *     DPC+GF chain (runDpcAndGfChain, spec-accurate S1/S3) produces the mosaic
 *     when the DPC shaders are ready, else the S3_PACK fallback (drawS3Pack)
 *     when they are not — this mirror models exactly that fallback.  At k>2
 *     the mosaic is NOT used at all: drawDemosaic runs the inline per-sample
 *     sampleSameColorNR (which reads raw sensor values at any zoom and catches
 *     hot pixels), with dpStrength/rawNrStrength passed through.  The mosaic
 *     read is a manual 4-tap texelFetch bilinear (GL_LINEAR is ILLEGAL on
 *     RGBA32F in ES 3.0; the device silently fell back to NEAREST, so the
 *     per-texel quantized phase estimates showed up as the "magenta mosaic"
 *     (zoom-in) and "more noise" (zoom-out) — the read is now format-safe).
 *     The shader-side k>=16 inline defer is subsumed by the k>2 gate in the
 *     live path and is modelled for completeness only.
 *
 * Asserts the two user-visible invariants headlessly before any APK handover:
 *   (a) a non-zero slider NEVER adds noise (gNoise(s1|s3) <= gNoise(0) );
 *   (b) zoom-in shows no magenta mosaic (magErr(s1|s3) bounded by baseline).
 */
class S3PreviewGLReproTest {

    private val SW = 96
    private val SH = 96
    private val SENSOR_CLIP = 1023
    private val SENSOR_BLACK = 64
    private val isoA = 0.0067f * 800f / 100f
    private val isoB = (0.33f * 800f / 100f) * (0.33f * 800f / 100f)
    private val BA_COLOR_MAP = intArrayOf(0, 1, 1, 2)
    private val CLIP_SCALAR = (SENSOR_CLIP - SENSOR_BLACK).toFloat()

    /** Mask knee: flat-branch "structure vs measured σ̂" threshold multiple (probe-only). */
    private var maskK = 3.0f

    /** Mode-3 regional flat-pull ω band: ω = clamp((ratio-lo)/(hi-lo),0,1).
     *  lo=1.5 sits at the pure-noise p95 (1.48) so noise keeps ω≈0 (bit-identical
     *  iavg pull); hi=2.5 is past the bulk of both foliage and line ratios.
     *  See the module comment. */
    private var omegaLo = 1.5f
    private var omegaHi = 2.5f

    private fun phase(x: Int, y: Int): Int = abs(x % 2) + abs(y % 2) * 2

    private fun sensorVal(v: ShortArray, x: Int, y: Int): Float {
        val cx = x.coerceIn(0, SW - 1)
        val cy = y.coerceIn(0, SH - 1)
        val raw = v[cy * SW + cx].toInt() and 0x3FF
        return max((raw - SENSOR_BLACK).toFloat(), 0f)
    }

    /** Mirror of the Stage-2 σ̂ pass (SigmaHatShaderProgram FRAGMENT_SHADER),
     *  evaluated from the RAW sensor values: per-CFA-phase MAD over the 8
     *  same-phase neighbouring cells (grid offsets ±1, clamped), mean over
     *  phases, floored at the ISO-model signal curve.  A masked S3 experiment
     *  re-keys the flat/structure decision on this measured noise instead of
     *  the under-reporting model σ. */
    private fun sigmaHatAt(v: ShortArray, sx: Int, sy: Int): Float {
        val cellsW = SW / 2
        val cellsH = SH / 2
        val cx = (sx / 2).coerceIn(0, cellsW - 1)
        val cy = (sy / 2).coerceIn(0, cellsH - 1)
        val nox = intArrayOf(1, -1, 0, 0, 1, -1, 1, -1)
        val noy = intArrayOf(0, 0, 1, -1, 1, 1, -1, -1)
        var sig2Total = 0f
        for (p in 0..3) {
            val px = p and 1
            val py = p shr 1
            val vals = FloatArray(8)
            for (k in 0..7) {
                val nx = (cx + nox[k]).coerceIn(0, cellsW - 1)
                val ny = (cy + noy[k]).coerceIn(0, cellsH - 1)
                vals[k] = sensorVal(v, 2 * nx + px, 2 * ny + py)
            }
            java.util.Arrays.sort(vals)
            val med = (vals[3] + vals[4]) * 0.5f
            val devs = FloatArray(8) { k -> abs(vals[k] - med) }
            java.util.Arrays.sort(devs)
            val mad = (devs[3] + devs[4]) * 0.5f
            sig2Total += 2.1981f * mad * mad
        }
        val sig2Mean = sig2Total * 0.25f
        val meanSignal = (sensorVal(v, 2 * cx, 2 * cy) + sensorVal(v, 2 * cx + 1, 2 * cy) +
            sensorVal(v, 2 * cx, 2 * cy + 1) + sensorVal(v, 2 * cx + 1, 2 * cy + 1)) * 0.25f
        val floor2 = max(isoA * max(meanSignal, 0f) + isoB, 1.0e-6f)
        val out2 = max(sig2Mean, floor2)
        return sqrt(max(out2, 1.0e-6f))
    }

    /** max |mean(nb cell) − mean(center cell)| over the 8 surrounding cells —
     *  a coarse (region-level) structure measure: on pure noise it collapses as
     *  σ/√N and its max-over-8 tail is bounded, on texture it tracks the
     *  structure that spans blocks. */
    private fun coarseDevAt(v: ShortArray, sx: Int, sy: Int): Float {
        val cellsW = SW / 2
        val cellsH = SH / 2
        val cx = (sx / 2).coerceIn(0, cellsW - 1)
        val cy = (sy / 2).coerceIn(0, cellsH - 1)
        fun cellMean(c: Int, r: Int): Float {
            val cc = c.coerceIn(0, cellsW - 1)
            val rr = r.coerceIn(0, cellsH - 1)
            return (sensorVal(v, 2 * cc, 2 * rr) + sensorVal(v, 2 * cc + 1, 2 * rr) +
                sensorVal(v, 2 * cc, 2 * rr + 1) + sensorVal(v, 2 * cc + 1, 2 * rr + 1)) * 0.25f
        }
        val center = cellMean(cx, cy)
        var m = 0f
        for (dy in -1..1) for (dx in -1..1) {
            if (dx == 0 && dy == 0) continue
            m = max(m, abs(cellMean(cx + dx, cy + dy) - center))
        }
        return m
    }

    /** Literal GL sampleSameColorNR (12-tap a-trim + DPC guard + S3 clip blend + directional I_D).
     *  At nrRadius<4 mirrors the preview-only 4-tap (step-2 axis neighbours)
     *  used when the demosaic box stays 4x4; capture/box<4 keep the 12-tap.
     *  maskSigma (measurement only, not shipped) re-keys the flat/structure
     *  threshold on the measured σ̂ so the "noisy region" mask can be probed. */
    private fun sameColorNR(v: ShortArray, sx: Int, sy: Int, s1: Float, s3: Float, nrRadius: Int = 4, mode: Int = 0, mCoarse: Float? = null): Float {
        val c = sensorVal(v, sx, sy)
        if (s1 <= 0f && s3 <= 0f) return c
        val (mn, mx, iavg) = if (nrRadius >= 4) {
            val nE = sensorVal(v, sx + 2, sy)
            val nW = sensorVal(v, sx - 2, sy)
            val nN = sensorVal(v, sx, sy - 2)
            val nS = sensorVal(v, sx, sy + 2)
            val nNE = sensorVal(v, sx + 2, sy - 2)
            val nNW = sensorVal(v, sx - 2, sy - 2)
            val nSE = sensorVal(v, sx + 2, sy + 2)
            val nSW = sensorVal(v, sx - 2, sy + 2)
            val nEE = sensorVal(v, sx + 4, sy)
            val nWW = sensorVal(v, sx - 4, sy)
            val nNN = sensorVal(v, sx, sy - 4)
            val nSS = sensorVal(v, sx, sy + 4)
            val smn = min(min(min(min(min(nE, nW), min(nN, nS)), min(nNE, nNW)), min(nSE, nSW)),
                min(min(nEE, nWW), min(nNN, nSS)))
            val smx = max(max(max(max(max(nE, nW), max(nN, nS)), max(nNE, nNW)), max(nSE, nSW)),
                max(max(nEE, nWW), max(nNN, nSS)))
            Triple(smn, smx,
                (nE + nW + nN + nS + nNE + nNW + nSE + nSW + nEE + nWW + nNN + nSS - smn - smx) / 10f)
        } else {
            val nE = sensorVal(v, sx + 2, sy)
            val nW = sensorVal(v, sx - 2, sy)
            val nN = sensorVal(v, sx, sy - 2)
            val nS = sensorVal(v, sx, sy + 2)
            val smn = min(min(nE, nW), min(nN, nS))
            val smx = max(max(nE, nW), max(nN, nS))
            Triple(smn, smx, (nE + nW + nN + nS - smn - smx) / 2f)
        }
        val sigma = sqrt(max(isoA * max(iavg, 0f) + isoB, 1f))
        val band = max((0.1f + 0.3f * s1) * max(iavg, 0f), (2f + 2f * s1) * sigma)

        // Directional I_D: smoothest direction pair (12-tap path only)
        var iDir = iavg
        // M2-style structure measure (mirror of the GLSL), hoisted so the S3
        // blend can share it with the S1 defect gate.
        var maxNb = 0f
        var minDev = 0f
        if (nrRadius >= 4) {
            val nE = sensorVal(v, sx + 2, sy)
            val nW = sensorVal(v, sx - 2, sy)
            val nN = sensorVal(v, sx, sy - 2)
            val nS = sensorVal(v, sx, sy + 2)
            val nNE = sensorVal(v, sx + 2, sy - 2)
            val nNW = sensorVal(v, sx - 2, sy - 2)
            val nSE = sensorVal(v, sx + 2, sy + 2)
            val nSW = sensorVal(v, sx - 2, sy + 2)
            val aH = (nE + nW) * 0.5f
            val aV = (nN + nS) * 0.5f
            val a45 = (nNE + nSW) * 0.5f
            val a135 = (nSE + nNW) * 0.5f
            val dH = abs(nE - nW)
            val dV = abs(nN - nS)
            val d45 = abs(nNE - nSW)
            val d135 = abs(nSE - nNW)
            iDir = aH
            if (dV < dH && dV <= d45 && dV <= d135) iDir = aV
            else if (d45 < dH && d45 <= dV && d45 <= d135) iDir = a45
            else if (d135 < dH && d135 <= dV && d135 <= d45) iDir = a135

            val nEE = sensorVal(v, sx + 4, sy)
            val nWW = sensorVal(v, sx - 4, sy)
            val nNN = sensorVal(v, sx, sy - 4)
            val nSS = sensorVal(v, sx, sy + 4)
            val tN = nNN + nNE + nNW + c - minOf(minOf(nNN, nNE), minOf(nNW, c)) - maxOf(maxOf(nNN, nNE), maxOf(nNW, c))
            val devN = abs(nN - tN * 0.5f)
            val tS = nSS + nSE + nSW + c - minOf(minOf(nSS, nSE), minOf(nSW, c)) - maxOf(maxOf(nSS, nSE), maxOf(nSW, c))
            val devS = abs(nS - tS * 0.5f)
            val tE = nEE + nNE + nSE + c - minOf(minOf(nEE, nNE), minOf(nSE, c)) - maxOf(maxOf(nEE, nNE), maxOf(nSE, c))
            val devE = abs(nE - tE * 0.5f)
            val tW = nWW + nNW + nSW + c - minOf(minOf(nWW, nNW), minOf(nSW, c)) - maxOf(maxOf(nWW, nNW), maxOf(nSW, c))
            val devW = abs(nW - tW * 0.5f)
            maxNb = maxOf(maxOf(devN, devS), maxOf(devE, devW))
            minDev = minOf(devN, devS, devE, devW)
        }

        var center = c
        if (max(s1, 0.85f * s3) > 0f) {
            val hot = (c > mx) && (c - iavg) > band
            val cold = (c < mn) && (iavg - c) > band
            if (hot || cold) {
                if (nrRadius >= 4) {
                    if (abs(c - iavg) > 6.0f * maxNb) {
                        center = c + (iDir - c) * max(max(s1, 0.85f * s3), 0.98f)
                    }
                } else {
                    center = c + (iavg - c) * max(max(s1, 0.85f * s3), 0.98f)
                }
            }
        }
        // Similarity-weighted (bilateral) S3 blend — mirror of the GLSL.
        // Full weight within 2.5*sigma of the corrected centre, taper to zero
        // by 5*sigma; the pull stays at 0.98*s3 (weights carry the edge protection).
        // See the GLSL for rationale.
        // maxNb is the structure detector: flat noise (maxNb <= 6*sigma_model)
        // keeps the robust α-trim pull (bit-identical to the baseline S3, so
        // the monotonicity gate can't pop on the noisy flat scene, where the
        // minDev window would collapse to ~0 in its low tail and starve the
        // denoising); a real edge/line (maxNb > 6*sigma_model) switches to the
        // weight window keyed on sg = max(sigma, 6*minDev), the smallest axis
        // deviation, which stays at the noise floor along a feature — the
        // cross-line taps land outside the window and the line is protected.
        var bavg = iavg
        if (nrRadius >= 4) {
            val sg = max(sigma, 6.0f * minDev)
            val flatLimit = if (mode == 1) maskK * sigmaHatAt(v, sx, sy) else 6.0f * sigma
            if (maxNb <= flatLimit) {
                if (mode == 2) {
                    // Directional flat target (probe): see the module comment —
                    // dead end: the noise minDev/maxNb tail overlaps a line's
                    // band, so any ω>0 bleeds onto noise and pops the boxAA step.
                    val r = if (maxNb > 1e-6f) minDev / maxNb else 1f
                    val omega = ((0.95f - r) / (0.95f - 0.75f)).coerceIn(0f, 1f)
                    bavg = iavg + (iDir - iavg) * omega
                } else if (mode == 3) {
                    // Regional (coarse/fine) flat target: the ω band slides with
                    // omegaLo/omegaHi so the pure-noise tail (gate p95=1.48) can
                    // be excluded (ω=0 -> bit-identical iavg pull) while foliage
                    // texture (80% >= 1.5) keeps ω=1.  See the module comment.
                    // mCoarse is the per-output-pixel coarse dev threaded from
                    // renderPreview (one read per fragment); null (probe sites)
                    // falls back to the per-sample read.
                    val ratio = if (maxNb > 1e-6f) (mCoarse ?: coarseDevAt(v, sx, sy)) / maxNb else 2f
                    val omega = ((ratio - omegaLo) / (omegaHi - omegaLo)).coerceIn(0f, 1f)
                    bavg = iavg + (iDir - iavg) * omega
                } else if (mode == 4) {
                    // Constant directional flat pull (probe): no discriminator at
                    // all — the flat branch blends the α-trim target toward the
                    // along-feature pair mean iDir by a fixed 0.5.  On a line
                    // iDir tracks the feature (pull loses the off-line taps so the
                    // line is preserved); on isotropic noise iDir ≈ iavg (both are
                    // trimmed means of the same taps) so the pull is nearly a no-op
                    // and the gate margin should hold.  If it does, this is the
                    // shippable answer: no σ̂, no extra pass, no discriminator.
                    bavg = iavg + (iDir - iavg) * 0.5f
                } else {
                    // flat patch: bit-identical to the baseline robust pull
                    bavg = iavg
                }
            } else {
                val tau = 2.5f * sg
                val invTau = 1f / tau
                var wsum = 0f
                bavg = 0f
                fun tapW(tx: Int, ty: Int) {
                    val t = sensorVal(v, sx + tx, sy + ty)
                    val w = (1f - max(abs(t - center) - tau, 0f) * invTau).coerceIn(0f, 1f)
                    bavg += w * t
                    wsum += w
                }
                tapW(2, 0); tapW(-2, 0); tapW(0, -2); tapW(0, 2)
                tapW(2, -2); tapW(-2, -2); tapW(2, 2); tapW(-2, 2)
                tapW(4, 0); tapW(-4, 0); tapW(0, -4); tapW(0, 4)
                if (wsum > 0f) bavg /= wsum else bavg = center
            }
        }
        return if (c < CLIP_SCALAR) center + (bavg - center) * (0.98f * s3) else center
    }

    // ------------------------------------------------------------------
    // Scenes
    // ------------------------------------------------------------------

    private fun buildFlatNoise(seed: Int, base: Float, sigma: Float): ShortArray {
        val rnd = Random(seed)
        val s = ShortArray(SW * SH)
        for (y in 0 until SH) for (x in 0 until SW) {
            val g = base + (rnd.nextFloat() - 0.5f) * 2f * sigma
            s[y * SW + x] = (SENSOR_BLACK + g).roundToInt().coerceIn(0, SENSOR_CLIP).toShort()
        }
        return s
    }

    /** Per-phase-imbalanced flat field (like a real sensor under neutral light:
     *  the red photosite base sits lower than green, blue in between).  Equal
     *  bases would make boxedBlend's cross-phase iavg drag a fixed point, which
     *  is why the equal-base flat scene can never show magenta. */
    private fun buildImbalancedFlat(seed: Int, baseByPhase: FloatArray, sigma: Float): ShortArray {
        val rnd = Random(seed)
        val s = ShortArray(SW * SH)
        for (y in 0 until SH) for (x in 0 until SW) {
            val p = phase(x, y)
            val g = baseByPhase[p] + (rnd.nextFloat() - 0.5f) * 2f * sigma
            s[y * SW + x] = (SENSOR_BLACK + g).roundToInt().coerceIn(0, SENSOR_CLIP).toShort()
        }
        return s
    }

    /** Left half = red-heavy raw, right half = blue-heavy raw (no per-cell noise). */
    private fun buildRedBlue(): ShortArray {
        val s = ShortArray(SW * SH)
        for (y in 0 until SH) for (x in 0 until SW) {
            val p = phase(x, y)
            val v = if (x < SW / 2) {
                if (BA_COLOR_MAP[p] == 0) 850f else 240f   // red half; G/B low
            } else {
                if (BA_COLOR_MAP[p] == 2) 850f else 240f   // blue half; R/G low
            }
            s[y * SW + x] = (SENSOR_BLACK + v).roundToInt().coerceIn(0, SENSOR_CLIP).toShort()
        }
        return s
    }

    // ------------------------------------------------------------------
    // The deployed GL pipeline, transcribed 1:1.
    // ------------------------------------------------------------------

    private class View(
        val gridW: Int,
        val gridH: Int,
        val flipX: Boolean,
        val flipY: Boolean,                 // y-flip is always on in preview
        val crop: FloatArray = floatArrayOf(0f, 0f, SENSOR_W2.toFloat(), SENSOR_H2.toFloat())
    ) {
        val k: Float get() = max(crop[2] / gridW, crop[3] / gridH)
        fun T(u: Float, axis: Int): Float {
            val flipped = if (axis == 0) flipX else flipY
            return if (flipped) 1f - u else u
        }
        fun TInv(u: Float, axis: Int): Float = T(u, axis)   // mirror is self-inverse
    }

    // Host gate (live PreviewRenderer): the sparse mosaic feeds the demosaic
    // only when previewZoomK <= 2.0.  The DPC+GF chain (needSparseGrid) takes
    // priority inside that range; the S3_PACK fallback this test mirrors runs
    // only when the DPC/rawDenoise shaders are not ready.  Beyond k=2 the
    // device never uses the mosaic — the demosaic's inline per-sample
    // sampleSameColorNR carries S1/S3 instead.
    private fun packActive(v: View): Boolean = v.k <= 2f

    // Host-side boxAA gate (live PreviewRenderer::renderBayerFrame): every
    // slider increase must keep every band's sigma non-increasing (monotone
    // walk probed across k=1.5..8) AND at-or-below the zero-slider baseline.
    // box3 is strictly dominated (4x4+4-tap is cheaper and smoother than
    // 3x3+12-tap), so the mapping emits only 4 (weak/mid s3, 4-tap ring) and
    // 2 (strong s3 >= 0.7, full 12-tap ring).  The pack regime (k<=2) uses its
    // own shader, boxAA=4 throughout.  Keep in sync with the host.
    private fun previewBoxAA(v: View, s1: Float, s3: Float): Int {
        if (v.k <= 2f) return 4
        return if (s3 >= 0.7f) 2 else 4
    }

    /** Box-AA base offset: even boxes (2) start at the pixel, odd/large (3,4)
     *  start at the left/top neighbour — mirrors demosaicBilinear. */
    private fun boxBase(sv: Float, box: Int): Int {
        val baseOff = if (box >= 3) -1 else 0
        return kotlin.math.floor(sv).toInt() + baseOff
    }

    /** Centered crop sub-window (zoom-in): cropW/H sensor cells on a fixed grid. */
    private fun zoomInCrop(cropW: Int, cropH: Int): FloatArray {
        val left = (SW - cropW) / 2
        val top = (SH - cropH) / 2
        return floatArrayOf(left.toFloat(), top.toFloat(), cropW.toFloat(), cropH.toFloat())
    }

    // --- S3_PACK pass -------------------------------------------------

    private fun buildPackGL(scene: ShortArray, v: View, s1: Float, s3: Float, gains: FloatArray? = null, mode: Int = 0): FloatArray {
        val GX = v.gridW
        val GY = v.gridH
        val out = FloatArray(GX * GY * 4)
        for (gy in 0 until GY) for (gx in 0 until GX) {
            val idx = floatArrayOf(gx.toFloat(), gy.toFloat())
            val n0 = floatArrayOf(v.T(idx[0] / GX, 0), v.T(idx[1] / GY, 1))
            val n1 = floatArrayOf(v.T((idx[0] + 1f) / GX, 0), v.T((idx[1] + 1f) / GY, 1))
            val loRaw = floatArrayOf(v.crop[0] + n0[0] * v.crop[2], v.crop[1] + n0[1] * v.crop[3])
            val hiRaw = floatArrayOf(v.crop[0] + n1[0] * v.crop[2], v.crop[1] + n1[1] * v.crop[3])
            val lo = floatArrayOf(min(loRaw[0], hiRaw[0]), min(loRaw[1], hiRaw[1]))
            val hi = floatArrayOf(max(loRaw[0], hiRaw[0]), max(loRaw[1], hiRaw[1]))
            val b0x = kotlin.math.floor(lo[0]).toInt()
            val b0y = kotlin.math.floor(lo[1]).toInt()
            val b1x = max(kotlin.math.floor(hi[0]).toInt() - 1, b0x)
            val b1y = max(kotlin.math.floor(hi[1]).toInt() - 1, b0y)
            val boxW = max(b1x - b0x + 1, 1)
            val boxH = max(b1y - b0y + 1, 1)
            val nCells = boxW * boxH
            val base = (gy * GX + gx) * 4
            val n0cx = v.T((idx[0] + 0.5f) / GX, 0)
            val n0cy = v.T((idx[1] + 0.5f) / GY, 1)
            val scx = kotlin.math.floor(v.crop[0] + n0cx * v.crop[2]).toInt()
            val scy = kotlin.math.floor(v.crop[1] + n0cy * v.crop[3]).toInt()

            if (nCells <= 1) {
                val px = abs(scx % 2)
                val py = abs(scy % 2)
                val mc = coarseDevAt(scene, scx.coerceIn(0, SW - 1), scy.coerceIn(0, SH - 1))
                for (p in 0..3) {
                    val phaseX = p and 1
                    val phaseY = p shr 1
                    out[base + p] = sameColorNR(
                        scene,
                        (scx + (px xor phaseX)).coerceIn(0, SW - 1),
                        (scy + (py xor phaseY)).coerceIn(0, SH - 1),
                        s1, s3,
                        mode = mode, mCoarse = mc
                    )
                }
                continue
            }

            val pSum = FloatArray(4)
            val pN = IntArray(4)
            for (yy in b0y..b1y) {
                for (xx in b0x..b1x) {
                    val clx = xx.coerceIn(0, SW - 1)
                    val cly = yy.coerceIn(0, SH - 1)
                    val p = phase(clx, cly)
                    pSum[p] += sensorVal(scene, clx, cly)
                    pN[p]++
                }
            }
            // A small footprint whose every present phase box-mean sits at clip
            // is an all-hot cluster (or a sub-quad highlight): the mn==mx
            // degenerate makes boxedBlend a no-op, so DPC each phase cell
            // individually through the anchored per-phase filter (same as the
            // 1-cell path) instead — removes the cluster exactly as the inline
            // path would (mirrors the shader).
            var allClip = true
            for (p in 0..3) if (pN[p] > 0 && pSum[p] / pN[p] < CLIP_SCALAR) allClip = false
            if (nCells <= 4 && allClip) {
                val px = abs(b0x % 2)
                val py = abs(b0y % 2)
                val mc = coarseDevAt(scene, b0x.coerceIn(0, SW - 1), b0y.coerceIn(0, SH - 1))
                for (p in 0..3) {
                    val phaseX = p and 1
                    val phaseY = p shr 1
                    out[base + p] = sameColorNR(
                        scene,
                        (b0x + (px xor phaseX)).coerceIn(0, SW - 1),
                        (b0y + (py xor phaseY)).coerceIn(0, SH - 1),
                        s1, s3,
                        mode = mode, mCoarse = mc
                    )
                }
                continue
            }
            var tot = 0f
            var tn = 0
            for (p in 0..3) if (pN[p] > 0) {
                val pm = pSum[p] / pN[p]
                // A phase whose box-mean sits at clip is a single hot cell (or a
                // small-site highlight): pulling it down 0.98x toward a poisoned
                // fillback would still leave a visible speck, so exclude it from
                // the fillback used for the absent phases (mirrors the shader).
                if (pm < CLIP_SCALAR) { tot += pm; tn++ }
            }
            if (tn == 0) {
                // Every present phase is at clip (genuine highlight): fall back
                // to the plain mean so the fillback stays populated.
                for (p in 0..3) if (pN[p] > 0) { tot += pSum[p] / pN[p]; tn++ }
                if (tn == 0) { tot = 0f; tn = 1 }
            }
            val fbk = tot / tn
            for (p in 0..3) if (pN[p] == 0) { pN[p] = 1; pSum[p] = fbk }

            if (nCells >= 16) {
                out[base + 0] = pSum[0] / pN[0]
                out[base + 1] = pSum[1] / pN[1]
                out[base + 2] = pSum[2] / pN[2]
                out[base + 3] = pSum[3] / pN[3]
                continue
            }

            // boxedBlend
            val b = floatArrayOf(pSum[0] / pN[0], pSum[1] / pN[1], pSum[2] / pN[2], pSum[3] / pN[3])
            val bl = boxedBlend(b, s1, s3, gains)
            out[base + 0] = bl[0]; out[base + 1] = bl[1]; out[base + 2] = bl[2]; out[base + 3] = bl[3]
        }
        return out
    }

    private fun boxedBlend(b: FloatArray, s1: Float, s3: Float, gains: FloatArray? = null): FloatArray {
        val gs = gains ?: floatArrayOf(1f, 1f, 1f, 1f)
        val bn = floatArrayOf(b[0] * gs[0], b[1] * gs[1], b[2] * gs[2], b[3] * gs[3])
        val mn = min(min(min(bn[0], bn[1]), bn[2]), bn[3])
        val mx = max(max(max(bn[0], bn[1]), bn[2]), bn[3])
        val iavg = (bn[0] + bn[1] + bn[2] + bn[3] - mn - mx) * 0.5f
        val sigma = sqrt(max(isoA * max(iavg, 0f) + isoB, 1f))
        val band = max((0.1f + 0.3f * s1) * max(iavg, 0f), (2f + 2f * s1) * sigma)
        val corr = max(s1, 0.85f * s3)
        val o = FloatArray(4)
        fun omxx(p: Int): Float {
            var m = -1f
            for (q in 0..3) if (q != p) m = max(m, bn[q])
            return m
        }
        fun omn(p: Int): Float {
            var m = Float.MAX_VALUE
            for (q in 0..3) if (q != p) m = min(m, bn[q])
            return m
        }
        for (p in 0..3) {
            val c = bn[p]
            var center = c
            if (corr > 0f) {
                val hot = (c > omxx(p)) && (c - iavg) > band
                val cold = (c < omn(p)) && (iavg - c) > band
                if (hot || cold) center = c + (iavg - c) * max(corr, 0.98f)
            }
            val outN = if (b[p] < CLIP_SCALAR) center + (iavg - center) * (0.98f * s3) else center
            o[p] = outN / gs[p]
        }
        return o
    }

    // --- DEMOSAIC pass -------------------------------------------------

    /** denoisedSampleRaw mirror: values channel-select by safePhase(cell). */
    private fun mosaicRead(
        pack: FloatArray, v: View, cx0: Int, cy0: Int,
        shiftX: Int = 0, shiftY: Int = 0
    ): Float {
        val GX = v.gridW
        val GY = v.gridH
        val invX = (cx0 - v.crop[0]) / v.crop[2]
        val invY = (cy0 - v.crop[1]) / v.crop[3]
        val aX = v.TInv(invX, 0)
        val aY = v.TInv(invY, 1)
        val tcX = ((aX * GX + 0.5f).coerceIn(0.5f, GX - 0.5f) + shiftX).coerceIn(0.5f, GX - 0.5f)
        val tcY = ((aY * GY + 0.5f).coerceIn(0.5f, GY - 0.5f) + shiftY).coerceIn(0.5f, GY - 0.5f)
        val want = phase(cx0, cy0)
        // GL_LINEAR + CLAMP_TO_EDGE weights for position tc (texel centre i+0.5).
        fun lerp1(tc: Float, G: Int): FloatArray {
            val t0f = kotlin.math.floor(tc - 0.5f).toInt()
            val t0 = t0f.coerceIn(0, G - 1)
            val t1 = (t0f + 1).coerceIn(0, G - 1)
            val w0 = 1f - min(max(tc - (t0 + 0.5f), 0f), 1f)
            return floatArrayOf(t0.toFloat(), t1.toFloat(), w0)
        }
        val wx = lerp1(tcX, GX)
        val wy = lerp1(tcY, GY)
        var acc = 0f
        val i00 = (wy[0].toInt() * GX + wx[0].toInt()) * 4 + want
        val i10 = (wy[0].toInt() * GX + wx[1].toInt()) * 4 + want
        val i01 = (wy[1].toInt() * GX + wx[0].toInt()) * 4 + want
        val i11 = (wy[1].toInt() * GX + wx[1].toInt()) * 4 + want
        acc += pack[i00] * (wx[2]) * (wy[2])
        acc += pack[i10] * (1f - wx[2]) * (wy[2])
        acc += pack[i01] * (wx[2]) * (1f - wy[2])
        acc += pack[i11] * (1f - wx[2]) * (1f - wy[2])
        return acc
    }

    /** Host-side σ-gate mirror of previewNrRadius: the inline neighbourhood
     *  filter drops to the 4-tap ring only where the demosaic box stays 4x4
     *  (the box supplies the steady averaging there); the 2x2 box keeps the
     *  full 12-tap.  Keep in sync with the host. */
    private fun previewNrRadius(boxAA: Int): Int = if (boxAA == 4) 2 else 4

    /** One denoisedSampleRaw tap: mosaic (pack active, k<16) or inline (stale pack). */
    private fun demosaicSample(
        scene: ShortArray, pack: FloatArray, v: View,
        s1: Float, s3: Float, cx0: Int, cy0: Int,
        shiftX: Int = 0, shiftY: Int = 0,
        nrRadius: Int = 4,
        mode: Int = 0,
        mCoarse: Float? = null
    ): Float {
        val cx = cx0.coerceIn(0, SW - 1)
        val cy = cy0.coerceIn(0, SH - 1)
        if (packActive(v)) {
            return mosaicRead(pack, v, cx, cy, shiftX, shiftY)
        }
        return sameColorNR(scene, cx, cy, s1, s3, nrRadius, mode, mCoarse)
    }

    /** demosaicAt: the standard 9-tap reconstruction (box-AA 4x4 host path). */
    private fun demosaicAtRGB(
        scene: ShortArray, pack: FloatArray, v: View,
        s1: Float, s3: Float,         cellX: Int, cellY: Int,
        shiftX: Int = 0, shiftY: Int = 0,
        nrRadius: Int = 4,
        mode: Int = 0,
        mCoarse: Float? = null
    ): FloatArray {
        val center = demosaicSample(scene, pack, v, s1, s3, cellX, cellY, shiftX, shiftY, nrRadius, mode, mCoarse)
        val nN = demosaicSample(scene, pack, v, s1, s3, cellX, cellY - 1, shiftX, shiftY, nrRadius, mode, mCoarse)
        val nS = demosaicSample(scene, pack, v, s1, s3, cellX, cellY + 1, shiftX, shiftY, nrRadius, mode, mCoarse)
        val nW = demosaicSample(scene, pack, v, s1, s3, cellX - 1, cellY, shiftX, shiftY, nrRadius, mode, mCoarse)
        val nE = demosaicSample(scene, pack, v, s1, s3, cellX + 1, cellY, shiftX, shiftY, nrRadius, mode, mCoarse)
        val nNW = demosaicSample(scene, pack, v, s1, s3, cellX - 1, cellY - 1, shiftX, shiftY, nrRadius, mode, mCoarse)
        val nNE = demosaicSample(scene, pack, v, s1, s3, cellX + 1, cellY - 1, shiftX, shiftY, nrRadius, mode, mCoarse)
        val nSW = demosaicSample(scene, pack, v, s1, s3, cellX - 1, cellY + 1, shiftX, shiftY, nrRadius, mode, mCoarse)
        val nSE = demosaicSample(scene, pack, v, s1, s3, cellX + 1, cellY + 1, shiftX, shiftY, nrRadius, mode, mCoarse)
        val color = BA_COLOR_MAP[phase(cellX, cellY)]
        return if (color == 0) {
            floatArrayOf(center, (nW + nE + nN + nS) * 0.25f, (nNW + nNE + nSW + nSE) * 0.25f)
        } else if (color == 2) {
            floatArrayOf((nNW + nNE + nSW + nSE) * 0.25f, (nW + nE + nN + nS) * 0.25f, center)
        } else {
            val colorNS = BA_COLOR_MAP[phase(cellX, cellY - 1)]
            if (colorNS == 0) {
                floatArrayOf((nN + nS) * 0.5f, center, (nW + nE) * 0.5f)
            } else {
                floatArrayOf((nW + nE) * 0.5f, center, (nN + nS) * 0.5f)
            }
        }
    }

    /** Dual-ring same-colour sample (mirror of the GLSL denoisedDualRing used by
     *  the fused smooth-box path): computes BOTH the ring-2 sample (lo, the 4x4
     *  box side) and the ring-4 sample (hi, the 2x2 box side) from one tap set.
     *  Each result must be bit-identical to a standalone sameColorNR call at its
     *  ring — the packed 4x4 ring-2 box's middle four cells are exactly the 2x2
     *  ring-4 box, so the fusion removes those four redundant re-evaluations per
     *  fragment without touching the output.  Arithmetic is copied verbatim from
     *  sameColorNR's two ring branches (hi with mode 3 = the shipped regional
     *  pull). */
    private fun denoisedDualRing(
        v: ShortArray, sx: Int, sy: Int,
        s1: Float, s3: Float, mCoarse: Float?
    ): Pair<Float, Float> {
        val c = sensorVal(v, sx, sy)
        if (s1 <= 0f && s3 <= 0f) return c to c
        val nE = sensorVal(v, sx + 2, sy)
        val nW = sensorVal(v, sx - 2, sy)
        val nN = sensorVal(v, sx, sy - 2)
        val nS = sensorVal(v, sx, sy + 2)

        // ---- lo: ring-2 path (verbatim copy of sameColorNR's nrRadius<4)
        val smn2 = min(min(nE, nW), min(nN, nS))
        val smx2 = max(max(nE, nW), max(nN, nS))
        val iavg2 = (nE + nW + nN + nS - smn2 - smx2) / 2f
        val sigma2 = sqrt(max(isoA * max(iavg2, 0f) + isoB, 1f))
        val band2 = max((0.1f + 0.3f * s1) * max(iavg2, 0f), (2f + 2f * s1) * sigma2)
        var center2 = c
        if (max(s1, 0.85f * s3) > 0f) {
            val hot2 = (c > smx2) && (c - iavg2) > band2
            val cold2 = (c < smn2) && (iavg2 - c) > band2
            if (hot2 || cold2) {
                center2 = c + (iavg2 - c) * max(max(s1, 0.85f * s3), 0.98f)
            }
        }
        val lo = if (c < CLIP_SCALAR) center2 + (iavg2 - center2) * (0.98f * s3) else center2

        // ---- hi: ring-4 path (verbatim copy of sameColorNR's nrRadius>=4, mode 3)
        val nNE = sensorVal(v, sx + 2, sy - 2)
        val nNW = sensorVal(v, sx - 2, sy - 2)
        val nSE = sensorVal(v, sx + 2, sy + 2)
        val nSW = sensorVal(v, sx - 2, sy + 2)
        val nEE = sensorVal(v, sx + 4, sy)
        val nWW = sensorVal(v, sx - 4, sy)
        val nNN = sensorVal(v, sx, sy - 4)
        val nSS = sensorVal(v, sx, sy + 4)
        val smn = min(min(min(min(min(nE, nW), min(nN, nS)), min(nNE, nNW)), min(nSE, nSW)),
            min(min(nEE, nWW), min(nNN, nSS)))
        val smx = max(max(max(max(max(nE, nW), max(nN, nS)), max(nNE, nNW)), max(nSE, nSW)),
            max(max(nEE, nWW), max(nNN, nSS)))
        val iavg = (nE + nW + nN + nS + nNE + nNW + nSE + nSW + nEE + nWW + nNN + nSS - smn - smx) / 10f
        val sigma = sqrt(max(isoA * max(iavg, 0f) + isoB, 1f))
        val band = max((0.1f + 0.3f * s1) * max(iavg, 0f), (2f + 2f * s1) * sigma)

        val aH = (nE + nW) * 0.5f
        val aV = (nN + nS) * 0.5f
        val a45 = (nNE + nSW) * 0.5f
        val a135 = (nSE + nNW) * 0.5f
        val dH = abs(nE - nW)
        val dV = abs(nN - nS)
        val d45 = abs(nNE - nSW)
        val d135 = abs(nSE - nNW)
        var iDir = aH
        if (dV < dH && dV <= d45 && dV <= d135) iDir = aV
        else if (d45 < dH && d45 <= dV && d45 <= d135) iDir = a45
        else if (d135 < dH && d135 <= dV && d135 <= d45) iDir = a135
        val tN = nNN + nNE + nNW + c - minOf(minOf(nNN, nNE), minOf(nNW, c)) - maxOf(maxOf(nNN, nNE), maxOf(nNW, c))
        val devN = abs(nN - tN * 0.5f)
        val tS = nSS + nSE + nSW + c - minOf(minOf(nSS, nSE), minOf(nSW, c)) - maxOf(maxOf(nSS, nSE), maxOf(nSW, c))
        val devS = abs(nS - tS * 0.5f)
        val tE = nEE + nNE + nSE + c - minOf(minOf(nEE, nNE), minOf(nSE, c)) - maxOf(maxOf(nEE, nNE), maxOf(nSE, c))
        val devE = abs(nE - tE * 0.5f)
        val tW = nWW + nNW + nSW + c - minOf(minOf(nWW, nNW), minOf(nSW, c)) - maxOf(maxOf(nWW, nNW), maxOf(nSW, c))
        val devW = abs(nW - tW * 0.5f)
        val maxNb = maxOf(maxOf(devN, devS), maxOf(devE, devW))
        val minDev = minOf(devN, devS, devE, devW)

        var center = c
        if (max(s1, 0.85f * s3) > 0f) {
            val hot = (c > smx) && (c - iavg) > band
            val cold = (c < smn) && (iavg - c) > band
            if (hot || cold) {
                if (abs(c - iavg) > 6.0f * maxNb) {
                    center = c + (iDir - c) * max(max(s1, 0.85f * s3), 0.98f)
                }
            }
        }

        val sg = max(sigma, 6.0f * minDev)
        var bavg: Float
        if (maxNb <= 6.0f * sigma) {
            val ratio = if (maxNb > 1e-6f) (mCoarse ?: coarseDevAt(v, sx, sy)) / maxNb else 2f
            val omega = ((ratio - omegaLo) / (omegaHi - omegaLo)).coerceIn(0f, 1f)
            bavg = iavg + (iDir - iavg) * omega
        } else {
            val tau = 2.5f * sg
            val invTau = 1f / tau
            var wsum = 0f
            bavg = 0f
            fun tapW(tx: Int, ty: Int) {
                val t = sensorVal(v, sx + tx, sy + ty)
                val w = (1f - max(abs(t - center) - tau, 0f) * invTau).coerceIn(0f, 1f)
                bavg += w * t
                wsum += w
            }
            tapW(2, 0); tapW(-2, 0); tapW(0, -2); tapW(0, 2)
            tapW(2, -2); tapW(-2, -2); tapW(2, 2); tapW(-2, 2)
            tapW(4, 0); tapW(-4, 0); tapW(0, -4); tapW(0, 4)
            if (wsum > 0f) bavg /= wsum else bavg = center
        }
        val hi = if (c < CLIP_SCALAR) center + (bavg - center) * (0.98f * s3) else center
        return lo to hi
    }

    /** One dual sample (mosaic read or inline), like demosaicSample. */
    private fun dualSample(
        scene: ShortArray, pack: FloatArray, v: View,
        s1: Float, s3: Float, cx0: Int, cy0: Int,
        shiftX: Int = 0, shiftY: Int = 0,
        mode: Int = 3,
        mCoarse: Float? = null
    ): Pair<Float, Float> {
        val cx = cx0.coerceIn(0, SW - 1)
        val cy = cy0.coerceIn(0, SH - 1)
        if (packActive(v)) {
            val m = mosaicRead(pack, v, cx, cy, shiftX, shiftY)
            return m to m
        }
        return denoisedDualRing(scene, cx, cy, s1, s3, mCoarse)
    }

    /** Dual demosaic (mirror of the GLSL demosaicAtDual): the fused box's
     *  mid-cell reconstruction, returning both the 4x4-side (ring-2) and the
     *  2x2-side (ring-4) RGB.  Bit-identical to demosaicAtRGB at each ring. */
    private fun demosaicAtDualRGB(
        scene: ShortArray, pack: FloatArray, v: View,
        s1: Float, s3: Float, cellX: Int, cellY: Int,
        shiftX: Int = 0, shiftY: Int = 0,
        mode: Int = 3,
        mCoarse: Float? = null
    ): Pair<FloatArray, FloatArray> {
        fun tap(ox: Int, oy: Int): Pair<Float, Float> =
            dualSample(scene, pack, v, s1, s3, cellX + ox, cellY + oy, shiftX, shiftY, mode, mCoarse)
        val tapC = tap(0, 0)
        val tapN = tap(0, -1)
        val tapS = tap(0, 1)
        val tapW = tap(-1, 0)
        val tapE = tap(1, 0)
        val tapNW = tap(-1, -1)
        val tapNE = tap(1, -1)
        val tapSW = tap(-1, 1)
        val tapSE = tap(1, 1)
        val color = BA_COLOR_MAP[phase(cellX, cellY)]
        fun lo(): Float = tapC.first
        fun hi(): Float = tapC.second
        fun loOf(p: Pair<Float, Float>): Float = p.first
        fun hiOf(p: Pair<Float, Float>): Float = p.second
        val loR: Float
        val loG: Float
        val loB: Float
        val hiR: Float
        val hiG: Float
        val hiB: Float
        if (color == 0) {
            loR = lo(); hiR = hi()
            loG = (loOf(tapW) + loOf(tapE) + loOf(tapN) + loOf(tapS)) * 0.25f
            hiG = (hiOf(tapW) + hiOf(tapE) + hiOf(tapN) + hiOf(tapS)) * 0.25f
            loB = (loOf(tapNW) + loOf(tapNE) + loOf(tapSW) + loOf(tapSE)) * 0.25f
            hiB = (hiOf(tapNW) + hiOf(tapNE) + hiOf(tapSW) + hiOf(tapSE)) * 0.25f
        } else if (color == 2) {
            loB = lo(); hiB = hi()
            loG = (loOf(tapW) + loOf(tapE) + loOf(tapN) + loOf(tapS)) * 0.25f
            hiG = (hiOf(tapW) + hiOf(tapE) + hiOf(tapN) + hiOf(tapS)) * 0.25f
            loR = (loOf(tapNW) + loOf(tapNE) + loOf(tapSW) + loOf(tapSE)) * 0.25f
            hiR = (hiOf(tapNW) + hiOf(tapNE) + hiOf(tapSW) + hiOf(tapSE)) * 0.25f
        } else {
            loG = lo(); hiG = hi()
            val colorNS = BA_COLOR_MAP[phase(cellX, cellY - 1)]
            if (colorNS == 0) {
                loR = (loOf(tapN) + loOf(tapS)) * 0.5f; loB = (loOf(tapW) + loOf(tapE)) * 0.5f
                hiR = (hiOf(tapN) + hiOf(tapS)) * 0.5f; hiB = (hiOf(tapW) + hiOf(tapE)) * 0.5f
            } else {
                loR = (loOf(tapW) + loOf(tapE)) * 0.5f; loB = (loOf(tapN) + loOf(tapS)) * 0.5f
                hiR = (hiOf(tapW) + hiOf(tapE)) * 0.5f; hiB = (hiOf(tapN) + hiOf(tapS)) * 0.5f
            }
        }
        return floatArrayOf(loR, loG, loB) to floatArrayOf(hiR, hiG, hiB)
    }

    /** demosaicBilinear + main(): render the full output image (all paths). */
    private fun renderPreview(
        scene: ShortArray, pack: FloatArray, v: View, s1: Float, s3: Float,
        shiftX: Int = 0, shiftY: Int = 0,
        boxAA: Int = previewBoxAA(v, s1, s3),
        nrRadius: Int = previewNrRadius(boxAA),
        mode: Int = 0,
        boxSmooth: Boolean = false
    ): Array<FloatArray> {
        val GX = v.gridW
        val GY = v.gridH
        val out = Array(GY) { FloatArray(GX * 3) }
        for (gy in 0 until GY) for (gx in 0 until GX) {
            val aX = (gx + 0.5f) / GX
            val aY = (gy + 0.5f) / GY
            val svX = v.crop[0] + v.T(aX, 0) * v.crop[2]
            val svY = v.crop[1] + v.T(aY, 1) * v.crop[3]
            var sum: FloatArray
            if (boxSmooth && v.k > 2f) {
                // Prototype of a monotone-by-construction box-AA: instead of
                // stepping 4->2 at s3>=0.7, blend the box-4 mean (16 samples,
                // 4-tap ring) and the box-2 mean (4 samples, 12-tap ring) by a
                // smoothstep on s3.  A continuous box removes the step the
                // directional flat pull pops.
                val w = boxMixW(s3)
                val base4 = boxBase(svX, 4)
                val baseY4 = boxBase(svY, 4)
                // One coarse dev per output pixel, read once at the box centre
                // (mirror of demosaicBilinear's per-fragment mCoarse).  Shared
                // by both smooth sub-boxes — this is the S3 perf fix.
                val mc = coarseDevAt(scene, (base4 + 2).coerceIn(0, SW - 1), (baseY4 + 2).coerceIn(0, SH - 1))
                var sum4 = floatArrayOf(0f, 0f, 0f)
                for (dy in 0 until 4) for (dx in 0 until 4) {
                    val c = demosaicAtRGB(
                        scene, pack, v, s1, s3,
                        (base4 + dx).coerceIn(0, SW - 1), (baseY4 + dy).coerceIn(0, SH - 1),
                        shiftX, shiftY, nrRadius = 2, mode = mode, mCoarse = mc
                    )
                    sum4[0] += c[0]; sum4[1] += c[1]; sum4[2] += c[2]
                }
                val base2 = boxBase(svX, 2)
                val baseY2 = boxBase(svY, 2)
                var sum2 = floatArrayOf(0f, 0f, 0f)
                for (dy in 0 until 2) for (dx in 0 until 2) {
                    val c = demosaicAtRGB(
                        scene, pack, v, s1, s3,
                        (base2 + dx).coerceIn(0, SW - 1), (baseY2 + dy).coerceIn(0, SH - 1),
                        shiftX, shiftY, nrRadius = 4, mode = mode, mCoarse = mc
                    )
                    sum2[0] += c[0]; sum2[1] += c[1]; sum2[2] += c[2]
                }
                val div4 = 1f / 16f
                val div2 = 1f / 4f
                sum = floatArrayOf(
                    (1f - w) * (sum4[0] * div4) + w * (sum2[0] * div2),
                    (1f - w) * (sum4[1] * div4) + w * (sum2[1] * div2),
                    (1f - w) * (sum4[2] * div4) + w * (sum2[2] * div2)
                )
            } else {
                val baseX = boxBase(svX, boxAA)
                val baseY = boxBase(svY, boxAA)
                // One coarse dev per output pixel at the box centre (box<=1:
                // base + 0 == the clamped sample, matching the GLSL 1:1 path).
                val mc = coarseDevAt(scene, (baseX + boxAA / 2).coerceIn(0, SW - 1), (baseY + boxAA / 2).coerceIn(0, SH - 1))
                sum = floatArrayOf(0f, 0f, 0f)
                for (dy in 0 until boxAA) for (dx in 0 until boxAA) {
                    val c = demosaicAtRGB(
                        scene, pack, v, s1, s3,
                        (baseX + dx).coerceIn(0, SW - 1),
                        (baseY + dy).coerceIn(0, SH - 1),
                        shiftX, shiftY,
                        nrRadius = nrRadius,
                        mode = mode, mCoarse = mc
                    )
                    sum[0] += c[0]; sum[1] += c[1]; sum[2] += c[2]
                }
                val div = 1f / (boxAA * boxAA)
                sum[0] *= div; sum[1] *= div; sum[2] *= div
            }
            val fin = composeMain(sum)
            out[gy][gx * 3] = fin[0]; out[gy][gx * 3 + 1] = fin[1]; out[gy][gx * 3 + 2] = fin[2]
        }
        return out
    }

    /** Smoothstep 0..1 across s3 in [0.65, 0.75] — the box-2 blend weight. */
    private fun boxMixW(s3: Float): Float {
        val t = ((s3 - 0.65f) / 0.10f).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    /** compensateNegatives + luma-attenuation from the shader main(). */
    private fun composeMain(rgb: FloatArray): FloatArray {
        var r = rgb[0] / CLIP_SCALAR
        var g = rgb[1] / CLIP_SCALAR
        var b = rgb[2] / CLIP_SCALAR
        val minimum = min(min(r, g), b)
        if (minimum < 0f) {
            val peak = max(max(r, g), b)
            val lifted = peak - minimum
            val ratio = if (lifted > 0f) peak / lifted else 0f
            r = max((r - minimum) * ratio, 0f)
            g = max((g - minimum) * ratio, 0f)
            b = max((b - minimum) * ratio, 0f)
        }
        val luma = 0.2126f * r + 0.7152f * g + 0.0722f * b
        val atten = sqrt(max(1f - luma, 0f))
        val peak = max(max(r, g), b)
        return floatArrayOf((r - peak) * atten + peak, (g - peak) * atten + peak, (b - peak) * atten + peak)
    }

    /** FULL shader main() mirror: demosaic -> xWB -> clip-neutralization ->
     *  xcolorMat -> /CLIP_SCALAR (the deployed DEMOSAIC_FRAGMENT_SHADER).
     *  The neighbours here re-run the pack read exactly as demosaicAt does. */
    private fun renderPreviewFull(
        scene: ShortArray, pack: FloatArray, v: View, s1: Float, s3: Float,
        wbR: Float, wbB: Float,
        colorMat: FloatArray? = null,
        shiftX: Int = 0, shiftY: Int = 0,
        boxAA: Int = previewBoxAA(v, s1, s3),
        nrRadius: Int = previewNrRadius(boxAA),
        mode: Int = 0
    ): Array<FloatArray> {
        val GX = v.gridW
        val GY = v.gridH
        val out = Array(GY) { FloatArray(GX * 3) }
        for (gy in 0 until GY) for (gx in 0 until GX) {
            val aX = (gx + 0.5f) / GX
            val aY = (gy + 0.5f) / GY
            val svX = v.crop[0] + v.T(aX, 0) * v.crop[2]
            val svY = v.crop[1] + v.T(aY, 1) * v.crop[3]
            val baseX = boxBase(svX, boxAA)
            val baseY = boxBase(svY, boxAA)
            val mc = coarseDevAt(scene, (baseX + boxAA / 2).coerceIn(0, SW - 1), (baseY + boxAA / 2).coerceIn(0, SH - 1))
            var sum = floatArrayOf(0f, 0f, 0f)
            for (dy in 0 until boxAA) for (dx in 0 until boxAA) {
                val c = demosaicAtRGB(
                    scene, pack, v, s1, s3,
                    (baseX + dx).coerceIn(0, SW - 1),
                    (baseY + dy).coerceIn(0, SH - 1),
                    shiftX, shiftY,
                    nrRadius = nrRadius,
                    mode = mode, mCoarse = mc
                )
                sum[0] += c[0]; sum[1] += c[1]; sum[2] += c[2]
            }
            val div = 1f / (boxAA * boxAA)
            sum[0] *= div; sum[1] *= div; sum[2] *= div
            // linearRGB *= u_wb_gains
            var r = sum[0] * wbR
            var g = sum[1]
            var b = sum[2] * wbB
            // clip-neutralization (u_clip_atten_factor -> exp 0.5, 709 luma)
            var nr = r / CLIP_SCALAR
            var ng = g / CLIP_SCALAR
            var nb = b / CLIP_SCALAR
            val minimum = min(min(nr, ng), nb)
            if (minimum < 0f) {
                val peak = max(max(nr, ng), nb)
                val lifted = peak - minimum
                val ratio = if (lifted > 0f) peak / lifted else 0f
                nr = max((nr - minimum) * ratio, 0f)
                ng = max((ng - minimum) * ratio, 0f)
                nb = max((nb - minimum) * ratio, 0f)
            }
            val luma = 0.2126f * nr + 0.7152f * ng + 0.0722f * nb
            val atten = sqrt(max(1f - luma, 0f))
            val peak = max(max(nr, ng), nb)
            nr = (nr - peak) * atten + peak
            ng = (ng - peak) * atten + peak
            nb = (nb - peak) * atten + peak
            r = nr * CLIP_SCALAR; g = ng * CLIP_SCALAR; b = nb * CLIP_SCALAR
            // u_color_mat * (rgb / CLIP_SCALAR)
            if (colorMat != null) {
                var x = r / CLIP_SCALAR; var y = g / CLIP_SCALAR; var z = b / CLIP_SCALAR
                r = (colorMat[0] * x + colorMat[1] * y + colorMat[2] * z) * CLIP_SCALAR
                g = (colorMat[3] * x + colorMat[4] * y + colorMat[5] * z) * CLIP_SCALAR
                b = (colorMat[6] * x + colorMat[7] * y + colorMat[8] * z) * CLIP_SCALAR
            }
            out[gy][gx * 3] = r; out[gy][gx * 3 + 1] = g; out[gy][gx * 3 + 2] = b
        }
        return out
    }

    // ------------------------------------------------------------------
    // Metrics
    // ------------------------------------------------------------------

    /** Interior output texels (sensor centre > 8 cells from any edge). */
    private fun interiorMask(v: View): List<IntArray> {
        val list = mutableListOf<IntArray>()
        for (gy in 0 until v.gridH) for (gx in 0 until v.gridW) {
            val aX = (gx + 0.5f) / v.gridW
            val aY = (gy + 0.5f) / v.gridH
            val svX = v.crop[0] + v.T(aX, 0) * v.crop[2]
            val svY = v.crop[1] + v.T(aY, 1) * v.crop[3]
            if (svX > 10f && svX < SW - 10f && svY > 10f && svY < SH - 10f) list.add(intArrayOf(gx, gy))
        }
        return list
    }

    private fun metrics(img: Array<FloatArray>, v: View): FloatArray {
        val interior = interiorMask(v)
        var mr = 0f
        var mrCnt = 0
        var gSum = 0f
        var gSum2 = 0f
        var n = 0
        for ((gx, gy) in interior) {
            val r = img[gy][gx * 3]
            val g = img[gy][gx * 3 + 1]
            val b = img[gy][gx * 3 + 2]
            val den = max(max(r, g), b)
            mr += abs(r - b) / max(den, 1e-4f)
            mrCnt++
            gSum += g; gSum2 += g * g; n++
        }
        val mean = gSum / n
        val sd = sqrt(max(gSum2 / n - mean * mean, 0f))
        return floatArrayOf(mr / mrCnt, sd)
    }

    /** Standard deviation of one output band (R=0, G=1, B=2) over the interior. */
    private fun bandSD(img: Array<FloatArray>, v: View, channel: Int): Float {
        val interior = interiorMask(v)
        var s = 0f
        var s2 = 0f
        var n = 0
        for ((gx, gy) in interior) {
            val x = img[gy][gx * 3 + channel]
            s += x; s2 += x * x; n++
        }
        val mean = s / n
        return sqrt(max(s2 / n - mean * mean, 0f))
    }

    // ------------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------------

    @Test
    fun reproduceMagentaAndNoise() {
        val s1 = 0.6f
        val s3 = 0.6f
        // (grid, cropW, cropH, mirror) on the fixed 96x96 sensor FBO:
        //   k=1   crop=96  grid=96  (1:1 preview)
        //   k=0.5 crop=48  grid=96  (2x zoom-in: crop < grid)
        //   k=0.25 crop=24 grid=96  (4x zoom-in)
        //   k=2   crop=96  grid=48  (zoom-out)
        //   k=4   crop=96  grid=24  (zoom-out: boxed means, pack carries it)
        //   k=8   crop=96  grid=12  (deep zoom-out: 64-cell boxes per phase)
        val configs = listOf(
            Config(96, 96, 96, false),    // k=1  back
            Config(96, 96, 96, true),     // k=1  front (mirror x)
            Config(96, 48, 48, false),    // k=0.5 back  (zoom-in)
            Config(96, 48, 48, true),     // k=0.5 front (zoom-in + mirror)
            Config(96, 24, 24, false),    // k=0.25 back (strong zoom-in)
            Config(96, 24, 24, true),     // k=0.25 front (strong zoom-in + mirror)
            Config(48, 96, 96, false),    // k=2  back
            Config(48, 96, 96, true),     // k=2  front
            Config(24, 96, 96, false),    // k=4  back
            Config(24, 96, 96, true),     // k=4  front
            Config(12, 96, 96, false),    // k=8  back
            Config(12, 96, 96, true)      // k=8  front
        )
        for (scene in listOf("flatNoise", "redBlue")) {
            println("== scene=$scene s1=$s1 s3=$s3 (pack at k<=2, inline beyond) ==")
            println("grid  crop  k    mirror   magErr   gNoise   (magErr/gNoise at s1=0 s3=0)")
            for (cfg in configs) {
                val v = View(cfg.grid, cfg.grid, cfg.mirror, true,
                    if (cfg.cropW == 96) floatArrayOf(0f, 0f, 96f, 96f) else zoomInCrop(cfg.cropW, cfg.cropH))
                val sceneArr = if (scene == "flatNoise") buildFlatNoise(7, 460f, 28f) else buildRedBlue()
                val pack = buildPackGL(sceneArr, v, s1, s3)
                val img = renderPreview(sceneArr, pack, v, s1, s3)
                val m = metrics(img, v)
                val pack0 = buildPackGL(sceneArr, v, 0f, 0f)
                val img0 = renderPreview(sceneArr, pack0, v, 0f, 0f)
                val m0 = metrics(img0, v)
                println("${cfg.grid}  ${cfg.cropW}  %.2f  %s  %.4f  %.3f   (%.4f / %.3f)  [pack]".format(
                    v.k, if (cfg.mirror) "mirrorX" else "back  ", m[0], m[1], m0[0], m0[1]
                ))

                if (scene == "flatNoise") {
                    // Invariant (a): a non-zero slider never adds preview noise.
                    // Only valid on a noisy flat field — S1/S3 denoise, so any
                    // growth in the green SD is an added-noise regression.
                    assertTrue(
                        "gNoise grew with sliders at grid=${cfg.grid} crop=${cfg.cropW} mirror=${cfg.mirror} " +
                            "(${m[1]} > ${m0[1]} + 0.001)",
                        m[1] <= m0[1] + 0.001f
                    )
                    // Invariant (b): no magenta overlay anywhere (bound like baseline).
                    assertTrue(
                        "magErr grew with sliders at grid=${cfg.grid} crop=${cfg.cropW} mirror=${cfg.mirror} " +
                            "(${m[0]} > ${m0[0]} * 1.5 + 0.02)",
                        m[0] <= m0[0] * 1.5f + 0.02f
                    )
                }
                // redBlue stays a print-only diagnostic: its colour boundary
                // legitimately warps local chroma, so SD/magErr are not
                // interpretable as artifact metrics there.
            }
        }
    }

    private class Config(val grid: Int, val cropW: Int, val cropH: Int, val mirror: Boolean)

    /**
     * Box-AA reduction acceptance gate (the host's inline S1/S3 cost fix):
     * with the smooth box active (the k>2 inline regime blends the 4x4+nr2
     * mean toward the 2x2+nr4 mean by smoothstep(s3,0.65,0.75)), each output
     * band's sigma must stay AT OR BELOW the zero-slider baseline (boxAA=4).
     * Any single band above baseline rejects the deployed demosaic — the
     * effective averaging must never get weaker than baseline at any strength.
     */
    @Test
    fun inlineReducedBoxKeepsBandsBounded() {
        val scene = buildFlatNoise(11, 460f, 28f)
        // k>2 regimes where the inline path is live and the box blend acts.
        val configs = listOf(
            Config(24, 96, 96, false),    // k=4  back
            Config(24, 96, 96, true),     // k=4  front
            Config(12, 96, 96, false),    // k=8  back
            Config(12, 96, 96, true)      // k=8  front
        )
        // Slider grid covering the blend zone (s3 0.65..0.75) plus both tails.
        val s3s = listOf(0f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.65f, 0.7f, 0.75f, 0.8f)
        val s1s = listOf(0.0f, 0.3f, 0.6f)
        val strengths = mutableListOf<FloatArray>()
        for (s3 in s3s) for (s1 in s1s) strengths.add(floatArrayOf(s1, s3))
        val failures = StringBuilder()
        println("== inline smooth-box per-band sigma vs zero-slider baseline ==")
        for (cfg in configs) {
            val v = View(cfg.grid, cfg.grid, cfg.mirror, true, floatArrayOf(0f, 0f, 96f, 96f))
            val back = if (cfg.mirror) "mirrorX" else "back  "
            val pack0 = buildPackGL(scene, v, 0f, 0f)
            val img0 = renderPreview(scene, pack0, v, 0f, 0f)   // baseline: box 4
            val sd0 = FloatArray(3) { bandSD(img0, v, it) }
            println("k=%.2f %s  baseline sigma R=%.3f G=%.3f B=%.3f".format(v.k, back, sd0[0], sd0[1], sd0[2]))
            for (s in strengths) {
                val pack = buildPackGL(scene, v, s[0], s[1], mode = 3)
                val img = renderPreview(scene, pack, v, s[0], s[1], mode = 3, boxSmooth = true)
                val sd = FloatArray(3) { bandSD(img, v, it) }
                println(
                    "  s1=%.1f s3=%.1f blend=%.2f  R=%.3f G=%.3f B=%.3f".format(
                        s[0], s[1], boxMixW(s[1]), sd[0], sd[1], sd[2]
                    )
                )
                val name = charArrayOf('R', 'G', 'B')
                for (ch in 0..2) {
                    if (sd[ch] > sd0[ch] + 0.001f) {
                        val msg = "band %s sigma %.3f > baseline %.3f at grid=%d mirror=%s s1=%.1f s3=%.1f blend=%.2f\n"
                            .format(name[ch], sd[ch], sd0[ch], cfg.grid, cfg.mirror, s[0], s[1], boxMixW(s[1]))
                        failures.append(msg)
                        println("  !! " + msg.trim())
                    }
                }
            }
        }
        assertTrue(
            "smooth box broke the per-band sigma gate:\n" + failures,
            failures.isEmpty()
        )
    }

    /**
     * Acceptance gate for the fused smooth-box path (GLSL demosaicAtDual):
     * the dual sample must be bit-identical to the two standalone ring-2 /
     * ring-4 demosaicAt calls it replaces, at exactly the mid-box positions
     * the smooth blend visits.  This is what makes the fusion
     * output-invariant — the redundant 4-of-20 per-fragment re-evaluations of
     * the k>2 smooth path are removed, nothing else changes.
     */
    @Test
    fun dualFusionMatchesStandalone() {
        val scene = buildFlatNoise(103, 460f, 28f)
        val configs = listOf(
            Config(24, 96, 96, false),    // k=4 back
            Config(24, 96, 96, true),     // k=4 mirror
            Config(12, 96, 96, false),    // k=8 back
            Config(12, 96, 96, true)      // k=8 mirror
        )
        // Slider grid covering the blend zone, both tails, and the off-case.
        val strengths = listOf(
            floatArrayOf(0.6f, 0.65f),
            floatArrayOf(0.3f, 0.7f),
            floatArrayOf(0f, 0.75f),
            floatArrayOf(0.6f, 0.8f),
            floatArrayOf(0f, 0f)
        )
        val mcFixed = coarseDevAt(scene, 48, 48)
        val mcVals = listOf<Float?>(null, mcFixed)
        var maxDev = 0f
        var cellCount = 0
        val failures = StringBuilder()
        for (cfg in configs) {
            val v = View(cfg.grid, cfg.grid, cfg.mirror, true, floatArrayOf(0f, 0f, 96f, 96f))
            for (s in strengths) {
                val pack = buildPackGL(scene, v, s[0], s[1], mode = 3)
                for (mc in mcVals) {
                    for (gy in 0 until v.gridH) for (gx in 0 until v.gridW) {
                        val aX = (gx + 0.5f) / v.gridW
                        val aY = (gy + 0.5f) / v.gridH
                        val svX = v.crop[0] + v.T(aX, 0) * v.crop[2]
                        val svY = v.crop[1] + v.T(aY, 1) * v.crop[3]
                        // Shared coarse dev (GLSL mCoarse = one per output pixel
                        // at the box centre, threaded to every ring sample).
                        val cmc = mc ?: coarseDevAt(
                            scene,
                            (boxBase(svX, 2) + 1).coerceIn(0, SW - 1),
                            (boxBase(svY, 2) + 1).coerceIn(0, SH - 1)
                        )
                        for (dy in 0 until 2) for (dx in 0 until 2) {
                            val cx = (boxBase(svX, 2) + dx).coerceIn(0, SW - 1)
                            val cy = (boxBase(svY, 2) + dy).coerceIn(0, SH - 1)
                            val loExp = demosaicAtRGB(scene, pack, v, s[0], s[1], cx, cy, nrRadius = 2, mode = 3, mCoarse = cmc)
                            val hiExp = demosaicAtRGB(scene, pack, v, s[0], s[1], cx, cy, nrRadius = 4, mode = 3, mCoarse = cmc)
                            val (loAct, hiAct) = demosaicAtDualRGB(scene, pack, v, s[0], s[1], cx, cy, mode = 3, mCoarse = cmc)
                            cellCount++
                            for (ch in 0..2) {
                                val dLo = abs(loExp[ch] - loAct[ch])
                                val dHi = abs(hiExp[ch] - hiAct[ch])
                                maxDev = max(maxDev, max(dLo, dHi))
                                if (loExp[ch] != loAct[ch] || hiExp[ch] != hiAct[ch]) {
                                    failures.append(
                                        "k=%.2f %s s1=%.1f s3=%.1f cell=(%d,%d) ch=%d loExp=%.9f loAct=%.9f hiExp=%.9f hiAct=%.9f\n"
                                            .format(v.k, if (cfg.mirror) "mirror" else "back", s[0], s[1], cx, cy, ch, loExp[ch], loAct[ch], hiExp[ch], hiAct[ch])
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        println("dualFusionMatchesStandalone: %d mid-box cells compared, maxDev=%.3e".format(cellCount, maxDev))
        assertTrue(
            "dual fusion drifted from standalone ring-2/ring-4 (must be bit-identical):\n" + failures,
            failures.isEmpty()
        )
    }

    /**
     * Diagnostic: which box size is safely reachable at every (s1, s3)?
     * Prints, for each combo, the smallest box whose per-band sigma stays at
     * or below the zero-slider baseline (the strict gate).  This is the table
     * that the host previewBoxAA mapping must be derived from — print-only,
     * no assertion.
     */
    @Test
    fun probeBoxAAWindow() {
        val scene = buildFlatNoise(17, 460f, 28f)
        for (grid in listOf(24, 12)) {   // k=4 and k=8
            val v = View(grid, grid, false, true, floatArrayOf(0f, 0f, 96f, 96f))
            val pack0 = buildPackGL(scene, v, 0f, 0f)
            val img0 = renderPreview(scene, pack0, v, 0f, 0f)
            val sd0 = FloatArray(3) { bandSD(img0, v, it) }
            val tol = 0.001f
            println("== probe k=%.1f: smallest box keeping every band sigma <= baseline (+%s) ==".format(v.k, tol))
            println("s1   s3   box4pass box3pass box2pass box1pass  -> minBox (maxRatio over boxes)")
            for (i1 in 0..10) for (i3 in 0..10) {
                val s1 = i1 / 10f
                val s3 = i3 / 10f
                if (s1 <= 0f && s3 <= 0f) continue
                val pack = buildPackGL(scene, v, s1, s3)
                val pass = BooleanArray(4)
                var maxRatio = 0f
                for (bi in 1..4) {   // box = bi
                    // box4 pairs with the 4-tap ring exactly as shipped; boxes
                    // 1-3 keep the full 12-tap ring.
                    val img = renderPreview(scene, pack, v, s1, s3, boxAA = bi, nrRadius = if (bi == 4) 2 else 4)
                    var ok = true
                    for (ch in 0..2) {
                        val sd = bandSD(img, v, ch)
                        maxRatio = max(maxRatio, sd / sd0[ch])
                        if (sd > sd0[ch] + tol) ok = false
                    }
                    pass[bi - 1] = ok
                }
                val minBox = if (pass[0]) 1 else if (pass[1]) 2 else if (pass[2]) 3 else 4
                println("%.1f %.1f   %s   %s   %s   %s   -> %d  (maxRatio=%.2f)".format(
                    s1, s3, pass[3], pass[2], pass[1], pass[0], minBox, maxRatio
                ))
            }
        }
    }

    /**
     * Diagnostic: monotonicity of per-band sigma as each slider rises, at
     * every zoom level (pack regime k<=2 AND inline regime k>2).  Walks s3 at
     * fixed s1 and flags any band that pops above the previous (weaker)
     * slider value.  Silently also home the first step (s3 from baseline, so
     * the origin bound is covered too).  `variant`: "shipped" uses the host
     * mapping; "box4" forces boxAA=4 + 4-tap ring at every strength (the box
     * keeps carrying the averaging, so sigma should move strictly down).
     * Print-only.
     */
    @Test
    fun probeSliderMonotonicity() {
        val scene = buildFlatNoise(29, 460f, 28f)
        val tol = 0.001f
        val s1s = listOf(0f, 0.3f, 0.6f)
        val s3s = listOf(0f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f)
        runMonotoneWalk(scene, tol, s1s, s3s, listOf("shipped", "box4"), mirror = false, gateTol = null)
    }

    private fun runMonotoneWalk(
        scene: ShortArray, tol: Float, s1s: List<Float>, s3s: List<Float>,
        variants: List<String>,
        mirror: Boolean,
        gateTol: Float?
    ) {
        val failures = StringBuilder()
        for (grid in listOf(64, 48, 32, 24, 12)) {   // k=1.5, 2, 3, 4, 8
            val v = View(grid, grid, mirror, true, floatArrayOf(0f, 0f, 96f, 96f))
            val pack0 = buildPackGL(scene, v, 0f, 0f)
            val sd0 = FloatArray(3) { bandSD(renderPreview(scene, pack0, v, 0f, 0f), v, it) }
            println("== probe monotonicity k=%.2f (grid=%d) %s baseline R=%.4f G=%.4f B=%.4f ==".format(
                v.k, grid, if (mirror) "mirror" else "back", sd0[0], sd0[1], sd0[2]))
            for (variant in variants) {
                val pops = StringBuilder()
                val probeMode = when (variant) {
                    "mask" -> 1
                    "dir" -> 2
                    "region" -> 3
                    "dirC" -> 4
                    else -> 0
                }
                val smooth = variant == "smooth0" || variant == "smooth3"
                var maxDelta = 0f   // max |band sigma(mode) − band sigma(shipped)| over the walk
                println("-- variant=$variant")
                for (s1 in s1s) {
                    var prev = sd0.copyOf()
                    for (s3 in s3s) {
                        if (s1 <= 0f && s3 <= 0f) continue
                        val pack = buildPackGL(scene, v, s1, s3, mode = probeMode)
                        val box = when (variant) {
                            "box4" -> 4
                            else -> previewBoxAA(v, s1, s3)
                        }
                        val nrR = when (variant) {
                            "box4" -> 2
                            else -> previewNrRadius(box)
                        }
                        val img = renderPreview(scene, pack, v, s1, s3, boxAA = box, nrRadius = nrR, mode = probeMode, boxSmooth = smooth)
                        val sd = FloatArray(3) { bandSD(img, v, it) }
                        val delta = if (smooth) {
                            // shipped + stepped box as the reference for how far
                            // the smooth/directional variant moves the noise bands.
                            val packS = buildPackGL(scene, v, s1, s3)
                            val imgS = renderPreview(scene, packS, v, s1, s3, boxAA = box, nrRadius = nrR)
                            FloatArray(3) { abs(sd[it] - bandSD(imgS, v, it)) }
                        } else {
                            FloatArray(3) { 0f }
                        }
                        for (ch in 0..2) if (delta[ch] > maxDelta) maxDelta = delta[ch]
                        var up = ""
                        var anyUp = false
                        for (ch in 0..2) {
                            if (sd[ch] > prev[ch] + tol) {
                                up += " %c:%.4f>%.4f".format(charArrayOf('R', 'G', 'B')[ch], sd[ch], prev[ch])
                                anyUp = true
                            }
                        }
                        if (gateTol != null) {
                            val gUp = (0..2).any { sd[it] > prev[it] + gateTol }
                            if (gUp) {
                                failures.append(
                                    "k=%.2f %s s1=%.1f s3=%.1f box=%d nr=%d  R=%.4f(G:%.4f) G=%.4f(G:%.4f) B=%.4f(G:%.4f)\n".format(
                                        v.k, if (mirror) "mirror" else "back", s1, s3, box, nrR,
                                        sd[0], prev[0], sd[1], prev[1], sd[2], prev[2]
                                    )
                                )
                            }
                        }
                        if (anyUp) pops.append("k=%.2f s1=%.1f s3=%.1f box=%d nr=%d%s\n".format(v.k, s1, s3, box, nrR, up))
                        println("  s1=%.2f s3=%.2f box=%d nr=%d  R=%.5f G=%.5f B=%.5f%s".format(
                            s1, s3, box, nrR, sd[0], sd[1], sd[2],
                            if (anyUp) "  <-- POP" else if (smooth) "  dR=%.5f dG=%.5f dB=%.5f".format(delta[0], delta[1], delta[2]) else ""))
                        prev = sd
                    }
                }
                println("variant=$variant pops: " + (if (pops.isEmpty()) "none" else pops.toString().trim()))
                if (smooth) println("variant=$variant maxDelta vs shipped: %.5f".format(maxDelta))
            }
        }
        if (gateTol != null) {
            assertTrue(
                "slider increase popped a band's sigma above the previous value (violates monotone denoising):\n" + failures,
                failures.isEmpty()
            )
        }
    }

    /**
     * Acceptance gate for "increasing S1/S3 must always reduce noise, at every
     * zoom level": walk each slider upward along the shipped mapping and
     * require every band's sigma to never rise above the previous (weaker)
     * slider value.  Runs both back/mirror configs across k=1.5..8 (pack and
     * inline regimes).  tol is set just above the >0.0005 pops the old box3
     * step-down produced, so those fail while sub-0.0003 rounding (see the
     * k=2 G band) passes.
     */
    @Test
    fun slidersNeverRaiseSigmaAtAnyZoom() {
        val scene = buildFlatNoise(29, 460f, 28f)
        val s1s = listOf(0f, 0.3f, 0.6f)
        val s3s = listOf(0f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f)
        // Deployed filter = smooth box-AA + regional directional flat pull
        // (mode 3): continuous box across s3, so the walk both gates the
        // monotonicity AND pins the deployed demosaic's per-band sigma.
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("smooth3"), mirror = false, gateTol = 0.0004f)
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("smooth3"), mirror = true, gateTol = 0.0004f)
    }

    /**
     * Experiment: re-key the S3 flat/structure threshold on the MEASURED σ̂
     * (the "only denoise noisy regions" mask) instead of the under-reporting
     * model σ.  On the gate scene σ̂ measures the real noise (~16), so with
     * K chosen above the pure-noise maxNb tail the flat branch keeps the full
     * α-trim pull and the gate must hold identically to "shipped".  Prints σ̂
     * stats so the tail landing can be checked.
     */
    @Test
    fun probeNoiseMaskMonotonicity() {
        val scene = buildFlatNoise(29, 460f, 28f)
        val s1s = listOf(0f, 0.3f, 0.6f)
        val s3s = listOf(0f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f)
        // σ̂ stats over the gate scene, once, for knee calibration.
        val shs = FloatArray(2000)
        var n = 0
        for (y in 8 until SH - 8) for (x in 8 until SW - 8) { if (n < 2000) shs[n++] = sigmaHatAt(scene, x, y) }
        java.util.Arrays.sort(shs, 0, n)
        println("sigmaHat gate scene: p10=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f".format(
            shs[0], shs[n * 5 / 10], shs[n * 9 / 10], shs[n * 99 / 100], shs[n - 1]))
        // r = minDev/maxNb on the gate scene (calibration for the directional
        // blend's omega band: it must stay 0 for noise, turn on for lines).
        val rs = FloatArray(4096)
        var rn = 0
        for (y in 8 until SH - 8) for (x in 8 until SW - 8) {
            val mm = structRatioAt(scene, x, y)
            if (rn < 4096 && mm > 0f) rs[rn++] = mm
        }
        java.util.Arrays.sort(rs, 0, rn)
        println("minDev/maxNb gate scene: p50=%.3f p90=%.3f p95=%.3f p99=%.3f".format(
            rs[rn / 2], rs[rn * 9 / 10], rs[rn * 95 / 100], rs[rn * 99 / 100]))
        // r2 = coarseDev/maxNb (regional coarse/fine ratio, mode=3): on noise the
        // block-mean spread collapses below the per-pixel max dev (σ/√N), so the
        // ratio should sit BELOW the 0.6..1.0 ω band leaving the flat pull
        // bit-identical; texture raised blocks push it over.
        val r2s = FloatArray(4096)
        var r2n = 0
        for (y in 8 until SH - 8) for (x in 8 until SW - 8) {
            if (r2n < 4096) {
                val mb = maxNbAt(scene, x, y)
                if (mb > 1e-6f) r2s[r2n++] = coarseDevAt(scene, x, y) / mb
            }
        }
        java.util.Arrays.sort(r2s, 0, r2n)
        println("coarseDev/maxNb gate scene: p50=%.3f p90=%.3f p95=%.3f p99=%.3f max=%.3f".format(
            r2s[r2n / 2], r2s[r2n * 9 / 10], r2s[r2n * 95 / 100], r2s[r2n * 99 / 100], r2s[r2n - 1]))
        coarseRatioCov(scene, "coarse/fine gate scene (бand coverage)")
        // mode=1 (σ̂-keyed flat gate) is the recorded negative result: gate-safe
        // but a no-op on lines (σ̂ MAD self-inflates where the line sits), so it
        // must KEEP the monotonicity gate.  mode=2 (directional flat target)
        // preserves lines on paper (46.9%->20% s3Lift) but the noise r tail
        // overlaps the line's band (p50=0.179), so it pops the boxAA 4->2 step
        // at s3>=0.7 — printed, not asserted, to record the dead end.  mode=3
        // (regional coarse/fine blend) is the current candidate, printed first so
        // the ω band can be calibrated to the measured noise-ratio tail before
        // asserting the gate.
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("mask"), mirror = false, gateTol = 0.0004f)
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("mask"), mirror = true, gateTol = 0.0004f)
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("dir"), mirror = false, gateTol = null)
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("region"), mirror = false, gateTol = null)
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("region"), mirror = true, gateTol = null)
        // mode=4 (constant directional flat pull): recorded negative result.  It
        // preserves the thin line (46.9% -> 30.4% s3Lift) but FAILS the production
        // gate at the boxAA 4->2 step (k=3.00 back s1=0.3 s3=0.7 box=2 nr=4:
        // R 0.0054>0.0052, G 0.0040>0.0036, B 0.0056>0.0052) — the flat-branch
        // pull target must stay bit-identical iavg for the gate to hold, so any
        // iDir blend (masked or constant) is excluded.  maxDelta vs shipped on
        // noise reaches 0.0013-0.0016 at k>=3.  Printed, not asserted.
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("dirC"), mirror = false, gateTol = null)
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("dirC"), mirror = true, gateTol = null)
        // Smooth box-AA prototype: with the box shape blended continuously over
        // s3 there is no 4->2 step for the directional pull to pop.  smooth0 =
        // shipped pull + smooth box (isolates the box change), smooth3 = regional
        // directional pull + smooth box (the full candidate).  Both must hold the
        // 0.0004 gate on both configs, or the smooth-box fix is rejected.
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("smooth0"), mirror = false, gateTol = 0.0004f)
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("smooth0"), mirror = true, gateTol = 0.0004f)
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("smooth3"), mirror = false, gateTol = 0.0004f)
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("smooth3"), mirror = true, gateTol = 0.0004f)
    }

    /** minDev/maxNb at a sensor pixel (mirror of the sameColorNR devs). */
    private fun structRatioAt(v: ShortArray, sx: Int, sy: Int): Float {
        val maxNb = maxNbAt(v, sx, sy)
        val minDev = minDevAt(v, sx, sy)
        return if (maxNb > 1e-6f) minDev / maxNb else 1f
    }

    /** coarseDev/maxNb distribution over a scene, with ω-band coverage so the
     *  mode-3 threshold can be placed between the noise tail and the texture.
     *  (0.6, 1.0, 1.5, 2.5, 4) are candidate rLo bands. */
    private fun coarseRatioCov(scene: ShortArray, label: String) {
        val rs = FloatArray(16384)
        var n = 0
        for (y in 12 until SH - 12) for (x in 12 until SW - 12) {
            if (n < 16384) {
                val mb = maxNbAt(scene, x, y)
                if (mb > 1e-6f) rs[n++] = coarseDevAt(scene, x, y) / mb
            }
        }
        if (n == 0) return
        java.util.Arrays.sort(rs, 0, n)
        fun frac(lo: Float, hi: Float): Int {
            var c = 0
            for (i in 0 until n) if (rs[i] >= lo && rs[i] < hi) c++
            return (c * 100f / n).roundToInt()
        }
        val p = { i: Int -> if (i < n) rs[i] else rs[n - 1] }
        println("%s  p50=%.2f p90=%.2f p95=%.2f p99=%.2f | ω>=0.6:0.6-1.0=%d%% 1.0-1.5=%d%% 1.5-2.5=%d%% 2.5-4=%d%% 4+=%d%%".format(
            label, p(n / 2), p(n * 9 / 10), p(n * 95 / 100), p(n * 99 / 100),
            frac(0.6f, 1.0f), frac(1.0f, 1.5f), frac(1.5f, 2.5f), frac(2.5f, 4f), frac(4f, 1e9f)))
    }

    /** max over the four axis-trimmed deviations (the structure detector). */
    private fun maxNbAt(v: ShortArray, sx: Int, sy: Int): Float {
        val c = sensorVal(v, sx, sy)
        val nN = sensorVal(v, sx, sy - 2)
        val nS = sensorVal(v, sx, sy + 2)
        val nE = sensorVal(v, sx + 2, sy)
        val nW = sensorVal(v, sx - 2, sy)
        val nNE = sensorVal(v, sx + 2, sy - 2)
        val nNW = sensorVal(v, sx - 2, sy - 2)
        val nSE = sensorVal(v, sx + 2, sy + 2)
        val nSW = sensorVal(v, sx - 2, sy + 2)
        val nNN = sensorVal(v, sx, sy - 4)
        val nSS = sensorVal(v, sx, sy + 4)
        val nEE = sensorVal(v, sx + 4, sy)
        val nWW = sensorVal(v, sx - 4, sy)
        val tN = nNN + nNE + nNW + c - minOf(minOf(nNN, nNE), minOf(nNW, c)) - maxOf(maxOf(nNN, nNE), maxOf(nNW, c))
        val devN = abs(nN - tN * 0.5f)
        val tS = nSS + nSE + nSW + c - minOf(minOf(nSS, nSE), minOf(nSW, c)) - maxOf(maxOf(nSS, nSE), maxOf(nSW, c))
        val devS = abs(nS - tS * 0.5f)
        val tE = nEE + nNE + nSE + c - minOf(minOf(nEE, nNE), minOf(nSE, c)) - maxOf(maxOf(nEE, nNE), maxOf(nSE, c))
        val devE = abs(nE - tE * 0.5f)
        val tW = nWW + nNW + nSW + c - minOf(minOf(nWW, nNW), minOf(nSW, c)) - maxOf(maxOf(nWW, nNW), maxOf(nSW, c))
        val devW = abs(nW - tW * 0.5f)
        return maxOf(maxOf(devN, devS), maxOf(devE, devW))
    }

    /** min over the four axis-trimmed deviations. */
    private fun minDevAt(v: ShortArray, sx: Int, sy: Int): Float {
        val c = sensorVal(v, sx, sy)
        val nN = sensorVal(v, sx, sy - 2)
        val nS = sensorVal(v, sx, sy + 2)
        val nE = sensorVal(v, sx + 2, sy)
        val nW = sensorVal(v, sx - 2, sy)
        val nNE = sensorVal(v, sx + 2, sy - 2)
        val nNW = sensorVal(v, sx - 2, sy - 2)
        val nSE = sensorVal(v, sx + 2, sy + 2)
        val nSW = sensorVal(v, sx - 2, sy + 2)
        val nNN = sensorVal(v, sx, sy - 4)
        val nSS = sensorVal(v, sx, sy + 4)
        val nEE = sensorVal(v, sx + 4, sy)
        val nWW = sensorVal(v, sx - 4, sy)
        val tN = nNN + nNE + nNW + c - minOf(minOf(nNN, nNE), minOf(nNW, c)) - maxOf(maxOf(nNN, nNE), maxOf(nNW, c))
        val devN = abs(nN - tN * 0.5f)
        val tS = nSS + nSE + nSW + c - minOf(minOf(nSS, nSE), minOf(nSW, c)) - maxOf(maxOf(nSS, nSE), maxOf(nSW, c))
        val devS = abs(nS - tS * 0.5f)
        val tE = nEE + nNE + nSE + c - minOf(minOf(nEE, nNE), minOf(nSE, c)) - maxOf(maxOf(nEE, nNE), maxOf(nSE, c))
        val devE = abs(nE - tE * 0.5f)
        val tW = nWW + nNW + nSW + c - minOf(minOf(nWW, nNW), minOf(nSW, c)) - maxOf(maxOf(nWW, nNW), maxOf(nSW, c))
        val devW = abs(nW - tW * 0.5f)
        return minOf(devN, devS, devE, devW)
    }

    /**
     * Purpose test for the noise-region mask: a thin bright line on noise must
     * survive S3 (the flat-branch α-trim pull is what smears it).  Measures
     * the green dip the line holds with S1-only vs S1+S3, mask off (shipped
     * V3) vs on (σ̂-keyed).  s3Lift is how much S3 dims the line; a working
     * mask must LOWER s3Lift (line preserved) while keeping the gate.
     */
    @Test
    fun probeNoiseMaskThinLine() {
        // 1px vertical line at sensor col 48, height +24 (~3x the sensor noise
        // 8), on a mono flat noise base 460.
        val base = buildFlatNoise(77, 460f, 8f)
        val line = base.copyOf()
        for (y in 0 until SH) line[y * SW + 48] =
            (SENSOR_BLACK + 460f + 24f + (Random(1).nextFloat() - 0.5f) * 2f * 8f)
                .roundToInt().coerceIn(0, SENSOR_CLIP).toShort()
        val v = View(96, 96, false, true, floatArrayOf(0f, 0f, 96f, 96f))
        val s1 = 0.3f
        val s3 = 0.6f
        println("== probeNoiseMaskThinLine (s1=$s1 s3=$s3) green band dip at the line ==")
        // Where does the coarse/fine ratio stand at the line vs off-line?  The
        // mode-3 ω band must sit between the noise ratio and the texture ratio.
        val onR = FloatArray(64)
        val offR = FloatArray(64)
        var onn = 0
        var offn = 0
        for (y in 40..55) {
            for (sx in listOf(48, 49)) if (onn < 64) onR[onn++] = (coarseDevAt(line, sx, y) / maxOf(maxNbAt(line, sx, y), 1e-6f))
            for (sx in listOf(45, 46, 50, 51)) if (offn < 64) offR[offn++] = (coarseDevAt(line, sx, y) / maxOf(maxNbAt(line, sx, y), 1e-6f))
        }
        java.util.Arrays.sort(onR, 0, onn)
        java.util.Arrays.sort(offR, 0, offn)
        println("coarse/fine ratio on-line: p50=%.2f p90=%.2f p99=%.2f   off-line: p50=%.2f p90=%.2f p99=%.2f".format(
            onR[onn / 2], onR[onn * 9 / 10], onR[onn * 99 / 100], offR[offn / 2], offR[offn * 9 / 10], offR[offn * 99 / 100]))
        val lift = FloatArray(5)
        for (mode in 0..4) {
            val dip1 = lineDip(line, v, s1, 0f, mode)
            val dip3 = lineDip(line, v, s1, s3, mode)
            lift[mode] = (dip1 - dip3) / max(dip1, 1e-6f)
            println("  mode=$mode  dip(s1)=%.4f dip(s1+s3)=%.4f  s3Lift=%.1f%%".format(dip1, dip3, lift[mode] * 100))
        }
        // Directional flat target (mode 2) must preserve the line materially
        // better than the shipped flat pull (mode 0).  The regional blend
        // (mode 3) is measured below; its ω band gets calibrated to the
        // printed ratio stats.  Mode 4 (constant directional pull) is the
        // shipping candidate — it must beat shipped by the same margin.
        assertTrue(
            "directional flat target must preserve the line better than shipped V3 " +
                "(s3Lift ${"%.1f".format(lift[0] * 100)}% -> ${"%.1f".format(lift[2] * 100)}%)",
            lift[2] < lift[0] - 0.10f
        )
    }

    private fun lineDip(scene: ShortArray, v: View, s1: Float, s3: Float, mode: Int): Float {
        val pack = buildPackGL(scene, v, s1, s3, mode = mode)
        val img = renderPreview(scene, pack, v, s1, s3, mode = mode)
        val row = 48
        val onLine = mutableListOf<Float>()
        val offLine = mutableListOf<Float>()
        for (gx in 44 until 52) {
            val g = img[row][gx * 3 + 1]
            if (gx in 47..49) onLine.add(g) else offLine.add(g)
        }
        val on = onLine.sum() / onLine.size
        val off = offLine.sum() / offLine.size
        return on - off
    }

    /**
     * Multiscale foliage-like texture: vertical leaf streaks (3 sine harmonics,
     * per-column contrast ~8-10/px), 4 single-px grass-blade spikes (+30), a
     * slow vertical undulation, on a mono base 460 (+ noise σ when requested).
     * The noise-free (σ=0) twin is the witness for PURE structure smearing.
     */
    private fun buildFoliage(seed: Int, base: Float, sigma: Float): ShortArray {
        val rnd = Random(seed)
        val bump = FloatArray(SW)
        for (x in 0 until SW) {
            bump[x] = 18f * sin(2.0 * PI * x / 16.0).toFloat() +
                12f * sin(2.0 * PI * x / 8.0 + 1.0).toFloat() +
                6f * sin(2.0 * PI * x / 4.0 + 2.0).toFloat()
        }
        val s = ShortArray(SW * SH)
        for (y in 0 until SH) for (x in 0 until SW) {
            var g = base + bump[x]
            if (x == 20 || x == 34 || x == 60 || x == 78) g += 30f
            g += 4f * sin(2.0 * PI * y / 24.0).toFloat()
            if (sigma > 0f) g += (rnd.nextFloat() - 0.5f) * 2f * sigma
            s[y * SW + x] = (SENSOR_BLACK + g).roundToInt().coerceIn(0, SENSOR_CLIP).toShort()
        }
        return s
    }

    /** Mean |G(gx+1) − G(gx)| over the green band — local texture/noise energy. */
    private fun texEnergy(img: Array<FloatArray>, v: View): Float {
        var sum = 0.0
        var n = 0
        for (gy in 4 until v.gridH - 4) {
            for (gx in 4 until v.gridW - 1) {
                sum += abs(img[gy][(gx + 1) * 3 + 1] - img[gy][gx * 3 + 1])
                n++
            }
        }
        return (sum / n).toFloat()
    }

    /**
     * Real-texture regime probe: on foliage-like texture (contrast ~8-10/px ≫
     * noise at the moderate σ used here) does the directional flat pull (modes
     * 2-4) keep the TEXTURE while S3 still removes the NOISE, vs the shipped
     * flat pull (mode 0)?  Two witnesses:
     *   - texLift  = 1 − E(clean scene with S3)/E(clean scene without): how much
     *     of the pure structure S3 flattens (0 = perfect preservation).  The
     *     clean scene filters out the noise term so this is structure-only.
     *   - noiseLift = 1 − E(noisy scene with S3)/E(noisy scene without): how
     *     much TOTAL energy S3 removes (noise + whatever structure it also
     *     takes).  A working mask keeps noiseLift near shipped while texLift
     *     drops well below shipped.
     */
    @Test
    fun probeFoliageTextureLift() {
        val clean = buildFoliage(1, 460f, 0f)
        val s1 = 0.3f
        val s3 = 0.6f
        val v = View(96, 96, false, true, floatArrayOf(0f, 0f, 96f, 96f))
        coarseRatioCov(clean, "coarse/fine clean foliage")
        for (sigma in listOf(8f, 14f)) {
            val noisy = buildFoliage(1, 460f, sigma)
            coarseRatioCov(noisy, "coarse/fine foliage sigma=$sigma")
        }
        val pack0 = buildPackGL(clean, v, 0f, 0f)
        val ec0 = texEnergy(renderPreview(clean, pack0, v, 0f, 0f), v)
        println("== probeFoliageTextureLift (s1=$s1 s3=$s3) clean energy=%.5f ==".format(ec0))
        val modes = listOf(0, 2, 3, 4)
        val bands = listOf(0.6f to 1.0f, 1.5f to 2.5f)
        for (sigma in listOf(8f, 14f)) {
            val noisy = buildFoliage(1, 460f, sigma)
            val pN0 = buildPackGL(noisy, v, 0f, 0f)
            val en0 = texEnergy(renderPreview(noisy, pN0, v, 0f, 0f), v)
            println("-- sigma=$sigma  noisy energy=%.5f".format(en0))
            for (mode in modes) {
                if (mode == 3) {
                    for ((lo, hi) in bands) {
                        omegaLo = lo; omegaHi = hi
                        val packN = buildPackGL(noisy, v, s1, s3, mode = mode)
                        val imgN = renderPreview(noisy, packN, v, s1, s3, mode = mode)
                        val noiseLift = 1f - texEnergy(imgN, v) / en0
                        val packC = buildPackGL(clean, v, s1, s3, mode = mode)
                        val imgC = renderPreview(clean, packC, v, s1, s3, mode = mode)
                        val texLift = 1f - texEnergy(imgC, v) / ec0
                        println("  mode=%d band=(%.1f,%.1f)  texLift=%.2f%%  noiseLift=%.2f%%".format(mode, lo, hi, texLift * 100, noiseLift * 100))
                    }
                    omegaLo = 0.6f; omegaHi = 1.0f
                } else {
                    val packN = buildPackGL(noisy, v, s1, s3, mode = mode)
                    val imgN = renderPreview(noisy, packN, v, s1, s3, mode = mode)
                    val noiseLift = 1f - texEnergy(imgN, v) / en0
                    val packC = buildPackGL(clean, v, s1, s3, mode = mode)
                    val imgC = renderPreview(clean, packC, v, s1, s3, mode = mode)
                    val texLift = 1f - texEnergy(imgC, v) / ec0
                    println("  mode=%d  texLift=%.2f%%  noiseLift=%.2f%%".format(mode, texLift * 100, noiseLift * 100))
                }
            }
        }
    }

    @Test
    fun probeHotPixelLeakDuringZoomIn() {
        val base = buildFlatNoise(2025, 460f, 28f)
        // One clipped hot pixel per CFA phase, spread so edge clamp never
        // corrupts the trim ring.
        val hot = listOf(
            24 to 24, 25 to 24, 24 to 25, 25 to 25,   // phases 0,1,2,3 (2x2 block)
            72 to 24, 72 to 48, 48 to 72, 24 to 72    // more, both parities
        )
        val leaky = base.copyOf()
        for ((hx, hy) in hot) leaky[hy * SW + hx] = SENSOR_CLIP.toShort()

        val grids = listOf(96, 90, 84, 78, 72, 66, 60, 54, 50, 48, 46, 44, 40, 38, 36, 34, 32, 30, 28, 24, 20, 16, 12)
        val visible = 0.05f   // output is 0..1; a surviving spike in the leaking
                              // channel reads ~1.0, dilution < 0.05 is suppressed
        var anyVisible = false
        for (grid in grids) {
            val v = View(grid, grid, false, true, floatArrayOf(0f, 0f, 96f, 96f))
            for (s1 in listOf(0.3f, 0.6f)) {
                val p = buildPackGL(leaky, v, s1, 0f)
                val imgH = renderPreview(leaky, p, v, s1, 0f)
                val imgC = renderPreview(base, buildPackGL(base, v, s1, 0f), v, s1, 0f)
                val bx = previewBoxAA(v, s1, 0f)
                val nr = previewNrRadius(bx)
                val mode = if (packActive(v)) "pack" else "inline"
                var leaks = 0
                var maxBump = 0f
                for ((hx, hy) in hot) {
                    val gx0 = max(((hx + 0.5f) / 96f * grid).toInt() - 2, 0)
                    val gy0 = max(((hy + 0.5f) / 96f * grid).toInt() - 2, 0)
                    val gx1 = min(gx0 + 4, grid - 1)
                    val gy1 = min(gy0 + 4, grid - 1)
                    var bump = 0f
                    for (gy in gy0..gy1) for (gx in gx0..gx1)
                        for (c in 0..2) bump = max(bump, imgH[gy][gx * 3 + c] - imgC[gy][gx * 3 + c])
                    maxBump = max(maxBump, bump)
                    if (bump > visible) leaks++
                }
                val flag = if (leaks > 0) "  <-- HOT PIXELS LEAK" else ""
                if (leaks > 0) anyVisible = true
                println("zoom-in k=%.2f (grid=%d) s1=%.1f %s box=%d nr=%d  leaks=%d/%d maxBump=%.4f%s".format(
                    v.k, grid, s1, mode, bx, nr, leaks, hot.size, maxBump, flag))
                if ((grid == 96 || grid == 48) && s1 == 0.3f) {
                    // Reverse-map hot@(72,48) to its pack texel; dump the p0 across the
                    // 3x3 texel neighbourhood so both k=1 (leak-free) and k=2 (leaking)
                    // show where the residual sits.
                    val rr = ((24f / 96f) * grid).toInt().coerceIn(2, grid - 3)
                    val cc = ((72f / 96f) * grid).toInt().coerceIn(2, grid - 3)
                    println("  pack grid=%d texel centre (%d,%d), 3x3 x 4ch:".format(grid, cc, rr))
                    for (gy in cc - 1..cc + 1) {
                        for (gx in rr - 1..rr + 1) {
                            val i = (gy * grid + gx) * 4
                            println("    (%d,%d): p0=%.1f p1=%.1f p2=%.1f p3=%.1f".format(
                                gx, gy, p[i + 0], p[i + 1], p[i + 2], p[i + 3]))
                        }
                    }
                    println("  sameColorNR(24,72)=%.1f sameColorNR(24,74)=%.1f sameColorNR(26,72)=%.1f".format(
                        sameColorNR(leaky, 24, 72, 0.3f, 0f), sameColorNR(leaky, 24, 74, 0.3f, 0f), sameColorNR(leaky, 26, 72, 0.3f, 0f)))
                    for ((hx, hy) in hot) {
                        val gx0 = max(((hx + 0.5f) / 96f * grid).toInt() - 2, 0)
                        val gy0 = max(((hy + 0.5f) / 96f * grid).toInt() - 2, 0)
                        val gx1 = min(gx0 + 4, grid - 1)
                        val gy1 = min(gy0 + 4, grid - 1)
                        var bgx = gx0; var bgy = gy0; var bc = 0; var bump = -1f; var hotVal = 0f; var ctl = 0f
                        for (gy in gy0..gy1) for (gx in gx0..gx1)
                            for (c in 0..2) {
                                val d = imgH[gy][gx * 3 + c] - imgC[gy][gx * 3 + c]
                                if (d > bump) { bump = d; bgx = gx; bgy = gy; bc = c; hotVal = imgH[gy][gx * 3 + c]; ctl = imgC[gy][gx * 3 + c] }
                            }
                        println("  hot@(%d,%d) phase=%d -> maxBump texel(%d,%d) ch%d leaky=%.4f ctl=%.4f bump=%.4f".format(
                            hx, hy, phase(hx, hy), bgx, bgy, bc, hotVal, ctl, bump))
                        if (bump > visible) {
                            // Replicate renderPreview's own box-AA loop for the maxBump
                            // texel and compare raw channel averages with the rendered.
                            val gx = bgx; val gy = bgy
                            val svX = v.crop[0] + v.T((gx + 0.5f) / v.gridW, 0) * v.crop[2]
                            val svY = v.crop[1] + v.T((gy + 0.5f) / v.gridH, 1) * v.crop[3]
                            val bx0 = boxBase(svX, 4)
                            val by0 = boxBase(svY, 4)
                            val ssum = FloatArray(3)
                            val sb = StringBuilder()
                            for (dy in 0..3) for (dx in 0..3) {
                                val cx = (bx0 + dx).coerceIn(0, SW - 1)
                                val cy = (by0 + dy).coerceIn(0, SH - 1)
                                val s = demosaicAtRGB(leaky, p, v, 0.3f, 0f, cx, cy, nrRadius = 2)
                                for (c in 0..2) ssum[c] += s[c]
                                sb.append("(%d,%d)p%d=%s ".format(cx, cy, phase(cx, cy),
                                    s.joinToString("/") { "%.0f".format(it) }))
                            }
                            for (c in 0..2) ssum[c] /= 16f
                            val pre = composeMain(ssum)
                            println("  repl k=%.2f texel(%d,%d) raw=(%.1f,%.1f,%.1f) pre=(%.3f,%.3f,%.3f) imgH=(%.3f,%.3f,%.3f) imgC=(%.3f,%.3f,%.3f)".format(
                                v.k, gx, gy, ssum[0], ssum[1], ssum[2], pre[0], pre[1], pre[2],
                                imgH[gy][gx * 3], imgH[gy][gx * 3 + 1], imgH[gy][gx * 3 + 2],
                                imgC[gy][gx * 3], imgC[gy][gx * 3 + 1], imgC[gy][gx * 3 + 2]))
                        }
                    }
                }
            }
        }
        println("visible hot-pixel leak during zoom-in: " + anyVisible)
    }

    // ------------------------------------------------------------------
    // Device-failure-mode probe: which single deltas actually produce a
    // per-texel magenta mosaic on flat noise?  Not part of the invariant
    // suite — diagnostic printout only (the CPU model is a fixed point, so a
    // probe row that explodes identifies the device-side mechanism to defend
    // against).
    // ------------------------------------------------------------------

    private fun magentaCount(img: Array<FloatArray>, v: View): Int {
        var cnt = 0
        var tot = 0
        for ((gx, gy) in interiorMask(v)) {
            val r = img[gy][gx * 3]
            val g = img[gy][gx * 3 + 1]
            val b = img[gy][gx * 3 + 2]
            val den = max(max(r, g), b)
            if (den > 1e-4f) {
                val rb = (r - b) / den
                val bg = (b - g) / den
                if (rb > 0.25f && bg > 0.25f) cnt++ else if (rb < -0.25f && bg < -0.25f) cnt++
                tot++
            }
        }
        return if (tot == 0) 0 else cnt
    }

    @Test
    fun probeDeviceFailureModes() {
        val s1 = 0.6f
        val s3 = 0.6f
        val scene = buildFlatNoise(7, 460f, 28f)
        val runs = listOf(
            Probe("k=0.5 basel", 96, 48, 48, 0, 0, floatArrayOf(24f, 24f, 48f, 48f)),
            Probe("k=0.5 shiftX=+1", 96, 48, 48, 1, 0, floatArrayOf(24f, 24f, 48f, 48f)),
            Probe("k=0.5 shiftY=+1", 96, 48, 48, 0, 1, floatArrayOf(24f, 24f, 48f, 48f)),
            Probe("k=0.5 shiftX=-1", 96, 48, 48, -1, 0, floatArrayOf(24f, 24f, 48f, 48f)),
            Probe("k=0.5 oddOrig", 96, 48, 48, 0, 0, floatArrayOf(23f, 25f, 48f, 48f)),
            Probe("k=0.5 shiftX+1 odd", 96, 48, 48, 1, 0, floatArrayOf(23f, 25f, 48f, 48f)),
            Probe("k=0.8 crop77", 96, 77, 77, 0, 0, floatArrayOf(9f, 9f, 77f, 77f)),
            Probe("k=0.8 shiftX+1", 96, 77, 77, 1, 0, floatArrayOf(9f, 9f, 77f, 77f)),
            Probe("k=0.8 oddOrig", 96, 78, 77, 0, 0, floatArrayOf(8f, 9f, 78f, 77f)),
            Probe("k=0.8 oddShift", 96, 78, 77, 1, 1, floatArrayOf(8f, 9f, 78f, 77f)),
            Probe("k=1.0 full", 96, 96, 96, 0, 0, floatArrayOf(0f, 0f, 96f, 96f)),
            Probe("k=1.0 shiftY=+1", 96, 96, 96, 0, 1, floatArrayOf(0f, 0f, 96f, 96f))
        )
        println("== probe: device-failure modes (flatNoise, s1=s3=0.6) ==")
        for (r in runs) {
            val v = View(r.grid, r.grid, false, true, r.crop)
            val pack = buildPackGL(scene, v, s1, s3)
            val img = renderPreview(scene, pack, v, s1, s3, r.sx, r.sy)
            val pack0 = buildPackGL(scene, v, 0f, 0f)
            val img0 = renderPreview(scene, pack0, v, 0f, 0f)
            val m = metrics(img, v)
            val m0 = metrics(img0, v)
            val mag = magentaCount(img, v)
            val mag0 = magentaCount(img0, v)
            println(
                "%-22s magErr=%.4f gNoise=%.3f magDN=%2d  (s0: magErr=%.4f gNoise=%.3f magDN=%2d)".format(
                    r.label, m[0], m[1], mag, m0[0], m0[1], mag0
                )
            )
        }
    }

    private class Probe(
        val label: String, val grid: Int, val cw: Int, val ch: Int,
        val sx: Int, val sy: Int, val crop: FloatArray
    )

    // ------------------------------------------------------------------
    // Cross-phase boxedBlend drag with IMbalanced phase bases (real-sensor
    // shape) + the full shader main() WB/colorMat chain.  The equal-base flat
    // scene above makes the drag a fixed point (iavg==center), which is why the
    // main suite can show magDN=0 while the device shows magenta: on a real
    // sensor R/G/B phase bases differ, the boxed mean's cross-phase iavg pulls
    // R or B toward the green phase, and the WB red/blue gains then magnify the
    // residual into a magenta cast.  With the live k<=2 gate, boxedBlend is
    // only reachable for 2..4-cell boxes (k≈1.2..2.0); the k=3..8 rows below
    // no longer take the boxed branch on device — they read inline, so their
    // pack-vs-inline delta is ~0 (kept as diagnostics of the device behaviour).
    // ------------------------------------------------------------------
    private fun magentaAvail(img: Array<FloatArray>, v: View): Float {
        // mean max(R-B,0)/max over interior; drives the assertion.
        val interior = interiorMask(v)
        var mr = 0f
        var n = 0
        for ((gx, gy) in interior) {
            val r = img[gy][gx * 3]
            val g = img[gy][gx * 3 + 1]
            val b = img[gy][gx * 3 + 2]
            val den = max(max(r, g), b)
            mr += max((r - b) / max(den, 1e-4f), 0f)
            n++
        }
        return if (n == 0) 0f else mr / n
    }

    @Test
    fun probeBoxedBlendImbalancedBases() {
        val s1 = 0.6f
        val s3 = 0.6f
        // Real-sensor-shaped neutral flat field: R phase lowest, B mid, G high.
        // Perfect WB neutralizes: wbR = G/R, wbB = G/B so a NON-dragged
        // per-phase mean reads back neutral after the full main() chain.
        val base = floatArrayOf(300f, 460f, 460f, 390f)
        val scene = buildImbalancedFlat(11, base, 12f)
        val wbR = base[1] / base[0]      // ~1.533
        val wbB = base[1] / base[3]      // ~1.179
        val runs = listOf(
            Probe("k=1.0 full   ", 96, 96, 96, 0, 0, floatArrayOf(0f, 0f, 96f, 96f)),
            Probe("k=2.0 out    ", 48, 96, 96, 0, 0, floatArrayOf(0f, 0f, 96f, 96f)),
            Probe("k=3.0 out    ", 32, 96, 96, 0, 0, floatArrayOf(0f, 0f, 96f, 96f)),
            Probe("k=3.2 boxed  ", 30, 96, 96, 0, 0, floatArrayOf(0f, 0f, 96f, 96f)),
            Probe("k=4.0 plain  ", 24, 96, 96, 0, 0, floatArrayOf(0f, 0f, 96f, 96f)),
            Probe("k=8.0 plain  ", 12, 96, 96, 0, 0, floatArrayOf(0f, 0f, 96f, 96f)),
            Probe("k=0.5 zoom-in", 96, 48, 48, 0, 0, floatArrayOf(24f, 24f, 48f, 48f))
        )
        println("== probe: boxedBlend cross-phase drag (imbalanced bases + WB in main) ==")
        println("grid  crop  k    mirror   packMag  inlineMag  (delta = pack drag effect)")
        val gains = floatArrayOf(wbR, 1f, 1f, wbB)
        for (r in runs) {
            val v = View(r.grid, r.grid, false, true, r.crop)
            val pack = buildPackGL(scene, v, s1, s3, gains)
            val img = renderPreviewFull(scene, pack, v, s1, s3, wbR, wbB)
            val pack0 = buildPackGL(scene, v, 0f, 0f)
            val img0 = renderPreviewFull(scene, pack0, v, 0f, 0f, wbR, wbB)
            val pm = magentaAvail(img, v)
            val im = magentaAvail(img0, v)
            // Inline baseline is the honest per-phase denoise; the pack must
            // NOT add a red excess over it anywhere in the boxed range.
            println("${r.grid}  ${r.cw}   %.2f  back   %.6f  %.6f  (delta=%.6f)".format(v.k, pm, im, pm - im))
            if (r.cw == 96) {
                assertTrue(
                    "pack added magenta over inline at grid=${r.grid} crop=${r.cw} (pack=$pm inline=$im)",
                    pm <= im + 0.001f
                )
            }
        }
    }
}
