package com.agx.camera.gpu

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
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

    private fun phase(x: Int, y: Int): Int = abs(x % 2) + abs(y % 2) * 2

    private fun sensorVal(v: ShortArray, x: Int, y: Int): Float {
        val cx = x.coerceIn(0, SW - 1)
        val cy = y.coerceIn(0, SH - 1)
        val raw = v[cy * SW + cx].toInt() and 0x3FF
        return max((raw - SENSOR_BLACK).toFloat(), 0f)
    }

    /** Literal GL sampleSameColorNR (12-tap a-trim + DPC guard + S3 clip blend).
     *  At nrRadius<4 mirrors the preview-only 4-tap (step-2 axis neighbours)
     *  used when the demosaic box stays 4x4; capture/box<4 keep the 12-tap. */
    private fun sameColorNR(v: ShortArray, sx: Int, sy: Int, s1: Float, s3: Float, nrRadius: Int = 4): Float {
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
        var center = c
        if (max(s1, 0.85f * s3) > 0f) {
            val hot = (c > mx) && (c - iavg) > band
            val cold = (c < mn) && (iavg - c) > band
            if (hot || cold) center = c + (iavg - c) * max(s1, 0.85f * s3)
        }
        return if (c < CLIP_SCALAR) center + (iavg - center) * (0.98f * s3) else center
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
    // 2 (strong s3 >= 0.6, full 12-tap ring).  The pack regime (k<=2) uses its
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

    private fun buildPackGL(scene: ShortArray, v: View, s1: Float, s3: Float, gains: FloatArray? = null): FloatArray {
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
                for (p in 0..3) {
                    val phaseX = p and 1
                    val phaseY = p shr 1
                    out[base + p] = sameColorNR(
                        scene,
                        (scx + (px xor phaseX)).coerceIn(0, SW - 1),
                        (scy + (py xor phaseY)).coerceIn(0, SH - 1),
                        s1, s3
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
            var tot = 0f
            var tn = 0
            for (p in 0..3) if (pN[p] > 0) { tot += pSum[p] / pN[p]; tn++ }
            if (tn == 0) { tot = 0f; tn = 1 }
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
                if (hot || cold) center = c + (iavg - c) * corr
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
        nrRadius: Int = 4
    ): Float {
        val cx = cx0.coerceIn(0, SW - 1)
        val cy = cy0.coerceIn(0, SH - 1)
        if (packActive(v)) {
            return mosaicRead(pack, v, cx, cy, shiftX, shiftY)
        }
        return sameColorNR(scene, cx, cy, s1, s3, nrRadius)
    }

    /** demosaicAt: the standard 9-tap reconstruction (box-AA 4x4 host path). */
    private fun demosaicAtRGB(
        scene: ShortArray, pack: FloatArray, v: View,
        s1: Float, s3: Float, cellX: Int, cellY: Int,
        shiftX: Int = 0, shiftY: Int = 0,
        nrRadius: Int = 4
    ): FloatArray {
        val center = demosaicSample(scene, pack, v, s1, s3, cellX, cellY, shiftX, shiftY, nrRadius)
        val nN = demosaicSample(scene, pack, v, s1, s3, cellX, cellY - 1, shiftX, shiftY, nrRadius)
        val nS = demosaicSample(scene, pack, v, s1, s3, cellX, cellY + 1, shiftX, shiftY, nrRadius)
        val nW = demosaicSample(scene, pack, v, s1, s3, cellX - 1, cellY, shiftX, shiftY, nrRadius)
        val nE = demosaicSample(scene, pack, v, s1, s3, cellX + 1, cellY, shiftX, shiftY, nrRadius)
        val nNW = demosaicSample(scene, pack, v, s1, s3, cellX - 1, cellY - 1, shiftX, shiftY, nrRadius)
        val nNE = demosaicSample(scene, pack, v, s1, s3, cellX + 1, cellY - 1, shiftX, shiftY, nrRadius)
        val nSW = demosaicSample(scene, pack, v, s1, s3, cellX - 1, cellY + 1, shiftX, shiftY, nrRadius)
        val nSE = demosaicSample(scene, pack, v, s1, s3, cellX + 1, cellY + 1, shiftX, shiftY, nrRadius)
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

    /** demosaicBilinear + main(): render the full output image (all paths). */
    private fun renderPreview(
        scene: ShortArray, pack: FloatArray, v: View, s1: Float, s3: Float,
        shiftX: Int = 0, shiftY: Int = 0,
        boxAA: Int = previewBoxAA(v, s1, s3),
        nrRadius: Int = previewNrRadius(boxAA)
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
            var sum = floatArrayOf(0f, 0f, 0f)
            for (dy in 0 until boxAA) for (dx in 0 until boxAA) {
                val c = demosaicAtRGB(
                    scene, pack, v, s1, s3,
                    (baseX + dx).coerceIn(0, SW - 1),
                    (baseY + dy).coerceIn(0, SH - 1),
                    shiftX, shiftY,
                    nrRadius = nrRadius
                )
                sum[0] += c[0]; sum[1] += c[1]; sum[2] += c[2]
            }
            val div = 1f / (boxAA * boxAA)
            sum[0] *= div; sum[1] *= div; sum[2] *= div
            val fin = composeMain(sum)
            out[gy][gx * 3] = fin[0]; out[gy][gx * 3 + 1] = fin[1]; out[gy][gx * 3 + 2] = fin[2]
        }
        return out
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
        nrRadius: Int = previewNrRadius(boxAA)
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
            var sum = floatArrayOf(0f, 0f, 0f)
            for (dy in 0 until boxAA) for (dx in 0 until boxAA) {
                val c = demosaicAtRGB(
                    scene, pack, v, s1, s3,
                    (baseX + dx).coerceIn(0, SW - 1),
                    (baseY + dy).coerceIn(0, SH - 1),
                    shiftX, shiftY,
                    nrRadius = nrRadius
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
     * with the reduced boxAA active (the k>2 inline regime, box 2x2 while
     * the sliders are up), each output band's sigma must stay AT OR BELOW the
     * zero-slider baseline (boxAA=4).  Any single band above baseline rejects
     * the boxAA step-down — the demosaic's effective averaging must never get
     * weaker than baseline at any slider strength.
     */
    @Test
    fun inlineReducedBoxKeepsBandsBounded() {
        val scene = buildFlatNoise(11, 460f, 28f)
        // k>2 regimes where the inline path is live and the box reduction acts.
        val configs = listOf(
            Config(24, 96, 96, false),    // k=4  back
            Config(24, 96, 96, true),     // k=4  front
            Config(12, 96, 96, false),    // k=8  back
            Config(12, 96, 96, true)      // k=8  front
        )
        // Slider grid covering every boxAA branch and the s3=0.7 edge (the only
        // remaining threshold), plus the S1-only s3=0 branch, crossed with S1.
        val s3s = listOf(0f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f)
        val s1s = listOf(0.0f, 0.3f, 0.6f)
        val strengths = mutableListOf<FloatArray>()
        for (s3 in s3s) for (s1 in s1s) strengths.add(floatArrayOf(s1, s3))
        val failures = StringBuilder()
        println("== inline reduced-box per-band sigma vs zero-slider baseline ==")
        for (cfg in configs) {
            val v = View(cfg.grid, cfg.grid, cfg.mirror, true, floatArrayOf(0f, 0f, 96f, 96f))
            val back = if (cfg.mirror) "mirrorX" else "back  "
            val pack0 = buildPackGL(scene, v, 0f, 0f)
            val img0 = renderPreview(scene, pack0, v, 0f, 0f)   // baseline: box 4
            val sd0 = FloatArray(3) { bandSD(img0, v, it) }
            println("k=%.2f %s  baseline sigma R=%.3f G=%.3f B=%.3f".format(v.k, back, sd0[0], sd0[1], sd0[2]))
            for (s in strengths) {
                val pack = buildPackGL(scene, v, s[0], s[1])
                val img = renderPreview(scene, pack, v, s[0], s[1])
                val box = previewBoxAA(v, s[0], s[1])
                val sd = FloatArray(3) { bandSD(img, v, it) }
                println(
                    "  s1=%.1f s3=%.1f box=%d  R=%.3f G=%.3f B=%.3f".format(
                        s[0], s[1], box, sd[0], sd[1], sd[2]
                    )
                )
                val name = charArrayOf('R', 'G', 'B')
                for (ch in 0..2) {
                    if (sd[ch] > sd0[ch] + 0.001f) {
                        val msg = "band %s sigma %.3f > baseline %.3f at grid=%d mirror=%s s1=%.1f s3=%.1f box=%d\n"
                            .format(name[ch], sd[ch], sd0[ch], cfg.grid, cfg.mirror, s[0], s[1], box)
                        failures.append(msg)
                        println("  !! " + msg.trim())
                    }
                }
            }
        }
        assertTrue(
            "boxAA step-down broke the per-band sigma gate:\n" + failures,
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
                println("-- variant=$variant")
                for (s1 in s1s) {
                    var prev = sd0.copyOf()
                    for (s3 in s3s) {
                        if (s1 <= 0f && s3 <= 0f) continue
                        val pack = buildPackGL(scene, v, s1, s3)
                        val box = when (variant) {
                            "box4" -> 4
                            else -> previewBoxAA(v, s1, s3)
                        }
                        val nrR = when (variant) {
                            "box4" -> 2
                            else -> previewNrRadius(box)
                        }
                        val img = renderPreview(scene, pack, v, s1, s3, boxAA = box, nrRadius = nrR)
                        val sd = FloatArray(3) { bandSD(img, v, it) }
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
                        println("  s1=%.2f s3=%.2f box=%d nr=%d  R=%.4f G=%.4f B=%.4f%s".format(
                            s1, s3, box, nrR, sd[0], sd[1], sd[2], if (anyUp) "  <-- POP" else ""))
                        prev = sd
                    }
                }
                println("variant=$variant pops: " + (if (pops.isEmpty()) "none" else pops.toString().trim()))
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
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("shipped"), mirror = false, gateTol = 0.0004f)
        runMonotoneWalk(scene, 0.001f, s1s, s3s, listOf("shipped"), mirror = true, gateTol = 0.0004f)
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