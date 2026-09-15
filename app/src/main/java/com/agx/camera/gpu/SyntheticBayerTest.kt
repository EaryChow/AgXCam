package com.agx.camera.gpu

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Synthetic Bayer frame generator + CPU reference implementations of the
 * RAW-domain stages (DPC, green-guided GF, MAD σ̂) used to validate the GPU
 * pipeline on-device against a ground-truth oracle (plan T1-style anchor test).
 *
 * Geometry mirrors the live pipeline in PreviewRenderer exactly:
 *  - sensor 192×160, 10-bit, RGGB (phase0=R, phase1=G, phase2=G, phase3=B),
 *    uniform black level 64.
 *  - output grid = demosaic FBO (960×720 by default); one texel per ~1/5
 *    sensor column / 1/4.5 sensor row (the C3 cell mapping).
 *  - field: base 500 DN; vertical edge to 650 DN at x=40; a per-phase marker
 *    block x∈[80,128), y∈[64,112) (R=840, G1=220, G2=230, B=120) to detect
 *    phase swapping / texel misalignment; four isolated defects (2 hot, 1 hot,
 *    1 cold across phases); additive Gaussian σ=8 DN elsewhere.
 *
 * The CPU reference mirrors the GLSL float math including clamping, insertion
 * sorts, and the 2nd-largest/poison fallbacks, so deltas beyond tolerance
 * indicate a real GPU-side indexing/unit bug rather than rounding.
 */
object SyntheticBayerTest {

    const val SENSOR_W = 192
    const val SENSOR_H = 160
    const val BIT_DEPTH = 10
    const val BLACK_LEVEL = 64
    const val WHITE_LEVEL = 1023

    val BLACK = intArrayOf(BLACK_LEVEL, BLACK_LEVEL, BLACK_LEVEL, BLACK_LEVEL)
    val COLOR_MAP = intArrayOf(0, 1, 1, 2)

    // Same-color 5x5 lattice offsets (sensor pixels), mirror of GLSL NDX/NDY.
    val NDX = intArrayOf(2, -2, 0, 0, 2, -2, 2, -2)
    val NDY = intArrayOf(0, 0, 2, -2, 2, 2, -2, -2)

    const val SIG2_FACTOR = 2.1981f // (1.4826)^2

    // Sensor region markers used by the probe suite.
    const val MARKER_X0 = 80; const val MARKER_X1 = 128
    const val MARKER_Y0 = 64; const val MARKER_Y1 = 112

    // ------------------------------------------------------------------
    // Generator
    // ------------------------------------------------------------------

    fun phase(x: Int, y: Int): Int = abs(x % 2) + abs(y % 2) * 2

    fun inMarker(x: Int, y: Int): Boolean = x in MARKER_X0 until MARKER_X1 && y in MARKER_Y0 until MARKER_Y1

    fun buildSensor(seed: Long = 20260916L): ShortArray {
        val s = ShortArray(SENSOR_W * SENSOR_H)
        val rnd = java.util.Random(seed)
        for (y in 0 until SENSOR_H) {
            for (x in 0 until SENSOR_W) {
                var base = if (x >= 40) 650 else 500
                if (inMarker(x, y)) {
                    base = when (phase(x, y)) {
                        0 -> 840
                        1 -> 220
                        2 -> 230
                        else -> 120
                    }
                }
                var v = base + (if (inMarker(x, y)) 0.0 else rnd.nextGaussian() * 8.0)
                // isolated defects (phase, hot/cold)
                if (x == 60 && y == 60) v += 380.0       // phase0 hot
                else if (x == 62 && y == 61) v -= 300.0  // phase2 cold
                else if (x == 65 && y == 63) v += 420.0  // phase3 hot
                else if (x == 62 && y == 65) v += 360.0  // phase2 hot
                s[y * SENSOR_W + x] = v.roundToInt().coerceIn(0, WHITE_LEVEL).toShort()
            }
        }
        return s
    }

    fun buildBayerBuffer(sensor: ShortArray): ByteBuffer {
        val bb = ByteBuffer.allocateDirect(SENSOR_W * SENSOR_H * 2).order(ByteOrder.nativeOrder())
        bb.asShortBuffer().apply { put(sensor); flip() }
        bb.position(0)
        return bb
    }

    // ------------------------------------------------------------------
    // Geometry mirror (cell mapping identical to the GLSL)
    // ------------------------------------------------------------------

    /** raw value (masked to bit depth) at a sensor position, clamped. */
    fun rawVal(sensor: ShortArray, x: Int, y: Int): Int {
        val cx = x.coerceIn(0, SENSOR_W - 1)
        val cy = y.coerceIn(0, SENSOR_H - 1)
        return sensor[cy * SENSOR_W + cx].toInt() and 0x3FF
    }

    /** black-subtracted SIGNED value — mirror of GLSL bayerAt(). */
    fun sensorValue(sensor: ShortArray, x: Int, y: Int): Float {
        val cx = x.coerceIn(0, SENSOR_W - 1)
        val cy = y.coerceIn(0, SENSOR_H - 1)
        return rawVal(sensor, cx, cy).toFloat() - BLACK[phase(cx, cy)].toFloat()
    }

    fun cellOriginX(texelX: Int, viewW: Int): Int =
        ((texelX + 0.5f) / viewW.toFloat() * SENSOR_W.toFloat()).toInt()

    // Mirror of the GPU cellOrigin(): the pipeline's vertex shader applies the
    // preview transform — centre-crop the sensor to the output aspect
    // (scaleX/scaleY < 1, here scaleX=1, scaleY≈0.9 for 192x160 → 960x720) then
    // a Y flip — so texel row ty owns sensor row
    //     floor( SENSOR_H * (0.5*(1+scaleY) - scaleY*(ty+0.5)/viewH) )
    // clamped to [0, SENSOR_H-1]. Without this the oracle reads an unmoved,
    // unflipped, unscaled row set and every boundary/marker probe diverges.
    fun cellOriginY(texelY: Int, viewW: Int, viewH: Int): Int {
        val sourceAspect = SENSOR_W.toFloat() / SENSOR_H.toFloat()
        val viewAspect = viewW.toFloat() / viewH.toFloat()
        val scaleY = if (sourceAspect > viewAspect) 1f else sourceAspect / viewAspect
        val syF = SENSOR_H.toFloat() * (0.5f * (1f + scaleY) - scaleY * (texelY + 0.5f) / viewH.toFloat())
        return syF.toInt().coerceIn(0, SENSOR_H - 1)
    }

    /** output-grid texel covering the cell that contains the given sensor position. */
    fun cellTexelFor(sensorX: Int, sensorY: Int, viewW: Int, viewH: Int): Pair<Int, Int> {
        val cx = sensorX - abs(sensorX % 2)
        val cy = sensorY - abs(sensorY % 2)
        val ox = (cx.toFloat() * viewW / SENSOR_W.toFloat()).toInt()
        val oy = (cy.toFloat() * viewH / SENSOR_H.toFloat()).toInt()
        return ox.coerceIn(0, viewW - 1) to oy.coerceIn(0, viewH - 1)
    }

    /**
     * Output texel whose cellOrigin OWNS the given sensor cell under the GPU
     * ownership map (the preview-transformed origin). The plain cellTexelFor
     * map is NOT the ownership inverse on Y, so probes that must land on a
     * specific sensor cell (defects, marker edges) look up the true owner.
     */
    fun ownerTexelFor(sensorX: Int, sensorY: Int, viewW: Int, viewH: Int): Pair<Int, Int> {
        var tx = 0
        while (tx < viewW && cellOriginX(tx, viewW) != sensorX) tx++
        var ty = 0
        while (ty < viewH && cellOriginY(ty, viewW, viewH) != sensorY) ty++
        return tx.coerceIn(0, viewW - 1) to ty.coerceIn(0, viewH - 1)
    }

    // ------------------------------------------------------------------
    // Insertion sort mirror (exact GLSL comparison semantics)
    // ------------------------------------------------------------------

    fun insertionSort(v: FloatArray) {
        for (i in 1 until v.size) {
            val x = v[i]
            var j = i - 1
            while (j >= 0 && v[j] > x) {
                v[j + 1] = v[j]
                j--
            }
            v[j + 1] = x
        }
    }

    // ------------------------------------------------------------------
    // DPC passes (GLSL mirrors)
    // ------------------------------------------------------------------

    fun isoSigmaSq(signal: Float, isoA: Float, isoB: Float): Float =
        maxOf(isoA * signal.coerceAtLeast(0f) + isoB, 1.0e-6f)

    fun isoSigma(signal: Float, isoA: Float, isoB: Float): Float =
        kotlin.math.sqrt(isoSigmaSq(signal, isoA, isoB))

    /** Pass 0 (AVG): α-trimmed mean of the 8 same-colour neighbours per phase. */
    fun avgCell(sensor: ShortArray, texelX: Int, texelY: Int, viewW: Int, viewH: Int): FloatArray {
        val scx = cellOriginX(texelX, viewW)
        val scy = cellOriginY(texelY, viewW, viewH)
        val px = abs(scx % 2)
        val py = abs(scy % 2)
        val r = FloatArray(4)
        for (p in 0 until 4) {
            val ccx = scx + (px xor (p and 1))
            val ccy = scy + (py xor (p shr 1))
            val vals = FloatArray(8) { k -> sensorValue(sensor, ccx + NDX[k], ccy + NDY[k]) }
            insertionSort(vals)
            var sum = 0f
            for (k in 1 until 7) sum += vals[k]
            r[p] = sum / 6f
        }
        return r
    }

    fun avgAt(sensor: ShortArray, sensorX: Int, sensorY: Int, p: Int, viewW: Int, viewH: Int): Float {
        val (tx, ty) = cellTexelFor(sensorX, sensorY, viewW, viewH)
        return avgCell(sensor, tx, ty, viewW, viewH)[p]
    }

    /** Pass 1 (DETECT): per-phase defect flags mirror. */
    fun flagCell(
        sensor: ShortArray, texelX: Int, texelY: Int, viewW: Int, viewH: Int,
        m1: Float, m2: Float, theta: Float, isoA: Float, isoB: Float, enabled: Boolean
    ): FloatArray {
        val r = FloatArray(4)
        if (!enabled) return r
        val scx = cellOriginX(texelX, viewW)
        val scy = cellOriginY(texelY, viewW, viewH)
        val px = abs(scx % 2)
        val py = abs(scy % 2)
        for (p in 0 until 4) {
            val ccx = scx + (px xor (p and 1))
            val ccy = scy + (py xor (p shr 1))
            val i = sensorValue(sensor, ccx, ccy)
            val iavg = avgAt(sensor, ccx, ccy, p, viewW, viewH)
            val sigma = isoSigma(iavg, isoA, isoB)
            val band = maxOf(m1 * maxOf(iavg, 0f), theta * sigma)
            val devC = i - iavg
            if (abs(devC) > band) {
                var maxNb = 0f
                for (k in 0 until 8) {
                    val nx = ccx + NDX[k]; val ny = ccy + NDY[k]
                    val nbI = sensorValue(sensor, nx, ny)
                    val nbAvg = avgAt(sensor, nx, ny, p, viewW, viewH)
                    maxNb = maxOf(maxNb, abs(nbI - nbAvg))
                }
                if (abs(devC) > m2 * maxNb) {
                    r[p] = if (devC > 0f) 1f else -1f
                }
            }
        }
        return r
    }

    /** 2nd largest / 2nd smallest over clean candidates (GLSL fallback mirror). */
    private fun orderedCandidate(cands: FloatArray, hot: Boolean): Float {
        var lo1 = 1.0e30f; var lo2 = 1.0e30f
        var hi1 = -1.0e30f; var hi2 = -1.0e30f
        for (v in cands) {
            if (abs(v) > 1.0e8f) continue
            if (v < lo1) { lo2 = lo1; lo1 = v } else if (v < lo2) { lo2 = v }
            if (v > hi1) { hi2 = hi1; hi1 = v } else if (v > hi2) { hi2 = v }
        }
        return if (hot) hi2 else lo2
    }

    /** Pass 2 (CORRECT) and Pass 3 (COUPLET) mirror. */
    fun correctedCell(
        sensor: ShortArray, texelX: Int, texelY: Int, viewW: Int, viewH: Int,
        m1: Float, m2: Float, theta: Float, isoA: Float, isoB: Float,
        strength: Float, enabled: Boolean, couplet: Boolean
    ): FloatArray {
        val scx = cellOriginX(texelX, viewW)
        val scy = cellOriginY(texelY, viewW, viewH)
        val px = abs(scx % 2)
        val py = abs(scy % 2)
        val r = FloatArray(4)

        var anyNeighbourFlagged = false
        if (couplet) {
            outer@ for (p in 0 until 4) {
                val ccx = scx + (px xor (p and 1))
                val ccy = scy + (py xor (p shr 1))
                for (k in 0 until 8) {
                    if (flagAt(sensor, ccx + NDX[k], ccy + NDY[k], p, viewW, viewH,
                            m1, m2, theta, isoA, isoB) != 0f) {
                        anyNeighbourFlagged = true
                        break@outer
                    }
                }
            }
        }

        for (p in 0 until 4) {
            val ccx = scx + (px xor (p and 1))
            val ccy = scy + (py xor (p shr 1))
            val i = sensorValue(sensor, ccx, ccy)
            val iavg = avgAt(sensor, ccx, ccy, p, viewW, viewH)
            val sigma = isoSigma(iavg, isoA, isoB)
            val band = maxOf(m1 * maxOf(iavg, 0f), theta * sigma)

            val cands = FloatArray(8) { k -> sensorValue(sensor, ccx + NDX[k], ccy + NDY[k]) }
            if (couplet) {
                for (k in 0 until 8) {
                    if (flagAt(sensor, ccx + NDX[k], ccy + NDY[k], p, viewW, viewH,
                            m1, m2, theta, isoA, isoB) != 0f) cands[k] = -1.0e9f
                }
            }

            var outv = maxOf(i, 0f)
            val needPassThrough = !enabled || (couplet && !anyNeighbourFlagged)
            if (!needPassThrough) {
                val devC = i - iavg
                val hot = devC > 0f && codeCheck(sensor, ccx, ccy, p, viewW, viewH, devC, m2, m1, theta, isoA, isoB)
                val cold = devC < 0f && codeCheck(sensor, ccx, ccy, p, viewW, viewH, -devC, m2, m1, theta, isoA, isoB)
                if (hot || cold) {
                    val e = cands[0]; val w = cands[1]; val n = cands[2]; val s = cands[3]
                    val ne = cands[4]; val sw = cands[5]; val se = cands[6]; val nw = cands[7]
                    val aH = (e + w) * 0.5f
                    val aV = (n + s) * 0.5f
                    val a45 = (ne + sw) * 0.5f
                    val a135 = (se + nw) * 0.5f
                    val dH = abs(e - w); val dV = abs(n - s)
                    val d45 = abs(ne - sw); val d135 = abs(se - nw)
                    var id = aH
                    if (dV < dH && dV <= d45 && dV <= d135) id = aV
                    else if (d45 < dH && d45 <= dV && d45 <= d135) id = a45
                    else if (d135 < dH && d135 <= dV && d135 <= d45) id = a135

                    if (couplet) {
                        val poisoned = abs(e) > 1.0e8f || abs(w) > 1.0e8f || abs(n) > 1.0e8f ||
                            abs(s) > 1.0e8f || abs(ne) > 1.0e8f || abs(sw) > 1.0e8f ||
                            abs(se) > 1.0e8f || abs(nw) > 1.0e8f
                        if (poisoned) {
                            var cnt = 0
                            for (v in cands) if (abs(v) <= 1.0e8f) cnt++
                            if (cnt >= 6) {
                                outv = maxOf(orderedCandidate(cands, hot), 0f)
                                r[p] = outv
                                continue
                            }
                        }
                    }

                    val m3 = m1
                    if (abs(id - iavg) <= maxOf(m3 * maxOf(iavg, 0f), theta * sigma)) {
                        outv = maxOf(id, 0f)
                    } else {
                        outv = maxOf(orderedCandidate(cands, hot), 0f)
                    }
                }
            }
            r[p] = maxOf(i, 0f) + (outv - maxOf(i, 0f)) * strength
        }
        return r
    }

    private fun codeCheck(
        sensor: ShortArray, ccx: Int, ccy: Int, p: Int, viewW: Int, viewH: Int,
        devMag: Float, m2: Float, m1: Float, theta: Float, isoA: Float, isoB: Float
    ): Boolean {
        var maxNb = 0f
        for (k in 0 until 8) {
            val nx = ccx + NDX[k]; val ny = ccy + NDY[k]
            val nbI = sensorValue(sensor, nx, ny)
            val nbAvg = avgAt(sensor, nx, ny, p, viewW, viewH)
            maxNb = maxOf(maxNb, abs(nbI - nbAvg))
        }
        return devMag > m2 * maxNb
    }

    private fun flagAt(
        sensor: ShortArray, sensorX: Int, sensorY: Int, p: Int, viewW: Int, viewH: Int,
        m1: Float, m2: Float, theta: Float, isoA: Float, isoB: Float
    ): Float {
        val (tx, ty) = cellTexelFor(sensorX, sensorY, viewW, viewH)
        return flagCell(sensor, tx, ty, viewW, viewH, m1, m2, theta, isoA, isoB, true)[p]
    }

    /** Raw pack mirror (DPC pass-through): max(I, 0) per phase. */
    fun rawPackCell(sensor: ShortArray, texelX: Int, texelY: Int, viewW: Int, viewH: Int): FloatArray {
        val scx = cellOriginX(texelX, viewW)
        val scy = cellOriginY(texelY, viewW, viewH)
        val px = abs(scx % 2)
        val py = abs(scy % 2)
        val r = FloatArray(4)
        for (p in 0 until 4) {
            val ccx = scx + (px xor (p and 1))
            val ccy = scy + (py xor (p shr 1))
            r[p] = maxOf(sensorValue(sensor, ccx, ccy), 0f)
        }
        return r
    }

    // ------------------------------------------------------------------
    // Stage 3 (green-guided guided filter) mirror
    // ------------------------------------------------------------------

    fun guidedCell(
        gridAt: (Int, Int) -> FloatArray,
        texelX: Int, texelY: Int, viewW: Int, viewH: Int,
        alpha: Float, eps: Float
    ): FloatArray {
        val center = gridAt(texelX, texelY)
        val g0 = (center[1] + center[2]) * 0.5f
        val r = FloatArray(4)
        for (p in 0 until 4) {
            var meanI = 0f; var meanG = 0f
            var sumII = 0f; var sumGG = 0f; var sumIG = 0f
            for (k in 0 until 25) {
                val tx = (texelX + (k % 5) - 2).coerceIn(0, viewW - 1)
                val ty = (texelY + (k / 5) - 2).coerceIn(0, viewH - 1)
                val c = gridAt(tx, ty)
                val i = c[p]
                val g = (c[1] + c[2]) * 0.5f
                meanI += i; meanG += g
                sumII += i * i; sumGG += g * g; sumIG += i * g
            }
            val invN = 1f / 25f
            meanI *= invN; meanG *= invN
            val varG = maxOf(sumGG * invN - meanG * meanG, 0f)
            val covIG = sumIG * invN - meanI * meanG
            val a = covIG / (varG + eps)
            val b = meanI - a * meanG
            val gfOut = a * g0 + b
            r[p] = maxOf(center[p] + (gfOut - center[p]) * (1f - alpha), 0f)
        }
        return r
    }

    // ------------------------------------------------------------------
    // Stage 2 (MAD σ̂) mirror
    // ------------------------------------------------------------------

    fun sigmaCell(
        gridAt: (Int, Int) -> FloatArray,
        texelX: Int, texelY: Int, viewW: Int, viewH: Int,
        isoA: Float, isoB: Float
    ): FloatArray {
        val nox = intArrayOf(1, -1, 0, 0, 1, -1, 1, -1)
        val noy = intArrayOf(0, 0, 1, -1, 1, 1, -1, -1)
        var sig2Total = 0f
        val sig2ByPhase = FloatArray(4)
        for (p in 0 until 4) {
            val vals = FloatArray(8)
            for (k in 0 until 8) {
                val tx = (texelX + nox[k]).coerceIn(0, viewW - 1)
                val ty = (texelY + noy[k]).coerceIn(0, viewH - 1)
                vals[k] = gridAt(tx, ty)[p]
            }
            insertionSort(vals)
            val med = (vals[3] + vals[4]) * 0.5f
            val devs = FloatArray(8) { k -> abs(vals[k] - med) }
            insertionSort(devs)
            val mad = (devs[3] + devs[4]) * 0.5f
            val sig2 = SIG2_FACTOR * mad * mad
            sig2ByPhase[p] = sig2
            sig2Total += sig2
        }
        val sig2Mean = sig2Total * 0.25f
        val center = gridAt(texelX, texelY)
        val meanSignal = (center[0] + center[1] + center[2] + center[3]) * 0.25f
        val floor2 = isoSigmaSq(meanSignal, isoA, isoB)
        val out2 = maxOf(sig2Mean, floor2)
        return floatArrayOf(out2, kotlin.math.sqrt(maxOf(out2, 1.0e-6f)), sig2ByPhase[0], sig2ByPhase[3])
    }

    // ------------------------------------------------------------------
    // Probe orchestration
    // ------------------------------------------------------------------

    data class SynthParams(
        val dpEnabled: Boolean,
        val m1: Float, val m2: Float, val theta: Float,
        val isoA: Float, val isoB: Float,
        val corrStrength: Float,
        val s3Active: Boolean, val alpha: Float, val eps: Float,
        val viewW: Int, val viewH: Int
    )

    /** Sensor-coordinate probe points exercising every test region. */
    fun probeSensorPoints(): List<Pair<Int, Int>> = listOf(
        MARKER_X0 to MARKER_Y0,      // marker cell origin (phase0 R=840)
        MARKER_X0 + 12 to MARKER_Y0 + 4, // interior marker
        78 to 64,                     // just LEFT of marker (edge of marker boundary)
        60 to 60,                     // phase0 hot
        62 to 61,                     // phase2 cold
        65 to 63,                     // phase3 hot
        62 to 65,                     // phase2 hot
        30 to 30,                     // flat region
        20 to 120,                    // flat region
        39 to 80                      // on the 500/650 vertical edge
    )

    /** Builds lazily-cached CPU references for a probe sensor point. */
    class CpuRefs(val sensor: ShortArray, val params: SynthParams) {
        var refs: MutableList<ProbeRef>? = null

        fun compute(): List<ProbeRef> {
            refs?.let { return it }
            val out = ArrayList<ProbeRef>()
            val pw = params.viewW; val ph = params.viewH
            for ((sx, sy) in probeSensorPoints()) {
                val (tx, ty) = ownerTexelFor(sx, sy, pw, ph)
                val avg = avgCell(sensor, tx, ty, pw, ph)
                val flags = flagCell(sensor, tx, ty, pw, ph,
                    params.m1, params.m2, params.theta, params.isoA, params.isoB, true)
                val grid = if (params.dpEnabled) correctedCell(
                    sensor, tx, ty, pw, ph,
                    params.m1, params.m2, params.theta, params.isoA, params.isoB,
                    params.corrStrength, true, true
                ) else rawPackCell(sensor, tx, ty, pw, ph)
                val gridAt: (Int, Int) -> FloatArray = { gx, gy ->
                    if (params.dpEnabled) correctedCell(
                        sensor, gx, gy, pw, ph,
                        params.m1, params.m2, params.theta, params.isoA, params.isoB,
                        params.corrStrength, true, true
                    ) else rawPackCell(sensor, gx, gy, pw, ph)
                }
                // The GPU σ̂ stage consumes finalSparseTex (S3 output when active),
                // so the oracle's σ̂ input must mirror the same chain.
                val s3At: (Int, Int) -> FloatArray = { gx, gy ->
                    if (params.s3Active) guidedCell(gridAt, gx, gy, pw, ph, params.alpha, params.eps)
                    else gridAt(gx, gy)
                }
                val s3 = s3At(tx, ty)
                val sigma = sigmaCell(s3At, tx, ty, pw, ph, params.isoA, params.isoB)
                out.add(
                    ProbeRef(sx, sy, tx, ty,
                        avg, flags, grid,
                        if (params.s3Active) s3 else grid, sigma)
                )
            }
            refs = out
            return out
        }
    }

    data class ProbeRef(
        val sx: Int, val sy: Int,
        val tx: Int, val ty: Int,
        val avg: FloatArray,
        val flags: FloatArray,
        val grid: FloatArray,     // after DPC couplet (or raw pack)
        val s3Grid: FloatArray,   // after Stage 3 (or grid if inactive)
        val sigma: FloatArray
    )
}