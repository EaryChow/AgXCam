package com.agx.camera.camera

import java.nio.ByteBuffer

enum class LensShadingState {
    IDLE, SAMPLING, PAUSED, CONVERGED
}

/**
 * Monte-Carlo lens-shading (vignette) estimator.
 *
 * The Xiaomi HAL on device reports SENSOR_INFO_LENS_SHADING_APPLIED=true yet
 * never emits a STATISTICS_LENS_SHADING_MAP, so there is no per-lens map to
 * consume. This estimator reconstructs one by
 * sampling live 16-bit Bayer frames:
 *
 *   - Accumulate CFA-aware green-phase samples (the least noisy channel and
 *     the least affected by color casts) into a coarse grid. Each sample is
 *     normalized by the current frame's green mean so exposure ramps between
 *     frames do not bias the accumulators.
 *   - Each frame's sample lattice is sub-block jittered, so consecutive frames
 *     do not sample the same pixels: consecutive frames of one static scene are
 *     highly correlated, and a short sampling window would therefore act like
 *     a deterministic "out of one frame" sample instead of a Monte-Carlo
 *     estimate. The jitter sweeps the whole stride window over a sample cycle
 *     and decorrelates the estimate across time.
 *   - A robust per-cell gain is built from the center reference, then radial
 *     bins in (r/R)^2 are reduced with a median and fitted with a low-order
 *     polynomial (constant + linear + quadratic in the squared radius). The
 *     smooth fit suppresses scene outliers a per-cell gain would chase.
 *   - The fitted grid is clipped to [1, 3] and emitted as a per-lens
 *     [LensShadingData] with all four channels equal (achromatic map); the
 *     caller feeds it through the identical [LensShadingData.toRgba16fFlipped]
 *     path used for HAL maps, so no renderer changes are needed.
 *   - Sampling runs continuously for a long, scene-averaging window. It only
 *     converges once the grid has stopped changing *and* the sampling has run
 *     for at least [minConvergeMillis] (so short, correlated scenes are not
 *     locked in), and it is forced to a final map after [maxConvergeMillis].
 *
 * The histogram is intentionally kept coarse: ~33x25 cells averaged over many
 * frames at [sampleStride] tolerates scenes that are not perfectly flat.
 */
class LensShadingEstimator(
    private val gridWidth: Int = 33,
    private val gridHeight: Int = 25,
    private val sampleStride: Int = 24,
    private val binCount: Int = 16,
    private val centerUTop: Float = 0.06f,
    private val minFramesBeforeConverge: Int = 150,
    private val convergeDelta: Float = 0.01f,
    private val convergeStreakRequired: Int = 5,
    private val minConvergeMillis: Long = 30_000L,
    private val maxConvergeMillis: Long = 120_000L,
    private val gainMax: Float = 3f,
    private val clock: () -> Long = { System.currentTimeMillis() }
) {
    private val lock = Any()

    private var state = LensShadingState.IDLE
    private var sensorWidth = 0
    private var sensorHeight = 0
    private var bayer = BayerPattern.RGGB
    private var whiteLevel = 1023
    private var blackLevel = 64f

    private var cellSum = DoubleArray(0)
    private var cellCount = LongArray(0)
    private var cellU = FloatArray(0)
    private var isGreenPhase = BooleanArray(4)

    private var framesSampled = 0
    private var activeMillis = 0L
    private var lastActiveStart = 0L
    private var lastGrid: FloatArray? = null
    private var lastMap: LensShadingData? = null
    private var lastMapDelta = Float.MAX_VALUE
    private var convergeStreak = 0

    val currentState: LensShadingState get() = synchronized(lock) { state }
    val sampledFrames: Int get() = synchronized(lock) { framesSampled }

    /** Accumulated sampling time (pause time excluded), ms. */
    val elapsedMillis: Long get() = synchronized(lock) {
        if (state == LensShadingState.SAMPLING) {
            activeMillis + (clock() - lastActiveStart).coerceAtLeast(0L)
        } else {
            activeMillis
        }
    }

    /** Max absolute cell gain change vs the previous rebuild; MAX_VALUE until
     *  the first map has been built. */
    val mapDelta: Float get() = synchronized(lock) { lastMapDelta }

    fun start(width: Int, height: Int, bayer: BayerPattern, whiteLevel: Int, blackLevel: Float) {
        synchronized(lock) {
            sensorWidth = width
            sensorHeight = height
            this.bayer = bayer
            this.whiteLevel = whiteLevel
            this.blackLevel = blackLevel
            this.isGreenPhase = BooleanArray(4) { bayer.colorMap()[it] == 1 }

            val n = gridWidth * gridHeight
            cellSum = DoubleArray(n)
            cellCount = LongArray(n)
            cellU = FloatArray(n) { i -> radiusSquaredOfCell(i % gridWidth, i / gridWidth) }

            framesSampled = 0
            activeMillis = 0L
            lastActiveStart = clock()
            lastGrid = null
            lastMap = null
            lastMapDelta = Float.MAX_VALUE
            convergeStreak = 0
            state = LensShadingState.SAMPLING
        }
    }

    fun pause() {
        synchronized(lock) {
            if (state == LensShadingState.SAMPLING) {
                activeMillis += (clock() - lastActiveStart).coerceAtLeast(0L)
                state = LensShadingState.PAUSED
            }
        }
    }

    fun resume() {
        synchronized(lock) {
            if (state == LensShadingState.PAUSED || state == LensShadingState.CONVERGED) {
                lastActiveStart = clock()
                state = LensShadingState.SAMPLING
            }
        }
    }

    fun reset() {
        synchronized(lock) {
            cellSum = DoubleArray(0)
            cellCount = LongArray(0)
            cellU = FloatArray(0)
            framesSampled = 0
            activeMillis = 0L
            lastGrid = null
            lastMap = null
            lastMapDelta = Float.MAX_VALUE
            convergeStreak = 0
            state = LensShadingState.IDLE
        }
    }

    /** Accumulate one 16-bit raw frame (little-endian, contiguous). Returns the
     *  freshly rebuilt map when the frame contributed data, otherwise null. */
    fun addFrame(buffer: ByteBuffer): LensShadingData? {
        synchronized(lock) {
            if (state != LensShadingState.SAMPLING) return null
            if (sensorWidth <= 0 || sensorHeight <= 0 || cellCount.size == 0) return null
            if (!accumulateCells(buffer)) return null
            framesSampled++
            val now = clock()
            activeMillis += (now - lastActiveStart).coerceAtLeast(0L)
            lastActiveStart = now
            val map = rebuildMap()
            if (map != null) lastMap = map
            // Hard cap: a scene that never settles still yields a usable map.
            if (maxConvergeMillis > 0L && activeMillis >= maxConvergeMillis) {
                state = LensShadingState.CONVERGED
            }
            return lastMap
        }
    }

    /** The last rebuilt map, without accumulating a new frame. */
    fun buildMap(): LensShadingData? = synchronized(lock) { lastMap }

    /** Two passes over the sparse green grid: frame green mean first, then
     *  exposure-normalized accumulation into cells. The lattice origin is
     *  jittered per frame inside a stride window so consecutive frames do not
     *  sample identical pixels (Monte-Carlo decorrelation across time). Each
     *  block still samples a 2x2 neighborhood so every CFA phase is hit
     *  regardless of the stride's parity. Returns false when the frame carried
     *  no usable signal. */
    private fun accumulateCells(buffer: ByteBuffer): Boolean {
        val w = sensorWidth
        val h = sensorHeight
        val step = sampleStride
        val jx = (framesSampled * 11 + 5) % 8
        val jy = (framesSampled * 17 + 7) % 8

        var meanSum = 0.0
        var meanCount = 0L
        var r = jy
        while (r < h) {
            var c = jx
            while (c < w) {
                var dr = 0
                while (dr < 2 && r + dr < h) {
                    var dc = 0
                    while (dc < 2 && c + dc < w) {
                        val pr = r + dr
                        val pc = c + dc
                        if (isGreenPhase[(pc and 1) + (pr and 1) * 2]) {
                            val sig = greenSignal(buffer, w, pr, pc)
                            if (sig > 0.0) {
                                meanSum += sig
                                meanCount++
                            }
                        }
                        dc++
                    }
                    dr++
                }
                c += step
            }
            r += step
        }
        if (meanCount < 200L) return false
        val invMean = 1.0 / (meanSum / meanCount)

        val sums = cellSum
        val counts = cellCount
        val gw = gridWidth
        val gh = gridHeight
        r = jy
        while (r < h) {
            var c = jx
            while (c < w) {
                var dr = 0
                while (dr < 2 && r + dr < h) {
                    var dc = 0
                    while (dc < 2 && c + dc < w) {
                        val pr = r + dr
                        val pc = c + dc
                        if (isGreenPhase[(pc and 1) + (pr and 1) * 2]) {
                            val sig = greenSignal(buffer, w, pr, pc)
                            if (sig > 0.0) {
                                val cell = (pc * gw / w) + (pr * gh / h) * gw
                                sums[cell] += sig * invMean
                                counts[cell]++
                            }
                        }
                        dc++
                    }
                    dr++
                }
                c += step
            }
            r += step
        }
        return true
    }

    private fun greenSignal(buffer: ByteBuffer, w: Int, r: Int, c: Int): Double {
        val idx = (r * w + c) * 2
        val v = (buffer.get(idx).toInt() and 0xFF) or
            ((buffer.get(idx + 1).toInt() and 0xFF) shl 8)
        // Reject clipped and near-black samples: neither carries usable
        // brightness structure, and dark noise would poison a corner bin.
        if (v >= (whiteLevel * 0.90).toInt()) return 0.0
        val sig = v - blackLevel
        return if (sig > 4.0) sig.toDouble() else 0.0
    }

    /** Fit + rebuild the gain grid from the accumulated cells. Convergence is
     *  gated on both a streak of stable rebuilds and minimum sampling time, so
     *  a short burst of one correlated scene cannot be locked in. */
    private fun rebuildMap(): LensShadingData? {
        val n = gridWidth * gridHeight
        val cellSun = cellSum
        val cellCnt = cellCount

        // Center reference: count-weighted mean over the inner region.
        var centerSum = 0.0
        var centerCnt = 0L
        for (i in 0 until n) {
            if (cellCnt[i] > 0 && cellU[i] < centerUTop) {
                centerSum += cellSun[i]
                centerCnt += cellCnt[i]
            }
        }
        if (centerCnt < 32L) return null
        val centerMean = centerSum / centerCnt
        if (centerMean <= 0.0) return null

        // Per-cell gains, clipped to >= 1 (correction only lifts shadows).
        val gains = FloatArray(n)
        for (i in 0 until n) {
            if (cellCnt[i] <= 0) {
                gains[i] = 1f
                continue
            }
            val g = (centerMean / (cellSun[i] / cellCnt[i])).toFloat()
            gains[i] = g.coerceIn(1f, gainMax)
        }

        // Radial bins in (r/R)^2, reduced by median for robustness.
        val binLists = Array(binCount) { mutableListOf<Float>() }
        for (i in 0 until n) {
            if (cellCnt[i] == 0L) continue
            var bin = (cellU[i] * binCount).toInt()
            bin = bin.coerceIn(0, binCount - 1)
            binLists[bin].add(gains[i])
        }
        val points = ArrayList<Pair<Float, Float>>(binCount)
        for (b in 0 until binCount) {
            if (binLists[b].isEmpty()) continue
            val median = median(binLists[b].toFloatArray())
            if (!median.isFinite() || median <= 0f) continue
            val u = ((b + 0.5f) / binCount).coerceIn(0f, 1f)
            points.add(u to median.coerceAtLeast(1f))
        }

        // 2nd-order fit in u = (r/R)^2: g(u) = c0 + c1*u + c2*u^2.
        val coeff = fitQuadratic(points) ?: return null

        val newGrid = FloatArray(n)
        for (i in 0 until n) {
            val v = coeff[0] + coeff[1] * cellU[i] + coeff[2] * cellU[i] * cellU[i]
            newGrid[i] = v.coerceIn(1f, gainMax)
        }

        // Convergence: the grid's max change across rebuilds, plus a streak.
        val prev = lastGrid
        lastMapDelta = if (prev == null) {
            Float.MAX_VALUE
        } else {
            var maxD = 0f
            for (i in 0 until n) {
                val d = kotlin.math.abs(newGrid[i] - prev[i])
                if (d > maxD) maxD = d
            }
            maxD
        }
        lastGrid = newGrid.copyOf()

        val timeOk = minConvergeMillis <= 0L || activeMillis >= minConvergeMillis
        if (lastMapDelta < convergeDelta) {
            convergeStreak++
            if (timeOk && convergeStreak >= convergeStreakRequired && framesSampled >= minFramesBeforeConverge) {
                state = LensShadingState.CONVERGED
            }
        } else {
            convergeStreak = 0
        }

        return lensShadingFromGrid(newGrid)
    }

    private fun lensShadingFromGrid(g: FloatArray): LensShadingData {
        val rows = Array(gridHeight) { r ->
            FloatArray(gridWidth) { c -> g[r * gridWidth + c] }
        }
        // Achromatic correction: every raw phase receives the same gain grid.
        val maxGain = g.maxOrNull() ?: 1f
        return LensShadingData(
            rGains = rows.map { it.copyOf() }.toTypedArray(),
            grGains = rows.map { it.copyOf() }.toTypedArray(),
            gbGains = rows.map { it.copyOf() }.toTypedArray(),
            bGains = rows.map { it.copyOf() }.toTypedArray(),
            width = gridWidth,
            height = gridHeight,
            available = maxGain > 1.02f
        )
    }

    private fun radiusSquaredOfCell(col: Int, row: Int): Float {
        if (sensorWidth <= 0 || sensorHeight <= 0) return 0f
        val cx = (col + 0.5f) / gridWidth - 0.5f
        val cy = (row + 0.5f) / gridHeight - 0.5f
        // Corner-normalized: cx,cy in [-0.5,0.5], so (cx^2+cy^2)*2 is 1.0 at
        // the corners and 0.0 at the optical center.
        return (cx * cx + cy * cy) * 2f
    }

    private fun median(values: FloatArray): Float {
        val v = values.copyOf()
        v.sort()
        val mid = v.size / 2
        return if (v.size % 2 == 1) v[mid] else (v[mid - 1] + v[mid]) * 0.5f
    }

    /** Least-squares 2nd-order polynomial through the radial-bin points via
     *  Gaussian elimination on the 3x3 normal equations. */
    private fun fitQuadratic(points: List<Pair<Float, Float>>): FloatArray? {
        if (points.size < 3) return null
        var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0; var s4 = 0.0
        var t0 = 0.0; var t1 = 0.0; var t2 = 0.0
        for ((u, g) in points) {
            val uu = u.toDouble()
            s0 += 1.0
            s1 += uu
            s2 += uu * uu
            s3 += uu * uu * uu
            s4 += uu * uu * uu * uu
            t0 += g
            t1 += g * uu
            t2 += g * uu * uu
        }
        val a = arrayOf(
            doubleArrayOf(s0, s1, s2, t0),
            doubleArrayOf(s1, s2, s3, t1),
            doubleArrayOf(s2, s3, s4, t2)
        )
        return gaussSolve(a)
    }

    private fun gaussSolve(a: Array<DoubleArray>): FloatArray? {
        val n = 3
        for (col in 0 until n) {
            var piv = col
            var pivAbs = kotlin.math.abs(a[col][col])
            for (row in col + 1 until n) {
                val v = kotlin.math.abs(a[row][col])
                if (v > pivAbs) {
                    piv = row
                    pivAbs = v
                }
            }
            if (pivAbs < 1e-12) return null
            if (piv != col) {
                val tmp = a[col]
                a[col] = a[piv]
                a[piv] = tmp
            }
            for (row in col + 1 until n) {
                val f = a[row][col] / a[col][col]
                for (k in col..n) {
                    a[row][k] -= f * a[col][k]
                }
            }
        }
        val x = DoubleArray(n)
        for (row in n - 1 downTo 0) {
            var sum = a[row][n]
            for (col in row + 1 until n) {
                sum -= a[row][col] * x[col]
            }
            x[row] = sum / a[row][row]
        }
        return FloatArray(n) { x[it].toFloat() }
    }
}

/** Apply a 0..1 [strength] to a learned gain map on the fly:
 *  g' = 1 + (g - 1) * strength. Unity strength returns [data] unchanged. */
fun scaleLensShadingGains(data: LensShadingData, strength: Float): LensShadingData {
    if (strength >= 1f) return data
    val coef = strength.coerceIn(0f, 1f)
    if (coef <= 0f) {
        return LensShadingData(
            rGains = oneGrid(data.height, data.width),
            grGains = oneGrid(data.height, data.width),
            gbGains = oneGrid(data.height, data.width),
            bGains = oneGrid(data.height, data.width),
            width = data.width,
            height = data.height,
            available = false
        )
    }
    fun scaled(rows: Array<FloatArray>): Array<FloatArray> =
        rows.map { row -> FloatArray(row.size) { c -> 1f + (row[c] - 1f) * coef } }.toTypedArray()
    return LensShadingData(
        rGains = scaled(data.rGains),
        grGains = scaled(data.grGains),
        gbGains = scaled(data.gbGains),
        bGains = scaled(data.bGains),
        width = data.width,
        height = data.height,
        available = data.available
    )
}

private fun oneGrid(h: Int, w: Int): Array<FloatArray> =
    Array(h) { FloatArray(w) { 1f } }