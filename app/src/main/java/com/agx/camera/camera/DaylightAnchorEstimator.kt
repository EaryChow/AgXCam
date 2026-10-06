package com.agx.camera.camera

// Runtime sampler for the device's own daylight white balance, reduced to the
// diagonal bridge between the reported sensor color matrices and the live RAW
// gain space.
//
// The DNG color matrices describe a nominal sensor. On real devices they are
// not always authored in the gain space the RAW stream actually lives in: the
// profile's answer to "what does the sensor say to a D65 white" then disagrees
// with the device's own daylight gains, and every path that trusts the
// profile's sensor model (the fixed white-balance presets, the Kelvin solve)
// renders the mismatch as a cast. The diagonal ratio between the two is a
// constant of the sensor, so it can be sampled once and kept, the way the
// lens shading map is sampled: feed COLOR_CORRECTION_GAINS readbacks from the
// HAL while it is pinned to its fixed DAYLIGHT mode, take the median of the
// converged stream, and hand it to RawColorProfile.
//
// Sampling runs continuously after convergence too: the estimate is a sliding
// median, so a slow drift in the vendor's daylight answer tracks instead of
// going stale.
class DaylightAnchorEstimator {

    enum class State { IDLE, SAMPLING, CONVERGED }

    /** Green-normalized device daylight gains with the sample count behind them. */
    data class Anchor(val rGain: Float, val bGain: Float, val frames: Int)

    private companion object {
        const val WINDOW = 32
        const val MIN_SAMPLES = 12
        const val SPREAD_EPS = 0.04f
        const val MIN_GAIN = 0.3f
        const val MAX_GAIN = 8f
    }

    private val lock = Any()

    @Volatile
    var state = State.IDLE
        private set

    var sampledFrames = 0
        private set

    // Ring of accepted (r, b) pairs, green-referenced.
    private val window = ArrayDeque<FloatArray>(WINDOW)

    private var anchor: Anchor? = null

    val currentAnchor: Anchor?
        get() = synchronized(lock) { anchor }

    fun start() = synchronized(lock) {
        window.clear()
        anchor = null
        sampledFrames = 0
        state = State.SAMPLING
    }

    fun reset() = synchronized(lock) {
        window.clear()
        anchor = null
        sampledFrames = 0
        state = State.IDLE
    }

    /**
     * Feed one HAL COLOR_CORRECTION_GAINS readback (R, Gr, Gb, B). Only
     * converged daylight-mode readbacks may reach this; the caller gates on
     * the AWB mode and state. Garbage (non-finite or wildly out-of-range
     * gains, e.g. a HAL mid-switch) is dropped. Returns the refreshed anchor,
     * or null while there is not one yet.
     */
    fun addSample(gains: FloatArray?): Anchor? = synchronized(lock) {
        if (state != State.SAMPLING && state != State.CONVERGED) return null
        if (gains == null || gains.size < 4) return null
        // Only the four CFA channels are read; a longer array may carry more.
        for (i in 0..3) {
            val v = gains[i]
            if (!v.isFinite() || v < MIN_GAIN || v > MAX_GAIN) return null
        }
        val gMean = (gains[1] + gains[2]) * 0.5f
        if (!gMean.isFinite() || gMean <= 0f) return null

        val r = gains[0] / gMean
        val b = gains[3] / gMean
        window.addLast(floatArrayOf(r, b))
        while (window.size > WINDOW) window.removeFirst()
        sampledFrames++

        val rs = window.map { it[0] }.sorted()
        val bs = window.map { it[1] }.sorted()
        val medR = median(rs)
        val medB = median(bs)
        anchor = Anchor(medR, medB, sampledFrames)

        if (state == State.SAMPLING && sampledFrames >= MIN_SAMPLES &&
            spread(rs, medR) < SPREAD_EPS && spread(bs, medB) < SPREAD_EPS
        ) {
            state = State.CONVERGED
        }
        anchor
    }

    private fun median(sorted: List<Float>): Float =
        if (sorted.size % 2 == 1) sorted[sorted.size / 2]
        else (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) * 0.5f

    // 10th..90th percentile range over the median: a scale-free stability
    // measure that converges when the vendor's daylight answer stops moving.
    private fun spread(sorted: List<Float>, med: Float): Float {
        if (sorted.isEmpty() || med <= 0f) return Float.MAX_VALUE
        val lo = sorted[(sorted.size * 0.1f).toInt().coerceIn(0, sorted.size - 1)]
        val hi = sorted[((sorted.size - 1) * 0.9f).toInt().coerceIn(0, sorted.size - 1)]
        return (hi - lo) / med
    }
}
