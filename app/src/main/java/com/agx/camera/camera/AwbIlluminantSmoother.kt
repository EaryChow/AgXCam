package com.agx.camera.camera

// Dual-rate temporal smoother for the AWB illuminant estimate.
//
// The estimator steps about every 15th frame; between steps the smoothed
// value is simply held. Three regimes:
//
//  - Acquire (startup, or a scene still in flux): the estimate is not
//    trusted yet, so it is tracked fast instead of smoothed; smoothing an
//    unconverged estimate would anchor to a wrong answer for a long time.
//    The very first trusted estimate is adopted exactly (starting from the
//    neutral [1,1,1,1] would drag the gains through a transient on the first
//    visible frames). Once the estimate holds still across HANDOVER
//    consecutive steps it is trusted as genuine, and adopted exactly again -
//    a trusted estimate should not keep a partially-converged residual for
//    the slow pole to bleed off.
//  - Slow pole: ~20 s half-life. This is what kills flicker when a light
//    source oscillates between two colors: an oscillating raw estimate never
//    stabilizes, so it never leaves the slow pole, and the held value barely
//    moves.
//  - Fast acquire: engages only when a new estimate is far from the smoothed
//    value AND stable across consecutive steps - a genuine new illuminant.
//    It persists until truly converged (below RELEASE_EPS), not merely below
//    the engage threshold, and bails back to the slow pole if the estimate
//    turns unstable.
//
// Threading: step/reset run on the raw frame thread; reset also runs from
// the preview pipeline thread on session changes. All state is guarded by
// the internal lock. The gains array is owned here and mutated only under
// the lock; readers take the reference once per estimator step.
class AwbIlluminantSmoother {

    private val lock = Any()

    // Smoothed illuminant stage, per CFA phase.
    val gains = FloatArray(GreyWorldEstimate.PHASE_COUNT) { 1f }

    private var prev: FloatArray? = null
    private var fastHoldSteps = 0
    private var fastAcquire = false
    private var acquiring = true
    private var stableSteps = 0
    private var snapNeutral = false
    private var snapAdaptation = false
    private var lastStepNanos = 0L

    fun reset() = synchronized(lock) {
        for (p in gains.indices) gains[p] = 1f
        prev = null
        fastHoldSteps = 0
        fastAcquire = false
        acquiring = true
        stableSteps = 0
        snapNeutral = false
        snapAdaptation = false
        lastStepNanos = 0L
    }

    /**
     * One estimator step: applies the EMA to [gains] and returns the alpha
     * used. The first step of a session adopts the estimate exactly and arms
     * the snap flags (see [consumeSnapNeutral] / [consumeSnapAdaptation]).
     */
    fun step(raw: FloatArray, nowNanos: Long): Float = synchronized(lock) {
        val dtSec = if (lastStepNanos > 0L) {
            ((nowNanos - lastStepNanos) / 1_000_000_000.0).toFloat().coerceIn(0.05f, 5f)
        } else {
            0.5f
        }
        lastStepNanos = nowNanos
        val alphaSlow = (1.0 - kotlin.math.exp(-dtSec / SLOW_TAU_S)).toFloat()
        val alphaFast = (1.0 - kotlin.math.exp(-dtSec / FAST_TAU_S)).toFloat()

        val previous = prev
        prev = raw.copyOf()
        if (previous == null) {
            for (p in raw.indices) gains[p] = raw[p]
            snapNeutral = true
            snapAdaptation = true
            return@synchronized 0f
        }
        var stable = true
        for (p in raw.indices) {
            if (kotlin.math.abs(raw[p] - previous[p]) > STABLE_EPS) stable = false
        }
        if (acquiring) {
            stableSteps = if (stable) stableSteps + 1 else 0
            if (stableSteps >= STABLE_HANDOVER) {
                acquiring = false
                fastHoldSteps = 0
                for (p in raw.indices) gains[p] = raw[p]
                return@synchronized 0f
            }
            for (p in raw.indices) gains[p] += alphaFast * (raw[p] - gains[p])
            return@synchronized alphaFast
        }
        var far = false
        for (p in raw.indices) {
            if (kotlin.math.abs(raw[p] - gains[p]) > FAR_EPS) far = true
        }
        if (fastAcquire) {
            var converged = true
            for (p in raw.indices) {
                if (kotlin.math.abs(raw[p] - gains[p]) > RELEASE_EPS) converged = false
            }
            if (!stable || converged) {
                fastAcquire = false
                fastHoldSteps = 0
                for (p in raw.indices) gains[p] += alphaSlow * (raw[p] - gains[p])
                return@synchronized alphaSlow
            }
            for (p in raw.indices) gains[p] += alphaFast * (raw[p] - gains[p])
            return@synchronized alphaFast
        }
        fastHoldSteps = if (far && stable) fastHoldSteps + 1 else 0
        if (fastHoldSteps >= FAST_HOLD_COUNT) {
            fastAcquire = true
            for (p in raw.indices) gains[p] += alphaFast * (raw[p] - gains[p])
            return@synchronized alphaFast
        }
        for (p in raw.indices) gains[p] += alphaSlow * (raw[p] - gains[p])
        return@synchronized alphaSlow
    }

    /** One-shot flag: the last [step] adopted the first estimate exactly. */
    fun consumeSnapNeutral(): Boolean = synchronized(lock) {
        val s = snapNeutral
        snapNeutral = false
        s
    }

    /** One-shot flag: the last [step] adopted the first estimate exactly. */
    fun consumeSnapAdaptation(): Boolean = synchronized(lock) {
        val s = snapAdaptation
        snapAdaptation = false
        s
    }

    /**
     * When the fixed daylight reference that multiplies this stage is
     * replaced, rescale the smoothed stage so the product the renderer
     * applies stays put. illuminant = level / (means * daylight) exactly, so
     * each phase moves by old/new and the product is preserved. Only
     * meaningful while the estimator owns the smoothing state; right after a
     * reset there is nothing to carry over.
     */
    fun retarget(oldDaylight: FloatArray?, newDaylight: FloatArray?, colorMap: IntArray) =
        synchronized(lock) {
            if (prev == null) return@synchronized
            if (oldDaylight == null || newDaylight == null) return@synchronized
            for (p in gains.indices) {
                val oldG = when (colorMap.getOrElse(p) { 1 }) {
                    0 -> oldDaylight[0]
                    2 -> oldDaylight[2]
                    else -> 1f
                }
                val newG = when (colorMap.getOrElse(p) { 1 }) {
                    0 -> newDaylight[0]
                    2 -> newDaylight[2]
                    else -> 1f
                }
                gains[p] = (gains[p] * oldG / newG)
                    .coerceIn(GreyWorldEstimate.MIN_GAIN, GreyWorldEstimate.MAX_GAIN)
            }
        }

    companion object {
        // Slow pole: ~20 s half-life.
        const val SLOW_TAU_S = 29.0
        // Fast acquire: ~1 s half-life.
        const val FAST_TAU_S = 1.5
        // Engage fast acquire on a deviation above FAR_EPS sustained while
        // the estimate stays stable; release on convergence below
        // RELEASE_EPS or on instability.
        const val FAR_EPS = 0.2f
        const val STABLE_EPS = 0.1f
        const val RELEASE_EPS = 0.05f
        const val FAST_HOLD_COUNT = 2
        // Consecutive stable estimates that promote the smoother from its
        // startup acquire phase to the slow pole.
        const val STABLE_HANDOVER = 3
    }
}
