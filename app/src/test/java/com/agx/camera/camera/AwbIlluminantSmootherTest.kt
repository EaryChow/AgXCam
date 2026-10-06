package com.agx.camera.camera

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

// Ports of the validated scenarios in exp_awb_smoothing.py: the dual-rate
// state machine must snap the first trusted estimate, hand over to the slow
// pole only on a settled estimate, fast-acquire genuine changes in seconds,
// and hold still under oscillation.
class AwbIlluminantSmootherTest {

    private val tung = floatArrayOf(0.85f, 1f, 1f, 1.10f)
    private val day = floatArrayOf(1.20f, 1f, 1f, 1.30f)
    private val rggb = intArrayOf(0, 1, 1, 2)

    // Half-seconds per estimator step, as on device.
    private var tNanos = 0L
    private fun step(s: AwbIlluminantSmoother, raw: FloatArray): Float {
        tNanos += 500_000_000L
        return s.step(raw, tNanos)
    }

    @Test
    fun firstEstimateSnapsExactlyAndArmsOneShotFlags() {
        val s = AwbIlluminantSmoother()
        val a = step(s, tung)
        assertEquals("snap step applies no EMA", 0f, a, 0f)
        assertArrayEquals(tung, s.gains, 1e-6f)
        assertTrue("neutral snap armed", s.consumeSnapNeutral())
        assertTrue("adaptation snap armed", s.consumeSnapAdaptation())
        assertFalse("neutral snap is one-shot", s.consumeSnapNeutral())
        assertFalse("adaptation snap is one-shot", s.consumeSnapAdaptation())
    }

    @Test
    fun unsettledStartupTracksFastThenHandsOverWithASnap() {
        val s = AwbIlluminantSmoother()
        // Wandering early estimates: never stable twice in a row.
        step(s, floatArrayOf(0.90f, 1f, 1f, 1.15f))
        step(s, floatArrayOf(0.80f, 1f, 1f, 1.05f))
        step(s, floatArrayOf(0.88f, 1f, 1f, 1.18f))
        assertTrue("still acquiring", step(s, tung) > 0.1f)
        // Settle: the assert step counted one stable estimate, so two more
        // promote to the slow pole and adopt the estimate exactly.
        step(s, tung)
        val a = step(s, tung)
        assertEquals("handover snaps", 0f, a, 0f)
        assertArrayEquals(tung, s.gains, 1e-6f)
    }

    @Test
    fun genuineChangeConvergesInSeconds() {
        val s = AwbIlluminantSmoother()
        repeat(8) { step(s, tung) }          // acquire + handover
        repeat(30) { step(s, tung) }         // settle on the slow pole
        var clearAt = -1
        for (i in 0 until 30) {
            step(s, day)
            val err = maxErr(s.gains, day)
            if (clearAt < 0 && err < 0.05f) clearAt = i
        }
        assertTrue("fast acquire never converged", clearAt >= 0)
        assertTrue("converged slowly: $clearAt steps", clearAt <= 10)
    }

    @Test
    fun oscillationNeverLeavesTheSlowPole() {
        val s = AwbIlluminantSmoother()
        val a = floatArrayOf(0.80f, 1f, 1f, 0.85f)
        val b = floatArrayOf(1.30f, 1f, 1f, 1.40f)
        repeat(8) { step(s, a) }             // acquire + handover on a
        var jitter = 0f
        var prev = s.gains.copyOf()
        for (i in 0 until 60) {
            step(s, if (i % 2 == 0) a else b)
            jitter = maxOf(jitter, maxErr(s.gains, prev))
            prev = s.gains.copyOf()
        }
        assertTrue("output jitter under oscillation: $jitter", jitter < 0.02f)
    }

    @Test
    fun slowDriftTracksGradually() {
        val s = AwbIlluminantSmoother()
        repeat(8) { step(s, tung) }
        // Small per-step walk toward daylight: stable, and its steady-state
        // lag stays below the fast-acquire threshold - the slow pole must
        // follow it without fast-acquiring. (Lag on a ramp is rate x tau/dt;
        // 0.002 per step keeps it under 0.05.)
        var cur = tung.copyOf()
        repeat(20) {
            for (p in cur.indices) cur[p] += (day[p] - tung[p]) * 0.002f
            step(s, cur)
        }
        assertTrue(
            "slow drift tracked: ${s.gains.contentToString()}",
            maxErr(s.gains, cur) < 0.05f
        )
    }

    @Test
    fun resetRestartsTheAcquirePhase() {
        val s = AwbIlluminantSmoother()
        step(s, tung)
        repeat(4) { step(s, tung) }
        s.reset()
        assertArrayEquals(FloatArray(4) { 1f }, s.gains, 0f)
        // First estimate after the reset snaps again.
        val a = step(s, day)
        assertEquals(0f, a, 0f)
        assertArrayEquals(day, s.gains, 1e-6f)
        assertTrue(s.consumeSnapNeutral())
    }

    @Test
    fun retargetPreservesTheAppliedProduct() {
        val s = AwbIlluminantSmoother()
        repeat(6) { step(s, tung) }          // acquire + handover
        val before = s.gains.copyOf()
        val oldD = floatArrayOf(2.1f, 1f, 1.44f)
        val newD = floatArrayOf(2.5f, 1f, 1.83f)
        s.retarget(oldD, newD, rggb)
        // Product per phase (phaseGains = illuminant * daylight) must not move.
        for (p in 0 until 4) {
            val oldG = when (rggb[p]) { 0 -> oldD[0]; 2 -> oldD[2]; else -> 1f }
            val newG = when (rggb[p]) { 0 -> newD[0]; 2 -> newD[2]; else -> 1f }
            assertEquals(
                "phase $p product",
                before[p] * oldG, s.gains[p] * newG, 1e-4f
            )
        }
        // Before the first estimate there is nothing to carry over.
        val fresh = AwbIlluminantSmoother()
        fresh.retarget(oldD, newD, rggb)
        assertArrayEquals(FloatArray(4) { 1f }, fresh.gains, 0f)
    }

    @Test
    fun alphasMatchTheDesignedTimeConstants() {
        val s = AwbIlluminantSmoother()
        // dt clamped to [0.05, 5] s; at the 0.5 s cadence the fast pole is
        // 1-exp(-0.5/1.5) and the slow pole 1-exp(-0.5/29).
        val a = step(s, tung)      // snap: 0
        assertEquals(0f, a, 0f)
        val aFast = step(s, floatArrayOf(0.86f, 1f, 1f, 1.11f))
        assertEquals((1.0 - kotlin.math.exp(-0.5 / 1.5)).toFloat(), aFast, 1e-4f)
        // Far away the slow pole still creeps: hand over, park on the slow
        // pole with a near-identical estimate, then check the alpha.
        repeat(8) { step(s, tung) }
        val aSlow = step(s, floatArrayOf(0.851f, 1f, 1f, 1.101f))
        assertEquals((1.0 - kotlin.math.exp(-0.5 / 29.0)).toFloat(), aSlow, 1e-4f)
        assertNotEquals(aFast, aSlow, 1e-3f)
    }

    private fun maxErr(a: FloatArray, b: FloatArray): Float {
        var m = 0f
        for (i in a.indices) m = maxOf(m, abs(a[i] - b[i]))
        return m
    }
}
