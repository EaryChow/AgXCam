package com.agx.camera.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class DaylightAnchorEstimatorTest {

    private fun gains(r: Float, b: Float) = floatArrayOf(r, 1f, 1f, b)

    private fun feed(est: DaylightAnchorEstimator, n: Int, r: Float, b: Float) {
        repeat(n) { est.addSample(gains(r, b)) }
    }

    @Test
    fun idleEstimatorAcceptsNothing() {
        val est = DaylightAnchorEstimator()
        assertEquals(DaylightAnchorEstimator.State.IDLE, est.state)
        repeat(20) { est.addSample(gains(2.5f, 1.8f)) }
        assertNull(est.currentAnchor)
        assertEquals(0, est.sampledFrames)
    }

    @Test
    fun aStableStreamConvergesToItsMedian() {
        val est = DaylightAnchorEstimator()
        est.start()
        // Vendor daylight answers wobble by a few percent frame to frame; the
        // anchor is the median of the stream, not any single readback.
        val seq = listOf(
            2.50f to 1.85f, 2.52f to 1.87f, 2.49f to 1.83f, 2.51f to 1.86f,
            2.50f to 1.85f, 2.53f to 1.88f, 2.48f to 1.84f, 2.51f to 1.86f,
            2.50f to 1.85f, 2.52f to 1.87f, 2.49f to 1.83f, 2.51f to 1.86f,
            2.50f to 1.85f, 2.52f to 1.87f, 2.49f to 1.84f, 2.51f to 1.86f
        )
        for ((r, b) in seq) est.addSample(gains(r, b))

        assertEquals(DaylightAnchorEstimator.State.CONVERGED, est.state)
        val anchor = est.currentAnchor!!
        assertClose("r", 2.505f, anchor.rGain, 0.01f)
        assertClose("b", 1.855f, anchor.bGain, 0.01f)
        assertEquals(seq.size, anchor.frames)
    }

    @Test
    fun garbageSamplesAreDropped() {
        val est = DaylightAnchorEstimator()
        est.start()
        val bad: List<FloatArray?> = listOf(
            null,
            floatArrayOf(1f, 1f, 1f),
            floatArrayOf(Float.NaN, 1f, 1f, 1.8f),
            floatArrayOf(2.5f, 1f, 1f, Float.POSITIVE_INFINITY),
            floatArrayOf(-2.5f, 1f, 1f, 1.8f),
            floatArrayOf(2.5f, 1f, 1f, 0f),
            floatArrayOf(2.5f, 1f, 1f, 1.8f, 9f)   // extra channels are fine to carry
        )
        for (g in bad) est.addSample(g)
        // Only the well-formed sample (the last one) landed.
        assertEquals(1, est.sampledFrames)
        assertNotNull(est.currentAnchor)
    }

    @Test
    fun aWildSampleDoesNotMoveTheMedian() {
        val est = DaylightAnchorEstimator()
        est.start()
        feed(est, 16, 2.50f, 1.85f)
        val before = est.currentAnchor!!
        // One mid-switch readback an order of magnitude off; the HAL reports
        // sane values again afterwards.
        est.addSample(gains(25f, 18.5f))
        est.addSample(gains(2.50f, 1.85f))
        val after = est.currentAnchor!!
        assertTrue("r moved", abs(after.rGain - before.rGain) < 0.01f)
        assertTrue("b moved", abs(after.bGain - before.bGain) < 0.01f)
    }

    @Test
    fun theMedianTracksASlowDriftAfterConvergence() {
        val est = DaylightAnchorEstimator()
        est.start()
        feed(est, 16, 2.50f, 1.85f)
        assertEquals(DaylightAnchorEstimator.State.CONVERGED, est.state)
        // Sampling keeps running after convergence: a persistent shift in the
        // vendor's daylight answer has to track, not freeze at first contact.
        feed(est, 32, 2.60f, 1.90f)
        val anchor = est.currentAnchor!!
        assertTrue("r did not track: ${anchor.rGain}", anchor.rGain > 2.55f)
        assertTrue("b did not track: ${anchor.bGain}", anchor.bGain > 1.87f)
    }

    @Test
    fun resetClearsEverything() {
        val est = DaylightAnchorEstimator()
        est.start()
        feed(est, 16, 2.5f, 1.85f)
        est.reset()
        assertEquals(DaylightAnchorEstimator.State.IDLE, est.state)
        assertEquals(0, est.sampledFrames)
        assertNull(est.currentAnchor)
        est.addSample(gains(2.5f, 1.85f))
        assertEquals(0, est.sampledFrames)
    }

    @Test
    fun greensAreGreenReferenced() {
        val est = DaylightAnchorEstimator()
        est.start()
        // Slightly split green sites average into the reference.
        est.addSample(floatArrayOf(2.68f, 0.999f, 1.001f, 1.851f))
        val anchor = est.currentAnchor!!
        assertClose("r", 2.68f, anchor.rGain, 1e-3f)
        assertClose("b", 1.851f, anchor.bGain, 1e-3f)
    }

    private fun assertClose(label: String, expected: Float, actual: Float, tolerance: Float) {
        assertTrue(
            "$label: expected $expected but was $actual (tolerance $tolerance)",
            abs(expected - actual) <= tolerance
        )
    }
}
