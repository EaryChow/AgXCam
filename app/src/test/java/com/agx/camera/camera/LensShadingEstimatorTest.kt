package com.agx.camera.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Verifies the Monte-Carlo lens-shading estimator (§15.1 fallback): sparse green
 * sampling accumulating into a coarse grid, radial-bin + polynomial fit, the
 * achromatic LensShadingData encoding, the pause/resume/converge life cycle the
 * settings panel drives, and the time-gated convergence that keeps short
 * correlated scenes from being locked in.
 */
class LensShadingEstimatorTest {

    // Small grids keep the fits and center-handling exerciseable in unit tests.
    private fun estimator(): LensShadingEstimator = LensShadingEstimator(
        gridWidth = 8,
        gridHeight = 6,
        sampleStride = 4,
        binCount = 12,
        centerUTop = 0.06f,
        minFramesBeforeConverge = 10,
        convergeDelta = 0.02f,
        convergeStreakRequired = 3,
        minConvergeMillis = 0L,
        maxConvergeMillis = 0L,
        gainMax = 3f
    )

    @Test
    fun addFrame_beforeStart_returnsNull() {
        val est = estimator()
        val buf = makeFrame(128, 96) { _, _, _ -> 64 + 700.0 }
        assertNull(est.addFrame(buf))
        assertEquals(LensShadingState.IDLE, est.currentState)
        assertEquals(0, est.sampledFrames)
    }

    @Test
    fun flatFrame_buildsIdentityAchromaticMap() {
        val est = estimator()
        est.start(128, 96, BayerPattern.RGGB, 1023, 64f)
        val buf = makeFrame(128, 96) { _, _, _ -> 64 + 700.0 }

        var map: LensShadingData? = null
        for (f in 0 until 20) {
            map = est.addFrame(buf)
            if (map != null) break
        }
        assertNotNull("flat frame should yield a map quickly", map)
        val m = map!!

        // All four phases carry the same achromatic gain grid.
        for (row in 0 until m.height) {
            for (col in 0 until m.width) {
                assertEquals(m.rGains[row][col], m.grGains[row][col], 0f)
                assertEquals(m.rGains[row][col], m.gbGains[row][col], 0f)
                assertEquals(m.rGains[row][col], m.bGains[row][col], 0f)
                assertTrue("flat field gain ~1", m.rGains[row][col] in 0.98f..1.05f)
            }
        }
        assertFalse("identity-ish map is marked unavailable", m.available)
    }

    @Test
    fun vignettedFrame_raisesCorners_andConverges() {
        val est = estimator()
        est.start(128, 96, BayerPattern.RGGB, 1023, 64f)
        val buf = makeFrame(128, 96) { c, r, _ ->
            val u = radiusSquared(c, r, 128, 96)
            64 + 700.0 * (1.0 - 0.35 * u)
        }

        var map: LensShadingData? = null
        var feed = 0
        while (feed < 40) {
            map = est.addFrame(buf)
            feed++
            if (est.currentState == LensShadingState.CONVERGED) break
        }
        val m = requireNotNull(map) { "vignetted frame should yield a map" }

        assertTrue("estimator should converge", est.currentState == LensShadingState.CONVERGED)
        assertTrue(est.sampledFrames >= 10)
        assertTrue("vignette map is marked available", m.available)

        val center = m.rGains[3][3]
        val corner = m.rGains[5][7]
        assertTrue("center gain near 1.0, was $center", center in 0.95f..1.15f)
        assertTrue("corner gain must exceed center by the vignette, was $corner", corner > 1.15f)
    }

    @Test
    fun timeGate_delaysConvergenceUntilMinConvergeMillis() {
        var now = 0L
        val est = LensShadingEstimator(
            gridWidth = 8,
            gridHeight = 6,
            sampleStride = 4,
            binCount = 12,
            centerUTop = 0.06f,
            minFramesBeforeConverge = 5,
            convergeDelta = 0.02f,
            convergeStreakRequired = 2,
            minConvergeMillis = 100_000L,
            maxConvergeMillis = 0L,
            gainMax = 3f,
            clock = { now }
        )
        est.start(128, 96, BayerPattern.RGGB, 1023, 64f)
        val buf = makeFrame(128, 96) { _, _, _ -> 64 + 700.0 }

        var map: LensShadingData? = null
        for (f in 0 until 20) {
            map = est.addFrame(buf)
        }
        assertNotNull(map)
        assertTrue("streak should have built", est.sampledFrames >= 10)
        assertFalse(
            "convergence must wait for the minimum sampling duration, state=${est.currentState}",
            est.currentState == LensShadingState.CONVERGED
        )

        now = 100_001L
        map = est.addFrame(buf)
        assertNotNull(map)
        assertTrue("time gate passed, flat field converges", est.currentState == LensShadingState.CONVERGED)
    }

    @Test
    fun maxConvergeMillis_forcesFinalizeEvenWithoutStreak() {
        var now = 0L
        val est = LensShadingEstimator(
            gridWidth = 8,
            gridHeight = 6,
            sampleStride = 4,
            binCount = 12,
            centerUTop = 0.06f,
            minFramesBeforeConverge = 1_000,
            convergeDelta = 0.001f,
            convergeStreakRequired = 1_000,
            minConvergeMillis = 1_000_000L,
            maxConvergeMillis = 50_000L,
            gainMax = 3f,
            clock = { now }
        )
        est.start(128, 96, BayerPattern.RGGB, 1023, 64f)
        val buf = makeFrame(128, 96) { _, _, _ -> 64 + 700.0 }

        var map: LensShadingData? = null
        for (f in 0 until 12) {
            val m = est.addFrame(buf)
            if (m != null && map == null) map = m
        }
        assertNotNull("center coverage builds within a few frames", map)
        assertFalse(
            "streak/time gate untouched, must not be converged yet, state=${est.currentState}",
            est.currentState == LensShadingState.CONVERGED
        )

        now = 60_000L
        map = est.addFrame(buf)
        assertNotNull(map)
        assertTrue("max duration forces the map finalization", est.currentState == LensShadingState.CONVERGED)
        assertTrue(est.elapsedMillis >= 60_000L)
    }

    @Test
    fun pause_haltsAccumulation_resumeContinues() {
        val est = estimator()
        est.start(128, 96, BayerPattern.RGGB, 1023, 64f)
        val buf = makeFrame(128, 96) { _, _, _ -> 64 + 700.0 }

        var map: LensShadingData? = null
        var f = 0
        while (f < 20 && map == null) {
            map = est.addFrame(buf)
            f++
        }
        assertNotNull(map)
        val framesBeforePause = est.sampledFrames
        assertTrue(framesBeforePause > 0)
        val elapsedBeforePause = est.elapsedMillis

        est.pause()
        assertEquals(LensShadingState.PAUSED, est.currentState)
        assertNull("paused: no new frames accepted", est.addFrame(buf))
        assertEquals(framesBeforePause, est.sampledFrames)

        est.resume()
        assertEquals(LensShadingState.SAMPLING, est.currentState)
        assertNotNull("resumed: frames accumulate again", est.addFrame(buf))
        assertTrue(est.sampledFrames > framesBeforePause)

        est.pause()
        assertTrue("pause freezes the sampling clock", est.elapsedMillis >= elapsedBeforePause)
    }

    @Test
    fun reset_clearsEstimator_relearnable() {
        val est = estimator()
        est.start(128, 96, BayerPattern.RGGB, 1023, 64f)
        val buf = makeFrame(128, 96) { _, _, _ -> 64 + 700.0 }
        var f = 0
        while (f < 20) {
            est.addFrame(buf)
            f++
        }
        assertTrue(est.sampledFrames > 0)

        est.reset()
        assertEquals(LensShadingState.IDLE, est.currentState)
        assertEquals(0, est.sampledFrames)
        assertEquals(0L, est.elapsedMillis)
        assertNull("reset: no accumulation until start()", est.addFrame(buf))
        assertNull("reset: last map dropped", est.buildMap())

        est.start(128, 96, BayerPattern.RGGB, 1023, 64f)
        assertNotNull("fresh start accumulates again", est.addFrame(buf))
        assertEquals(LensShadingState.SAMPLING, est.currentState)
    }

    @Test
    fun scaleLensShadingGains_scalesAroundUnity() {
        val data = LensShadingData(
            rGains = arrayOf(floatArrayOf(1f, 2f), floatArrayOf(1.5f, 1f)),
            grGains = arrayOf(floatArrayOf(1f, 2f), floatArrayOf(1.5f, 1f)),
            gbGains = arrayOf(floatArrayOf(1f, 2f), floatArrayOf(1.5f, 1f)),
            bGains = arrayOf(floatArrayOf(1f, 2f), floatArrayOf(1.5f, 1f)),
            width = 2,
            height = 2,
            available = true
        )

        assertSame("unity strength returns the same instance", data, scaleLensShadingGains(data, 1f))

        val half = scaleLensShadingGains(data, 0.5f)
        assertEquals(1f, half.rGains[0][0], 1e-4f)
        assertEquals(1.5f, half.rGains[0][1], 1e-4f)
        assertEquals(1.25f, half.rGains[1][0], 1e-4f)

        val off = scaleLensShadingGains(data, 0f)
        for (row in 0 until off.height) {
            for (col in 0 until off.width) {
                assertEquals("zero strength collapses to identity", 1f, off.rGains[row][col], 1e-4f)
            }
        }
        assertFalse(off.available)
        assertNotSame(data, off)
    }

    private fun makeFrame(
        w: Int,
        h: Int,
        fill: (c: Int, r: Int, phase: Int) -> Double
    ): ByteBuffer {
        val buf = ByteBuffer.allocate(w * h * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (r in 0 until h) {
            for (c in 0 until w) {
                val phase = (c % 2) + (r % 2) * 2
                val v = fill(c, r, phase).toDouble().coerceIn(0.0, 1023.0)
                buf.putShort((v + 0.5).toInt().toShort())
            }
        }
        buf.rewind()
        return buf
    }

    private fun radiusSquared(c: Int, r: Int, w: Int, h: Int): Double {
        val cx = (c + 0.5) / w - 0.5
        val cy = (r + 0.5) / h - 0.5
        return (cx * cx + cy * cy) * 2.0
    }
}