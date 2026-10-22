package com.agx.camera.color

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WhiteBalanceMathModeTest {

    private val epsilon = 1e-9

    @Test
    fun ambienceTint_isZeroAtAndAboveTheRampStart() {
        assertEquals(0.0, WhiteBalanceMath.ambienceTint(3700.0), epsilon)
        assertEquals(0.0, WhiteBalanceMath.ambienceTint(4200.0), epsilon)
    }

    @Test
    fun ambienceTint_isFullAtAndBelowTheRampEnd() {
        assertEquals(30.0, WhiteBalanceMath.ambienceTint(2900.0), epsilon)
        assertEquals(30.0, WhiteBalanceMath.ambienceTint(2400.0), epsilon)
    }

    @Test
    fun ambienceTint_rampsAcrossTheBand() {
        assertEquals(30.0, WhiteBalanceMath.ambienceTint(3350.0), 1e-6)
        assertEquals(15.0, WhiteBalanceMath.ambienceTint(3550.0), 1e-6)
        assertEquals(0.0, WhiteBalanceMath.ambienceTint(3700.0), 1e-6)
    }

    @Test
    fun modeTargetCct_slidesUpFromTheMeanEstimate() {
        assertEquals(5250.0, WhiteBalanceMath.modeTargetCct(4000.0), epsilon)
        assertEquals(3100.0, WhiteBalanceMath.modeTargetCct(1850.0), epsilon)
    }

    @Test
    fun modeTargetCct_capsShortOfD65() {
        assertEquals(6490.0, WhiteBalanceMath.modeTargetCct(6000.0), epsilon)
        assertEquals(6490.0, WhiteBalanceMath.modeTargetCct(7000.0), epsilon)
    }

    @Test
    fun modeTargetTint_addsTheBlockOffsetAndTheAmbienceBias() {
        // 0.004 duv is 12 tint units; 3000 K sits below the full-ramp end.
        assertEquals(12.0 + 30.0,
            WhiteBalanceMath.modeTargetTint(0.004, 3000.0), 1e-6)
        // Cool means carry no ambience bias.
        assertEquals(-6.0, WhiteBalanceMath.modeTargetTint(-0.002, 5200.0), 1e-6)
    }

    @Test
    fun modeTargetXy_matchesTheKelvinPair() {
        val (expectedX, expectedY) = WhiteBalanceMath.kelvinToXy(
            WhiteBalanceMath.modeTargetCct(3600.0).toFloat(),
            WhiteBalanceMath.modeTargetTint(0.003, 3600.0).toFloat()
        )
        val (x, y) = WhiteBalanceMath.modeTargetXy(0.003, 3600.0)
        assertEquals(expectedX, x, 1e-6f)
        assertEquals(expectedY, y, 1e-6f)
    }

    @Test
    fun modeTargetXy_movesTowardGreenWithPositiveBlockDuv() {
        val meanCct = 3800.0
        val (_, yNeutral) = WhiteBalanceMath.modeTargetXy(0.0, meanCct)
        val (_, yGreen) = WhiteBalanceMath.modeTargetXy(0.004, meanCct)
        val (_, yMagenta) = WhiteBalanceMath.modeTargetXy(-0.004, meanCct)
        assertTrue("positive duv raises y", yGreen > yNeutral)
        assertTrue("negative duv lowers y", yMagenta < yNeutral)
    }

    @Test
    fun signedLocusDistance_isPositiveOnTheGreenSide() {
        val (xGreen, yGreen) = WhiteBalanceMath.kelvinToXy(5000f, 10f)
        val green = WhiteBalanceMath.signedLocusDistance(xGreen, yGreen)
        assertNotNull(green)
        assertEquals(10.0 / 3000.0, green!![0], 1e-4)

        val (xMagenta, yMagenta) = WhiteBalanceMath.kelvinToXy(5000f, -10f)
        val magenta = WhiteBalanceMath.signedLocusDistance(xMagenta, yMagenta)
        assertNotNull(magenta)
        assertEquals(-10.0 / 3000.0, magenta!![0], 1e-4)
    }

    @Test
    fun signedLocusDistance_isNearZeroOnTheLocus() {
        val (x, y) = WhiteBalanceMath.kelvinToXy(4000f, 0f)
        val d = WhiteBalanceMath.signedLocusDistance(x, y)
        assertNotNull(d)
        assertEquals(0.0, d!![0], 1e-4)
        assertEquals(4000.0, d[1], 25.0)
    }

    @Test
    fun signedLocusDistance_reportsD65OnTheGreenSide() {
        val d = WhiteBalanceMath.signedLocusDistance(0.3127f, 0.3290f)
        assertNotNull(d)
        assertTrue("D65 sits on the green side of the locus", d!![0] > 0.0)
        assertEquals(6500.0, d[1], 60.0)
    }

    @Test
    fun signedLocusDistance_rejectsChromaticitiesOutsideUv() {
        assertEquals(null, WhiteBalanceMath.signedLocusDistance(0.0f, -0.3f))
    }
}
