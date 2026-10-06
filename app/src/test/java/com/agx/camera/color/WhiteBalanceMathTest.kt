package com.agx.camera.color

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs

class WhiteBalanceMathTest {

    private val EPSILON = 0.05f

    private fun vecEquals(a: FloatArray, b: FloatArray, eps: Float = EPSILON) {
        assertEquals("array length", a.size, b.size)
        for (i in a.indices) {
            assertEquals("index $i", b[i], a[i], eps)
        }
    }

    @Test
    fun kelvinToXy_d65() {
        val (x, y) = WhiteBalanceMath.kelvinToXy(6500f)
        assertEquals("x near D65", 0.3127f, x, 0.05f)
        assertTrue("y should be in valid range", y in 0.2f..0.5f)
    }

    @Test
    fun kelvinToXy_illuminantA() {
        val (x, _) = WhiteBalanceMath.kelvinToXy(2856f)
        assertTrue("warm temp x should be > D65 x (0.3127)", x > 0.31f)
    }

    @Test
    fun kelvinToXy_clampsTint() {
        val (_, y1) = WhiteBalanceMath.kelvinToXy(5500f, 200f)
        assertTrue("tint high y should be <= 1.0", y1 <= 1.0f)
        val (_, y2) = WhiteBalanceMath.kelvinToXy(5500f, -200f)
        assertTrue("tint low y should be >= 0.0", y2 >= 0.0f)
    }

    @Test
    fun kelvinToXy_clampsKelvin() {
        val (_, yLow) = WhiteBalanceMath.kelvinToXy(500f)
        val (_, yHigh) = WhiteBalanceMath.kelvinToXy(50000f)
        assertTrue(yLow >= 0.0f)
        assertTrue(yHigh <= 1.0f)
    }

    @Test
    fun kelvinToXy_defaultD65() {
        // The default Kelvin/Tint pair must resolve to D65 itself, so the
        // default KELVIN preset is exactly the DAYLIGHT preset.
        val (x, y) = WhiteBalanceMath.kelvinToXy(6504f, 9.6f)
        assertEquals("default x near D65", 0.3127f, x, 0.001f)
        assertEquals("default y near D65", 0.3290f, y, 0.001f)
    }

    @Test
    fun kelvinToXy_positiveTintMovesTowardGreen() {
        // Duv = tint / 3000, and positive Duv is the green side of the locus.
        var prev = WhiteBalanceMath.kelvinToXy(6504f, -60f).second
        for (tint in -40..60 step 10) {
            val cur = WhiteBalanceMath.kelvinToXy(6504f, tint.toFloat()).second
            assertTrue("y should increase with tint (tint=$tint: $cur <= $prev)", cur > prev)
            prev = cur
        }
    }

    @Test
    fun kelvinToXy_tintIsPerpendicularToLocus() {
        // The Duv offset is applied along the locus normal, so shifting tint at a
        // fixed CCT must not move the point along the locus. The CCT read back
        // from the locus is unchanged, hence the locus normal component dominates
        // the motion: comparing symmetric tint swings, the two results must sit on
        // opposite sides of the locus point in v while x moves toward the locus.
        val locus = WhiteBalanceMath.kelvinToXy(6504f, 0f)
        val green = WhiteBalanceMath.kelvinToXy(6504f, 12f)
        val magenta = WhiteBalanceMath.kelvinToXy(6504f, -12f)
        assertTrue("green side raises y", green.second > locus.second)
        assertTrue("magenta side lowers y", magenta.second < locus.second)
        // The normal direction is stable, so the midpoint of a symmetric swing
        // returns to the locus point rather than drifting along it.
        val midX = (green.first + magenta.first) / 2f
        val midY = (green.second + magenta.second) / 2f
        assertEquals("midpoint x", locus.first, midX, 0.0005f)
        assertEquals("midpoint y", locus.second, midY, 0.0005f)
    }

    @Test
    fun kelvinToXy_planckianAnchors() {
        // CIE Planckian locus reference chromaticities (CIE 018:2019 table).
        val illuminantA = WhiteBalanceMath.kelvinToXy(2856f)
        assertEquals("Illuminant A x", 0.4476f, illuminantA.first, 0.002f)
        assertEquals("Illuminant A y", 0.4074f, illuminantA.second, 0.002f)

        val sixFive = WhiteBalanceMath.kelvinToXy(6500f)
        assertEquals("6500K x", 0.3135f, sixFive.first, 0.002f)
        assertEquals("6500K y", 0.3236f, sixFive.second, 0.002f)

        val twoK = WhiteBalanceMath.kelvinToXy(2000f)
        assertEquals("2000K x", 0.5267f, twoK.first, 0.002f)
        assertEquals("2000K y", 0.4133f, twoK.second, 0.002f)

        val tenK = WhiteBalanceMath.kelvinToXy(10000f)
        assertEquals("10000K x", 0.2807f, tenK.first, 0.002f)
        assertEquals("10000K y", 0.2884f, tenK.second, 0.002f)
    }

    @Test
    fun kelvinToXy_continuousNoSeam() {
        var prev = WhiteBalanceMath.kelvinToXy(2000f)
        var temp = 2025f
        while (temp <= 10000f) {
            val cur = WhiteBalanceMath.kelvinToXy(temp)
            val dx = abs(cur.first - prev.first)
            val dy = abs(cur.second - prev.second)
            assertTrue("jump in x near ${temp}K: $dx", dx < 0.01f)
            assertTrue("jump in y near ${temp}K: $dy", dy < 0.01f)
            prev = cur
            temp += 25f
        }
    }

    @Test
    fun colorMatrix_identity() {
        val id = ColorMatrix.identity()
        val expected = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        vecEquals(expected, id.m)
    }

    @Test
    fun colorMatrix_multiplyIdentity() {
        val a = ColorMatrix.diagonal(2.0f, 3.0f, 4.0f)
        val result = ColorMatrix.multiply(a, ColorMatrix.identity())
        vecEquals(a.m, result.m)
    }

    @Test
    fun colorMatrix_inverseRoundtrip() {
        val a = ColorMatrix.diagonal(2.0f, 3.0f, 4.0f)
        val inv = ColorMatrix.inverse(a)
        val product = ColorMatrix.multiply(a, inv)
        val expected = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        vecEquals(expected, product.m, 0.01f)
    }

    @Test
    fun colorMatrix_determinant() {
        val diag = ColorMatrix.diagonal(2.0f, 3.0f, 4.0f)
        assertEquals(24.0f, ColorMatrix.determinant(diag), 0.01f)
        assertEquals(1.0f, ColorMatrix.determinant(ColorMatrix.identity()), 0.01f)
    }

    @Test
    fun colorMatrix_rgbToRGB_roundtrip() {
        val m = ColorMatrix.rgbToRGB(ColorMatrix.REC709, ColorMatrix.REC2020)
        val inv = ColorMatrix.rgbToRGB(ColorMatrix.REC2020, ColorMatrix.REC709)
        val product = ColorMatrix.multiply(m, inv)
        val expected = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        vecEquals(expected, product.m, 0.02f)
    }
}
