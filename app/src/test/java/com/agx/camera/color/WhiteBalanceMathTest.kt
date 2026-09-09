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

    private fun matMulVec(m: ColorMatrix.Mat3, v: FloatArray): FloatArray = floatArrayOf(
        m.m[0] * v[0] + m.m[1] * v[1] + m.m[2] * v[2],
        m.m[3] * v[0] + m.m[4] * v[1] + m.m[5] * v[2],
        m.m[6] * v[0] + m.m[7] * v[1] + m.m[8] * v[2]
    )

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
    fun bradfordCAT_identity() {
        val d65 = Pair(ColorMatrix.D65_X, ColorMatrix.D65_Y)
        val cat = WhiteBalanceMath.chromaticAdaptationBradford(d65, d65)
        val expected = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        vecEquals(expected, cat.m, 0.01f)
    }

    @Test
    fun bradfordCAT_symmetric() {
        val d65 = Pair(ColorMatrix.D65_X, ColorMatrix.D65_Y)
        val warm = Pair(0.4476f, 0.4074f)
        val forward = WhiteBalanceMath.chromaticAdaptationBradford(d65, warm)
        val backward = WhiteBalanceMath.chromaticAdaptationBradford(warm, d65)
        val product = ColorMatrix.multiply(forward, backward)
        val identity = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        vecEquals(identity, product.m, 0.02f)
    }

    @Test
    fun kelvinSceneLinearTo709_d65_wellFormed() {
        val mat = WhiteBalanceMath.buildSceneLinearTo709(
            kelvin = 6500f,
            tint = 0f,
            calibrationMatrix = ColorMatrix.identity(),
            referenceToXyz = ColorMatrix.identity()
        )
        assertTrue("no NaN", mat.m.all { !it.isNaN() })
        assertTrue("no Infinity", mat.m.all { !it.isInfinite() })
        assertTrue("non-degenerate determinant", kotlin.math.abs(ColorMatrix.determinant(mat)) > 0.01f)
        val d65xyz = floatArrayOf(
            ColorMatrix.D65_X / ColorMatrix.D65_Y,
            1.0f,
            (1 - ColorMatrix.D65_X - ColorMatrix.D65_Y) / ColorMatrix.D65_Y
        )
        val result = matMulVec(mat, d65xyz)
        assertTrue("all channels positive", result.all { it > 0.0f })
        val maxRatio = result.max() / result.min()
        assertTrue("channel ratio < 3 at D65 (got $maxRatio)", maxRatio < 3.0f)
    }

    @Test
    fun kelvinToXy_defaultD65() {
        val (x, y) = WhiteBalanceMath.kelvinToXy(6300f, -14f)
        assertEquals("default x near D65", 0.3127f, x, 0.003f)
        assertEquals("default y near D65", 0.3290f, y, 0.003f)
        val cat = WhiteBalanceMath.chromaticAdaptationBradford(
            Pair(x, y), Pair(ColorMatrix.D65_X, ColorMatrix.D65_Y)
        )
        val identity = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        vecEquals(identity, cat.m, 0.01f)
    }

    @Test
    fun kelvinToXy_planckianAnchors() {
        val illuminantA = WhiteBalanceMath.kelvinToXy(2856f)
        assertEquals("Illuminant A x", 0.4476f, illuminantA.first, 0.01f)
        assertEquals("Illuminant A y", 0.4074f, illuminantA.second, 0.01f)

        val sixFive = WhiteBalanceMath.kelvinToXy(6500f)
        assertEquals("6500K x", 0.3135f, sixFive.first, 0.01f)
        assertEquals("6500K y", 0.3236f, sixFive.second, 0.01f)

        val twoK = WhiteBalanceMath.kelvinToXy(2000f)
        assertEquals("2000K x", 0.5267f, twoK.first, 0.01f)
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
    fun kelvinSceneLinearTo709_neutralizesItsOwnIlluminant() {
        for (temp in listOf(2200f, 3000f, 4000f, 5500f, 6500f, 8000f, 10000f)) {
            val mat = WhiteBalanceMath.buildSceneLinearTo709(
                kelvin = temp,
                calibrationMatrix = ColorMatrix.identity(),
                referenceToXyz = ColorMatrix.identity()
            )
            val (x, y) = WhiteBalanceMath.kelvinToXy(temp)
            val sourceWhiteXyz = floatArrayOf(x / y, 1.0f, (1 - x - y) / y)
            val result = matMulVec(mat, sourceWhiteXyz)
            assertTrue("no NaN at ${temp}K", result.all { !it.isNaN() })
            assertTrue("no Inf at ${temp}K", result.all { !it.isInfinite() })
            val spread = result.max() - result.min()
            assertTrue("illuminant at ${temp}K should map near-neutral (spread $spread)", spread < 0.1f)
            assertTrue("illuminant at ${temp}K should map near 1:1:1, got $result", result.all { it > 0.5f && it < 1.5f })
        }
    }

    @Test
    fun kelvinSceneLinearTo709_warmPreset_onNeutralFeed_boostsBlue() {
        val matWarm = WhiteBalanceMath.buildSceneLinearTo709(
            kelvin = 3000f,
            calibrationMatrix = ColorMatrix.identity(),
            referenceToXyz = ColorMatrix.identity()
        )
        val neutral = floatArrayOf(0.5f, 0.5f, 0.5f)
        val result = matMulVec(matWarm, neutral)
        assertTrue(
            "assuming warm (3000K) light on a neutral feed should cool the output (B > R), got B=${result[2]} R=${result[0]}",
            result[2] > result[0]
        )
    }

    @Test
    fun kelvinSceneLinearTo709_coolPreset_onNeutralFeed_boostsRed() {
        val matCool = WhiteBalanceMath.buildSceneLinearTo709(
            kelvin = 8000f,
            calibrationMatrix = ColorMatrix.identity(),
            referenceToXyz = ColorMatrix.identity()
        )
        val neutral = floatArrayOf(0.5f, 0.5f, 0.5f)
        val result = matMulVec(matCool, neutral)
        assertTrue(
            "assuming cool (8000K) light on a neutral feed should warm the output (R > B), got R=${result[0]} B=${result[2]}",
            result[0] > result[2]
        )
    }

    @Test
    fun kelvinSceneLinearTo709_extremeKelvin_noNan() {
        val mat = WhiteBalanceMath.buildSceneLinearTo709(
            kelvin = 2000f,
            calibrationMatrix = ColorMatrix.identity(),
            referenceToXyz = ColorMatrix.identity()
        )
        assertTrue("no NaN in matrix", mat.m.all { !it.isNaN() })
        assertTrue("no Infinity in matrix", mat.m.all { !it.isInfinite() })
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
