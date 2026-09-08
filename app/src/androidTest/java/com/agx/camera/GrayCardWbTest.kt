package com.agx.camera

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.agx.camera.camera.GrayCardSampler
import com.agx.camera.color.ColorMatrix
import com.agx.camera.color.WhiteBalanceMath
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class GrayCardWbTest {

    private fun bitmapToNv21(bitmap: Bitmap): Triple<ByteBuffer, ByteBuffer, ByteBuffer> {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val yBuf = ByteBuffer.allocateDirect(w * h)
        val uBuf = ByteBuffer.allocateDirect((w / 2) * (h / 2))
        val vBuf = ByteBuffer.allocateDirect((w / 2) * (h / 2))

        for (row in 0 until h) {
            for (col in 0 until w) {
                val px = pixels[row * w + col]
                val r = Color.red(px) / 255.0
                val g = Color.green(px) / 255.0
                val b = Color.blue(px) / 255.0
                val y = (0.299 * r + 0.587 * g + 0.114 * b).coerceIn(0.0, 1.0)
                yBuf.put((y * 255).toInt().toByte())

                if (row % 2 == 0 && col % 2 == 0) {
                    val u = (-0.168736 * r - 0.331264 * g + 0.5 * b + 0.5).coerceIn(0.0, 1.0)
                    val v = (0.5 * r - 0.418688 * g - 0.081312 * b + 0.5).coerceIn(0.0, 1.0)
                    uBuf.put((u * 255).toInt().toByte())
                    vBuf.put((v * 255).toInt().toByte())
                }
            }
        }

        yBuf.position(0)
        uBuf.position(0)
        vBuf.position(0)
        return Triple(yBuf, uBuf, vBuf)
    }

    private fun grayBitmap(w: Int, h: Int, rScale: Float = 1.0f, gScale: Float = 1.0f, bScale: Float = 1.0f): Bitmap {
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val baseGray = 128
        val r = (baseGray * rScale).toInt().coerceIn(0, 255)
        val g = (baseGray * gScale).toInt().coerceIn(0, 255)
        val b = (baseGray * bScale).toInt().coerceIn(0, 255)
        val color = Color.rgb(r, g, b)
        val pixels = IntArray(w * h) { color }
        bitmap.setPixels(pixels, 0, w, 0, 0, w, h)
        return bitmap
    }

    @Test
    fun grayCardSampler_neutralGray_returnsUnityGains() {
        val w = 64; val h = 64
        val bitmap = grayBitmap(w, h)
        val (yBuf, uBuf, vBuf) = bitmapToNv21(bitmap)
        bitmap.recycle()

        val sampler = GrayCardSampler()
        var gains: GrayCardSampler.Gains? = null
        sampler.onSampleComplete = { g -> gains = g }
        sampler.activate()
        sampler.sample(yBuf, uBuf, vBuf, w, h, w / 2f, h / 2f, w, h)

        assertNotNull("callback fired", gains)
        assertEquals("gainR should be ~1.0", 1.0f, gains!!.gainR, 0.1f)
        assertEquals("gainB should be ~1.0", 1.0f, gains!!.gainB, 0.1f)
    }

    @Test
    fun grayCardSampler_tintedGray_computesCorrectGains() {
        val w = 64; val h = 64
        val bitmap = grayBitmap(w, h, rScale = 0.8f, gScale = 1.0f, bScale = 1.2f)
        val (yBuf, uBuf, vBuf) = bitmapToNv21(bitmap)
        bitmap.recycle()

        val sampler = GrayCardSampler()
        var gains: GrayCardSampler.Gains? = null
        sampler.onSampleComplete = { g -> gains = g }
        sampler.activate()
        sampler.sample(yBuf, uBuf, vBuf, w, h, w / 2f, h / 2f, w, h)

        assertNotNull("callback fired", gains)
        assertTrue("gainR should be > 1.0 (R is weak)", gains!!.gainR > 1.0f)
        assertTrue("gainB should be < 1.0 (B is strong)", gains!!.gainB < 1.0f)
    }

    @Test
    fun grayCardWB_neutralGray_pipeline_neutralOutput() {
        val w = 64; val h = 64
        val bitmap = grayBitmap(w, h)
        val (yBuf, uBuf, vBuf) = bitmapToNv21(bitmap)
        bitmap.recycle()

        val sampler = GrayCardSampler()
        var gains: GrayCardSampler.Gains? = null
        sampler.onSampleComplete = { g -> gains = g }
        sampler.activate()
        sampler.sample(yBuf, uBuf, vBuf, w, h, w / 2f, h / 2f, w, h)

        assertNotNull(gains)
        val grayCardMatrix = ColorMatrix.diagonal(gains!!.gainR, gains.gainG, gains.gainB)
        val wbMat = WhiteBalanceMath.buildGrayCardSceneLinearTo709(grayCardMatrix)
        val neutral = floatArrayOf(0.5f, 0.5f, 0.5f)
        val result = matMulVec(wbMat, neutral)

        val ratioRG = result[0] / result[1]
        val ratioBG = result[2] / result[1]
        assertEquals("R/G neutral", 1.0f, ratioRG, 0.15f)
        assertEquals("B/G neutral", 1.0f, ratioBG, 0.15f)
    }

    @Test
    fun grayCardWB_tintedGray_pipeline_correctsColor() {
        val w = 64; val h = 64
        val rScale = 0.7f; val gScale = 1.0f; val bScale = 1.3f
        val bitmap = grayBitmap(w, h, rScale = rScale, gScale = gScale, bScale = bScale)
        val (yBuf, uBuf, vBuf) = bitmapToNv21(bitmap)
        bitmap.recycle()

        val sampler = GrayCardSampler()
        var gains: GrayCardSampler.Gains? = null
        sampler.onSampleComplete = { g -> gains = g }
        sampler.activate()
        sampler.sample(yBuf, uBuf, vBuf, w, h, w / 2f, h / 2f, w, h)

        assertNotNull(gains)
        val grayCardMatrix = ColorMatrix.diagonal(gains!!.gainR, gains.gainG, gains.gainB)
        val wbMat = WhiteBalanceMath.buildGrayCardSceneLinearTo709(grayCardMatrix)
        val tinted = floatArrayOf(0.35f, 0.5f, 0.65f)
        val result = matMulVec(wbMat, tinted)

        val ratioRG = result[0] / result[1]
        val ratioBG = result[2] / result[1]
        assertTrue("R/G should be closer to 1.0 after correction", abs(ratioRG - 1.0f) < 0.05f)
        assertTrue("B/G should be closer to 1.0 after correction", abs(ratioBG - 1.0f) < 0.05f)
    }

    @Test
    fun kelvinWB_3000K_neutralizesItsIlluminant() {
        val wbMat = WhiteBalanceMath.buildSceneLinearTo709(
            kelvin = 3000f,
            calibrationMatrix = ColorMatrix.identity(),
            referenceToXyz = ColorMatrix.identity()
        )
        val (x, y) = WhiteBalanceMath.kelvinToXy(3000f)
        val sourceWhiteXyz = floatArrayOf(x / y, 1.0f, (1 - x - y) / y)
        val result = matMulVec(wbMat, sourceWhiteXyz)
        assertTrue("no NaN", result.all { !it.isNaN() })
        val spread = result.max() - result.min()
        assertTrue("3000K white should map near-neutral (spread $spread)", spread < 0.1f)
    }

    @Test
    fun kelvinWB_3000K_onNeutralFeed_boostsBlue() {
        val wbMat = WhiteBalanceMath.buildSceneLinearTo709(
            kelvin = 3000f,
            calibrationMatrix = ColorMatrix.identity(),
            referenceToXyz = ColorMatrix.identity()
        )
        val neutral = floatArrayOf(0.5f, 0.5f, 0.5f)
        val result = matMulVec(wbMat, neutral)
        assertTrue("assuming 3000K light on a neutral feed should cool (B > G)", result[2] > result[1])
    }

    @Test
    fun kelvinWB_6500K_neutral() {
        val wbMat = WhiteBalanceMath.buildSceneLinearTo709(
            kelvin = 6500f,
            calibrationMatrix = ColorMatrix.identity(),
            referenceToXyz = ColorMatrix.identity()
        )
        val neutral = floatArrayOf(0.5f, 0.5f, 0.5f)
        val result = matMulVec(wbMat, neutral)
        val ratioRG = result[0] / result[1]
        val ratioBG = result[2] / result[1]
        assertEquals("R/G at D65", 1.0f, ratioRG, 0.15f)
        assertEquals("B/G at D65", 1.0f, ratioBG, 0.15f)
    }

    @Test
    fun kelvinWB_extremeTemps_noNan() {
        for (temp in listOf(2000f, 3000f, 4000f, 5500f, 6500f, 8000f, 10000f)) {
            val mat = WhiteBalanceMath.buildSceneLinearTo709(
                kelvin = temp,
                calibrationMatrix = ColorMatrix.identity(),
                referenceToXyz = ColorMatrix.identity()
            )
            assertTrue("no NaN at ${temp}K", mat.m.all { !it.isNaN() })
            assertTrue("no Inf at ${temp}K", mat.m.all { !it.isInfinite() })
        }
    }

    @Test
    fun syntheticGenerator_grayCardBitmap_isUniform() {
        val gen = SyntheticBayerGenerator(64, 64)
        val bitmap = gen.generateGrayCard()
        val pixels = IntArray(64 * 64)
        bitmap.getPixels(pixels, 0, 64, 0, 0, 64, 64)
        bitmap.recycle()

        val first = pixels[0]
        for (px in pixels) {
            assertEquals("all pixels should be uniform", first, px)
        }
    }

    @Test
    fun syntheticGenerator_colorChart_hasEightColors() {
        val gen = SyntheticBayerGenerator(64, 64)
        val bitmap = gen.generateColorChart()
        val pixels = IntArray(64 * 64)
        bitmap.getPixels(pixels, 0, 64, 0, 0, 64, 64)
        bitmap.recycle()

        val uniqueColors = pixels.toSet().size
        assertTrue("color chart should have at least 8 unique colors", uniqueColors >= 8)
    }

    private fun matMulVec(m: ColorMatrix.Mat3, v: FloatArray): FloatArray = floatArrayOf(
        m.m[0] * v[0] + m.m[1] * v[1] + m.m[2] * v[2],
        m.m[3] * v[0] + m.m[4] * v[1] + m.m[5] * v[2],
        m.m[6] * v[0] + m.m[7] * v[1] + m.m[8] * v[2]
    )
}
