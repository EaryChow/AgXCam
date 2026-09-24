package com.agx.camera.camera

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Verifies the CPU-side stage of lens shading correction: the
 * per-channel gain grids are packed into an RGBA16F half-float texture, the
 * channels are permuted to match the active CFA so each raw phase picks up its
 * own plane's gain, and rows are vertically flipped (Android's map is row 0 at
 * the top; glTexImage2D places the first row at the bottom).
 */
class LensShadingMapTest {

    @Test
    fun halfFloat_encodesOneAs0x3C00() {
        val data = makeData(1, 1) { _, _, _ -> 1.0f }
        val pixels = data.toRgba16fFlipped(BayerPattern.RGGB)
        for (p in pixels) {
            assertEquals(0x3C00.toShort(), p)
        }
    }

    @Test
    fun halfFloat_roundTripKnownValues() {
        for (v in floatArrayOf(0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 0.25f)) {
            assertEquals(v, halfToFloat(floatToHalf(v)), 1e-3f)
        }
    }

    @Test
    fun rggb_keepsChannelOrder_andFlipsRows() {
        val w = 2
        val h = 2
        // Distinct value per (channel, row, col): R=1.., Gr, Gb, B picked up by
        // each texel, so a wrong channel order or a missing flip is obvious.
        val data = makeData(w, h) { ch, r, c -> (ch * 100 + r * 10 + c).toFloat() + 1f }
        val pixels = data.toRgba16fFlipped(BayerPattern.RGGB)

        // RGGB permutation is identity [R, Gr, Gb, B]. Output row 0 must hold
        // sensor row h-1 and so on.
        for (outRow in 0 until h) {
            val srcRow = h - 1 - outRow
            assertPixel(
                pixels, w, outRow, 0,
                floatArrayOf(
                    data.rGains[srcRow][0],
                    data.grGains[srcRow][0],
                    data.gbGains[srcRow][0],
                    data.bGains[srcRow][0]
                )
            )
        }
    }

    @Test
    fun bggr_permutesChannels() {
        val w = 1
        val h = 1
        val data = makeData(w, h) { ch, _, _ -> ch * 10f + 1f } // R=1, Gr=11, Gb=21, B=31
        val pixels = data.toRgba16fFlipped(BayerPattern.BGGR)
        val out = FloatArray(4) { halfToFloat(pixels[it]) }
        // BGGR swaps RGGB: B->R, Gb->G, Gr->B, R->A.
        assertEquals(31f, out[0], 1e-3f)
        assertEquals(21f, out[1], 1e-3f)
        assertEquals(11f, out[2], 1e-3f)
        assertEquals(1f, out[3], 1e-3f)
    }

    @Test
    fun grbg_permutesChannels() {
        val w = 1
        val h = 1
        val data = makeData(w, h) { ch, _, _ -> ch * 10f + 1f }
        val pixels = data.toRgba16fFlipped(BayerPattern.GRBG)
        val out = FloatArray(4) { halfToFloat(pixels[it]) }
        // GRBG: Gr->R, R->G, B->B, Gb->A.
        assertEquals(11f, out[0], 1e-3f)
        assertEquals(1f, out[1], 1e-3f)
        assertEquals(31f, out[2], 1e-3f)
        assertEquals(21f, out[3], 1e-3f)
    }

    @Test
    fun gbrg_permutesChannels() {
        val w = 1
        val h = 1
        val data = makeData(w, h) { ch, _, _ -> ch * 10f + 1f }
        val pixels = data.toRgba16fFlipped(BayerPattern.GBRG)
        val out = FloatArray(4) { halfToFloat(pixels[it]) }
        // GBRG: Gr->R, B->G, R->B, Gb->A.
        assertEquals(11f, out[0], 1e-3f)
        assertEquals(31f, out[1], 1e-3f)
        assertEquals(1f, out[2], 1e-3f)
        assertEquals(21f, out[3], 1e-3f)
    }

    @Test
    fun imageData_isRowOrderedAfterFlip() {
        val w = 1
        val h = 3
        val data = makeData(w, h) { ch, r, _ -> (ch * 100 + r).toFloat() + 1f }
        val pixels = data.toRgba16fFlipped(BayerPattern.RGGB)
        // Four floats (channels) per pixel, hog pixels grouped per row.
        for (outRow in 0 until h) {
            val srcRow = h - 1 - outRow
            val base = outRow * w * 4
            assertEquals(data.rGains[srcRow][0], halfToFloat(pixels[base]), 1e-3f)
        }
    }

    private fun makeData(
        w: Int,
        h: Int,
        fill: (channel: Int, row: Int, col: Int) -> Float
    ): LensShadingData {
        val cols = { ch: Int ->
            Array(h) { row -> FloatArray(w) { col -> fill(ch, row, col) } }
        }
        return LensShadingData(
            rGains = cols(0),
            grGains = cols(1),
            gbGains = cols(2),
            bGains = cols(3),
            width = w,
            height = h,
            available = true
        )
    }

    private fun assertPixel(pixels: ShortArray, width: Int, row: Int, col: Int, expected: FloatArray) {
        val base = (row * width + col) * 4
        for (ch in 0 until 4) {
            assertEquals(expected[ch], halfToFloat(pixels[base + ch]), 1e-3f)
        }
    }

    private fun floatToHalf(f: Float): Short {
        val bits = java.lang.Float.floatToRawIntBits(f)
        val sign = (bits ushr 16) and 0x8000
        var exp = ((bits ushr 23) and 0xFF) - 127 + 15
        var mantissa = bits and 0x7FFFFF
        if (exp <= 0) {
            exp = 0
            mantissa = 0
        } else if (exp >= 31) {
            exp = 31
            mantissa = 0
        } else {
            mantissa = mantissa shr 13
        }
        return (sign or (exp shl 10) or mantissa).toShort()
    }

    private fun halfToFloat(h: Short): Float {
        val bits = h.toInt() and 0xFFFF
        val sign = if ((bits and 0x8000) != 0) -1f else 1f
        val exponent = (bits shr 10) and 0x1F
        val mantissa = (bits and 0x3FF).toFloat() / 0x400
        if (exponent == 0) return sign * mantissa * (1f / 0x8000)
        if (exponent == 31) return if (mantissa == 0f) sign * Float.POSITIVE_INFINITY else Float.NaN
        return sign * (1f + mantissa) * Math.pow(2.0, (exponent - 15).toDouble()).toFloat()
    }
}