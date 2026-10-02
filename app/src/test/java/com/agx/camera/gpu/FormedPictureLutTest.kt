package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Formed-picture LUT: monotonicity, domain clamping and the entry cap.
 *
 * The table is a stand-in for full AgX per texel, so it does not have to be
 * bit-exact with the pipeline. It does have to be monotone and clamped: the
 * structure map indexes it by log2 of the linear input, and a non-monotone or
 * overshooting table would put the luminance band in the wrong place exactly
 * where it matters, at the toe.
 */
class FormedPictureLutTest {

    private val lut = FormedPictureLut()

    @Test
    fun defaultEntryCountFitsTheShaderUniform() {
        // The shader declares u_formed_lut[MAX_ENTRIES]; a table larger than that
        // would be a GL error and would silently lose its tail.
        assertTrue(
            "default table has ${lut.table.size} entries, shader allows ${FormedPictureLut.MAX_ENTRIES}",
            lut.table.size <= FormedPictureLut.MAX_ENTRIES
        )
        assertEquals(FormedPictureLut.MAX_ENTRIES, lut.table.size)
    }

    @Test
    fun tableIsMonotoneNonDecreasing() {
        for (i in 1 until lut.table.size) {
            assertTrue(
                "table[$i]=${lut.table[i]} must be >= table[${i - 1}]=${lut.table[i - 1]}",
                lut.table[i] >= lut.table[i - 1] - 1.0e-6f
            )
        }
    }

    @Test
    fun tableStaysInsideZeroToOne() {
        for (v in lut.table) {
            assertTrue("value $v below zero", v >= 0f)
            assertTrue("value $v above one", v <= 1.0001f)
        }
    }

    @Test
    fun formedLuminanceIsMonotone() {
        var prev = -1f
        var linear = 1.0e-4f
        while (linear <= 8.0f) {
            val y = lut.formedLuminance(linear)
            assertTrue("non-monotone at linear=$linear ($prev -> $y)", y >= prev - 1.0e-6f)
            prev = y
            linear *= 1.1f
        }
    }

    @Test
    fun outOfDomainInputClampsRatherThanExtrapolates() {
        assertEquals(lut.table[0], lut.formedLuminance(0f), 0f)
        assertEquals(lut.table[0], lut.formedLuminance(-1f), 0f)
        assertEquals(
            lut.table[lut.table.size - 1],
            lut.formedLuminance(1.0e9f),
            0f
        )
        assertEquals(lut.table[0], lut.formedLuminance(0f), 0f)
    }

    @Test
    fun nonFiniteInputReturnsTheFloor() {
        assertEquals(0f, lut.formedLuminance(Float.NaN), 0f)
        assertEquals(0f, lut.formedLuminance(Float.POSITIVE_INFINITY), 0f)
        assertEquals(0f, lut.formedLuminance(Float.NEGATIVE_INFINITY), 0f)
    }

    @Test
    fun floorIsTheCurveMinimumAndLeavesToeRoom() {
        // The floor is AgX's own black point, not a hard zero, so the
        // assertion is that it is the minimum and leaves most of the range
        // above it. Demanding a literal zero would mean rejecting the real
        // curve rather than the distillation.
        var minAt = 0
        for (i in lut.table.indices) {
            if (lut.table[i] < lut.table[minAt]) minAt = i
        }
        assertEquals("floor must be the table minimum", 0, minAt)
        assertTrue(
            "floor ${lut.table[0]} must leave headroom below the mid pivot",
            lut.table[0] < 0.5f * lut.table[lut.table.size / 2]
        )
    }

    @Test
    fun lookupMatchesTheTableItself() {
        // A stop that lands exactly on an entry must return that entry, with no
        // interpolation drift.
        val stop = lut.domainMinStop
        val linear = Math.pow(2.0, stop.toDouble()).toFloat()
        assertEquals(lut.table[0], lut.formedLuminance(linear), 1.0e-5f)
    }

    @Test
    fun exposureGainShiftsTheCurve() {
        val dark = FormedPictureLut(exposureEv = 0.0f)
        val bright = FormedPictureLut(exposureEv = 3.0f)
        val linear = 0.2f
        assertTrue(
            "a higher exposure gain must map to a higher formed luminance " +
                "(${bright.formedLuminance(linear)} vs ${dark.formedLuminance(linear)})",
            bright.formedLuminance(linear) >= dark.formedLuminance(linear)
        )
    }

    @Test
    fun spowfPreservesSignForNegativeBases() {
        // The sign is kept even for a fractional exponent. That matches the
        // AgX reference implementation this mirrors, and a negative input is
        // clamped to zero before the call in the real curve, so the behaviour
        // never reaches the table.
        assertEquals(-8f, FormedPictureLut.spowf(-2f, 3f), 1.0e-4f)
        assertEquals(0.125f, FormedPictureLut.spowf(2f, -3f), 1.0e-6f)
        assertEquals(0f, FormedPictureLut.spowf(0f, 2f), 0f)
    }

    @Test
    fun srgbOetfIsTheStandardCurve() {
        assertEquals(0f, FormedPictureLut.srgbOetf(0f), 0f)
        assertEquals(1f, FormedPictureLut.srgbOetf(1f), 1.0e-5f)
        // Below the linear segment the OETF is the 12.92x line.
        assertEquals(0.0031308f * 12.92f, FormedPictureLut.srgbOetf(0.0031308f), 1.0e-6f)
        assertTrue(FormedPictureLut.srgbOetf(0.5f) > 0.5f)
    }

    

    @Test(expected = IllegalArgumentException::class)
    fun tooFewEntriesIsRejected() {
        FormedPictureLut(entries = 4)
    }

    @Test(expected = IllegalArgumentException::class)
    fun tooManyEntriesIsRejected() {
        // Must fail loudly rather than upload past the declared uniform size.
        FormedPictureLut(entries = FormedPictureLut.MAX_ENTRIES + 1)
    }
}