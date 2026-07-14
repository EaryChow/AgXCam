package com.agx.camera.gpu

import com.agx.camera.camera.BayerPattern
import org.junit.Assert.*
import org.junit.Test

class BayerPatternTest {

    @Test
    fun rggb_phaseIndex() {
        assertEquals(0, BayerPattern.RGGB.phaseIndex(0, 0))
        assertEquals(1, BayerPattern.RGGB.phaseIndex(1, 0))
        assertEquals(2, BayerPattern.RGGB.phaseIndex(0, 1))
        assertEquals(3, BayerPattern.RGGB.phaseIndex(1, 1))
    }

    @Test
    fun rggb_colorMap() {
        val map = BayerPattern.RGGB.colorMap()
        assertArrayEquals(intArrayOf(0, 1, 1, 2), map)
    }

    @Test
    fun grbg_colorMap() {
        val map = BayerPattern.GRBG.colorMap()
        assertArrayEquals(intArrayOf(1, 0, 2, 1), map)
    }

    @Test
    fun gbrg_colorMap() {
        val map = BayerPattern.GBRG.colorMap()
        assertArrayEquals(intArrayOf(1, 2, 0, 1), map)
    }

    @Test
    fun bggr_colorMap() {
        val map = BayerPattern.BGGR.colorMap()
        assertArrayEquals(intArrayOf(2, 1, 1, 0), map)
    }

    @Test
    fun rggb_phaseIndex_is_consistent_with_colorMap() {
        for (x in 0..1) {
            for (y in 0..1) {
                val phase = BayerPattern.RGGB.phaseIndex(x, y)
                val color = BayerPattern.RGGB.colorMap()[phase]
                assertTrue("color should be 0, 1, or 2", color in 0..2)
            }
        }
    }

    @Test
    fun bggr_phaseIndex_is_consistent_with_colorMap() {
        for (x in 0..1) {
            for (y in 0..1) {
                val phase = BayerPattern.BGGR.phaseIndex(x, y)
                val color = BayerPattern.BGGR.colorMap()[phase]
                assertTrue("color should be 0, 1, or 2", color in 0..2)
            }
        }
    }

    @Test
    fun fromString_valid() {
        assertEquals(BayerPattern.RGGB, BayerPattern.fromString("bayer_rggb"))
        assertEquals(BayerPattern.GRBG, BayerPattern.fromString("bayer_grbg"))
        assertEquals(BayerPattern.GBRG, BayerPattern.fromString("bayer_gbrg"))
        assertEquals(BayerPattern.BGGR, BayerPattern.fromString("bayer_bggr"))
    }

    @Test(expected = IllegalArgumentException::class)
    fun fromString_invalid() {
        BayerPattern.fromString("bayer_unknown")
    }

    @Test
    fun fromAndroidConstant_valid() {
        assertEquals(BayerPattern.RGGB, BayerPattern.fromAndroidConstant(0))
        assertEquals(BayerPattern.GRBG, BayerPattern.fromAndroidConstant(1))
        assertEquals(BayerPattern.GBRG, BayerPattern.fromAndroidConstant(2))
        assertEquals(BayerPattern.BGGR, BayerPattern.fromAndroidConstant(3))
    }

    @Test
    fun phaseIndex_allPatterns_cover_all_positions() {
        for (pattern in BayerPattern.entries) {
            val phases = mutableSetOf<Int>()
            for (y in 0..3) {
                for (x in 0..3) {
                    phases.add(pattern.phaseIndex(x, y))
                }
            }
            assertEquals("Pattern $pattern should cover all 4 phases", 4, phases.size)
            assertTrue("Phases should be 0..3", phases.containsAll(setOf(0, 1, 2, 3)))
        }
    }
}
