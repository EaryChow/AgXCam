package com.agx.camera.camera

import org.junit.Assert.assertEquals
import org.junit.Test

class LensPreviewSizeTest {

    private val lens5Ladder = listOf(
        320 to 240,
        640 to 480,
        800 to 600,
        1920 to 1440,
        4080 to 3060,
        4080 to 3072
    )

    @Test
    fun inheritedSensorAspectDoesNotSelectFullSensorFor240p() {
        // Lens 5's 4080x3072 mode is 1.328125, while its resolution option
        // can arrive with that exact aspect. The 240p preview budget must
        // still choose 320x240, not the full 12.5 MP sensor mode.
        val picked = choosePreviewSize(lens5Ladder, 102_240L, 4080f / 3072f)
        assertEquals(320, picked?.first)
        assertEquals(240, picked?.second)
    }

    @Test
    fun nearMatchingAspectStillRespectsThe1080pBudget() {
        val picked = choosePreviewSize(lens5Ladder, 2_073_600L, 4080f / 3072f)
        assertEquals(1920, picked?.first)
        assertEquals(1440, picked?.second)
    }

    @Test
    fun aDifferentAspectFamilyIsNotAbsorbedByTheTolerance() {
        val with169 = listOf(320 to 240, 1600 to 1200, 1920 to 1080)
        val picked = choosePreviewSize(with169, 2_073_600L, 16f / 9f)
        assertEquals(1920, picked?.first)
        assertEquals(1080, picked?.second)
    }
}
