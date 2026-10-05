package com.agx.camera.camera

import org.junit.Assert.assertEquals
import org.junit.Test

class KelvinSliderMappingTest {

    @Test
    fun kelvinIndex_endpointsMatchLayoutRange() {
        // Layout declares kelvin_slider android:max="80".
        assertEquals(0, KelvinState.kelvinToSliderIndex(2000f))
        assertEquals(80, KelvinState.kelvinToSliderIndex(10000f))
        assertEquals(2000f, KelvinState.sliderIndexToKelvin(0), 0.001f)
        assertEquals(10000f, KelvinState.sliderIndexToKelvin(80), 0.001f)
    }

    @Test
    fun tintIndex_endpointsMatchLayoutRange() {
        // Layout declares tint_slider android:max="200".
        assertEquals(0, KelvinState.tintToSliderIndex(-100f))
        assertEquals(200, KelvinState.tintToSliderIndex(100f))
        assertEquals(-100f, KelvinState.sliderIndexToTint(0), 0.001f)
        assertEquals(100f, KelvinState.sliderIndexToTint(200), 0.001f)
    }

    @Test
    fun tintIndex_zeroTintIsCentred() {
        assertEquals(100, KelvinState.tintToSliderIndex(0f))
    }

    @Test
    fun mapping_isExactInverseAcrossFullRange() {
        for (i in 0..80) {
            val kelvin = KelvinState.sliderIndexToKelvin(i)
            assertEquals("kelvin index $i", i, KelvinState.kelvinToSliderIndex(kelvin))
        }
        for (i in 0..200) {
            val tint = KelvinState.sliderIndexToTint(i)
            assertEquals("tint index $i", i, KelvinState.tintToSliderIndex(tint))
        }
    }

    @Test
    fun mapping_clampsOutOfRangeValues() {
        assertEquals(0, KelvinState.kelvinToSliderIndex(500f))
        assertEquals(0, KelvinState.kelvinToSliderIndex(-4000f))
        assertEquals(80, KelvinState.kelvinToSliderIndex(50000f))
        assertEquals(0, KelvinState.tintToSliderIndex(-999f))
        assertEquals(200, KelvinState.tintToSliderIndex(999f))
        assertEquals(10000f, KelvinState.sliderIndexToKelvin(999), 0.001f)
        assertEquals(-100f, KelvinState.sliderIndexToTint(-5), 0.001f)
    }

    @Test
    fun defaultState_handleSitsAtNearestIndex() {
        // DEFAULT_KELVIN and DEFAULT_TINT are off-grid, so the handle shows the
        // nearest index while the state keeps its exact value.
        assertEquals(45, KelvinState.kelvinToSliderIndex(KelvinState.DEFAULT_KELVIN))
        assertEquals(109, KelvinState.tintToSliderIndex(KelvinState.DEFAULT_TINT))
        assertEquals(6500f, KelvinState.sliderIndexToKelvin(45), 0.001f)
        assertEquals(9f, KelvinState.sliderIndexToTint(109), 0.001f)
    }

    @Test
    fun defaultState_survivesRoundTripThroughIndex() {
        // Dragging the handle to the default position snaps the value onto the
        // slider grid; the state is only exact until the user touches a slider.
        val kelvin = KelvinState.sliderIndexToKelvin(
            KelvinState.kelvinToSliderIndex(KelvinState.DEFAULT_KELVIN)
        )
        assertEquals(6500f, kelvin, 0.001f)
    }
}