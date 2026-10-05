package com.agx.camera.camera

enum class WhiteBalanceMode {
    AUTO,
    KELVIN,
    DAYLIGHT,
    CLOUDY,
    INCANDESCENT,
    FLUORESCENT,
    TWILIGHT,
    SHADE
}

data class KelvinState(
    val kelvin: Float = DEFAULT_KELVIN,
    val tint: Float = DEFAULT_TINT
) {
    companion object {
        // D65 sits off the Planckian locus. 6504 K is the CCT of the nearest
        // locus point and +9.6 tint (Duv = tint / 3000 = +0.0032) is the
        // perpendicular offset in CIE 1960 uv that lands on D65 itself, so the
        // Kelvin mode chromatic adaptation is the identity at these values.
        const val DEFAULT_KELVIN = 6504f
        const val DEFAULT_TINT = 9.6f

        // The Kelvin and tint SeekBars carry raw indices rather than the physical
        // quantities: kelvin_slider spans 2000..10000 K in 100 K steps (layout max
        // 80), tint_slider spans -100..+100 in steps of 1 (layout max 200).
        // Both directions are needed: the listener turns an index into a value and
        // syncAllSliders turns the stored value back into a handle position. The
        // defaults are off-grid, so the handle can only show the nearest index and
        // the readout is formatted from the stored value, not from the index.
        const val SLIDER_KELVIN_MIN = 2000f
        const val SLIDER_KELVIN_STEP = 100f
        const val SLIDER_KELVIN_LAST_INDEX = 80
        const val SLIDER_TINT_MIN = -100f
        const val SLIDER_TINT_STEP = 1f
        const val SLIDER_TINT_LAST_INDEX = 200

        fun kelvinToSliderIndex(kelvin: Float): Int =
            ((kelvin - SLIDER_KELVIN_MIN) / SLIDER_KELVIN_STEP)
                .toInt()
                .coerceIn(0, SLIDER_KELVIN_LAST_INDEX)

        fun sliderIndexToKelvin(index: Int): Float =
            SLIDER_KELVIN_MIN + index.coerceIn(0, SLIDER_KELVIN_LAST_INDEX) * SLIDER_KELVIN_STEP

        fun tintToSliderIndex(tint: Float): Int =
            ((tint - SLIDER_TINT_MIN) / SLIDER_TINT_STEP)
                .toInt()
                .coerceIn(0, SLIDER_TINT_LAST_INDEX)

        fun sliderIndexToTint(index: Int): Float =
            SLIDER_TINT_MIN + index.coerceIn(0, SLIDER_TINT_LAST_INDEX) * SLIDER_TINT_STEP
    }
}
