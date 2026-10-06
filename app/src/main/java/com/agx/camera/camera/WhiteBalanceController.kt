package com.agx.camera.camera

import com.agx.camera.color.ColorMatrix

enum class WhiteBalanceMode {
    AUTO,
    KELVIN,
    DAYLIGHT,
    CLOUDY,
    INCANDESCENT,
    FLUORESCENT,
    TWILIGHT,
    SHADE;

    // The scene illuminant each preset asserts, as a CIE 1931 2-degree chromaticity.
    // A preset is the user naming the light rather than the frame being measured for
    // it, so this is the one number the mode contributes; the gains whiten the
    // sensor's response to this light and the matrix adapts this light onto D65, and
    // both come from here. That makes a preset the same mechanism as AUTO with the
    // measurement replaced by a constant, and KELVIN the same mechanism with the
    // constant replaced by a parametrization (kelvinToXy).
    //
    // Grounded in the CIE 1931 2-degree table. Three of them follow the
    // illuminant mapping the DNG specification gives for these light sources,
    // so a HAL's preset gains and this chromaticity describe the same light:
    //   INCANDESCENT -> A, FLUORESCENT -> F2, DAYLIGHT -> D65.
    // The rest have no CIE standard; these are documented choices, each a defined
    // illuminant chosen to sit in the right direction from D65 (cooler/bluer, the way
    // these scenes actually read) rather than a number pulled from the air:
    //   CLOUDY  -> C, north-sky/overcast daylight, a shade cooler than D65.
    //   SHADE   -> D75, open shade lit by blue sky, cooler again.
    //   TWILIGHT-> a design point below D75, blue-hour light, which is off the
    //              Planckian locus and has no standard at all.
    // DAYLIGHT references the pipeline's own D65 constant rather than repeating the
    // CIE digits, so the daylight preset is exactly the identity adaptation against
    // the same white the renderer adapts to.
    fun sceneXy(): FloatArray? = when (this) {
        AUTO, KELVIN -> null
        DAYLIGHT -> floatArrayOf(ColorMatrix.D65_X, ColorMatrix.D65_Y)      // CIE D65
        INCANDESCENT -> floatArrayOf(0.44757f, 0.40745f)  // CIE A
        FLUORESCENT -> floatArrayOf(0.37208f, 0.37529f)   // CIE F2
        CLOUDY -> floatArrayOf(0.31006f, 0.31616f)        // CIE C
        SHADE -> floatArrayOf(0.29902f, 0.31485f)         // CIE D75
        TWILIGHT -> floatArrayOf(0.28f, 0.29f)            // design choice, blue hour
    }
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
