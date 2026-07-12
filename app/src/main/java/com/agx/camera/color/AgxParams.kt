package com.agx.camera.color

data class AgxParams(
    val contrast: Float = 2.4f,
    val toe: Float = 1.5f,
    val shoulder: Float = 1.5f,
    val rgbRotation: FloatArray = floatArrayOf(0.0373f, -0.0214f, -0.0532f),
    val purityAttenuation: FloatArray = floatArrayOf(32.9652f, 28.0513f, 12.4754f),
    val reverseRgbRotation: FloatArray = floatArrayOf(0f, 0f, 0f),
    val restorePurity: FloatArray = floatArrayOf(32.3174f, 28.3256f, 3.7433f),
    val tintingScale: Float = 0f,
    val tintingHue: Float = 0f,
    val nrStrength: Float = 0f,
    val usePreForPost: Boolean = true
) {
    fun toInsetParams(): AgxPrecomputer.InsetParams = AgxPrecomputer.InsetParams(
        rgbRotation = rgbRotation.copyOf(),
        purityAttenuation = purityAttenuation.copyOf(),
        usePreForPost = usePreForPost,
        reverseRgbRotation = reverseRgbRotation.copyOf(),
        restorePurity = restorePurity.copyOf(),
        tintingScale = tintingScale,
        tintingHue = tintingHue
    )

    companion object {
        fun fromInsetParams(p: AgxPrecomputer.InsetParams, contrast: Float = 2.4f, toe: Float = 1.5f, shoulder: Float = 1.5f, nrStrength: Float = 0f) = AgxParams(
            contrast = contrast,
            toe = toe,
            shoulder = shoulder,
            rgbRotation = p.rgbRotation.copyOf(),
            purityAttenuation = p.purityAttenuation.copyOf(),
            reverseRgbRotation = p.reverseRgbRotation.copyOf(),
            restorePurity = p.restorePurity.copyOf(),
            tintingScale = p.tintingScale,
            tintingHue = p.tintingHue,
            nrStrength = nrStrength,
            usePreForPost = p.usePreForPost
        )
    }
}
