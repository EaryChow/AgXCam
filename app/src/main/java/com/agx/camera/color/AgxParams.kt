package com.agx.camera.color

data class AgxParams(
    val contrast: Float = 2.4f,
    val toe: Float = 1.5f,
    val shoulder: Float = 1.5f,
    val rotation: FloatArray = floatArrayOf(0.0373f, -0.0214f, -0.0532f),
    val attenuation: FloatArray = floatArrayOf(32.9652f, 28.0513f, 12.4754f),
    val reverseRotation: FloatArray = floatArrayOf(0f, 0f, 0f),
    val purityBoost: FloatArray = floatArrayOf(32.3174f, 28.3256f, 3.7433f),
    val tintingScale: Float = 0f,
    val tintingHue: Float = 0f,
    val nrStrength: Float = 0f,
    val usePreForPost: Boolean = true
) {
    fun toInsetParams(): AgxPrecomputer.InsetParams = AgxPrecomputer.InsetParams(
        rotation = rotation.copyOf(),
        attenuation = attenuation.copyOf(),
        usePreForPost = usePreForPost,
        reverseRotation = reverseRotation.copyOf(),
        purityBoost = purityBoost.copyOf(),
        tintingScale = tintingScale,
        tintingHue = tintingHue
    )

    companion object {
        fun fromInsetParams(p: AgxPrecomputer.InsetParams, contrast: Float = 2.4f, toe: Float = 1.5f, shoulder: Float = 1.5f, nrStrength: Float = 0f) = AgxParams(
            contrast = contrast,
            toe = toe,
            shoulder = shoulder,
            rotation = p.rotation.copyOf(),
            attenuation = p.attenuation.copyOf(),
            reverseRotation = p.reverseRotation.copyOf(),
            purityBoost = p.purityBoost.copyOf(),
            tintingScale = p.tintingScale,
            tintingHue = p.tintingHue,
            nrStrength = nrStrength,
            usePreForPost = p.usePreForPost
        )
    }
}
