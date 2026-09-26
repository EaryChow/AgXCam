package com.agx.camera.color

data class AgxParams(
    val contrast: Float = 3.0f,
    val toe: Float = 1.5f,
    val shoulder: Float = 2.5f,
    val middleGray: Float = 18f,
    val rotation: FloatArray = floatArrayOf(0.1222f, -0.1222f, -0.053f),
    val attenuation: FloatArray = floatArrayOf(45.0f, 50.0f, 30.0f),
    val reverseRotation: FloatArray = floatArrayOf(0.1222f, -0.1222f, -0.053f),
    val purityBoost: FloatArray = floatArrayOf(20.0f, 30.0f, 4.0f),
    val tintingScale: Float = 0f,
    val tintingHue: Float = 0f,
    val vibrance: Float = 0.6f,
    val nrStrength: Float = 0f,
    val useRotationForReverse: Boolean = true,
    val useAttenuationForBoost: Boolean = false
) {
    fun toInsetParams(): AgxPrecomputer.InsetParams = AgxPrecomputer.InsetParams(
        rotation = rotation.copyOf(),
        attenuation = attenuation.copyOf(),
        useRotationForReverse = useRotationForReverse,
        useAttenuationForBoost = useAttenuationForBoost,
        reverseRotation = reverseRotation.copyOf(),
        purityBoost = purityBoost.copyOf(),
        tintingScale = tintingScale,
        tintingHue = tintingHue
    )

    companion object {
        fun fromInsetParams(p: AgxPrecomputer.InsetParams, contrast: Float = 2.4f, toe: Float = 1.5f, shoulder: Float = 1.5f, middleGray: Float = 18f, vibrance: Float = 0.6f, nrStrength: Float = 0f) = AgxParams(
            contrast = contrast,
            toe = toe,
            shoulder = shoulder,
            middleGray = middleGray,
            vibrance = vibrance,
            rotation = p.rotation.copyOf(),
            attenuation = p.attenuation.copyOf(),
            reverseRotation = p.reverseRotation.copyOf(),
            purityBoost = p.purityBoost.copyOf(),
            tintingScale = p.tintingScale,
            tintingHue = p.tintingHue,
            nrStrength = nrStrength,
            useRotationForReverse = p.useRotationForReverse,
            useAttenuationForBoost = p.useAttenuationForBoost
        )
    }
}
