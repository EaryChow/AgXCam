package com.agx.camera.camera

enum class WhiteBalanceMode {
    AUTO,
    KELVIN,
    GRAY_CARD,
    DAYLIGHT,
    CLOUDY,
    INCANDESCENT,
    FLUORESCENT,
    TWILIGHT,
    SHADE
}

data class KelvinState(
    val kelvin: Float = 6300f,
    val tint: Float = -14f
)
