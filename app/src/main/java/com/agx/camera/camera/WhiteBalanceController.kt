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
    val kelvin: Float = 5500f,
    val tint: Float = 0f
)
