package com.agx.camera.camera

/** Shared preview-resolution targets, expressed as total pixel counts. */
object PreviewResolution {
    const val P240 = 426 * 240
    const val P360 = 640 * 360
    const val P480 = 854 * 480
    const val P720 = 1280 * 720
    const val P1080 = 1920 * 1080

    val OPTIONS: List<Int> = listOf(P240, P360, P480, P720, P1080)
    const val DEFAULT = P480
}
