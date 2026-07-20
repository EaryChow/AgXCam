package com.agx.camera.camera

import android.graphics.Rect
import android.hardware.camera2.params.MeteringRectangle
import android.os.Handler

class AutofocusController(
    private val cameraManager: Camera2Manager,
    private val handler: Handler
) {
    var isLocked = false
        private set

    private var currentRect: MeteringRectangle? = null

    /**
     * u, v: normalized coordinates (0..1) in the camera frame, before sensor
     * rotation — i.e. the output of MainActivity.viewToFrameCoords().
     */
    fun setFocusPoint(u: Float, v: Float, activeArray: Rect, sensorOrientation: Int, isFrontCamera: Boolean) {
        if (activeArray.width() <= 0 || activeArray.height() <= 0) return
        currentRect = normalizedToMeteringRect(u, v, activeArray, sensorOrientation, isFrontCamera)
        if (!isLocked) cameraManager.setMeteringRegion(currentRect)
    }

    fun lock() {
        isLocked = true
        cameraManager.lockFocusAndExposure()
    }

    fun unlock() {
        isLocked = false
        cameraManager.setMeteringRegion(currentRect) // resume region AF at same point
    }

    private fun normalizedToMeteringRect(
        u: Float, v: Float, activeArray: Rect, sensorOrientation: Int, isFrontCamera: Boolean
    ): MeteringRectangle {
        var nx = u
        var ny = v
        when (sensorOrientation) {          // frame -> sensor rotation
            90  -> { nx = v;      ny = 1f - u }
            180 -> { nx = 1f - u; ny = 1f - v }
            270 -> { nx = 1f - v; ny = u }
        }
        if (isFrontCamera) nx = 1f - nx      // undo preview mirroring

        val cx = activeArray.left + nx * activeArray.width()
        val cy = activeArray.top + ny * activeArray.height()
        val halfW = activeArray.width() * 0.05f   // region ~= 10% of frame width
        val halfH = activeArray.height() * 0.05f

        return MeteringRectangle(
            Rect(
                (cx - halfW).toInt().coerceAtLeast(activeArray.left),
                (cy - halfH).toInt().coerceAtLeast(activeArray.top),
                (cx + halfW).toInt().coerceAtMost(activeArray.right),
                (cy + halfH).toInt().coerceAtMost(activeArray.bottom)
            ),
            MeteringRectangle.METERING_WEIGHT_MAX
        )
    }
}