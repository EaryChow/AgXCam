package com.agx.camera.camera

import android.graphics.Rect
import android.hardware.camera2.params.MeteringRectangle
import android.os.Handler
import com.agx.camera.CrashLogger

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
    fun setFocusPoint(u: Float, v: Float, activeArray: Rect, isFrontCamera: Boolean, triggerScan: Boolean = true) {
        if (activeArray.width() <= 0 || activeArray.height() <= 0) return
        currentRect = normalizedToMeteringRect(u, v, activeArray, isFrontCamera)
        if (!isLocked) {
            val rect = currentRect ?: return
            if (triggerScan) cameraManager.setMeteringRegion(rect)
            else cameraManager.updateMeteringRegion(rect)
        }
    }

    /**
     * Live-drag: retarget the focus region without starting a new scan. The
     * region box follows the indicator on the HAL; a fresh scan is triggered on
     * release.
     */
    fun moveFocusPoint(u: Float, v: Float, activeArray: Rect, isFrontCamera: Boolean) {
        if (activeArray.width() <= 0 || activeArray.height() <= 0) return
        currentRect = normalizedToMeteringRect(u, v, activeArray, isFrontCamera)
        if (!isLocked) {
            val rect = currentRect ?: return
            cameraManager.moveMeteringRegion(rect)
        }
    }

    /**
     * Set the independent auto-exposure metering region (auto exposure mode).
     * Maps to the same 33% box geometry as the AF region.
     */
    fun setExposurePoint(u: Float, v: Float, activeArray: Rect, isFrontCamera: Boolean) {
        if (activeArray.width() <= 0 || activeArray.height() <= 0) return
        cameraManager.setAeRegion(normalizedToMeteringRect(u, v, activeArray, isFrontCamera))
    }

    /** Live-drag: retarget the independent AE metering region without a scan. */
    fun moveExposurePoint(u: Float, v: Float, activeArray: Rect, isFrontCamera: Boolean) {
        if (activeArray.width() <= 0 || activeArray.height() <= 0) return
        cameraManager.moveAeRegion(normalizedToMeteringRect(u, v, activeArray, isFrontCamera))
    }

    fun lock() {
        isLocked = true
        cameraManager.lockFocus()
    }

    fun unlock() {
        isLocked = false
        // Clear the manager-side lock state first, otherwise the preview re-arm
        // keeps AUTO + manual hold and the fresh region scan below gets overridden.
        cameraManager.unlockFocus()
        cameraManager.setMeteringRegion(currentRect) // resume region AF at same point
    }

    private fun normalizedToMeteringRect(
        u: Float, v: Float, activeArray: Rect, isFrontCamera: Boolean
    ): MeteringRectangle {
        var nx = u
        var ny = v
        // The renderer displays the sensor frame without rotating it (only the
        // GL V-flip + aspect crop), so the tap maps 1:1 to sensor coordinates.
        // For the front camera the preview is mirrored like the renderer does.
        if (isFrontCamera) nx = 1f - nx

        // AF geometry: the region is a constant 33% of the FULL
        // sensor active array, centered on the tapped pixel, independent of any
        // zoom. The HAL is always run full-frame during preview (SCALER_CROP_REGION
        // is only applied to still captures), so regions in active-array space
        // keep working regardless of the shutter zoom factor.
        val cx = activeArray.left + nx * activeArray.width()
        val cy = activeArray.top + ny * activeArray.height()

        val halfW = activeArray.width() * 0.165f
        val halfH = activeArray.height() * 0.165f
        val left = (cx - halfW).toInt().coerceAtLeast(activeArray.left)
        val top = (cy - halfH).toInt().coerceAtLeast(activeArray.top)
        val right = (cx + halfW).toInt().coerceAtMost(activeArray.right)
        val bottom = (cy + halfH).toInt().coerceAtMost(activeArray.bottom)

        CrashLogger.log(TAG, "Metering box: fullSensor box=${Rect(left, top, right, bottom)} tapSensor=(${cx.toInt()},${cy.toInt()})")

        return MeteringRectangle(
            Rect(left, top, right, bottom),
            500
        )
    }

    companion object {
        private const val TAG = "AutofocusController"
    }
}