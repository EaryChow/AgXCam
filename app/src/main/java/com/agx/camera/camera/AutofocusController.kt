package com.agx.camera.camera

import android.hardware.camera2.*
import android.hardware.camera2.params.MeteringRectangle
import android.os.Handler
import android.util.Log
import android.util.SizeF

class AutofocusController(
    private val cameraManager: Camera2Manager,
    private val handler: Handler
) {

    enum class State {
        CONTINUOUS,
        FOCUSING,
        FOCUSED,
        FAILED
    }

    var state = State.CONTINUOUS
        private set

    var onStateChanged: ((State) -> Unit)? = null

    fun onTapToFocus(viewWidth: Int, viewHeight: Int, tapX: Float, tapY: Float) {
        if (state == State.FOCUSING) return

        val meteringRect = tapToMeteringRect(viewWidth, viewHeight, tapX, tapY)
        cameraManager.startAfAeLock(meteringRect, handler,
            onCaptureStarted = {
                state = State.FOCUSING
                onStateChanged?.invoke(State.FOCUSING)
            },
            onCaptureCompleted = { result ->
                val afState = result.get(CaptureResult.CONTROL_AF_STATE)
                when (afState) {
                    CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED -> {
                        state = State.FOCUSED
                        onStateChanged?.invoke(State.FOCUSED)
                    }
                    CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED -> {
                        state = State.FAILED
                        onStateChanged?.invoke(State.FAILED)
                    }
                    else -> {
                        state = State.FAILED
                        onStateChanged?.invoke(State.FAILED)
                    }
                }
            }
        )
    }

    fun revertToContinuous() {
        state = State.CONTINUOUS
        cameraManager.unlockAeAwb()
        cameraManager.startContinuousAf()
        onStateChanged?.invoke(State.CONTINUOUS)
    }

    private fun tapToMeteringRect(viewWidth: Int, viewHeight: Int, tapX: Float, tapY: Float): MeteringRectangle {
        val halfW = 150f.coerceAtMost(viewWidth / 4f)
        val halfH = 150f.coerceAtMost(viewHeight / 4f)
        val left = (tapX - halfW).toInt().coerceAtLeast(0)
        val top = (tapY - halfH).toInt().coerceAtLeast(0)
        val right = (tapX + halfW).toInt().coerceAtMost(viewWidth)
        val bottom = (tapY + halfH).toInt().coerceAtMost(viewHeight)
        return MeteringRectangle(
            android.graphics.Rect(left, top, right, bottom),
            MeteringRectangle.METERING_WEIGHT_MAX
        )
    }

    companion object {
        private const val TAG = "AutofocusController"
    }
}
