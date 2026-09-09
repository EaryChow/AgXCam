package com.agx.camera.camera

import android.util.Log

class ShutterController {

    enum class State {
        IDLE,
        PRECAPTURE,
        CAPTURING
    }

    var state = State.IDLE
        private set

    var onStateChanged: ((State) -> Unit)? = null

    fun onPrecaptureStarted() {
        transitionTo(State.PRECAPTURE)
    }

    fun onCaptureSubmitted() {
        transitionTo(State.CAPTURING)
    }

    fun onCaptureComplete() {
        transitionTo(State.IDLE)
    }

    fun onCaptureFailed() {
        transitionTo(State.IDLE)
    }

    private fun transitionTo(newState: State) {
        if (state == newState) return
        state = newState
        Log.d(TAG, "Shutter state: $newState")
        onStateChanged?.invoke(newState)
    }

    companion object {
        private const val TAG = "ShutterController"
    }
}