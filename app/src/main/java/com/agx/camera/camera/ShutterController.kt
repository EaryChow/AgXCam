package com.agx.camera.camera

import android.os.Handler
import android.os.SystemClock
import android.util.Log

class ShutterController(private val handler: Handler) {

    enum class State {
        IDLE,
        PRECAPTURE,
        CAPTURING,
        COOLDOWN
    }

    var state = State.IDLE
        private set

    var onStateChanged: ((State) -> Unit)? = null
    var onCooldownTick: ((Int) -> Unit)? = null

    private var cooldownEndTime = 0L

    private val cooldownChecker = object : Runnable {
        override fun run() {
            if (state != State.COOLDOWN) return
            val remaining = ((cooldownEndTime - SystemClock.elapsedRealtime()) / 1000).toInt().coerceAtLeast(0)
            onCooldownTick?.invoke(remaining)
            if (remaining > 0) {
                handler.postDelayed(this, 500)
            } else {
                transitionTo(State.IDLE)
            }
        }
    }

    fun onPrecaptureStarted() {
        transitionTo(State.PRECAPTURE)
    }

    fun onCaptureSubmitted() {
        transitionTo(State.CAPTURING)
    }

    fun onCaptureComplete() {
        startCooldown()
    }

    fun onCaptureFailed() {
        startCooldown()
    }

    private fun startCooldown() {
        cooldownEndTime = SystemClock.elapsedRealtime() + COOLDOWN_MS
        transitionTo(State.COOLDOWN)
        handler.postDelayed(cooldownChecker, 500)
    }

    fun cancelCooldown() {
        handler.removeCallbacks(cooldownChecker)
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
        private const val COOLDOWN_MS = 3000L
    }
}
