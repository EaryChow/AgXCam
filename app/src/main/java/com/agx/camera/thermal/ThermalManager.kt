package com.agx.camera.thermal

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.util.Log

class ThermalManager(context: Context) {

    enum class State {
        NORMAL,
        WARM,
        HOT,
        CRITICAL
    }

    private val appContext = context.applicationContext

    private var baselineFrameTimeMs = Float.MAX_VALUE
    private var frameCount = 0
    private var warmupComplete = false
    private var consecutiveSlowFrames = 0

    var currentState = State.NORMAL
        private set

    var isTorchActive = false

    var onStateChanged: ((State) -> Unit)? = null

    // Hysteresis: prevent rapid state transitions
    private var lastStateChangeTime = 0L
    private val minStateDurationMs = 10_000 // 10 seconds minimum before allowing state change

    val isWarmupComplete: Boolean get() = warmupComplete

    val isCaptureBlocked: Boolean
        get() {
            if (!warmupComplete) {
                return pollBatteryTemperature() >= 50.0f
            }
            return currentState == State.CRITICAL
        }

    val isPreviewReduced: Boolean
        get() = currentState == State.CRITICAL && consecutiveSlowFrames > 5

    fun reset() {
        baselineFrameTimeMs = Float.MAX_VALUE
        frameCount = 0
        warmupComplete = false
        consecutiveSlowFrames = 0
        currentState = State.NORMAL
    }

    fun onFrameRendered(frameTimeMs: Float) {
        frameCount++

        if (!warmupComplete) {
            if (frameTimeMs < baselineFrameTimeMs) {
                baselineFrameTimeMs = frameTimeMs
            }
            if (frameCount >= 60) {
                warmupComplete = true
                Log.d(TAG, "Thermal baseline established: ${baselineFrameTimeMs}ms")
            }
            return
        }

        if (baselineFrameTimeMs > 0 && frameTimeMs > baselineFrameTimeMs * 1.5f) {
            consecutiveSlowFrames++
        } else {
            consecutiveSlowFrames = 0
        }

        evaluateState()
    }

    fun pollBatteryTemperature(): Float {
        val intent = appContext.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tempTenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        return tempTenths / 10.0f
    }

    private fun evaluateState() {
        val batteryC = pollBatteryTemperature()

        val warmThreshold = if (isTorchActive) 42.0f else 48.0f
        val hotThreshold = if (isTorchActive) 47.0f else 52.0f
        val criticalThreshold = if (isTorchActive) 50.0f else 55.0f

        val newState = when {
            batteryC >= criticalThreshold -> State.CRITICAL
            batteryC >= hotThreshold -> State.HOT
            batteryC >= warmThreshold -> State.WARM
            consecutiveSlowFrames > 20 -> State.HOT
            consecutiveSlowFrames > 10 -> State.WARM
            else -> State.NORMAL
        }

        // Hysteresis: prevent rapid state transitions
        val now = System.currentTimeMillis()
        if (newState != currentState && now - lastStateChangeTime >= minStateDurationMs) {
            currentState = newState
            lastStateChangeTime = now
            Log.d(TAG, "Thermal state: $currentState (battery=${batteryC}°C, slowFrames=$consecutiveSlowFrames, torch=$isTorchActive)")
            onStateChanged?.invoke(currentState)
        }
    }

    companion object {
        private const val TAG = "ThermalManager"
    }
}
