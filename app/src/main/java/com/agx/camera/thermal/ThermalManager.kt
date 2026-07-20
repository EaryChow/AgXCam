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

    var currentState = State.NORMAL
        private set

    var isTorchActive = false

    var onStateChanged: ((State) -> Unit)? = null

    // Hysteresis: prevent rapid state transitions
    private var lastStateChangeTime = 0L
    private val minStateDurationMs = 10_000 // 10 seconds minimum before allowing state change

    // Temperature smoothing (EMA)
    private var smoothedTemp: Float? = null
    private val tempEmaAlpha = 0.3f // lower = smoother, 0.3 = ~6 sample window

    // Return hysteresis: temp must drop this far below a threshold before stepping down
    private val returnHysteresisC = 3.0f

    val isCaptureBlocked: Boolean
        get() {
            evaluateState()
            return currentState == State.CRITICAL
        }

    val isPreviewReduced: Boolean
        get() = currentState == State.CRITICAL

    fun reset() {
        smoothedTemp = null
        currentState = State.NORMAL
        lastStateChangeTime = 0L
    }

    fun onFrameRendered(frameTimeMs: Float) {
        // No-op: frame timing no longer triggers thermal state changes
    }

    fun pollBatteryTemperature(): Float {
        val intent = appContext.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tempTenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        return tempTenths / 10.0f
    }

    private fun smoothTemperature(raw: Float): Float {
        val prev = smoothedTemp
        return if (prev == null) {
            smoothedTemp = raw
            raw
        } else {
            val smoothed = prev + tempEmaAlpha * (raw - prev)
            smoothedTemp = smoothed
            smoothed
        }
    }

    private fun evaluateState() {
        val rawBatteryC = pollBatteryTemperature()
        val batteryC = smoothTemperature(rawBatteryC)

        val warmThreshold = if (isTorchActive) 42.0f else 48.0f
        val hotThreshold = if (isTorchActive) 47.0f else 52.0f
        val criticalThreshold = if (isTorchActive) 50.0f else 55.0f

        val newState = when {
            batteryC >= criticalThreshold -> State.CRITICAL
            batteryC >= hotThreshold -> State.HOT
            batteryC >= warmThreshold -> State.WARM
            else -> State.NORMAL
        }

        // Hysteresis: prevent rapid state transitions
        val now = System.currentTimeMillis()
        if (now - lastStateChangeTime < minStateDurationMs) return

        val resolved = if (newState.ordinal < currentState.ordinal) {
            // Stepping down: require temp to drop below threshold minus hysteresis
            val requiredDrop = when (currentState) {
                State.CRITICAL -> criticalThreshold - returnHysteresisC
                State.HOT -> hotThreshold - returnHysteresisC
                State.WARM -> warmThreshold - returnHysteresisC
                State.NORMAL -> 0f
            }
            if (batteryC < requiredDrop) newState else currentState
        } else {
            newState
        }

        if (resolved != currentState) {
            currentState = resolved
            lastStateChangeTime = now
            Log.d(TAG, "Thermal state: $currentState (battery=${batteryC}°C raw=${rawBatteryC}°C, torch=$isTorchActive)")
            onStateChanged?.invoke(currentState)
        }
    }

    companion object {
        private const val TAG = "ThermalManager"
    }
}
