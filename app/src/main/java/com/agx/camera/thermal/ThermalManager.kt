package com.agx.camera.thermal

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import kotlin.math.exp

/**
 * Thermal gatekeeper for the camera app.
 *
 * Fixed thresholds (Celsius): WARM = 38, HOT = 40, CRITICAL = 42. These are
 * no longer shifted by the torch — high temperature is dangerous for the
 * battery regardless of what the camera is doing.
 *
 * ---------------------------------------------------------------------------
 * FUTURE VIDEO MODE (not yet implemented — this comment is a contract, not a
 * promise that the code below handles video):
 *  - Starting a recording while in WARM: the session is capped at 720p30.
 *  - Recording started in NORMAL that warms up mid-recording: drop the
 *    encoder bitrate and skip frames AT ENCODE TIME (30fps source -> 20fps
 *    encode inside a valid 30fps container). The file stays valid and the
 *    recording is never interrupted; the viewfinder goes blurry/slightly
 *    janky and a corner warning explains why.
 *  - Going CRITICAL mid-recording: force-stop and FINALIZE the recording
 *    (file properly saved), THEN shut down the RAW stream, showing the same
 *    message used for still captures.
 * ---------------------------------------------------------------------------
 *
 * Temperature detection uses an ASYMMETRIC EMA, not a symmetric one:
 *  - Rise is FAST (~7.5s effective window): critical protection must not lag
 *    while the temperature is climbing toward 42.
 *  - Fall is SLOW (~45s effective window): batteries cool slowly, and a slow
 *    decay is a built-in de-jitter (asymmetric hysteresis) so a value hovering
 *    around 37.9/38.0 cannot flap states.
 *
 * Two values are tracked:
 *  - the RAW battery temperature is what the UI displays ("right now it is
 *    38.4°C" — the user wants the honest current number);
 *  - the SMOOTHED value is what drives state transitions (stable level).
 *
 * The smoothing window must never be smaller than the real sample cadence of
 * ACTION_BATTERY_CHANGED (some devices only emit it every few seconds), so the
 * update cadence is auto-detected at runtime and the per-step alpha is derived
 * from the measured interval.
 */
class ThermalManager(context: Context) {

    enum class State {
        NORMAL,
        WARM,
        HOT,
        CRITICAL
    }

    enum class TorchShutdownReason {
        DURATION_LIMIT,
        HOT,
        CRITICAL
    }

    // --- Fixed thresholds (Celsius) ---
    val warmThreshold = 38.0f
    val hotThreshold = 40.0f
    val criticalThreshold = 42.0f

    // --- Throughput throttling knobs consumed by the camera/render layers ---
    // These are the ONLY knobs thermal policy may move: frame rate and output
    // resolution. Exposure semantics (shutter, ISO, compensation) are never
    // touched — in manual mode the user's exposure settings always win.
    // Thermal policy only moves throughput (fps, resolution), never imaging
    // semantics: when it gets hot the viewfinder gets laggy/blurry, but the
    // exposure always obeys the user.
    val throttledPreviewFps = 15
    val maxPreviewResolutionDim = 480
    val warmTorchDurationLimitMs = 3 * 60 * 1000L

    private val appContext = context.applicationContext

    @Volatile
    var currentState = State.NORMAL
        private set

    var isTorchActive = false
        set(value) {
            field = value
            torchEnabledElapsedRealtime = if (value) clockMsProvider() else 0L
            torchShutdownFired = false
        }

    /** Mockable clock for deterministic unit tests. */
    internal var clockMsProvider: () -> Long = { SystemClock.elapsedRealtime() }

    var onStateChanged: ((State) -> Unit)? = null
    var onTemperatureUpdate: ((displayC: Float, judgedC: Float) -> Unit)? = null
    var onTorchForcedOff: ((TorchShutdownReason) -> Unit)? = null

    val isCaptureBlocked: Boolean
        get() {
            evaluateState()
            return currentState == State.CRITICAL
        }

    val isPreviewReduced: Boolean
        get() {
            evaluateState()
            return currentState != State.NORMAL
        }

    /** Honest "right now" reading for the on-screen number. */
    val displayTemperatureC: Float
        get() = latestBatteryTemperatureC

    internal val judgedTemperatureC: Float
        get() = smoothedTemp ?: latestBatteryTemperatureC

    // --- Raw battery temperature cache, fed by the runtime receiver ---
    @Volatile
    var latestBatteryTemperatureC: Float = readBatteryTemperature()
        private set

    // --- Auto-detected ACTION_BATTERY_CHANGED cadence ---
    @Volatile
    internal var sampleIntervalMs = 2_000L
    private val cadenceSamples = ArrayList<Long>()
    @Volatile
    private var lastBatterySeenElapsed = 0L
    @Volatile
    private var lastBatteryValue = Float.NaN

    // --- Asymmetric EMA state ---
    @Volatile
    private var smoothedTemp: Float? = null
    /** Rise window (seconds cast): de-jitter strongly resisted on the way up. */
    private val tauRiseMs = 7_500f
    /** Fall window: slow decay acts as built-in hysteresis. */
    private val tauFallMs = 45_000f
    /** Extra margin below a threshold before dropping state after the EMA. */
    private val stepDownMarginC = 0.25f

    // --- Torch tracking ---
    @Volatile
    private var torchEnabledElapsedRealtime = 0L
    @Volatile
    private var torchShutdownFired = false

    private var pollHandlerThread: HandlerThread? = null
    private var pollHandler: Handler? = null
    @Volatile
    private var pollRunning = false

    private val batteryReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val tenths = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)
            val c = tenths / 10.0f
            latestBatteryTemperatureC = c
            measureCadence(c)
        }
    }

    /** Runtime battery source: a registered sticky broadcast, cheaper than polled queries. */
    private fun readBatteryTemperature(): Float {
        val intent = appContext.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tempTenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        return tempTenths / 10.0f
    }

    private fun measureCadence(c: Float) {
        val now = SystemClock.elapsedRealtime()
        if (lastBatterySeenElapsed != 0L && !lastBatteryValue.isNaN() && c != lastBatteryValue) {
            val dt = now - lastBatterySeenElapsed
            if (dt in 200..120_000) {
                cadenceSamples.add(dt)
                if (cadenceSamples.size > 16) cadenceSamples.removeAt(0)
                val sorted = cadenceSamples.sorted()
                sampleIntervalMs = sorted[sorted.size / 2].coerceIn(1_000L, 60_000L)
            }
        }
        lastBatterySeenElapsed = now
        lastBatteryValue = c
    }

    fun start() {
        if (pollRunning) return
        pollRunning = true
        // Sticky battery broadcast: delivers the current value immediately and
        // re-delivers whenever the battery actually changes. This is a system
        // broadcast, so no export flag is required on API 33+.
        appContext.registerReceiver(batteryReceiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val thread = HandlerThread("ThermalPoll").also { it.start() }
        pollHandlerThread = thread
        pollHandler = Handler(thread.looper)
        schedulePoll(0)
    }

    fun stop() {
        if (!pollRunning) return
        pollRunning = false
        runCatching { appContext.unregisterReceiver(batteryReceiver) }
        pollHandler?.removeCallbacksAndMessages(null)
        pollHandlerThread?.quitSafely()
        pollHandlerThread = null
    }

    private fun schedulePoll(delayMs: Long) {
        val h = pollHandler ?: return
        if (!pollRunning) return
        h.postDelayed({
            if (!pollRunning) return@postDelayed
            evaluateState()
            schedulePoll(POLL_INTERVAL_MS)
        }, delayMs)
    }

    fun reset() {
        smoothedTemp = null
        currentState = State.NORMAL
        cadenceSamples.clear()
        sampleIntervalMs = 2_000L
        lastBatterySeenElapsed = 0L
        lastBatteryValue = Float.NaN
        torchShutdownFired = false
    }

    fun onFrameRendered(frameTimeMs: Float) {
        // No-op: frame timing no longer triggers thermal state changes.
    }

    /** Asymmetric EMA. Internal for deterministic tests. */
    internal fun smoothTemperature(raw: Float): Float {
        val prev = smoothedTemp
        if (prev == null) {
            smoothedTemp = raw
            return raw
        }
        val dtSec = (sampleIntervalMs / 1000f).coerceIn(0.5f, 120f)
        val alphaRise = 1f - exp(-dtSec / (tauRiseMs / 1000f))
        val alphaFall = 1f - exp(-dtSec / (tauFallMs / 1000f))
        val newVal = if (raw > prev) {
            prev + (raw - prev) * alphaRise
        } else {
            prev + (raw - prev) * alphaFall
        }
        smoothedTemp = newVal
        return newVal
    }

    internal fun seedSmoothedTemperature(value: Float) {
        smoothedTemp = value
    }

    @Synchronized
    private fun evaluateState() {
        val raw = latestBatteryTemperatureC
        val judged = smoothTemperature(raw)

        val newState = when {
            judged >= criticalThreshold -> State.CRITICAL
            judged >= hotThreshold -> State.HOT
            judged >= warmThreshold -> State.WARM
            else -> State.NORMAL
        }

        var resolved = newState
        if (newState.ordinal < currentState.ordinal) {
            // Stepping down: the slow-fall EMA already de-jitters, but keep a
            // small margin below the line so a reading resting right at the
            // threshold doesn't flip state the instant it dips a fraction.
            val requiredBelow = when (currentState) {
                State.CRITICAL -> criticalThreshold - stepDownMarginC
                State.HOT -> hotThreshold - stepDownMarginC
                State.WARM -> warmThreshold - stepDownMarginC
                State.NORMAL -> Float.MAX_VALUE
            }
            if (!(judged < requiredBelow)) resolved = currentState
        }

        if (resolved != currentState) {
            val prev = currentState
            currentState = resolved
            Log.d(TAG, "Thermal state: $prev -> $currentState (battery=${judged}°C raw=${raw}°C torch=$isTorchActive)")
            onStateChanged?.invoke(currentState)
        }

        enforceTorchPolicy(currentState)

        onTemperatureUpdate?.invoke(raw, judged)
    }

    private fun enforceTorchPolicy(state: State) {
        if (!isTorchActive) return
        when (state) {
            State.WARM -> {
                // Torch allowed while warm, but only for a bounded duration.
                if (clockMsProvider() - torchEnabledElapsedRealtime >= warmTorchDurationLimitMs) {
                    fireTorchShutdown(TorchShutdownReason.DURATION_LIMIT)
                }
            }
            State.HOT -> {
                // Torch prohibited outright once hot; shut it down immediately.
                fireTorchShutdown(TorchShutdownReason.HOT)
            }
            State.CRITICAL -> {
                // Torch off FIRST, then the RAW stream. MainActivity additionally
                // teardowns the flash before suppressing the preview; the ordering
                // documented here is the contract that keeps this true.
                fireTorchShutdown(TorchShutdownReason.CRITICAL)
            }
            State.NORMAL -> {
                // Cool again: nothing to do; a fresh warm episode re-arms the
                // 3-minute timer via the isTorchActive setter when re-enabled.
            }
        }
    }

    private fun fireTorchShutdown(reason: TorchShutdownReason) {
        if (torchShutdownFired) return
        torchShutdownFired = true
        Log.d(TAG, "Torch force-off: $reason")
        onTorchForcedOff?.invoke(reason)
    }

    companion object {
        private const val TAG = "ThermalManager"
        private const val POLL_INTERVAL_MS = 1_000L
    }
}
