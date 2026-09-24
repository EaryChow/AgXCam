package com.agx.camera.phase1.week7

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import com.agx.camera.thermal.ThermalManager
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.Mockito.*

class ThermalThresholdTest {

    private lateinit var thermalManager: ThermalManager

    private fun createContextWithBatteryTemp(tempTenths: Int): Context {
        val context = mock(Context::class.java)
        val appContext = mock(Context::class.java)
        `when`(context.applicationContext).thenReturn(appContext)

        val intent = mock(Intent::class.java)
        `when`(intent.getIntExtra(eq(BatteryManager.EXTRA_TEMPERATURE), anyInt())).thenReturn(tempTenths)
        `when`(appContext.registerReceiver(isNull(), any(IntentFilter::class.java))).thenReturn(intent)

        return context
    }

    @Before
    fun setUp() {
        val context = createContextWithBatteryTemp(250)
        thermalManager = ThermalManager(context)
    }

    @Test
    fun initialState_isNormal() {
        assertEquals(ThermalManager.State.NORMAL, thermalManager.currentState)
    }

    @Test
    fun transition_toWarm_at38C() {
        val context = createContextWithBatteryTemp(380)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked // triggers evaluateState
        assertEquals(ThermalManager.State.WARM, tm.currentState)
    }

    @Test
    fun transition_toHot_at40C() {
        val context = createContextWithBatteryTemp(400)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.HOT, tm.currentState)
    }

    @Test
    fun transition_toCritical_at42C() {
        val context = createContextWithBatteryTemp(420)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.CRITICAL, tm.currentState)
    }

    // Torch no longer shifts the thresholds: 38/40/42 apply regardless.
    @Test
    fun torchMode_thresholdsAreNotShifted_warm() {
        val context = createContextWithBatteryTemp(380)
        val tm = ThermalManager(context)
        tm.isTorchActive = true
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.WARM, tm.currentState)
    }

    @Test
    fun torchMode_thresholdsAreNotShifted_hot() {
        val context = createContextWithBatteryTemp(400)
        val tm = ThermalManager(context)
        tm.isTorchActive = true
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.HOT, tm.currentState)
    }

    @Test
    fun torchMode_thresholdsAreNotShifted_critical() {
        val context = createContextWithBatteryTemp(420)
        val tm = ThermalManager(context)
        tm.isTorchActive = true
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.CRITICAL, tm.currentState)
    }

    @Test
    fun normalMode_noTransitionBelowThresholds() {
        val context = createContextWithBatteryTemp(350)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.NORMAL, tm.currentState)
    }

    @Test
    fun reset_clearsAllState() {
        val context = createContextWithBatteryTemp(420)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.CRITICAL, tm.currentState)

        tm.reset()

        assertEquals(ThermalManager.State.NORMAL, tm.currentState)
    }

    @Test
    fun isCaptureBlocked_criticalState() {
        val context = createContextWithBatteryTemp(420)
        val tm = ThermalManager(context)
        assertTrue(tm.isCaptureBlocked)
    }

    @Test
    fun isCaptureBlocked_hotState() {
        val context = createContextWithBatteryTemp(400)
        val tm = ThermalManager(context)
        assertFalse(tm.isCaptureBlocked)
    }

    @Test
    fun isCaptureBlocked_normalState() {
        val context = createContextWithBatteryTemp(250)
        val tm = ThermalManager(context)
        assertFalse(tm.isCaptureBlocked)
    }

    @Test
    fun isPreviewReduced_criticalState() {
        val context = createContextWithBatteryTemp(420)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertTrue(tm.isPreviewReduced)
    }

    @Test
    fun isPreviewReduced_hotState() {
        val context = createContextWithBatteryTemp(400)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertTrue(tm.isPreviewReduced)
    }

    @Test
    fun isPreviewReduced_warmState() {
        val context = createContextWithBatteryTemp(380)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertTrue(tm.isPreviewReduced)
    }

    @Test
    fun onStateChanged_callback_fires() {
        val context = createContextWithBatteryTemp(420)
        val tm = ThermalManager(context)

        var callbackFired = false
        var receivedState: ThermalManager.State? = null
        tm.onStateChanged = { state ->
            callbackFired = true
            receivedState = state
        }

        tm.isCaptureBlocked

        assertTrue(callbackFired)
        assertEquals(ThermalManager.State.CRITICAL, receivedState)
    }

    @Test
    fun onStateChanged_notFiredWhenNoChange() {
        val context = createContextWithBatteryTemp(420)
        val tm = ThermalManager(context)

        var callCount = 0
        tm.onStateChanged = { callCount++ }

        tm.isCaptureBlocked
        val countAfterFirst = callCount

        tm.isCaptureBlocked
        assertEquals(countAfterFirst, callCount)
    }

    // --- Asymmetric EMA: rise moves much faster than fall for the same step ---
    @Test
    fun asymmetricEma_risesFasterThanFalls() {
        val context = createContextWithBatteryTemp(250)
        val tm = ThermalManager(context)

        tm.seedSmoothedTemperature(37.0f)
        val afterRise = tm.smoothTemperature(40.0f) // 3.0 deg up-step
        val afterFall = tm.smoothTemperature(37.0f) // same size step down

        val riseMove = afterRise - 37.0f
        val fallMove = afterRise - afterFall

        assertTrue("rise should be substantial, was $riseMove", riseMove > 0.5f)
        assertTrue("fall should be much slower than rise", fallMove < riseMove * 0.3f)
    }

    // --- Display number is the raw reading, not the smoothed judgment value ---
    @Test
    fun displayShowsRawNotSmoothed() {
        val context = createContextWithBatteryTemp(380)
        val tm = ThermalManager(context)

        tm.seedSmoothedTemperature(42.0f) // force a stale, much higher judged value
        tm.isCaptureBlocked

        assertEquals(38.0f, tm.displayTemperatureC, 0.001f)
        assertNotEquals(38.0f, tm.judgedTemperatureC, 0.001f)
    }

    // --- Torch policy ---
    @Test
    fun torchWarmLimit_forcesOffAfter3Min() {
        val context = createContextWithBatteryTemp(380)
        val tm = ThermalManager(context)
        var fired: ThermalManager.TorchShutdownReason? = null
        tm.onTorchForcedOff = { fired = it }

        var now = 1000L
        tm.clockMsProvider = { now }
        tm.isTorchActive = true
        now += 60_000L // 1 minute still under the limit
        tm.isCaptureBlocked
        assertEquals(null, fired)

        now += 180_000L // 4 minutes total: past the 3-minute budget
        tm.isCaptureBlocked
        assertEquals(ThermalManager.TorchShutdownReason.DURATION_LIMIT, fired)
    }

    @Test
    fun torch_hotState_forcesOffImmediately() {
        val context = createContextWithBatteryTemp(400)
        val tm = ThermalManager(context)
        var fired: ThermalManager.TorchShutdownReason? = null
        tm.onTorchForcedOff = { fired = it }
        tm.clockMsProvider = { 1000L }

        tm.isTorchActive = true
        tm.isCaptureBlocked

        assertEquals(ThermalManager.TorchShutdownReason.HOT, fired)
    }

    @Test
    fun torch_criticalState_forcesOffBeforeShuttingDown() {
        val context = createContextWithBatteryTemp(420)
        val tm = ThermalManager(context)
        var fired: ThermalManager.TorchShutdownReason? = null
        tm.onTorchForcedOff = { fired = it }
        tm.clockMsProvider = { 1000L }

        tm.isTorchActive = true
        tm.isCaptureBlocked

        assertEquals(ThermalManager.TorchShutdownReason.CRITICAL, fired)
    }

    @Test
    fun torchForcedOff_firesOnlyOnceUntilReEnabled() {
        val context = createContextWithBatteryTemp(400)
        val tm = ThermalManager(context)
        var fireCount = 0
        tm.onTorchForcedOff = { fireCount++ }
        tm.clockMsProvider = { 1000L }
        tm.isTorchActive = true

        tm.isCaptureBlocked
        tm.isCaptureBlocked
        assertEquals(1, fireCount)

        tm.isTorchActive = false
        tm.isTorchActive = true
        tm.isCaptureBlocked
        assertEquals(2, fireCount)
    }

    // --- Thermal protection gate (user-facing opt-out) ---

    @Test
    fun thermalProtectionEnabled_defaultsToTrue() {
        assertEquals(true, thermalManager.thermalProtectionEnabled)
    }

    @Test
    fun protectionPref_roundTripsThroughProvider() {
        val context = createContextWithBatteryTemp(400)
        val tm = ThermalManager(context)
        // Mirrors MainActivity wiring: provider maps the persisted preference
        // key to a Boolean, read at actuator time (never cached).
        val stored = mutableMapOf("thermal_protection_enabled" to true)
        tm.thermalProtectionEnabledProvider = { stored["thermal_protection_enabled"] ?: true }

        assertEquals(true, tm.thermalProtectionEnabled)

        stored["thermal_protection_enabled"] = false
        assertEquals(false, tm.thermalProtectionEnabled)

        stored["thermal_protection_enabled"] = true
        assertEquals(true, tm.thermalProtectionEnabled)
    }

    @Test
    fun protectionOff_stateMachineAndCallbacksStillDrive() {
        val context = createContextWithBatteryTemp(250)
        val tm = ThermalManager(context)
        var fireCount = 0
        tm.onTorchForcedOff = { fireCount++ }
        val observed = ArrayList<ThermalManager.State>()
        tm.onStateChanged = { observed.add(it) }
        tm.thermalProtectionEnabledProvider = { false }
        tm.clockMsProvider = { 1000L }
        tm.isTorchActive = true

        tm.setSimulatedBatteryTemperature(38.0f)
        tm.seedSmoothedTemperature(38.0f)
        tm.forceEvaluation()
        assertEquals(ThermalManager.State.WARM, tm.currentState)

        tm.setSimulatedBatteryTemperature(40.0f)
        tm.seedSmoothedTemperature(40.0f)
        tm.forceEvaluation()
        assertEquals(ThermalManager.State.HOT, tm.currentState)

        tm.setSimulatedBatteryTemperature(42.0f)
        tm.seedSmoothedTemperature(42.0f)
        tm.forceEvaluation()
        assertEquals(ThermalManager.State.CRITICAL, tm.currentState)

        // The state machine + callbacks kept running the whole time...
        assertEquals(
            listOf(
                ThermalManager.State.WARM,
                ThermalManager.State.HOT,
                ThermalManager.State.CRITICAL
            ),
            observed
        )
        // ...but no torch actuator ever fired.
        assertEquals(0, fireCount)
    }

    @Test
    fun protectionOff_torchNeverForcedOff_andNoDurationCap() {
        val context = createContextWithBatteryTemp(380)
        val tm = ThermalManager(context)
        var fired: ThermalManager.TorchShutdownReason? = null
        tm.onTorchForcedOff = { fired = it }
        var now = 1000L
        tm.clockMsProvider = { now }
        tm.thermalProtectionEnabledProvider = { false }

        tm.isTorchActive = true
        tm.forceEvaluation()
        // 10 minutes warm: far past the 3-minute cap if protection were on.
        now += 600_000L
        tm.forceEvaluation()
        assertEquals(ThermalManager.State.WARM, tm.currentState)
        assertEquals(null, fired)

        tm.setSimulatedBatteryTemperature(40.0f)
        tm.seedSmoothedTemperature(40.0f)
        tm.forceEvaluation() // HOT: would YANK the torch if protection were on.
        assertEquals(ThermalManager.State.HOT, tm.currentState)
        assertEquals(null, fired)

        tm.setSimulatedBatteryTemperature(42.0f)
        tm.seedSmoothedTemperature(42.0f)
        tm.forceEvaluation()
        assertEquals(ThermalManager.State.CRITICAL, tm.currentState)
        assertEquals(null, fired)
    }

    @Test
    fun protectionOff_toggleOnWhileHot_reappliesHotPolicy() {
        val context = createContextWithBatteryTemp(400)
        val tm = ThermalManager(context)
        var fired: ThermalManager.TorchShutdownReason? = null
        tm.onTorchForcedOff = { fired = it }
        tm.clockMsProvider = { 1000L }

        tm.thermalProtectionEnabledProvider = { false }
        tm.isTorchActive = true
        tm.forceEvaluation()
        assertEquals(ThermalManager.State.HOT, tm.currentState)
        assertEquals(null, fired)

        // Toggle protection back on: the forced evaluation must re-apply the
        // HOT actuator (torch off) immediately, not on the next poll.
        tm.thermalProtectionEnabledProvider = { true }
        tm.forceEvaluation()
        assertEquals(ThermalManager.TorchShutdownReason.HOT, fired)
    }
}
