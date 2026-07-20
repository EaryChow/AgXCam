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
    fun transition_toWarm_at48C() {
        val context = createContextWithBatteryTemp(480)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked // triggers evaluateState
        assertEquals(ThermalManager.State.WARM, tm.currentState)
    }

    @Test
    fun transition_toHot_at52C() {
        val context = createContextWithBatteryTemp(520)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.HOT, tm.currentState)
    }

    @Test
    fun transition_toCritical_at55C() {
        val context = createContextWithBatteryTemp(550)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.CRITICAL, tm.currentState)
    }

    @Test
    fun torchMode_warmThresholdIs42C() {
        val context = createContextWithBatteryTemp(420)
        val tm = ThermalManager(context)
        tm.isTorchActive = true
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.WARM, tm.currentState)
    }

    @Test
    fun torchMode_hotThresholdIs47C() {
        val context = createContextWithBatteryTemp(470)
        val tm = ThermalManager(context)
        tm.isTorchActive = true
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.HOT, tm.currentState)
    }

    @Test
    fun torchMode_criticalThresholdIs50C() {
        val context = createContextWithBatteryTemp(500)
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
        val context = createContextWithBatteryTemp(550)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertEquals(ThermalManager.State.CRITICAL, tm.currentState)

        tm.reset()

        assertEquals(ThermalManager.State.NORMAL, tm.currentState)
    }

    @Test
    fun isCaptureBlocked_criticalState() {
        val context = createContextWithBatteryTemp(550)
        val tm = ThermalManager(context)
        assertTrue(tm.isCaptureBlocked)
    }

    @Test
    fun isCaptureBlocked_hotState() {
        val context = createContextWithBatteryTemp(520)
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
        val context = createContextWithBatteryTemp(550)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertTrue(tm.isPreviewReduced)
    }

    @Test
    fun isPreviewReduced_hotState() {
        val context = createContextWithBatteryTemp(520)
        val tm = ThermalManager(context)
        tm.isCaptureBlocked
        assertFalse(tm.isPreviewReduced)
    }

    @Test
    fun onStateChanged_callback_fires() {
        val context = createContextWithBatteryTemp(550)
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
        val context = createContextWithBatteryTemp(550)
        val tm = ThermalManager(context)

        var callCount = 0
        tm.onStateChanged = { callCount++ }

        tm.isCaptureBlocked
        val countAfterFirst = callCount

        tm.isCaptureBlocked
        assertEquals(countAfterFirst, callCount)
    }
}
