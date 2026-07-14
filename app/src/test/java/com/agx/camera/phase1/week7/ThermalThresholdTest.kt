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
    fun warmup_completesAfter60Frames() {
        assertFalse(thermalManager.isWarmupComplete)

        for (i in 1..59) {
            thermalManager.onFrameRendered(16.0f)
        }
        assertFalse(thermalManager.isWarmupComplete)

        thermalManager.onFrameRendered(16.0f)
        assertTrue(thermalManager.isWarmupComplete)
    }

    @Test
    fun warmup_baselineSetToMinimumFrameTime() {
        for (i in 1..30) {
            thermalManager.onFrameRendered(20.0f)
        }
        for (i in 31..60) {
            thermalManager.onFrameRendered(14.0f)
        }

        assertTrue(thermalManager.isWarmupComplete)
        assertEquals(ThermalManager.State.NORMAL, thermalManager.currentState)
    }

    @Test
    fun transition_toWarm_at40C() {
        val context = createContextWithBatteryTemp(400)
        val tm = ThermalManager(context)

        for (i in 1..60) tm.onFrameRendered(16.0f)
        assertTrue(tm.isWarmupComplete)

        tm.onFrameRendered(16.0f)
        assertEquals(ThermalManager.State.WARM, tm.currentState)
    }

    @Test
    fun transition_toHot_at45C() {
        val context = createContextWithBatteryTemp(450)
        val tm = ThermalManager(context)

        for (i in 1..60) tm.onFrameRendered(16.0f)
        assertTrue(tm.isWarmupComplete)

        tm.onFrameRendered(16.0f)
        assertEquals(ThermalManager.State.HOT, tm.currentState)
    }

    @Test
    fun transition_toCritical_at50C() {
        val context = createContextWithBatteryTemp(500)
        val tm = ThermalManager(context)

        for (i in 1..60) tm.onFrameRendered(16.0f)
        assertTrue(tm.isWarmupComplete)

        tm.onFrameRendered(16.0f)
        assertEquals(ThermalManager.State.CRITICAL, tm.currentState)
    }

    @Test
    fun torchMode_warmThresholdIs37C() {
        val context = createContextWithBatteryTemp(370)
        val tm = ThermalManager(context)
        tm.isTorchActive = true

        for (i in 1..60) tm.onFrameRendered(16.0f)
        assertTrue(tm.isWarmupComplete)

        tm.onFrameRendered(16.0f)
        assertEquals(ThermalManager.State.WARM, tm.currentState)
    }

    @Test
    fun torchMode_hotThresholdIs42C() {
        val context = createContextWithBatteryTemp(420)
        val tm = ThermalManager(context)
        tm.isTorchActive = true

        for (i in 1..60) tm.onFrameRendered(16.0f)
        assertTrue(tm.isWarmupComplete)

        tm.onFrameRendered(16.0f)
        assertEquals(ThermalManager.State.HOT, tm.currentState)
    }

    @Test
    fun torchMode_criticalThresholdIs47C() {
        val context = createContextWithBatteryTemp(470)
        val tm = ThermalManager(context)
        tm.isTorchActive = true

        for (i in 1..60) tm.onFrameRendered(16.0f)
        assertTrue(tm.isWarmupComplete)

        tm.onFrameRendered(16.0f)
        assertEquals(ThermalManager.State.CRITICAL, tm.currentState)
    }

    @Test
    fun normalMode_noTransitionBelowThresholds() {
        val context = createContextWithBatteryTemp(350)
        val tm = ThermalManager(context)

        for (i in 1..61) tm.onFrameRendered(16.0f)
        assertEquals(ThermalManager.State.NORMAL, tm.currentState)
    }

    @Test
    fun reset_clearsAllState() {
        val context = createContextWithBatteryTemp(500)
        val tm = ThermalManager(context)

        for (i in 1..61) tm.onFrameRendered(16.0f)
        assertEquals(ThermalManager.State.CRITICAL, tm.currentState)
        assertTrue(tm.isWarmupComplete)

        tm.reset()

        assertEquals(ThermalManager.State.NORMAL, tm.currentState)
        assertFalse(tm.isWarmupComplete)
    }

    @Test
    fun isCaptureBlocked_criticalState() {
        val context = createContextWithBatteryTemp(500)
        val tm = ThermalManager(context)

        for (i in 1..61) tm.onFrameRendered(16.0f)
        assertTrue(tm.isCaptureBlocked)
    }

    @Test
    fun isCaptureBlocked_hotState() {
        val context = createContextWithBatteryTemp(450)
        val tm = ThermalManager(context)

        for (i in 1..61) tm.onFrameRendered(16.0f)
        assertTrue(tm.isCaptureBlocked)
    }

    @Test
    fun isCaptureBlocked_normalState() {
        val context = createContextWithBatteryTemp(250)
        val tm = ThermalManager(context)

        for (i in 1..61) tm.onFrameRendered(16.0f)
        assertFalse(tm.isCaptureBlocked)
    }

    @Test
    fun isPreviewReduced_criticalWithSlowFrames() {
        val context = createContextWithBatteryTemp(500)
        val tm = ThermalManager(context)

        for (i in 1..60) tm.onFrameRendered(16.0f)

        for (i in 1..7) tm.onFrameRendered(30.0f)

        assertTrue(tm.isPreviewReduced)
    }

    @Test
    fun isPreviewReduced_hotState_noSlowFrames() {
        val context = createContextWithBatteryTemp(450)
        val tm = ThermalManager(context)

        for (i in 1..60) tm.onFrameRendered(16.0f)

        tm.onFrameRendered(16.0f)
        assertFalse(tm.isPreviewReduced)
    }

    @Test
    fun onStateChanged_callback_fires() {
        val context = createContextWithBatteryTemp(500)
        val tm = ThermalManager(context)

        var callbackFired = false
        var receivedState: ThermalManager.State? = null
        tm.onStateChanged = { state ->
            callbackFired = true
            receivedState = state
        }

        for (i in 1..60) tm.onFrameRendered(16.0f)
        tm.onFrameRendered(16.0f)

        assertTrue(callbackFired)
        assertEquals(ThermalManager.State.CRITICAL, receivedState)
    }

    @Test
    fun onStateChanged_notFiredWhenNoChange() {
        val context = createContextWithBatteryTemp(500)
        val tm = ThermalManager(context)

        var callCount = 0
        tm.onStateChanged = { callCount++ }

        for (i in 1..61) tm.onFrameRendered(16.0f)
        val countAfterFirst = callCount

        for (i in 1..5) tm.onFrameRendered(16.0f)
        assertEquals(countAfterFirst, callCount)
    }
}
