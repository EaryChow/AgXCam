package com.agx.camera.phase1.week7

import android.os.Handler
import com.agx.camera.thermal.CriticalBlinkController
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito.any
import org.mockito.Mockito.atLeastOnce
import org.mockito.Mockito.eq
import org.mockito.Mockito.mock
import org.mockito.Mockito.times
import org.mockito.Mockito.verify

class CriticalBlinkControllerTest {

    @Test
    fun start_isIdempotent_singleTickChain() {
        val handler = mock(Handler::class.java)
        var visibility = false
        val controller = CriticalBlinkController(handler) { visible -> visibility = visible }

        controller.start()
        // Re-entrant: a second start must never spawn a second tick chain.
        controller.start()
        controller.start()

        verify(handler, times(1)).post(any(Runnable::class.java))
        assertTrue(controller.isActive)
        assertTrue(visibility) // visible immediately on start
    }

    @Test
    fun start_showsWarningImmediately() {
        val handler = mock(Handler::class.java)
        var visibility = false
        val controller = CriticalBlinkController(handler) { visible -> visibility = visible }

        controller.start()

        assertTrue(visibility)
    }

    @Test
    fun tick_togglesVisibilityAndReschedules_butNeverDoublePosts() {
        val handler = mock(Handler::class.java)
        var visibility = false
        val controller = CriticalBlinkController(handler) { visible -> visibility = visible }
        controller.start()

        // Drive the single posted runnable the way the handler would.
        val captor = ArgumentCaptor.forClass(Runnable::class.java)
        verify(handler).post(captor.capture())
        captor.value.run()

        assertFalse(visibility) // first tick hides the warning
        verify(handler).postDelayed(any(Runnable::class.java), eq(350L))
        verify(handler, times(1)).post(any(Runnable::class.java)) // only the single initial chain
    }

    @Test
    fun stop_hidesWarning_andIsIdempotent() {
        val handler = mock(Handler::class.java)
        var visibility = false
        val controller = CriticalBlinkController(handler) { visible -> visibility = visible }
        controller.start()
        controller.stop()
        controller.stop() // idempotent: no-op after first stop

        assertFalse(controller.isActive)
        assertFalse(visibility)
        verify(handler, atLeastOnce()).removeCallbacks(any(Runnable::class.java))
    }

    @Test
    fun tickAfterStop_doesNothing() {
        val handler = mock(Handler::class.java)
        var visibility = false
        val visibilityChanges = ArrayList<Boolean>()
        val controller = CriticalBlinkController(handler) { visible ->
            visibility = visible
            visibilityChanges.add(visible)
        }
        controller.start()
        controller.stop()
        // start shows (true), stop hides (false) - exactly those two flips.
        assertEquals(listOf(true, false), visibilityChanges)
        assertFalse(visibility)

        // A stale tick that already left the active window must be ignored.
        val captor = ArgumentCaptor.forClass(Runnable::class.java)
        verify(handler).post(captor.capture())
        captor.value.run()

        assertEquals(listOf(true, false), visibilityChanges)
        assertFalse(visibility)
    }
}
