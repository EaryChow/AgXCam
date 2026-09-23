package com.agx.camera.thermal

import android.os.Handler

/**
 * Drives the flashing full-screen warning shown when thermal protection is
 * disabled and the battery reaches CRITICAL.
 *
 * The controller only ever toggles a boolean through the injected lambda; the
 * owner decides which view/state that maps to. All callbacks happen on the
 * injected Handler (MainActivity passes the main looper), so the lambda is free
 * to touch views directly.
 *
 * start() is idempotent: a second start while already running is a no-op, so
 * re-entry (state re-transition, toggle spam, cold-start double-call) can never
 * spawn a second tick chain or double-pace the blink.
 */
class CriticalBlinkController(
    private val handler: Handler,
    private val toggleVisibility: (Boolean) -> Unit
) {

    private var active = false
    private var visible = false

    val isActive: Boolean
        get() = active

    /** Idempotent start: at most one tick chain is ever in flight. */
    @Synchronized
    fun start() {
        if (active) return
        active = true
        visible = true
        toggleVisibility(true)
        handler.post(tick)
    }

    @Synchronized
    fun stop() {
        if (!active) return
        active = false
        handler.removeCallbacks(tick)
        toggleVisibility(false)
        visible = false
    }

    private val tick = object : Runnable {
        override fun run() {
            synchronized(this@CriticalBlinkController) {
                if (!active) return
                visible = !visible
                toggleVisibility(visible)
                handler.postDelayed(this, TICK_MS)
            }
        }
    }

    companion object {
        private const val TICK_MS = 350L
    }
}
