package com.agx.camera.thermal

/**
 * Process-wide interlock for the probe-only thermal run.
 *
 * The thermal harness must be measured with the shipped camera/preview pipeline
 * stopped, not merely avoided by launching a different activity. The normal
 * MainActivity checks this guard on create/resume and also registers a callback
 * so an already-running instance closes its camera session as soon as a probe
 * run starts. The guard stays set for the whole native loop, including while
 * the ThermalProbeActivity is backgrounded.
 */
object ThermalRunGuard {
    @Volatile
    var active = false
        private set

    @Volatile
    var onActivated: (() -> Unit)? = null

    fun activate() {
        active = true
        onActivated?.invoke()
    }

    fun deactivate() {
        active = false
    }
}
