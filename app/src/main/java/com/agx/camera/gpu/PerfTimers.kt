package com.agx.camera.gpu

import java.util.LinkedHashMap

/**
 * CPU/wall-clock stage timers for the measurement bundle.
 *
 * Only records while the "Record processing time" switch is on. GPU pass
 * timings live in GpuTimerQueries; this covers the host-side stages that the
 * GPU query book cannot see (list validation, uploads, capture readback, JPEG
 * and DNG work).
 */
object PerfTimers {
    @Volatile var enabled = false
        private set

    private val lock = Any()
    private val histograms = LinkedHashMap<String, PassTimingHistogram>()

    fun setEnabled(value: Boolean) {
        synchronized(lock) {
            if (!value) histograms.clear()
            enabled = value
        }
    }

    fun clear() {
        synchronized(lock) { histograms.clear() }
    }

    fun record(name: String, nanos: Long) {
        if (!enabled || nanos < 0L) return
        synchronized(lock) {
            histograms.getOrPut(name) { PassTimingHistogram(name) }.add(nanos)
        }
    }

    fun <T> measure(name: String, block: () -> T): T {
        if (!enabled) return block()
        val t0 = System.nanoTime()
        return try {
            block()
        } finally {
            record(name, System.nanoTime() - t0)
        }
    }

    fun report(): String {
        if (!enabled) return ""
        synchronized(lock) {
            if (histograms.isEmpty()) return "CPU/wall stage timing: no samples yet"
            val sb = StringBuilder()
            sb.append("== CPU/wall stage timing ==\n")
            for (h in histograms.values) {
                sb.append("      ")
                    .append(h.name())
                    .append(" n=").append(h.sampleCount())
                    .append(" mean=").append(fmt(h.meanNs()))
                    .append(" p50<=").append(fmt(h.percentileNs(0.50)))
                    .append(" p95<=").append(fmt(h.percentileNs(0.95)))
                    .append(" min=").append(fmt(h.minNs()))
                    .append(" max=").append(fmt(h.maxNs()))
                    .append('\n')
            }
            return sb.toString()
        }
    }

    private fun fmt(ns: Long): String = String.format("%.3fms", ns / 1.0e6)
}
