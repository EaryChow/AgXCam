package com.agx.camera.gpu

import android.content.Context
import android.os.PowerManager

/**
 * The device's thermal throttle state, sampled at frame rate.
 *
 * This exists because "the frame got slower" and "the GPU clocked down" look
 * identical in a timing histogram. When a run showed the same pass costing more
 * on a *smaller* target, the two candidate explanations were a code change and
 * a clock drop, and nothing in the report could tell them apart. Reporting the
 * status turns that from an argument into a measurement.
 *
 * It also gates the regression comparison. A throttled baseline makes every
 * later change look like an improvement, because the baseline was taken under
 * worse conditions than the candidate. Keying segments on the throttled flag
 * means the existing "compare only matching segment keys" rule already refuses
 * to compare across it, without a second comparison rule to keep in sync.
 *
 * Levels are collapsed into two key states rather than all seven. Android lets
 * the status wander between MODERATE and SEVERE, and keying each separately
 * would open a new segment on every wander until the segment cap dropped the
 * baseline. The exact level is still reported per frame and per segment, so the
 * detail survives; only the grouping is coarse.
 */
class ThermalStatus(context: Context) {

    private val powerManager =
        context.applicationContext.getSystemService(Context.POWER_SERVICE) as? PowerManager

    /**
     * Current Android thermal status, or [THERMAL_STATUS_NONE] when the
     * platform does not expose one.
     *
     * Sampled per frame rather than held: the value is cheap to read and the
     * interesting transitions happen mid-run, which is exactly when a cached
     * value would be stale by the time anyone read the report.
     */
    fun current(): Int = try {
        powerManager?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE
    } catch (_: SecurityException) {
        // The API is advisory and can be denied on locked-down builds. An
        // unknown status must not be reported as a cool device.
        THERMAL_STATUS_UNKNOWN
    }

    companion object {
        /** Not a real Android level: this build could not read one. */
        const val THERMAL_STATUS_UNKNOWN = -1

        /**
         * First status at which Android documents throttling as possible.
         *
         * MODERATE is "throttling may be required", not "is required", so a
         * segment at exactly this level is flagged rather than trusted. Erring
         * toward exclusion costs a re-run; erring toward inclusion silently
         * poisons every later comparison against the segment.
         */
        const val THROTTLE_AT = 3 // PowerManager.THERMAL_STATUS_MODERATE

        /** Short label for a raw status level. */
        fun text(status: Int): String = when (status) {
            -1 -> "unknown"
            0 -> "none"
            1 -> "light"
            2 -> "moderate-"
            3 -> "moderate"
            4 -> "severe"
            5 -> "critical"
            6 -> "emergency"
            7 -> "shutdown"
            else -> "level$status"
        }

        /** True when the level is at or past the documented throttling point. */
        fun isThrottled(status: Int): Boolean = status >= THROTTLE_AT
    }
}