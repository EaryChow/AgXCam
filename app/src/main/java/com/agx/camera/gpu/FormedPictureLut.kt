package com.agx.camera.gpu

import kotlin.math.max
import kotlin.math.pow

/**
 * Formed-picture luminance in 1D.
 *
 * The structure map's luminance band has to be judged on the formed picture,
 * not on the linear camera-native value: S5 works in linear RGB, and a
 * luminance band taken there puts the toe judgement in the wrong place
 * entirely. Running full AgX per pixel is too expensive, so this distills the
 * curve into a 1D table.
 *
 * What goes in: the AgX log curve and sigmoid parameters plus the ev pp
 * exposure gain applied before AgX. What stays out: the colour matrices and
 * the low-side compensation, which are near identity in the log domain for a
 * grey input. This tracks the coarse toe / mid pivot / shoulder banding and is
 * not a bit-exact match with the final picture. The table is rebuilt when any
 * of its inputs change; the per-pixel cost is one lookup.
 */
class FormedPictureLut(
    logMin: Float = -10f,
    logMax: Float = 6.5f,
    logMidgray: Float = 0.5f,
    displayMidgray: Float = 0.48f,
    contrast: Float = 2.4f,
    toe: Float = 1.5f,
    shoulder: Float = 1.5f,
    val exposureEv: Float = 1.5f,
    entries: Int = MAX_ENTRIES
) {

    val table: FloatArray = FloatArray(entries)

    /**
     * Identifies every input the table was built from. Callers compare this
     * against the current session values to decide whether to rebuild, so a
     * moved AgX slider cannot leave a stale table in place.
     */
    val configKey: String = "logMin=$logMin logMax=$logMax mid=$logMidgray " +
        "disp=$displayMidgray contrast=$contrast toe=$toe shoulder=$shoulder ev=$exposureEv"

    // Input domain is log-spaced in scene stops: a linear table would spend
    // most of its resolution on the clipped regions and none on the toe, and
    // the toe is where the luminance banding matters most.
    val domainMinStop: Float = -8f
    val domainMaxStop: Float = 2f
    private val entriesCount = entries

    init {
        require(entries >= 8) { "formed-picture LUT needs at least 8 entries" }
        require(entries <= MAX_ENTRIES) {
            // The shader declares u_formed_lut[MAX_ENTRIES]; uploading more
            // than it was compiled for is a GL error and the table would be
            // silently truncated, so the cap is enforced here instead.
            "formed-picture LUT is limited to $MAX_ENTRIES entries (shader uniform size)"
        }
        val gain = 2.0.pow((exposureEv - 1.5).toDouble()).toFloat()
        for (i in 0 until entries) {
            val stop = domainMinStop + (domainMaxStop - domainMinStop) * i / (entries - 1).toFloat()
            val linear = 2.0.pow(stop.toDouble()).toFloat()
            table[i] = evaluate(linear, gain, logMin, logMax, logMidgray, displayMidgray, contrast, toe, shoulder)
        }
        // Keep the unused scale explicit: table[0] is the domain floor, so a
        // linear input below it clamps rather than extrapolating.
        check(table[0] >= 0f) { "formed-picture LUT floor must be non-negative" }
        check(table[entries - 1] <= 1.0001f) { "formed-picture LUT ceiling must stay in 0..1" }
    }

    /** Formed luminance for a linear capacity-normalised input. */
    fun formedLuminance(linear: Float): Float {
        if (!linear.isFinite()) return 0f
        if (linear <= 0f) return table[0]
        val stop = (kotlin.math.ln(linear.toDouble()) / kotlin.math.ln(2.0)).toFloat()
        if (stop <= domainMinStop) return table[0]
        if (stop >= domainMaxStop) return table[entriesCount - 1]
        val pos = (stop - domainMinStop) / (domainMaxStop - domainMinStop) * (entriesCount - 1)
        val i0 = pos.toInt().coerceIn(0, entriesCount - 2)
        val f = pos - i0
        return table[i0] + (table[i0 + 1] - table[i0]) * f
    }

    companion object {

        /** Must match `u_formed_lut[MAX_ENTRIES]` in StructureMapShaderProgram. */
        const val MAX_ENTRIES = 256

        /**
         * The distilled AgX curve for a grey input: log, sigmoid, the 2.4
         * encode and the sRGB OETF, with the ev pp gain folded in ahead of the
         * log. The colour stages are omitted by design - see the class comment.
         */
        fun evaluate(
            linear: Float,
            exposureGain: Float,
            logMin: Float, logMax: Float,
            logMidgray: Float, displayMidgray: Float,
            contrast: Float, toe: Float, shoulder: Float
        ): Float {
            val floorLog = 0.18f * 2.0.pow(logMin.toDouble()).toFloat()
            var v = max(linear, floorLog) * exposureGain
            if (v <= 0f) return 0f
            // lin2log
            v = (kotlin.math.ln(v / 0.18f) / kotlin.math.ln(2f)) + (-logMin)
            v = v / ((-logMin) + logMax)
            v = v.coerceIn(0f, 1f)
            // sigmoid
            v = sigmoid(v, shoulder, toe, contrast, logMidgray, displayMidgray)
            // 2.4 encode
            v = spowf(v, 2.4f)
            // sRGB OETF
            return srgbOetf(max(v, 0f))
        }

        fun spowf(a: Float, b: Float): Float {
            val s = if (a < 0f) -1f else 1f
            return s * kotlin.math.abs(a.toDouble()).pow(b.toDouble()).toFloat()
        }

        fun sigmoid(x: Float, sp: Float, tp: Float, slope: Float, px: Float, py: Float): Float {
            val s0 = 1.0f
            val t0 = 0.0f
            val ss = spowf(
                (spowf(slope * ((s0 - px) / (1.0f - py)), sp) - 1.0f) * spowf(slope * (s0 - px), -sp),
                -1.0f / sp
            )
            val ms = slope * (x - px) / ss
            val fs = ms / spowf(1.0f + spowf(ms, sp), 1.0f / sp)
            val ts = spowf(
                (spowf(slope * ((px - t0) / py), tp) - 1.0f) * spowf(slope * (px - t0), -tp),
                -1.0f / tp
            )
            val mr = (slope * (x - px)) / -ts
            val ft = mr / spowf(1.0f + spowf(mr, tp), 1.0f / tp)
            return if (x >= px) ss * fs + py else (-ts * ft) + py
        }

        fun srgbOetf(linear: Float): Float =
            if (linear <= 0.0031308f) linear * 12.92f
            else 1.055f * linear.toDouble().pow(1.0 / 2.4).toFloat() - 0.055f
    }
}