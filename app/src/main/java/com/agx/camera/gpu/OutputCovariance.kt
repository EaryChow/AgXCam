package com.agx.camera.gpu

import kotlin.math.max
import kotlin.math.sqrt

/**
 * Output-domain 3x3 RGB noise covariance, measured on S5 round-1 output.
 *
 * This is the measurement point the Tier-2 revision was waiting on. The
 * assumed-rho scan in [CovarianceSensitivity] walks rho over the range the
 * demosaic is known to leave behind and reports how far the marginal
 * propagation underestimates luma; it cannot say which rho is real, because a
 * Bayer-domain sample cannot see a correlation the demosaic has not created
 * yet. That has to come out of the output domain, on real pixels.
 *
 * The estimator is the ordinary second moment of a zero-mean noise field:
 * take a texel, subtract the local mean of its 3x3 neighbourhood, and
 * accumulate the outer products. Subtracting the local mean is what makes the
 * residual zero-mean - without it every measurement would be dominated by
 * scene gradient, and the "noise" would be the picture.
 *
 * Only texels whose whole 3x3 neighbourhood classifies PLAIN are admitted. A
 * neighbourhood straddling a stroke has the scene's own structure in its mean,
 * which biases the residual toward whatever the structure gradient was, and a
 * covariance built on that is not a noise covariance. The class map already
 * exists, so the gate is free.
 *
 * Two properties make the numbers directly comparable to the scan:
 *
 *  - The result is a full 3x3, not a scalar rho. The scan's equicorrelated
 *    model is then evaluated against measured cross terms instead of
 *    assumed ones, so "rho" stops being a free parameter.
 *  - Only ratios are reported (underestimate, overestimate), never absolute
 *    variances. Those are scale-invariant, which means the measurement does
 *    not need to know the pipeline's working range or the sensor's DN scale -
 *    a factor that is easy to get wrong and impossible to notice, because a
 *    wrong scale still produces a plausible-looking percentage.
 *
 * This runs on the armed forensic grab, not per frame. The per-frame global
 * reduction the scan's own doc forbids is exactly what this defers to grab
 * time, so the runtime discipline is unchanged.
 */
object OutputCovariance {

    /** Reported when the grab could not produce a usable sample. */
    const val NO_SAMPLE = "output-domain covariance: no flat sample on this grab"

    /** Structure-map class channel; at or below this counts as PLAIN. */
    const val CLASS_PLAIN_MAX = 0.5f

    /**
     * Flat neighbours required out of the 3x3, the centre included, before a
     * texel is admitted.
     *
     * Requiring all nine was the first cut and it was wrong: PLAIN is an SNR
     * ratio against a noise prediction, so on a scene that straddles a
     * threshold it interleave speckle-wise with its neighbours and no texel
     * ever has nine PLAIN ones - the gate reports "no sample" while sitting on
     * hundreds of flat texels. A majority keeps scene leakage bounded (the
     * local mean is taken over flat neighbours only, so stroke values never
     * enter it) without demanding a classification pattern that a single
     * threshold cannot produce.
     */
    const val MIN_FLAT_NEIGHBOURS = 5

    /**
     * Below this the correlation is a division by noise. A handful of samples
     * also produces a matrix that looks fine and means nothing, so the gate is
     * on sample count rather than on the matrix being well formed.
     */
    const val MIN_SAMPLES = 32L

    data class Measurement(
        val samples: Long,
        val sigmaR: Float,
        val sigmaG: Float,
        val sigmaB: Float,
        val rhoRG: Float?,
        val rhoRB: Float?,
        val rhoGB: Float?,
        val lumaMarginal: Float,
        val lumaExact: Float,
        val chromaMarginal: Float,
        val chromaExact: Float
    ) {
        /**
         * How far the marginal propagation underestimates luma, which is the
         * direction that matters: luma feeds epsilon directly, so an
         * underestimate there is the one that biases every classification and
         * every dose downstream.
         */
        val lumaUnderestimate: Float
            get() = if (lumaExact > 1.0e-12f) {
                max((lumaExact - lumaMarginal) / lumaExact, 0f)
            } else 0f

        /** The mirror figure. Chroma running hot is the safe direction. */
        val chromaOverestimate: Float
            get() = if (chromaMarginal > 1.0e-12f) {
                max((chromaMarginal - chromaExact) / chromaMarginal, 0f)
            } else 0f

        val criterionExceeded: Boolean
            get() = lumaUnderestimate > CovarianceSensitivity.LUMA_UNDERESTIMATE_LIMIT

        fun report(): String {
            val sb = StringBuilder()
            sb.append("output-domain covariance: n=").append(samples)
                .append(" sigmaR=").append(String.format("%.5f", sigmaR))
                .append(" sigmaG=").append(String.format("%.5f", sigmaG))
                .append(" sigmaB=").append(String.format("%.5f", sigmaB))
                .append(" rhoRG=").append(fmt(rhoRG))
                .append(" rhoRB=").append(fmt(rhoRB))
                .append(" rhoGB=").append(fmt(rhoGB))
                .append('\n')
            sb.append("  luma   marginal=").append(String.format("%.5f", lumaMarginal))
                .append(" exact=").append(String.format("%.5f", lumaExact))
                .append(" under=").append(String.format("%.2f%%", lumaUnderestimate * 100f))
                .append(" | chroma over=").append(String.format("%.2f%%", chromaOverestimate * 100f))
                .append('\n')
            sb.append("  -> MEASURED, criterion ")
                .append(if (criterionExceeded) "EXCEEDED" else "within limit")
                .append(" (limit ")
                .append(String.format("%.1f%%", CovarianceSensitivity.LUMA_UNDERESTIMATE_LIMIT * 100f))
                .append("): ")
                .append(
                    if (criterionExceeded) {
                        "marginal propagation underestimates epsilon_luma; the S5 ratio denominator runs low"
                    } else {
                        "marginal propagation is adequate at this sample"
                    }
                )
            return sb.toString()
        }

        private fun fmt(v: Float?): String =
            if (v == null) "n/a" else String.format("%.4f", v)
    }

    /** Sampling step: enough texels for a stable second moment, few enough to stay cheap. */
    fun defaultStride(width: Int, height: Int): Int =
        max(1, minOf(width, height) / 48)

    /**
     * Accumulates the noise second-moment matrix over PLAIN interiors and
     * returns the derived sigmas, correlations and the resulting luma /
     * chroma comparison, or null when there are too few admitted samples.
     *
     * [round1] and [structure] are RGBA32F readbacks of equal size; the class
     * channel is read from [structure]'s blue component, matching the layout
     * the structure map writes. Either may be null, which is how a grab that
     * could not read one of them reports "no sample" instead of measuring
     * against a zeroed buffer.
     */
    fun measure(
        round1: FloatArray?,
        structure: FloatArray?,
        width: Int,
        height: Int,
        stride: Int = defaultStride(width, height)
    ): Measurement? {
        if (width < 3 || height < 3) return null
        if (round1 == null || structure == null) return null
        val texels = width * height
        if (round1.size < texels * 4) return null
        if (structure.size < texels * 4) return null
        val step = max(1, stride)

        var s00 = 0.0; var s01 = 0.0; var s02 = 0.0
        var s11 = 0.0; var s12 = 0.0; var s22 = 0.0
        var n = 0L

        var y = 1
        while (y < height - 1) {
            var x = 1
            while (x < width - 1) {
                val ci = (y * width + x) * 4
                if (isFlatClass(structure[ci + 2])) {
                    // Mean over the flat members of the neighbourhood only, so a
                    // neighbouring stroke cannot pull the baseline and turn
                    // scene gradient into apparent noise.
                    var sumR = 0.0; var sumG = 0.0; var sumB = 0.0
                    var flat = 0
                    var finite = true
                    var dy = -1
                    while (dy <= 1 && finite) {
                        var dx = -1
                        while (dx <= 1) {
                            val j = ((y + dy) * width + (x + dx)) * 4
                            if (isFlatClass(structure[j + 2])) {
                                val r = round1[j]
                                val g = round1[j + 1]
                                val b = round1[j + 2]
                                if (!r.isFinite() || !g.isFinite() || !b.isFinite()) {
                                    finite = false
                                    break
                                }
                                sumR += r
                                sumG += g
                                sumB += b
                                flat++
                            }
                            dx++
                        }
                        dy++
                    }
                    if (finite && flat >= MIN_FLAT_NEIGHBOURS) {
                        val inv = 1.0 / flat
                        val dR = round1[ci].toDouble() - sumR * inv
                        val dG = round1[ci + 1].toDouble() - sumG * inv
                        val dB = round1[ci + 2].toDouble() - sumB * inv
                        s00 += dR * dR
                        s01 += dR * dG
                        s02 += dR * dB
                        s11 += dG * dG
                        s12 += dG * dB
                        s22 += dB * dB
                        n++
                    }
                }
                x += step
            }
            y += step
        }
        if (n < MIN_SAMPLES) return null

        val inv = 1.0 / n
        val vR = max(s00 * inv, 0.0)
        val vG = max(s11 * inv, 0.0)
        val vB = max(s22 * inv, 0.0)
        val cRG = s01 * inv
        val cRB = s02 * inv
        val cGB = s12 * inv

        val sR = sqrt(vR).toFloat()
        val sG = sqrt(vG).toFloat()
        val sB = sqrt(vB).toFloat()

        val wR = CovarianceSensitivity.LUMA_R
        val wG = CovarianceSensitivity.LUMA_G
        val wB = CovarianceSensitivity.LUMA_B
        val lumaMarginal = (wR * wR * vR + wG * wG * vG + wB * wB * vB)
        val lumaExact = (
            wR * wR * vR + wG * wG * vG + wB * wB * vB +
                2.0 * (wR * wG * cRG + wR * wB * cRB + wG * wB * cGB)
            )

        var chromaMarginal = 0.0
        var chromaExact = 0.0
        for (row in CovarianceSensitivity.chromaRows()) {
            chromaMarginal += marginalVar(row, vR, vG, vB)
            chromaExact += fullVar(row, vR, vG, vB, cRG, cRB, cGB)
        }

        return Measurement(
            samples = n,
            sigmaR = sR,
            sigmaG = sG,
            sigmaB = sB,
            rhoRG = corr(cRG, vR, vG),
            rhoRB = corr(cRB, vR, vB),
            rhoGB = corr(cGB, vG, vB),
            lumaMarginal = sqrt(max(lumaMarginal, 0.0)).toFloat(),
            lumaExact = sqrt(max(lumaExact, 0.0)).toFloat(),
            chromaMarginal = sqrt(max(chromaMarginal, 0.0)).toFloat(),
            chromaExact = sqrt(max(chromaExact, 0.0)).toFloat()
        )
    }

    /**
     * Flat ground, which the classifier calls PLAIN.
     *
     * TEXTURE and EDGE are excluded because their stroke values would enter the
     * local mean and report scene structure as noise covariance. The majority
     * vote in [MIN_FLAT_NEIGHBOURS] covers the boundary texels this gate keeps
     * out of the centre.
     */
    private fun isFlatClass(cls: Float): Boolean = cls <= CLASS_PLAIN_MAX

    /** Covariance undefined when either side has no variance to divide by. */
    private fun corr(c: Double, vA: Double, vB: Double): Float? {
        if (vA <= 0.0 || vB <= 0.0) return null
        val r = c / sqrt(vA * vB)
        if (!r.isFinite()) return null
        return r.toFloat().coerceIn(-1f, 1f)
    }

    private fun marginalVar(w: FloatArray, vR: Double, vG: Double, vB: Double): Double =
        w[0] * w[0] * vR + w[1] * w[1] * vG + w[2] * w[2] * vB

    private fun fullVar(
        w: FloatArray,
        vR: Double, vG: Double, vB: Double,
        cRG: Double, cRB: Double, cGB: Double
    ): Double =
        marginalVar(w, vR, vG, vB) +
            2.0 * (w[0] * w[1] * cRG + w[0] * w[2] * cRB + w[1] * w[2] * cGB)
}