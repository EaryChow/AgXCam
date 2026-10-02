package com.agx.camera.gpu

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Covariance sensitivity scan.
 *
 * The per-channel sigma estimates are propagated through a fixed linear
 * YC1C2 transform using marginal variances only:
 *
 *   var(L) = L * diag(sigma2) * L^T
 *
 * That is exact for uncorrelated channels. It is not what the sensor
 * produces: demosaic leaves the channels correlated, and the correlation is
 * measurable only in the output domain (a Bayer-domain measurement cannot
 * see it, because the correlation is created by the demosaic).
 *
 * The known failure direction is fixed: with positive channel correlation the
 * marginal propagation UNDERESTIMATES the luma variance and OVERESTIMATES the
 * chroma variances. Luma feeds epsilon directly, so an underestimate is the
 * one that matters; chroma running hot is the safe direction.
 *
 * The scan therefore walks rho in [0.4, 0.89] x the full ISO range and asks a
 * single numeric question: what is the worst-case relative epsilon_luma
 * underestimate? Above 5% the output-domain 3x3 covariance measurement point
 * becomes required.
 */
object CovarianceSensitivity {

    /** Max relative epsilon_luma underestimate before the scan flags the point. */
    const val LUMA_UNDERESTIMATE_LIMIT = 0.05f

    /** Correlation range the demosaic is known to leave behind. */
    const val RHO_MIN = 0.40f
    const val RHO_MAX = 0.89f

    /** YC1C2-style luma weights (the transform's first row). */
    const val LUMA_R = 0.299f
    const val LUMA_G = 0.587f
    const val LUMA_B = 0.114f

    data class Point(
        val rho: Float,
        val iso: Int,
        val epsilonLumaMarginal: Float,
        val epsilonLumaExact: Float,
        val underestimate: Float,
        val chromaOverestimate: Float
    )

    data class ScanResult(
        val points: List<Point>,
        val maxLumaUnderestimate: Float,
        val worstRho: Float,
        val worstIso: Int,
        val criterionExceeded: Boolean
    ) {
        /** Verdict text for the bundle. */
        fun verdict(): String = String.format(
            "covariance scan: max epsilon_luma underestimate=%.2f%% at rho=%.2f iso=%d; limit=%.1f%% -> %s",
            maxLumaUnderestimate * 100f, worstRho, worstIso,
            LUMA_UNDERESTIMATE_LIMIT * 100f,
            if (criterionExceeded) "TRIGGERED (output-domain 3x3 covariance measurement point required)" else "not triggered"
        )

        fun report(): String {
            val sb = StringBuilder()
            sb.append(verdict()).append('\n')
            sb.append("  rho    iso    epsL_marginal  epsL_exact   under    chroma_over\n")
            for (p in points) {
                sb.append(String.format(
                    "  %.2f  %6d  %13.4f  %11.4f  %6.2f%%  %8.2f%%\n",
                    p.rho, p.iso, p.epsilonLumaMarginal, p.epsilonLumaExact,
                    p.underestimate * 100f, p.chromaOverestimate * 100f
                ))
            }
            return sb.toString()
        }
    }

    /**
     * Luma variance with marginal propagation: sum of w_i^2 * sigma2_i.
     */
    fun lumaVarianceMarginal(sigma2R: Float, sigma2G: Float, sigma2B: Float): Float =
        LUMA_R * LUMA_R * sigma2R + LUMA_G * LUMA_G * sigma2G + LUMA_B * LUMA_B * sigma2B

    /**
     * Luma variance with the inter-channel correlation terms included.
     * For an equicorrelated set, each cross term is 2*rho*w_i*w_j*sigma_i*sigma_j.
     */
    fun lumaVarianceExact(
        sigmaR: Float, sigmaG: Float, sigmaB: Float, rho: Float
    ): Float {
        var v = lumaVarianceMarginal(sigmaR * sigmaR, sigmaG * sigmaG, sigmaB * sigmaB)
        v += 2f * rho * LUMA_R * LUMA_G * sigmaR * sigmaG
        v += 2f * rho * LUMA_R * LUMA_B * sigmaR * sigmaB
        v += 2f * rho * LUMA_G * LUMA_B * sigmaG * sigmaB
        return max(v, 0f)
    }

    /**
     * Chroma energy is the part of the C1/C2 plane orthogonal to luma. Under
     * equicorrelation it shrinks (the channels move together), so the exact
     * chroma variance is lower than the marginal propagation - which is why
     * chroma runs HOT on the marginal path, the safe direction.
     */
    fun chromaVarianceMarginal(sigma2R: Float, sigma2G: Float, sigma2B: Float): Float {
        // Sum of the two orthogonal chroma basis rows' variances, using the
        // same 1/sqrt(3) scaled Cb/Cr rows the transform uses.
        val c1 = floatArrayOf(-0.168736f, -0.331264f, 0.5f)
        val c2 = floatArrayOf(0.5f, -0.418688f, -0.081312f)
        return dotSq(c1, sigma2R, sigma2G, sigma2B) + dotSq(c2, sigma2R, sigma2G, sigma2B)
    }

    fun chromaVarianceExact(sigmaR: Float, sigmaG: Float, sigmaB: Float, rho: Float): Float {
        val c1 = floatArrayOf(-0.168736f, -0.331264f, 0.5f)
        val c2 = floatArrayOf(0.5f, -0.418688f, -0.081312f)
        return orthogonalVar(c1, sigmaR, sigmaG, sigmaB, rho) +
            orthogonalVar(c2, sigmaR, sigmaG, sigmaB, rho)
    }

    private fun dotSq(w: FloatArray, s2R: Float, s2G: Float, s2B: Float): Float =
        w[0] * w[0] * s2R + w[1] * w[1] * s2G + w[2] * w[2] * s2B

    private fun orthogonalVar(w: FloatArray, sR: Float, sG: Float, sB: Float, rho: Float): Float {
        var v = dotSq(w, sR * sR, sG * sG, sB * sB)
        v += 2f * rho * w[0] * w[1] * sR * sG
        v += 2f * rho * w[0] * w[2] * sR * sB
        v += 2f * rho * w[1] * w[2] * sG * sB
        return max(v, 0f)
    }

    /**
     * Full sweep. `isoModel` returns the per-channel sigma at a mid signal
     * level for an ISO, which is what the scan feeds in - the same ratio
     * shape the runtime profile measures.
     */
    fun sweep(
        isoModel: (Int) -> FloatArray,
        rhos: FloatArray = defaultRhos(),
        isos: IntArray = defaultIsos()
    ): ScanResult {
        val points = ArrayList<Point>()
        var maxUnder = 0f
        var worstRho = RHO_MIN
        var worstIso = isos.firstOrNull() ?: 0

        for (iso in isos) {
            val s = isoModel(iso)
            val sR = max(s[0], 1.0e-6f)
            val sG = max(s[1], 1.0e-6f)
            val sB = max(s[2], 1.0e-6f)
            val epsLumaMarginal = sqrt(lumaVarianceMarginal(sR * sR, sG * sG, sB * sB))
            val epsCromaMarginal = sqrt(chromaVarianceMarginal(sR * sR, sG * sG, sB * sB))
            for (rho in rhos) {
                val epsLumaExact = sqrt(lumaVarianceExact(sR, sG, sB, rho))
                val epsCromaExact = sqrt(chromaVarianceExact(sR, sG, sB, rho))
                val under = if (epsLumaExact > 1.0e-9f) {
                    (epsLumaExact - epsLumaMarginal) / epsLumaExact
                } else 0f
                val over = if (epsCromaMarginal > 1.0e-9f) {
                    (epsCromaMarginal - epsCromaExact) / epsCromaMarginal
                } else 0f
                points.add(Point(rho, iso, epsLumaMarginal, epsLumaExact, max(under, 0f), max(over, 0f)))
                if (under > maxUnder) {
                    maxUnder = under
                    worstRho = rho
                    worstIso = iso
                }
            }
        }
        return ScanResult(points, maxUnder, worstRho, worstIso, maxUnder > LUMA_UNDERESTIMATE_LIMIT)
    }

    /** Correlation grid: 0.40 to 0.89 inclusive in even steps. */
    fun defaultRhos(): FloatArray {
        val steps = 10
        val out = FloatArray(steps)
        for (i in 0 until steps) out[i] = RHO_MIN + (RHO_MAX - RHO_MIN) * i / (steps - 1).toFloat()
        return out
    }

    /** ISO grid: the bucket ladder the runtime profile keys on. */
    fun defaultIsos(): IntArray {
        val out = ArrayList<Int>()
        var iso = 100
        while (iso <= 102400) {
            out.add(iso)
            iso *= 2
        }
        return out.toIntArray()
    }

    /** ISO-model per-channel sigma at a mid signal, using the shipped model. */
    fun isoModelSigmas(iso: Int, signal: Float = 300f): FloatArray {
        val a = com.agx.camera.camera.NoiseModel.photonCoeff(iso)
        val b = com.agx.camera.camera.NoiseModel.readNoiseVariance(iso)
        // Per-channel photon response spread: G sees 2x the photons of R/B in
        // a Bayer cell, so its shot variance is roughly half for the same DN
        // signal. Read noise is common.
        val shotR = a * signal
        val shotG = a * signal * 0.5f
        val shotB = a * signal * 0.8f
        return floatArrayOf(
            sqrt(shotR + b),
            sqrt(shotG + b),
            sqrt(shotB + b)
        )
    }

    /** Convenience wrapper over the shipped noise model. */
    fun sweepShippedModel(): ScanResult = sweep({ iso -> isoModelSigmas(iso) })

    /** True when the luma side is underestimated at this point. */
    fun isUnderestimated(p: Point): Boolean = abs(p.underestimate) > LUMA_UNDERESTIMATE_LIMIT
}