package com.agx.camera.gpu

import com.agx.camera.camera.NoiseProfileStats
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Covariance sensitivity scan: the 5% criterion is executed here, not argued.
 *
 * The scan exists because marginal variance propagation is the wrong model for
 * demosaiced output. What these tests pin down is the direction of each error
 * (luma under, chroma over) and that the shipped model actually trips or does
 * not trip the written criterion - whichever way it lands, the report has to say
 * the same thing every run.
 */
class CovarianceSensitivityTest {

    @Test
    fun marginalLumaVarianceMatchesTheWeightedSum() {
        val v = CovarianceSensitivity.lumaVarianceMarginal(4f, 9f, 16f)
        val expected = 0.299f * 0.299f * 4f + 0.587f * 0.587f * 9f + 0.114f * 0.114f * 16f
        assertEquals(expected, v, 1.0e-6f)
    }

    @Test
    fun exactLumaVarianceExceedsMarginalUnderPositiveCorrelation() {
        // This is the load-bearing sign. With rho > 0 the exact variance is
        // larger, so the marginal path UNDERestimates the luma variance, and
        // luma feeds epsilon directly.
        val sR = 2.0f
        val sG = 3.0f
        val sB = 4.0f
        val marginal = CovarianceSensitivity.lumaVarianceMarginal(sR * sR, sG * sG, sB * sB)
        val exact = CovarianceSensitivity.lumaVarianceExact(sR, sG, sB, 0.6f)
        assertTrue("exact=$exact must exceed marginal=$marginal", exact > marginal)
        // At rho = 0 the two forms must agree exactly.
        assertEquals(
            marginal,
            CovarianceSensitivity.lumaVarianceExact(sR, sG, sB, 0.0f),
            1.0e-5f
        )
    }

    @Test
    fun exactLumaVarianceNeverGoesNegative() {
        // A high rho with mismatched channel sigmas can push the raw quadratic
        // form below zero; the clamp keeps sqrt() from producing NaN.
        val v = CovarianceSensitivity.lumaVarianceExact(20.0f, 0.5f, 0.5f, 0.89f)
        assertTrue("got $v", v >= 0f)
    }

    @Test
    fun chromaUnderestimationIsTheSafeDirection() {
        val sR = 2.0f
        val sG = 3.0f
        val sB = 4.0f
        val marginal = CovarianceSensitivity.chromaVarianceMarginal(sR * sR, sG * sG, sB * sB)
        val exact = CovarianceSensitivity.chromaVarianceExact(sR, sG, sB, 0.6f)
        assertTrue(
            "under positive correlation chroma must run hot on the marginal path (marginal=$marginal exact=$exact)",
            marginal >= exact
        )
    }

    @Test
    fun scanIsMonotonicInRho() {
        // More correlation means a bigger luma underestimate. If this is not
        // monotone the maximum the scan reports is not trustworthy.
        val result = CovarianceSensitivity.sweep(
            isoModel = { iso -> CovarianceSensitivity.isoModelSigmas(iso) },
            rhos = floatArrayOf(0.2f, 0.4f, 0.6f, 0.8f, 0.89f),
            isos = intArrayOf(1600)
        )
        val byRho = result.points.sortedBy { it.rho }
        for (i in 1 until byRho.size) {
            assertTrue(
                "underestimate must not decrease as rho rises: ${byRho[i - 1].underestimate} -> ${byRho[i].underestimate}",
                byRho[i].underestimate >= byRho[i - 1].underestimate - 1.0e-6f
            )
        }
    }

    @Test
    fun sweepGridCoversTheFullIsoLadderAndCorrelationRange() {
        val rhos = CovarianceSensitivity.defaultRhos()
        assertTrue(rhos.size >= 2)
        assertEquals(CovarianceSensitivity.RHO_MIN, rhos.first(), 1.0e-6f)
        assertEquals(CovarianceSensitivity.RHO_MAX, rhos.last(), 1.0e-6f)

        val isos = CovarianceSensitivity.defaultIsos()
        assertTrue(isos.first() <= 100)
        assertTrue(isos.last() >= 102400)
    }

    @Test
    fun criterionFlagMatchesTheNumberItIsComparedAgainst() {
        val result = CovarianceSensitivity.sweepShippedModel()
        val expectedTrigger = result.maxLumaUnderestimate > CovarianceSensitivity.LUMA_UNDERESTIMATE_LIMIT
        assertEquals(
            "criterionExceeded must be exactly (max > limit); max=${result.maxLumaUnderestimate}",
            expectedTrigger,
            result.criterionExceeded
        )
    }

    @Test
    fun worstCaseIsActuallyTheWorstCase() {
        val result = CovarianceSensitivity.sweepShippedModel()
        val worst = result.points.maxByOrNull { it.underestimate }!!
        assertEquals(worst.rho, result.worstRho, 1.0e-6f)
        assertEquals(worst.iso, result.worstIso)
        assertEquals(worst.underestimate, result.maxLumaUnderestimate, 1.0e-6f)
    }

    @Test
    fun verdictTextIsAsciiAndNamesTheLimit() {
        val text = CovarianceSensitivity.sweepShippedModel().report()
        assertTrue(text, text.all { it.code < 128 })
        assertTrue(text, text.contains("covariance scan"))
        assertTrue(text, text.contains("limit=5.0%"))
        // Whichever way the shipped model lands, the report has to commit to
        // one of the two states rather than staying silent.
        assertTrue(
            text,
            text.contains("TRIGGERED") || text.contains("not triggered")
        )
    }

    @Test
    fun isoModelSigmasArePositiveAndGIsQuieterThanR() {
        for (iso in intArrayOf(100, 400, 1600, 6400, 25600, 102400)) {
            val s = CovarianceSensitivity.isoModelSigmas(iso)
            assertTrue("iso=$iso sR=${s[0]}", s[0] > 0f)
            assertTrue("iso=$iso sG=${s[1]}", s[1] > 0f)
            assertTrue("iso=$iso sB=${s[2]}", s[2] > 0f)
            assertTrue("iso=$iso: G must see more photons than R", s[1] < s[0])
        }
    }

    @Test
    fun noiseFloorRisesWithIso() {
        var prev = 0f
        for (iso in intArrayOf(100, 400, 1600, 6400, 25600)) {
            val s = CovarianceSensitivity.isoModelSigmas(iso)
            assertTrue("sigma must rise with ISO at $iso (was $prev now ${s[0]})", s[0] > prev)
            prev = s[0]
        }
    }

    @Test
    fun underestimationFlagUsesTheSameLimitAsTheScan() {
        val result = CovarianceSensitivity.sweepShippedModel()
        for (p in result.points) {
            assertEquals(
                "point rho=${p.rho} iso=${p.iso}",
                absOf(p.underestimate) > CovarianceSensitivity.LUMA_UNDERESTIMATE_LIMIT,
                CovarianceSensitivity.isUnderestimated(p)
            )
        }
    }

    @Test
    fun profileAndScanShareTheSameIsoBuckets() {
        // The scan walks the profile's bucket ladder; if the two ever drifted
        // the report would compare mismatched populations.
        val scanIsos = CovarianceSensitivity.defaultIsos().toList()
        val table = NoiseProfileStats.offlineRatioTable()
        for (iso in scanIsos) {
            assertNotNull("no offline ratio for bucket $iso", table[iso])
            assertEquals(
                "scan ISO $iso must be its own bucket",
                iso,
                NoiseProfileStats.isoBucketFor(iso)
            )
        }
    }

    private fun absOf(v: Float): Float = if (v < 0f) -v else v
}