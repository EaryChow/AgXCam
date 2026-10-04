package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Output-domain covariance measurement.
 *
 * The estimator subtracts a 3x3 local mean, which attenuates white noise by
 * 8/9. That factor is common to every channel and every cross term, so it must
 * cancel out of the correlations and out of the reported ratios. The tests
 * below lean on that: the synthetic fields are built at unit variance and the
 * expected numbers are derived from the construction, not from a golden run.
 */
class OutputCovarianceTest {

    private val w = 96
    private val h = 96
    private val texels = w * h

    /** Structure map with every texel PLAIN (class channel 0). */
    private fun allPlain(): FloatArray {
        val s = FloatArray(texels * 4)
        for (i in 0 until texels) s[i * 4 + 2] = 0f
        return s
    }

    private fun put(data: FloatArray, x: Int, y: Int, r: Float, g: Float, b: Float) {
        val i = (y * w + x) * 4
        data[i] = r
        data[i + 1] = g
        data[i + 2] = b
    }

    private fun setClass(structure: FloatArray, x: Int, y: Int, cls: Float) {
        structure[(y * w + x) * 4 + 2] = cls
    }

    /**
     * R = z, G = rho*z + sqrt(1-rho^2)*z2, B = rho*z + sqrt(1-rho^2)*z3 with
     * independent standard normals. rho(R,G) = rho(R,B) = rho and
     * rho(G,B) = rho^2, so this is deliberately NOT equicorrelated: it
     * distinguishes a measured 3x3 from the scan's single-rho model.
     */
    private fun correlatedField(rho: Float, seed: Int): FloatArray {
        val out = FloatArray(texels * 4)
        val rnd = Random(seed)
        val k = sqrt(maxOf(0f, 1f - rho * rho))
        for (y in 0 until h) {
            for (x in 0 until w) {
                val z = rnd.nextDouble(-1.0, 1.0)
                val z2 = rnd.nextDouble(-1.0, 1.0)
                val z3 = rnd.nextDouble(-1.0, 1.0)
                put(
                    out, x, y,
                    z.toFloat(),
                    (rho * z + k * z2).toFloat(),
                    (rho * z + k * z3).toFloat()
                )
            }
        }
        return out
    }

    private fun rho(r: Float?) = r ?: error("expected a correlation, got null")

    @Test
    fun uncorrelatedNoiseMeasuresZeroCorrelation() {
        val out = FloatArray(texels * 4)
        val rnd = Random(7)
        for (y in 0 until h) {
            for (x in 0 until w) {
                put(
                    out, x, y,
                    rnd.nextFloat() - 0.5f,
                    rnd.nextFloat() - 0.5f,
                    rnd.nextFloat() - 0.5f
                )
            }
        }
        val m = OutputCovariance.measure(out, allPlain(), w, h, stride = 3)
        assertNotNull(m)
        assertTrue(abs(rho(m!!.rhoRG)) < 0.20f)
        assertTrue(abs(rho(m.rhoRB)) < 0.20f)
        assertTrue(abs(rho(m.rhoGB)) < 0.20f)
    }

    @Test
    fun knownCorrelationComesBackUnbiased() {
        val target = 0.8f
        val out = correlatedField(target, seed = 11)
        val m = OutputCovariance.measure(out, allPlain(), w, h, stride = 3)
        assertNotNull(m)
        // The 8/9 local-mean attenuation is common to numerator and denominator,
        // so it must not move the correlation.
        assertEquals(target, rho(m!!.rhoRG), 0.03f)
        assertEquals(target, rho(m.rhoRB), 0.03f)
        // G and B share z, so their correlation is rho^2 - this is the pair that
        // proves the measurement kept the off-diagonal structure rather than
        // collapsing to one scalar.
        assertEquals(target * target, rho(m.rhoGB), 0.04f)
    }

    @Test
    fun lumaUnderestimateMatchesTheAnalyticCovariance() {
        val target = 0.8f
        val m = OutputCovariance.measure(correlatedField(target, seed = 12), allPlain(), w, h, stride = 1)
        assertNotNull(m)
        val vR = m!!.sigmaR * m.sigmaR
        val vG = m.sigmaG * m.sigmaG
        val vB = m.sigmaB * m.sigmaB
        val cRG = target * vR
        val cRB = target * vR
        val cGB = target * target * vR
        val a = CovarianceSensitivity.LUMA_R
        val b = CovarianceSensitivity.LUMA_G
        val c = CovarianceSensitivity.LUMA_B
        val marginal = a * a * vR + b * b * vG + c * c * vB
        val exact = marginal +
            2.0f * (a * b * cRG + a * c * cRB + b * c * cGB)
        val expected = (sqrt(exact) - sqrt(marginal)) / sqrt(exact)
        // Sampling error on a correlation goes as 1/sqrt(n); this is a finite
        // sample, so the tolerance is set by the estimator, not by the algebra.
        assertEquals(expected, m.lumaUnderestimate, 0.03f)
        assertTrue(m.criterionExceeded)
    }

    @Test
    fun lumaUnderestimateHitsTheAnalyticCeilingWhenChannelsMatch() {
        // Every channel carries the same noise, which is the strongest possible
        // correlation, and it puts the marginal propagation at its worst. The
        // ceiling is analytic: with equal variances and unit-sum luma weights the
        // exact luma variance is sum(w)^2 * v and the marginal one is
        // sum(w^2) * v, so the underestimate cannot exceed 1 - sqrt(sum(w^2)),
        // about 33%. A figure above that would mean the cross terms were counted
        // wrong, and this is the only case that can detect it.
        val out = FloatArray(texels * 4)
        val rnd = Random(13)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val z = rnd.nextFloat() - 0.5f
                put(out, x, y, z, z, z)
            }
        }
        val m = OutputCovariance.measure(out, allPlain(), w, h, stride = 1)
        assertNotNull(m)
        val a = CovarianceSensitivity.LUMA_R
        val b = CovarianceSensitivity.LUMA_G
        val c = CovarianceSensitivity.LUMA_B
        val ceiling = 1.0f - sqrt(a * a + b * b + c * c)
        assertEquals(ceiling, m!!.lumaUnderestimate, 0.02f)
        assertTrue(m.criterionExceeded)
        assertTrue(rho(m.rhoRG) > 0.99f)
    }

    @Test
    fun scaleOfTheInputDoesNotChangeTheReportedRatios() {
        val field = correlatedField(0.6f, seed = 14)
        val scaled = FloatArray(field.size) { field[it] * 1000f }
        val a = OutputCovariance.measure(field, allPlain(), w, h, stride = 3)!!
        val b = OutputCovariance.measure(scaled, allPlain(), w, h, stride = 3)!!
        assertEquals(a.lumaUnderestimate, b.lumaUnderestimate, 1.0e-6f)
        assertEquals(a.chromaOverestimate, b.chromaOverestimate, 1.0e-6f)
        assertEquals(a.sigmaR * 1000f, b.sigmaR, a.sigmaR * 1.0e-3f)
    }

    @Test
    fun structureTexelsAreExcludedFromThePopulation() {
        // Half the frame is TEXTURE. If the gate did nothing, the texture side
        // would drag the sigmas and the whole measurement with it.
        val plain = correlatedField(0.5f, seed = 15)
        val structure = allPlain()
        for (y in 0 until h / 2) {
            for (x in 0 until w) setClass(structure, x, y, 1f)
        }
        val masked = OutputCovariance.measure(plain, structure, w, h, stride = 3)
        val unmasked = OutputCovariance.measure(plain, allPlain(), w, h, stride = 3)
        assertNotNull(masked)
        assertNotNull(unmasked)
        // Roughly half the admitted interiors disappear.
        assertTrue(masked!!.samples < unmasked!!.samples * 0.75f)
        // The surviving population is drawn from the same generator, so the
        // correlation is unchanged; only the count moves.
        assertEquals(rho(unmasked.rhoRG), rho(masked.rhoRG), 0.08f)
    }

    @Test
    fun darkPlainIsAdmittedAsFlatGround() {
        // DARK_PLAIN is a PLAIN promoted by the toe threshold, so it is flat
        // ground and belongs in the population. Excluding it is what made the
        // first device run report "no PLAIN sample" while sitting on hundreds of
        // flat texels.
        val field = correlatedField(0.6f, seed = 20)
        val structure = allPlain()
        for (y in 0 until h) {
            for (x in 0 until w) {
                if ((x + y) % 2 == 0) setClass(structure, x, y, 3f)
            }
        }
        val m = OutputCovariance.measure(field, structure, w, h, stride = 3)
        assertNotNull(m)
        assertTrue(
            "checkerboard PLAIN/DARK_PLAIN is all flat and must still measure",
            m!!.samples > 0
        )
        assertEquals(0.6f, rho(m.rhoRG), 0.05f)
    }

    @Test
    fun interleavedDarkPlainStillMeasuresOnAToeStraddlingScene() {
        // Speckled PLAIN/DARK_PLAIN: no texel has nine PLAIN neighbours, which is
        // exactly the scene that starved the original all-nine gate.
        val field = correlatedField(0.7f, seed = 21)
        val structure = allPlain()
        val rnd = Random(22)
        for (i in 0 until texels) {
            structure[i * 4 + 2] = if (rnd.nextFloat() < 0.5f) 0f else 3f
        }
        val m = OutputCovariance.measure(field, structure, w, h, stride = 3)
        assertNotNull(m)
        assertTrue(m!!.samples > 0)
    }

    @Test
    fun aLoneFlatTexelInATexturedFieldIsNotEnough() {
        // One PLAIN texel surrounded by TEXTURE has no flat neighbourhood, so it
        // is rejected on the neighbour count rather than measured against a mean
        // dominated by stroke values.
        val field = FloatArray(texels * 4)
        val structure = allPlain()
        for (i in 0 until texels) setClassClass(structure, i, 1f)
        setClass(structure, w / 2, h / 2, 0f)
        assertNull(OutputCovariance.measure(field, structure, w, h, stride = 3))
    }

    private fun setClassClass(structure: FloatArray, index: Int, cls: Float) {
        structure[index * 4 + 2] = cls
    }

    @Test
    fun flatNeighbourMeanIgnoresColocatedTexture() {
        // A structured texel inside the neighbourhood must not drag the local
        // mean, or the residual inherits the stroke's own value and reads as
        // noise. Same construction on both sides, one side with a TEXTURE
        // neighbour: the measured correlation must barely move.
        val field = correlatedField(0.5f, seed = 23)
        val clean = allPlain()
        val speckled = allPlain()
        val rnd = Random(24)
        for (i in 0 until texels) {
            if (rnd.nextFloat() < 0.05f) setClassClass(speckled, i, 1f)
        }
        val a = OutputCovariance.measure(field, clean, w, h, stride = 3)!!
        val b = OutputCovariance.measure(field, speckled, w, h, stride = 3)!!
        assertEquals(rho(a.rhoRG), rho(b.rhoRG), 0.05f)
    }

    @Test
    fun fullyStructuredFrameYieldsNoSampleRatherThanZeroes() {
        val out = FloatArray(texels * 4)
        val structure = allPlain()
        for (y in 0 until h) {
            for (x in 0 until w) setClass(structure, x, y, 1f)
        }
        assertNull(OutputCovariance.measure(out, structure, w, h, stride = 3))
    }

    @Test
    fun absentPlanesYieldNoSample() {
        assertNull(OutputCovariance.measure(null, allPlain(), w, h))
        assertNull(OutputCovariance.measure(FloatArray(texels * 4), null, w, h))
    }

    @Test
    fun degenerateGeometryIsRejected() {
        val out = FloatArray(texels * 4)
        assertNull(OutputCovariance.measure(out, allPlain(), 2, 2))
        assertNull(OutputCovariance.measure(out, allPlain(), 0, 0))
        // A short plane must not be indexed past its end.
        assertNull(OutputCovariance.measure(FloatArray(16), allPlain(), w, h))
    }

    @Test
    fun tooFewSamplesIsRefused() {
        // A 4x4 field has at most a couple of admissible interiors, far below
        // MIN_SAMPLES, and a second-moment matrix from that reads as confident.
        val small = 4
        val plane = FloatArray(small * small * 4)
        val structure = FloatArray(small * small * 4)
        assertNull(OutputCovariance.measure(plane, structure, small, small))
    }

    @Test
    fun constantFieldProducesNoNaNAndNoCorrelation() {
        val out = FloatArray(texels * 4) { 0.4f }
        val m = OutputCovariance.measure(out, allPlain(), w, h, stride = 3)
        assertNotNull(m)
        // Zero variance means the correlation is undefined, so it must read as
        // absent rather than as 0/0 or a plausible-looking number.
        assertNull(m!!.rhoRG)
        assertNull(m.rhoGB)
        assertEquals(0f, m.lumaUnderestimate, 0f)
        assertEquals(0f, m.sigmaR, 1.0e-6f)
    }

    @Test
    fun nonFiniteSamplesAreSkippedNotPropagated() {
        val field = correlatedField(0.7f, seed = 16)
        for (y in 0 until h) {
            for (x in 0 until w) {
                if ((x + y) % 7 == 0) put(field, x, y, Float.NaN, 0f, 0f)
            }
        }
        val m = OutputCovariance.measure(field, allPlain(), w, h, stride = 3)
        assertNotNull(m)
        assertTrue(m!!.sigmaR.isFinite() && m.sigmaG.isFinite() && m.sigmaB.isFinite())
        assertTrue(m.lumaExact.isFinite() && m.lumaMarginal.isFinite())
    }

    @Test
    fun classGateThresholdIsInclusiveOfPlain() {
        val field = correlatedField(0.5f, seed = 17)
        val structure = allPlain()
        // Exactly at the threshold: still PLAIN, so still admitted.
        for (i in 0 until texels) structure[i * 4 + 2] = OutputCovariance.CLASS_PLAIN_MAX
        val atThreshold = OutputCovariance.measure(field, structure, w, h, stride = 3)
        assertNotNull(atThreshold)
        // One step above: excluded everywhere.
        for (i in 0 until texels) structure[i * 4 + 2] = OutputCovariance.CLASS_PLAIN_MAX + 0.01f
        assertNull(OutputCovariance.measure(field, structure, w, h, stride = 3))
    }

    @Test
    fun strideOnlyChangesTheSampleCountNotTheEstimate() {
        val field = correlatedField(0.75f, seed = 18)
        val dense = OutputCovariance.measure(field, allPlain(), w, h, stride = 1)!!
        val sparse = OutputCovariance.measure(field, allPlain(), w, h, stride = 7)!!
        assertTrue(sparse.samples < dense.samples)
        assertEquals(rho(dense.rhoRG), rho(sparse.rhoRG), 0.06f)
        assertEquals(dense.lumaUnderestimate, sparse.lumaUnderestimate, 0.02f)
    }

    @Test
    fun reportNamesTheMeasurementAndItsVerdict() {
        val m = OutputCovariance.measure(correlatedField(0.8f, seed = 19), allPlain(), w, h, stride = 3)!!
        val text = m.report()
        assertTrue(text.contains("output-domain covariance"))
        assertTrue(text.contains("MEASURED"))
        assertTrue(text.contains("EXCEEDED"))
        assertTrue(text.contains("rhoRG="))
        assertTrue(OutputCovariance.NO_SAMPLE.contains("no flat sample"))
    }

    @Test
    fun chromaRowsAreSharedWithTheAssumedRhoScan() {
        // Both consumers must project onto the same chroma basis, or the measured
        // and assumed chroma figures are not comparable.
        val rows = CovarianceSensitivity.chromaRows()
        assertEquals(2, rows.size)
        assertEquals(3, rows[0].size)
        rows[0][0] = 99f
        assertTrue(CovarianceSensitivity.chromaRows()[0][0] != 99f)
    }
}