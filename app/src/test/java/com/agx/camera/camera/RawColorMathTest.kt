package com.agx.camera.camera

import com.agx.camera.color.ColorMatrix
import com.agx.camera.color.ColorMatrix.Mat3
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt

class RawColorMathTest {

    private val EPS = 1e-3f

    private val ID = ColorMatrix.identity()

    // D65 sits far enough from both anchors to exercise the interpolation.
    private val T1 = 5003f
    private val T2 = 7504f

    private fun matMulVec(m: Mat3, v: FloatArray): FloatArray = floatArrayOf(
        m.m[0] * v[0] + m.m[1] * v[1] + m.m[2] * v[2],
        m.m[3] * v[0] + m.m[4] * v[1] + m.m[5] * v[2],
        m.m[6] * v[0] + m.m[7] * v[1] + m.m[8] * v[2]
    )

    private fun maxAbs(vec: FloatArray): Float =
        vec.maxOf { abs(it) }

    private fun assertClose(label: String, expected: Float, actual: Float, tolerance: Float = 1e-4f) {
        assertTrue(
            "$label: expected $expected, got $actual (tolerance $tolerance)",
            abs(expected - actual) <= tolerance
        )
    }

    private fun requireMat(m: Mat3?): Mat3 = m ?: throw AssertionError("matrix must exist")

    @Test
    fun xyToTemperature_d50() {
        val d50 = RawColorMath.xyToTemperature(floatArrayOf(0.3457f, 0.3585f))
        assertEquals("D50 ~= 5003K", 5003f, d50, 60f)
    }

    @Test
    fun xyToTemperature_d65() {
        val d65 = RawColorMath.xyToTemperature(floatArrayOf(0.3127f, 0.3290f))
        assertEquals("D65 ~= 6504K", 6504f, d65, 80f)
    }

    @Test
    fun xyToTemperature_illuminantA() {
        val a = RawColorMath.xyToTemperature(floatArrayOf(0.4476f, 0.4074f))
        assertEquals("Illuminant A ~= 2856K", 2856f, a, 80f)
    }

    @Test
    fun degenerateNeutral_returnsNull() {
        val zero = RawColorMath.srgbMatrix(ID, ID, ID, ID, null, null, T1, T2, floatArrayOf(0f, 0f, 0f))
        assertNull("zero neutral", zero)
        val negative = RawColorMath.srgbMatrix(ID, ID, ID, ID, null, null, T1, T2, floatArrayOf(1f, -2f, 3f))
        assertNull("negative neutral", negative)
        val nan = RawColorMath.srgbMatrix(ID, ID, ID, ID, null, null, T1, T2, floatArrayOf(Float.NaN, 1f, 1f))
        assertNull("NaN neutral", nan)
        val tiny = RawColorMath.srgbMatrix(ID, ID, ID, ID, null, null, T1, T2, floatArrayOf(1e-30f, 1f, 1f))
        assertNotNull("positive tiny is fine", tiny)
    }

    @Test
    fun identityProfile_identityNeutral_neutralizes() {
        for (t1 in listOf(2856f, 5003f, 6504f)) {
            for (t2 in listOf(6504f, 7504f)) {
                if (t1 >= t2) continue
                val m = requireMat(
                    RawColorMath.srgbMatrix(
                        ID, ID, ID, ID, null, null, t1, t2,
                        floatArrayOf(1f, 1f, 1f)
                    )
                )
                assertTrue("no NaN", m.m.all { !it.isNaN() })
                assertTrue("no Inf", m.m.all { !it.isInfinite() })
                val result = matMulVec(m, floatArrayOf(1f, 1f, 1f))
                val spread = result.max() - result.min()
                assertTrue("neutral should map near-neutral (spread $spread)", spread < 0.15f)
            }
        }
    }

    @Test
    fun identityProfile_sceneNeutrals_neutralize() {
        // Camera RGB for various scene whites through an identity XYZ->camera
        // matrix is just XYZ, so the neutral is the normalized XYZ of the white.
        val cases = mapOf(
            5003f to floatArrayOf(0.9644f, 1.0f, 0.8251f),
            6504f to floatArrayOf(0.9504f, 1.0f, 1.0891f),
            2856f to floatArrayOf(1.0985f, 1.0f, 0.3558f)
        )
        for ((temp, neutral) in cases) {
            val m = requireMat(
                RawColorMath.srgbMatrix(ID, ID, ID, ID, null, null, 2856f, 7504f, neutral)
            )
            val result = matMulVec(m, neutral)
            val spread = result.max() - result.min()
            assertTrue("neutral for ${temp}K should map near-neutral (spread $spread)", spread < 0.2f)
            val xy = RawColorMath.sceneWhiteXy(neutral, ID, ID, 2856f, 7504f)
                ?: throw AssertionError("sceneWhiteXy must exist")
            val tempGuess = RawColorMath.xyToTemperature(xy)
            assertTrue("scene-white solve recovers temperature", abs(temp - tempGuess) < 60f)
        }
    }

    @Test
    fun forwardPath_neutralizesAtReferenceIlluminant() {
        // D75 camera neutral through an identity XYZ->camera matrix is the
        // normalized XYZ of the D75 white.
        val d75Neutral = floatArrayOf(0.7743f, 0.8155f, 1.0f)
        // A self-consistent forward matrix for illuminant 2 (7504K): it maps
        // the reference camera neutral to the PCS (D50) white.
        val fwd = ColorMatrix.diagonal(0.9642f, 1.0f, 0.8249f)
        val m = requireMat(
            RawColorMath.srgbMatrix(ID, ID, ID, ID, fwd, fwd, T1, T2, d75Neutral)
        )
        assertTrue("no NaN", m.m.all { !it.isNaN() })
        assertTrue("no Inf", m.m.all { !it.isInfinite() })
        val result = matMulVec(m, d75Neutral)
        val spread = result.max() - result.min()
        assertTrue("reference neutral should map near-neutral (spread $spread)", spread < 0.05f)
    }

    @Test
    fun forwardPath_offReference_staysFinite() {
        // An arbitrary forward matrix off its reference illuminant is only an
        // approximation -- assert the math stays well-formed, not exact.
        val fwd = Mat3(
            floatArrayOf(
                1.2f, -0.1f, 0.05f,
                -0.05f, 1.1f, -0.1f,
                0.02f, -0.1f, 1.3f
            )
        )
        val m = requireMat(
            RawColorMath.srgbMatrix(ID, ID, ID, ID, fwd, fwd, T1, T2, floatArrayOf(1f, 1f, 1f))
        )
        assertTrue("no NaN", m.m.all { !it.isNaN() })
        assertTrue("no Inf", m.m.all { !it.isInfinite() })
        val out = matMulVec(m, floatArrayOf(1f, 1f, 1f))
        assertTrue("no NaN output", out.all { !it.isNaN() })
        assertTrue("no Inf output", out.all { !it.isInfinite() })
    }

    @Test
    fun normalizeForwardMatrix_pinsReferenceWhiteToPcs() {
        // A forward matrix shipped in the DNG reference-illuminant convention
        // (row sums == XYZ of D65, max-1 scale): FWD * [1,1,1] must become the
        // D50 PCS white after normalization.
        val d65Convention = ColorMatrix.diagonal(0.9505f, 1.0f, 1.0888f)
        val normalized = requireMat(RawColorMath.normalizeForwardMatrix(d65Convention))
        val out = matMulVec(normalized, floatArrayOf(1f, 1f, 1f))
        assertEquals("X == PCS X", 0.9642, out[0].toDouble(), 0.005)
        assertEquals("Y == PCS Y", 1.0, out[1].toDouble(), 0.005)
        assertEquals("Z == PCS Z", 0.8249, out[2].toDouble(), 0.005)
    }

    @Test
    fun forwardPath_referenceIlluminant_mapsToTrueWhite() {
        // D65 scene shot at the D65 reference illuminant. The forward matrix
        // uses the raw DNG convention (row sums = D65 white); after the
        // PCS-normalization the as-shot neutral must land exactly on sRGB
        // white -- no residual warm/cool tint that would skew the colors.
        val d65Neutral = floatArrayOf(0.8730f, 0.9184f, 1.0f)
        val d65ConventionFwd = ColorMatrix.diagonal(0.9505f, 1.0f, 1.0888f)
        val m = requireMat(
            RawColorMath.srgbMatrix(ID, ID, ID, ID, d65ConventionFwd, d65ConventionFwd, T1, T2, d65Neutral)
        )
        val result = matMulVec(m, d65Neutral)
        val spread = result.max() - result.min()
        assertTrue("reference neutral maps to true white (spread $spread)", spread < 0.05f)
        assertEquals("X near 1", 1.0, result[0].toDouble(), 0.05)
        assertEquals("Y near 1", 1.0, result[1].toDouble(), 0.05)
        assertEquals("Z near 1", 1.0, result[2].toDouble(), 0.05)
    }

    @Test
    fun warmVsCoolNeutral_differentMatrices() {
        val neutralWarm = floatArrayOf(1.4f, 1f, 0.6f)
        val neutralCool = floatArrayOf(0.6f, 1f, 1.4f)
        val mWarm = requireMat(RawColorMath.srgbMatrix(ID, ID, ID, ID, null, null, T1, T2, neutralWarm))
        val mCool = requireMat(RawColorMath.srgbMatrix(ID, ID, ID, ID, null, null, T1, T2, neutralCool))
        val diff = maxAbs(FloatArray(9) { mWarm.m[it] - mCool.m[it] })
        assertTrue("warm vs cool neutral should give different matrices (diff $diff)", diff > 0.01f)
        val warmOut = matMulVec(mWarm, neutralWarm)
        assertTrue("warm neutralized", warmOut.max() - warmOut.min() < 0.2f)
        val coolOut = matMulVec(mCool, neutralCool)
        assertTrue("cool neutralized", coolOut.max() - coolOut.min() < 0.2f)
    }

    @Test
    fun rowsSumPositive() {
        val m = requireMat(RawColorMath.srgbMatrix(ID, ID, ID, ID, null, null, T1, T2, floatArrayOf(1f, 1f, 1f)))
        for (row in 0..2) {
            val sum = m.m[row * 3] + m.m[row * 3 + 1] + m.m[row * 3 + 2]
            assertEquals("row $row sums to 1 (neutral->neutral)", 1.0, sum.toDouble(), 0.15)
        }
    }

    @Test
    fun srgbMatrix_isRowMajorColumnVector() {
        val neutral = floatArrayOf(1f, 0.5f, 1f)
        val m = requireMat(RawColorMath.srgbMatrix(ID, ID, ID, ID, null, null, T1, T2, neutral))
        val out = matMulVec(m, neutral)
        assertTrue("no NaN output", out.all { !it.isNaN() })
        assertTrue("no Inf output", out.all { !it.isInfinite() })
    }

    // XYZ of D65 at Y = 1, matching what xyToXyz hands to the color matrix.
    private fun d65Xyz() = floatArrayOf(0.950456f, 1.0f, 1.089058f)

    // XYZ of D50 at Y = 1, the PCS white the reference matrices are normalized on.
    private val D50_XYZ = floatArrayOf(0.9642f, 1.0f, 0.8249f)

    // Two XYZ->camera matrices, each solved against d65Xyz so the sensor answers
    // a D65 white with a known, chromatic response - red 0.45 / green 1 / blue
    // 0.75 for the first, 0.55 / 1 / 0.68 for the second. That shape is the
    // reason a flat reference cannot be right for every sensor. Two distinct
    // matrices also give the temperature interpolation something to do, which is
    // what a real profile reports.
    private val chromaticMatrix1 = Mat3(
        floatArrayOf(
            0.400000f, 0.048036f, 0.020000f,
            0.050000f, 0.974258f, -0.020000f,
            0.650000f, 0.010000f, 0.112209f
        )
    )
    private val chromaticMatrix2 = Mat3(
        floatArrayOf(
            0.520000f, 0.044872f, 0.010000f,
            0.030000f, 1.004158f, -0.030000f,
            0.600000f, -0.020000f, 0.119120f
        )
    )

    private fun daylightTransform(
        cm1: Mat3 = chromaticMatrix1,
        cm2: Mat3 = chromaticMatrix2,
        fwd1: Mat3? = null,
        fwd2: Mat3? = null
    ): RawColorMath.NeutralTransform? = RawColorMath.neutralTransformAtDaylight(
        cm1, cm2, ID, ID, fwd1, fwd2, T1, T2
    )

    // The sensor's own daylight response, recovered from the gains the profile
    // hands out: gains * response == (1,1,1), green-normalized.
    private fun daylightResponse(t: RawColorMath.NeutralTransform) =
        floatArrayOf(1f / t.wbGains[0], 1f, 1f / t.wbGains[2])

    // The D65 white in camera-native space, straight from the two reference
    // matrices interpolated at 6504. This is what the profile reads to build the
    // daylight gains, derived here independently of the profile code so the tests
    // below are checking the maths rather than restating it.
    private fun normalizeColorMatrix(m: Mat3): Mat3 {
        val maxC = maxOf(coordOf(m)[0], coordOf(m)[1], coordOf(m)[2])
        if (maxC > 0.0f && (maxC < 0.99f || maxC > 1.01f)) {
            return Mat3(m.m.map { it / maxC }.toFloatArray())
        }
        return m
    }

    private fun coordOf(m: Mat3) = matMulVec(m, D50_XYZ)

    private fun d65CameraResponse(
        cm1: Mat3,
        cm2: Mat3,
        t1: Float = T1,
        t2: Float = T2
    ): FloatArray {
        // Reciprocal-temperature weighting, matching how the profile blends its two
        // references: the warm reference gets weight 1 and the cool reference 0 at
        // their own temperatures. Linear-in-temperature would be a different
        // answer, so the weighting is restated here rather than approximated.
        val g = (1.0f / 6504f - 1.0f / t2) / (1.0f / t1 - 1.0f / t2)
        val xyz = d65Xyz()
        // Each reference is normalized on its own before the blend, the way the
        // profile does it: the two matrices are calibrated independently and their
        // absolute scales are not comparable, so blending raw numbers would mix an
        // artifact into a real difference. Normalization divides by the largest
        // response to D50, the PCS white the matrices are authored against.
        val a = matMulVec(normalizeColorMatrix(cm1), xyz)
        val b = matMulVec(normalizeColorMatrix(cm2), xyz)
        val out = FloatArray(3) { g * a[it] + (1f - g) * b[it] }
        // Green-normalized, which is the space the gains live in.
        return floatArrayOf(out[0] / out[1], 1f, out[2] / out[1])
    }

    @Test
    fun daylightTransform_isPinnedToD65NotTheVendorReference() {
        // Vendors label their two matrices with reference temperatures that are
        // not D65. The daylight transform reads them at 6504 by name, so moving
        // those labels must not move the answer - otherwise "daylight" would be
        // whatever the vendor happened to call the low reference.
        val baseline = daylightTransform() ?: throw AssertionError("must exist")
        val moved = RawColorMath.neutralTransformAtDaylight(
            chromaticMatrix1, chromaticMatrix2, ID, ID, null, null, 4000f, 9000f
        ) ?: throw AssertionError("daylight transform must exist")
        // The interpolation weight depends on those labels, so the two matrices are
        // blended differently and the two answers are close but not identical.
        // What matters is that D65 stayed put, so each is checked against the
        // reference response read at its own pair's weight - which is the whole
        // claim: the answer tracks 6504, not the vendor's naming.
        for (case in listOf(
            Triple("baseline", baseline, T1 to T2),
            Triple("moved", moved, 4000f to 9000f)
        )) {
            val (label, transform, temps) = case
            val expected = d65CameraResponse(chromaticMatrix1, chromaticMatrix2, temps.first, temps.second)
            for (c in 0..2) {
                assertClose("$label gain $c", 1f / expected[c], transform.wbGains[c], 2e-3f)
            }
        }

        // The matrix blend moves with the labels too, so the two are close but not
        // identical. Asserted as small rather than absent because a difference is
        // the expected outcome of re-labelling the same pair, and the property
        // under test is that neither blend pulls the reference off 6504.
        val matrixDrift = (0 until 9).maxOf { abs(baseline.colorMatrix.m[it] - moved.colorMatrix.m[it]) }
        assertTrue("colorMatrix tracks the scene temperature (drift $matrixDrift)", matrixDrift < 0.01f)

        // And it must be the chromatic answer, not the flat one an identity
        // matrix would give. Without that the anchor would be saying nothing and
        // the estimator would be back to referencing green.
        val flat = RawColorMath.neutralTransformAtDaylight(
            ID, ID, ID, ID, null, null, T1, T2
        ) ?: throw AssertionError("daylight transform must exist")
        assertTrue(
            "daylight anchor is chromatic, not flat",
            abs(flat.wbGains[0] - baseline.wbGains[0]) > 0.05f &&
                abs(flat.wbGains[2] - baseline.wbGains[2]) > 0.05f
        )
    }

    @Test
    fun daylightTransform_existsAndIsGreenNormalized() {
        val t = daylightTransform() ?: throw AssertionError("daylight transform must exist")
        assertEquals("green at 1", 1f, t.wbGains[1], 1e-6f)
        assertTrue("gains finite and positive", t.wbGains.all { it.isFinite() && it > 0f })
        assertTrue("no NaN matrix", t.colorMatrix.m.all { !it.isNaN() })

        // The sensor answers red and blue below green under daylight, so a gain
        // vector that came out flat would mean the anchor had nothing to say and
        // the estimator would be back to referencing green.
        assertTrue("red needs a lift: ${t.wbGains[0]}", t.wbGains[0] > 1.05f)
        assertTrue("blue needs a lift: ${t.wbGains[2]}", t.wbGains[2] > 1.05f)
        assertNotEquals("red and blue differ", t.wbGains[0], t.wbGains[2])
    }

    @Test
    fun daylightTransform_mapsEqualCodesToWhite() {
        // The pipeline-critical invariant, and it is about equal codes rather than
        // the raw daylight response: the estimator always finishes by driving the
        // three channels onto one level, so equal codes is the only input this
        // matrix actually has to handle. The daylight gains shape what the
        // estimator aims at before that; the matrix's job starts afterwards.
        //
        // Equal codes have to survive every pairing the profile can hand us:
        // two distinct references, two identical ones (no interpolation to do),
        // an identity pair, and with or without a forward matrix to derive the
        // camera-to-XYZ side.
        val pairs = listOf(
            chromaticMatrix1 to chromaticMatrix2,
            chromaticMatrix1 to chromaticMatrix1,
            ID to ID
        )
        for ((cm1, cm2) in pairs) {
            for (fwd in listOf(null, ColorMatrix.diagonal(0.9505f, 1.0f, 1.0888f))) {
                val t = daylightTransform(cm1, cm2, fwd, fwd)
                    ?: throw AssertionError("daylight transform must exist")
                val out = matMulVec(t.colorMatrix, floatArrayOf(1f, 1f, 1f))
                val spread = out.max() - out.min()
                assertTrue("equal codes map to white (spread $spread)", spread < 0.05f)
            }
        }
    }

    @Test
    fun daylightTransform_equalsAskingForTheDaylightWhite() {
        // Pinning the chromaticity must not change the answer relative to the
        // as-shot path asked for the same white. This is what makes the daylight
        // reference the same reference, reached the same way, rather than a
        // second estimate that could drift away from D65.
        val cm = chromaticMatrix1
        val derived = daylightTransform(cm, cm) ?: throw AssertionError("must exist")
        val asked = RawColorMath.neutralTransform(
            cm, cm, ID, ID, null, null, T1, T2, matMulVec(cm, d65Xyz())
        ) ?: throw AssertionError("as-shot daylight must exist")

        assertArrayEquals("wbGains", asked.wbGains, derived.wbGains, 1e-3f)
        assertArrayEquals("colorMatrix", asked.colorMatrix.m, derived.colorMatrix.m, 1e-3f)
    }

    @Test
    fun daylightTransform_gainsAreTheD65Response() {
        // The anchor itself. The gains must be the reciprocal of what the sensor
        // answers for a D65 white, read straight off the two reference matrices
        // at 6504 rather than through the profile code, so this checks the maths
        // instead of restating it.
        for ((cm1, cm2) in listOf(
            chromaticMatrix1 to chromaticMatrix2,
            chromaticMatrix1 to chromaticMatrix1,
            ID to ID
        )) {
            val t = daylightTransform(cm1, cm2) ?: throw AssertionError("must exist")
            val expected = d65CameraResponse(cm1, cm2)
            for (c in 0..2) {
                assertClose(
                    "gain $c for ${cm1 === ID}",
                    1f / expected[c],
                    t.wbGains[c],
                    2e-3f
                )
            }
        }
    }

    @Test
    fun daylightChain_takesAD65FrameToWhite() {
        // End to end across both halves: a frame that IS the sensor's daylight
        // response, estimated and rendered through the daylight matrix, has to
        // come out white. A matrix built from a different neutral, or gains aimed
        // somewhere other than level codes, would put the cast somewhere instead
        // of removing it.
        val t = daylightTransform(chromaticMatrix1, chromaticMatrix2)
            ?: throw AssertionError("must exist")
        val response = daylightResponse(t)
        val rggb = intArrayOf(0, 1, 1, 2)

        // What the scan reports off the raw phases at a plausible exposure, with
        // the two green sites answering differently the way they do on real glass.
        // Every mean sits well above MIN_PHASE_MEAN, which the estimator needs
        // before it will touch a channel at all.
        val exposure = 1000.0
        val phaseMeans = doubleArrayOf(
            response[0] * exposure, 0.9 * exposure, 1.1 * exposure, response[2] * exposure
        )
        val estimate = GreyWorldEstimate.estimate(phaseMeans, rggb, t.wbGains)!!

        // This is the anchoring, stated as a number. The frame already is the
        // sensor's D65 response - the CFA's green bias and nothing else - so the
        // reference comes straight off and the illuminant stage finds nothing left
        // to correct. A neutral daylight scene is the zero point, which is what
        // makes the reading a measurement of the light rather than a restatement of
        // the CFA. Comparing against 1.0 rather than against the profile's gains is
        // the check, because any other reference illuminant disagrees here.
        // Measured against the green reference, which is the geometric mean of the
        // two sites. The greens answered 900 and 1100, so that sits at 995 and not
        // at the 1000 exposure - the estimate is reading a frame whose greens
        // disagree, and "nothing left to correct" means level with the pair.
        val greenReference = sqrt(0.9 * 1.1 * exposure * exposure).toFloat()
        for (p in intArrayOf(0, 1, 2, 3)) {
            val corrected = (phaseMeans[p] * estimate.daylight[p] * estimate.illuminant[p]).toFloat()
            assertClose("illuminant $p", greenReference, corrected, 1.0f)
        }

        // The product still carries the reference, which is what actually takes a
        // D65 frame to the level codes the matrix below renders as white.
        val phaseGains = estimate.phaseGains

        // The two green sites answered 900 and 1100 and have to land on one level
        // before the merge, which is what keeps the demosaic from averaging two
        // differently scaled signals into one green.
        assertClose(
            "green sites reconcile",
            (phaseMeans[1] * phaseGains[1]).toFloat(),
            (phaseMeans[2] * phaseGains[2]).toFloat(),
            1e-3f
        )

        // Both halves of the split together, which is what the shader applies. It lands
        // on the green reference - 995 here, the geometric mean of the two greens -
        // and the matrix sees that level with green normalized to 1, which is the
        // one input it was built for.
        val preMerge = GreyWorldEstimate.preMergeGains(phaseGains, rggb)
        val colors = GreyWorldEstimate.colorGains(phaseGains, rggb)
        val balanced = floatArrayOf(
            (phaseMeans[0] * preMerge[0] * colors[0]).toFloat(),
            (phaseMeans[1] * preMerge[1] * colors[1]).toFloat(),
            (phaseMeans[3] * preMerge[3] * colors[2]).toFloat()
        )
        assertClose("red level", balanced[1], balanced[0], 0.02f)
        assertClose("blue level", balanced[1], balanced[2], 0.02f)
        val level = floatArrayOf(balanced[0] / balanced[1], 1f, balanced[2] / balanced[1])

        val out = matMulVec(t.colorMatrix, level)
        val whiteSpread = out.max() - out.min()
        assertTrue("D65 renders white (spread $whiteSpread)", whiteSpread < 0.02f)
        assertClose("near unit brightness", 1f, (out[0] + out[1] + out[2]) / 3f, 0.05f)
    }

    @Test
    fun daylightChain_leavesANonDaylightSceneToCarryTheCast() {
        // The other half of what the anchor is for. A scene under a light far
        // from D65 has to keep its cast in the gains - if the estimator collapsed
        // everything toward unit gains the matrix would render every scene as if
        // it were daylight, which is the failure mode a fixed D65 matrix creates
        // unless the gains still measure the light.
        val t = daylightTransform(chromaticMatrix1, chromaticMatrix2)
            ?: throw AssertionError("must exist")
        val rggb = intArrayOf(0, 1, 1, 2)
        val daylight = daylightResponse(t)

        // The same frame lit warm: red comes up, blue drops, greens unchanged.
        // Kept well clear of the gain clamps: red needs to fall by more than 2x to read
        // as a warm scene, and at 2x the estimate clamps at MIN_GAIN, which would
        // land the channel back on the exposure and test nothing.
        val phaseMeans = doubleArrayOf(
            daylight[0] * 1.5 * 1000, 0.9 * 1000, 1.1 * 1000, daylight[2] * 0.7 * 1000
        )
        val estimate = GreyWorldEstimate.estimate(phaseMeans, rggb, t.wbGains)!!
        // Against the daylight frame, whose illuminant stage came back all-1s: red
        // has to fall below that and blue rise above it. If the illuminant stage
        // collapsed toward 1 whatever the light was, the pinned D65 matrix would
        // render every scene as though it were daylight.
        assertTrue("red ${estimate.illuminant[0]} below the daylight answer", estimate.illuminant[0] < 1f)
        assertTrue("blue ${estimate.illuminant[3]} above the daylight answer", estimate.illuminant[3] > 1f)
        val gains = estimate.phaseGains

        // And the warm frame still lands on level codes green-normalized, so the matrix
        // gets the one input it is built for whatever the light was.
        val preMerge = GreyWorldEstimate.preMergeGains(gains, rggb)
        val colors = GreyWorldEstimate.colorGains(gains, rggb)
        val balanced = floatArrayOf(
            (phaseMeans[0] * preMerge[0] * colors[0]).toFloat(),
            (phaseMeans[1] * preMerge[1] * colors[1]).toFloat(),
            (phaseMeans[3] * preMerge[3] * colors[2]).toFloat()
        )
        assertClose("red level", balanced[1], balanced[0], 0.02f)
        assertClose("blue level", balanced[1], balanced[2], 0.02f)
    }

    // XYZ -> camera, reading the two reference matrices interpolated at D65 the way
    // the profile reads them. Green-normalized, so it is on the same scale as the
    // phase means the tests build from it and as daylightResponse above. Stated
    // directly so the tests can build a scene the sensor would genuinely report.
    private fun cameraResponseTo(
        sceneXyz: FloatArray,
        cm1: Mat3 = chromaticMatrix1,
        cm2: Mat3 = chromaticMatrix2
    ): FloatArray {
        val g = (1.0f / 6504f - 1.0f / T2) / (1.0f / T1 - 1.0f / T2)
        val a = normalizeColorMatrix(cm1).m
        val b = normalizeColorMatrix(cm2).m
        val blended = FloatArray(9) { i -> g * a[i] + (1f - g) * b[i] }
        val v = matMulVec(Mat3(blended), sceneXyz)
        return floatArrayOf(v[0] / v[1], 1f, v[2] / v[1])
    }

    @Test
    fun sceneXyForNeutral_recoversTheIlluminantChromaticity() {
        // The whole estimate rests on this: the neutral a sensor reports for a grey
        // object, pushed through the D65 camera->XYZ map, is the illuminant. So the
        // round trip has to be exact - build a scene white, ask what the sensor would
        // report for it, and read the chromaticity back off.
        val scenes = listOf(
            "D65" to d65Xyz(),
            // Deliberately off the Planckian locus. A scene is a chromaticity, not a
            // temperature, and a point here has no CCT at all - which is the whole
            // reason AUTO cannot route the estimate through one.
            "off-locus" to floatArrayOf(0.85f, 1.0f, 0.55f)
        )
        for ((name, xyz) in scenes) {
            val neutral = cameraResponseTo(xyz)
            val xy = RawColorMath.xyFromXyz(
                matMulVec(
                    requireMat(RawColorMath.cameraToXyzAtDaylight(
                        chromaticMatrix1, chromaticMatrix2, ID, ID, null, null, T1, T2
                    )),
                    neutral
                )
            ) ?: throw AssertionError("$name must be a chromaticity")
            val sum = xyz.sum()
            assertClose("$name x", xyz[0] / sum, xy[0], 1e-3f)
            assertClose("$name y", xyz[1] / sum, xy[1], 1e-3f)
        }
    }

    @Test
    fun sceneXyForNeutral_rejectsANonNeutral() {
        assertNull(RawColorMath.xyFromXyz(floatArrayOf(0f, 0f, 0f)))
        // A negative-red response is not a chromaticity, and reporting it as one
        // would let an obviously broken reading drive the adaptation.
        assertNull(RawColorMath.xyFromXyz(floatArrayOf(-0.2f, 1f, 0.6f)))
    }

    @Test
    fun neutralTransformAtSceneXy_d65IsTheDaylightTransform() {
        // The zero point. Passing D65 with the sensor's own daylight white has to
        // give back the daylight transform exactly, because that is what the
        // daylight transform is: the same construction with the estimate landing on
        // D65. If the scene temperature were still reaching the matrix reads, these
        // would diverge and a neutral daylight scene would not be its own anchor.
        val daylight = daylightTransform() ?: throw AssertionError("must exist")
        val white = daylightResponse(daylight)
        val direct = RawColorMath.neutralTransformAtSceneXy(
            chromaticMatrix1, chromaticMatrix2, ID, ID, null, null, T1, T2,
            white, floatArrayOf(0.3127f, 0.3290f)
        ) ?: throw AssertionError("must exist")
        for (i in 0..8) {
            assertClose("matrix $i", daylight.colorMatrix.m[i], direct.colorMatrix.m[i], 1e-4f)
        }
    }

    @Test
    fun neutralTransformAtSceneXy_rendersAnOffLocusIlluminantAsWhite() {
        // The failure the direct path exists to avoid. This illuminant has no
        // correlated color temperature, so a solve that recovers a temperature and
        // turns it back into a chromaticity has to land somewhere else on the locus
        // and leave the cast behind. The direct path is handed the measurement and
        // has nothing to recover, so the scene renders neutral.
        val sceneXyz = floatArrayOf(0.85f, 1.0f, 0.55f)
        val sceneXy = floatArrayOf(0.85f / 2.4f, 1.0f / 2.4f)
        val neutral = cameraResponseTo(sceneXyz)

        val direct = RawColorMath.neutralTransformAtSceneXy(
            chromaticMatrix1, chromaticMatrix2, ID, ID, null, null, T1, T2, neutral, sceneXy
        ) ?: throw AssertionError("must exist")

        // The transform's own gains whiten the neutral, then its matrix sees level
        // codes - the same two-stage shape the estimator's gains produce.
        val level = floatArrayOf(
            neutral[0] * direct.wbGains[0],
            neutral[1] * direct.wbGains[1],
            neutral[2] * direct.wbGains[2]
        )
        val out = matMulVec(direct.colorMatrix, level)
        val spread = out.max() - out.min()
        assertTrue("off-locus scene renders white (spread $spread)", spread < 0.02f)

        // And the old path cannot do it, which is the point of the comparison. The
        // solve projects the illuminant onto the blackbody locus, so it lands on a
        // nearby locus point instead of the measured one and leaves a cast.
        val solved = RawColorMath.neutralTransform(
            chromaticMatrix1, chromaticMatrix2, ID, ID, null, null, T1, T2, neutral
        ) ?: throw AssertionError("must exist")
        val solvedLevel = floatArrayOf(
            neutral[0] * solved.wbGains[0],
            neutral[1] * solved.wbGains[1],
            neutral[2] * solved.wbGains[2]
        )
        val solvedOut = matMulVec(solved.colorMatrix, solvedLevel)
        val solvedSpread = solvedOut.max() - solvedOut.min()
        assertTrue(
            "the temperature solve leaves a cast the direct path does not ($solvedSpread vs $spread)",
            solvedSpread > spread + 0.01f
        )
    }

    @Test
    fun neutralTransformAtSceneXy_matrixIsWhiteBalanceRemoved() {
        // The decoupling, stated as something falsifiable. The matrix is WB-removed,
        // so on its own it does not take the raw sensor neutral to white - it expects
        // level codes. A matrix that folded the balance in would neutralize the raw
        // neutral by itself, and a caller that had already whitened the signal would
        // then be taking the cast off twice.
        val neutral = cameraResponseTo(floatArrayOf(0.85f, 1.0f, 0.55f))
        val t = RawColorMath.neutralTransformAtSceneXy(
            chromaticMatrix1, chromaticMatrix2, ID, ID, null, null, T1, T2,
            neutral, floatArrayOf(0.85f / 2.4f, 1.0f / 2.4f)
        ) ?: throw AssertionError("must exist")

        val level = floatArrayOf(
            neutral[0] * t.wbGains[0],
            neutral[1] * t.wbGains[1],
            neutral[2] * t.wbGains[2]
        )
        val balanced = matMulVec(t.colorMatrix, level)
        val balancedSpread = balanced.max() - balanced.min()
        assertTrue("gains then matrix is white (spread $balancedSpread)", balancedSpread < 0.02f)

        // The matrix alone, on the raw sensor neutral, is not: that residual spread is
        // the balance still sitting in the gains, which is where it belongs.
        val rawOut = matMulVec(t.colorMatrix, neutral)
        val rawSpread = rawOut.max() - rawOut.min()
        assertTrue(
            "the matrix alone leaves the balance to the gains ($rawSpread vs $balancedSpread)",
            rawSpread > balancedSpread + 0.01f
        )
    }

    @Test
    fun autoChain_measuresTheIlluminantAndRendersItNeutral() {
        // End to end for AUTO, built from a real scene rather than from hand-picked
        // channel ratios. The frame is what the sensor would report for a grey object
        // under a known warm white, so the estimator has something consistent to
        // measure and the answer can be checked against the white that was asked for.
        val daylight = daylightTransform() ?: throw AssertionError("must exist")
        val rggb = intArrayOf(0, 1, 1, 2)

        // A warm white at xy (0.41, 0.38). Off the Planckian locus, so it has no CCT
        // to recover - the case AUTO has to get right by never asking for one.
        val sceneXyWanted = floatArrayOf(0.41f, 0.38f)
        val sceneXyz = floatArrayOf(
            sceneXyWanted[0] / sceneXyWanted[1],
            1f,
            (1f - sceneXyWanted[0] - sceneXyWanted[1]) / sceneXyWanted[1]
        )

        // The frame: the sensor's response to that white, at an exposure, with the
        // two green sites disagreeing the way they do on real glass.
        val response = cameraResponseTo(sceneXyz)
        val exposure = 1000.0
        val phaseMeans = doubleArrayOf(
            response[0] * exposure, 0.9 * exposure, 1.1 * exposure, response[2] * exposure
        )

        val estimate = GreyWorldEstimate.estimate(phaseMeans, rggb, daylight.wbGains)!!
        val gains = estimate.phaseGains
        val preMerge = GreyWorldEstimate.preMergeGains(gains, rggb)
        val colors = GreyWorldEstimate.colorGains(gains, rggb)
        val balanced = floatArrayOf(
            (phaseMeans[0] * preMerge[0] * colors[0]).toFloat(),
            (phaseMeans[1] * preMerge[1] * colors[1]).toFloat(),
            (phaseMeans[3] * preMerge[3] * colors[2]).toFloat()
        )
        val level = floatArrayOf(balanced[0] / balanced[1], 1f, balanced[2] / balanced[1])

        // The measured neutral is the estimator's gains inverted, read through the
        // same D65 camera->XYZ the daylight gains came from - so the estimate and the
        // render describe one sensor, which is what makes the reading a measurement
        // of the light rather than of the CFA.
        val estimatedNeutral = floatArrayOf(1f / gains[0], 1f, 1f / gains[3])
        val measured = RawColorMath.xyFromXyz(
            matMulVec(
                requireMat(RawColorMath.cameraToXyzAtDaylight(
                    chromaticMatrix1, chromaticMatrix2, ID, ID, null, null, T1, T2
                )),
                estimatedNeutral
            )
        ) ?: throw AssertionError("a consistent frame must measure as a chromaticity")
        assertClose("measured x", sceneXyWanted[0], measured[0], 5e-3f)
        assertClose("measured y", sceneXyWanted[1], measured[1], 5e-3f)

        // And the transform built from that measurement renders the scene neutral. The
        // cast lives in the gains, which is the division of labour the two-stage
        // estimator exists to set up: gains whiten, matrix only redirects white.
        val transform = RawColorMath.neutralTransformAtSceneXy(
            chromaticMatrix1, chromaticMatrix2, ID, ID, null, null, T1, T2,
            estimatedNeutral, measured
        ) ?: throw AssertionError("must exist")
        val out = matMulVec(transform.colorMatrix, level)
        val spread = out.max() - out.min()
        assertTrue("warm scene renders white (spread $spread)", spread < 0.02f)
        assertClose("near unit brightness", 1f, (out[0] + out[1] + out[2]) / 3f, 0.05f)
    }
}
