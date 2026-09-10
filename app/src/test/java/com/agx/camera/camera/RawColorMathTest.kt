package com.agx.camera.camera

import com.agx.camera.color.ColorMatrix
import com.agx.camera.color.ColorMatrix.Mat3
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs

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
}