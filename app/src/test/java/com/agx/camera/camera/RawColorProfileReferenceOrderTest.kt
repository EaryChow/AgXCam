package com.agx.camera.camera

import com.agx.camera.color.ColorMatrix
import com.agx.camera.color.ColorMatrix.Mat3
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs

/**
 * Guards the pairing between each reference illuminant's temperature and the
 * matrices measured under it.
 *
 * [RawColorProfile] has to order the two references warm-first because the
 * interpolation weight reaches 1 at t1 and 0 at t2. Sorting the temperatures on
 * their own - which is what this used to do - leaves each matrix attached to the
 * *other* reference, and then every entry point documented as "read at D65"
 * returns whichever illuminant happened to be in slot 2 instead of D65's own
 * response.
 *
 * The fixtures below are real device values from a phone whose ColorMatrix1 is
 * authored at D65 and ColorMatrix2 at illuminant A, so the reported order is
 * 6504 K / 2856 K and the sort runs. A correct pairing therefore reads D65 from
 * slot 2; the buggy one read illuminant A from slot 2 and put a warm cast on
 * the daylight path.
 */
class RawColorProfileReferenceOrderTest {

    // XYZ of D65 at Y = 1, the white the daylight path is anchored on.
    private val D65_XYZ = floatArrayOf(0.950456f, 1.0f, 1.089058f)

    // ColorMatrix1 of the fixture, authored at CalibrationIlluminant1 = D65.
    private val colorMatrixD65 = Mat3(
        floatArrayOf(
            0.8359375f, -0.171875f, -0.1328125f,
            -0.46875f, 1.3984375f, 0.046875f,
            -0.0859375f, 0.3359375f, 0.40625f
        )
    )

    // ColorMatrix2 of the fixture, authored at CalibrationIlluminant2 = Light A.
    private val colorMatrixA = Mat3(
        floatArrayOf(
            1.28125f, -0.484375f, -0.2265625f,
            -0.5859375f, 1.59375f, 0.140625f,
            -0.046875f, 0.1796875f, 0.703125f
        )
    )

    private val calibration = Mat3(
        floatArrayOf(1.0234375f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1.0078125f)
    )

    private val forward = Mat3(
        floatArrayOf(
            0.6328125f, 0.109375f, 0.21875f,
            0.21875f, 0.7578125f, 0.0234375f,
            -0.0390625f, -0.453125f, 1.3203125f
        )
    )

    private val reportedD65 =
        RawColorProfile.ReferenceIlluminant(6504f, colorMatrixD65, calibration, forward)
    private val reportedA =
        RawColorProfile.ReferenceIlluminant(2856f, colorMatrixA, calibration, forward)

    private fun matMulVec(m: Mat3, v: FloatArray): FloatArray = floatArrayOf(
        m.m[0] * v[0] + m.m[1] * v[1] + m.m[2] * v[2],
        m.m[3] * v[0] + m.m[4] * v[1] + m.m[5] * v[2],
        m.m[6] * v[0] + m.m[7] * v[1] + m.m[8] * v[2]
    )

    private fun greenNormalized(v: FloatArray): FloatArray {
        val g = v[1]
        return floatArrayOf(v[0] / g, v[1] / g, v[2] / g)
    }

    /**
     * The regression test. The fixture reports D65 first and A second, so the
     * order has to run, and the whole point is that the matrices travel with
     * their temperatures while it does.
     *
     * The bug this replaces swapped only the two temperatures, which left
     * 2856 K beside ColorMatrix1 and 6504 K beside ColorMatrix2 - illuminant A's
     * temperature beside the D65 matrix.
     */
    @Test
    fun orderReferences_keepsEachMatrixBesideTheTemperatureItWasMeasuredAt() {
        val (warm, cool) = RawColorProfile.orderReferences(reportedD65, reportedA)

        assertEquals("warm reference is illuminant A", 2856f, warm.kelvin, 0f)
        assertEquals("cool reference is D65", 6504f, cool.kelvin, 0f)
        assertSame(
            "the warm slot must carry ColorMatrix2, not ColorMatrix1",
            colorMatrixA, warm.colorMatrix
        )
        assertSame(
            "the cool slot must carry ColorMatrix1, not ColorMatrix2",
            colorMatrixD65, cool.colorMatrix
        )
        assertSame("calibration travels with its illuminant", calibration, warm.calibration)
        assertSame("forward matrix travels with its illuminant", forward, cool.forwardMatrix)

        // The invariant the interpolation depends on, stated directly rather than
        // implied by two independent assertions.
        assertTrue(
            "slot 1 must be the warmer reference",
            warm.kelvin <= cool.kelvin
        )
    }

    /**
     * The ordering has to be a property of the pair, not of which sensor channel
     * the vendor filled in first. Same association either way round.
     */
    @Test
    fun orderReferences_givesTheSameAssociationFromEitherReportedOrder() {
        val (warmFirst, coolFirst) = RawColorProfile.orderReferences(reportedD65, reportedA)
        val (coolFirstSwapped, warmFirstSwapped) = RawColorProfile.orderReferences(reportedA, reportedD65)

        assertEquals(warmFirst.kelvin, coolFirstSwapped.kelvin, 0f)
        assertEquals(coolFirst.kelvin, warmFirstSwapped.kelvin, 0f)
        assertSame(warmFirst.colorMatrix, coolFirstSwapped.colorMatrix)
        assertSame(coolFirst.colorMatrix, warmFirstSwapped.colorMatrix)
    }

    /**
     * An already-warm-first pair is left alone, and a degenerate pair is stable.
     * Neither should need special handling, but both are the sort of thing a
     * sort quietly gets wrong.
     */
    @Test
    fun orderReferences_leavesAnOrderedOrDegeneratePairAlone() {
        val (warm, cool) = RawColorProfile.orderReferences(reportedA, reportedD65)
        assertEquals("warm first is not reversed", 2856f, warm.kelvin, 0f)
        assertEquals("cool second is not reversed", 6504f, cool.kelvin, 0f)

        val same = RawColorProfile.ReferenceIlluminant(6504f, colorMatrixD65, calibration, forward)
        val (a, b) = RawColorProfile.orderReferences(same, same)
        assertEquals("equal temperatures keep reported order", 6504f, a.kelvin, 0f)
        assertEquals("equal temperatures keep reported order", 6504f, b.kelvin, 0f)
        assertSame(a.colorMatrix, b.colorMatrix)
    }

    /**
     * End to end through the fixed pairing: a D65 grey, evaluated with the matrix
     * the ordering selects, has to render neutral.
     *
     * The scene white is the fixture's own ColorMatrix1 under D65, which is the
     * only way to get an independent ground truth - the sensor codes a white
     * object at exactly the value the D65 matrix predicts, so if the render picks
     * the wrong reference the numbers move.
     */
    @Test
    fun daylightTransform_throughTheFixedPairing_rendersAD65GreyWhite() {
        val (warm, cool) = RawColorProfile.orderReferences(reportedD65, reportedA)

        val t = RawColorMath.neutralTransformAtDaylight(
            // cm1 is paired with t1 and cm2 with t2: slot 1 carries the warm
            // reference, slot 2 the cool one, and each matrix sits beside its own
            // temperature. This is the invariant the sort exists to preserve.
            warm.colorMatrix ?: throw AssertionError("warm matrix must exist"),
            cool.colorMatrix ?: throw AssertionError("cool matrix must exist"),
            warm.calibration ?: throw AssertionError("warm calibration must exist"),
            cool.calibration ?: throw AssertionError("cool calibration must exist"),
            warm.forwardMatrix,
            cool.forwardMatrix,
            warm.kelvin,
            cool.kelvin
        ) ?: throw AssertionError("daylight transform must exist")

        // The codes a white object reports under D65, from ColorMatrix1 itself.
        val sceneWhite = greenNormalized(matMulVec(colorMatrixD65, D65_XYZ))

        // What the shader applies: gains first, then the color matrix.
        val balanced = floatArrayOf(
            sceneWhite[0] * t.wbGains[0],
            sceneWhite[1] * t.wbGains[1],
            sceneWhite[2] * t.wbGains[2]
        )
        val out = matMulVec(t.colorMatrix, balanced)

        val rg = out[0] / out[1]
        val bg = out[2] / out[1]
        assertEquals("D65 grey renders neutral (R/G $rg)", 1f, rg, 0.01f)
        assertEquals("D65 grey renders neutral (B/G $bg)", 1f, bg, 0.01f)
    }

    /**
     * The control that makes [daylightTransform_throughTheFixedPairing_rendersAD65GreyWhite]
     * mean something: feed the maths the association the buggy ordering produced
     * and show the cast reappears. Without this a permanently neutral result would
     * pass either way and prove nothing about the pairing.
     */
    @Test
    fun daylightTransform_withTheOldAssociation_stillCasts() {
        val t = RawColorMath.neutralTransformAtDaylight(
            colorMatrixD65,   // ColorMatrix1 (D65) held beside 2856 K...
            colorMatrixA,     // ...and ColorMatrix2 (A) beside 6504 K. The bug.
            calibration, calibration, forward, forward,
            2856f, 6504f
        ) ?: throw AssertionError("daylight transform must exist")

        val sceneWhite = greenNormalized(matMulVec(colorMatrixD65, D65_XYZ))
        val balanced = floatArrayOf(
            sceneWhite[0] * t.wbGains[0],
            sceneWhite[1] * t.wbGains[1],
            sceneWhite[2] * t.wbGains[2]
        )
        val out = matMulVec(t.colorMatrix, balanced)

        val rg = out[0] / out[1]
        val bg = out[2] / out[1]
        // The warm-heavy, blue-light signature the buggy ordering produces.
        // Asserted loosely enough to survive float noise, tightly enough that
        // "neutral" cannot sneak through.
        assertTrue("the old association is warm-heavy (R/G $rg)", rg > 1.2f)
        assertTrue("the old association is blue-light (B/G $bg)", bg < 0.9f)
        assertTrue(
            "the two associations differ at all (got $rg/$bg)",
            abs(rg - 1f) > 0.1f
        )
    }

    /**
     * The ordering makes the result a property of the pair, not of which slot the
     * vendor reported each illuminant in. Same profile, both reported orders, one
     * answer - and that answer is neutral.
     */
    @Test
    fun daylightTransform_isTheSameThroughOrderingFromEitherReportedOrder() {
        fun render(r1: RawColorProfile.ReferenceIlluminant,
                   r2: RawColorProfile.ReferenceIlluminant): Pair<Float, Float> {
            val (warm, cool) = RawColorProfile.orderReferences(r1, r2)
            val t = RawColorMath.neutralTransformAtDaylight(
                warm.colorMatrix ?: throw AssertionError("warm matrix must exist"),
                cool.colorMatrix ?: throw AssertionError("cool matrix must exist"),
                warm.calibration ?: throw AssertionError("warm calibration must exist"),
                cool.calibration ?: throw AssertionError("cool calibration must exist"),
                warm.forwardMatrix,
                cool.forwardMatrix,
                warm.kelvin,
                cool.kelvin
            ) ?: throw AssertionError("must exist")

            val sceneWhite = greenNormalized(matMulVec(colorMatrixD65, D65_XYZ))
            val out = matMulVec(
                t.colorMatrix,
                floatArrayOf(
                    sceneWhite[0] * t.wbGains[0],
                    sceneWhite[1] * t.wbGains[1],
                    sceneWhite[2] * t.wbGains[2]
                )
            )
            return (out[0] / out[1]) to (out[2] / out[1])
        }

        val (rgReported, bgReported) = render(reportedD65, reportedA)
        val (rgReversed, bgReversed) = render(reportedA, reportedD65)

        assertEquals("R/G does not depend on the reported slot", rgReported, rgReversed, 1e-3f)
        assertEquals("B/G does not depend on the reported slot", bgReported, bgReversed, 1e-3f)

        // And the answer itself, since a pair of identical wrong answers would
        // still satisfy the two assertions above.
        assertEquals("D65 grey renders neutral (R/G $rgReported)", 1f, rgReported, 0.01f)
        assertEquals("D65 grey renders neutral (B/G $bgReported)", 1f, bgReported, 0.01f)
    }

    /**
     * The device that started this: both reference illuminants reported as
     * 2856 K while ColorMatrix1 is the D65 matrix. There is nothing to order, so
     * the daylight read has to fall back to the primary matrix rather than to
     * whichever slot the reciprocal-temperature weight happens to land on.
     *
     * The two candidate answers are far apart here - 1/(ColorMatrix1 @ D65) is
     * [2.100, 1, 1.441] and 1/(ColorMatrix2 @ D65) is [2.445, 1, 1.321] - and the
     * second one is the yellow this whole investigation started from, so the test
     * asserts the value rather than only the neutrality it produces.
     */
    @Test
    fun daylightTransform_withBothReferencesAtTheSameTemperature_readsColorMatrix1() {
        val t = RawColorMath.neutralTransformAtDaylight(
            colorMatrixD65, colorMatrixA, calibration, calibration, forward, forward,
            2856f, 2856f
        ) ?: throw AssertionError("daylight transform must exist")

        // The codes a white object reports under D65, from ColorMatrix1 itself.
        val sceneWhite = greenNormalized(matMulVec(colorMatrixD65, D65_XYZ))
        assertEquals("daylight gains are 1/(ColorMatrix1 @ D65)",
            sceneWhite[1] / sceneWhite[0], t.wbGains[0], 1e-3f)
        assertEquals("daylight gains are 1/(ColorMatrix1 @ D65)",
            sceneWhite[1] / sceneWhite[2], t.wbGains[2], 1e-3f)
        assertEquals("red gain", 2.100f, t.wbGains[0], 5e-3f)
        assertEquals("blue gain", 1.441f, t.wbGains[2], 5e-3f)
        assertTrue("the read must not land on the illuminant-A matrix (${t.wbGains[0]})",
            t.wbGains[0] < 2.3f)

        val balanced = floatArrayOf(
            sceneWhite[0] * t.wbGains[0],
            sceneWhite[1] * t.wbGains[1],
            sceneWhite[2] * t.wbGains[2]
        )
        val out = matMulVec(t.colorMatrix, balanced)
        assertEquals("D65 grey renders neutral (R/G)", 1f, out[0] / out[1], 0.01f)
        assertEquals("D65 grey renders neutral (B/G)", 1f, out[2] / out[1], 0.01f)
    }

    /**
     * The same fallback on the path AUTO measures a scene chromaticity through:
     * a neutral pushed through the D65 camera->XYZ map has to come back at D65,
     * which is only true if the map was built from the D65 matrix. A map built
     * from the illuminant-A matrix answers the sensor's D65 response with A's own
     * chromaticity, and every estimate from there is anchored on the wrong light.
     */
    @Test
    fun cameraToXyzAtDaylight_withBothReferencesAtTheSameTemperature_stillAnchorsOnD65() {
        val cameraToXyz = RawColorMath.cameraToXyzAtDaylight(
            colorMatrixD65, colorMatrixA, calibration, calibration, forward, forward,
            2856f, 2856f
        ) ?: throw AssertionError("camera to XYZ must exist")

        val sceneWhite = greenNormalized(matMulVec(colorMatrixD65, D65_XYZ))
        val xyz = matMulVec(cameraToXyz, sceneWhite)
        val xy = RawColorMath.xyFromXyz(xyz) ?: throw AssertionError("must be a chromaticity")

        assertEquals("D65 x", D65_XY[0], xy[0], 2e-3f)
        assertEquals("D65 y", D65_XY[1], xy[1], 2e-3f)
    }

    private val D65_XY = floatArrayOf(0.3127f, 0.3290f)
}
