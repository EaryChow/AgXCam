package com.agx.camera.camera

import com.agx.camera.color.ColorMatrix
import com.agx.camera.color.ColorMatrix.Mat3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * A white balance mode is a statement about the scene illuminant, so the mode's
 * chromaticity is the one number it contributes and everything else in the render
 * derives from it. These tests pin that number where it is a standard, and pin the
 * consequences that make the table more than decoration.
 */
class WhiteBalanceSceneXyTest {

    private val EPS = 1e-4f

    private fun matMulVec(m: Mat3, v: FloatArray): FloatArray = floatArrayOf(
        m.m[0] * v[0] + m.m[1] * v[1] + m.m[2] * v[2],
        m.m[3] * v[0] + m.m[4] * v[1] + m.m[5] * v[2],
        m.m[6] * v[0] + m.m[7] * v[1] + m.m[8] * v[2]
    )

    private fun xyOf(mode: WhiteBalanceMode): FloatArray {
        val xy = mode.sceneXy()
        assertNotNull("${mode.name} must name an illuminant", xy)
        return xy!!
    }

    @Test
    fun everyPresetNamesAnIlluminant() {
        // The six presets are the user declaring the light. AUTO measures it and
        // KELVIN parametrizes it, so those two are the only modes with no constant.
        val presets = listOf(
            WhiteBalanceMode.DAYLIGHT,
            WhiteBalanceMode.CLOUDY,
            WhiteBalanceMode.INCANDESCENT,
            WhiteBalanceMode.FLUORESCENT,
            WhiteBalanceMode.TWILIGHT,
            WhiteBalanceMode.SHADE
        )
        assertEquals(presets.size, WhiteBalanceMode.entries.size - 2)
        for (mode in presets) {
            val xy = xyOf(mode)
            assertTrue("${mode.name} x in gamut", xy[0] > 0f && xy[0] < 1f)
            assertTrue("${mode.name} y in gamut", xy[1] > 0f && xy[1] < 1f)
            assertTrue("${mode.name} inside the chromaticity triangle", xy[0] + xy[1] < 1f)
        }
        assertNull(WhiteBalanceMode.AUTO.sceneXy())
        assertNull(WhiteBalanceMode.KELVIN.sceneXy())
    }

    @Test
    fun groundedPresetsMatchTheCieTable() {
        // These four follow the illuminant mapping the DNG specification gives
        // for these light sources, so the HAL's preset gains and this
        // chromaticity describe the same light. CIE 1931 2-degree standard
        // observer.
        val expected = mapOf(
            WhiteBalanceMode.INCANDESCENT to floatArrayOf(0.44757f, 0.40745f), // A
            WhiteBalanceMode.FLUORESCENT to floatArrayOf(0.37208f, 0.37529f),  // F2
            WhiteBalanceMode.CLOUDY to floatArrayOf(0.31006f, 0.31616f),       // C
            WhiteBalanceMode.SHADE to floatArrayOf(0.29902f, 0.31485f)         // D75
        )
        for ((mode, xy) in expected) {
            val got = xyOf(mode)
            assertEquals("${mode.name} x", xy[0], got[0], EPS)
            assertEquals("${mode.name} y", xy[1], got[1], EPS)
        }
        // DAYLIGHT is checked against the pipeline's own anchor rather than the CIE
        // digits, so that the daylight preset is exactly the identity adaptation.
        assertEquals(ColorMatrix.D65_X, xyOf(WhiteBalanceMode.DAYLIGHT)[0], 0f)
        assertEquals(ColorMatrix.D65_Y, xyOf(WhiteBalanceMode.DAYLIGHT)[1], 0f)
    }

    @Test
    fun daylightIsTheD65AnchorSoTheAdaptationIsTheIdentity() {
        // The whole pipeline renders onto D65, so the daylight preset adapting to
        // D65 has to be the identity. If this drifts, a neutral daylight scene
        // acquires a cast in the one mode that is supposed to be the reference.
        assertEquals(ColorMatrix.D65_X, xyOf(WhiteBalanceMode.DAYLIGHT)[0], EPS)
        assertEquals(ColorMatrix.D65_Y, xyOf(WhiteBalanceMode.DAYLIGHT)[1], EPS)
    }

    @Test
    fun presetsRunCoolerThanDaylightInTheOrderTheScenesRead() {
        // Not a colorimetry rule - a claim about the table being internally ordered.
        // Overcast, open shade and blue hour all read cooler than direct daylight,
        // and each step in the UI goes further from D65. Tungsten and fluorescent are
        // warm instead, which is the whole point of reaching for them.
        val d65y = xyOf(WhiteBalanceMode.DAYLIGHT)[1]
        assertTrue("cloudy below daylight", xyOf(WhiteBalanceMode.CLOUDY)[1] < d65y)
        assertTrue("shade below cloudy", xyOf(WhiteBalanceMode.SHADE)[1] < xyOf(WhiteBalanceMode.CLOUDY)[1])
        assertTrue("twilight below shade", xyOf(WhiteBalanceMode.TWILIGHT)[1] < xyOf(WhiteBalanceMode.SHADE)[1])
        assertTrue("tungsten above daylight", xyOf(WhiteBalanceMode.INCANDESCENT)[1] > d65y)
        assertTrue("fluorescent above daylight", xyOf(WhiteBalanceMode.FLUORESCENT)[1] > d65y)
    }

    @Test
    fun eachPresetProducesADistinctAdaptation() {
        // Guards against two presets silently collapsing onto the same matrix, which
        // would make one of the UI entries a no-op. D65 is excluded because it is
        // the identity and is pinned as such by the test above.
        val seen = mutableMapOf<String, WhiteBalanceMode>()
        for (mode in WhiteBalanceMode.entries) {
            val xy = mode.sceneXy() ?: continue
            val key = "${xy[0]},${xy[1]}"
            assertNull("${mode.name} duplicates ${seen[key]}", seen[key])
            seen[key] = mode
        }
        assertEquals(6, seen.size)
    }

    @Test
    fun presetIlluminantDrivesTheMatrixAndTheGainsStayTheHalOnes() {
        // The mechanism the presets now share with AUTO: a known scene chromaticity
        // plus the sensor's response to it, with white balance split out. Tungsten
        // under CIE A is the case that matters - it is far enough from D65 that
        // using D65 would leave a large, obvious cast.
        val a = xyOf(WhiteBalanceMode.INCANDESCENT)
        val d65 = xyOf(WhiteBalanceMode.DAYLIGHT)

        // A sensor whose D65 response is the daylight white. Under illuminant A it
        // reports a warmer, redder neutral - this is what the HAL's gains encode.
        val cm = referenceMatrix()
        val daylightWhite = matMulVec(cm, xyzFromXy(ColorMatrix.D65_X, ColorMatrix.D65_Y))
        val tungstenWhite = matMulVec(cm, xyzFromXy(a[0], a[1]))

        val t = RawColorMath.neutralTransformAtSceneXy(
            cm, cm, IDENTITY, IDENTITY, null, null, 5003f, 7504f,
            tungstenWhite, a
        ) ?: throw AssertionError("must exist")

        // The gains whiten the sensor's response to the preset illuminant...
        val level = floatArrayOf(
            tungstenWhite[0] * t.wbGains[0],
            tungstenWhite[1] * t.wbGains[1],
            tungstenWhite[2] * t.wbGains[2]
        )
        assertEquals(level[1], level[0], 1e-3f)
        assertEquals(level[1], level[2], 1e-3f)

        // ...and the matrix puts that level on the output white, which is the CAT
        // from the preset's illuminant onto D65.
        val out = matMulVec(t.colorMatrix, level)
        assertTrue("tungsten renders white (spread ${out.max() - out.min()})", out.max() - out.min() < 0.02f)

        // The decisive check: the tungsten matrix is not the daylight matrix. If the
        // preset were being treated as "any light" or the table were being ignored,
        // these would be equal and tungsten would render with daylight's cast.
        val d = RawColorMath.neutralTransformAtSceneXy(
            cm, cm, IDENTITY, IDENTITY, null, null, 5003f, 7504f,
            daylightWhite, d65
        ) ?: throw AssertionError("must exist")
        var maxDiff = 0f
        for (i in 0..8) maxDiff = maxOf(maxDiff, abs(t.colorMatrix.m[i] - d.colorMatrix.m[i]))
        assertTrue("tungsten and daylight matrices differ (max $maxDiff)", maxDiff > 0.05f)
    }

    @Test
    fun presetSolvedAgainstTheDaylightWhite_keepsTheGainsFixedAndStillRendersTheScene() {
        // The pairing the presets actually use: the profile's own daylight white
        // as the neutral, so the gains are the sensor's green-bias removal and do
        // not move with the scene, and the mode's chromaticity is what decides
        // the matrix. The scene this is checked against is lit by the preset, so
        // a neutral under it has to come out neutral - the split changed, the
        // render did not.
        val cm = referenceMatrix()
        val daylightWhite = matMulVec(cm, xyzFromXy(ColorMatrix.D65_X, ColorMatrix.D65_Y))
        val a = xyOf(WhiteBalanceMode.INCANDESCENT)
        val tungstenWhite = matMulVec(cm, xyzFromXy(a[0], a[1]))

        val t = RawColorMath.neutralTransformAtSceneXy(
            cm, cm, IDENTITY, IDENTITY, null, null, 5003f, 7504f,
            daylightWhite, a
        ) ?: throw AssertionError("must exist")

        // The gains whiten the sensor's daylight white, not its answer to the
        // scene: the same numbers the daylight transform would solve for.
        val day = RawColorMath.neutralTransformAtDaylight(
            cm, cm, IDENTITY, IDENTITY, null, null, 5003f, 7504f
        ) ?: throw AssertionError("must exist")
        for (i in 0..2) {
            assertEquals("gain $i", day.wbGains[i], t.wbGains[i], 1e-4f)
        }

        // And the sensor's response to the preset illuminant, through those fixed
        // gains and the preset's matrix, still renders white.
        val level = floatArrayOf(
            tungstenWhite[0] * t.wbGains[0],
            tungstenWhite[1] * t.wbGains[1],
            tungstenWhite[2] * t.wbGains[2]
        )
        val out = matMulVec(t.colorMatrix, level)
        assertTrue(
            "tungsten renders white (spread ${out.max() - out.min()})",
            out.max() - out.min() < 0.02f
        )
    }

    @Test
    fun daylightPresetIsExactlyTheDaylightTransform() {
        // The zero point again, at the mode level: DAYLIGHT + the sensor's D65
        // white has to reproduce the pinned daylight transform bit for bit, because
        // that is what "anchored on D65" has to mean operationally.
        val cm = referenceMatrix()
        val daylightWhite = matMulVec(cm, xyzFromXy(ColorMatrix.D65_X, ColorMatrix.D65_Y))
        val daylight = RawColorMath.neutralTransformAtDaylight(
            cm, cm, IDENTITY, IDENTITY, null, null, 5003f, 7504f
        ) ?: throw AssertionError("must exist")
        val viaMode = RawColorMath.neutralTransformAtSceneXy(
            cm, cm, IDENTITY, IDENTITY, null, null, 5003f, 7504f,
            daylightWhite, WhiteBalanceMode.DAYLIGHT.sceneXy()!!
        ) ?: throw AssertionError("must exist")
        for (i in 0..8) {
            assertEquals("matrix $i", daylight.colorMatrix.m[i], viaMode.colorMatrix.m[i], EPS)
        }
    }

    private val IDENTITY = ColorMatrix.identity()

    // A camera (XYZ -> camera) reference matrix that answers the D65 white with a
    // known chromatic response, red 0.45 / green 1 / blue 0.75. Built by
    // constructing that response and inverting it, so the fixture states the sensor
    // it is emulating instead of restating the profile's normalization. The presets
    // do not interpolate between two references, so one matrix is enough here.
    private fun referenceMatrix(): Mat3 {
        val d65 = xyzFromXy(ColorMatrix.D65_X, ColorMatrix.D65_Y)
        val response = floatArrayOf(0.45f * d65[0], 1.0f * d65[1], 0.75f * d65[2])
        return ColorMatrix.inverse(Mat3(
            floatArrayOf(
                1f / response[0], 0f, 0f,
                0f, 1f / response[1], 0f,
                0f, 0f, 1f / response[2]
            )
        ))
    }

    private fun xyzFromXy(x: Float, y: Float): FloatArray =
        floatArrayOf(x / y, 1f, (1f - x - y) / y)
}
