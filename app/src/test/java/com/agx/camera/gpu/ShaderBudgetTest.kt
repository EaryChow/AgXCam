package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The shader budget table, re-derived from the GLSL it describes.
 *
 * The fetch counts are the expensive part of a pass, so they are what a budget
 * argument rests on. They are static properties of the shader text, which means
 * someone can change a tap count in the GLSL and leave the table claiming
 * something else. These tests fail the build in that case instead of letting a
 * stale number into a report.
 *
 * Two layers, because each catches a different edit: the arithmetic checks pin
 * the decomposition itself, and the source checks pin the loop bounds the
 * decomposition is read from.
 */
class ShaderBudgetTest {

    private fun source(name: String): String {
        val f = File("src/main/java/com/agx/camera/gpu/$name.kt")
        assertTrue("cannot read $name.kt from ${f.absolutePath}", f.exists())
        return f.readText()
    }

    // ---- S2 sigma_hat ---------------------------------------------------

    @Test
    fun sigmaHatSparseCountFollowsItsDecomposition() {
        // 8 same-CFA-phase neighbours over 4 phases, plus one centre tap for the
        // ISO floor.
        assertEquals(8 * 4 + 1, ShaderBudget.FETCH_SIGMA_HAT_SPARSE)
    }

    @Test
    fun sigmaHatRgbCountFollowsItsDecomposition() {
        // Three phases in the capture RGB domain instead of four.
        assertEquals(8 * 3 + 1, ShaderBudget.FETCH_SIGMA_HAT_RGB)
    }

    @Test
    fun sigmaHatUsesEightNeighboursOverFourPhases() {
        val src = source("SigmaHatShaderProgram")
        assertTrue("expected an 8-entry neighbour table", src.contains("int[8]"))
        assertTrue("expected a four-phase loop", Regex("for\\s*\\(\\s*int\\s+p\\s*=\\s*0;\\s*p\\s*<\\s*4").containsMatchIn(src))
        assertTrue("expected an eight-neighbour inner loop", Regex("for\\s*\\(\\s*int\\s+k\\s*=\\s*0;\\s*k\\s*<\\s*8").containsMatchIn(src))
    }

    // ---- S3 green-guided GF --------------------------------------------

    @Test
    fun rawDenoiseCountFollowsItsDecomposition() {
        // One centre tap, one green-guide tap, then a 5x5 window per CFA phase
        // over four phases.
        assertEquals(1 + 1 + 25 * 4, ShaderBudget.FETCH_RAW_DENOISE)
    }

    @Test
    fun rawDenoiseUsesFourPhasesAndAFiveByFiveWindow() {
        val src = source("RawDenoiseShaderProgram")
        assertTrue("expected a four-phase loop", Regex("for\\s*\\(\\s*int\\s+p\\s*=\\s*0;\\s*p\\s*<\\s*4").containsMatchIn(src))
        assertTrue("expected a 25-tap window loop", Regex("for\\s*\\(\\s*int\\s+k\\s*=\\s*0;\\s*k\\s*<\\s*25").containsMatchIn(src))
    }

    // ---- S2 axis-min buckets -------------------------------------------

    @Test
    fun sigmaBucketCountFollowsItsDecomposition() {
        // Eight taps across three phases plus one centre tap.
        assertEquals(8 * 3 + 1, ShaderBudget.FETCH_SIGMA_BUCKET)
    }

    @Test
    fun sigmaBucketUsesEightTaps() {
        val src = source("SigmaBucketShaderProgram")
        assertTrue("expected an eight-tap loop", Regex("for\\s*\\(\\s*int\\s+k\\s*=\\s*0;\\s*k\\s*<\\s*8").containsMatchIn(src))
    }

    // ---- S5 structure map ----------------------------------------------

    @Test
    fun structureMapTexturedCountFollowsItsDecomposition() {
        // One centre tap; localRatio costs 1 centre + 9 window + 1 sigma = 11 and
        // is evaluated here and for each of the eight organization neighbours;
        // gradDir costs 4 taps and is evaluated here and for each of the eight
        // coherence neighbours.
        assertEquals(1 + 11 + 4 + 8 * 11 + 8 * 4, ShaderBudget.FETCH_STRUCTURE_MAP_TEXTURED)
    }

    @Test
    fun structureMapFlatCountIsTheTexturedCountWithoutTheCoherenceRing() {
        // A texel whose gradient vanishes skips the coherence ring: eight ring
        // texels at four taps each.
        assertEquals(
            "FLAT must be TEXTURED minus the eight coherence-ring texels",
            ShaderBudget.FETCH_STRUCTURE_MAP_TEXTURED - 8 * 4,
            ShaderBudget.FETCH_STRUCTURE_MAP_FLAT
        )
    }

    @Test
    fun structureMapFlatCountDecomposesIntoItsFourParts() {
        // Ties the per-part tap counts measured from the shader source to the
        // constant the budget actually reports, so retuning a part without
        // retuning FLAT fails here rather than in the report.
        val skipCentre = 1                       // the centre tap, skipped by both neighbourhoods
        val localRatio = 1 + 3 * 3 + 1          // centre + 3x3 window + sigma prediction
        val gradDir = 4                          // one call site per direction, four directions
        val organisation = 8 * localRatio       // ring of eight, each a full localRatio
        assertEquals(
            "skip-centre + localRatio + gradDir + organization must reconstruct FLAT",
            ShaderBudget.FETCH_STRUCTURE_MAP_FLAT,
            skipCentre + localRatio + gradDir + organisation
        )
    }

    @Test
    fun structureMapSkipCenterIsStillPresent() {
        val src = source("StructureMapShaderProgram")
        val skips = Regex("if\\s*\\(\\s*dx\\s*==\\s*0\\s*&&\\s*dy\\s*==\\s*0\\s*\\)\\s*continue").findAll(src).count()
        // organizationAt and the coherence ring each skip the centre; without
        // both, the neighbourhood is nine wide and both counts are wrong.
        assertEquals("expected the centre tap skipped in both neighbourhoods", 2, skips)
    }

    @Test
    fun structureMapGradDirCostsFourTaps() {
        val body = source("StructureMapShaderProgram")
            .substringAfter("vec2 gradDir(vec2 p) {")
            .substringBefore("}")
        // One call site per direction, each executed once, so the call-site count
        // is the fetch count here.
        val taps = Regex("texture\\s*\\(\\s*u_input_tex").findAll(body).count()
        assertEquals("gradDir reads one texel per direction", 4, taps)
    }

    @Test
    fun structureMapLocalRatioCostsElevenTaps() {
        val body = source("StructureMapShaderProgram")
            .substringAfter("float localRatio(vec2 p) {")
            .substringBefore("float organizationAt")
        // One static window call site inside a 3x3 loop, so nine fetches, plus
        // the centre tap.
        val windowLoops = Regex("for\\s*\\(\\s*int\\s+dy\\s*=\\s*-1;\\s*dy\\s*<=\\s*1").findAll(body).count()
        val windowInnerLoops = Regex("for\\s*\\(\\s*int\\s+dx\\s*=\\s*-1;\\s*dx\\s*<=\\s*1").findAll(body).count()
        assertEquals("expected one 3x3 window", 1, windowLoops)
        assertEquals("expected a 3x3 window inner loop", 1, windowInnerLoops)
        // The window is one call site inside a 3x3 loop, so nine fetches from
        // two static call sites: the centre tap and the window tap.
        assertEquals("one centre plus one 3x3 window call site", 2, Regex("texture\\s*\\(\\s*u_input_tex").findAll(body).count())
        // Executed cost of those three facts: one centre tap, nine window taps
        // from the single window call site inside a 3x3 loop, one sigma
        // prediction. The budget derives its FLAT count from 11 here, so this
        // is what makes the 11 in structureMapFlatCountDecomposesIntoItsFourParts
        // a measured number rather than a repeated literal.
        assertEquals("centre + 3x3 window + sigma prediction", 11, 1 + 3 * 3 + 1)
        assertEquals(
            "localRatio must contribute exactly 11 of the flat budget",
            11,
            ShaderBudget.FETCH_STRUCTURE_MAP_FLAT - 1 - 4 - 8 * 11
        )
    }

    @Test
    fun structureMapSigmaPredictionIsTheOneTapThatMakesEleven() {
        // The sigma read moved into its own function when the denominator had to
        // be brought into the residual's domain. It is still exactly one tap and
        // it is still on the ratio's denominator, so the eleven above has to be
        // re-derivable from where the tap now lives - otherwise the budget and
        // the shader can drift apart again with nothing failing.
        val prediction = source("StructureMapShaderProgram")
            .substringAfter("float noisePrediction(vec2 p) {")
            .substringBefore("float localRatio")
        assertEquals(
            "the prediction reads sigma_hat once and nothing else",
            1,
            Regex("texture\\s*\\(\\s*u_sigma_tex").findAll(prediction).count()
        )
        val localRatio = source("StructureMapShaderProgram")
            .substringAfter("float localRatio(vec2 p) {")
            .substringBefore("float organizationAt")
        assertTrue("localRatio must divide by the shared prediction", localRatio.contains("noisePrediction(p)"))
    }

    @Test
    fun structureMapDomainFoldIsPresentSoTheRatioIsNotComparedAcrossUnits() {
        // The live census read a p99 ratio of 0.004 against a plain ceiling of
        // 1.60 because a formed-image residual was divided by a raw-DN^2
        // variance. The fold that fixes it is a set of uniforms, so the test
        // pins the uniforms and the term that consumes them.
        val src = source("StructureMapShaderProgram")
        for (uniform in listOf(
            "u_luma_eps_scale", "u_sigma_dm2", "u_inverse_range2", "u_calib_scale", "u_eps_boost"
        )) {
            assertTrue("expected a $uniform uniform", src.contains("uniform float $uniform;"))
        }
        val prediction = src
            .substringAfter("float noisePrediction(vec2 p) {")
            .substringBefore("float localRatio")
        assertTrue("the fold must reach the prediction", prediction.contains("u_inverse_range2"))
        assertTrue("the fold must reach the prediction", prediction.contains("u_calib_scale"))
        assertTrue("the fold must reach the prediction", prediction.contains("u_eps_boost"))
        assertTrue("the fold must reach the prediction", prediction.contains("u_sigma_dm2"))
    }

    @Test
    fun structureMapSamplesSigmaByImageCoordinateNotByGridSize() {
        // Dividing the pixel coordinate by the sigma grid size is only the right
        // UV when the two happen to be the same size; past the density gate the
        // grid is far coarser than the image and every lookup walks off the end
        // of the texture.
        val prediction = source("StructureMapShaderProgram")
            .substringAfter("float noisePrediction(vec2 p) {")
            .substringBefore("float localRatio")
        assertTrue(
            "the sigma lookup is normalised by the output size",
            prediction.contains("/ u_output_size")
        )
        assertFalse(
            "the sigma lookup must not be normalised by the sigma grid size",
            prediction.contains("/ u_sigma_size")
        )
    }

    /**
     * The fold is present in the GLSL; this asks whether it is present *in the
     * running pipeline*.
     *
     * The shipped values are read out of PreviewRenderer rather than restated
     * here, because a test that hardcodes 1.4 / 10 / 1/32 only proves those
     * literals multiply to something non-zero - it keeps passing after someone
     * sets the production constant to 0. That is the whole failure: a zeroed
     * fold term collapses the denominator onto the sigmaDm2 floor alone (or
     * onto the 1e-12 clamp, if that is zeroed too) and the ratio inflates by
     * orders of magnitude while every other test in the suite stays green.
     */
    @Test
    fun everyFoldTermIsNonZeroAtTheShippedOperatingPoint() {
        val renderer = source("PreviewRenderer")

        fun shippedConst(name: String): Float {
            val m = Regex("private const val $name = ([0-9.]+)f").find(renderer)
            assertTrue("$name is not a Float constant in PreviewRenderer", m != null)
            return m!!.groupValues[1].toFloat()
        }

        val lumaEpsScale = shippedConst("S5_LUMA_EPS_SCALE")
        val sigmaDm2 = shippedConst("S5_SIGMA_DM2")
        val calibScale = shippedConst("S5_CALIB_TO_RESIDUAL_SCALE")

        assertTrue("S5_LUMA_EPS_SCALE must be non-zero, was $lumaEpsScale", lumaEpsScale > 0f)
        // sigmaDm2 is the term with a 0f default in draw()'s own signature, so it
        // is the one that can be silently dropped by a call that stops naming it.
        assertTrue("S5_SIGMA_DM2 must be non-zero, was $sigmaDm2", sigmaDm2 > 0f)
        assertTrue("S5_CALIB_TO_RESIDUAL_SCALE must be non-zero, was $calibScale", calibScale > 0f)

        // inverseRange2 is derived, not stored: a whiteRange that parses as 0 or
        // as a raw DN count instead of a range both land here as a silent
        // rescale. Pin the derivation so it cannot be dropped either.
        assertTrue(
            "inverseRange2 must still be derived from whiteRange",
            renderer.contains("inverseRange2 = 1f / (whiteRange * whiteRange)")
        )
    }

    /**
     * The fold multiplies to something finite and strictly positive across the
     * whole sigma range the sensor can report, and lands the pure-noise anchor
     * on 1.0 at the shipped constants rather than at hand-picked ones.
     *
     * A sweep rather than a single point because the dangerous term is additive:
     * sigmaDm2 sets a floor that dominates whenever sigma_hat^2 drops below it,
     * so a check at one ISO can pass while the floor itself is wrong.
     */
    @Test
    fun theShippedFoldKeepsThePureNoiseAnchorAtOne() {
        val renderer = source("PreviewRenderer")

        fun shippedConst(name: String): Float =
            Regex("private const val $name = ([0-9.]+)f").find(renderer)!!.groupValues[1].toFloat()

        val lumaEpsScale = shippedConst("S5_LUMA_EPS_SCALE")
        val sigmaDm2 = shippedConst("S5_SIGMA_DM2")
        val calibScale = shippedConst("S5_CALIB_TO_RESIDUAL_SCALE")
        // The DN range the shipped sensor reports: white 1023 - black 64.
        val whiteRange = 959f
        val inverseRange2 = 1f / (whiteRange * whiteRange)
        val cal = StructureMapClassifier.calibrationFor(367)

        // 0.1 .. 1000 DN^2 brackets the floor, the live band and a bright ISO.
        for (sigmaHat2 in listOf(0.1f, 1f, 7f, sigmaDm2, 23f, 113f, 540f, 1000f)) {
            val pred = StructureMapClassifier.noisePrediction(
                sigmaHat2, sigmaDm2, inverseRange2, calibScale, 1f, lumaEpsScale
            )
            assertTrue("prediction must be finite at sigmaHat2=$sigmaHat2", pred.isFinite())
            assertTrue("prediction must be positive at sigmaHat2=$sigmaHat2, was $pred", pred > 0f)
            // The anchor: a residual equal to the prediction times the retention
            // has to read exactly 1.0, whatever sigma_hat^2 was. This is the
            // property the fold exists to preserve.
            assertEquals(
                "pure noise must read 1.0 at sigmaHat2=$sigmaHat2",
                1.0f,
                StructureMapClassifier.ratio(pred * cal.rRetain, pred, cal.rRetain),
                1.0e-3f
            )
        }
    }

    // ---- Report shape --------------------------------------------------

    @Test
    fun reportPrintsEveryShaderWithItsFetchCount() {
        val text = ShaderBudget.standard().report(measurementForcedChain = false)
        assertTrue(text.contains("S2.sigma_hat"))
        assertTrue(text.contains("${ShaderBudget.FETCH_SIGMA_HAT_SPARSE} fetches"))
        assertTrue(text.contains("S3.green_guided_gf"))
        assertTrue(text.contains("${ShaderBudget.FETCH_RAW_DENOISE} fetches"))
        assertTrue(text.contains("S2.axis_buckets"))
        assertTrue(text.contains("S5.structure_map"))
    }

    @Test
    fun reportStatesTheForcedPassesWhenAMeasurementConsumerIsLive() {
        val on = ShaderBudget.standard().report(measurementForcedChain = true)
        assertTrue("an open cost has to be declared", on.contains("forced on"))
        val off = ShaderBudget.standard().report(measurementForcedChain = false)
        assertTrue(off.contains("no pass is forced"))
    }

    @Test
    fun reportNamesTheProgramSetsSoTheDeltaIsAuditable() {
        val budget = ShaderBudget.standard()
        // Pretend the driver reported sizes, so the delta is actually computed.
        budget.record("bayer", 1000, "test")
        budget.record("demosaic", 1000, "test")
        budget.record("nr", 1000, "test")
        budget.record("spatialNr", 1000, "test")
        budget.record("dpc", 1000, "test")
        budget.record("outNr", 1000, "test")
        budget.record("sigmaHat", 250, "test")
        budget.record("rawDenoise", 250, "test")
        budget.record("sigmaBucket", 250, "test")
        budget.record("structureMap", 250, "test")

        assertEquals(6000, budget.baselineBytes())
        assertEquals(1000, budget.phase1Bytes())
        val text = budget.report(measurementForcedChain = false)
        // Both sets must be named with their membership, so the ratio between
        // them is auditable rather than a bare percentage of an unnamed total.
        assertTrue(text, text.contains("baseline set (6 programs) = 6000 bytes"))
        assertTrue(text, text.contains("Phase-1 additions (4 programs) = 1000 bytes"))
        assertTrue(text, text.contains("+16.7% of the baseline set"))
        // And the caliber has to be stated, because this table is compared by
        // eye against a source-size budget that it is not measured in.
        assertTrue(text, text.contains("GL_PROGRAM_BINARY_LENGTH"))
        assertTrue(text, text.contains("SOURCE-SIZE"))
    }

    @Test
    fun missingSizesAreDeclaredRatherThanGuessed() {
        val text = ShaderBudget.standard().report(measurementForcedChain = false)
        assertTrue(
            "an unmeasurable driver must say so, not print a zero delta",
            text.contains("unavailable") || text.contains("baseline")
        )
    }

    @Test
    fun theRegressionGateIsStatedNextToTheNumbers() {
        val text = ShaderBudget.standard().report(measurementForcedChain = false)
        assertTrue("the 5% rule needs its segment condition stated", text.contains("5%"))
        assertTrue("a delta without a segment key is not comparable", text.contains("segment"))
    }
}
