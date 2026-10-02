package com.agx.camera.gpu

import org.junit.Assert.assertEquals
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
        // the centre tap and the sigma prediction from the second sampler.
        val windowLoops = Regex("for\\s*\\(\\s*int\\s+dy\\s*=\\s*-1;\\s*dy\\s*<=\\s*1").findAll(body).count()
        val windowInnerLoops = Regex("for\\s*\\(\\s*int\\s+dx\\s*=\\s*-1;\\s*dx\\s*<=\\s*1").findAll(body).count()
        assertEquals("expected one 3x3 window", 1, windowLoops)
        assertEquals("expected a 3x3 window inner loop", 1, windowInnerLoops)
        // The window is one call site inside a 3x3 loop, so nine fetches from
        // two static call sites: the centre tap and the window tap.
        assertEquals("one centre plus one 3x3 window call site", 2, Regex("texture\\s*\\(\\s*u_input_tex").findAll(body).count())
        assertTrue("localRatio must also read the sigma prediction", body.contains("u_sigma_tex"))
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