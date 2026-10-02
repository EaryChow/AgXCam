package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The pipeline state block, tested without a GL context.
 *
 * This exists because an empty measurement section is otherwise undecidable from
 * the report alone: "no samples yet" reads the same whether the pass never ran,
 * the gate that starts it was closed, or the benchmark was off. Only the gate
 * truth separates those, so the block has to state both the switch truth and the
 * pass truth, and the two must not be the same variable.
 */
class PipelineGatesTest {

    private fun gates(
        s1Raw: Int = 0, s3Raw: Int = 0, s5Raw: Int = 0,
        zoomK: Float = 6.4f,
        needSparseGrid: Boolean = true,
        sparseGridRendered: Boolean = false,
        outDenoiseActive: Boolean = true,
        passSigma: Boolean = true,
        passSwgf: Boolean = false,
        passOutNr: Boolean = true,
        passBucket: Boolean = true,
        bucketDrawn: Boolean = false,
        passStructureMap: Boolean = true,
        sigmaSource: PipelineGates.SigmaSource = PipelineGates.SigmaSource.OFFSCREEN_CHAIN,
        lumaSource: PipelineGates.LumaSource = PipelineGates.LumaSource.S5_INLINE,
        thermalStatus: Int = 0,
        appThermalTier: String = "NORMAL",
        viewfinderFps: Float = 29.7f,
        structureMapDrawn: Boolean = false,
        sparse: Boolean = sparseGridRendered,
        w: Int = 640,
        h: Int = 480
    ) = PipelineGates(
        s1Raw = s1Raw, s3Raw = s3Raw, s5Raw = s5Raw,
        s1Strength = s1Raw / 100f, s3Strength = s3Raw / 100f, s5Strength = s5Raw / 100f,
        outDenoiseActive = outDenoiseActive,
        needSparseGrid = needSparseGrid,
        sparseGridRendered = sparseGridRendered,
        zoomK = zoomK,
        viewWidth = w, viewHeight = h,
        passSigma = passSigma, passSwgf = passSwgf, passOutNr = passOutNr,
        passBucket = passBucket, bucketDrawn = bucketDrawn,
        passStructureMap = passStructureMap,
        structureMapDrawn = structureMapDrawn,
        sigmaSource = sigmaSource, lumaSource = lumaSource,
        thermalStatus = thermalStatus,
        appThermalTier = appThermalTier,
        viewfinderFps = viewfinderFps,
        // Derived from the same inputs the renderer uses, so a test that varies
        // the gate fields gets the key those fields imply.
        segmentKey = PipelineGates.segmentKeyFor(
            s1Raw = s1Raw, s3Raw = s3Raw, s5Raw = s5Raw,
            sparseChainRegime = sparse, viewWidth = w, viewHeight = h,
            appThermalTier = appThermalTier,
            thermalPlatformThrottled = ThermalStatus.isThrottled(thermalStatus)
        )
    )

    @Test
    fun continuousValuesStayOutOfTheChangeDetectionString() {
        // shouldLog re-emits when stateText() differs from the last one, so a
        // value that moves every frame - the viewfinder rate - turns a
        // change-detected line into one line per frame. Nothing errors; the log
        // just fills with a state that never changed. Thermal level and the app
        // tier are discrete and belong in the string; the rate does not.
        val logger = PipelineStateLog()
        assertTrue(logger.shouldLog(gates(viewfinderFps = 29.7f)))
        assertFalse(
            "a moving viewfinder rate must not re-emit the pipeline state line",
            logger.shouldLog(gates(viewfinderFps = 29.2f))
        )
        assertFalse(logger.shouldLog(gates(viewfinderFps = 12.0f)))

        assertFalse(gates().stateText().contains("viewfinder"))
        assertTrue(gates().frameRateText().contains("29.7"))
        assertTrue(
            "the rate must say it is not a pass timing",
            gates().frameRateText().contains("not a GPU pass timing")
        )
    }

    @Test
    fun aThermalChangeDoesReEmitThePipelineStateLine() {
        val logger = PipelineStateLog()
        assertTrue(logger.shouldLog(gates(thermalStatus = 0, appThermalTier = "NORMAL")))
        assertTrue(
            "heating the device must produce a line on its own",
            logger.shouldLog(gates(thermalStatus = 3, appThermalTier = "WARM"))
        )
        assertTrue(logger.shouldLog(gates(thermalStatus = 4, appThermalTier = "HOT")))
    }

    @Test
    fun reportTextCarriesTheRateThatStateTextOmits() {
        val report = gates(viewfinderFps = 17.5f).reportText()
        assertTrue(report, report.contains("viewfinder fps=17.5"))
        assertTrue(report, report.contains("thermal=none"))
        assertTrue(report, report.contains("appTier=NORMAL"))
    }

    // ---- The block has to be self-describing ---------------------------

    @Test
    fun stateBlockNamesBothThermalSources() {
        // The app's own tier is the one that matters for reading a size change:
        // ThermalManager drops the preview resolution itself at WARM, so a
        // segment can change size with no change in the platform status.
        val cool = gates(thermalStatus = 0, appThermalTier = "NORMAL").stateText()
        assertTrue(cool, cool.contains("thermal=none"))
        assertTrue(cool, cool.contains("appTier=NORMAL"))
        assertFalse(cool, cool.contains("THROTTLED"))

        val hot = gates(thermalStatus = 4, appThermalTier = "HOT").stateText()
        assertTrue(hot, hot.contains("thermal=severe"))
        assertTrue(hot, hot.contains("THROTTLED"))
        assertTrue(hot, hot.contains("appTier=HOT"))
    }

    @Test
    fun theStructureMapNoteReportsEligibilityAndDrawingSeparately() {
        // Reporting only eligibility made the note claim a draw on the nineteen
        // frames in every twenty where there was none, while never saying the
        // pass was skipped. Both states are now stated.
        val drewIt = gates(passStructureMap = true, structureMapDrawn = true).structureMapPathNote()
        assertTrue(drewIt, drewIt.contains("drew this frame"))

        val eligibleNotDrawn =
            gates(passStructureMap = true, structureMapDrawn = false).structureMapPathNote()
        assertFalse(eligibleNotDrawn, eligibleNotDrawn.contains("drew this frame"))
        assertTrue(
            eligibleNotDrawn,
            eligibleNotDrawn.contains("not this frame's census frame")
        )
        assertTrue(eligibleNotDrawn, eligibleNotDrawn.contains("1/20"))

        val ineligible = gates(passStructureMap = false, structureMapDrawn = false)
            .structureMapPathNote()
        assertTrue(ineligible, ineligible.contains("not eligible this frame"))
        assertFalse(ineligible, ineligible.contains("census"))
    }

    @Test
    fun anArmedGrabForcesACensusFrame() {
        // Without this, 19 of every 20 grabs attached the previous census
        // frame's structure map while meta("frame") named the current one.
        // Mirrors the renderer's gate.
        fun censusDue(measurementActive: Boolean, grabArmed: Boolean, cadenceHit: Boolean) =
            (measurementActive || grabArmed) && (grabArmed || cadenceHit)

        assertTrue(
            "an armed grab is a census frame even off the cadence",
            censusDue(measurementActive = false, grabArmed = true, cadenceHit = false)
        )
        assertTrue(censusDue(measurementActive = true, grabArmed = false, cadenceHit = true))
        assertFalse(
            "an unarmed grab off the cadence is not a census frame",
            censusDue(measurementActive = false, grabArmed = false, cadenceHit = false)
        )
        assertFalse(
            "benchmark off and nothing armed must cost nothing",
            censusDue(measurementActive = false, grabArmed = false, cadenceHit = true)
        )
    }

    @Test
    fun passesTextMarksDrewThisFrameDistinctlyFromEligible() {
        // The bucket draw is census-gated, so eligibility is true on nineteen
        // frames in twenty where the pass did not run. Reporting it bare read
        // as "the bucket pass ran", which is the same conflation the structure
        // map had until it got its own drawn flag.
        val skipped = gates(passBucket = true, bucketDrawn = false).passesText()
        assertTrue(skipped, skipped.contains("bucket=true "))
        assertFalse("a cadence skip must not look like a draw", skipped.contains("bucket=true*"))
        assertTrue("the legend has to be present or the mark is unreadable", skipped.contains("* = drew this frame"))

        val drew = gates(passBucket = true, bucketDrawn = true).passesText()
        assertTrue(drew, drew.contains("bucket=true*"))

        val smapDrew = gates(passStructureMap = true, structureMapDrawn = true).passesText()
        assertTrue(smapDrew, smapDrew.contains("smap=true*"))
    }

    @Test
    fun cadenceGatedPassesAgreeWithTheNoteThatExplainsThem() {
        // Two surfaces report the same two facts about the same frame; if they
        // can disagree the report contradicts itself.
        val eligibleNotDrawn = gates(
            passBucket = true, bucketDrawn = false,
            passStructureMap = true, structureMapDrawn = false
        )
        val passes = eligibleNotDrawn.passesText()
        assertTrue(passes, passes.contains("bucket=true "))
        assertTrue(passes, passes.contains("smap=true "))
        assertTrue(
            eligibleNotDrawn.structureMapPathNote(),
            eligibleNotDrawn.structureMapPathNote().contains("not this frame's census frame")
        )
    }

    @Test
    fun segmentKeyCarriesBothThermalSources() {
        // The platform flag and the app tier are separate fields on purpose: the
        // app can be in WARM while the platform still reports none, and the tier
        // is the one that decides throttling.
        val platformHot = gates(thermalStatus = 3, appThermalTier = "NORMAL").segmentKey
        assertTrue(platformHot.platformThrottled)
        assertFalse(platformHot.isThrottled())

        val appWarm = gates(thermalStatus = 0, appThermalTier = "WARM").segmentKey
        assertFalse(appWarm.platformThrottled)
        assertTrue(appWarm.isThrottled())
    }

    @Test
    fun segmentKeyFollowsTheAppTierAndNotThePlatformStatus() {
        // Pins the regression: keying on the platform value left a WARM run
        // sharing NORMAL's segment whenever WARM only dropped the frame rate.
        val normal = gates(thermalStatus = 0, appThermalTier = "NORMAL").segmentKey
        val warm = gates(thermalStatus = 0, appThermalTier = "WARM").segmentKey
        assertFalse(normal == warm)
    }

    @Test
    fun theGatesKeyIsTheKeyItWereHandedRatherThanARebuiltOne() {
        // The report's segment list and this block have to describe one
        // segmentation. Since the key is passed in, that is now structural: this
        // asserts the block reports back exactly the key the frame entered, so a
        // future refactor that reintroduces a rebuild here fails rather than
        // silently describing different segmentation from the one timed.
        val handed = PipelineGates.segmentKeyFor(0, 32, 40, true, 640, 480, "WARM", false)
        assertEquals(handed, gates(s3Raw = 32, s5Raw = 40, sparse = true, appThermalTier = "WARM").segmentKey)
        assertEquals(
            "the handed key must survive verbatim, including a regime the block's own fields do not imply",
            "0-32-40-sp-640x480-WARM",
            handed.tag()
        )
    }

    @Test
    fun segmentKeyForIsTheSingleProducerAndAgreesWithTheGatesFields() {
        // The factory and the gate fields describe the same frame, so a key
        // built from the fields must equal one built by the factory. Before the
        // key became a constructor field these were two independent expressions
        // of the same rule, which is how they came to disagree.
        val built = gates(
            s1Raw = 12, s3Raw = 34, s5Raw = 56, sparse = true, w = 320, h = 240,
            thermalStatus = 0, appThermalTier = "HOT"
        )
        assertEquals(
            built.segmentKey,
            PipelineGates.segmentKeyFor(
                s1Raw = 12, s3Raw = 34, s5Raw = 56, sparseChainRegime = true,
                viewWidth = 320, viewHeight = 240, appThermalTier = "HOT",
                thermalPlatformThrottled = false
            )
        )
    }

    @Test
    fun aFrameGrabAloneCannotSwitchS5OntoASigmaEstimate() {
        // The blocking regression. Arming a grab counts as a measurement
        // consumer, so past the density gate (k>2) the offscreen chain runs and
        // sigma_hat produces an estimate. If that estimate could reach S5, then
        // checking the raw attachment would change the rendered frame's noise
        // model - with useIsoSigma flipping to false and nothing in the pixels
        // to say so.
        //
        // Mirrors the renderer's rule: S5 may consume sigma derived from the
        // image path's own S3 output, or from a measurement-only chain while
        // measurement is deliberately on. A grab alone is neither.
        assertTrue(
            "the image path's own chain may feed S5",
            sigmaReachesImagePath(sigmaAvailable = true, sparseGridRendered = true, measurementActive = false)
        )
        assertTrue(
            "measurement mode deliberately re-routes S5 and says so in the report",
            sigmaReachesImagePath(sigmaAvailable = true, sparseGridRendered = false, measurementActive = true)
        )
        assertFalse(
            "a grab alone must leave S5 on its ISO fallback",
            sigmaReachesImagePath(sigmaAvailable = true, sparseGridRendered = false, measurementActive = false)
        )
    }

    @Test
    fun anAbsentSigmaEstimateLeavesS5OnTheIsoFallback() {
        for (sparse in listOf(true, false)) {
            for (measuring in listOf(true, false)) {
                assertFalse(
                    "no estimate means ISO fallback regardless of mode",
                    sigmaReachesImagePath(sigmaAvailable = false, sparseGridRendered = sparse, measurementActive = measuring)
                )
            }
        }
    }

    /**
     * The renderer's S5 sigma gate, mirrored.
     *
     * Duplicated rather than shared because the rule is one boolean expression
     * inside a 3,700-line render loop, and reaching across to it from a unit
     * test would mean exposing the frame loop. If the renderer changes this
     * expression, this mirror has to change with it - which is the point: the
     * two are now asserted from the same documented rule instead of one of them
     * being untested.
     */
    private fun sigmaReachesImagePath(
        sigmaAvailable: Boolean,
        sparseGridRendered: Boolean,
        measurementActive: Boolean
    ): Boolean = sigmaAvailable && (sparseGridRendered || measurementActive)

    @Test
    fun stateTextNamesEverySwitchAndTheZoomFactor() {
        val text = gates(s1Raw = 10, s3Raw = 32, s5Raw = 40).stateText()

        assertTrue(text.contains("s1=10"))
        assertTrue(text.contains("s3=32"))
        assertTrue(text.contains("s5=40"))
        assertTrue(text.contains("needSparseGrid=true"))
        assertTrue(text.contains("sparseGridRendered=false"))
        assertTrue(text.contains("outDenoiseActive=true"))
        assertTrue(text.contains("zoomK=6.40"))
        assertTrue(text.contains("view=640x480"))
    }

    @Test
    fun stateTextCarriesBothRawSlidersAndDerivedStrengths() {
        // The raw number is what a reader reproduces from the UI; the derived
        // strength is what the shader was handed. Both, or a retuned mapping
        // makes old reports unreadable.
        val text = gates(s3Raw = 32).stateText()

        assertTrue(text.contains("s3=32"))
        assertTrue(text.contains("s3=0.320"))
    }

    @Test
    fun passTruthIsReportedSeparatelyFromSwitchTruth() {
        // S5 was asked for, but the sigma pass did not run, so outnr=true and
        // sigma=false. Collapsing them would hide the very failure the block
        // exists to surface.
        val text = gates(outDenoiseActive = true, passSigma = false, passSwgf = false).passesText()

        assertTrue(text.contains("sigma=false"))
        assertTrue(text.contains("swgf=false"))
        assertTrue(text.contains("outnr=true"))
    }

    @Test
    fun sigmaSourceIsNamedSoTheDefinitionCannotBeMistaken() {
        val text = gates(sigmaSource = PipelineGates.SigmaSource.OFFSCREEN_CHAIN).passesText()
        assertTrue(text.contains("measurement offscreen"))

        val none = gates(sigmaSource = PipelineGates.SigmaSource.NONE).passesText()
        assertTrue("a frame with no estimate must say so", none.contains("none"))
    }

    @Test
    fun structureMapNamesBothOfItsInputs() {
        val text = gates().structureMapPathNote()
        assertTrue(text.contains("luma="))
        assertTrue(text.contains("S5 output"))
        assertTrue(text.contains("sigma="))
    }

    @Test
    fun structureMapReportsWhenItDidNotRun() {
        val text = gates(lumaSource = PipelineGates.LumaSource.NOT_USED).structureMapPathNote()
        assertTrue(text.contains("did not run"))
    }

    @Test
    fun structureMapSaysWhenS5WasInactive() {
        // With S5 off there is no S5 output to characterise, and saying so keeps
        // the map from being read as a statement about a path that never ran.
        val text = gates(outDenoiseActive = false, lumaSource = PipelineGates.LumaSource.DEMOSAIC_NO_S5)
            .structureMapPathNote()
        assertTrue(text.contains("S5 inactive"))
    }

    // ---- The density gate, the root cause of the empty sections ---------

    @Test
    fun densityNoteExplainsAnEmptyMeasurementPastTheGate() {
        val note = gates(sparseGridRendered = false, zoomK = 6.4f).gridDensityNote()

        assertNotNull("k above the gate is the case that needs explaining", note)
        assertTrue(note!!.contains("k=6.40"))
        assertTrue(note.contains("k<=2"))
    }

    @Test
    fun densityNoteIsAbsentInsideTheGate() {
        assertNull(
            "inside the gate there is nothing surprising to explain",
            gates(sparseGridRendered = true, zoomK = 1.2f).gridDensityNote()
        )
    }

    @Test
    fun blockSaysTheChainRanOffscreenRatherThanForTheImage() {
        val text = gates().reportText()
        assertTrue(text.contains("passes:"))
        assertTrue(text.contains("sigma source"))
    }

    // ---- Segment key and block must not disagree -----------------------

    @Test
    fun segmentKeyIsDerivedFromTheSameRawValuesTheBlockPrints() {
        val g = gates(s1Raw = 10, s3Raw = 32, s5Raw = 40, sparseGridRendered = false)
        val key = g.segmentKey

        assertEquals(10, key.s1Raw)
        assertEquals(32, key.s3Raw)
        assertEquals(40, key.s5Raw)
        assertEquals("the key's regime is the rendered one", "inline(k>2)", key.zoomRegimeText())
    }

    @Test
    fun segmentKeyUsesTheViewSizeNotTheOutputSize() {
        val key = gates().segmentKey
        assertEquals(640, key.viewWidth)
        assertEquals(480, key.viewHeight)
    }

    // ---- When to print --------------------------------------------------

    @Test
    fun firstFrameAlwaysPrints() {
        val log = PipelineStateLog()
        assertTrue("the block must appear at least once", log.shouldLog(gates()))
    }

    @Test
    fun unchangedGatesDoNotPrintAgain() {
        val log = PipelineStateLog()
        val g = gates()

        assertTrue(log.shouldLog(g))
        assertFalse("an unchanging frame must not flood the log", log.shouldLog(g))
        assertFalse(log.shouldLog(g))
    }

    @Test
    fun aSliderMovePrintsAgain() {
        val log = PipelineStateLog()
        log.shouldLog(gates(s3Raw = 32))
        assertTrue(log.shouldLog(gates(s3Raw = 40)))
    }

    @Test
    fun aGateFlipPrintsEvenWhenTheSegmentKeyIsUnchanged() {
        // The chain shaders finishing compilation flips a pass without moving any
        // slider. Printing only on segment switches would miss exactly this.
        val log = PipelineStateLog()
        val before = gates(passSwgf = false)
        val after = gates(passSwgf = true)

        assertEquals("the segment key is identical", before.segmentKey, after.segmentKey)
        assertTrue(log.shouldLog(before))
        assertTrue("a pass flip must be visible", log.shouldLog(after))
    }

    @Test
    fun aSigmaSourceChangePrintsEvenWhenTheKeyIsUnchanged() {
        val log = PipelineStateLog()
        val offscreen = gates(sigmaSource = PipelineGates.SigmaSource.OFFSCREEN_CHAIN)
        val none = gates(sigmaSource = PipelineGates.SigmaSource.NONE)

        assertEquals(offscreen.segmentKey, none.segmentKey)
        assertTrue(log.shouldLog(offscreen))
        assertTrue(log.shouldLog(none))
    }

    @Test
    fun resetForcesTheBlockToPrintAgain() {
        val log = PipelineStateLog()
        val g = gates()
        log.shouldLog(g)
        assertFalse(log.shouldLog(g))

        log.reset()
        assertFalse(log.hasLogged())
        assertTrue(log.shouldLog(g))
    }
}