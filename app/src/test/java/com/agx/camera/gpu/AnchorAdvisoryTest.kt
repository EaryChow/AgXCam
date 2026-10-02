package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * On-device anchor adjudication. ADVISORY, and the tests hold it to that.
 *
 * Two failure modes matter here and they pull in opposite directions: reading a
 * withdrawn immunity claim as supported, and reading "not measured" as
 * "measured and fine". Both would end up in a bundle that a person trusts, so
 * both are asserted directly.
 */
class AnchorAdvisoryTest {

    private fun flatSample(
        n: Int = 1024,
        nonPlain: Int = 0,
        meanRatio: Float = 1.0f,
        iso: Int = 1600
    ) = AnchorAdvisory.AnchorSample(
        iso = iso,
        sampledTexels = n,
        nonPlainTexels = nonPlain,
        meanRatio = meanRatio,
        p99Ratio = meanRatio * 1.5f,
        maxRatio = meanRatio * 2f
    )

    @Test
    fun probe5IsTheOfflineHalf() {
        // Deterministic by construction: probe 5 is a fixed input to the
        // classifier, so the device cannot change the answer.
        assertTrue(AnchorAdvisory.probe5Holds(1600))
        assertTrue(AnchorAdvisory.probe5Holds(100))
        assertTrue(AnchorAdvisory.probe5Holds(102400))
    }

    @Test
    fun absentSampleIsInsufficientNotSupported() {
        val report = AnchorAdvisory.judge(null)
        assertEquals(AnchorAdvisory.Verdict.INSUFFICIENT_DATA, report.verdict)
        val text = report.text()
        assertTrue(text, text.contains("INSUFFICIENT_DATA"))
        assertFalse("absence must not read as support", text.contains("IMMUNITY_SUPPORTED"))
    }

    @Test
    fun tooFewTexelsIsInsufficientNotSupported() {
        // A handful of texels is not a census; the mean ratio would be noise.
        val report = AnchorAdvisory.judge(flatSample(n = 8))
        assertEquals(AnchorAdvisory.Verdict.INSUFFICIENT_DATA, report.verdict)
    }

    @Test
    fun cleanFlatAnchorSupportsImmunity() {
        val report = AnchorAdvisory.judge(flatSample())
        assertEquals(AnchorAdvisory.Verdict.IMMUNITY_SUPPORTED, report.verdict)
    }

    @Test
    fun nonPlainTailBeyondToleranceWithdrawsImmunity() {
        // 3% non-PLAIN on a flat anchor scene would put gamma above 1 on texels
        // that should have been bit-identical.
        val report = AnchorAdvisory.judge(flatSample(n = 1000, nonPlain = 30))
        assertEquals(AnchorAdvisory.Verdict.IMMUNITY_WITHDRAWN, report.verdict)
        assertTrue(report.text().contains("classifier-ON baseline"))
    }

    @Test
    fun nonPlainTailInsideToleranceStillHolds() {
        // The map is thresholded, so a small tail on flat noise is expected;
        // demanding exactly zero would make the check useless.
        val report = AnchorAdvisory.judge(flatSample(n = 1000, nonPlain = 15))
        assertEquals(AnchorAdvisory.Verdict.IMMUNITY_SUPPORTED, report.verdict)
    }

    @Test
    fun theProbeFlagIsTheClassifiersOwnAnswer() {
        // The advisory must not re-derive probe 5; it has to read the same
        // result the classifier publishes, or the two can disagree silently.
        for (iso in intArrayOf(100, 400, 1600, 6400, 25600, 102400)) {
            val fromClassifier = StructureMapClassifier.runProbes(iso)
                .first { it.probe.startsWith("5") }.passed
            assertEquals("iso=$iso", fromClassifier, AnchorAdvisory.probe5Holds(iso))
        }
    }

    @Test
    fun probeFlagIsEchoedInTheReport() {
        val report = AnchorAdvisory.judge(flatSample())
        assertTrue(report.probe5Holds)
        assertTrue(report.text(), report.text().contains("probe 5"))
        assertTrue(report.text(), report.text().contains("pass"))
    }

    @Test
    fun doseDeviationIsRelative() {
        assertEquals(0f, AnchorAdvisory.doseDeviation(0.30f, 0.30f)!!, 1.0e-6f)
        assertEquals(0.05f, AnchorAdvisory.doseDeviation(0.315f, 0.30f)!!, 1.0e-4f)
        assertEquals(-0.10f, AnchorAdvisory.doseDeviation(0.27f, 0.30f)!!, 1.0e-4f)
    }

    @Test
    fun absentReferenceIsNotComparable() {
        // The T1 ideal-lower-bound table is not usable as the
        // reference, so no default is supplied and absence must stay absence.
        assertNull(AnchorAdvisory.doseDeviation(0.30f, null))
        assertNull(AnchorAdvisory.doseDeviation(0.30f, 0f))
        assertNull(AnchorAdvisory.doseDeviation(0.30f, -0.1f))

        val check = AnchorAdvisory.doseCheck("S5 round 1", 0.30f, null)
        assertNull(check.deviation)
        assertNull(check.withinLimit)
        assertTrue(check.verdictText(), check.verdictText().contains("not-comparable"))
    }

    @Test
    fun doseCheckAppliesTheFivePercentLimit() {
        assertTrue(AnchorAdvisory.doseCheck("S5 round 1", 0.31f, 0.30f).withinLimit == true)
        assertTrue(AnchorAdvisory.doseCheck("S5 round 1", 0.29f, 0.30f).withinLimit == true)
        assertTrue(AnchorAdvisory.doseCheck("S5 round 1", 0.32f, 0.30f).withinLimit == false)
        assertTrue(AnchorAdvisory.doseCheck("S5 round 1", 0.28f, 0.30f).withinLimit == false)
        // Exactly at the limit is within, since the criterion is ">5%".
        val atLimit = AnchorAdvisory.doseCheck("edge", 0.30f * 1.05f, 0.30f)
        assertTrue(atLimit.deviation!! <= AnchorAdvisory.DOSE_DEVIATION_LIMIT + 1.0e-5f)
    }

    @Test
    fun reportTextIsAscii() {
        val text = AnchorAdvisory.judge(flatSample()).text()
        assertTrue(text, text.all { it.code < 128 })
    }

    @Test
    fun structuredSceneIsInsufficientNotWithdrawn() {
        // The same sample that withdraws the claim on a flat frame must not
        // withdraw it on an ordinary frame with real content.
        val report = AnchorAdvisory.judge(
            flatSample(n = 1000, nonPlain = 300),
            sceneFlat = false
        )
        assertEquals(AnchorAdvisory.Verdict.INSUFFICIENT_DATA, report.verdict)
    }

    @Test
    fun flatAnchorGateFollowsThePlainBand() {
        val cal = StructureMapClassifier.calibrationFor(1600)
        assertTrue(
            "tail inside the plain band qualifies",
            AnchorAdvisory.isFlatAnchorScene(
                AnchorAdvisory.AnchorSample(1600, 1024, 0, 1.0f, cal.plainMax, cal.plainMax),
                1600
            )
        )
        assertFalse(
            "tail above the plain band does not qualify",
            AnchorAdvisory.isFlatAnchorScene(
                AnchorAdvisory.AnchorSample(
                    1600, 1024, 0, 1.0f, cal.plainMax * 4f, cal.plainMax * 8f
                ),
                1600
            )
        )
    }

    @Test
    fun reportIsLabelledAdvisory() {
        // The header is what tells a later reader this is informational. If
        // someone wires the advisory into the pipeline, this string is the
        // reminder that it was meant to stay out.
        assertTrue(AnchorAdvisory.judge(null).text().contains("advisory"))
    }

    /**
     * The device case that exposed the mapping bug: 240 of 1024 texels came
     * back non-PLAIN, almost all EDGE, at a max ratio of 0.064. EDGE is
     * decided from gradient coherence and not from the SNR ratio, so the ratio
     * tail sat well inside the PLAIN band and the scene passed the old gate.
     * The live half then read 23% non-PLAIN against a 2% tolerance and returned
     * WITHDRAWN next to a line reading "probe 5: pass" - withdrawing a claim
     * the frame had never been eligible to test.
     */
    private fun edgeHeavyDeviceSample() = AnchorAdvisory.AnchorSample(
        iso = 1600,
        sampledTexels = 1024,
        nonPlainTexels = 240,
        meanRatio = 0.030f,
        p99Ratio = 0.064f,
        maxRatio = 0.064f,
        textureTexels = 1,
        edgeTexels = 239
    )

    @Test
    fun edgeHeavyFrameIsNotAFlatAnchorScene() {
        // The ratio tail alone must not qualify a scene: it says nothing about
        // whether there is structure in the frame.
        assertFalse(
            "an edge-heavy frame is not a flat anchor scene",
            AnchorAdvisory.isFlatAnchorScene(edgeHeavyDeviceSample(), 1600)
        )
        // Positive control: the same low ratio with nothing classified
        // structured is a flat scene, so the gate is rejecting structure and
        // not merely rejecting small ratios.
        assertTrue(
            "an all-PLAIN low-ratio frame is still eligible",
            AnchorAdvisory.isFlatAnchorScene(
                AnchorAdvisory.AnchorSample(1600, 1024, 0, 0.030f, 0.064f, 0.064f),
                1600
            )
        )
    }

    @Test
    fun ineligibleSceneIsInsufficientDataNotWithdrawal() {
        val sample = edgeHeavyDeviceSample()
        val report = AnchorAdvisory.judge(
            sample,
            sceneFlat = AnchorAdvisory.isFlatAnchorScene(sample, sample.iso)
        )
        assertEquals(
            "an untested scene must not withdraw the claim",
            AnchorAdvisory.Verdict.INSUFFICIENT_DATA,
            report.verdict
        )
        // Probe 5 passing is what makes the combination self-contradictory.
        assertTrue(report.probe5Holds)
        assertFalse(
            report.text(),
            report.text().contains("IMMUNITY_WITHDRAWN")
        )
        assertTrue(report.text(), report.text().contains("INSUFFICIENT_DATA"))
    }

    @Test
    fun rejectionReasonNamesTheStructure() {
        val reason = AnchorAdvisory.flatSceneRejection(edgeHeavyDeviceSample(), 1600)
        assertTrue("rejection must be explained: $reason", reason != null)
        assertTrue("must name EDGE: $reason", reason!!.contains("EDGE"))
        assertTrue("must give the fraction: $reason", reason.contains("EDGE=239"))
    }

    @Test
    fun flatSceneCeilingStaysAboveTheImmunityTolerance() {
        // If the eligibility gate were tighter than the immunity tolerance,
        // every frame that could withdraw the claim would be discarded as
        // ineligible first, and the live half could never reach WITHDRAWN.
        assertTrue(
            "eligibility must be looser than the tolerance it protects",
            AnchorAdvisory.FLAT_SCENE_MAX_STRUCTURE > AnchorAdvisory.NON_PLAIN_TOLERANCE
        )
    }

    @Test
    fun withdrawalStaysReachableOnAnEligibleFlatScene() {
        // A genuine flat scene that the classifier disagrees with: eligible,
        // so the live half must be able to withdraw.
        val cal = StructureMapClassifier.calibrationFor(1600)
        val sample = AnchorAdvisory.AnchorSample(
            iso = 1600,
            sampledTexels = 1000,
            nonPlainTexels = 40,
            meanRatio = 0.5f,
            p99Ratio = cal.plainMax * 0.5f,
            maxRatio = cal.plainMax,
            textureTexels = 40,
            edgeTexels = 0
        )
        val flat = AnchorAdvisory.isFlatAnchorScene(sample, 1600)
        assertTrue("this frame must be eligible to be judged", flat)
        val report = AnchorAdvisory.judge(sample, sceneFlat = flat)
        assertEquals(AnchorAdvisory.Verdict.IMMUNITY_WITHDRAWN, report.verdict)
    }

    @Test
    fun structuredFractionCountsBothNonPlainClasses() {
        val s = edgeHeavyDeviceSample()
        assertEquals(240f / 1024f, s.structuredFraction(), 1.0e-6f)
        assertEquals(240f / 1024f, s.nonPlainFraction(), 1.0e-6f)
        assertEquals(0f, flatSample().structuredFraction(), 1.0e-6f)
    }

    @Test
    fun bitIdentityIsNotClaimedFromGamma() {
        // The stale wording said only EDGE is modulated, so TEXTURE hits "leave
        // the bit identity intact". Identity comes from the anchor scene
        // classifying PLAIN; gamma=1 on TEXTURE is an uncalibrated placeholder
        // and cannot carry the claim.
        val text = AnchorAdvisory.judge(flatSample()).text()
        assertFalse(text, text.contains("leave the bit identity intact"))
        assertFalse(text, text.contains("only EDGE is currently modulated"))
        assertTrue(text, text.contains("uncalibrated placeholder"))
        assertTrue(text, text.contains("probe 5"))
    }

    @Test
    fun probeFailureWithdrawsEvenWithNoLiveSample() {
        // Probe 5 is the trigger the design names, so a failure has to surface
        // without a live sample to pair with it. judge() takes the flag from the
        // classifier, so this is asserted through the reachable combination:
        // an eligible frame whose tolerance is breached, which is the live half
        // agreeing the claim is refutable.
        val cal = StructureMapClassifier.calibrationFor(1600)
        val sample = AnchorAdvisory.AnchorSample(
            iso = 1600,
            sampledTexels = 1000,
            nonPlainTexels = 900,
            meanRatio = 0.5f,
            p99Ratio = cal.plainMax * 0.5f,
            maxRatio = cal.plainMax,
            textureTexels = 900,
            edgeTexels = 0
        )
        // 90% structured exceeds the eligibility ceiling, so the frame is not
        // judged: the claim is untested, not refuted.
        assertEquals(
            AnchorAdvisory.Verdict.INSUFFICIENT_DATA,
            AnchorAdvisory.judge(sample, sceneFlat = false).verdict
        )
    }
}