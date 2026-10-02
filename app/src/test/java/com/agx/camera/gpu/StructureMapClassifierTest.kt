package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Structure-map classifier: the five frozen probes, and the SNR ratio
 * definition they exist to defend.
 *
 * Probe 5 is the load-bearing one. The whole anchor immunity claim is
 * conditional on a pure-noise patch classifying PLAIN; if probe 5 fails the
 * claim is withdrawn and anchor acceptance falls back to the classifier-ON
 * baseline. So probe 5 is asserted directly rather than only through the
 * aggregate.
 */
class StructureMapClassifierTest {

    @Test
    fun allFiveProbesPass() {
        val results = StructureMapClassifier.runProbes(1600)
        assertEquals(5, results.size)
        for (p in results) {
            assertTrue(
                "probe '${p.probe}' got ${StructureMapClassifier.className(p.class_)}, expected ${StructureMapClassifier.className(p.expected)} (${p.note})",
                p.passed
            )
        }
        assertTrue(StructureMapClassifier.probesPass(1600))
    }

    @Test
    fun probe5IsWhatProbe5Says() {
        // Named explicitly so the immunity claim has a direct assertion and
        // cannot quietly ride on the aggregate probesPass() flag.
        val p5 = StructureMapClassifier.runProbes(1600).first { it.probe.startsWith("5") }
        assertEquals(StructureMapClassifier.PLAIN, p5.class_)
        assertEquals(StructureMapClassifier.PLAIN, p5.expected)
        assertTrue(p5.passed)
    }

    @Test
    fun probesHoldAcrossTheWholeIsoLadder() {
        for (iso in intArrayOf(100, 400, 1600, 6400, 25600, 102400)) {
            assertTrue("probes fail at iso=$iso", StructureMapClassifier.probesPass(iso))
        }
    }

    @Test
    fun twoClassFallbackIsNotTriggered() {
        // Probe 4 (pure chroma edge, flat luma) landing on EDGE is what proves
        // the coherence path sees something the luma-side ratio cannot.
        assertFalse(
            "probe 4 must land on EDGE or the three-class separation has failed",
            StructureMapClassifier.needsTwoClassFallback(1600)
        )
    }

    @Test
    fun ratioCarriesTheRetentionFactorInTheDenominator() {
        val sigma2 = 100f
        // A pure-noise patch lands at roughly r_retain * sigma^2, so its ratio
        // is ~1. A bare sigma^2 denominator would make this read 1/r_retain.
        val r = StructureMapClassifier.calibrationFor(1600).rRetain
        val ratio = StructureMapClassifier.ratio(1.0f * r * sigma2, sigma2, r)
        assertEquals(1.0f, ratio, 1.0e-4f)
    }

    @Test
    fun ratioRejectsDegenerateInputs() {
        assertEquals(0f, StructureMapClassifier.ratio(10f, 0f, 0.3f), 0f)
        assertEquals(0f, StructureMapClassifier.ratio(10f, 100f, 0f), 0f)
        assertEquals(0f, StructureMapClassifier.ratio(Float.NaN, 100f, 0.3f), 0f)
        assertEquals(0f, StructureMapClassifier.ratio(-5f, 100f, 0.3f), 0f)
    }

    @Test
    fun calibrationLadderIsMonotonic() {
        val cals = StructureMapClassifier.allCalibrations()
        for (i in 1 until cals.size) {
            val a = cals[i - 1]
            val b = cals[i]
            assertTrue("rRetain must rise with ISO", b.rRetain >= a.rRetain)
            assertTrue("textureMin must rise with ISO", b.textureMin >= a.textureMin)
            assertTrue("plainMax must not fall with ISO", b.plainMax >= a.plainMax)
        }
    }

    @Test
    fun everyProfileBucketWithoutItsOwnRowFloorsToTheRungBelow() {
        // The calibration ladder is deliberately sparser than the profile
        // ladder. Pinning the whole mapping keeps the effective granularity
        // visible instead of looking like a row per bucket.
        val expected = mapOf(
            100 to 100,
            200 to 100,
            400 to 400,
            800 to 400,
            1600 to 1600,
            3200 to 1600,
            6400 to 6400,
            12800 to 6400,
            25600 to 25600,
            51200 to 25600,
            102400 to 102400
        )
        for ((profileBucket, rung) in expected) {
            assertEquals(
                "profile bucket $profileBucket",
                rung,
                StructureMapClassifier.calibrationFor(profileBucket).isoBucket
            )
        }
    }

    @Test
    fun calibrationForPicksTheNearestLowerBucket() {
        assertEquals(100, StructureMapClassifier.calibrationFor(100).isoBucket)
        // The calibration ladder is 100/400/1600/..., and ISO 500 floors to
        // bucket 400, so it takes the 400 row. ISO 300 floors to 200, which has
        // no row of its own, so it falls back to the 100 row below it.
        assertEquals(400, StructureMapClassifier.calibrationFor(500).isoBucket)
        assertEquals(100, StructureMapClassifier.calibrationFor(300).isoBucket)
        assertEquals(400, StructureMapClassifier.calibrationFor(400).isoBucket)
        assertEquals(1600, StructureMapClassifier.calibrationFor(1600).isoBucket)
        assertEquals(1600, StructureMapClassifier.calibrationFor(3000).isoBucket)
        assertEquals(102400, StructureMapClassifier.calibrationFor(102400).isoBucket)
    }

    @Test
    fun classificationIsDeterministicForRepeatedCalls() {
        val cal = StructureMapClassifier.calibrationFor(1600)
        val sigma2 = 100f
        val ratio = StructureMapClassifier.ratio(9f * sigma2, sigma2, cal.rRetain)
        val first = StructureMapClassifier.classify(1600, ratio, 0.22f, 0.85f)
        for (i in 0 until 50) {
            assertEquals(first, StructureMapClassifier.classify(1600, ratio, 0.22f, 0.85f))
        }
    }

    @Test
    fun edgesAreDecidedByCoherenceNotByRatio() {
        val cal = StructureMapClassifier.calibrationFor(1600)
        // Low luma-side ratio, high coherence: still EDGE, because that is the
        // case a luma-only criterion misses.
        val ratio = StructureMapClassifier.ratio(1.0f * 100f, 100f, cal.rRetain)
        assertEquals(StructureMapClassifier.EDGE, StructureMapClassifier.classify(1600, ratio, 0.90f, 0.1f))
    }

    @Test
    fun textureNeedsBothRatioAndOrganization() {
        val cal = StructureMapClassifier.calibrationFor(1600)
        val ratio = StructureMapClassifier.ratio(9f * 100f, 100f, cal.rRetain)
        // High ratio but scattered (noise-like organization) must not be TEXTURE.
        assertEquals(StructureMapClassifier.PLAIN, StructureMapClassifier.classify(1600, ratio, 0.0f, 0.1f))
        // Connected organization with the same ratio is TEXTURE.
        assertEquals(StructureMapClassifier.TEXTURE, StructureMapClassifier.classify(1600, ratio, 0.0f, 0.9f))
    }

    @Test
    fun textureHoldsInsideTheBandBetweenTheTwoThresholds() {
        val cal = StructureMapClassifier.calibrationFor(1600)
        // Between plainMax and textureMin the classification depends on history:
        // TEXTURE holds because it has to clear a higher bar to drop.
        val inBand = cal.plainMax * 1.05f
        assertTrue("probe ratio must sit inside the band", inBand < cal.textureMin)
        assertEquals(
            StructureMapClassifier.TEXTURE,
            StructureMapClassifier.classify(1600, inBand, 0.0f, 0.9f, previous = StructureMapClassifier.TEXTURE)
        )
        // The same ratio from a PLAIN previous frame stays PLAIN, so the band
        // cannot be entered from below.
        assertEquals(
            StructureMapClassifier.PLAIN,
            StructureMapClassifier.classify(1600, inBand, 0.0f, 0.9f, previous = StructureMapClassifier.PLAIN)
        )
    }

    @Test
    fun ratioBelowPlainMaxDropsImmediately() {
        val cal = StructureMapClassifier.calibrationFor(1600)
        // At or below plainMax the classifier does not consult history at all.
        assertEquals(
            StructureMapClassifier.PLAIN,
            StructureMapClassifier.classify(
                1600, cal.plainMax, 0.0f, 0.9f,
                previous = StructureMapClassifier.TEXTURE
            )
        )
        assertEquals(
            StructureMapClassifier.PLAIN,
            StructureMapClassifier.classify(
                1600, cal.plainMax * 0.5f, 0.0f, 0.9f,
                previous = StructureMapClassifier.TEXTURE
            )
        )
    }

    @Test
    fun twoClassModeNeverReturnsEdge() {
        for (iso in intArrayOf(100, 1600, 102400)) {
            for (r in listOf(0.1f, 1.5f, 2.5f, 5f, 20f)) {
                for (org in listOf(0f, 0.5f, 1f)) {
                    val c = StructureMapClassifier.classifyTwoClass(iso, r, org)
                    assertTrue(
                        "two-class mode returned $c at iso=$iso ratio=$r org=$org",
                        c == StructureMapClassifier.PLAIN || c == StructureMapClassifier.NON_PLAIN
                    )
                }
            }
        }
    }

    @Test
    fun gammaIsOneForPlainAndTextureUntilTextureIsCalibrated() {
        // Current state, not the design: TEXTURE is meant to sit below 1 so the
        // pass lets texture through, and it has no calibrated value yet, so it
        // sits at 1. This pins the placeholder and will fail when a real
        // gamma_texture is calibrated in.
        assertEquals(1f, StructureMapClassifier.gammaFor(StructureMapClassifier.PLAIN), 0f)
        assertEquals(1f, StructureMapClassifier.gammaFor(StructureMapClassifier.TEXTURE), 0f)
        assertTrue(StructureMapClassifier.gammaFor(StructureMapClassifier.EDGE) > 1f)
    }

    @Test
    fun aLumIsMinimumAtThePivotAndGrowsAtBothEnds() {
        val pivot = 0.45f
        val atPivot = StructureMapClassifier.aLum(pivot / 2f + 0.0001f, pivot)
        val below = StructureMapClassifier.aLum(0.01f, pivot)
        val above = StructureMapClassifier.aLum(0.95f, pivot)
        assertTrue("below ($below) must exceed the pivot ($atPivot)", below > atPivot)
        assertTrue("above ($above) must exceed the pivot ($atPivot)", above > atPivot)
    }

    @Test
    fun aLumIsClampedToItsInputRange() {
        // Out-of-range formed luminance clamps rather than extrapolating.
        assertEquals(
            StructureMapClassifier.aLum(0f, 0.45f),
            StructureMapClassifier.aLum(-0.5f, 0.45f),
            1.0e-6f
        )
        assertEquals(
            StructureMapClassifier.aLum(1f, 0.45f),
            StructureMapClassifier.aLum(1.5f, 0.45f),
            1.0e-6f
        )
    }

    @Test
    fun elevationIsAtLeastOne() {
        val base = StructureMapClassifier.jndBase(0.5f)
        assertTrue(StructureMapClassifier.elevation(0f, base, false) >= 1f)
        assertTrue(StructureMapClassifier.elevation(100f, base, false) >= 1f)
        // Edge blocks keep their gain above 1 even when masking would lower it.
        assertTrue(StructureMapClassifier.elevation(0f, base, true) >= 1f)
        assertEquals(1f, StructureMapClassifier.elevation(10f, 0f, false), 1.0e-6f)
    }

    @Test
    fun reportIsAsciiAndRecordsThePendingRecalibration() {
        val text = StructureMapClassifier.report(1600)
        assertTrue(text, text.all { it.code < 128 })
        assertTrue(text, text.contains("PENDING"))
        assertTrue(text, text.contains("probe 5") || text.contains("noise disguised"))
        assertTrue(text, text.contains("two-class fallback"))
    }

    @Test
    fun residualAndSigmaRoundTrip() {
        val sigma2 = 100f
        val target = 3.5f
        val residual = StructureMapClassifier.residualFor(target, sigma2, 1600)
        val sigma = StructureMapClassifier.sigmaFor(residual, target, 1600)
        assertEquals(sigma2, sigma * sigma, 1.0e-2f)
    }

    @Test
    fun stabilizerLimitsClassJumps() {
        val s = StructureMapStabilizer()
        assertEquals(1, s.slewLimit)
        // PLAIN(0) -> EDGE(2) is two classes; the slew limit moves one.
        val out = s.update(key = 42, candidate = StructureMapClassifier.EDGE)
        assertEquals(StructureMapClassifier.TEXTURE, out)
        assertEquals(StructureMapClassifier.TEXTURE, s.current(42))
        assertEquals(StructureMapClassifier.EDGE, s.update(42, StructureMapClassifier.EDGE))
    }

    @Test
    fun stabilizerTracksPerKey() {
        val s = StructureMapStabilizer()
        s.update(1, StructureMapClassifier.TEXTURE)
        // Reaching EDGE from the PLAIN default takes two frames: the slew
        // limit is 1 class per frame, so the first update lands on TEXTURE.
        assertEquals(StructureMapClassifier.TEXTURE, s.update(2, StructureMapClassifier.EDGE))
        assertEquals(StructureMapClassifier.EDGE, s.update(2, StructureMapClassifier.EDGE))
        assertEquals(StructureMapClassifier.TEXTURE, s.current(1))
        assertEquals(StructureMapClassifier.EDGE, s.current(2))
        // An untracked key starts at PLAIN rather than at another key's value.
        assertEquals(StructureMapClassifier.PLAIN, s.current(3))
    }

    @Test
    fun stabilizerStabilityUsesTheChangedFraction() {
        val s = StructureMapStabilizer()
        assertTrue("an empty map is trivially stable", s.isStable(changed = 0, total = 0))
        assertTrue(s.isStable(maxChangedFraction = 0.05f, changed = 5, total = 100))
        assertFalse(s.isStable(maxChangedFraction = 0.05f, changed = 6, total = 100))
    }

    @Test
    fun stabilizerReportIsAscii() {
        val s = StructureMapStabilizer()
        assertTrue(s.report().contains("empty"))
        s.update(1, StructureMapClassifier.TEXTURE)
        s.update(2, StructureMapClassifier.EDGE)
        val text = s.report()
        assertTrue(text, text.all { it.code < 128 })
        assertTrue(text, text.contains("TEXTURE"))
        // Drive the second key all the way to EDGE so the report has to name it.
        s.update(2, StructureMapClassifier.EDGE)
        val withEdge = s.report()
        assertTrue(withEdge, withEdge.contains("EDGE"))
        assertNotNull(withEdge)
    }
}