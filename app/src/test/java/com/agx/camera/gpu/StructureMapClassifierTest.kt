package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

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
        val pred = StructureMapClassifier.noisePrediction(sigma2, 0f, 1f, 1f, 1f)
        val ratio = StructureMapClassifier.ratio(1.0f * r * pred, pred, r)
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
    fun theHostProbeAndTheDeviceAgreeOnWherePureNoiseLands() {
        // The whole point of the folded denominator: a pure-noise patch has to
        // land at the same ratio on the host and on the GPU, or the bands are
        // tuned against a map that does not exist. At the shipped luma_eps_scale
        // the unfolded form put the probe at 1.0 and the device at 0.714.
        val sigma2 = 100f
        val r = StructureMapClassifier.calibrationFor(1600).rRetain
        val pred = StructureMapClassifier.noisePrediction(sigma2, 0f, 1f, 1f, 1f)
        val probe = StructureMapClassifier.ratio(pred * r, pred, r)
        assertEquals(1.0f, probe, 1.0e-4f)
        // And the fold is not free, which is what the unfolded form hid.
        assertTrue(
            "the prediction must actually carry the epsilon scale",
            pred > sigma2
        )
        // The device expression is the same product in the same order, so the
        // ratio of the two forms is the epsilon scale and nothing else.
        assertEquals(
            StructureMapClassifier.LUMA_EPS_SCALE,
            pred / sigma2,
            1.0e-4f
        )
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
        val ratio = StructureMapClassifier.ratio(
            9f * StructureMapClassifier.noisePrediction(sigma2, 0f, 1f, 1f, 1f),
            StructureMapClassifier.noisePrediction(sigma2, 0f, 1f, 1f, 1f),
            cal.rRetain
        )
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
        val ratio = StructureMapClassifier.ratio(
            StructureMapClassifier.noisePrediction(100f, 0f, 1f, 1f, 1f),
            StructureMapClassifier.noisePrediction(100f, 0f, 1f, 1f, 1f),
            cal.rRetain
        )
        assertEquals(StructureMapClassifier.EDGE, StructureMapClassifier.classify(1600, ratio, 0.90f, 0.1f))
    }

    @Test
    fun textureNeedsBothRatioAndOrganization() {
        val cal = StructureMapClassifier.calibrationFor(1600)
        val ratio = StructureMapClassifier.ratio(
            9f * StructureMapClassifier.noisePrediction(100f, 0f, 1f, 1f, 1f),
            StructureMapClassifier.noisePrediction(100f, 0f, 1f, 1f, 1f),
            cal.rRetain
        )
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
    fun noisePredictionAppliesTheDomainFoldTheRatioWasMissing() {
        // The ratio divides a measured formed-image residual by this prediction,
        // so the prediction has to carry the same folds S5's epsilon carries.
        // Without them the census read a p99 of 0.004 against a 1.60 ceiling and
        // the map looked degenerate rather than mis-scaled.
        val sigmaHat2 = 113f
        val sigmaDm2 = 10f
        val whiteRange = 959f
        val inverseRange2 = 1f / (whiteRange * whiteRange)
        val calibScale = 1f / 32f
        val pred = StructureMapClassifier.noisePrediction(sigmaHat2, sigmaDm2, inverseRange2, calibScale, 1f)
        // A raw DN^2 residual read as if it were already in image units is the
        // 2.9e7 error the fold removes; without it the prediction is off by
        // whiteRange^2 / calibScale.
        val unfolded = 1.4f * (sigmaHat2 + sigmaDm2)
        assertEquals(unfolded, pred / (inverseRange2 * calibScale), unfolded * 1.0e-3f)
        // And it lands in the same order of magnitude as the S5 luma epsilon
        // itself, which is the whole point: the ratio is dimensionless only if
        // the two sides are the same quantity.
        val epsY = 1.4f * (sigmaHat2 + sigmaDm2) * inverseRange2 * calibScale
        assertEquals(epsY, pred, 1.0e-9f)
    }

    @Test
    fun noisePredictionRejectsDegenerateInputs() {
        assertEquals(0f, StructureMapClassifier.noisePrediction(0f, 10f, 1e-6f, 1f / 32f, 1f), 0f)
        assertEquals(
            0f,
            StructureMapClassifier.noisePrediction(Float.NaN, 10f, 1e-6f, 1f / 32f, 1f),
            0f
        )
    }

    @Test
    fun theProbeAnchorSitsBelowThePlainCeilingWithTheFoldApplied() {
        // Probe 1 and probe 5 only mean anything if the ratio they construct is
        // the same ratio the shader computes. Feeding the same-domain residual
        // through the folded prediction has to land on both sides of plainMax,
        // which is what makes them a test of the shipped criterion rather than
        // of a literal.
        val cal = StructureMapClassifier.calibrationFor(1600)
        val inverseRange2 = 1f / (959f * 959f)
        val pred = StructureMapClassifier.noisePrediction(113f, 10f, inverseRange2, 1f / 32f, 1f)
        val atNoise = StructureMapClassifier.ratio(pred * cal.rRetain, pred, cal.rRetain)
        assertEquals(1.0f, atNoise, 1.0e-3f)
        assertTrue(
            "the pure-noise anchor has to classify PLAIN under the shipped bands",
            atNoise <= cal.plainMax
        )
        val atTexture = StructureMapClassifier.ratio(9f * pred * cal.rRetain, pred, cal.rRetain)
        assertTrue("a textured patch has to clear the texture band", atTexture >= cal.textureMin)
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
        // Nothing in the image path consumes the value today. It is pinned so a
        // dormant helper that gets "fixed" looks like the calibration change it
        // is, rather than passing silently because no shader reads it.
        assertEquals(
            "stabilizedGamma must stay the slew-limited form of the same table",
            1.15f,
            StructureMapStabilizer().let { s ->
                s.update(1, StructureMapClassifier.EDGE)
                s.stabilizedGamma(1, StructureMapClassifier.EDGE)
            },
            0f
        )
    }

    @Test
    fun classNameNamesEveryClassTheMapCanWrite() {
        assertEquals("PLAIN", StructureMapClassifier.className(StructureMapClassifier.PLAIN))
        assertEquals("TEXTURE", StructureMapClassifier.className(StructureMapClassifier.TEXTURE))
        assertEquals("EDGE", StructureMapClassifier.className(StructureMapClassifier.EDGE))
        assertEquals(
            "a class outside the three the map writes must say so rather than borrow a name",
            "?", StructureMapClassifier.className(3)
        )
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
