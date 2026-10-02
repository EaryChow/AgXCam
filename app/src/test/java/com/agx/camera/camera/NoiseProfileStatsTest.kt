package com.agx.camera.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Runtime sigma profile: the double-key EMA.
 *
 * The two keys are the whole reason this class exists. The Stage-3 blend
 * changes the residual the estimator sees, so one EMA shared across S3-on and
 * S3-off averages two populations and reads plausible while being wrong in
 * both. These tests exist to keep that from quietly regressing back to a single
 * key, and to keep "not measured" from reading as "measured and fine".
 */
class NoiseProfileStatsTest {

    @Test
    fun isoBucketsDoubleFromOneHundred() {
        assertEquals(100, NoiseProfileStats.isoBucketFor(50))
        assertEquals(100, NoiseProfileStats.isoBucketFor(100))
        assertEquals(100, NoiseProfileStats.isoBucketFor(101))
        assertEquals(100, NoiseProfileStats.isoBucketFor(150))
        assertEquals(200, NoiseProfileStats.isoBucketFor(200))
        assertEquals(200, NoiseProfileStats.isoBucketFor(201))
        assertEquals(200, NoiseProfileStats.isoBucketFor(300))
        assertEquals(400, NoiseProfileStats.isoBucketFor(400))
        assertEquals(1600, NoiseProfileStats.isoBucketFor(1600))
        assertEquals(1600, NoiseProfileStats.isoBucketFor(1601))
        assertEquals(6400, NoiseProfileStats.isoBucketFor(6400))
        assertEquals(102400, NoiseProfileStats.isoBucketFor(200000))
    }

    @Test
    fun bucketsAreFloorsNotCeilings() {
        // An ISO just above a rung belongs to that rung, not the next one:
        // rounding up would halve the number of real ISOs landing in each EMA
        // population for no reason.
        for (iso in intArrayOf(101, 102, 150, 199)) {
            assertEquals("iso=$iso", 100, NoiseProfileStats.isoBucketFor(iso))
        }
        for (iso in intArrayOf(201, 300, 399)) {
            assertEquals("iso=$iso", 200, NoiseProfileStats.isoBucketFor(iso))
        }
    }

    @Test
    fun s3StateIsASeparateKey() {
        val stats = NoiseProfileStats()
        stats.update(iso = 1600, s3Active = false, meanSigmaChroma = 10f, meanSigmaLuma = 1f)
        stats.update(iso = 1600, s3Active = true, meanSigmaChroma = 20f, meanSigmaLuma = 1f)

        // Same ISO, different S3 state: the two rows must not blend.
        assertEquals(1L, stats.sampleCount(1600, false))
        assertEquals(1L, stats.sampleCount(1600, true))
        assertEquals(10f, stats.ratio(1600, false)!!, 1.0e-4f)
        assertEquals(20f, stats.ratio(1600, true)!!, 1.0e-4f)
    }

    @Test
    fun isoBucketIsASeparateKey() {
        val stats = NoiseProfileStats()
        stats.update(iso = 1500, s3Active = true, meanSigmaChroma = 12f, meanSigmaLuma = 1f)
        stats.update(iso = 3200, s3Active = true, meanSigmaChroma = 14f, meanSigmaLuma = 1f)
        assertEquals(12f, stats.ratio(1500, true)!!, 1.0e-4f)
        assertEquals(14f, stats.ratio(3200, true)!!, 1.0e-4f)
    }

    @Test
    fun unmeasuredKeyReadsNullNotZero() {
        val stats = NoiseProfileStats()
        stats.update(iso = 1600, s3Active = false, meanSigmaChroma = 10f, meanSigmaLuma = 1f)
        // A ratio of 0 would read as "perfectly balanced" and would silently
        // pass a deviation check; absence has to stay absence.
        assertNull(stats.ratio(1600, true))
        assertNull(stats.ratio(6400, false))
        assertNull(stats.meanSigmaChroma(6400, false))
        assertEquals(0L, stats.sampleCount(6400, false))
    }

    @Test
    fun invalidSamplesAreRejected() {
        val stats = NoiseProfileStats()
        stats.update(1600, false, Float.NaN, 1f)
        stats.update(1600, false, 10f, Float.NaN)
        stats.update(1600, false, 10f, 0f)
        stats.update(1600, false, 10f, -1f)
        stats.update(1600, false, Float.POSITIVE_INFINITY, 1f)
        assertEquals(0L, stats.sampleCount(1600, false))
        assertNull(stats.ratio(1600, false))
    }

    @Test
    fun firstSampleSeedsTheEmaAndLaterOnesAreWeighted() {
        val stats = NoiseProfileStats()
        stats.emaAlpha = 0.5f
        stats.update(1600, false, 10f, 1f)
        assertEquals(10f, stats.meanSigmaChroma(1600, false)!!, 1.0e-4f)
        stats.update(1600, false, 20f, 1f)
        assertEquals(15f, stats.meanSigmaChroma(1600, false)!!, 1.0e-4f)
        stats.update(1600, false, 20f, 1f)
        assertEquals(17.5f, stats.meanSigmaChroma(1600, false)!!, 1.0e-4f)
    }

    @Test
    fun emaAlphaIsClampedToTheUsableRange() {
        val stats = NoiseProfileStats()
        stats.emaAlpha = 0f
        stats.update(1600, false, 10f, 1f)
        stats.update(1600, false, 100f, 1f)
        // alpha 0 would freeze the row forever; it is clamped to 0.001.
        assertTrue(stats.meanSigmaChroma(1600, false)!! > 10f)

        val jumpy = NoiseProfileStats()
        jumpy.emaAlpha = 5f
        jumpy.update(1600, false, 10f, 1f)
        jumpy.update(1600, false, 20f, 1f)
        // alpha 5 is clamped to 1, so the newest sample wins outright.
        assertEquals(20f, jumpy.meanSigmaChroma(1600, false)!!, 1.0e-4f)
    }

    @Test
    fun deviationIsSignedAndRelative() {
        val stats = NoiseProfileStats()
        stats.update(1600, false, meanSigmaChroma = 52f, meanSigmaLuma = 1f)
        assertEquals(0f, stats.deviationFromOffline(1600, false, 52f)!!, 1.0e-5f)
        // Measured is half the reference, so the deviation is negative.
        assertEquals(-0.5f, stats.deviationFromOffline(1600, false, 104f)!!, 1.0e-4f)
        // Measured is double the reference, so the deviation is positive.
        assertEquals(1.0f, stats.deviationFromOffline(1600, false, 26f)!!, 1.0e-4f)
    }

    @Test
    fun missingOfflineValueIsNotComparable() {
        val stats = NoiseProfileStats()
        stats.update(1600, false, 52f, 1f)
        assertNull(stats.deviationFromOffline(1600, false, 0f))
        assertFalse("an invalid reference must not read as deviated", stats.isDeviated(1600, false, 0f))
        // No measurement at all must also stay non-deviating, not become 0%.
        assertNull(stats.deviationFromOffline(6400, false, 52f))
        assertFalse(stats.isDeviated(6400, false, 52f))
    }

    @Test
    fun deviationWarningTripsOnlyPastTheThreshold() {
        val stats = NoiseProfileStats()
        stats.update(1600, false, 52f, 1f)
        assertFalse(stats.isDeviated(1600, false, 52f))
        // Reference 20% above measured gives -16.7%, inside the 25% band.
        assertFalse(stats.isDeviated(1600, false, 52f * 1.2f))
        // Reference 40% below measured gives +66.7%, past the band.
        assertTrue(stats.isDeviated(1600, false, 52f * 0.6f))
        // Reference double measured gives -50%, past the band.
        assertTrue(stats.isDeviated(1600, false, 52f * 2f))
    }

    @Test
    fun resetClearsEveryRow() {
        val stats = NoiseProfileStats()
        stats.update(1600, false, 10f, 1f)
        stats.update(6400, true, 12f, 1f)
        stats.reset()
        assertNull(stats.ratio(1600, false))
        assertNull(stats.ratio(6400, true))
        assertTrue(stats.report().contains("no samples yet"))
    }

    @Test
    fun reportIsAsciiAndShowsBothKeyParts() {
        val stats = NoiseProfileStats()
        stats.update(1600, false, 52f, 1f)
        stats.update(1600, true, 58f, 1f)
        val text = stats.report(NoiseProfileStats.offlineRatioTable())
        assertTrue(text, text.all { it.code < 128 })
        assertTrue(text, text.contains("s3=off"))
        assertTrue(text, text.contains("s3=on"))
        assertTrue(text, text.contains("iso<=1600"))
    }

    @Test
    fun offlineTableIsMonotonicAndAnchored() {
        val table = NoiseProfileStats.offlineRatioTable()
        assertTrue(table.size >= 6)
        assertNotNull(table[100])
        assertNotNull(table[102400])
        // The chroma side scales with the noise floor, so the table must rise
        // with ISO; a flat or falling table would be a transcription error.
        var prev = Float.NEGATIVE_INFINITY
        for ((bucket, ratio) in table) {
            assertTrue("bucket=$bucket ratio=$ratio must exceed previous $prev", ratio > prev)
            prev = ratio
        }
    }

    @Test
    fun offlineRatioInterpolatesBetweenAnchors() {
        val at400 = NoiseProfileStats.offlineRatioFor(400)
        val at800 = NoiseProfileStats.offlineRatioFor(800)
        val at1600 = NoiseProfileStats.offlineRatioFor(1600)
        assertTrue("$at800 must lie between $at400 and $at1600", at800 > at400 && at800 < at1600)
    }

    @Test
    fun clampIsoStaysInsideTheLadder() {
        assertEquals(100, NoiseProfileStats.clampIso(1))
        assertEquals(100, NoiseProfileStats.clampIso(50))
        assertEquals(1600, NoiseProfileStats.clampIso(1600))
        assertEquals(102400, NoiseProfileStats.clampIso(999999))
    }
}