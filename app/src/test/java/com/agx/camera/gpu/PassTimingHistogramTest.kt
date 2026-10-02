package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Headless checks on the per-pass GPU timing histogram.
 *
 * The point of these is the reporting contract, not the math: the bundle is
 * read by a person comparing two builds, so a percentile that can come out
 * below the minimum sample, or a report that hides an empty histogram behind a
 * plausible-looking number, is worse than no report at all.
 */
class PassTimingHistogramTest {

    @Test
    fun emptyHistogramReportsNoSamples() {
        val h = PassTimingHistogram("S1.dpc")
        assertEquals(0L, h.sampleCount())
        assertEquals(0L, h.minNs())
        assertEquals(0L, h.maxNs())
        assertEquals(0L, h.meanNs())
        assertEquals(0L, h.percentileNs(0.5))
        assertTrue(h.report(), h.report().contains("n=0"))
        assertTrue(h.histogramText(), h.histogramText().contains("(empty)"))
    }

    @Test
    fun negativeSamplesAreClampedRatherThanStored() {
        val h = PassTimingHistogram("S3.gf")
        h.add(-5L)
        assertEquals(1L, h.sampleCount())
        // A GL query can never report negative time; clamping to 0 keeps a
        // driver artifact out of the mean instead of skewing it low.
        assertEquals(0L, h.minNs())
        assertEquals(0L, h.meanNs())
    }

    @Test
    fun minMaxMeanTrackExactSamples() {
        val h = PassTimingHistogram("S5.main1")
        h.add(1_000_000L)
        h.add(3_000_000L)
        h.add(2_000_000L)
        assertEquals(3L, h.sampleCount())
        assertEquals(1_000_000L, h.minNs())
        assertEquals(3_000_000L, h.maxNs())
        assertEquals(2_000_000L, h.meanNs())
    }

    @Test
    fun percentilesAreNeverOptimistic() {
        val h = PassTimingHistogram("S5.main2")
        // 100 samples spanning 1us..10ms.
        for (i in 1..100) h.add(i * 100_000L)

        val p50 = h.percentileNs(0.50)
        val p95 = h.percentileNs(0.95)
        assertTrue("p50=$p50 must not be below the true median", p50 >= 5_000_000L)
        assertTrue("p95=$p95 must not be below the true p95", p95 >= 9_500_000L)
        assertTrue("p50=$p50 <= p95=$p95", p50 <= p95)
    }

    @Test
    fun percentilesNeverUnderstateTheSampleTheyDescribe() {
        // Buckets report their upper edge, so a percentile may exceed the true
        // value. It must never fall below it: an understated percentile is the
        // direction that hides a regression.
        val h = PassTimingHistogram("S5.main1")
        for (i in 1..100) h.add(i * 100_000L)
        for (p in listOf(0.01, 0.25, 0.5, 0.75, 0.95, 0.99)) {
            val target = (100.0 * p).toLong().coerceAtLeast(1L)
            val trueValue = target * 100_000L
            val reported = h.percentileNs(p)
            assertTrue("p=$p reported=$reported must be >= true=$trueValue", reported >= trueValue)
        }
    }

    @Test
    fun percentileFractionIsClamped() {
        val h = PassTimingHistogram("S1.passthrough")
        h.add(2_000_000L)
        // Out-of-range fractions must degrade to the endpoints, not throw.
        assertEquals(h.percentileNs(0.0), h.percentileNs(-5.0))
        assertEquals(h.percentileNs(1.0), h.percentileNs(5.0))
    }

    @Test
    fun resetClearsEverything() {
        val h = PassTimingHistogram("S3.gf")
        h.add(1_000_000L)
        h.add(2_000_000L)
        h.reset()
        assertEquals(0L, h.sampleCount())
        assertEquals(0L, h.maxNs())
        assertEquals(0L, h.minNs())
        assertTrue(h.report().contains("n=0"))
    }

    @Test
    fun reportMarksPercentilesAsUpperBounds() {
        // Bucket edges are 1.5x apart, so with a tight cluster of samples the
        // p95 bucket's top edge sits above maxNs. The device log showed
        // "p95=3.330ms max=2.998ms" for S3.gf, which reads as a contradiction
        // unless the report says the percentile is a bound. Lock the marker so
        // a future format tweak cannot quietly drop it and make the timing
        // table look self-contradictory again.
        val h = PassTimingHistogram("S3.gf")
        repeat(120) { h.add(2_990_000L + (it % 10) * 1_000L) }
        val report = h.report()
        assertTrue(report, h.percentileNs(0.95) > h.maxNs())
        assertTrue(report, report.contains("p50<=${String.format("%.3f", h.percentileNs(0.50) / 1.0e6)}"))
        assertTrue(report, report.contains("p95<=${String.format("%.3f", h.percentileNs(0.95) / 1.0e6)}"))
    }

    @Test
    fun reportAndHistogramTextAreAscii() {
        val h = PassTimingHistogram("S5.statsH1")
        for (i in 1..50) h.add(i * 250_000L)
        val report = h.report()
        assertTrue(report, report.all { it.code < 128 })
        val dump = h.histogramText()
        assertTrue(dump, dump.all { it.code < 128 })
        assertTrue(report, report.contains("S5.statsH1"))
        assertTrue(dump, dump.contains("S5.statsH1"))
    }

    @Test
    fun oversizedSampleLandsInTheLastBucketRatherThanBeingDropped() {
        val h = PassTimingHistogram("S5.main1")
        // 60s is past the last bucket edge; the sample must still be counted,
        // because dropping it would hide a stall rather than report it.
        h.add(60_000_000_000L)
        assertEquals(1L, h.sampleCount())
        assertEquals(60_000_000_000L, h.maxNs())
        // The sample must land in a bucket that actually covers it, so the
        // percentile reports it rather than the top edge of a much lower one.
        assertTrue(
            "p50=${h.percentileNs(0.5)} must cover the 60s sample",
            h.percentileNs(0.5) >= 60_000_000_000L
        )
    }

@Test
    fun histogramCoversEveryValueTheTimerCanStillReport() {
        // The timer can report anything below the 32-bit wrap (4.294967296 s),
        // so the histogram must place samples there in real buckets rather than
        // collapsing them into a top bucket - otherwise a reported percentile
        // would sit far below the sample and hide a multi-second stall.
        val h = PassTimingHistogram("S5.main1")
        val readableCeiling = 4_294_967_295L
        for (ns in listOf(1_000L, 1_000_000L, 500_000_000L, 2_000_000_000L, readableCeiling)) {
            h.add(ns)
        }
        assertEquals(5L, h.sampleCount())
        assertEquals(readableCeiling, h.maxNs())
        // p100 targets the top sample, so it is the value that proves the
        // ceiling lands in a bucket that actually covers it.
        assertTrue(
            "p100=${h.percentileNs(1.0)} must cover the ${readableCeiling}ns sample",
            h.percentileNs(1.0) >= readableCeiling
        )
    }
}