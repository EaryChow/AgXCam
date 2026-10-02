package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Segment grouping and cross-segment attribution, tested without a GL context.
 *
 * The rule that matters most here is the one a single global histogram cannot
 * express: a sample belongs to the configuration that was live when its query
 * was opened, not the one live when the result arrived. Results land a frame or
 * two late, so attributing them at harvest time would push them across a slider
 * change into a segment whose key claims they were captured under other
 * settings.
 */
class TimingSegmentsTest {

    private fun key(
        s1: Int = 0,
        s3: Int = 0,
        s5: Int = 0,
        sparse: Boolean = true,
        w: Int = 640,
        h: Int = 480,
        appTier: String = "NORMAL",
        platformThrottled: Boolean = false
    ) = MeasurementSegment(s1, s3, s5, sparse, w, h, appTier, platformThrottled)

    @Test
    fun sameKeyKeepsOneSegmentAcrossFrames() {
        val segments = TimingSegments()
        repeat(10) { segments.enterFrame(key(s3 = 32)) }

        assertEquals(1, segments.segmentCount)
        assertEquals(10, segments.all()[0].frames)
    }

    @Test
    fun appTierSplitsSegmentsEvenWhenThePlatformReportsNoThrottle() {
        // The regression this keying rule exists to prevent. A recorded run had
        // segment headers reading "thermal=none" while ThermalManager had
        // already gone WARM and capped the preview at 480: the platform API is
        // less sensitive than the app's own judgement, because the app also
        // judges on frame time. Keying on the platform value protected nothing,
        // and it only looked like it worked because the resolution cap happened
        // to split the segment anyway. Had WARM only dropped the frame rate,
        // both tiers would have shared one segment and the throttled samples
        // would have become the baseline.
        val segments = TimingSegments()
        repeat(40) { segments.enterFrame(key(s3 = 32, appTier = "NORMAL", platformThrottled = false)) }
        repeat(40) { segments.enterFrame(key(s3 = 32, appTier = "WARM", platformThrottled = false)) }

        assertEquals("WARM must not share NORMAL's segment", 2, segments.segmentCount)
        assertEquals(40, segments.all()[0].frames)
        assertEquals(40, segments.all()[1].frames)
        assertFalse(segments.all()[0].key.isThrottled())
        assertTrue(segments.all()[1].key.isThrottled())
    }

    @Test
    fun everyAppTierAboveNormalCountsAsThrottled() {
        // WARM and HOT share the 480p cap but HOT also kills the flash, and
        // CRITICAL suppresses preview and the RAW stream outright, so each is
        // its own cost world and none of them may serve as a baseline.
        for (tier in listOf("WARM", "HOT", "CRITICAL")) {
            assertTrue(tier, MeasurementSegment(0, 0, 0, true, 640, 480, tier, false).isThrottled())
        }
        assertFalse(MeasurementSegment(0, 0, 0, true, 640, 480, "NORMAL", false).isThrottled())
        assertEquals("NORMAL", MeasurementSegment.APP_TIER_NORMAL)
    }

    @Test
    fun platformThrottleAloneStillSplitsTheKey() {
        // Kept as its own field so a genuine platform-throttled run is never
        // compared against a cool one, and printed so a reader can see when the
        // two sources disagreed - which is itself the diagnostic signal.
        val cool = key(platformThrottled = false)
        val platformHot = key(platformThrottled = true)
        assertFalse(cool == platformHot)
        assertFalse(
            "platform throttling must not by itself mark the segment throttled",
            platformHot.isThrottled()
        )
        assertTrue(platformHot.label().contains("platformTherm=throttled"))
    }

    @Test
    fun segmentLabelNamesBothThermalSourcesAndTheThrottleVerdict() {
        val label = key(appTier = "HOT", platformThrottled = false).label()
        assertTrue(label, label.contains("appTier=HOT"))
        assertTrue(label, label.contains("platformTherm=none"))
        assertTrue(label, label.contains("THROTTLED"))
    }

    @Test
    fun reportNamesThrottledSegmentsAndSaysTheyAreNotBaselines() {
        val segments = TimingSegments()
        segments.enterFrame(key(s3 = 32, appTier = "WARM"))

        val report = segments.report()
        assertTrue(report, report.contains("THROTTLED"))
        assertTrue(report, report.contains("must not be used as the"))
        assertTrue(report, report.contains("appTier"))
    }

    @Test
    fun reportStillNamesThePlatformLevelAsContext() {
        // Platform thermal stays reported per segment even though it no longer
        // decides anything: "platform said none while the app said WARM" is the
        // observation that explains why the two fields are separate.
        val segments = TimingSegments()
        val seg = segments.enterFrame(key(s3 = 32, appTier = "WARM"))
        segments.noteFrameContext(seg.seq, thermalStatus = 0, zoomK = 8.53f)

        val report = segments.report()
        assertTrue(report, report.contains("platformThermal=none"))
        assertTrue(report, report.contains("appTier=WARM"))
    }

    @Test
    fun statusBelowThePlatformThresholdDoesNotFlagThrottling() {
        assertFalse(ThermalStatus.isThrottled(0))
        assertFalse(ThermalStatus.isThrottled(1))
        assertFalse(ThermalStatus.isThrottled(2))
        assertTrue(ThermalStatus.isThrottled(3))
        assertTrue(ThermalStatus.isThrottled(7))
    }

    @Test
    fun zoomIsRecordedPerSegmentWithoutSplittingIt() {
        // Zoom moved 6.40 to 8.53 in the same frame the preview size changed, so
        // the key cannot tell those two apart. The range has to be visible in
        // the report even though it must not create a segment.
        val segments = TimingSegments()
        val seg = segments.enterFrame(key(s3 = 32))
        segments.noteFrameContext(seg.seq, thermalStatus = 0, zoomK = 6.40f)
        segments.noteFrameContext(seg.seq, thermalStatus = 2, zoomK = 8.53f)

        assertEquals("one segment must hold both zooms", 1, segments.segmentCount)
        val only = segments.all()[0]
        assertEquals(6.40f, only.minZoomK, 0.001f)
        assertEquals(8.53f, only.maxZoomK, 0.001f)
        assertEquals("worst level seen wins", 2, only.worstThermal)
        assertTrue(segments.report().contains("k=6.40..8.53"))
    }

    @Test
    fun reportCollapsesTheZoomRangeWhenItNeverMoved() {
        val segments = TimingSegments()
        val seg = segments.enterFrame(key(s3 = 32))
        repeat(5) { segments.noteFrameContext(seg.seq, thermalStatus = 0, zoomK = 8.53f) }

        assertTrue(segments.report().contains("k=8.53"))
    }

    @Test
    fun eachSliderMoveOpensANewSegment() {
        val segments = TimingSegments()
        segments.enterFrame(key(s1 = 0))
        segments.enterFrame(key(s1 = 10))
        segments.enterFrame(key(s3 = 32))
        segments.enterFrame(key(s5 = 40))

        assertEquals(4, segments.segmentCount)
    }

    @Test
    fun viewSizeChangeOpensANewSegment() {
        val segments = TimingSegments()
        segments.enterFrame(key(w = 640, h = 480))
        segments.enterFrame(key(w = 1280, h = 720))

        assertEquals("a different view is a different cost", 2, segments.segmentCount)
    }

    @Test
    fun zoomEntersAsARegimeNotAsTheContinuousFactor() {
        val segments = TimingSegments()
        segments.enterFrame(key(sparse = true))
        segments.enterFrame(key(sparse = false))

        assertEquals("the two cost worlds are distinct segments", 2, segments.segmentCount)
        assertEquals("sparse(k<=2)", segments.all()[0].key.zoomRegimeText())
        assertEquals("inline(k>2)", segments.all()[1].key.zoomRegimeText())
    }

    @Test
    fun activityOtherThanAKeyChangeDoesNotOpenASegment() {
        // ISO is not in the key, and neither is per-pass query traffic. If
        // either of those opened a segment, auto-exposure alone would leave no
        // histogram with a usable sample count.
        val segments = TimingSegments()
        val seg = segments.enterFrame(key(s3 = 32))
        repeat(20) {
            segments.noteBegin(seg.seq)
            segments.addSample("S2.sigma_post_s3", 1_000_000L, seg.seq)
            segments.enterFrame(key(s3 = 32))
        }

        assertEquals(1, segments.segmentCount)
        assertEquals(20, seg.sampleCount())
    }

    @Test
    fun sampleLandsInTheSegmentItsQueryWasOpenedAgainst() {
        val segments = TimingSegments()
        val before = segments.enterFrame(key(s3 = 32))
        segments.noteBegin(before.seq)

        // The frame ends, the slider moves, and the result only arrives now.
        val after = segments.enterFrame(key(s3 = 0))
        segments.addSample("S2.sigma_post_s3", 1_000_000L, before.seq)

        assertEquals("late result must not follow the current segment", 0, after.sampleCount())
        assertEquals(1, before.sampleCount())
    }

    @Test
    fun segmentWithAnOpenQuerySurvivesAKeyChange() {
        val segments = TimingSegments()
        val before = segments.enterFrame(key(s3 = 32))
        segments.noteBegin(before.seq)
        segments.enterFrame(key(s3 = 0))

        assertTrue("frozen segment still holds an outstanding query", segments.all().any { it.seq == before.seq })
        assertTrue(segments.addSample("S5", 2_000_000L, before.seq))
        assertEquals(1, before.sampleCount())
    }

    @Test
    fun noSegmentMeansNothingCanBeFiled() {
        val segments = TimingSegments()
        assertEquals(TimingSegments.NO_SEGMENT, segments.currentSeq())
        assertFalse(segments.addSample("S5", 1_000_000L, TimingSegments.NO_SEGMENT))
        assertFalse("a sample with nowhere to go must not claim a segment", segments.noteBegin(TimingSegments.NO_SEGMENT))
    }

    @Test
    fun settledQueryReleasesItsInFlightCount() {
        val segments = TimingSegments()
        val seg = segments.enterFrame(key())
        segments.noteBegin(seg.seq)
        segments.noteSettled(seg.seq)
        segments.noteSettled(seg.seq)

        assertEquals("a discard must not leave the segment pinned", 0, seg.inFlight)
    }

    @Test
    fun capDropsTheOldestAndTheReportSaysSo() {
        val segments = TimingSegments()
        for (s5 in 0..12) {
            val seg = segments.enterFrame(key(s5 = s5))
            segments.addSample("S5", 1_000_000L, seg.seq)
        }

        assertEquals(TimingSegments.MAX_SEGMENTS, segments.segmentCount)
        assertEquals(13 - TimingSegments.MAX_SEGMENTS, segments.droppedSegmentCount)

        val report = segments.report()
        assertTrue("dropped segments must be visible", report.contains("dropped"))
        assertTrue(report.contains("cap of ${TimingSegments.MAX_SEGMENTS}"))
    }

    @Test
    fun capPrefersToDropASettledSegmentOverOneWithAQueryOpen() {
        val segments = TimingSegments()
        val first = segments.enterFrame(key(s5 = 0))
        segments.noteBegin(first.seq)
        for (s5 in 1..TimingSegments.MAX_SEGMENTS) {
            segments.enterFrame(key(s5 = s5))
        }

        assertTrue("the in-flight segment must not be the victim", segments.all().any { it.seq == first.seq })
    }

    @Test
    fun shortSegmentIsFlaggedLowConfidence() {
        val segments = TimingSegments()
        val short = segments.enterFrame(key(s5 = 40))
        repeat(TimingSegments.LOW_CONFIDENCE_FRAMES) {
            segments.enterFrame(key(s1 = 1))
        }
        repeat(TimingSegments.LOW_CONFIDENCE_FRAMES) {
            segments.enterFrame(key(s1 = 2))
        }

        assertTrue("two frames must not read like a settled measurement", short.isLowConfidence())
        assertTrue(segments.report().contains("low-confidence"))
    }

    @Test
    fun settledSegmentIsNotLowConfidence() {
        val segments = TimingSegments()
        val seg = segments.enterFrame(key())
        repeat(TimingSegments.LOW_CONFIDENCE_FRAMES - 1) { segments.enterFrame(key()) }

        assertFalse(seg.isLowConfidence())
    }

    @Test
    fun reportPrintsEverySegmentNotOnlyTheNewest() {
        val segments = TimingSegments()
        for (s5 in 0..3) {
            val seg = segments.enterFrame(key(s5 = s5))
            segments.addSample("S2.sigma_post_s3", 1_000_000L, seg.seq)
        }
        val report = segments.report()

        for (s5 in 0..3) {
            assertTrue("segment s5=$s5 missing from the report", report.contains("s5=$s5"))
        }
        assertTrue(report.contains("S2.sigma_post_s3"))
    }

    @Test
    fun reportNamesTheConfigurationBehindEveryNumber() {
        val segments = TimingSegments()
        val seg = segments.enterFrame(key(s1 = 10, s3 = 32, s5 = 40, sparse = false, w = 1280, h = 720))
        segments.addSample("S5", 1_000_000L, seg.seq)

        val report = segments.report()
        assertTrue(report.contains("s1=10"))
        assertTrue(report.contains("s3=32"))
        assertTrue(report.contains("s5=40"))
        assertTrue(report.contains("inline(k>2)"))
        assertTrue(report.contains("view=1280x720"))
    }

    @Test
    fun resetClearsEverything() {
        val segments = TimingSegments()
        segments.enterFrame(key())
        segments.reset()

        assertEquals(0, segments.segmentCount)
        assertNull(segments.current())
        assertEquals(0, segments.droppedSegmentCount)
    }

    @Test
    fun tagsDistinguishSegmentsWithTheSameSliders() {
        // The tag is what a compact log line carries, so it has to include the
        // regime and view or two segments would be indistinguishable there.
        assertNotEquals(key(sparse = true).tag(), key(sparse = false).tag())
        assertNotEquals(key(w = 640).tag(), key(w = 1280).tag())
        assertEquals(key(sparse = true, w = 640).tag(), key(sparse = true, w = 640).tag())
    }
}