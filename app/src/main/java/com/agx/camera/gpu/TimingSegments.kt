package com.agx.camera.gpu

/**
 * Identity of one timing segment.
 *
 * A timing number only means something next to the configuration that produced
 * it, so the key carries everything that changes the cost: the three denoise
 * sliders, which of the two RAW cost worlds ran, and the preview size.
 *
 * The slider values are the raw 0-100 numbers rather than the derived
 * strengths. The raw number is what a reader can reproduce from the UI, so if
 * the slider-to-strength mapping is retuned later the already-captured segments
 * stay readable under the settings they were actually taken at instead of being
 * silently reinterpreted under the new mapping.
 *
 * Zoom enters as the two-valued regime and not as the continuous factor: the
 * only thing the factor decides is whether the sparse grid chain ran (k <= 2)
 * or the demosaic's inline denoise ran (k > 2). Those are two separate cost
 * worlds and nothing between them is reachable without crossing that gate.
 *
 * ISO is deliberately not part of the key. Auto-exposure walks it continuously,
 * so keying on it would open a new segment on nearly every frame and no
 * histogram would ever accumulate a usable sample count. The ISO distribution
 * is recorded separately by the profile statistics.
 *
 * Thermal state is part of the key, as two separate fields, because they
 * disagree and both carry information.
 *
 * [appTier] is the one that matters. ThermalManager judges from smoothed
 * battery temperature plus observed frame time, so it reaches WARM while the
 * platform thermal API is still reporting `none`; it is also what actually caps
 * the preview resolution. Keying on it is what keeps a throttled segment from
 * being used as a baseline, because a clock drop and a self-imposed resolution
 * cut both rescale the passes a reader would compare.
 *
 * [platformThrottled] is the platform's own view, kept in the key so two
 * segments that differ on it are never compared, and printed so a reader can
 * see when the platform and the app disagreed - which is itself the signal that
 * the app's frame-rate heuristic is reacting before the platform notices.
 *
 * appTier is keyed at full resolution rather than collapsed to a boolean, the
 * way the platform status is. The four tiers are distinct cost worlds -
 * WARM/HOT cap the preview at 480, CRITICAL suppresses preview and the RAW
 * stream - and the value is already de-jittered by ThermalManager's slow-fall
 * EMA plus a step-down margin, so it changes rarely enough to key on.
 */
data class MeasurementSegment(
    val s1Raw: Int,
    val s3Raw: Int,
    val s5Raw: Int,
    val sparseChainRegime: Boolean,
    val viewWidth: Int,
    val viewHeight: Int,
    val appTier: String,
    val platformThrottled: Boolean
) {

    fun zoomRegimeText(): String = if (sparseChainRegime) "sparse(k<=2)" else "inline(k>2)"

    /**
     * Whether this segment was captured while something other than the
     * platform's own reading was throttling the device.
     *
     * Deliberately keyed on the app's tier rather than the platform status.
     * ThermalManager judges from smoothed battery temperature *and* observed
     * frame time, so it escalates to WARM while the Android thermal API still
     * reports `none` - which is exactly what a run recorded: segment headers
     * read `thermal=none` while the app had already capped the preview at 480.
     * Keying on the platform value therefore protected nothing. Worse, it
     * protected nothing only in the lucky case where the throttle happened to
     * move the resolution, since that split the segment anyway; had WARM only
     * dropped the frame rate, the throttled samples would have landed in the
     * same segment as the cool ones and quietly become the baseline.
     */
    fun isThrottled(): Boolean = appTier != APP_TIER_NORMAL

    fun label(): String =
        "s1=$s1Raw s3=$s3Raw s5=$s5Raw zoom=${zoomRegimeText()} view=${viewWidth}x$viewHeight" +
            " appTier=$appTier platformTherm=${if (platformThrottled) "throttled" else "none"}" +
            (if (isThrottled()) " THROTTLED" else "")

    /** Short identifier, used where the full key would crowd the line. */
    fun tag(): String =
        "${s1Raw}-${s3Raw}-${s5Raw}-${if (sparseChainRegime) "sp" else "in"}-${viewWidth}x$viewHeight" +
            "-$appTier${if (platformThrottled) "-ptherm" else ""}"

    companion object {
        /**
         * Tier name meaning "nothing is throttling the preview".
         *
         * Spelled as ThermalManager.State.NORMAL.name rather than imported, so
         * the key stays comparable if the enum is renamed: a silent mismatch
         * would make every segment read as throttled, or worse, none of them.
         */
        const val APP_TIER_NORMAL = "NORMAL"
    }
}

/**
 * Timing samples grouped by the pipeline configuration that produced them.
 *
 * The benchmark has to work in any slider combination and has to say which
 * combination each number came from, so samples land in a segment rather than
 * in one global histogram. A segment holds its key for as long as that key is
 * current and is frozen into the session list when the key moves. The report
 * prints every segment, not just the newest, because the comparison a reader
 * usually wants is between two older ones.
 *
 * A query is stamped with the segment that was current when the query was
 * opened, never the one current when its result is harvested. Results arrive a
 * frame or two late, so attributing them at harvest time would push samples
 * across a boundary into a segment whose key claims they were captured under
 * different settings.
 *
 * Access is synchronized because arm and disarm arrive on whichever thread
 * flipped the switch, which is the UI thread in practice, while the frame and
 * harvest calls come from the GL thread.
 *
 * There are no GL calls here, so the grouping rules are unit-testable on their
 * own, which is what makes the cross-segment attribution rule checkable without
 * a device.
 */
class TimingSegments(private val maxSegments: Int = MAX_SEGMENTS) {

    class Segment(val key: MeasurementSegment, val seq: Int) {
        private val histograms = LinkedHashMap<String, PassTimingHistogram>()

        var frames: Int = 0
            internal set

        var inFlight: Int = 0
            internal set

        /**
         * Worst platform thermal level seen in this segment, and the zoom range
         * observed.
         *
         * Neither is the throttle decision - that is [MeasurementSegment.appTier]
         * in the key. These are context. Zoom is the more consequential of the
         * two: it moved 6.40 to 8.53 in the same frame that the preview size
         * changed 640x480 to 480x360, and the segment key cannot tell those
         * apart. A reader comparing those two segments needs to see that two
         * things moved at once, otherwise the cost difference gets attributed
         * to whichever of them they noticed first.
         */
        var worstThermal: Int = 0
            internal set

        var minZoomK: Float = Float.MAX_VALUE
            internal set

        var maxZoomK: Float = -Float.MAX_VALUE
            internal set

        /** Zoom range as a short suffix, collapsed when it never moved. */
        fun zoomRangeText(): String =
            if (maxZoomK - minZoomK < 0.005f) String.format("k=%.2f", minZoomK)
            else String.format("k=%.2f..%.2f", minZoomK, maxZoomK)

        internal fun noteContext(thermalStatus: Int, zoomK: Float) {
            if (thermalStatus > worstThermal) worstThermal = thermalStatus
            if (zoomK < minZoomK) minZoomK = zoomK
            if (zoomK > maxZoomK) maxZoomK = zoomK
        }

        fun histogramFor(passName: String): PassTimingHistogram =
            histograms.getOrPut(passName) { PassTimingHistogram(passName) }

        fun passes(): List<PassTimingHistogram> = histograms.values.toList()

        fun sampleCount(): Long {
            var n = 0L
            for (h in histograms.values) n += h.sampleCount()
            return n
        }

        /**
         * Fewer than [LOW_CONFIDENCE_FRAMES] frames is not enough for the
         * percentiles to mean anything. A slider nudged for two seconds must not
         * leave behind numbers that read like a settled measurement.
         */
        fun isLowConfidence(): Boolean = frames < LOW_CONFIDENCE_FRAMES
    }

    private val segments = ArrayList<Segment>()
    private var nextSeq = 0
    private var droppedSegments = 0
    private var droppedSamples = 0L

    val segmentCount: Int
        @Synchronized get() = segments.size

    val droppedSegmentCount: Int
        @Synchronized get() = droppedSegments

    val droppedSampleCount: Long
        @Synchronized get() = droppedSamples

    val lowConfidenceThreshold: Int get() = LOW_CONFIDENCE_FRAMES

    @Synchronized
    fun current(): Segment? = segments.lastOrNull()

    @Synchronized
    fun currentSeq(): Int = current()?.seq ?: NO_SEGMENT

    @Synchronized
    fun all(): List<Segment> = segments.toList()

    /**
     * Opens a frame under [key], freezing the previous segment when the key
     * moved. Returns the segment this frame belongs to.
     *
     * Freezing does not close a segment. Results for queries opened before the
     * switch are still outstanding, so the frozen segment stays in the list and
     * still accepts those samples until the cap retires it.
     */
    @Synchronized
    fun enterFrame(key: MeasurementSegment): Segment {
        val cur = current()
        if (cur != null && cur.key == key) {
            cur.frames++
            return cur
        }
        val seg = Segment(key, nextSeq++)
        seg.frames = 1
        segments.add(seg)
        evictOverflow()
        return seg
    }

    /**
     * Files the per-frame context that is recorded but not keyed.
     *
     * Kept apart from [enterFrame] because the key is a value and this is an
     * observation: recording zoom and thermal here means a segment stays
     * readable without letting either one split segments.
     */
    @Synchronized
    fun noteFrameContext(seq: Int, thermalStatus: Int, zoomK: Float) {
        findBySeq(seq)?.noteContext(thermalStatus, zoomK)
    }

    /** Records that a query was opened against [seq]. */
    @Synchronized
    fun noteBegin(seq: Int): Boolean {
        val seg = findBySeq(seq) ?: return false
        seg.inFlight++
        return true
    }

    /**
     * Files a harvested sample under the segment its query was opened against.
     * Returns false when that segment has already been retired, which happens
     * once the cap drops a segment that still had queries outstanding.
     */
    @Synchronized
    fun addSample(passName: String, nanos: Long, seq: Int): Boolean {
        val seg = findBySeq(seq)
        if (seg == null) {
            droppedSamples++
            return false
        }
        seg.histogramFor(passName).add(nanos)
        seg.inFlight--
        return true
    }

    /**
     * Settles a query that produced no usable reading, such as one discarded
     * because the context was reset. The segment still has to be told, or it
     * would sit pinned as in-flight and never become evictable.
     */
    @Synchronized
    fun noteSettled(seq: Int) {
        val seg = findBySeq(seq) ?: return
        if (seg.inFlight > 0) seg.inFlight--
    }

    /** Drops every segment. Used when the benchmark is armed or disarmed. */
    @Synchronized
    fun reset() {
        segments.clear()
        droppedSegments = 0
        droppedSamples = 0L
    }

    /**
     * Drops the oldest segments past the cap so a long session cannot grow the
     * histogram set without bound. A segment with queries still outstanding is
     * only evicted when no settled segment is available to make room; the
     * samples it was holding are then counted as dropped and said so in the
     * report rather than vanishing quietly.
     */
    private fun evictOverflow() {
        while (segments.size > maxSegments) {
            var victim: Segment? = segments.firstOrNull { it.inFlight == 0 }
            if (victim == null) victim = segments[0]
            val gone = victim ?: break
            segments.remove(gone)
            droppedSegments++
            droppedSamples += gone.sampleCount()
        }
    }

    private fun findBySeq(seq: Int): Segment? {
        if (seq == NO_SEGMENT) return null
        for (s in segments) if (s.seq == seq) return s
        return null
    }

    /**
     * Every segment in the session, oldest first, each with its per-pass
     * summary and histogram. Not only the newest: the segment a reader wants to
     * compare against is normally one they already scrolled past.
     */
    @Synchronized
    fun report(): String {
        if (segments.isEmpty()) return "timing segments: none yet (no frame has been timed)"
        val sb = StringBuilder()
        sb.append("timing segments: ").append(segments.size)
            .append(" of at most ").append(maxSegments).append(", oldest first\n")
        for (seg in segments) {
            sb.append("  [").append(seg.seq).append("] ")
                .append(seg.key.label())
                .append(" frames=").append(seg.frames)
                .append(" platformThermal=").append(ThermalStatus.text(seg.worstThermal))
                .append(" ").append(seg.zoomRangeText())
            if (seg.isLowConfidence()) {
                sb.append(" low-confidence (fewer than ").append(LOW_CONFIDENCE_FRAMES)
                    .append(" frames)")
            }
            sb.append('\n')
            if (seg.passes().isEmpty()) {
                sb.append("      no pass produced a sample in this segment\n")
                continue
            }
            for (hist in seg.passes()) {
                sb.append("      ").append(hist.report()).append('\n')
            }
            for (hist in seg.passes()) {
                for (line in hist.histogramText().lines()) {
                    if (line.isEmpty()) continue
                    sb.append("      ").append(line).append('\n')
                }
            }
        }
        if (droppedSegments > 0) {
            sb.append("note: ").append(droppedSegments)
                .append(" oldest segment(s) dropped past the cap of ").append(maxSegments)
                .append(", taking ").append(droppedSamples)
                .append(" already-harvested sample(s) with them\n")
        }
        val throttled = segments.filter { it.key.isThrottled() }
        if (throttled.isNotEmpty()) {
            sb.append("note: ").append(throttled.size).append(" of ").append(segments.size)
                .append(" segment(s) were captured while the app's own thermal tier was above ")
                .append(MeasurementSegment.APP_TIER_NORMAL).append(". Those segments are kept for the record but must")
                .append(" not be used as the baseline for a regression comparison: a clock drop")
                .append(" rescales every pass at once, and the app's WARM/HOT tier also cuts the")
                .append(" preview resolution itself, so a throttled reference makes any candidate")
                .append(" measure as an improvement. Compare segments whose keys match, appTier")
                .append(" included.\n")
        }
        return sb.toString()
    }

    companion object {
        /**
         * Eight segments is enough to hold a slider walk across one preview
         * session while bounding the histograms to a few kilobytes. Past that
         * the oldest is dropped and the report says so.
         */
        const val MAX_SEGMENTS = 8

        /** Under this many frames a segment's percentiles are not trustworthy. */
        const val LOW_CONFIDENCE_FRAMES = 30

        const val NO_SEGMENT = -1
    }
}