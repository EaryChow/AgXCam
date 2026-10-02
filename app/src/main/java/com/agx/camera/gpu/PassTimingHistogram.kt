package com.agx.camera.gpu

/**
 * Per-pass GPU time histogram. One instance per pass name, fed by the
 * measurement switch and reported into the debug bundle.
 *
 * Samples are nanoseconds straight out of a GL_TIME_ELAPSED query, so they
 * measure GPU elapsed time.
 *
 * Storage is a log-spaced bucket array plus exact min/max/sum/count, which
 * keeps the per-frame cost to one add on the preview thread. Percentile reads
 * come off the bucket boundaries; within a bucket the upper edge is reported,
 * so a percentile reads slightly high.
 */
class PassTimingHistogram(private val name: String, bucketCount: Int = HISTOGRAM_BUCKETS) {

    private val counts = LongArray(bucketCount)
    private val bucketTopNs = LongArray(bucketCount)
    private var sampleCount = 0L
    private var sumNs = 0L
    private var minNs = Long.MAX_VALUE
    private var maxNs = 0L

    init {
        // Precompute the bucket upper edges once; bucket i holds
        // (nsToBucketLower(i), nsToBucketUpper(i)] in nanoseconds.
        var edge = FIRST_BUCKET_TOP_NS
        for (i in 0 until bucketCount) {
            bucketTopNs[i] = edge
            edge = edge + (edge shr 1) + 1
        }
    }

    fun add(nanos: Long) {
        val ns = if (nanos < 0L) 0L else nanos
        var i = bucketOf(ns)
        if (i >= counts.size) i = counts.size - 1
        counts[i]++
        sampleCount++
        sumNs += ns
        if (ns < minNs) minNs = ns
        if (ns > maxNs) maxNs = ns
    }

    fun reset() {
        java.util.Arrays.fill(counts, 0L)
        sampleCount = 0L
        sumNs = 0L
        minNs = Long.MAX_VALUE
        maxNs = 0L
    }

    fun name(): String = name

    fun sampleCount(): Long = sampleCount

    fun minNs(): Long = if (sampleCount == 0L) 0L else minNs

    fun maxNs(): Long = maxNs

    fun meanNs(): Long = if (sampleCount == 0L) 0L else sumNs / sampleCount

    /** Percentile in nanoseconds, from the bucket edges. 0 when empty. */
    fun percentileNs(fraction: Double): Long {
        if (sampleCount == 0L) return 0L
        val f = fraction.coerceIn(0.0, 1.0)
        val target = ((sampleCount.toDouble()) * f).toLong().coerceAtLeast(1L)
        var seen = 0L
        for (i in counts.indices) {
            seen += counts[i]
            if (seen >= target) return bucketTopNs[i]
        }
        return maxNs
    }

    /**
     * ASCII one-block report, e.g. "S5.main2 n=310 mean=2.14ms p50<=2.05ms p95<=3.90ms".
     *
     * Percentiles print as upper bounds. They are read off 1.5x-spaced bucket
     * edges, so the 95th percentile can land in a bucket whose top sits above
     * the largest sample observed: a p95 of 3.90ms next to a max of 3.4ms is
     * the histogram being coarse, not a contradiction. The `<=` says which it
     * is. Printing a bare number there invites the reader to conclude the
     * timing data is broken, which is how a real fault would be dismissed.
     */
    fun report(): String {
        if (sampleCount == 0L) return "$name n=0 (no GPU samples)"
        return String.format(
            "%s n=%d mean=%.3fms p50<=%.3fms p95<=%.3fms min=%.3fms max=%.3fms",
            name, sampleCount,
            meanNs() / 1.0e6,
            percentileNs(0.50) / 1.0e6,
            percentileNs(0.95) / 1.0e6,
            minNs() / 1.0e6,
            maxNs() / 1.0e6
        )
    }

    /**
     * Log-spaced histogram dump. Bucket edges are 1.5x apart so the printed
     * table stays short while still separating sub-millisecond passes from
     * double-digit millisecond ones.
     */
    fun histogramText(): String {
        val sb = StringBuilder()
        sb.append("[$name] samples=").append(sampleCount).append('\n')
        var seen = 0L
        var lowerNs = 0L
        for (i in counts.indices) {
            val c = counts[i]
            val upperNs = bucketTopNs[i]
            if (c > 0L) {
                seen += c
                sb.append("  ")
                    .append(fmtBucket(lowerNs))
                    .append("..")
                    .append(fmtBucket(upperNs))
                    .append("ms  ")
                    .append(c)
                    .append("  ")
                    .append(bar(c, sampleCount))
                    .append('\n')
            }
            lowerNs = upperNs + 1
        }
        if (sampleCount == 0L) sb.append("  (empty)\n")
        return sb.toString()
    }

    private fun bucketOf(ns: Long): Int {
        // Same walk as the init loop: find the first bucket whose top edge
        // covers ns. Linear because bucketCount is small (32) and this runs
        // once per sampled pass per frame.
        var edge = FIRST_BUCKET_TOP_NS
        for (i in 0 until bucketTopNs.size) {
            if (ns <= edge) return i
            edge = edge + (edge shr 1) + 1
        }
        return bucketTopNs.size
    }

    companion object {
        // 1us first edge, 1.5x growth. 64 buckets reach ~2 days, comfortably past
        // any stall worth timing, so a sample always lands in a bucket that
        // covers it.
        private const val FIRST_BUCKET_TOP_NS = 1_000L
        const val HISTOGRAM_BUCKETS = 64

        private fun fmtBucket(ns: Long): String = String.format("%.3f", ns / 1.0e6)

        private fun bar(count: Long, total: Long): String {
            val width = 40
            val filled = if (total <= 0L) 0 else ((count.toDouble() / total.toDouble()) * width).toInt()
            val n = filled.coerceIn(0, width)
            val sb = StringBuilder(width)
            for (i in 0 until width) sb.append(if (i < n) '#' else '.')
            return sb.toString()
        }
    }
}