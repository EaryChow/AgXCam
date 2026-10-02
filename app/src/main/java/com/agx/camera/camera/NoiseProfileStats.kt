package com.agx.camera.camera

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Runtime sigma profile statistics, keyed on TWO things: the ISO bucket and
 * whether Stage 3 ran.
 *
 * The single ratio the rest of the package needs is
 *
 *   mean(sigma_hat_C) / mean(sigma_hat_luma)
 *
 * where sigma_hat_C is the per-channel chroma estimate and sigma_hat_luma the
 * luma estimate, both from the same texel neighbourhood. The chroma side is
 * scaled x64 downstream, so a ratio around 64 is the expected order of
 * magnitude; the number exists so the chroma scale factor is anchored on
 * measured data instead of a guess that silently drifts with ISO.
 *
 * The double key is not decoration. The Stage-3 blend changes the residual
 * the estimator sees, so an EMA shared between S3-on and S3-off averages two
 * different populations and reads plausible while being wrong in both. Any
 * S3-on/off acceptance comparison must therefore look at two separate rows.
 *
 * EMA, not a running mean: the estimator's own per-texel variance is large
 * and the ratio jitters frame to frame; a plain mean of ratios is dominated
 * by whichever frames happened to be darkest.
 */
class NoiseProfileStats {

    /** EMA weight for a new sample. Slow enough to ignore single-frame outliers. */
    var emaAlpha: Float = DEFAULT_ALPHA

    /** Relative deviation from the anchor ratio above that raises a warning. */
    var deviationWarnFraction: Float = DEFAULT_DEVIATION_WARN

    private data class Key(val isoBucket: Int, val s3Active: Boolean)

    private class Row {
        var meanSigmaChroma = 0f
        var meanSigmaLuma = 0f
        var samples = 0L
        var lastIso = 0
        var lastUpdateNanos = 0L
    }

    private val rows = LinkedHashMap<Key, Row>()

    /** Feeds one measurement. Rejects non-finite and non-positive inputs. */
    fun update(iso: Int, s3Active: Boolean, meanSigmaChroma: Float, meanSigmaLuma: Float, nowNanos: Long = 0L) {
        if (!meanSigmaChroma.isFinite() || !meanSigmaLuma.isFinite()) return
        if (meanSigmaLuma <= 0f) return
        val key = Key(isoBucketFor(iso), s3Active)
        val row = rows.getOrPut(key) { Row() }
        val alpha = emaAlpha.coerceIn(0.001f, 1f)
        if (row.samples == 0L) {
            row.meanSigmaChroma = meanSigmaChroma
            row.meanSigmaLuma = meanSigmaLuma
        } else {
            row.meanSigmaChroma += alpha * (meanSigmaChroma - row.meanSigmaChroma)
            row.meanSigmaLuma += alpha * (meanSigmaLuma - row.meanSigmaLuma)
        }
        row.samples++
        row.lastIso = iso
        row.lastUpdateNanos = nowNanos
    }

    fun sampleCount(iso: Int, s3Active: Boolean): Long =
        rows[Key(isoBucketFor(iso), s3Active)]?.samples ?: 0L

    fun ratio(iso: Int, s3Active: Boolean): Float? {
        val row = rows[Key(isoBucketFor(iso), s3Active)] ?: return null
        if (row.samples == 0L) return null
        val luma = row.meanSigmaLuma
        if (luma <= 0f) return null
        return row.meanSigmaChroma / luma
    }

    fun meanSigmaChroma(iso: Int, s3Active: Boolean): Float? =
        rows[Key(isoBucketFor(iso), s3Active)]?.meanSigmaChroma

    fun meanSigmaLuma(iso: Int, s3Active: Boolean): Float? =
        rows[Key(isoBucketFor(iso), s3Active)]?.meanSigmaLuma

    /**
     * Deviation of the measured ratio from the committed anchor ratio.
     * Null when either side is unavailable - an absent anchor must
     * read as "not comparable", never as "0% deviation".
     */
    fun deviationFromOffline(iso: Int, s3Active: Boolean, offlineRatio: Float): Float? {
        val measured = ratio(iso, s3Active) ?: return null
        if (offlineRatio <= 0f) return null
        return (measured - offlineRatio) / offlineRatio
    }

    /** True when the measured ratio has drifted far enough to flag it. */
    fun isDeviated(iso: Int, s3Active: Boolean, offlineRatio: Float): Boolean {
        val d = deviationFromOffline(iso, s3Active, offlineRatio) ?: return false
        return abs(d) > deviationWarnFraction
    }

    fun reset() {
        rows.clear()
    }

    /** ASCII block for the debug bundle. One line per populated key. */
    fun report(offlineRatios: Map<Int, Float> = emptyMap()): String {
        if (rows.isEmpty()) return "noise profile stats: no samples yet"
        val sb = StringBuilder()
        sb.append("noise profile stats (EMA double key: ISO bucket x S3 state)\n")
        for ((key, row) in rows) {
            val luma = row.meanSigmaLuma
            val ratioText = if (luma > 0f) String.format("%.3f", row.meanSigmaChroma / luma) else "n/a"
            sb.append("  iso<=")
                .append(key.isoBucket)
                .append(" s3=").append(if (key.s3Active) "on" else "off")
                .append(" n=").append(row.samples)
                .append(" lastIso=").append(row.lastIso)
                .append(" meanSigC=").append(String.format("%.4f", row.meanSigmaChroma))
                .append(" meanSigL=").append(String.format("%.4f", row.meanSigmaLuma))
                .append(" ratioC/L=").append(ratioText)
            val offline = offlineRatios[key.isoBucket]
            if (offline != null) {
                val d = deviationFromOffline(key.isoBucket, key.s3Active, offline)
                if (d != null) {
                    sb.append(" offline=").append(String.format("%.3f", offline))
                    sb.append(" dev=").append(String.format("%+.1f%%", d * 100f))
                    if (isDeviated(key.isoBucket, key.s3Active, offline)) sb.append(" DEVIATED")
                }
            }
            sb.append('\n')
        }
        return sb.toString()
    }

    companion object {
        private const val DEFAULT_ALPHA = 0.05f
        private const val DEFAULT_DEVIATION_WARN = 0.25f

        /**
         * ISO buckets double from 100 up. A bucket is a floor, not a
         * nearest-match, so two neighbouring ISOs inside one bucket share an
         * EMA - the point is to average over a population, not to track a
         * nominal ISO that the sensor never actually delivers.
         */
        fun isoBucketFor(iso: Int): Int {
            val i = iso.coerceAtLeast(1)
            var bucket = ISO_BUCKET_BASE
            // Floor, not ceiling: bucket(i) is the largest rung at or below i.
            // Rounding up would push every ISO just above a rung (101, 201,
            // 401...) into the next bucket, which halves how many real ISOs
            // land in each EMA population for no reason.
            while (bucket < ISO_BUCKET_MAX && i >= bucket * 2) bucket *= 2
            return min(bucket, ISO_BUCKET_MAX)
        }

        private const val ISO_BUCKET_BASE = 100
        private const val ISO_BUCKET_MAX = 102400

        /**
         * Chroma mean over luma mean for a bucket, from the committed anchor
         * table below.
         *
         * The anchors are transcribed from an offline noise measurement and are
         * the only place those numbers live; nothing outside this file is
         * consulted at runtime. Used only as the comparison baseline, since the
         * runtime value replaces it once enough samples have landed for a
         * bucket.
         */
        fun offlineRatioFor(isoBucket: Int): Float {
            val anchors = listOf(
                Pair(100, 41.0f),
                Pair(400, 47.0f),
                Pair(1600, 52.0f),
                Pair(6400, 58.0f),
                Pair(25600, 63.0f),
                Pair(102400, 66.0f)
            )
            val b = isoBucket.coerceAtLeast(ISO_BUCKET_BASE)
            if (b <= anchors.first().first) return anchors.first().second
            for (k in 1 until anchors.size) {
                val (x0, y0) = anchors[k - 1]
                val (x1, y1) = anchors[k]
                if (b <= x1) {
                    val t = if (x1 > x0) (ln(b.toDouble()) - ln(x0.toDouble())) / (ln(x1.toDouble()) - ln(x0.toDouble())) else 0.0
                    return y0 + (y1 - y0) * t.toFloat()
                }
            }
            return anchors.last().second
        }

        /** Convenience: the full offline table keyed by ISO bucket. */
        fun offlineRatioTable(): Map<Int, Float> {
            val out = LinkedHashMap<Int, Float>()
            var b = ISO_BUCKET_BASE
            while (b <= ISO_BUCKET_MAX) {
                out[b] = offlineRatioFor(b)
                b *= 2
            }
            return out
        }

        /** Clamp helper shared with the covariance scan's ISO walk. */
        fun clampIso(iso: Int): Int = max(ISO_BUCKET_BASE, min(iso, ISO_BUCKET_MAX))
    }
}