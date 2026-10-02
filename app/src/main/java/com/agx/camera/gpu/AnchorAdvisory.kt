package com.agx.camera.gpu

import kotlin.math.abs

/**
 * On-device anchor adjudication. ADVISORY ONLY.
 *
 * The anchor immunity claim is conditional on this probe:
 *
 *   If the pure-noise anchor scene classifies PLAIN, then gamma is identically
 *   1 on it and the main path stays bit-identical. If probe 5 fails, the
 *   immunity claim is withdrawn and anchor acceptance switches to the
 *   "classifier ON" baseline.
 *
 * That conditional is decided in two places, and both are needed. Probe 5 is a
 * fixed offline construction, so it is answered once by
 * [StructureMapClassifier.runProbes] and cannot be moved on device. What the
 * device can add is the other half: whether the live structure map actually
 * agrees with the probe on a flat anchor scene. A classifier that passes its
 * probe on paper and labels real flat-noise texels TEXTURE would break the
 * main-path identity claim silently, and nothing in the rendered image would
 * show it.
 *
 * Nothing here gates or changes the pipeline. It reads the diagnostic map and
 * reports; the rendering path never consults the result. A failed verdict is a
 * signal to investigate rather than a build failure.
 *
 * The per-round dose re-check (<=5% deviation against the measured
 * device reference) lives here too because it is the same kind of question: a
 * measured number against a fixed reference. The reference is a parameter, so a
 * missing reference reports as not comparable.
 */
object AnchorAdvisory {

    /** The anchor is acceptable while deviation stays within this. */
    const val DOSE_DEVIATION_LIMIT = 0.05f

    /**
     * Fraction of a flat anchor frame the classifier may still call structured.
     *
     * Deliberately looser than [NON_PLAIN_TOLERANCE]. Its job is to decide
     * whether the frame is eligible to be judged at all, so it has to sit above
     * the tolerance; otherwise every frame that trips the tolerance would be
     * thrown out as not-flat first and the live half could never withdraw
     * anything, which is the one verdict it exists to reach.
     */
    const val FLAT_SCENE_MAX_STRUCTURE = 0.10f

    /**
     * Fraction of sampled texels allowed to classify non-PLAIN on a flat anchor
     * scene. The map is thresholded on an SNR ratio, so a nonzero tail is
     * expected even on flat noise; the tolerance bounds the tail rather than
     * demanding an impossible exactly-zero.
     */
    const val NON_PLAIN_TOLERANCE = 0.02f

    /** Outcome classes. A boolean cannot carry "not measured" as its own value. */
    enum class Verdict {
        /** The anchor scene stayed PLAIN; the immunity claim is supported. */
        IMMUNITY_SUPPORTED,

        /** Non-PLAIN texels exceeded the tolerance; the claim is withdrawn. */
        IMMUNITY_WITHDRAWN,

        /** No sample yet. */
        INSUFFICIENT_DATA
    }

    data class AnchorSample(
        val iso: Int,
        val sampledTexels: Int,
        val nonPlainTexels: Int,
        val meanRatio: Float,
        val p99Ratio: Float,
        val maxRatio: Float,
        val textureTexels: Int = 0,
        val edgeTexels: Int = 0
    ) {
        fun nonPlainFraction(): Float =
            if (sampledTexels <= 0) 0f else nonPlainTexels.toFloat() / sampledTexels.toFloat()

        /**
         * Fraction of the sample the classifier called structured.
         *
         * Tracked per class because the two non-PLAIN classes are evidence of
         * different things, and only their union matters for the question this
         * object asks.
         */
        fun structuredFraction(): Float {
            if (sampledTexels <= 0) return 0f
            return (textureTexels + edgeTexels).toFloat() / sampledTexels.toFloat()
        }
    }

    data class DoseCheck(
        val stage: String,
        val measured: Float,
        val reference: Float?,
        val deviation: Float?,
        val withinLimit: Boolean?
    ) {
        fun verdictText(): String = when {
            reference == null -> "$stage measured=${f3(measured)} reference=absent not-comparable"
            deviation == null -> "$stage measured=${f3(measured)} reference=${f3(reference)} not-comparable"
            withinLimit == true -> "$stage measured=${f3(measured)} reference=${f3(reference)} dev=${signedPct(deviation)} within ${pct(DOSE_DEVIATION_LIMIT)}"
            else -> "$stage measured=${f3(measured)} reference=${f3(reference)} dev=${signedPct(deviation)} exceeds ${pct(DOSE_DEVIATION_LIMIT)}"
        }
    }

    data class Report(
        val probe5Holds: Boolean,
        val verdict: Verdict,
        val sample: AnchorSample?,
        val doseChecks: List<DoseCheck>,
        val sceneFlat: Boolean = true,
        val sceneRejection: String? = null
    ) {
        fun text(): String {
            val sb = StringBuilder()
            sb.append("anchor adjudication (advisory)\n")
            sb.append("  probe 5 (offline pure-noise must classify PLAIN): ")
                .append(if (probe5Holds) "pass" else "fail")
                .append('\n')
            val s = sample
            if (s == null) {
                sb.append("  live anchor sample: none - record one on a flat anchor scene\n")
            } else {
                val cal = StructureMapClassifier.calibrationFor(s.iso)
                sb.append("  live anchor sample iso=").append(s.iso)
                    .append(" n=").append(s.sampledTexels)
                    .append(" nonPLAIN=").append(s.nonPlainTexels)
                    .append(" (").append(String.format("%.2f%%", s.nonPlainFraction() * 100f)).append(")")
                    .append('\n')
                sb.append("    class census PLAIN=").append(s.sampledTexels - s.nonPlainTexels)
                    .append(" TEXTURE=").append(s.textureTexels)
                    .append(" EDGE=").append(s.edgeTexels)
                    .append('\n')
                sb.append("    ratio mean=").append(String.format("%.3f", s.meanRatio))
                    .append(" p99=").append(String.format("%.3f", s.p99Ratio))
                    .append(" max=").append(String.format("%.3f", s.maxRatio))
                    .append("  plain<=").append(String.format("%.2f", cal.plainMax))
                    .append(" texture>=").append(String.format("%.2f", cal.textureMin))
                    .append('\n')
                // Bit identity is a property of the anchor scene classifying
                // PLAIN, not of which classes happen to carry gamma=1. The
                // gamma values below are uncalibrated placeholders, so the
                // identity argument cannot rest on them either way.
                sb.append("    gamma by class would be ")
                    .append(String.format("%.2f", StructureMapClassifier.gammaFor(StructureMapClassifier.PLAIN)))
                    .append(" (PLAIN) / ")
                    .append(String.format("%.2f", StructureMapClassifier.gammaFor(StructureMapClassifier.TEXTURE)))
                    .append(" (TEXTURE) / ")
                    .append(String.format("%.2f", StructureMapClassifier.gammaFor(StructureMapClassifier.EDGE)))
                    .append(" (EDGE), uncalibrated placeholders; bit identity rests on ")
                    .append("probe 5 classifying the flat anchor scene PLAIN, and a ")
                    .append("non-PLAIN hit there would break it whatever gamma that class carries")
            }
            if (sceneRejection != null) {
                sb.append("  scene rejected as a flat anchor: ").append(sceneRejection).append('\n')
            }
            sb.append("  immunity verdict: ").append(verdict.name)
                .append(
                    when (verdict) {
                        Verdict.IMMUNITY_WITHDRAWN -> " - fall back to the classifier-ON baseline"
                        Verdict.INSUFFICIENT_DATA ->
                            " - not a judgement; probe 5 " +
                                (if (probe5Holds) "still holds" else "does not hold") +
                                ", the live scene was not eligible, so record one on a flat anchor scene"
                        Verdict.IMMUNITY_SUPPORTED -> " - the flat anchor scene stayed PLAIN"
                    }
                )
                .append('\n')
            if (doseChecks.isNotEmpty()) {
                sb.append("  per-round dose re-check (limit ").append(pct(DOSE_DEVIATION_LIMIT)).append("):\n")
                for (d in doseChecks) sb.append("    ").append(d.verdictText()).append('\n')
            }
            return sb.toString()
        }
    }

    /**
     * Offline half. Probe 5 is a fixed construction, so this is deterministic
     * and unit-testable; it does not depend on anything read on device.
     */
    fun probe5Holds(iso: Int = DEFAULT_PROBE_ISO): Boolean =
        StructureMapClassifier.runProbes(iso).firstOrNull { it.probe.startsWith("5") }?.passed == true

    /**
     * Live half. A null sample, a too-small sample, or a scene that is not
     * flat enough to serve as an anchor resolves to [Verdict.INSUFFICIENT_DATA]
     * rather than to a pass or a withdrawal.
     *
     * `sceneFlat` is the caller's judgement that the frame has enough flat
     * content to test. Without it any ordinary textured frame would register
     * non-PLAIN texels and withdraw a claim that was never in question.
     *
     * The three verdicts are kept strictly apart:
     *  - WITHDRAWN means the immunity claim is refuted. Either probe 5 failed,
     *    which is the one trigger the design names, or the live classifier
     *    disagreed with it on a scene that was eligible to be judged.
     *  - INSUFFICIENT_DATA means the claim was not tested at all: there is no
     *    sample, or the frame was not a flat anchor. A scene that is not flat
     *    says nothing about immunity, so it must never read as a withdrawal.
     *  - IMMUNITY_SUPPORTED means an eligible flat scene stayed PLAIN.
     */
    fun judge(
        sample: AnchorSample?,
        doseChecks: List<DoseCheck> = emptyList(),
        probeIso: Int = DEFAULT_PROBE_ISO,
        sceneFlat: Boolean = true
    ): Report {
        val probe = probe5Holds(probeIso)
        val eligible = sample != null && sample.sampledTexels >= MIN_TEXELS && sceneFlat
        val verdict = when {
            // The design's stated trigger. Checked before eligibility so a
            // failed probe still withdraws even with no live sample to pair.
            !probe -> Verdict.IMMUNITY_WITHDRAWN
            !eligible -> Verdict.INSUFFICIENT_DATA
            sample.nonPlainFraction() > NON_PLAIN_TOLERANCE -> Verdict.IMMUNITY_WITHDRAWN
            else -> Verdict.IMMUNITY_SUPPORTED
        }
        val rejection = if (sceneFlat || sample == null) {
            null
        } else {
            flatSceneRejection(sample, sample.iso)
        }
        return Report(probe, verdict, sample, doseChecks, sceneFlat, rejection)
    }

    /**
     * A frame qualifies as a flat anchor when it shows no structure to judge.
     *
     * Two independent conditions, and the second one is the one that matters:
     *
     *  - the 99th percentile ratio is still inside the PLAIN band, and
     *  - the classifier did not find real structure in it.
     *
     * The ratio tail alone cannot establish flatness, because EDGE is decided
     * from gradient coherence and not from the SNR ratio at all. A frame full
     * of edges therefore carries a ratio tail well inside the PLAIN band while
     * being visibly not a flat anchor. Judging such a frame on its non-PLAIN
     * count reads a scene that was never eligible as if the classifier had
     * disagreed with probe 5, and withdraws a claim that was never under test.
     * On the device frame that produced this, 240 of 1024 texels were
     * non-PLAIN, almost all of them EDGE, at a max ratio of 0.064.
     */
    fun isFlatAnchorScene(
        sample: AnchorSample,
        iso: Int
    ): Boolean {
        val cal = StructureMapClassifier.calibrationFor(iso)
        if (sample.p99Ratio > cal.plainMax) return false
        return sample.structuredFraction() <= FLAT_SCENE_MAX_STRUCTURE
    }

    /** Why [isFlatAnchorScene] said no, or null when it said yes. */
    fun flatSceneRejection(sample: AnchorSample, iso: Int): String? {
        val cal = StructureMapClassifier.calibrationFor(iso)
        if (sample.p99Ratio > cal.plainMax) {
            return "ratio tail ${f3(sample.p99Ratio)} clears plainMax=${f3(cal.plainMax)}"
        }
        val structured = sample.structuredFraction()
        if (structured > FLAT_SCENE_MAX_STRUCTURE) {
            return "classifier found structure in ${pct(structured)} of the frame " +
                "(TEXTURE=${sample.textureTexels} EDGE=${sample.edgeTexels}), " +
                "above the ${pct(FLAT_SCENE_MAX_STRUCTURE)} a flat anchor may contain"
        }
        return null
    }

    /**
     * Deviation of a measured attenuation against a reference, or null when the
     * reference is absent or non-positive. T1's ideal-lower-bound table is not a
     * valid reference here, so there is no default: the caller passes
     * the device value or gets null.
     */
    fun doseDeviation(measured: Float, reference: Float?): Float? {
        if (reference == null || !(reference > 0f)) return null
        if (!measured.isFinite()) return null
        return (measured - reference) / reference
    }

    fun doseCheck(stage: String, measured: Float, reference: Float?): DoseCheck {
        val d = doseDeviation(measured, reference)
        return DoseCheck(
            stage = stage,
            measured = measured,
            reference = reference,
            deviation = d,
            withinLimit = d?.let { abs(it) <= DOSE_DEVIATION_LIMIT }
        )
    }

    private const val DEFAULT_PROBE_ISO = 1600

    /** Below this the mean ratio is noise in the estimate, not a measurement. */
    private const val MIN_TEXELS = 64

    private fun f3(v: Float): String = String.format("%.4f", v)
    private fun pct(v: Float): String = String.format("%.0f%%", v * 100f)
    private fun signedPct(v: Float): String = String.format("%+.2f%%", v * 100f)
}