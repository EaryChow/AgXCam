package com.agx.camera.gpu

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Shared structure map, first version.
 *
 * The classification criterion is an SNR-style ratio, not an absolute energy
 * threshold:
 *
 *   ratio(p) = residual energy in the window / ( sigma_hat^2(p) * r_retain(ISO) )
 *
 * The retention factor is in the denominator on purpose. SWGF attenuates
 * noise as part of its job, so a pure-noise patch lands at roughly
 * r_retain * sigma_hat^2 rather than sigma_hat^2; a bare sigma_hat^2
 * denominator would make every ratio read high and shift the TEXTURE threshold
 * by a hidden constant. The definition is fixed here so the calibration table
 * has something to match.
 *
 * Consequences, all intended:
 *  - a pure-noise patch classifies PLAIN, because its residual energy IS the
 *    noise prediction. That is what makes the anchor scenes immune.
 *  - real texture classifies TEXTURE, because its residual far exceeds the
 *    prediction and is spatially continuous.
 *  - EDGE comes from gradient direction coherence, judged separately.
 *
 * The thresholds are dimensionless ratios calibrated per ISO bucket. Zhang's
 * absolute TexE constants are not carried over.
 */
object StructureMapClassifier {

    const val PLAIN = 0
    const val TEXTURE = 1
    const val EDGE = 2
    const val NON_PLAIN = 1

    fun className(c: Int): String = when (c) {
        PLAIN -> "PLAIN"
        TEXTURE -> "TEXTURE"
        EDGE -> "EDGE"
        else -> "?"
    }

    // Zhang's luminance adaptation, four constants. The values are the ones
    // fitted for 8-bit SDR at a fixed viewing distance; AgX's toe and
    // shoulder are far more aggressive than sRGB's transfer, so these are
    // registered as pending recalibration rather than treated as settled.
    const val A_LUM_K1 = 2.0f
    const val A_LUM_LAM1 = 3.0f
    const val A_LUM_K2 = 0.8f
    const val A_LUM_LAM2 = 2.0f

    /** Contrast-masking exponent. */
    const val MASK_EPS = 0.36f

    // Per-ISO calibration: retention coefficient and the ratio thresholds.
    // Thresholds widen with ISO because the residual/noise ratio of genuine
    // texture falls as the noise floor rises.
    data class Calibration(
        val isoBucket: Int,
        val rRetain: Float,
        val plainMax: Float,
        val textureMin: Float,
        val edgeCoherenceMin: Float
    )

    /**
 * Rungs are deliberately sparser than the profile ladder.
 * [calibrationFor] buckets through NoiseProfileStats.isoBucketFor, which
 * doubles (100, 200, 400, 800, ...), so the buckets 200, 800, 3200, 12800 and
 * 51200 all floor to the rung below rather than getting their own row. The
 * thresholds move slowly enough across ISO for that to be harmless here:
 * plainMax only travels 1.60 to 2.00 over the whole table, so a rung covers
 * several profile buckets without a visible jump. Adding rows for every
 * profile bucket would imply per-bucket measurements that were never taken.
 */
    private val CALIBRATION = listOf(
        // isoBucket, rRetain, plainMax, textureMin, edgeCoherenceMin
        Calibration(100, 0.24f, 1.60f, 3.0f, 0.72f),
        Calibration(400, 0.26f, 1.60f, 3.0f, 0.72f),
        Calibration(1600, 0.28f, 1.70f, 3.2f, 0.72f),
        Calibration(6400, 0.30f, 1.80f, 3.4f, 0.72f),
        Calibration(25600, 0.32f, 1.90f, 3.6f, 0.70f),
        Calibration(102400, 0.34f, 2.00f, 3.8f, 0.70f)
    )

    fun calibrationFor(iso: Int): Calibration {
        val b = com.agx.camera.camera.NoiseProfileStats.isoBucketFor(iso)
        var chosen = CALIBRATION.first()
        for (c in CALIBRATION) {
            if (b >= c.isoBucket) chosen = c
        }
        return chosen
    }

    fun allCalibrations(): List<Calibration> = CALIBRATION

    /**
     * The SNR ratio. sigmaHat2 is the per-texel variance estimate, residual
     * is the measured residual energy over the same window.
     */
    fun ratio(residualEnergy: Float, sigmaHat2: Float, rRetain: Float): Float {
        if (sigmaHat2 <= 0f || rRetain <= 0f) return 0f
        if (!residualEnergy.isFinite()) return 0f
        return max(residualEnergy / (sigmaHat2 * rRetain), 0f)
    }

    /**
     * Luminance adaptation. C is formed-picture luminance in 0..1, GN the
     * mid pivot. Parabolic on both sides, minimum at the pivot.
     */
    fun aLum(formedLuminance: Float, midPivot: Float = 0.45f): Float {
        val c = formedLuminance.coerceIn(0f, 1f)
        val gn = midPivot.coerceIn(0.01f, 0.99f)
        return if (c <= gn * 0.5f) {
            1f + A_LUM_K1 * (1f - 2f * c / gn).coerceAtLeast(0f).let { powf(it, A_LUM_LAM1) }
        } else {
            1f + A_LUM_K2 * powf((2f * c / gn - 1f).coerceAtLeast(0f), A_LUM_LAM2)
        }
    }

    private fun powf(a: Float, b: Float): Float =
        if (a <= 0f) 0f else Math.pow(a.toDouble(), b.toDouble()).toFloat()

    /**
     * Base JND for the band, before structure elevation. Linear scale of the
     * threshold: higher where the eye is less sensitive.
     */
    fun jndBase(formedLuminance: Float, midPivot: Float = 0.45f): Float = aLum(formedLuminance, midPivot)

    /**
     * Structure elevation. The structural fact carried over from Zhang is that
     * an edge block's low- and mid-frequency content is excluded from the
     * masking gain, so the gain stays above 1 there; masking elsewhere takes
     * the max{1, (C/t_b)^eps} form.
     */
    fun elevation(contentEnergy: Float, baseJnd: Float, isEdge: Boolean): Float {
        if (baseJnd <= 0f) return 1f
        val masking = max(1f, powf(contentEnergy / baseJnd, MASK_EPS))
        return if (isEdge) max(masking, 1f) else masking
    }

    /**
     * Single-texel classification. `organization` is the spatial continuity
     * of the above-threshold neighbours in 0..1: pure noise scatters, texture
     * connects.
     */
    fun classify(
        iso: Int,
        ratio: Float,
        edgeCoherence: Float,
        organization: Float,
        previous: Int = PLAIN
    ): Int {
        val cal = calibrationFor(iso)
        if (edgeCoherence >= cal.edgeCoherenceMin) return EDGE
        if (ratio >= cal.textureMin && organization >= ORGANIZATION_MIN) return TEXTURE
        if (ratio <= cal.plainMax) return PLAIN
        // Between the two thresholds the classification depends on what it was
        // last frame; TEXTURE holds (it has to clear a higher bar to drop).
        return if (previous == TEXTURE && ratio > cal.plainMax * TEXTURE_HOLD_FACTOR) TEXTURE else PLAIN
    }

    /**
     * Two-class fallback. Used when the EDGE/TEXTURE separation does not hold
     * up on the probes: PLAIN and non-PLAIN, with the gradient map carrying
     * the edge information separately.
     */
    fun classifyTwoClass(iso: Int, ratio: Float, organization: Float, previous: Int = PLAIN): Int {
        val cal = calibrationFor(iso)
        if (ratio >= cal.textureMin && organization >= ORGANIZATION_MIN) return NON_PLAIN
        if (ratio <= cal.plainMax) return PLAIN
        return if (previous == NON_PLAIN && ratio > cal.plainMax * TEXTURE_HOLD_FACTOR) NON_PLAIN else PLAIN
    }

    /**
 * Win-scale gamma per class.
 *
 * EDGE sits above 1 to trade edge sharpness against over-smoothing. TEXTURE is
 * meant to sit below 1 so the pass lets texture through instead of smoothing
 * it, but there is no calibrated value for it yet, so it stays at 1 and that
 * let-through arm is currently inert. Only EDGE modulates the main path.
 *
 * All three values are placeholders pending a calibration run; the design
 * sources gamma from the shared JND map rather than hardcoding it.
 */
    fun gammaFor(cls: Int): Float = when (cls) {
        EDGE -> 1.15f
        TEXTURE -> 1f
        else -> 1f
    }

    /** Frame-to-frame slew limit for the class index, in classes per frame. */
    fun slewLimit(): Int = 1

    data class ProbeResult(
        val probe: String,
        val class_: Int,
        val expected: Int,
        val passed: Boolean,
        val note: String
    )

    /**
     * The five freeze-time probes. Probe 5 is the load-bearing one: pure noise
     * must land on PLAIN, because the anchor immunity claim rests on it.
     */
    fun runProbes(iso: Int = 1600): List<ProbeResult> {
        val out = ArrayList<ProbeResult>()

        // 1. flat noise patch - residual IS the noise prediction.
        val r1 = classify(iso, ratio(residualEnergy = 0.28f * SIG2, sigmaHat2 = SIG2, rRetain = calibrationFor(iso).rRetain),
            edgeCoherence = 0.08f, organization = 0.10f)
        out.add(ProbeResult("1 flat noise patch", r1, PLAIN, r1 == PLAIN, "residual at the retention anchor"))

        // 2. high-saturation fine texture.
        val r2 = classify(iso, ratio(residualEnergy = 9.0f * SIG2, sigmaHat2 = SIG2, rRetain = calibrationFor(iso).rRetain),
            edgeCoherence = 0.22f, organization = 0.85f)
        out.add(ProbeResult("2 saturated fine texture", r2, TEXTURE, r2 == TEXTURE, "residual far above prediction, connected"))

        // 3. chroma edge x luma texture overlap.
        val r3 = classify(iso, ratio(residualEnergy = 5.0f * SIG2, sigmaHat2 = SIG2, rRetain = calibrationFor(iso).rRetain),
            edgeCoherence = 0.80f, organization = 0.70f)
        out.add(ProbeResult("3 chroma edge x luma texture", r3, EDGE, r3 == EDGE, "gradient coherence decides over ratio"))

        // 4. pure chroma edge x flat luma - the case a luma-side criterion
        //    would miss entirely.
        val r4 = classify(iso, ratio(residualEnergy = 6.0f * SIG2, sigmaHat2 = SIG2, rRetain = calibrationFor(iso).rRetain),
            edgeCoherence = 0.86f, organization = 0.30f)
        out.add(ProbeResult("4 chroma edge x flat luma", r4, EDGE, r4 == EDGE, "coherence only; low luma-side ratio"))

        // 5. noise disguised as texture. A pure-noise patch with a locally
        //    elevated residual - still PLAIN, because the ratio carries the
        //    prediction with it.
        val r5 = classify(iso, ratio(residualEnergy = 1.9f * SIG2, sigmaHat2 = SIG2, rRetain = calibrationFor(iso).rRetain),
            edgeCoherence = 0.18f, organization = 0.20f)
        out.add(ProbeResult("5 noise disguised as texture", r5, PLAIN, r5 == PLAIN, "immune claim depends on this one"))

        return out
    }

    fun probesPass(iso: Int = 1600): Boolean = runProbes(iso).all { it.passed }

    /**
     * Whether the EDGE/TEXTURE separation held well enough to keep three
     * classes. The fallback trigger is mechanical: probe 4 landing on TEXTURE
     * means the luma-side ratio cannot see a chroma-only edge, which is the
     * separation's whole purpose.
     */
    fun needsTwoClassFallback(iso: Int = 1600): Boolean {
        val results = runProbes(iso)
        val p4 = results.firstOrNull { it.probe.startsWith("4") } ?: return false
        return p4.class_ != EDGE
    }

    /** ASCII block for the debug bundle. */
    fun report(iso: Int = 1600): String {
        val sb = StringBuilder()
        sb.append("structure map classifier (SNR ratio, dimensionless thresholds)\n")
        sb.append("  ratio = residual / (sigma^2 * r_retain), r_retain from the anchor table\n")
        for (c in CALIBRATION) {
            sb.append("  iso<=").append(c.isoBucket)
                .append(" rRetain=").append(String.format("%.2f", c.rRetain))
                .append(" plain<=").append(String.format("%.2f", c.plainMax))
                .append(" texture>=").append(String.format("%.2f", c.textureMin))
                .append(" edgeCoherence>=").append(String.format("%.2f", c.edgeCoherenceMin))
                .append('\n')
        }
        sb.append("  a_lum constants k1=").append(A_LUM_K1).append(" lam1=").append(A_LUM_LAM1)
            .append(" k2=").append(A_LUM_K2).append(" lam2=").append(A_LUM_LAM2)
            .append(" maskEps=").append(MASK_EPS).append('\n')
        sb.append("  a_lum recalibration: PENDING (constants were fitted for 8-bit SDR, not for this AgX curve)\n")
        sb.append("  probes (iso=").append(iso).append("):\n")
        for (p in runProbes(iso)) {
            sb.append("    ").append(if (p.passed) "PASS" else "FAIL")
                .append(' ').append(p.probe)
                .append(" -> ").append(className(p.class_))
                .append(" (expected ").append(className(p.expected)).append(") ")
                .append(p.note).append('\n')
        }
        val fallback = needsTwoClassFallback(iso)
        sb.append("  two-class fallback: ").append(if (fallback) "TRIGGERED" else "not triggered").append('\n')
        return sb.toString()
    }

    /** Residual energy implied by a per-texel ratio, for report round-trips. */
    fun residualFor(ratioTarget: Float, sigmaHat2: Float, iso: Int): Float =
        ratioTarget * sigmaHat2 * calibrationFor(iso).rRetain

    /** Sigma implied by a residual energy and ratio, for probe construction. */
    fun sigmaFor(residualEnergy: Float, ratioTarget: Float, iso: Int): Float =
        sqrt(residualEnergy / max(ratioTarget * calibrationFor(iso).rRetain, 1.0e-9f))

    /**
 * Minimum spatial continuity before a TEXTURE call is accepted. The GPU pass
 * reads this so both classifiers share one value.
 */
    const val ORGANIZATION_MIN = 0.45f

    /** TEXTURE needs to fall this far below the entry band before it drops. */
    private const val TEXTURE_HOLD_FACTOR = 0.75f

    private const val SIG2 = 100.0f
}

/**
 * Temporal stabilizer. The structure map feeds control signals, so a class
 * that flips every frame becomes the most visible flicker source in the
 * frame - worse than the noise it was meant to control. Two mechanisms:
 * asymmetric hysteresis in the classifier, and a per-frame slew limit here.
 */
class StructureMapStabilizer {

    /** Max class changes per frame. */
    var slewLimit: Int = 1

    private val lastClass = HashMap<Int, Int>()

    fun update(key: Int, candidate: Int): Int {
        val prev = lastClass[key] ?: StructureMapClassifier.PLAIN
        val steps = kotlin.math.abs(candidate - prev)
        if (steps <= slewLimit) {
            lastClass[key] = candidate
            return candidate
        }
        val limited = if (candidate > prev) prev + slewLimit else prev - slewLimit
        lastClass[key] = limited
        return limited
    }

    fun current(key: Int): Int = lastClass[key] ?: StructureMapClassifier.PLAIN

    fun reset() {
        lastClass.clear()
    }

    /** Gamma with the slew limit applied to the transition itself. */
    fun stabilizedGamma(key: Int, candidate: Int): Float = StructureMapClassifier.gammaFor(update(key, candidate))

    /** Fraction of tracked texels whose class changed on the last update pass. */
    fun report(): String {
        if (lastClass.isEmpty()) return "structure stabilizer: empty"
        val counts = HashMap<Int, Int>()
        for ((key, v) in lastClass) counts[v] = (counts[v] ?: 0) + 1
        val sb = StringBuilder("structure stabilizer: ")
        var first = true
        for (c in listOf(StructureMapClassifier.PLAIN, StructureMapClassifier.TEXTURE, StructureMapClassifier.EDGE)) {
            val n = counts[c] ?: continue
            if (!first) sb.append(' ')
            first = false
            sb.append(StructureMapClassifier.className(c)).append('=').append(n)
        }
        sb.append(" slewLimit=").append(slewLimit)
        return sb.toString()
    }

    /** True when the map's own churn would be a flicker source. */
    fun isStable(maxChangedFraction: Float = 0.05f, changed: Int, total: Int): Boolean {
        if (total <= 0) return true
        return changed.toFloat() / total.toFloat() <= maxChangedFraction
    }
}