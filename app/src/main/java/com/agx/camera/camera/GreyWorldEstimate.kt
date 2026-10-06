package com.agx.camera.camera

import java.nio.ByteBuffer
import kotlin.math.sqrt

// Grey world white balance estimated per CFA phase.
//
// A Bayer mosaic carries two green sites, and they are two different filters
// over two different halves of the sensor, so under a neutral illuminant they do
// not have to answer with the same number. Reducing them to a single green
// before deriving the gains hides that difference: the demosaic then merges two
// differently scaled signals into one green channel, and nothing applied after
// the merge can tell how much of the result came from which site. Each phase is
// therefore normalized on its own, against the geometric mean of the two green
// sites, which is the reference that leaves the green pair balanced instead of
// tilting one site toward the other.
//
// The resulting four gains are split into the two places that have to be
// applied for that to hold:
//   preMergeGains - the green sites only, folded into each sample before the
//                   demosaic combines sites,
//   colorGains    - red and blue, referenced to green, applied as one multiply
//                   after the merge.
// Together they reconstruct phaseGains exactly, with every colour scaled once.
object GreyWorldEstimate {

    /** CFA phase count; also the length of every gains array here. */
    const val PHASE_COUNT = 4

    private const val RED = 0
    private const val BLUE = 2

    // A phase mean at or under this carries no colour information: a black
    // frame, a fully masked region, or a site read outside the sensor.
    private const val MIN_PHASE_MEAN = 1.0

    internal const val MIN_GAIN = 0.5f
    internal const val MAX_GAIN = 8f

    /** What one sparse pass over the cells found. */
    class CellScan(
        /** Black-subtracted mean per CFA phase, zero where a phase saw nothing. */
        val means: DoubleArray,
        val cellsUsed: Int,
        val cellsClipped: Int
    )

    /**
     * Black-subtracted mean of every CFA phase, walked one CFA cell at a time on a
     * fixed cell stride so the cost does not scale with the frame.
     *
     * The walk steps over 2x2 cells rather than over single sites, because a
     * clipped filter only means something relative to its neighbours: a saturated
     * green next to two healthy greens and a healthy red has still had the cell it
     * belongs to damaged, and averaging the survivors of that cell into the phase
     * means is what skews the reference. A cell with any filter at or above the
     * sensor's own white level is therefore dropped whole, all four phases of it.
     * That is also why the four phases of an accepted cell need no separate
     * bookkeeping: a 2x2 cell holds exactly one filter of each phase, so every
     * phase is averaged over the same cell set and the counts cannot diverge.
     *
     * Cells are anchored on even coordinates, so the four filters land on phases
     * 0, 1, 2, 3 in raster order under the same (x and 1) + (y and 1) * 2 mapping
     * the demosaic's reverse map uses. Which of those phases is red, green or
     * blue is then the colour map's business, not the walk's - one traversal is
     * correct for all four Bayer orders.
     *
     * [whiteLevel] is the sensor's reported saturation code, so "clipped" is the
     * ADC's own answer rather than a brightness guess. A non-positive value means
     * the frame came without one, and then nothing is rejected.
     */
    fun scan(
        buffer: ByteBuffer,
        width: Int,
        height: Int,
        blackLevels: IntArray,
        cellStride: Int,
        whiteLevel: Int
    ): CellScan {
        val step = cellStride.coerceAtLeast(1)
        val clip = if (whiteLevel > 0) whiteLevel else Int.MAX_VALUE
        val cellCols = width / 2
        val cellRows = height / 2
        val sums = DoubleArray(PHASE_COUNT)
        var used = 0
        var clipped = 0
        for (cy in 0 until cellRows step step) {
            val rowTop = (cy * 2) * width
            val rowBottom = rowTop + width
            for (cx in 0 until cellCols step step) {
                val col = cx * 2
                val top = (rowTop + col) * 2
                val bottom = (rowBottom + col) * 2
                val p0 = rawAt(buffer, top)
                val p1 = rawAt(buffer, top + 2)
                val p2 = rawAt(buffer, bottom)
                val p3 = rawAt(buffer, bottom + 2)
                if (p0 >= clip || p1 >= clip || p2 >= clip || p3 >= clip) {
                    clipped++
                    continue
                }
                sums[0] += p0
                sums[1] += p1
                sums[2] += p2
                sums[3] += p3
                used++
            }
        }
        val means = DoubleArray(PHASE_COUNT) { p ->
            if (used > 0) sums[p] / used - blackLevels.getOrElse(p) { 0 } else 0.0
        }
        return CellScan(means, used, clipped)
    }

    // RAW arrives little-endian: low byte first.
    private fun rawAt(buffer: ByteBuffer, index: Int): Int =
        (buffer.get(index).toInt() and 0xFF) or ((buffer.get(index + 1).toInt() and 0xFF) shl 8)

    // The two green sites in phase order, empty when the colour map does not
    // describe a normal Bayer mosaic.
    fun greenPhases(colorMap: IntArray): IntArray =
        (0 until PHASE_COUNT).filter { colorMap.getOrElse(it) { 1 } == 1 }.toIntArray()

    /** The red and blue phases in that order. */
    private fun redBluePhases(colorMap: IntArray): IntArray {
        var redPhase = 0
        var bluePhase = PHASE_COUNT - 1
        for (p in 0 until PHASE_COUNT) {
            when (colorMap.getOrElse(p) { 1 }) {
                RED -> redPhase = p
                BLUE -> bluePhase = p
            }
        }
        return intArrayOf(redPhase, bluePhase)
    }

    /**
     * The two stages of the estimate, kept apart on purpose.
     *
     * [daylight] is the fixed gain that removes the sensor's own response to a
     * neutral object under D65 - per lens, and not something the scene has any
     * say in. It is the whitening gain, not the response: the response is its
     * reciprocal, so [daylight] above 1 on a channel means the sensor answers
     * below green there.
     *
     * [illuminant] is what the light actually was, measured on the frame with
     * [daylight] already divided out. This is the reading worth logging: a D65
     * scene gives all-1s here, so what is left is the scene's departure from
     * daylight and nothing else.
     *
     * [phaseGains] is the product, which is the one array the renderer wants -
     * reference off, illuminant corrected, frame on level codes green-normalized,
     * which is the input the profile's daylight color matrix is built against.
     */
    class Estimate(
        /** Per-phase daylight whitening gain; all-1s when there was no profile to read. */
        val daylight: FloatArray,
        /** Per-phase departure from daylight; all-1s when the scene was D65. */
        val illuminant: FloatArray,
        /** [daylight] * [illuminant] per phase: what the renderer applies. */
        val phaseGains: FloatArray
    )

    /**
     * The full estimate, or null when a phase carries no usable signal.
     *
     * [daylightGains] are the fixed gain that removes the sensor's own response to a
     * neutral object under D65, as [RawColorMath] reads it off the color matrices
     * - green at 1, and on red and blue the reciprocal of what the CFA answers, so
     * a channel the sensor answers weakly gets a gain above 1. They are divided
     * out *before* the estimate, and that ordering is the whole point.
     *
     * A CFA response is green-biased by construction: the green filter passes far
     * more of any light than red or blue does, so the pre-white-balance signal for
     * a genuinely D65 scene is strongly chromatic green. Estimating on that raw
     * signal answers with the CFA's own bias rather than with the light - what it
     * really reports is the sensor's response to daylight, which every D65 scene
     * shares and which therefore swamps the part that varies. Taking the daylight
     * gains off first leaves the estimator looking only at the illuminant.
     *
     * The two green sites always share one target in the estimate: the geometric
     * mean of the two sites, which is what keeps the pair balanced for the demosaic
     * merge. The daylight gains carry green at 1, so that reconciliation is
     * untouched by the first stage.
     *
     * A [daylightGains] that is not three finite positive numbers leaves the first
     * stage out, which degrades to estimating straight off the raw signal.
     */
    fun estimate(
        phaseMeans: DoubleArray,
        colorMap: IntArray,
        daylightGains: FloatArray? = null
    ): Estimate? {
        val greens = greenPhases(colorMap)
        if (greens.size != 2) return null
        val anchor = daylightGains?.takeIf { d -> d.size >= 3 && d.all { it.isFinite() && it > 0f } }
        val redBlue = redBluePhases(colorMap)

        // Stage one: the fixed gain that takes the sensor's own daylight response off,
        // which is a constant and no part of the estimate.
        val daylight = FloatArray(PHASE_COUNT) { p ->
            if (anchor == null) {
                1f
            } else {
                when (p) {
                    redBlue[0] -> anchor[0]
                    redBlue[1] -> anchor[2]
                    else -> 1f
                }
            }
        }

        // Stage two: what the light did, measured on the corrected signal.
        val corrected = DoubleArray(PHASE_COUNT) { p -> phaseMeans[p] * daylight[p] }
        if (corrected.any { it <= MIN_PHASE_MEAN }) return null
        val level = sqrt(corrected[greens[0]] * corrected[greens[1]])
        val illuminant = FloatArray(PHASE_COUNT) { p ->
            (level / corrected[p]).toFloat().coerceIn(MIN_GAIN, MAX_GAIN)
        }
        val phaseGains = combine(illuminant, daylight)
        return Estimate(daylight, illuminant, phaseGains)
    }

    /**
     * The two stages multiplied per phase and clamped as one product.
     *
     * A caller that smooths [Estimate.illuminant] across frames and then rebuilds
     * the renderer array itself must go through here, not multiply directly: the
     * clamp belongs to the product, since a smoothed illuminant is already inside
     * [MIN_GAIN]/[MAX_GAIN] but its product with the daylight gain is not
     * necessarily. Clamping the two independently, or not at all, gives the
     * renderer a different array than [Estimate.phaseGains] describes.
     */
    fun combine(illuminant: FloatArray, daylight: FloatArray): FloatArray {
        require(illuminant.size == PHASE_COUNT && daylight.size == PHASE_COUNT) {
            "both stages are per phase, so both need $PHASE_COUNT entries"
        }
        return FloatArray(PHASE_COUNT) { p ->
            (illuminant[p] * daylight[p]).coerceIn(MIN_GAIN, MAX_GAIN)
        }
    }

    /**
     * [estimate] reduced to the one array the renderer applies, or null when a
     * phase carries no usable signal.
     */
    fun phaseGains(
        phaseMeans: DoubleArray,
        colorMap: IntArray,
        daylightGains: FloatArray? = null
    ): FloatArray? = estimate(phaseMeans, colorMap, daylightGains)?.phaseGains

    /**
     * The part of the solution that has to land before the demosaic merges
     * sites: the green sites carry their gain, red and blue stay at 1 because
     * their gain is the post-merge multiply in colorGains.
     *
     * A null [phaseGains] means the source only speaks red/green/blue - a sensor
     * profile, HAL state, Kelvin settings - so there is nothing per-phase to
     * split and the answer is all-1s. That is not a formality: the demosaic
     * multiplies green by u_wb_gains after the merge, so filling this slot from a
     * red/green/blue source would scale green twice. Passing null is how a caller
     * opts out of the split, and it is the only safe thing to do here.
     */
    fun preMergeGains(phaseGains: FloatArray?, colorMap: IntArray): FloatArray {
        if (phaseGains == null) return FloatArray(PHASE_COUNT) { 1f }
        val greens = greenPhases(colorMap).toSet()
        return FloatArray(PHASE_COUNT) { p ->
            if (p in greens) phaseGains[p] else 1f
        }
    }

    /** The same solution as red and blue gains referenced to green at 1.0. */
    fun colorGains(phaseGains: FloatArray, colorMap: IntArray): FloatArray {
        val redBlue = redBluePhases(colorMap)
        return floatArrayOf(phaseGains[redBlue[0]], 1f, phaseGains[redBlue[1]])
    }
}