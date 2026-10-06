package com.agx.camera.camera

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.sqrt

class GreyWorldEstimateTest {

    private val rggb = intArrayOf(0, 1, 1, 2)
    private val grbg = intArrayOf(1, 0, 2, 1)
    private val noBlack = intArrayOf(0, 0, 0, 0)
    private val size = 8

    // 8x8 is 4x4 cells, so a cell stride of 1 visits all sixteen and a stride of 2
    // visits every other one. Both are exercised below.
    private val white = 1023

    // Phase order for RGGB: red, green on odd columns, green on even columns,
    // blue. A neutral grey frame whose two green sites answer differently is
    // exactly the case the single-green estimate cannot see.
    private fun mosaic(red: Int, greenA: Int, greenB: Int, blue: Int): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(size * size * 2)
        for (y in 0 until size) {
            for (x in 0 until size) {
                val value = when (rggb[(x and 1) + (y and 1) * 2]) {
                    0 -> red
                    2 -> blue
                    else -> if ((x and 1) == 1) greenB else greenA
                }
                val i = (y * size + x) * 2
                buffer.put(i, (value and 0xFF).toByte())
                buffer.put(i + 1, ((value shr 8) and 0xFF).toByte())
            }
        }
        buffer.clear()
        return buffer
    }

    private fun scan(
        buffer: ByteBuffer,
        black: IntArray = noBlack,
        stride: Int = 1,
        whiteLevel: Int = white
    ) = GreyWorldEstimate.scan(buffer, size, size, black, stride, whiteLevel)

    private fun means(
        buffer: ByteBuffer,
        black: IntArray = noBlack,
        stride: Int = 1
    ) = scan(buffer, black, stride).means

    @Test
    fun eachPhaseReportsOnlyItsOwnSite() {
        val buffer = mosaic(red = 100, greenA = 200, greenB = 400, blue = 50)

        assertArrayEquals(doubleArrayOf(100.0, 400.0, 200.0, 50.0), means(buffer), 1e-9)
    }

    @Test
    fun aStrideVisitsWholeCellsOnTheSamePhase() {
        val buffer = mosaic(red = 100, greenA = 200, greenB = 400, blue = 50)

        // Cells are anchored on even coordinates, so a walk that steps by whole
        // cells cannot drift onto a neighbouring phase the way an odd pixel step
        // would. The phase means are stride-independent for that reason.
        assertArrayEquals(means(buffer, stride = 1), means(buffer, stride = 2), 1e-9)
        assertArrayEquals(means(buffer, stride = 1), means(buffer, stride = 3), 1e-9)
        assertArrayEquals(means(buffer, stride = 1), means(buffer, stride = 4), 1e-9)
    }

    @Test
    fun eachPhaseSubtractsItsOwnBlackLevel() {
        val buffer = mosaic(red = 100, greenA = 200, greenB = 400, blue = 50)

        assertArrayEquals(
            doubleArrayOf(36.0, 304.0, 72.0, 18.0),
            means(buffer, black = intArrayOf(64, 96, 128, 32)),
            1e-9
        )
    }

    @Test
    fun bothGreensLandOnTheSameNumber() {
        // The two green sites differ 4x on the sensor. Under a neutral illuminant
        // they are the same light through the same glass, so both have to answer
        // the same number afterwards. That is what stops the demosaic merging two
        // differently scaled signals into a single green channel.
        val buffer = mosaic(red = 100, greenA = 200, greenB = 800, blue = 50)
        val phaseMeans = means(buffer)
        val gains = GreyWorldEstimate.phaseGains(phaseMeans, rggb)!!

        val reference = sqrt(phaseMeans[1] * phaseMeans[2])
        assertClose("green A", reference.toFloat(), (phaseMeans[1] * gains[1]).toFloat())
        assertClose("green B", reference.toFloat(), (phaseMeans[2] * gains[2]).toFloat())
        assertClose("red", reference.toFloat(), (phaseMeans[0] * gains[0]).toFloat())
        assertClose("blue", reference.toFloat(), (phaseMeans[3] * gains[3]).toFloat())

        // The split is real, not one averaged green dressed up as a pair.
        assertNotEquals(gains[1], gains[2])
    }

    @Test
    fun aNeutralFrameAsksForNoChange() {
        val buffer = mosaic(red = 640, greenA = 640, greenB = 640, blue = 640)
        val gains = GreyWorldEstimate.phaseGains(means(buffer), rggb)!!

        gains.forEachIndexed { p, gain -> assertClose("phase $p", 1f, gain, 1e-3f) }
    }

    // A sensor never answers with equal codes on its three filters, so a frame of
    // equal numbers is not what a neutral scene looks like on the CFA. This is the
    // sensor's daylight response with green at 1, and the gains that take it to
    // level codes - the pair the profile hands over for a D65 reference.
    private val daylightResponse = floatArrayOf(0.45f, 1f, 0.75f)
    private val daylightGains = floatArrayOf(
        1f / daylightResponse[0], 1f, 1f / daylightResponse[2]
    )

    @Test
    fun aDaylightNeutralFrameAsksForNoChange() {
        // This frame is what the sensor reports for a neutral object under D65: the
        // CFA's green bias, nothing else. With that divided out first, the light is
        // left holding nothing - the illuminant stage is all-1s, which is the whole
        // claim: the reading is a measurement of the light, and there was no light
        // to measure beyond the reference.
        val buffer = mosaic(red = 450, greenA = 1000, greenB = 1000, blue = 750)
        val estimate = GreyWorldEstimate.estimate(means(buffer), rggb, daylightGains)!!

        estimate.illuminant.forEachIndexed { p, gain ->
            assertClose("illuminant $p", 1f, gain, 1e-3f)
        }

        // The product the renderer applies is the daylight gains, because that is
        // still what takes a D65 frame to level codes. Only the illuminant stage is
        // the measurement.
        assertClose("red", daylightGains[0], estimate.phaseGains[0], 1e-3f)
        assertClose("blue", daylightGains[2], estimate.phaseGains[3], 1e-3f)
    }

    @Test
    fun withoutTheDaylightReferenceTheEstimateIsDominatedByTheCfa() {
        // The same D65 frame read without the reference comes back strongly
        // red-deficient and blue-deficient. Nothing in the scene is off-neutral -
        // the estimate is reporting the CFA's own response to the light, which is
        // a constant every D65 scene shares. That constant is what swamps the part
        // that actually varies, and it is why the reference has to come off first.
        val buffer = mosaic(red = 450, greenA = 1000, greenB = 1000, blue = 750)
        val unanchored = GreyWorldEstimate.estimate(means(buffer), rggb)!!

        // The magnitude is the failure. On a frame lit by exactly the daylight the
        // CFA is anchored to, the unanchored reading is still more than 2x on red -
        // it reports a strong red cast where there is none at all. That bias is
        // baked into the green filter, identical on every daylight scene the sensor
        // ever sees, and it lands in the same number the light itself moves. No
        // threshold on the reading can separate the two.
        assertTrue("unanchored red ${unanchored.illuminant[0]}", unanchored.illuminant[0] > 2f)

        // The same frame with the daylight reference taken off first: all-1s,
        // because the light was exactly what the reference describes.
        val anchored = GreyWorldEstimate.estimate(means(buffer), rggb, daylightGains)!!
        anchored.illuminant.forEachIndexed { p, gain ->
            assertClose("anchored illuminant $p", 1f, gain, 1e-3f)
        }

        // And a genuinely warm scene still comes back warm, so the reference is
        // not simply flattening the estimate.
        val tungsten = GreyWorldEstimate.estimate(
            means(mosaic(red = 700, greenA = 1000, greenB = 1000, blue = 600)),
            rggb,
            daylightGains
        )!!
        assertTrue("tungsten red ${tungsten.illuminant[0]}", tungsten.illuminant[0] < 1f)
        assertTrue("tungsten blue ${tungsten.illuminant[3]}", tungsten.illuminant[3] > 1f)
    }

    @Test
    fun theDaylightAnchorLeavesTheGreenSitesAlone() {
        val buffer = mosaic(red = 300, greenA = 200, greenB = 800, blue = 400)
        val phaseMeans = means(buffer)
        val plain = GreyWorldEstimate.estimate(phaseMeans, rggb)!!
        val anchored = GreyWorldEstimate.estimate(phaseMeans, rggb, daylightGains)!!

        // The daylight reference carries green at 1, so the green pair's target is
        // unchanged: the geometric mean of the two sites either way, which is what
        // keeps the demosaic merge balanced.
        assertClose("green A", plain.illuminant[1], anchored.illuminant[1])
        assertClose("green B", plain.illuminant[2], anchored.illuminant[2])
        // Still two distinct answers, not one green averaged twice.
        assertNotEquals(anchored.illuminant[1], anchored.illuminant[2])
        // Red and blue are the ones that moved.
        assertNotEquals("red", plain.illuminant[0], anchored.illuminant[0])
        assertNotEquals("blue", plain.illuminant[3], anchored.illuminant[3])
        // And the reference itself is passed through untouched on the greens.
        assertClose("daylight green A", 1f, anchored.daylight[1])
        assertClose("daylight green B", 1f, anchored.daylight[2])
    }

    @Test
    fun theDaylightAnchorLeavesAWarmSceneAlone() {
        // Tungsten: more red and less blue than daylight. The anchor still reads it
        // as off-neutral and still corrects it - red down, blue up - but it
        // corrects by less, because daylight itself already answers below green on
        // red and blue. Referencing green alone asks for the difference between the
        // scene and a flat response, which is a bigger correction than the
        // difference between the scene and what the sensor does under daylight.
        val buffer = mosaic(red = 700, greenA = 1000, greenB = 1000, blue = 600)
        val phaseMeans = means(buffer)
        val plain = GreyWorldEstimate.estimate(phaseMeans, rggb)!!
        val anchored = GreyWorldEstimate.estimate(phaseMeans, rggb, daylightGains)!!

        // Red is over-represented under tungsten, so it comes back; blue is
        // under-represented, so it goes up.
        assertTrue("red ${anchored.illuminant[0]} should fall below 1", anchored.illuminant[0] < 1f)
        assertTrue("blue ${anchored.illuminant[3]} should rise above 1", anchored.illuminant[3] > 1f)
        // Both moves are smaller than the flat-reference version's, because daylight
        // itself already answers below green on red and blue.
        assertTrue(
            "red ${anchored.illuminant[0]} vs ${plain.illuminant[0]}",
            anchored.illuminant[0] < plain.illuminant[0]
        )
        assertTrue(
            "blue ${anchored.illuminant[3]} vs ${plain.illuminant[3]}",
            anchored.illuminant[3] < plain.illuminant[3]
        )
    }

    @Test
    fun theDaylightAnchorFollowsTheBayerPattern() {
        // GRBG puts its greens on phases 0 and 3, so the same daylight frame has to
        // be written the same way or the anchor lands on the wrong filters.
        val phaseMeans = doubleArrayOf(1000.0, 450.0, 750.0, 1000.0)
        val estimate = GreyWorldEstimate.estimate(phaseMeans, grbg, daylightGains)!!
        val reference = sqrt(phaseMeans[0] * phaseMeans[3])

        // Corrected means land on one level green-normalized whichever phase holds
        // which filter. GRBG puts green on 0 and 3, so the reference is their
        // geometric mean rather than phases 1 and 2.
        assertClose("red", reference.toFloat(), (phaseMeans[1] * estimate.phaseGains[1]).toFloat(), 1e-2f)
        assertClose("blue", reference.toFloat(), (phaseMeans[2] * estimate.phaseGains[2]).toFloat(), 1e-2f)
        assertClose("green A", reference.toFloat(), (phaseMeans[0] * estimate.phaseGains[0]).toFloat(), 1e-2f)
        assertClose("green B", reference.toFloat(), (phaseMeans[3] * estimate.phaseGains[3]).toFloat(), 1e-2f)

        // The reference landed on the filters that are not green, so this is the
        // case where a mix-up would put the CFA's bias on the wrong pair.
        assertTrue("red ${estimate.phaseGains[1]}", estimate.phaseGains[1] > 1f)
        assertTrue("blue ${estimate.phaseGains[2]}", estimate.phaseGains[2] > 1f)
        estimate.illuminant.forEachIndexed { p, gain ->
            assertClose("illuminant $p", 1f, gain, 1e-3f)
        }
    }

    @Test
    fun theAnchoredEstimateLandsOnLevelCodes() {
        // Whatever the light was, the corrected frame has to come out level
        // green-normalized. That is the invariant the profile's colour matrix is
        // built against, and it is what makes the returned array safe to hand
        // straight to the renderer as one product of both stages.
        // Every mean stays under the scan's white level: a cell holding a clipped
        // filter is dropped whole, so one value at 1023 would empty the frame and
        // leave the estimate nothing to work from.
        val frames = listOf(
            "daylight" to mosaic(red = 450, greenA = 1000, greenB = 1000, blue = 750),
            "tungsten" to mosaic(red = 700, greenA = 1000, greenB = 1000, blue = 600),
            "overcast" to mosaic(red = 400, greenA = 1000, greenB = 1000, blue = 800),
            "split greens" to mosaic(red = 450, greenA = 600, greenB = 1000, blue = 750)
        )
        for ((label, buffer) in frames) {
            val phaseMeans = means(buffer)
            val gains = GreyWorldEstimate.estimate(phaseMeans, rggb, daylightGains)!!.phaseGains
            val preMerge = GreyWorldEstimate.preMergeGains(gains, rggb)
            val colors = GreyWorldEstimate.colorGains(gains, rggb)
            val balanced = floatArrayOf(
                (phaseMeans[0] * preMerge[0] * colors[0]).toFloat(),
                (phaseMeans[1] * preMerge[1] * colors[1]).toFloat(),
                (phaseMeans[3] * preMerge[3] * colors[2]).toFloat()
            )
            assertClose("$label red", balanced[1], balanced[0], 1e-2f)
            assertClose("$label blue", balanced[1], balanced[2], 1e-2f)
        }
    }

    @Test
    fun theFloorIsCheckedAfterTheDaylightReferenceComesOff() {
        // MIN_PHASE_MEAN is a floor on usable signal, not on the frame being
        // neutral, and it applies to what the estimator is about to divide by. A
        // frame the sensor barely answers at all is refused either way, because a
        // gain built on a near-black mean is a guess.
        val dim = mosaic(red = 0, greenA = 1, greenB = 1, blue = 0)
        assertNull(GreyWorldEstimate.estimate(means(dim), rggb, daylightGains))
        assertNull(GreyWorldEstimate.estimate(means(dim), rggb))

        // The floor guards the corrected signal, so it sees the product rather than
        // the raw mean. Red and blue at 0.5 are under the floor on their own but
        // come out above it once the reference lifts them, and the estimate is
        // legitimate either way - there is real signal there to work from.
        val faint = mosaic(red = 1, greenA = 1000, greenB = 1000, blue = 1)
        assertNull(GreyWorldEstimate.estimate(means(faint), rggb))
        assertNotNull(GreyWorldEstimate.estimate(means(faint), rggb, daylightGains))
    }

    @Test
    fun aMalformedDaylightAnchorFallsBackToGreen() {
        // A broken profile must not take the estimate with it. Ignoring the anchor
        // and falling back to the green-referenced answer is wrong by a fixed factor
        // and visible; returning null would stall the gains instead.
        val buffer = mosaic(red = 300, greenA = 200, greenB = 800, blue = 400)
        val phaseMeans = means(buffer)
        val plain = GreyWorldEstimate.estimate(phaseMeans, rggb)!!

        val malformed = listOf(
            null,
            floatArrayOf(0f, 1f, 1f),
            floatArrayOf(1f, 0f, 1f),
            floatArrayOf(1f, 1f, Float.NaN),
            floatArrayOf(Float.POSITIVE_INFINITY, 1f, 1f),
            floatArrayOf(-1f, 1f, 1f),
            floatArrayOf(1f, 1f)
        )
        for (anchor in malformed) {
            val estimate = GreyWorldEstimate.estimate(phaseMeans, rggb, anchor)
            assertNotNull("anchor ${anchor?.toList()}", estimate)
            assertArrayEquals("daylight ${anchor?.toList()}", plain.daylight, estimate!!.daylight, 1e-6f)
            assertArrayEquals("illuminant ${anchor?.toList()}", plain.illuminant, estimate.illuminant, 1e-6f)
            assertArrayEquals("gains ${anchor?.toList()}", plain.phaseGains, estimate.phaseGains, 1e-6f)
        }

        // A longer array is not malformed, it is just carrying more than the three
        // filters use. Reading the first three matches normalize() and the rest of
        // the profile math, which all accept anything from three up.
        val longer = GreyWorldEstimate.estimate(
            phaseMeans, rggb, daylightGains + floatArrayOf(9f)
        )!!
        val expected = GreyWorldEstimate.estimate(phaseMeans, rggb, daylightGains)!!
        assertArrayEquals(expected.daylight, longer.daylight, 1e-6f)
        assertArrayEquals(expected.illuminant, longer.illuminant, 1e-6f)
        assertArrayEquals(expected.phaseGains, longer.phaseGains, 1e-6f)
    }

    @Test
    fun aSmoothedProductIsClampedAsOneNumber() {
        // The app smooths the illuminant across frames and rebuilds the renderer
        // array itself, so the clamp has to belong to the product: a smoothed
        // illuminant is already inside the bounds, but multiplied by the daylight
        // gain it need not be. Both sides have to come out of the same definition
        // or the renderer holds an array estimate() would not describe.
        val estimate = GreyWorldEstimate.estimate(
            means(mosaic(red = 1, greenA = 1000, greenB = 1000, blue = 1)), rggb, daylightGains
        )!!
        assertArrayEquals(
            estimate.phaseGains,
            GreyWorldEstimate.combine(estimate.illuminant, estimate.daylight),
            0f
        )

        // A pair that multiplies out past the bound is clamped, not passed through.
        val pushy = floatArrayOf(4f, 1f, 4f, 1f)
        assertArrayEquals(
            floatArrayOf(8f, 1f, 8f, 1f),
            GreyWorldEstimate.combine(pushy, pushy),
            0f
        )
        val faint = floatArrayOf(0.4f, 1f, 0.4f, 1f)
        assertArrayEquals(
            floatArrayOf(0.5f, 1f, 0.5f, 1f),
            GreyWorldEstimate.combine(faint, faint),
            0f
        )
    }

    @Test
    fun theSplitStillReconstructsTheAnchoredSolution() {
        val buffer = mosaic(red = 300, greenA = 200, greenB = 800, blue = 400)
        val phaseGains = GreyWorldEstimate.phaseGains(means(buffer), rggb, daylightGains)!!
        val preMerge = GreyWorldEstimate.preMergeGains(phaseGains, rggb)
        val colors = GreyWorldEstimate.colorGains(phaseGains, rggb)

        assertArrayEquals(floatArrayOf(1f, phaseGains[1], phaseGains[2], 1f), preMerge, 1e-6f)
        assertArrayEquals(floatArrayOf(phaseGains[0], 1f, phaseGains[3]), colors, 1e-6f)

        // The anchor changes the red and blue targets, not the split, so the two
        // halves still reconstruct the per-phase solution exactly.
        for (p in 0 until GreyWorldEstimate.PHASE_COUNT) {
            val combined = preMerge[p] * when (rggb[p]) {
                0 -> colors[0]
                2 -> colors[2]
                else -> colors[1]
            }
            assertEquals("phase $p", phaseGains[p], combined, 1e-5f)
        }
    }

    @Test
    fun withoutAPerPhaseSolutionGreenIsNotScaledTwice() {
        // Sensor profile, HAL state and Kelvin only speak red/green/blue. Green
        // reaches the shader through the post-merge u_wb_gains multiply, so a
        // pre-merge green gain on top of that would square it. Passing null has to
        // produce an all-1s split regardless of what green the caller holds, which
        // is the case that used to be left to a convention nobody enforced.
        val preMerge = GreyWorldEstimate.preMergeGains(null, rggb)

        assertArrayEquals(floatArrayOf(1f, 1f, 1f, 1f), preMerge, 0f)
        // Same under GRBG, where the greens are phases 0 and 3.
        assertArrayEquals(
            floatArrayOf(1f, 1f, 1f, 1f),
            GreyWorldEstimate.preMergeGains(null, grbg),
            0f
        )
    }

    @Test
    fun aNullSplitWouldDoubleGreenOnlyIfItBorrowedTheRgbGains() {
        // The failure this guards against, stated as arithmetic: feeding the
        // red/green/blue gains through the per-phase split scales green by
        // wbGainG in the shader twice, so the split has to come back neutral
        // instead of tracking them.
        val colorGains = floatArrayOf(1.8f, 1.4f, 2.2f)
        val asPhases = FloatArray(GreyWorldEstimate.PHASE_COUNT) { p ->
            when (rggb[p]) {
                0 -> colorGains[0]
                2 -> colorGains[2]
                else -> colorGains[1]
            }
        }

        val borrowed = GreyWorldEstimate.preMergeGains(asPhases, rggb)
        val greens = GreyWorldEstimate.greenPhases(rggb)
        for (g in greens) {
            assertEquals(
                "green $g scaled twice",
                colorGains[1] * colorGains[1],
                borrowed[g] * colorGains[1],
                1e-5f
            )
        }

        // And the null path is what actually ships, so green lands on the single
        // post-merge multiply.
        val shipped = GreyWorldEstimate.preMergeGains(null, rggb)
        for (g in greens) {
            assertEquals(colorGains[1], shipped[g] * colorGains[1], 1e-5f)
        }
    }

    @Test
    fun gainsAreSplitIntoPreMergeGreensAndPostMergeColours() {
        val buffer = mosaic(red = 100, greenA = 200, greenB = 800, blue = 50)
        val phaseGains = GreyWorldEstimate.phaseGains(means(buffer), rggb)!!
        val preMerge = GreyWorldEstimate.preMergeGains(phaseGains, rggb)
        val colors = GreyWorldEstimate.colorGains(phaseGains, rggb)

        // Only the green sites have to land before the merge; red and blue are
        // applied once, after it.
        assertArrayEquals(floatArrayOf(1f, phaseGains[1], phaseGains[2], 1f), preMerge, 1e-6f)
        assertArrayEquals(floatArrayOf(phaseGains[0], 1f, phaseGains[3]), colors, 1e-6f)

        // The two halves together reconstruct the per-phase solution exactly, so
        // no colour is applied twice and none is dropped.
        for (p in 0 until GreyWorldEstimate.PHASE_COUNT) {
            val combined = preMerge[p] * when (rggb[p]) {
                0 -> colors[0]
                2 -> colors[2]
                else -> colors[1]
            }
            assertEquals("phase $p", phaseGains[p], combined, 1e-5f)
        }
    }

    @Test
    fun theGreenSplitFollowsTheBayerPattern() {
        // GRBG puts its greens on phases 0 and 3, not 1 and 2.
        assertArrayEquals(intArrayOf(0, 3), GreyWorldEstimate.greenPhases(grbg))
        assertArrayEquals(intArrayOf(1, 2), GreyWorldEstimate.greenPhases(rggb))

        val buffer = mosaic(red = 100, greenA = 200, greenB = 800, blue = 50)
        val phaseMeans = means(buffer)
        val phaseGains = GreyWorldEstimate.phaseGains(phaseMeans, grbg)!!
        val preMerge = GreyWorldEstimate.preMergeGains(phaseGains, grbg)

        // Greens carry their gain before the merge; red and blue wait until after it.
        assertClose("green A", phaseGains[0], preMerge[0])
        assertClose("red", 1f, preMerge[1])
        assertClose("blue", 1f, preMerge[2])
        assertClose("green B", phaseGains[3], preMerge[3])

        // Both greens still reach the same number under this pattern too.
        val reference = sqrt(phaseMeans[0] * phaseMeans[3])
        assertClose(
            "green A",
            reference.toFloat(),
            (phaseMeans[0] * preMerge[0]).toFloat()
        )
        assertClose(
            "green B",
            reference.toFloat(),
            (phaseMeans[3] * preMerge[3]).toFloat()
        )
    }

    @Test
    fun aBlackPhaseYieldsNoEstimate() {
        val buffer = mosaic(red = 100, greenA = 200, greenB = 800, blue = 0)

        assertNull(GreyWorldEstimate.phaseGains(means(buffer), rggb))
    }

    @Test
    fun gainsStayInsideTheUsableRange() {
        // Blue reads 2 against greens at 1000: the true gain is 500, which would
        // push every blue channel past full sensor capacity, so it is held at the
        // ceiling.
        val buffer = mosaic(red = 1000, greenA = 1000, greenB = 1000, blue = 2)
        val gains = GreyWorldEstimate.phaseGains(means(buffer), rggb)!!

        gains.forEach { gain ->
            assertTrue("gain $gain out of range", gain in 0.5f..8f)
        }
        assertClose("clamped blue", 8f, gains[3])
    }

    // --- Clipped-region exclusion ---

    // Overwrite one filter at (x, y) with a saturated code.
    private fun ByteBuffer.saturate(x: Int, y: Int) = write(x, y, white)

    private fun ByteBuffer.write(x: Int, y: Int, value: Int) {
        val i = (y * size + x) * 2
        put(i, (value and 0xFF).toByte())
        put(i + 1, ((value shr 8) and 0xFF).toByte())
    }

    @Test
    fun aFullyClippedFrameLeavesNothingToAverage() {
        val buffer = mosaic(red = 100, greenA = 200, greenB = 800, blue = 50)
        for (y in 0 until size) {
            for (x in 0 until size) buffer.saturate(x, y)
        }

        val result = scan(buffer)

        assertEquals(0, result.cellsUsed)
        assertEquals(16, result.cellsClipped)
        assertArrayEquals(doubleArrayOf(0.0, 0.0, 0.0, 0.0), result.means, 1e-9)
        assertNull(GreyWorldEstimate.phaseGains(result.means, rggb))
    }

    @Test
    fun oneClippedFilterTakesItsWholeCellWithIt() {
        // A neutral frame, then a single filter driven to the white level. That
        // filter's three healthy neighbours sit in the same cell, and a cell that
        // kept them would still answer with a mean the sensor never measured.
        val buffer = mosaic(red = 640, greenA = 640, greenB = 640, blue = 640)
        buffer.saturate(2, 2)

        val result = scan(buffer)

        assertEquals(1, result.cellsClipped)
        assertEquals(15, result.cellsUsed)
        // Every surviving cell reads the same neutral value, so the surviving
        // phases still ask for no change.
        val gains = GreyWorldEstimate.phaseGains(result.means, rggb)!!
        gains.forEach { gain -> assertClose("surviving phase", 1f, gain, 1e-3f) }
    }

    @Test
    fun aClippedGreenDoesNotSkewTheGreenReference() {
        // Greens at 800, red and blue at 100: the estimator would otherwise pull
        // the green reference up and ask for a red gain far above 1.
        val buffer = mosaic(red = 100, greenA = 800, greenB = 800, blue = 100)
        val clean = GreyWorldEstimate.phaseGains(means(buffer), rggb)!!

        // Saturate one green of each site, in two different cells.
        buffer.saturate(1, 0)
        buffer.saturate(2, 1)
        val filtered = GreyWorldEstimate.phaseGains(means(buffer).also {
            assertEquals(2, scan(buffer).cellsClipped)
        }, rggb)!!

        for (p in 0 until GreyWorldEstimate.PHASE_COUNT) {
            assertEquals("phase $p", clean[p], filtered[p], 2e-3f)
        }
    }

    @Test
    fun aClippedRegionLeavesTheRestOfTheFrameIntact() {
        // Everything left of the midline is neutral; the right half is driven to
        // clip, as a blown window or an overcast sky edge would be. The estimate
        // has to come from the half that still carries signal.
        val buffer = ByteBuffer.allocateDirect(size * size * 2)
        for (y in 0 until size) {
            for (x in 0 until size) {
                val value = if (x < size / 2) {
                    when (rggb[(x and 1) + (y and 1) * 2]) {
                        0 -> 100
                        2 -> 800
                        else -> 400
                    }
                } else {
                    white
                }
                val i = (y * size + x) * 2
                buffer.put(i, (value and 0xFF).toByte())
                buffer.put(i + 1, ((value shr 8) and 0xFF).toByte())
            }
        }
        buffer.clear()

        val result = scan(buffer, stride = 1)

        // Two of the four cell columns are clipped, all four rows.
        assertEquals(8, result.cellsUsed)
        assertEquals(8, result.cellsClipped)
        // Exactly the values the unclipped half was written with.
        assertArrayEquals(doubleArrayOf(100.0, 400.0, 400.0, 800.0), result.means, 1e-9)
    }

    @Test
    fun theWhiteLevelIsTheSensorsOwnNotAGuess() {
        // Everything well under both candidate levels, so only the two filters
        // written afterwards can tip a cell over.
        val buffer = mosaic(red = 100, greenA = 200, greenB = 200, blue = 50)
        buffer.write(4, 0, 900)
        buffer.write(2, 1, 900)

        // At the sensor's real 10-bit level nothing is clipped.
        assertEquals(0, scan(buffer, whiteLevel = white).cellsClipped)
        // At a 9-bit level the two 900s clip, and each takes its own cell with it.
        assertEquals(2, scan(buffer, whiteLevel = 511).cellsClipped)
    }

    @Test
    fun anUnknownWhiteLevelRejectsNothing() {
        val buffer = mosaic(red = 100, greenA = 200, greenB = 800, blue = 50)
        for (y in 0 until size) {
            for (x in 0 until size) buffer.saturate(x, y)
        }

        val result = scan(buffer, whiteLevel = 0)

        // Nothing can be judged clipped without a level, so the walk still
        // produces an estimate rather than going dark.
        assertEquals(0, result.cellsClipped)
        assertEquals(16, result.cellsUsed)
    }

    @Test
    fun theStridedWalkStillFindsASmallClippedPatch() {
        val buffer = mosaic(red = 640, greenA = 640, greenB = 640, blue = 640)
        // A 2x2 patch: exactly one cell.
        for (y in 2..3) {
            for (x in 2..3) buffer.saturate(x, y)
        }

        assertEquals(1, scan(buffer, stride = 1).cellsClipped)
        // The stride skips cells, so a patch that lands between two visits is not
        // guaranteed to be seen - stated here because the estimate tolerates it:
        // the gains are unchanged at 1 either way.
        val gains = GreyWorldEstimate.phaseGains(means(buffer, stride = 2), rggb)!!
        gains.forEach { gain -> assertClose("gain", 1f, gain, 1e-3f) }
    }

    @Test
    fun blackSubtractionHappensAfterTheClipTest() {
        // A dark but unclipped frame must still produce means, and a frame whose
        // filters all sit at black must not be read as clipped.
        val dark = mosaic(red = 64, greenA = 64, greenB = 64, blue = 64)

        val result = scan(dark, black = intArrayOf(64, 64, 64, 64))

        assertEquals(0, result.cellsClipped)
        assertEquals(16, result.cellsUsed)
        assertArrayEquals(doubleArrayOf(0.0, 0.0, 0.0, 0.0), result.means, 1e-9)
    }

    private fun assertClose(label: String, expected: Float, actual: Float, tolerance: Float = 1e-4f) {
        assertTrue(
            "$label: expected $expected but was $actual (tolerance $tolerance)",
            abs(expected - actual) <= tolerance
        )
    }
}