package com.agx.camera.camera

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer

class WbModeTest {

    private val rggb = intArrayOf(0, 1, 1, 2)
    private val noBlack = intArrayOf(0, 0, 0, 0)
    private val white = 1023

    // 8x8 mosaic is 4x4 CFA cells; a cell stride of 1 with 2x2-cell blocks
    // gives a 2x2 block grid.
    private val size = 8

    private fun mosaic(red: Int, greenA: Int, greenB: Int, blue: Int): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(size * size * 2)
        for (y in 0 until size) {
            for (x in 0 until size) {
                val value = when (rggb[(x and 1) + (y and 1) * 2]) {
                    0 -> red
                    2 -> blue
                    else -> if ((x and 1) == 1) greenB else greenA
                }
                setSample(buffer, x, y, value)
            }
        }
        buffer.clear()
        return buffer
    }

    private fun setSample(buffer: ByteBuffer, x: Int, y: Int, value: Int) {
        val i = (y * size + x) * 2
        buffer.put(i, (value and 0xFF).toByte())
        buffer.put(i + 1, ((value shr 8) and 0xFF).toByte())
    }

    private fun blocks(
        buffer: ByteBuffer,
        stride: Int = 1,
        whiteLevel: Int = white
    ) = WbMode.scanBlocks(buffer, size, size, stride, whiteLevel, blockCells = 2)

    @Test
    fun scanBlocksSumsWholeBlocksOnThePhaseGrid() {
        val buffer = mosaic(red = 100, greenA = 200, greenB = 400, blue = 50)
        val blocks = blocks(buffer)

        assertEquals(2, blocks.rows)
        assertEquals(2, blocks.cols)
        assertEquals(4, blocks.size)
        // Four cells per block, each cell one sample of every phase, in the
        // scan's phase order: red, green on odd columns, green on even
        // columns, blue.
        for (b in 0 until blocks.size) {
            assertEquals(4, blocks.counts[b])
            assertArrayEquals(
                doubleArrayOf(400.0, 1600.0, 800.0, 200.0),
                doubleArrayOf(
                    blocks.sums[b * 4], blocks.sums[b * 4 + 1],
                    blocks.sums[b * 4 + 2], blocks.sums[b * 4 + 3]
                ),
                1e-9
            )
        }
    }

    @Test
    fun scanBlocksDropsAWholeCellWithAClippedFilter() {
        val buffer = mosaic(red = 100, greenA = 200, greenB = 400, blue = 50)
        // One cell of the first block reads full capacity on one filter only.
        setSample(buffer, 2, 2, white)
        val blocks = blocks(buffer)

        assertEquals(3, blocks.counts[0])
        assertArrayEquals(
            doubleArrayOf(300.0, 1200.0, 600.0, 150.0),
            doubleArrayOf(
                blocks.sums[0], blocks.sums[1], blocks.sums[2], blocks.sums[3]
            ),
            1e-9
        )
        assertEquals(4, blocks.counts[1])
    }

    @Test
    fun scanBlocksDropsThePartialBlockTail() {
        // 8 pixels is 4 cells; a block of 3 cells keeps only the first three,
        // so the block grid is 1x1 and the fourth cell is not visited.
        val buffer = mosaic(red = 100, greenA = 200, greenB = 400, blue = 50)
        val blocks = WbMode.scanBlocks(buffer, size, size, 1, white, blockCells = 3)

        assertEquals(1, blocks.rows)
        assertEquals(1, blocks.cols)
        assertEquals(9, blocks.counts[0])
    }

    @Test
    fun blockSamplesReportTheGreenNormalizedNeutral() {
        // Corrected response [200, 200, 200, 50]: the level is 200, the
        // illuminant gains are [1, 1, 1, 4], the daylight gains double red, so
        // the neutral is [0.5, 1, 0.25].
        val blocks = WbMode.Blocks(
            rows = 1, cols = 1,
            sums = doubleArrayOf(100.0, 200.0, 200.0, 50.0),
            counts = intArrayOf(1)
        )
        val samples = WbMode.blockSamples(
            blocks, noBlack, rggb, floatArrayOf(2f, 1f, 1f)
        )

        assertTrue(samples.valid[0])
        assertEquals(0.5f, samples.neutrals[0], 1e-6f)
        assertEquals(1f, samples.neutrals[1], 1e-6f)
        assertEquals(0.25f, samples.neutrals[2], 1e-6f)
    }

    @Test
    fun blockSamplesSubtractBlackLevelsFirst() {
        val blocks = WbMode.Blocks(
            rows = 1, cols = 1,
            sums = doubleArrayOf(164.0, 264.0, 264.0, 114.0),
            counts = intArrayOf(1)
        )
        val samples = WbMode.blockSamples(
            blocks, intArrayOf(64, 64, 64, 64), rggb, null
        )

        // Means [100, 200, 200, 50], level 200, no daylight: [0.5, 1, 0.25].
        assertTrue(samples.valid[0])
        assertEquals(0.5f, samples.neutrals[0], 1e-6f)
        assertEquals(0.25f, samples.neutrals[2], 1e-6f)
    }

    @Test
    fun blockSamplesDropBlocksWithoutSignal() {
        val blocks = WbMode.Blocks(
            rows = 1, cols = 2,
            sums = doubleArrayOf(100.0, 200.0, 200.0, 50.0, 0.0, 200.0, 200.0, 50.0),
            counts = intArrayOf(1, 0)
        )
        val samples = WbMode.blockSamples(blocks, noBlack, rggb, null)

        assertTrue(samples.valid[0])
        assertTrue(!samples.valid[1])
    }

    @Test
    fun medianOddCount() {
        assertEquals(3.0, WbMode.median(doubleArrayOf(5.0, 1.0, 3.0), 3), 1e-12)
    }

    @Test
    fun medianEvenCountAveragesTheMiddlePair() {
        assertEquals(2.5, WbMode.median(doubleArrayOf(4.0, 1.0, 3.0, 2.0), 4), 1e-12)
    }

    @Test
    fun medianIgnoresEntriesPastTheCount() {
        assertEquals(2.0, WbMode.median(doubleArrayOf(9.0, 2.0, 1.0, 100.0, 100.0), 3), 1e-12)
    }

    @Test
    fun medianNeutralIsComponentWise() {
        val samples = WbMode.Samples(
            neutrals = floatArrayOf(
                0.5f, 1f, 2.0f,
                0.3f, 1f, 3.0f,
                0.4f, 1f, 1.0f
            ),
            valid = booleanArrayOf(true, true, true)
        )
        val neutral = WbMode.medianNeutral(samples)

        assertNotNull(neutral)
        assertEquals(0.4f, neutral!![0], 1e-6f)
        assertEquals(1f, neutral[1], 1e-6f)
        assertEquals(2.0f, neutral[2], 1e-6f)
    }

    @Test
    fun medianNeutralReturnsNullWithoutValidBlocks() {
        val samples = WbMode.Samples(
            neutrals = FloatArray(6),
            valid = booleanArrayOf(false, false)
        )
        assertNull(WbMode.medianNeutral(samples))
    }

    @Test
    fun medianNeutralDropsBlocksBelowTheGateWeight() {
        val samples = WbMode.Samples(
            neutrals = floatArrayOf(
                0.50f, 1f, 2.00f,
                0.52f, 1f, 2.02f,
                0.20f, 1f, 4.00f,
                0.18f, 1f, 4.10f
            ),
            valid = booleanArrayOf(true, true, true, true)
        )
        // The last two blocks read as coloured content and earn no weight.
        // Two of four survive - exactly half - so the fallback does not apply.
        val weightOf: (FloatArray) -> Float = { n -> if (n[0] > 0.3f) 1f else 0f }
        val neutral = WbMode.medianNeutral(samples, weightOf)!!

        assertEquals(0.51f, neutral[0], 1e-6f)
        assertEquals(2.01f, neutral[2], 1e-6f)
    }

    @Test
    fun medianNeutralFallsBackWhenTooFewBlocksSurvive() {
        val samples = WbMode.Samples(
            neutrals = floatArrayOf(
                0.50f, 1f, 2.00f,
                0.20f, 1f, 4.00f
            ),
            valid = booleanArrayOf(true, true)
        )
        // Nothing survives the gate; the unweighted median is the estimate.
        val neutral = WbMode.medianNeutral(samples) { 0f }!!

        assertEquals(0.35f, neutral[0], 1e-6f)
        assertEquals(3.00f, neutral[2], 1e-6f)
    }

}
