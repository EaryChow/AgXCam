package com.agx.camera.camera

import java.nio.ByteBuffer

// White-balance mode: the illuminant from block medians.
//
// A grey-world sum is dominated by whatever carries the most signal: signage, a
// warm facade, a lamp. Averaging the cell responses into blocks first and taking
// the median of the block neutrals keeps that content from dragging the
// illuminant. The result is the illuminant stage the shared smoother consumes,
// so the existing temporal machinery applies to it unchanged.
object WbMode {

    /** Cells of the estimator's stride grid averaged into one median block. */
    const val BLOCK_CELLS = 8

    /** Cell responses of the sparse walk, grouped by block. */
    class Blocks(
        val rows: Int,
        val cols: Int,
        /** Raw code sums per block, four per block in phase order. */
        val sums: DoubleArray,
        /** Cells that contributed to each block. */
        val counts: IntArray
    ) {
        val size: Int get() = rows * cols
    }

    /** Per-block neutrals of the median estimate. */
    class Samples(
        /** Green-normalized neutral per block, three entries each, green at 1. */
        val neutrals: FloatArray,
        /** False where a block carried no usable signal. */
        val valid: BooleanArray
    ) {
        val size: Int get() = valid.size
    }

    /**
     * One pass over the stride grid, summing the cells of each block. Cells are
     * 2x2 and anchored on even coordinates, as in [GreyWorldEstimate.scan]; a
     * cell with any filter at or above the sensor's white level is dropped
     * whole. Only whole blocks are visited, so the block fields stay rectangular.
     */
    fun scanBlocks(
        buffer: ByteBuffer,
        width: Int,
        height: Int,
        cellStride: Int,
        whiteLevel: Int,
        blockCells: Int = BLOCK_CELLS
    ): Blocks {
        val step = cellStride.coerceAtLeast(1)
        val block = blockCells.coerceAtLeast(1)
        val clip = if (whiteLevel > 0) whiteLevel else Int.MAX_VALUE
        val sampledCols = (width / 2 + step - 1) / step
        val sampledRows = (height / 2 + step - 1) / step
        val cols = sampledCols / block
        val rows = sampledRows / block
        val sums = DoubleArray(rows * cols * GreyWorldEstimate.PHASE_COUNT)
        val counts = IntArray(rows * cols)
        for (br in 0 until rows) {
            for (bc in 0 until cols) {
                val index = br * cols + bc
                for (i in 0 until block) {
                    val cy = (br * block + i) * step
                    val rowTop = (cy * 2) * width
                    val rowBottom = rowTop + width
                    for (j in 0 until block) {
                        val col = (bc * block + j) * step * 2
                        val top = (rowTop + col) * 2
                        val bottom = (rowBottom + col) * 2
                        val p0 = rawAt(buffer, top)
                        val p1 = rawAt(buffer, top + 2)
                        val p2 = rawAt(buffer, bottom)
                        val p3 = rawAt(buffer, bottom + 2)
                        if (p0 >= clip || p1 >= clip || p2 >= clip || p3 >= clip) continue
                        val base = index * GreyWorldEstimate.PHASE_COUNT
                        sums[base] += p0
                        sums[base + 1] += p1
                        sums[base + 2] += p2
                        sums[base + 3] += p3
                        counts[index]++
                    }
                }
            }
        }
        return Blocks(rows, cols, sums, counts)
    }

    // RAW arrives little-endian: low byte first.
    private fun rawAt(buffer: ByteBuffer, index: Int): Int =
        (buffer.get(index).toInt() and 0xFF) or ((buffer.get(index + 1).toInt() and 0xFF) shl 8)

    /**
     * Per-block neutral from the block averages.
     *
     * The whole block is dropped when any phase of its average carries no
     * usable signal, the same rule the mean estimate applies to its phase
     * means.
     */
    fun blockSamples(
        blocks: Blocks,
        blackLevels: IntArray,
        colorMap: IntArray,
        daylight: FloatArray?
    ): Samples {
        val count = blocks.size
        val neutrals = FloatArray(count * 3)
        val valid = BooleanArray(count)
        val layout = channels(colorMap, daylight)
        if (count == 0 || layout.greens.size != 2) return Samples(neutrals, valid)
        val means = DoubleArray(GreyWorldEstimate.PHASE_COUNT)
        for (b in 0 until count) {
            val cells = blocks.counts[b]
            if (cells == 0) continue
            for (p in 0 until GreyWorldEstimate.PHASE_COUNT) {
                means[p] = blocks.sums[b * GreyWorldEstimate.PHASE_COUNT + p] / cells -
                    blackLevels.getOrElse(p) { 0 }
            }
            val estimate = GreyWorldEstimate.estimate(means, colorMap, daylight) ?: continue
            neutrals[b * 3] = 1f / estimate.phaseGains[layout.redPhase]
            neutrals[b * 3 + 1] = 1f
            neutrals[b * 3 + 2] = 1f / estimate.phaseGains[layout.bluePhase]
            valid[b] = true
        }
        return Samples(neutrals, valid)
    }

    /** Per-phase daylight whitening and the phase indices the estimator uses. */
    private class Channels(
        val greens: IntArray,
        val redPhase: Int,
        val bluePhase: Int,
        val whitening: DoubleArray
    )

    private fun channels(colorMap: IntArray, daylight: FloatArray?): Channels {
        val anchor = daylight?.takeIf { d -> d.size >= 3 && d.all { it.isFinite() && it > 0f } }
        val redPhase = phaseOf(colorMap, 0)
        val bluePhase = phaseOf(colorMap, 2)
        val whitening = DoubleArray(GreyWorldEstimate.PHASE_COUNT) { p ->
            when {
                anchor == null -> 1.0
                p == redPhase -> anchor[0].toDouble()
                p == bluePhase -> anchor[2].toDouble()
                else -> 1.0
            }
        }
        return Channels(GreyWorldEstimate.greenPhases(colorMap), redPhase, bluePhase, whitening)
    }

    // Phase of a colour in the CFA map, defaulting the way the estimator does.
    private fun phaseOf(colorMap: IntArray, color: Int): Int {
        var phase = if (color == 0) 0 else GreyWorldEstimate.PHASE_COUNT - 1
        for (p in 0 until GreyWorldEstimate.PHASE_COUNT) {
            if (colorMap.getOrElse(p) { 1 } == color) phase = p
        }
        return phase
    }

    /**
     * Component-wise median of the valid blocks' neutrals, or null when no block
     * carried signal. The median is taken per component, green stays at 1.
     */
    fun medianNeutral(samples: Samples): FloatArray? {
        var count = 0
        for (i in 0 until samples.size) if (samples.valid[i]) count++
        if (count == 0) return null
        val red = DoubleArray(count)
        val blue = DoubleArray(count)
        var j = 0
        for (i in 0 until samples.size) {
            if (!samples.valid[i]) continue
            red[j] = samples.neutrals[i * 3].toDouble()
            blue[j] = samples.neutrals[i * 3 + 2].toDouble()
            j++
        }
        return floatArrayOf(
            median(red, red.size).toFloat(),
            1f,
            median(blue, blue.size).toFloat()
        )
    }

    /**
     * The block-median estimate as the four-phase illuminant stage the shared
     * smoother consumes: the median green-normalized neutral inverted through
     * the same daylight whitening the mean estimate uses.
     */
    fun medianIlluminant(
        buffer: java.nio.ByteBuffer,
        width: Int,
        height: Int,
        blackLevels: IntArray,
        colorMap: IntArray,
        daylight: FloatArray?,
        cellStride: Int,
        whiteLevel: Int
    ): FloatArray? {
        val blocks = scanBlocks(buffer, width, height, cellStride, whiteLevel)
        val samples = blockSamples(blocks, blackLevels, colorMap, daylight)
        val neutral = medianNeutral(samples) ?: return null
        val layout = channels(colorMap, daylight)
        if (layout.greens.size != 2) return null
        val illuminant = FloatArray(GreyWorldEstimate.PHASE_COUNT) { 1f }
        for (p in 0 until GreyWorldEstimate.PHASE_COUNT) {
            when (p) {
                layout.redPhase -> {
                    val pg = 1f / neutral[0]
                    illuminant[p] = (pg / layout.whitening[p].toFloat())
                        .coerceIn(GreyWorldEstimate.MIN_GAIN, GreyWorldEstimate.MAX_GAIN)
                }
                layout.bluePhase -> {
                    val pg = 1f / neutral[2]
                    illuminant[p] = (pg / layout.whitening[p].toFloat())
                        .coerceIn(GreyWorldEstimate.MIN_GAIN, GreyWorldEstimate.MAX_GAIN)
                }
            }
        }
        return illuminant
    }

    /**
     * Median of the first [count] entries of [values], reordering them in place.
     * An even count averages the two middle values.
     */
    fun median(values: DoubleArray, count: Int): Double {
        if (count <= 0) return Double.NaN
        val middle = (count - 1) / 2
        selectKth(values, count, middle)
        if (count % 2 == 1) return values[middle]
        var upper = Double.POSITIVE_INFINITY
        for (i in middle + 1 until count) if (values[i] < upper) upper = values[i]
        return (values[middle] + upper) * 0.5
    }

    // Partial sort: afterwards every entry before [k] is at or under it.
    private fun selectKth(a: DoubleArray, count: Int, k: Int) {
        var left = 0
        var right = count - 1
        while (left < right) {
            val pivot = a[(left + right) ushr 1]
            var i = left
            var j = right
            while (i <= j) {
                while (a[i] < pivot) i++
                while (a[j] > pivot) j--
                if (i <= j) {
                    val t = a[i]
                    a[i] = a[j]
                    a[j] = t
                    i++
                    j--
                }
            }
            when {
                k <= j -> right = j
                k >= i -> left = i
                else -> return
            }
        }
    }

}
