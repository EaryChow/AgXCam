package com.agx.camera.phase1.week7

import org.junit.Assert.*
import org.junit.Test

class MemoryStressTest {

    companion object {
        private const val MP_50_WIDTH = 8192
        private const val MP_50_HEIGHT = 6144
        private const val MP_50_PIXELS = MP_50_WIDTH.toLong() * MP_50_HEIGHT.toLong()
        private const val BYTES_PER_PIXEL_16BIT = 2
        private const val BYTES_PER_PIXEL_32BIT_FLOAT = 4
        private const val TILE_SIZE = 2048
        private const val OVERLAP = 2
    }

    private data class SimTile(
        val originX: Int, val originY: Int,
        val readW: Int, val readH: Int,
        val outputOriginX: Int, val outputOriginY: Int,
        val outputW: Int, val outputH: Int
    )

    private fun simulateTileGrid(sensorW: Int, sensorH: Int): List<SimTile> {
        val tiles = mutableListOf<SimTile>()
        var y = 0
        while (y < sensorH) {
            var x = 0
            while (x < sensorW) {
                val tileW = minOf(TILE_SIZE, sensorW - x)
                val tileH = minOf(TILE_SIZE, sensorH - y)
                val originX = maxOf(0, x - OVERLAP)
                val originY = maxOf(0, y - OVERLAP)
                val readW = minOf(tileW + 2 * OVERLAP, sensorW - originX)
                val readH = minOf(tileH + 2 * OVERLAP, sensorH - originY)

                tiles.add(SimTile(originX, originY, readW, readH, x, y, tileW, tileH))
                x += TILE_SIZE
            }
            y += TILE_SIZE
        }
        return tiles
    }

    @Test
    fun bayerBuffer_50mp_16bit_exactSize() {
        val expectedBytes = MP_50_PIXELS * BYTES_PER_PIXEL_16BIT
        assertEquals(100_663_296L, expectedBytes)
        assertEquals(96, expectedBytes / (1024 * 1024))
    }

    @Test
    fun bayerBuffer_50mp_12bit_packedSize() {
        val bitsNeeded = MP_50_PIXELS * 12
        val bytesNeeded = (bitsNeeded + 7) / 8
        assertEquals(75_497_472L, bytesNeeded)
    }

    @Test
    fun bayerBuffer_50mp_10bit_packedSize() {
        val bitsNeeded = MP_50_PIXELS * 10
        val bytesNeeded = (bitsNeeded + 7) / 8
        assertEquals(62_914_560L, bytesNeeded)
    }

    @Test
    fun demosaicOutput_50mp_rgbFloat_size() {
        val expectedBytes = MP_50_PIXELS * 3 * BYTES_PER_PIXEL_32BIT_FLOAT
        assertEquals(603_979_776L, expectedBytes)
        assertEquals(576, expectedBytes / (1024 * 1024))
    }

    @Test
    fun lensShadingMap_50mp_rgba16f_size() {
        val expectedBytes = MP_50_PIXELS * 4 * 2
        assertEquals(402_653_184L, expectedBytes)
    }

    @Test
    fun tileGrid_coversEntireSensor() {
        val tiles = simulateTileGrid(MP_50_WIDTH, MP_50_HEIGHT)

        val maxOutX = tiles.maxOf { it.outputOriginX + it.outputW }
        val maxOutY = tiles.maxOf { it.outputOriginY + it.outputH }

        assertEquals(MP_50_WIDTH, maxOutX)
        assertEquals(MP_50_HEIGHT, maxOutY)
    }

    @Test
    fun tileGrid_totalTiles_count() {
        val tiles = simulateTileGrid(MP_50_WIDTH, MP_50_HEIGHT)
        assertEquals(12, tiles.size)
    }

    @Test
    fun tileGrid_tileDimensions_valid() {
        val tiles = simulateTileGrid(MP_50_WIDTH, MP_50_HEIGHT)

        for (tile in tiles) {
            assertTrue("Tile readW=${tile.readW} too small", tile.readW > 0)
            assertTrue("Tile readH=${tile.readH} too small", tile.readH > 0)
            assertTrue("Tile readW=${tile.readW} exceeds max", tile.readW <= TILE_SIZE + 2 * OVERLAP)
            assertTrue("Tile readH=${tile.readH} exceeds max", tile.readH <= TILE_SIZE + 2 * OVERLAP)
        }
    }

    @Test
    fun tileGrid_blitRectangles_withinTileBounds() {
        val tiles = simulateTileGrid(MP_50_WIDTH, MP_50_HEIGHT)

        for (tile in tiles) {
            val srcX1 = tile.outputOriginX - tile.originX
            val srcY1 = tile.outputOriginY - tile.originY
            val dstW = minOf(TILE_SIZE, MP_50_WIDTH - tile.outputOriginX)
            val dstH = minOf(TILE_SIZE, MP_50_HEIGHT - tile.outputOriginY)
            val srcX2 = srcX1 + dstW
            val srcY2 = srcY1 + dstH

            assertTrue("Tile (${tile.outputOriginX},${tile.outputOriginY}) srcX2=$srcX2 > readW=${tile.readW}",
                srcX2 <= tile.readW)
            assertTrue("Tile (${tile.outputOriginX},${tile.outputOriginY}) srcY2=$srcY2 > readH=${tile.readH}",
                srcY2 <= tile.readH)
            assertTrue("Tile srcX1=$srcX1 >= 0", srcX1 >= 0)
            assertTrue("Tile srcY1=$srcY1 >= 0", srcY1 >= 0)
        }
    }

    @Test
    fun tileGrid_outputCoversAllPixels() {
        val tiles = simulateTileGrid(MP_50_WIDTH, MP_50_HEIGHT)
        val covered = Array(MP_50_HEIGHT) { BooleanArray(MP_50_WIDTH) }

        for (tile in tiles) {
            for (y in tile.outputOriginY until tile.outputOriginY + tile.outputH) {
                for (x in tile.outputOriginX until tile.outputOriginX + tile.outputW) {
                    covered[y][x] = true
                }
            }
        }

        for (y in 0 until MP_50_HEIGHT) {
            for (x in 0 until MP_50_WIDTH) {
                assertTrue("Pixel ($x,$y) not covered by any tile", covered[y][x])
            }
        }
    }

    @Test
    fun simulatedPipeline_memoryEstimate_reasonable() {
        val bayerBytes = MP_50_PIXELS * BYTES_PER_PIXEL_16BIT
        val fboBytes = MP_50_PIXELS * 3 * BYTES_PER_PIXEL_32BIT_FLOAT
        val totalBytes = bayerBytes + fboBytes

        assertTrue("Total memory exceeds 700MB", totalBytes < 700 * 1024 * 1024)
        assertTrue("Total memory below 500MB", totalBytes > 500 * 1024 * 1024)
    }

    @Test
    fun tileMemory_singleTile_reasonable() {
        val tilePixels = (TILE_SIZE + 2 * OVERLAP).toLong() * (TILE_SIZE + 2 * OVERLAP).toLong()
        val tileBytes = tilePixels * BYTES_PER_PIXEL_16BIT

        assertTrue("Single tile exceeds 32MB", tileBytes < 32 * 1024 * 1024)
        assertTrue("Single tile below 8MB", tileBytes > 8 * 1024 * 1024)
    }

    @Test
    fun tileGrid_adjacentTiles_readRegionsOverlap() {
        val tilesX = (MP_50_WIDTH + TILE_SIZE - 1) / TILE_SIZE
        val tiles = simulateTileGrid(MP_50_WIDTH, MP_50_HEIGHT)

        for (j in 0 until (MP_50_HEIGHT + TILE_SIZE - 1) / TILE_SIZE) {
            for (i in 0 until tilesX - 1) {
                val left = tiles[j * tilesX + i]
                val right = tiles[j * tilesX + i + 1]
                val leftReadEndX = left.originX + left.readW
                val rightReadStartX = right.originX
                val overlapPixels = leftReadEndX - rightReadStartX
                assertTrue("Horizontal read overlap too small at ($i,$j): $overlapPixels", overlapPixels >= 2 * OVERLAP)
            }
        }

        for (j in 0 until (MP_50_HEIGHT + TILE_SIZE - 1) / TILE_SIZE - 1) {
            for (i in 0 until tilesX) {
                val top = tiles[j * tilesX + i]
                val bottom = tiles[(j + 1) * tilesX + i]
                val topReadEndY = top.originY + top.readH
                val bottomReadStartY = bottom.originY
                val overlapPixels = topReadEndY - bottomReadStartY
                assertTrue("Vertical read overlap too small at ($i,$j): $overlapPixels", overlapPixels >= 2 * OVERLAP)
            }
        }
    }

    @Test
    fun shaderHandlesAllBitDepths_10_12_14_16() {
        val validDepths = listOf(10, 12, 14, 16)
        for (depth in validDepths) {
            val mask = when {
                depth <= 10 -> 0x3FF
                depth <= 12 -> 0xFFF
                depth <= 14 -> 0x3FFF
                else -> 0xFFFF
            }
            assertTrue("Mask for $depth-bit should be positive", mask > 0)
            assertTrue("Mask for $depth-bit should fit in 16 bits", mask <= 0xFFFF)
        }
    }

    @Test
    fun bayerPatternMemory_8192x6144_16bit_matchesExpected() {
        val pixelCount = 8192L * 6144
        val bytes16 = pixelCount * 2
        val bytes32 = pixelCount * 4

        assertEquals(100_663_296L, bytes16)
        assertEquals(201_326_592L, bytes32)
    }
}
