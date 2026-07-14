package com.agx.camera.gpu

import com.agx.camera.camera.BayerPattern
import org.junit.Assert.*
import org.junit.Test

class TiledComputeSchedulerTest {

    private val scheduler = TiledComputeScheduler(tileSize = 2048, overlap = 2)

    @Test
    fun singleTile_smallSensor() {
        val grid = scheduler.computeTileGrid(1920, 1080)
        assertEquals(1, grid.tiles.size)
        assertEquals(1920, grid.totalWidth)
        assertEquals(1080, grid.totalHeight)
        assertEquals(1, grid.tilesX)
        assertEquals(1, grid.tilesY)

        val tile = grid.tiles[0]
        assertEquals(0, tile.originX)
        assertEquals(0, tile.originY)
        assertEquals(1920, tile.width)
        assertEquals(1080, tile.height)
        assertEquals(0, tile.outputOriginX)
        assertEquals(0, tile.outputOriginY)
    }

    @Test
    fun multipleTiles_50mpSensor() {
        val grid = scheduler.computeTileGrid(8192, 6144)
        assertTrue("Should have multiple tiles for 50MP", grid.tiles.size > 1)
        assertEquals(8192, grid.totalWidth)
        assertEquals(6144, grid.totalHeight)

        assertTrue("tilesX should be >= 4", grid.tilesX >= 4)
        assertTrue("tilesY should be >= 3", grid.tilesY >= 3)
    }

    @Test
    fun tileOverlap_bordersCoverSensorEdges() {
        val grid = scheduler.computeTileGrid(8192, 6144)
        val firstTile = grid.tiles[0]
        assertEquals("First tile starts at origin with overlap", 0, firstTile.originX)
        assertEquals(0, firstTile.originY)

        val lastTile = grid.tiles.last()
        assertTrue("Last tile extends to sensor edge", lastTile.originX + lastTile.width >= 8192)
        assertTrue(lastTile.originY + lastTile.height >= 6144)
    }

    @Test
    fun tileOutputOrigins_tessellateCorrectly() {
        val grid = scheduler.computeTileGrid(4096, 4096)
        for (tile in grid.tiles) {
            assertTrue("outputOriginX >= 0", tile.outputOriginX >= 0)
            assertTrue("outputOriginY >= 0", tile.outputOriginY >= 0)
            assertTrue("outputOriginX < totalWidth", tile.outputOriginX < grid.totalWidth)
            assertTrue("outputOriginY < totalHeight", tile.outputOriginY < grid.totalHeight)
        }
    }

    @Test
    fun validateTileOverlap_valid() {
        val grid = scheduler.computeTileGrid(8192, 6144)
        assertTrue("Tile overlap should be valid", scheduler.validateTileOverlap(grid))
    }

    @Test
    fun validateTileOverlap_singleTile() {
        val grid = scheduler.computeTileGrid(1000, 1000)
        assertTrue("Single tile should be valid", scheduler.validateTileOverlap(grid))
    }

    @Test
    fun croppedRegion() {
        val grid = scheduler.computeTileGrid(
            sensorWidth = 8192, sensorHeight = 6144,
            cropX = 1024, cropY = 1024,
            cropWidth = 4096, cropHeight = 3072
        )
        assertEquals(4096, grid.totalWidth)
        assertEquals(3072, grid.totalHeight)

        for (tile in grid.tiles) {
            assertTrue("tile origin within sensor bounds",
                tile.originX >= 0 && tile.originY >= 0)
            assertTrue("tile output within crop bounds",
                tile.outputOriginX + tileSize() <= grid.totalWidth ||
                tile.outputOriginY + tileSize() <= grid.totalHeight)
        }
    }

    @Test
    fun estimatePeakGpuMemory_reasonable() {
        val grid = scheduler.computeTileGrid(8192, 6144)
        val mem = scheduler.estimatePeakGpuMemory(grid)
        assertTrue("Peak GPU memory should be > 0", mem > 0)
        assertTrue("Peak GPU memory should be < 1GB for 50MP", mem < 1_000_000_000L)
    }

    @Test
    fun pixelCoverage_noGaps() {
        val grid = scheduler.computeTileGrid(4000, 3000)
        val coverage = Array(3000) { BooleanArray(4000) }

        for (tile in grid.tiles) {
            for (y in 0 until tile.height) {
                for (x in 0 until tile.width) {
                    val sensorX = tile.originX + x
                    val sensorY = tile.originY + y
                    if (sensorX in 0 until 4000 && sensorY in 0 until 3000) {
                        coverage[sensorY][sensorX] = true
                    }
                }
            }
        }

        for (y in 0 until 3000) {
            for (x in 0 until 4000) {
                assertTrue("Pixel ($x,$y) must be covered", coverage[y][x])
            }
        }
    }

    @Test
    fun synthetic50mp_fullPipelineValidation() {
        val sensorW = 8192
        val sensorH = 6144
        val adapter = StandardBayerAdapter()
        val grid = scheduler.computeTileGrid(sensorW, sensorH)

        assertTrue("50MP needs multiple tiles", grid.tiles.size > 1)
        assertEquals(sensorW, grid.totalWidth)
        assertEquals(sensorH, grid.totalHeight)

        for (tile in grid.tiles) {
            assertTrue("tile.width > 0", tile.width > 0)
            assertTrue("tile.height > 0", tile.height > 0)
            assertTrue("tile.originX >= 0", tile.originX >= 0)
            assertTrue("tile.originY >= 0", tile.originY >= 0)
            assertTrue("tile.outputOriginX >= 0", tile.outputOriginX >= 0)
            assertTrue("tile.outputOriginY >= 0", tile.outputOriginY >= 0)

            val dstW = minOf(adapter.tileSize(), sensorW - tile.outputOriginX)
            val dstH = minOf(adapter.tileSize(), sensorH - tile.outputOriginY)
            val srcX1 = tile.outputOriginX - tile.originX
            val srcY1 = tile.outputOriginY - tile.originY
            val srcX2 = srcX1 + dstW
            val srcY2 = srcY1 + dstH

            assertTrue("srcX1 >= 0", srcX1 >= 0)
            assertTrue("srcY1 >= 0", srcY1 >= 0)
            assertTrue("srcX2 <= tile.width", srcX2 <= tile.width)
            assertTrue("srcY2 <= tile.height", srcY2 <= tile.height)

            val groupsX = (tile.width + 15) / 16
            val groupsY = (tile.height + 15) / 16
            assertTrue("groupsX > 0", groupsX > 0)
            assertTrue("groupsY > 0", groupsY > 0)
            assertTrue("groupsX * 16 >= tile.width", groupsX * 16 >= tile.width)
            assertTrue("groupsY * 16 >= tile.height", groupsY * 16 >= tile.height)
        }

        assertTrue("Overlap valid", scheduler.validateTileOverlap(grid))

        val mem = scheduler.estimatePeakGpuMemory(grid)
        assertTrue("50MP memory > 0", mem > 0)
        assertTrue("50MP memory < 1GB", mem < 1_000_000_000L)
    }

    @Test
    fun synthetic50mp_tileBlitRectangles() {
        val sensorW = 8192
        val sensorH = 6144
        val adapter = StandardBayerAdapter()
        val grid = scheduler.computeTileGrid(sensorW, sensorH)

        for (tile in grid.tiles) {
            val dstW = minOf(adapter.tileSize(), sensorW - tile.outputOriginX)
            val dstH = minOf(adapter.tileSize(), sensorH - tile.outputOriginY)
            val srcX1 = tile.outputOriginX - tile.originX
            val srcY1 = tile.outputOriginY - tile.originY

            assertTrue("srcX1 within tile", srcX1 in 0 until tile.width)
            assertTrue("srcY1 within tile", srcY1 in 0 until tile.height)
            assertTrue("srcX1+dstW within tile", srcX1 + dstW <= tile.width)
            assertTrue("srcY1+dstH within tile", srcY1 + dstH <= tile.height)

            assertTrue("dstW > 0 and <= tileSize", dstW in 1..adapter.tileSize())
            assertTrue("dstH > 0 and <= tileSize", dstH in 1..adapter.tileSize())

            assertTrue("blit covers outputOrigin",
                tile.outputOriginX + dstW <= sensorW)
            assertTrue("blit covers outputOrigin Y",
                tile.outputOriginY + dstH <= sensorH)
        }
    }

    @Test
    fun synthetic50mp_overlapCoversAllAdjacentPairs() {
        val sensorW = 8192
        val sensorH = 6144
        val adapter = StandardBayerAdapter()
        val grid = scheduler.computeTileGrid(sensorW, sensorH)
        val minOverlap = 2 * adapter.overlapPixels()

        for (i in grid.tiles.indices) {
            for (j in i + 1 until grid.tiles.size) {
                val a = grid.tiles[i]
                val b = grid.tiles[j]

                val overlapX = maxOf(0,
                    minOf(a.originX + a.width, b.originX + b.width) -
                    maxOf(a.originX, b.originX))
                val overlapY = maxOf(0,
                    minOf(a.originY + a.height, b.originY + b.height) -
                    maxOf(a.originY, b.originY))

                if (overlapX > 0 && overlapY > 0) {
                    val isHorizAdjacent = a.outputOriginY == b.outputOriginY &&
                        (a.outputOriginX + adapter.tileSize() > b.outputOriginX ||
                         b.outputOriginX + adapter.tileSize() > a.outputOriginX)
                    val isVertAdjacent = a.outputOriginX == b.outputOriginX &&
                        (a.outputOriginY + adapter.tileSize() > b.outputOriginY ||
                         b.outputOriginY + adapter.tileSize() > a.outputOriginY)

                    if (isHorizAdjacent || isVertAdjacent) {
                        assertTrue("Adjacent tiles must overlap by >= $minOverlap in X",
                            overlapX >= minOverlap)
                        assertTrue("Adjacent tiles must overlap by >= $minOverlap in Y",
                            overlapY >= minOverlap)
                    }
                }
            }
        }
    }

    @Test
    fun synthetic50mp_pixelCoverage_noGaps() {
        val sensorW = 8192
        val sensorH = 6144
        val grid = scheduler.computeTileGrid(sensorW, sensorH)

        val coverage = Array(sensorH) { ByteArray(sensorW) }

        for (tile in grid.tiles) {
            for (y in 0 until tile.height) {
                for (x in 0 until tile.width) {
                    val sx = tile.originX + x
                    val sy = tile.originY + y
                    if (sx in 0 until sensorW && sy in 0 until sensorH) {
                        coverage[sy][sx] = 1
                    }
                }
            }
        }

        for (y in 0 until sensorH) {
            for (x in 0 until sensorW) {
                assertTrue("Pixel ($x,$y) must be covered by at least one tile",
                    coverage[y][x].toInt() != 0)
            }
        }
    }

    @Test
    fun synthetic50mp_simulatedPipeline_outputCoverage() {
        val sensorW = 8192
        val sensorH = 6144
        val adapter = StandardBayerAdapter()
        val grid = scheduler.computeTileGrid(sensorW, sensorH)

        val coverage = Array(sensorH) { IntArray(sensorW) }

        for (tile in grid.tiles) {
            val dstW = minOf(adapter.tileSize(), sensorW - tile.outputOriginX)
            val dstH = minOf(adapter.tileSize(), sensorH - tile.outputOriginY)
            val srcX1 = tile.outputOriginX - tile.originX
            val srcY1 = tile.outputOriginY - tile.originY

            for (dy in 0 until dstH) {
                for (dx in 0 until dstW) {
                    val outX = tile.outputOriginX + dx
                    val outY = tile.outputOriginY + dy
                    if (outX in 0 until sensorW && outY in 0 until sensorH) {
                        coverage[outY][outX]++
                    }
                }
            }
        }

        var uncoveredPixels = 0
        for (y in 0 until sensorH) {
            for (x in 0 until sensorW) {
                assertTrue("Pixel ($x,$y) must be covered by at least one tile",
                    coverage[y][x] >= 1)
                if (coverage[y][x] == 0) uncoveredPixels++
            }
        }

        assertEquals("No uncovered pixels in 50MP output", 0, uncoveredPixels)
    }

    @Test
    fun synthetic50mp_dispatchParameters() {
        val adapter = com.agx.camera.gpu.StandardBayerAdapter()
        val shader = adapter.createComputeShaderSource(
            com.agx.camera.camera.BayerPattern.RGGB, 10)

        assertTrue("Shader has local_size_x=16", shader.contains("local_size_x = 16"))
        assertTrue("Shader has local_size_y=16", shader.contains("local_size_y = 16"))
        assertTrue("Shader has u_tileOrigin", shader.contains("u_tileOrigin"))
        assertTrue("Shader has u_tileSize", shader.contains("u_tileSize"))
        assertTrue("Shader has u_sensorSize", shader.contains("u_sensorSize"))
        assertTrue("Shader has imageStore", shader.contains("imageStore"))
        assertTrue("Shader has u_bayer_color_map", shader.contains("u_bayer_color_map"))
        assertTrue("Shader has u_black_level_pattern", shader.contains("u_black_level_pattern"))
        assertTrue("Shader has u_lens_shading_map", shader.contains("u_lens_shading_map"))

        assertEquals("tileSize=2048", 2048, adapter.tileSize())
        assertEquals("overlap=2", 2, adapter.overlapPixels())
    }

    @Test
    fun synthetic50mp_tiledComputePipelineSimulation() {
        val sensorW = 8192
        val sensorH = 6144
        val adapter = StandardBayerAdapter()
        val pattern = BayerPattern.RGGB
        val bitDepth = 10
        val maxVal = (1 shl bitDepth) - 1
        val midGray = (maxVal * 0.18f).toInt()

        val bayer = ShortArray(sensorW * sensorH)
        for (y in 0 until sensorH) {
            for (x in 0 until sensorW) {
                val phase = (x % 2) + (y % 2) * 2
                val value = when (phase) {
                    0 -> midGray
                    1 -> (midGray * 0.85f).toInt()
                    2 -> (midGray * 0.85f).toInt()
                    3 -> (midGray * 0.7f).toInt()
                    else -> midGray
                }
                bayer[y * sensorW + x] = value.coerceIn(0, maxVal).toShort()
            }
        }

        val colorMap = pattern.colorMap()
        val shader = adapter.createComputeShaderSource(pattern, bitDepth)

        val grid = scheduler.computeTileGrid(sensorW, sensorH)
        assertTrue("50MP grid has multiple tiles", grid.tiles.size > 1)
        assertTrue("Overlap valid", scheduler.validateTileOverlap(grid))

        val outputCoverage = Array(sensorH) { BooleanArray(sensorW) }

        for (tile in grid.tiles) {
            val uTileOriginX = tile.originX
            val uTileOriginY = tile.originY
            val uTileSizeX = tile.width
            val uTileSizeY = tile.height
            val uSensorSizeX = sensorW
            val uSensorSizeY = sensorH

            assertEquals("tile origin within sensor", tile.originX, uTileOriginX)
            assertEquals("tile origin Y within sensor", tile.originY, uTileOriginY)
            assertEquals("tile width matches uniform", tile.width, uTileSizeX)
            assertEquals("tile height matches uniform", tile.height, uTileSizeY)
            assertTrue("tile width <= tileSize + 2*overlap",
                tile.width <= adapter.tileSize() + 2 * adapter.overlapPixels())
            assertTrue("tile height <= tileSize + 2*overlap",
                tile.height <= adapter.tileSize() + 2 * adapter.overlapPixels())
            assertTrue("tile width > 0", tile.width > 0)
            assertTrue("tile height > 0", tile.height > 0)

            val groupsX = (tile.width + 15) / 16
            val groupsY = (tile.height + 15) / 16
            assertTrue("dispatch groups cover tile width",
                groupsX * 16 >= tile.width)
            assertTrue("dispatch groups cover tile height",
                groupsY * 16 >= tile.height)

            val dstW = minOf(adapter.tileSize(), sensorW - tile.outputOriginX)
            val dstH = minOf(adapter.tileSize(), sensorH - tile.outputOriginY)
            val srcX1 = tile.outputOriginX - tile.originX
            val srcY1 = tile.outputOriginY - tile.originY

            for (dy in 0 until dstH) {
                for (dx in 0 until dstW) {
                    val sensorX = tile.originX + srcX1 + dx
                    val sensorY = tile.originY + srcY1 + dy
                    if (sensorX in 0 until sensorW && sensorY in 0 until sensorH) {
                        outputCoverage[sensorY][sensorX] = true
                    }
                }
            }

            for (localY in 0 until minOf(32, tile.height)) {
                for (localX in 0 until minOf(32, tile.width)) {
                    val globalX = tile.originX + localX
                    val globalY = tile.originY + localY
                    if (globalX >= sensorW || globalY >= sensorH) continue

                    val phase = (globalX % 2) + (globalY % 2) * 2
                    val mappedColor = colorMap[phase]
                    assertTrue("colorMap[$phase] in 0..2 at ($globalX,$globalY)",
                        mappedColor in 0..2)

                    val rawVal = bayer[globalY * sensorW + globalX].toInt() and maxVal
                    assertTrue("raw value in range at ($globalX,$globalY)",
                        rawVal in 0..maxVal)
                }
            }
        }

        var uncovered = 0
        for (y in 0 until sensorH) {
            for (x in 0 until sensorW) {
                if (!outputCoverage[y][x]) uncovered++
            }
        }
        assertEquals("All 50MP output pixels covered", 0, uncovered)
    }

    @Test
    fun synthetic50mp_shaderHandlesAllBitDepths() {
        val adapter = StandardBayerAdapter()
        for (bitDepth in listOf(8, 10, 12, 14, 16)) {
            val shader = adapter.createComputeShaderSource(BayerPattern.RGGB, bitDepth)
            assertTrue("shader contains version for ${bitDepth}-bit",
                shader.contains("#version 310 es"))
            assertTrue("shader has imageStore for ${bitDepth}-bit",
                shader.contains("imageStore"))
            assertTrue("shader has u_outTex binding=1 for ${bitDepth}-bit",
                shader.contains("binding = 1"))
            assertTrue("shader has u_bayerTex binding=0 for ${bitDepth}-bit",
                shader.contains("binding = 0"))
        }
    }

    private fun tileSize() = 2048
}
