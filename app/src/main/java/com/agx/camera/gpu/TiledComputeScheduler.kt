package com.agx.camera.gpu

import android.opengl.GLES20
import android.opengl.GLES30
import android.util.Log

private const val GL_HALF_FLOAT = 0x140B

class TiledComputeScheduler(
    private val tileSize: Int = 2048,
    private val overlap: Int = 2
) {

    data class Tile(
        val originX: Int,
        val originY: Int,
        val width: Int,
        val height: Int,
        val outputOriginX: Int,
        val outputOriginY: Int
    )

    data class TileGrid(
        val tiles: List<Tile>,
        val totalWidth: Int,
        val totalHeight: Int,
        val tilesX: Int,
        val tilesY: Int
    )

    fun computeTileGrid(
        sensorWidth: Int,
        sensorHeight: Int,
        cropX: Int = 0,
        cropY: Int = 0,
        cropWidth: Int = sensorWidth,
        cropHeight: Int = sensorHeight
    ): TileGrid {
        val tiles = mutableListOf<Tile>()
        var tilesX = 0
        var tilesY = 0

        var y = cropY
        while (y < cropY + cropHeight) {
            var x = cropX
            tilesX = 0
            while (x < cropX + cropWidth) {
                val tileW = minOf(tileSize, cropX + cropWidth - x)
                val tileH = minOf(tileSize, cropY + cropHeight - y)

                val originX = maxOf(0, x - overlap)
                val originY = maxOf(0, y - overlap)

                val readW = minOf(tileW + 2 * overlap, sensorWidth - originX)
                val readH = minOf(tileH + 2 * overlap, sensorHeight - originY)

                tiles.add(Tile(
                    originX = originX,
                    originY = originY,
                    width = readW,
                    height = readH,
                    outputOriginX = x - cropX,
                    outputOriginY = y - cropY
                ))

                tilesX++
                x += tileSize
            }
            tilesY++
            y += tileSize
        }

        Log.d(TAG, "Tile grid: ${tiles.size} tiles (${tilesX}x${tilesY}), " +
                "sensor=${sensorWidth}x${sensorHeight}, crop=${cropWidth}x${cropHeight}")

        return TileGrid(tiles, cropWidth, cropHeight, tilesX, tilesY)
    }

    fun estimatePeakGpuMemory(grid: TileGrid): Long {
        val tilePixels = tileSize.toLong() * tileSize.toLong()
        val intermediateBytes = tilePixels * 8L
        val outputBytes = grid.totalWidth.toLong() * grid.totalHeight.toLong() * 8L
        return intermediateBytes + outputBytes
    }

    fun validateTileOverlap(grid: TileGrid): Boolean {
        for (i in grid.tiles.indices) {
            val tile = grid.tiles[i]
            if (tile.width <= 0 || tile.height <= 0) {
                Log.e(TAG, "Invalid tile $i: ${tile.width}x${tile.height}")
                return false
            }
            if (tile.originX < 0 || tile.originY < 0) {
                Log.e(TAG, "Negative origin for tile $i: (${tile.originX}, ${tile.originY})")
                return false
            }
            if (tile.outputOriginX < 0 || tile.outputOriginY < 0) {
                Log.e(TAG, "Negative output origin for tile $i")
                return false
            }
        }

        for (i in 0 until grid.tiles.size - 1) {
            val current = grid.tiles[i]
            val next = grid.tiles[i + 1]
            if (current.outputOriginX + tileSize <= next.outputOriginX &&
                current.outputOriginY + tileSize <= next.outputOriginY) {
                continue
            }
            if (current.outputOriginX + tileSize > next.outputOriginX &&
                current.outputOriginY + tileSize > next.outputOriginY) {
                val overlapX = current.outputOriginX + tileSize - next.outputOriginX
                val overlapY = current.outputOriginY + tileSize - next.outputOriginY
                if (overlapX < 2 * overlap || overlapY < 2 * overlap) {
                    Log.e(TAG, "Insufficient overlap between tiles $i and ${i+1}: " +
                            "${overlapX}x${overlapY} (need ${2*overlap}x${2*overlap})")
                    return false
                }
            }
        }

        for (i in grid.tiles.indices) {
            val vertIdx = i + grid.tilesX
            if (vertIdx >= grid.tiles.size) continue
            val current = grid.tiles[i]
            val below = grid.tiles[vertIdx]
            if (current.outputOriginY + tileSize > below.outputOriginY &&
                current.outputOriginX + tileSize > below.outputOriginX) {
                val overlapX = current.outputOriginX + tileSize - below.outputOriginX
                val overlapY = current.outputOriginY + tileSize - below.outputOriginY
                if (overlapX < 2 * overlap || overlapY < 2 * overlap) {
                    Log.e(TAG, "Insufficient vertical overlap between tiles $i and $vertIdx: " +
                            "${overlapX}x${overlapY} (need ${2*overlap}x${2*overlap})")
                    return false
                }
            }
        }

        return true
    }

    fun executeTiledPipeline(
        bayerTextureId: Int,
        outputFboId: Int,
        outputTextureId: Int,
        outputWidth: Int,
        outputHeight: Int,
        grid: TileGrid,
        dispatchTile: (Tile, Int, Int) -> Unit,
        onTileComplete: ((Int, Int) -> Unit)? = null
    ) {
        var tilesProcessed = 0

        for (tile in grid.tiles) {
            val tileFbo = createTileFbo(tile.width, tile.height)
            if (tileFbo == null) {
                Log.e(TAG, "Failed to create tile FBO for tile at (${tile.originX}, ${tile.originY})")
                continue
            }

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, tileFbo.first)
            GLES30.glViewport(0, 0, tile.width, tile.height)

            dispatchTile(tile, tileFbo.first, tileFbo.second)

            GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, tileFbo.first)
            GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, outputFboId)

            val srcX1 = tile.outputOriginX - tile.originX
            val srcY1 = tile.outputOriginY - tile.originY
            val dstW = minOf(tileSize, outputWidth - tile.outputOriginX)
            val dstH = minOf(tileSize, outputHeight - tile.outputOriginY)
            val srcX2 = srcX1 + dstW
            val srcY2 = srcY1 + dstH

            GLES30.glBlitFramebuffer(
                srcX1, srcY1, srcX2, srcY2,
                tile.outputOriginX, tile.outputOriginY,
                tile.outputOriginX + dstW, tile.outputOriginY + dstH,
                GLES20.GL_COLOR_BUFFER_BIT, GLES20.GL_LINEAR
            )

            GLES30.glDeleteFramebuffers(1, intArrayOf(tileFbo.first), 0)
            GLES30.glDeleteTextures(1, intArrayOf(tileFbo.second), 0)

            tilesProcessed++
            onTileComplete?.invoke(tilesProcessed, grid.tiles.size)
        }

        Log.d(TAG, "Tiled pipeline complete: $tilesProcessed/${grid.tiles.size} tiles")
    }

    private fun createTileFbo(width: Int, height: Int): Pair<Int, Int>? {
        val texBuf = IntArray(1)
        GLES30.glGenTextures(1, texBuf, 0)
        val texId = texBuf[0]

        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, texId)
        GLES30.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F,
            width, height, 0,
            GLES30.GL_RGBA, GL_HALF_FLOAT, null
        )
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        val fboBuf = IntArray(1)
        GLES30.glGenFramebuffers(1, fboBuf, 0)
        val fboId = fboBuf[0]

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, texId, 0
        )

        val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
        if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "Tile FBO incomplete: $status")
            GLES30.glDeleteFramebuffers(1, intArrayOf(fboId), 0)
            GLES30.glDeleteTextures(1, intArrayOf(texId), 0)
            return null
        }

        return Pair(fboId, texId)
    }

    companion object {
        private const val TAG = "TiledComputeScheduler"
        const val DEFAULT_TILE_SIZE = 2048
        const val DEFAULT_OVERLAP = 2
    }
}
