package com.agx.camera.camera

import android.util.Log
import java.nio.ByteBuffer
import kotlin.math.abs

class GrayCardSampler {

    data class Gains(val gainR: Float, val gainG: Float, val gainB: Float)

    var isActive = false
        private set

    var onSampleComplete: ((Gains) -> Unit)? = null

    fun activate() {
        isActive = true
        Log.d(TAG, "Gray card sampler activated — tap preview to sample")
    }

    fun deactivate() {
        isActive = false
    }

    fun sample(
        yPlane: ByteBuffer, uPlane: ByteBuffer, vPlane: ByteBuffer,
        yuvWidth: Int, yuvHeight: Int,
        tapX: Float, tapY: Float,
        previewWidth: Int, previewHeight: Int
    ) {
        if (!isActive) return

        val sampleSize = 64
        val scaleX = yuvWidth.toFloat() / previewWidth
        val scaleY = yuvHeight.toFloat() / previewHeight

        val cx = (tapX * scaleX).toInt().coerceIn(sampleSize / 2, yuvWidth - sampleSize / 2)
        val cy = (tapY * scaleY).toInt().coerceIn(sampleSize / 2, yuvHeight - sampleSize / 2)
        val x0 = cx - sampleSize / 2
        val y0 = cy - sampleSize / 2

        var sumR = 0.0; var sumG = 0.0; var sumB = 0.0
        var count = 0

        for (dy in 0 until sampleSize) {
            val yRow = y0 + dy
            if (yRow < 0 || yRow >= yuvHeight) continue
            val uvRow = yRow / 2

            for (dx in 0 until sampleSize) {
                val xPos = x0 + dx
                if (xPos < 0 || xPos >= yuvWidth) continue
                val uvCol = xPos / 2

                val yVal = (yPlane.get(yRow * yuvWidth + xPos).toInt() and 0xFF) / 255.0
                val uVal = (uPlane.get(uvRow * (yuvWidth / 2) + uvCol).toInt() and 0xFF) / 255.0 - 0.5
                val vVal = (vPlane.get(uvRow * (yuvWidth / 2) + uvCol).toInt() and 0xFF) / 255.0 - 0.5

                val r = (yVal + 1.402 * vVal).coerceIn(0.0, 1.0)
                val g = (yVal - 0.344136 * uVal - 0.714136 * vVal).coerceIn(0.0, 1.0)
                val b = (yVal + 1.772 * uVal).coerceIn(0.0, 1.0)

                sumR += r; sumG += g; sumB += b
                count++
            }
        }

        if (count == 0) return

        val avgR = (sumR / count).coerceAtLeast(0.001)
        val avgG = (sumG / count).coerceAtLeast(0.001)
        val avgB = (sumB / count).coerceAtLeast(0.001)

        val gainR = (avgG / avgR).toFloat().coerceIn(0.1f, 10.0f)
        val gainG = 1.0f
        val gainB = (avgG / avgB).toFloat().coerceIn(0.1f, 10.0f)

        Log.d(TAG, "Gray card sample: avgR=$avgR avgG=$avgG avgB=$avgB → gains=($gainR, $gainG, $gainB)")

        isActive = false
        onSampleComplete?.invoke(Gains(gainR, gainG, gainB))
    }

    companion object {
        private const val TAG = "GrayCardSampler"
    }
}
