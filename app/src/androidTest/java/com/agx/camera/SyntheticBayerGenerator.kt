package com.agx.camera

import android.graphics.Bitmap
import android.graphics.Color

class SyntheticBayerGenerator(private val width: Int, private val height: Int) {

    private var frameIndex = 0

    fun generateFrame(): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val phase = (x % 2) + (y % 2) * 2
                val nx = x.toFloat() / width
                val ny = y.toFloat() / height

                val t = frameIndex / 30.0f
                val wave = (Math.sin((nx * 4 + t).toDouble()) * 0.3 + 0.5).toFloat()
                val checker = ((x / 4 + y / 4) % 2).toFloat()

                val r: Float
                val g: Float
                val b: Float

                when (phase) {
                    0 -> { // R
                        r = wave * 0.8f + checker * 0.2f
                        g = 0f
                        b = 0f
                    }
                    1 -> { // Gr
                        r = 0f
                        g = wave * 0.7f + checker * 0.3f
                        b = 0f
                    }
                    2 -> { // Gb
                        r = 0f
                        g = wave * 0.7f + checker * 0.3f
                        b = 0f
                    }
                    3 -> { // B
                        r = 0f
                        g = 0f
                        b = wave * 0.6f + checker * 0.2f
                    }
                    else -> { r = 0f; g = 0f; b = 0f }
                }

                val ri = (r.coerceIn(0f, 1f) * 255).toInt()
                val gi = (g.coerceIn(0f, 1f) * 255).toInt()
                val bi = (b.coerceIn(0f, 1f) * 255).toInt()
                pixels[y * width + x] = Color.rgb(ri, gi, bi)
            }
        }

        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        frameIndex++
        return bitmap
    }

    fun generateGrayCard(kelvin: Float = 5500f): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val midGray = 0.18f

        val (rGain, gGain, bGain) = if (kelvin < 5600f && kelvin > 3200f) {
            val warmFactor = ((5600f - kelvin) / 2400f).coerceIn(0f, 1f)
            Triple(1.0f + warmFactor * 0.3f, 1.0f, 1.0f - warmFactor * 0.2f)
        } else if (kelvin >= 5600f) {
            val coolFactor = ((kelvin - 5600f) / 4400f).coerceIn(0f, 1f)
            Triple(1.0f - coolFactor * 0.15f, 1.0f, 1.0f + coolFactor * 0.25f)
        } else {
            Triple(1.0f, 1.0f, 1.0f)
        }

        val ri = (midGray * rGain * 255).toInt().coerceIn(0, 255)
        val gi = (midGray * gGain * 255).toInt().coerceIn(0, 255)
        val bi = (midGray * bGain * 255).toInt().coerceIn(0, 255)
        val color = Color.rgb(ri, gi, bi)
        val pixels = IntArray(width * height) { color }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    fun generateColorChart(): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val colors = intArrayOf(
            Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW,
            Color.CYAN, Color.MAGENTA, Color.WHITE, Color.rgb(128, 128, 128)
        )
        val cellW = width / 4
        val cellH = height / 2
        val pixels = IntArray(width * height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val cellX = (x / cellW).coerceAtMost(3)
                val cellY = (y / cellH).coerceAtMost(1)
                val idx = cellY * 4 + cellX
                pixels[y * width + x] = colors[idx]
            }
        }

        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
