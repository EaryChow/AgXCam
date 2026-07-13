package com.agx.camera.io

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.opengl.GLES20
import android.util.Log
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object JpegEncoder {

    private const val TAG = "JpegEncoder"
    private var nativeAvailable: Boolean? = null

    init {
        try {
            System.loadLibrary("jpeg_encoder")
            nativeAvailable = true
            Log.d(TAG, "Native libjpeg-turbo encoder loaded")
        } catch (e: UnsatisfiedLinkError) {
            nativeAvailable = false
            Log.d(TAG, "Native encoder unavailable, using Bitmap.compress fallback")
        }
    }

    @JvmStatic
    private external fun nativeEncodeRgba(buffer: ByteBuffer, width: Int, height: Int, quality: Int): ByteArray?

    fun yuvImageToBitmap(image: Image): Bitmap {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val width = image.width
        val height = image.height

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val nv21 = ByteArray(width * height * 3 / 2)
        var pos = 0

        for (row in 0 until height) {
            val yOffset = row * yRowStride
            for (col in 0 until width) {
                nv21[pos++] = yBuffer.get(yOffset + col)
            }
        }

        val uvHeight = height / 2
        val uvWidth = width / 2
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                val uvOffset = row * uvRowStride + col * uvPixelStride
                nv21[pos++] = vBuffer.get(uvOffset)
                nv21[pos++] = uBuffer.get(uvOffset)
            }
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val bytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    fun encodeToJpeg(bitmap: Bitmap, quality: Int): ByteArray {
        if (nativeAvailable == true) {
            val buffer = ByteBuffer.allocateDirect(bitmap.width * bitmap.height * 4).order(ByteOrder.nativeOrder())
            bitmap.copyPixelsToBuffer(buffer)
            buffer.rewind()
            val result = nativeEncodeRgba(buffer, bitmap.width, bitmap.height, quality)
            if (result != null) return result
            Log.w(TAG, "Native encode returned null, falling back to Bitmap.compress")
        }
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, stream)
        return stream.toByteArray()
    }

    fun readFboToBitmapFlipped(fboTextureId: Int, width: Int, height: Int): Bitmap {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTextureId)

        val buffer = ByteBuffer.allocateDirect(width * height * 4).order(ByteOrder.nativeOrder())
        GLES20.glPixelStorei(GLES20.GL_PACK_ALIGNMENT, 1)
        GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buffer)
        GLES20.glPixelStorei(GLES20.GL_PACK_ALIGNMENT, 4)
        buffer.rewind()

        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            val flippedY = height - 1 - y
            buffer.position((flippedY * width) * 4)
            for (x in 0 until width) {
                val r = buffer.get().toInt() and 0xFF
                val g = buffer.get().toInt() and 0xFF
                val b = buffer.get().toInt() and 0xFF
                val a = buffer.get().toInt() and 0xFF
                pixels[y * width + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
