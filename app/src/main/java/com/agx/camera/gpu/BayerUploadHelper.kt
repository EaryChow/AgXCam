package com.agx.camera.gpu

import android.hardware.HardwareBuffer
import android.media.Image
import android.opengl.EGL14
import android.opengl.EGLDisplay
import android.opengl.GLES20
import android.opengl.GLES30
import android.os.Build
import android.util.Log
import java.lang.reflect.Method
import java.nio.ByteBuffer
import java.nio.ByteOrder

class BayerUploadHelper(private val eglDisplay: EGLDisplay) {

    enum class UploadPath {
        ZERO_COPY,
        CPU_COPY,
        FAILED
    }

    var activePath: UploadPath = UploadPath.FAILED
        private set

    var zeroCopySupported: Boolean = false
        private set

    private var eglImageKHR: Long = 0L
    private var pendingImage: Image? = null
    private var cpuStagingBuffer: ByteBuffer? = null
    private var lastFrameWidth = 0
    private var lastFrameHeight = 0
    private var grallocLogged = false

    private var eglCreateImageKHRMethod: Method? = null
    private var eglDestroyImageKHRMethod: Method? = null
    private var glEGLImageTargetTexture2DOESMethod: Method? = null
    private var eglNoImageKHRValue: Long = 0L
    private var eglNativeBufferAndroidValue: Int = 0

    fun probeZeroCopySupport(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            zeroCopySupported = false
            return false
        }

        try {
            val glExtensions = GLES20.glGetString(0x1F03)
            val hasGLOesEglImage = glExtensions?.contains("GL_OES_EGL_image") == true

            if (hasGLOesEglImage) {
                initEglExtMethods()
                zeroCopySupported = eglCreateImageKHRMethod != null
            } else {
                zeroCopySupported = false
            }
        } catch (e: Exception) {
            Log.w(TAG, "Zero-copy probe failed: ${e.message}")
            zeroCopySupported = false
        }

        Log.d(TAG, "Zero-copy support: $zeroCopySupported")
        return zeroCopySupported
    }

    private fun initEglExtMethods() {
        try {
            val eglExtClass = Class.forName("android.opengl.EGLExt")

            eglCreateImageKHRMethod = eglExtClass.getMethod(
                "eglCreateImageKHR",
                EGLDisplay::class.java, Long::class.javaPrimitiveType,
                Int::class.javaPrimitiveType, Long::class.javaPrimitiveType, IntArray::class.java
            )

            eglDestroyImageKHRMethod = eglExtClass.getMethod(
                "eglDestroyImageKHR",
                EGLDisplay::class.java, Long::class.javaPrimitiveType
            )

            glEGLImageTargetTexture2DOESMethod = eglExtClass.getMethod(
                "glEGLImageTargetTexture2DOES",
                Int::class.javaPrimitiveType, Long::class.javaPrimitiveType
            )

            val noImageField = EGL14::class.java.getField("EGL_NO_IMAGE_KHR")
            eglNoImageKHRValue = noImageField.getLong(null)

            val nativeBufferField = eglExtClass.getField("EGL_NATIVE_BUFFER_ANDROID")
            eglNativeBufferAndroidValue = nativeBufferField.getInt(null)

            Log.d(TAG, "EGLExt methods initialized")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to init EGLExt methods: ${e.message}")
            eglCreateImageKHRMethod = null
        }
    }

    fun uploadFrame(
        image: Image,
        targetTextureId: Int,
        sensorWidth: Int,
        sensorHeight: Int
    ): UploadPath {
        pendingImage = image

        if (zeroCopySupported) {
            val result = tryZeroCopyUpload(image, targetTextureId, sensorWidth, sensorHeight)
            if (result == UploadPath.ZERO_COPY) {
                activePath = UploadPath.ZERO_COPY
                return result
            }
            Log.w(TAG, "Zero-copy upload failed, falling back to CPU copy")
        }

        val result = tryCpuCopyUpload(image, targetTextureId, sensorWidth, sensorHeight)
        activePath = if (result == UploadPath.CPU_COPY) result else UploadPath.FAILED
        return result
    }

    private fun tryZeroCopyUpload(
        image: Image,
        targetTextureId: Int,
        sensorWidth: Int,
        sensorHeight: Int
    ): UploadPath {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return UploadPath.FAILED
        val method = eglCreateImageKHRMethod ?: return UploadPath.FAILED

        val hardwareBuffer = try {
            @Suppress("NewApi")
            image.hardwareBuffer
        } catch (e: Exception) {
            Log.w(TAG, "hardwareBuffer() failed: ${e.message}")
            return UploadPath.FAILED
        }

        if (hardwareBuffer == null) {
            Log.w(TAG, "hardwareBuffer null for RAW_SENSOR")
            return UploadPath.FAILED
        }

        if (!grallocLogged) {
            logGrallocInfo(hardwareBuffer)
            grallocLogged = true
        }

        try {
            @Suppress("NewApi")
            eglImageKHR = method.invoke(
                null, eglDisplay, EGL14.EGL_NO_CONTEXT,
                eglNativeBufferAndroidValue, hardwareBuffer, intArrayOf(EGL14.EGL_NONE)
            ) as Long

            if (eglImageKHR == eglNoImageKHRValue || eglImageKHR == 0L) {
                Log.w(TAG, "eglCreateImageKHR failed: 0x${Integer.toHexString(EGL14.eglGetError())}")
                @Suppress("NewApi")
                hardwareBuffer.close()
                return UploadPath.FAILED
            }

            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, targetTextureId)
            glEGLImageTargetTexture2DOESMethod?.invoke(null, GLES20.GL_TEXTURE_2D, eglImageKHR)

            val glError = GLES20.glGetError()
            if (glError != GLES20.GL_NO_ERROR) {
                Log.w(TAG, "glEGLImageTargetTexture2DOES failed: 0x${Integer.toHexString(glError)}")
                destroyEglImage()
                @Suppress("NewApi")
                hardwareBuffer.close()
                return UploadPath.FAILED
            }

            lastFrameWidth = sensorWidth
            lastFrameHeight = sensorHeight
            Log.d(TAG, "Zero-copy upload succeeded: ${sensorWidth}x${sensorHeight}")
            return UploadPath.ZERO_COPY
        } catch (e: Exception) {
            Log.w(TAG, "Zero-copy upload exception: ${e.message}")
            destroyEglImage()
            @Suppress("NewApi")
            hardwareBuffer.close()
            return UploadPath.FAILED
        }
    }

    private fun tryCpuCopyUpload(
        image: Image,
        targetTextureId: Int,
        sensorWidth: Int,
        sensorHeight: Int
    ): UploadPath {
        val plane = image.planes.firstOrNull() ?: return UploadPath.FAILED
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val bytesPerPixel = 2
        val stridePixels = rowStride / bytesPerPixel

        val bufferSize = buffer.remaining()
        val expectedSize = sensorWidth * sensorHeight * bytesPerPixel
        if (bufferSize < expectedSize / 2) {
            Log.w(TAG, "Buffer too small: $bufferSize bytes, expected ~$expectedSize")
            return UploadPath.FAILED
        }

        val staging = ensureStagingBuffer(expectedSize)
        staging.position(0)
        staging.put(buffer)

        GLES30.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, targetTextureId)

        if (stridePixels > sensorWidth) {
            GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, stridePixels)
        }

        staging.position(0)
        GLES30.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_R16UI,
            sensorWidth, sensorHeight, 0,
            GLES30.GL_RED_INTEGER, GLES20.GL_UNSIGNED_SHORT, staging
        )

        if (stridePixels > sensorWidth) {
            GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, 0)
        }

        val glError = GLES20.glGetError()
        if (glError != GLES20.GL_NO_ERROR) {
            Log.w(TAG, "CPU copy glTexImage2D failed: 0x${Integer.toHexString(glError)}")
            return UploadPath.FAILED
        }

        lastFrameWidth = sensorWidth
        lastFrameHeight = sensorHeight
        Log.d(TAG, "CPU copy upload succeeded: ${sensorWidth}x${sensorHeight}, stride=$stridePixels")
        return UploadPath.CPU_COPY
    }

    fun releaseFrame() {
        destroyEglImage()
        insertFenceSync()
        pendingImage?.close()
        pendingImage = null
    }

    private fun insertFenceSync() {
        try {
            val fence = GLES30.glFenceSync(GLES30.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
            if (fence != 0L) {
                val result = GLES30.glClientWaitSync(fence, GLES30.GL_SYNC_FLUSH_COMMANDS_BIT, 10_000_000L)
                GLES30.glDeleteSync(fence)
                if (result == GLES30.GL_TIMEOUT_EXPIRED) {
                    Log.w(TAG, "Fence sync timed out, falling back to glFinish")
                    GLES20.glFinish()
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Fence sync failed, falling back to glFinish: ${e.message}")
            try { GLES20.glFinish() } catch (_: Exception) {}
        }
    }

    fun finishFrame() {
        try {
            GLES20.glFinish()
        } catch (e: Exception) {
            Log.w(TAG, "glFinish failed: ${e.message}")
        }
    }

    fun destroy() {
        releaseFrame()
        cpuStagingBuffer = null
    }

    private fun destroyEglImage() {
        if (eglImageKHR != eglNoImageKHRValue && eglImageKHR != 0L) {
            try {
                eglDestroyImageKHRMethod?.invoke(null, eglDisplay, eglImageKHR)
            } catch (e: Exception) {
                Log.w(TAG, "eglDestroyImageKHR failed: ${e.message}")
            }
            eglImageKHR = 0L
        }
    }

    @Suppress("NewApi")
    private fun logGrallocInfo(hb: HardwareBuffer) {
        try {
            val formatField = HardwareBuffer::class.java.getMethod("getFormat")
            val format = formatField.invoke(hb) as Int
            val widthField = HardwareBuffer::class.java.getMethod("getWidth")
            val w = widthField.invoke(hb) as Int
            val heightField = HardwareBuffer::class.java.getMethod("getHeight")
            val h = heightField.invoke(hb) as Int
            val usageField = HardwareBuffer::class.java.getMethod("getUsage")
            val usage = usageField.invoke(hb) as Long
            Log.i(TAG, "AHardwareBuffer_describe: " +
                    "format=0x${Integer.toHexString(format)}, " +
                    "width=$w, height=$h, " +
                    "usage=0x${java.lang.Long.toHexString(usage)}")
        } catch (e: Exception) {
            Log.w(TAG, "AHardwareBuffer_describe logging failed: ${e.message}")
        }
    }

    private fun ensureStagingBuffer(size: Int): ByteBuffer {
        val existing = cpuStagingBuffer
        if (existing != null && existing.capacity() >= size) {
            return existing
        }
        val buf = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
        cpuStagingBuffer = buf
        Log.d(TAG, "Allocated CPU staging buffer: $size bytes")
        return buf
    }

    companion object {
        private const val TAG = "BayerUploadHelper"
    }
}
