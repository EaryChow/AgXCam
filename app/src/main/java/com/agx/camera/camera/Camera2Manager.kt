package com.agx.camera.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size

class Camera2Manager(private val context: Context) {

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null

    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    private var currentFlashMode: FlashMode = FlashMode.OFF
    private var currentLens: LensInfo? = null

    var onFrameAvailable: ((android.media.Image) -> Unit)? = null
    var onSessionReady: ((Int, Int) -> Unit)? = null
    var onError: ((String) -> Unit)? = null

    private val imageListener = ImageReader.OnImageAvailableListener { reader ->
        val image = reader.acquireLatestImage() ?: return@OnImageAvailableListener
        try {
            onFrameAvailable?.invoke(image)
        } finally {
            image.close()
        }
    }

    fun startBackgroundThread() {
        backgroundThread = HandlerThread("Camera2Background").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
        } catch (_: InterruptedException) {}
        backgroundThread = null
        backgroundHandler = null
    }

    @SuppressLint("MissingPermission")
    fun openCamera(lens: LensInfo, previewSize: Size, onError: ((String) -> Unit)? = null) {
        currentLens = lens
        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager

        Log.d(TAG, "Opening camera ${lens.cameraId}, preview size: ${previewSize.width}x${previewSize.height}")

        imageReader = ImageReader.newInstance(
            previewSize.width, previewSize.height,
            ImageFormat.YUV_420_888, 3
        ).apply {
            setOnImageAvailableListener(imageListener, backgroundHandler)
        }

        cameraManager.openCamera(lens.cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                createSession(camera, previewSize)
            }

            override fun onDisconnected(camera: CameraDevice) {
                Log.w(TAG, "Camera disconnected")
                camera.close()
                cameraDevice = null
                onError?.invoke("Camera disconnected by another app")
            }

            override fun onError(camera: CameraDevice, error: Int) {
                Log.e(TAG, "Camera error: $error")
                camera.close()
                cameraDevice = null
                onError?.invoke("Camera error: $error")
            }
        }, backgroundHandler)
    }

    private fun createSession(camera: CameraDevice, previewSize: Size) {
        val surfaces = listOf(imageReader!!.surface)

        @Suppress("DEPRECATION")
        camera.createCaptureSession(surfaces, object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                captureSession = session
                onSessionReady?.invoke(previewSize.width, previewSize.height)
                startPreview()
            }

            override fun onConfigureFailed(session: CameraCaptureSession) {
                Log.e(TAG, "Session configuration failed")
                onError?.invoke("Camera session configuration failed")
            }
        }, backgroundHandler)
    }

    private fun startPreview() {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(imageReader!!.surface)
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            currentFlashMode.applyToRequest(this)
        }

        session.setRepeatingRequest(request.build(), null, backgroundHandler)
        Log.d(TAG, "Preview started with flash mode: $currentFlashMode")
    }

    fun setFlashMode(mode: FlashMode) {
        currentFlashMode = mode
        startPreview()
    }

    fun close() {
        try {
            captureSession?.close()
        } catch (_: Exception) {}
        captureSession = null

        try {
            cameraDevice?.close()
        } catch (_: Exception) {}
        cameraDevice = null

        try {
            imageReader?.close()
        } catch (_: Exception) {}
        imageReader = null
    }

    companion object {
        private const val TAG = "Camera2Manager"
    }
}
