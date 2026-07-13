package com.agx.camera.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.hardware.camera2.params.MeteringRectangle
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size

class Camera2Manager(private val context: Context) {

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var previewReader: ImageReader? = null
    private var captureReader: ImageReader? = null

    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    private var currentFlashMode: FlashMode = FlashMode.OFF
    private var currentLens: LensInfo? = null
    var captureSize: Size = Size(640, 480)
        private set

    var onFrameAvailable: ((Image) -> Unit)? = null
    var onSessionReady: ((Int, Int) -> Unit)? = null
    var onError: ((String) -> Unit)? = null
    var onDisconnected: (() -> Unit)? = null

    private var sessionRetryCount = 0
    private val maxSessionRetries = 1

    private val previewListener = ImageReader.OnImageAvailableListener { reader ->
        val image = reader.acquireLatestImage() ?: return@OnImageAvailableListener
        try {
            onFrameAvailable?.invoke(image)
        } finally {
            image.close()
        }
    }

    val currentFlashModeForExif: FlashMode get() = currentFlashMode

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

        previewReader = ImageReader.newInstance(
            previewSize.width, previewSize.height,
            ImageFormat.YUV_420_888, 3
        ).apply {
            setOnImageAvailableListener(previewListener, backgroundHandler)
        }

        val chars = cameraManager.getCameraCharacteristics(lens.cameraId)
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val maxSize = map?.getOutputSizes(ImageFormat.YUV_420_888)
            ?.maxByOrNull { it.width.toLong() * it.height.toLong() }
            ?: previewSize
        captureSize = Size(maxSize.width, maxSize.height)

        Log.d(TAG, "Capture size: ${captureSize.width}x${captureSize.height}")

        captureReader = ImageReader.newInstance(
            captureSize.width, captureSize.height,
            ImageFormat.YUV_420_888, 1
        )

        sessionRetryCount = 0

        cameraManager.openCamera(lens.cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                createSession(camera, previewSize)
            }

            override fun onDisconnected(camera: CameraDevice) {
                Log.w(TAG, "Camera disconnected")
                camera.close()
                cameraDevice = null
                captureSession = null
                onDisconnected?.invoke()
            }

            override fun onError(camera: CameraDevice, error: Int) {
                Log.e(TAG, "Camera error: $error")
                camera.close()
                cameraDevice = null
                captureSession = null
                onError?.invoke("Camera error: $error")
            }
        }, backgroundHandler)
    }

    private fun createSession(camera: CameraDevice, previewSize: Size) {
        val surfaces = listOf(previewReader!!.surface, captureReader!!.surface)

        @Suppress("DEPRECATION")
        camera.createCaptureSession(surfaces, object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                captureSession = session
                sessionRetryCount = 0
                onSessionReady?.invoke(previewSize.width, previewSize.height)
                startPreview()
            }

            override fun onConfigureFailed(session: CameraCaptureSession) {
                Log.e(TAG, "Session configuration failed (attempt ${sessionRetryCount + 1})")
                if (sessionRetryCount < maxSessionRetries) {
                    sessionRetryCount++
                    val handler = backgroundHandler ?: return
                    handler.postDelayed({
                        val cam = cameraDevice ?: return@postDelayed
                        createSession(cam, previewSize)
                    }, 500)
                } else {
                    onError?.invoke("Camera session configuration failed")
                }
            }
        }, backgroundHandler)
    }

    private fun startPreview() {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
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

    fun startContinuousAf() {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            currentFlashMode.applyToRequest(this)
        }
        session.setRepeatingRequest(request.build(), null, backgroundHandler)
    }

    fun startAfAeLock(
        meteringRect: MeteringRectangle,
        handler: Handler,
        onCaptureStarted: () -> Unit,
        onCaptureCompleted: (TotalCaptureResult) -> Unit
    ) {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
            set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(meteringRect))
            set(CaptureRequest.CONTROL_AE_REGIONS, arrayOf(meteringRect))
            set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER_START)
            set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
            currentFlashMode.applyToRequest(this)
        }

        session.capture(request.build(), object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureStarted(session: CameraCaptureSession, request: CaptureRequest, timestamp: Long, frameNumber: Long) {
                onCaptureStarted()
            }

            override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                val afState = result.get(CaptureResult.CONTROL_AF_STATE)
                if (afState != null && afState != CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN) {
                    session.stopRepeating()
                    val lockRequest = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                        addTarget(previewReader!!.surface)
                        set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
                        set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(meteringRect))
                        set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_CANCEL)
                        set(CaptureRequest.CONTROL_AE_LOCK, true)
                        set(CaptureRequest.CONTROL_AWB_LOCK, true)
                        currentFlashMode.applyToRequest(this)
                    }
                    session.setRepeatingRequest(lockRequest.build(), null, backgroundHandler)
                    onCaptureCompleted(result)
                }
            }
        }, handler)
    }

    fun unlockAeAwb() {
        startPreview()
    }

    fun captureStill(onCaptureAvailable: (Image, TotalCaptureResult) -> Unit, onCaptureFailed: () -> Unit) {
        val camera = cameraDevice ?: run { onCaptureFailed(); return }
        val session = captureSession ?: run { onCaptureFailed(); return }

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(captureReader!!.surface)
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
            set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
            currentFlashMode.applyToRequest(this)
        }

        session.capture(request.build(), object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                val image = captureReader?.acquireLatestImage()
                if (image != null) {
                    onCaptureAvailable(image, result)
                } else {
                    Log.e(TAG, "Capture image not available at onCaptureCompleted")
                    onCaptureFailed()
                }
            }

            override fun onCaptureFailed(session: CameraCaptureSession, request: CaptureRequest, failure: CaptureFailure) {
                Log.e(TAG, "Capture failed: ${failure.reason}")
                onCaptureFailed()
            }
        }, backgroundHandler)
    }

    fun resetHard() {
        try {
            captureSession?.close()
        } catch (_: Exception) {}
        captureSession = null

        try {
            cameraDevice?.close()
        } catch (_: Exception) {}
        cameraDevice = null

        try {
            previewReader?.close()
        } catch (_: Exception) {}
        previewReader = null

        try {
            captureReader?.close()
        } catch (_: Exception) {}
        captureReader = null
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
            previewReader?.close()
        } catch (_: Exception) {}
        previewReader = null

        try {
            captureReader?.close()
        } catch (_: Exception) {}
        captureReader = null
    }

    companion object {
        private const val TAG = "Camera2Manager"
    }
}
