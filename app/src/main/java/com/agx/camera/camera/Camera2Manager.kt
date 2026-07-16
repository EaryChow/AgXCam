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
import com.agx.camera.CrashLogger

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

    private var availableAfModes: IntArray = intArrayOf()
    private var availableAeModes: IntArray = intArrayOf()

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
        CrashLogger.log(TAG, "openCamera id=${lens.cameraId} preview=${previewSize.width}x${previewSize.height} level=${lens.hardwareLevel} raw=${lens.hasRawSensor}")

        try {
            previewReader = ImageReader.newInstance(
                previewSize.width, previewSize.height,
                ImageFormat.YUV_420_888, 3
            ).apply {
                setOnImageAvailableListener(previewListener, backgroundHandler)
            }

            val chars = cameraManager.getCameraCharacteristics(lens.cameraId)
            availableAfModes = chars.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: intArrayOf()
            availableAeModes = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES) ?: intArrayOf()
            CrashLogger.log(TAG, "AF modes: ${availableAfModes.toList()}, AE modes: ${availableAeModes.toList()}")
            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val maxSize = map?.getOutputSizes(ImageFormat.YUV_420_888)
                ?.maxByOrNull { it.width.toLong() * it.height.toLong() }
                ?: previewSize
            captureSize = Size(maxSize.width, maxSize.height)

            Log.d(TAG, "Capture size: ${captureSize.width}x${captureSize.height}")
            CrashLogger.log(TAG, "Capture size: ${captureSize.width}x${captureSize.height}")

            captureReader = ImageReader.newInstance(
                captureSize.width, captureSize.height,
                ImageFormat.YUV_420_888, 1
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set up camera streams: ${e.message}", e)
            CrashLogger.logException(TAG, e)
            resetHard()
            onError?.invoke("Camera setup failed: ${e.message}")
            return
        }

        sessionRetryCount = 0

        try {
            cameraManager.openCamera(lens.cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    CrashLogger.log(TAG, "onOpened camera=${camera.id}")
                    cameraDevice = camera
                    createSession(camera, previewSize)
                }

                override fun onDisconnected(camera: CameraDevice) {
                    Log.w(TAG, "Camera disconnected")
                    CrashLogger.log(TAG, "onDisconnected camera=${camera.id}")
                    camera.close()
                    cameraDevice = null
                    captureSession = null
                    onDisconnected?.invoke()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    Log.e(TAG, "Camera error: $error")
                    CrashLogger.log(TAG, "Camera onError error=$error camera=${camera.id}")
                    camera.close()
                    cameraDevice = null
                    captureSession = null
                    onError?.invoke("Camera error: $error")
                }
            }, backgroundHandler)
        } catch (e: Exception) {
            Log.e(TAG, "openCamera failed: ${e.message}", e)
            CrashLogger.logException(TAG, e)
            resetHard()
            onError?.invoke("Camera open failed: ${e.message}")
        }
    }

    private fun createSession(camera: CameraDevice, previewSize: Size) {
        val surfaces = mutableListOf(previewReader!!.surface, captureReader!!.surface)
        CrashLogger.log(TAG, "createSession surfaces=${surfaces.size}")

        @Suppress("DEPRECATION")
        try {
            camera.createCaptureSession(surfaces, object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    CrashLogger.log(TAG, "onConfigured session=${session.device.id}")
                    captureSession = session
                    sessionRetryCount = 0
                    onSessionReady?.invoke(previewSize.width, previewSize.height)
                    startPreview()
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    Log.e(TAG, "Session configuration failed (attempt ${sessionRetryCount + 1})")
                    CrashLogger.log(TAG, "onConfigureFailed attempt=${sessionRetryCount + 1}")
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
        } catch (e: CameraAccessException) {
            Log.e(TAG, "createCaptureSession failed: ${e.message}", e)
            CrashLogger.logException(TAG, e)
            onError?.invoke("Camera session creation failed: ${e.message}")
        }
    }

    private fun startPreview() {
        val camera = cameraDevice ?: run { CrashLogger.log(TAG, "startPreview: cameraDevice is null"); return }
        val session = captureSession ?: run { CrashLogger.log(TAG, "startPreview: captureSession is null"); return }

        val afMode = safeAfMode(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        CrashLogger.log(TAG, "startPreview afMode=$afMode flash=$currentFlashMode")

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            currentFlashMode.applyToRequest(this, availableAeModes)
        }

        try {
            session.setRepeatingRequest(request.build(), null, backgroundHandler)
            Log.d(TAG, "Preview started with flash mode: $currentFlashMode")
        } catch (e: CameraAccessException) {
            Log.e(TAG, "setRepeatingRequest failed: ${e.message}", e)
            CrashLogger.logException(TAG, e)
        }
    }

    fun setFlashMode(mode: FlashMode) {
        currentFlashMode = mode
        startPreview()
    }

    fun startContinuousAf() {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return

        val afMode = safeAfMode(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            currentFlashMode.applyToRequest(this, availableAeModes)
        }
        try {
            session.setRepeatingRequest(request.build(), null, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "startContinuousAf setRepeatingRequest failed: ${e.message}", e)
        }
    }

    fun startAfAeLock(
        meteringRect: MeteringRectangle,
        handler: Handler,
        onCaptureStarted: () -> Unit,
        onCaptureCompleted: (TotalCaptureResult) -> Unit
    ) {
        val camera = cameraDevice ?: run { CrashLogger.log(TAG, "startAfAeLock: cameraDevice is null"); return }
        val session = captureSession ?: run { CrashLogger.log(TAG, "startAfAeLock: captureSession is null"); return }

        val afMode = safeAfMode(CaptureRequest.CONTROL_AF_MODE_AUTO)
        CrashLogger.log(TAG, "startAfAeLock afMode=$afMode rect=(${meteringRect.x},${meteringRect.y},${meteringRect.width},${meteringRect.height})")

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(meteringRect))
            set(CaptureRequest.CONTROL_AE_REGIONS, arrayOf(meteringRect))
            set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER_START)
            set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
            currentFlashMode.applyToRequest(this, availableAeModes)
        }

        try {
            session.capture(request.build(), object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureStarted(session: CameraCaptureSession, request: CaptureRequest, timestamp: Long, frameNumber: Long) {
                    onCaptureStarted()
                }

                override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                    val afState = result.get(CaptureResult.CONTROL_AF_STATE)
                    if (afState != null && afState != CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN) {
                        try {
                            session.stopRepeating()
                        } catch (e: CameraAccessException) {
                            Log.e(TAG, "stopRepeating failed: ${e.message}", e)
                        }
                        val lockRequest = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                            addTarget(previewReader!!.surface)
                            set(CaptureRequest.CONTROL_AF_MODE, afMode)
                            set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(meteringRect))
                            set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_CANCEL)
                            set(CaptureRequest.CONTROL_AE_LOCK, true)
                            set(CaptureRequest.CONTROL_AWB_LOCK, true)
                            currentFlashMode.applyToRequest(this, availableAeModes)
                        }
                        try {
                            session.setRepeatingRequest(lockRequest.build(), null, backgroundHandler)
                        } catch (e: CameraAccessException) {
                            Log.e(TAG, "lock setRepeatingRequest failed: ${e.message}", e)
                        }
                        onCaptureCompleted(result)
                    }
                }
            }, handler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "startAfAeLock capture failed: ${e.message}", e)
            CrashLogger.logException(TAG, e)
        }
    }

    fun unlockAeAwb() {
        startPreview()
    }

    fun captureStill(onCaptureAvailable: (Image, TotalCaptureResult) -> Unit, onCaptureFailed: () -> Unit) {
        val camera = cameraDevice ?: run { CrashLogger.log(TAG, "captureStill: cameraDevice is null"); onCaptureFailed(); return }
        val session = captureSession ?: run { CrashLogger.log(TAG, "captureStill: captureSession is null"); onCaptureFailed(); return }

        val afMode = safeAfMode(CaptureRequest.CONTROL_AF_MODE_AUTO)
        val aeMode = safeAeMode(CaptureRequest.CONTROL_AE_MODE_ON)
        CrashLogger.log(TAG, "captureStill afMode=$afMode aeMode=$aeMode")

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(captureReader!!.surface)
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            set(CaptureRequest.CONTROL_AE_MODE, aeMode)
            currentFlashMode.applyToRequest(this, availableAeModes)
        }

        try {
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
        } catch (e: CameraAccessException) {
            Log.e(TAG, "captureStill failed: ${e.message}", e)
            CrashLogger.logException(TAG, e)
            onCaptureFailed()
        }
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

        availableAfModes = intArrayOf()
        availableAeModes = intArrayOf()
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

        availableAfModes = intArrayOf()
        availableAeModes = intArrayOf()
    }

    private fun safeAfMode(preferred: Int): Int {
        if (preferred in availableAfModes) return preferred
        if (CaptureRequest.CONTROL_AF_MODE_AUTO in availableAfModes) return CaptureRequest.CONTROL_AF_MODE_AUTO
        if (CaptureRequest.CONTROL_AF_MODE_OFF in availableAfModes) return CaptureRequest.CONTROL_AF_MODE_OFF
        return CaptureRequest.CONTROL_AF_MODE_OFF
    }

    private fun safeAeMode(preferred: Int): Int {
        if (preferred in availableAeModes) return preferred
        if (CaptureRequest.CONTROL_AE_MODE_ON in availableAeModes) return CaptureRequest.CONTROL_AE_MODE_ON
        if (CaptureRequest.CONTROL_AE_MODE_OFF in availableAeModes) return CaptureRequest.CONTROL_AE_MODE_OFF
        return CaptureRequest.CONTROL_AE_MODE_OFF
    }

    companion object {
        private const val TAG = "Camera2Manager"
    }
}
