package com.agx.camera.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.hardware.camera2.*
import android.hardware.camera2.params.MeteringRectangle
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.util.Size
import com.agx.camera.CrashLogger
import java.lang.reflect.Field
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class Camera2Manager(private val context: Context) {

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var previewReader: ImageReader? = null
    private var captureReader: ImageReader? = null

    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    private var currentFlashMode: FlashMode = FlashMode.OFF
    private var currentLens: LensInfo? = null
    private var cameraCharacteristics: CameraCharacteristics? = null
    var captureSize: Size = Size(640, 480)
        private set

    private var availableAfModes: IntArray = intArrayOf()
    private var availableAeModes: IntArray = intArrayOf()

    var onFrameAvailable: ((Image) -> Unit)? = null
    var onSessionReady: ((Int, Int) -> Unit)? = null
    var onError: ((String) -> Unit)? = null
    var onDisconnected: (() -> Unit)? = null
    var onAutoExposureReadout: ((Int, Long) -> Unit)? = null
    var onMaxZoomReady: ((Float) -> Unit)? = null

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

    // Exposure compensation
    private var currentExposureComp = 0
    var aeExposureCompRange: IntRange? = null
        private set
    var aeExposureStep: Float = 1.0f
        private set

    // White balance
    private var currentAwbMode = CaptureRequest.CONTROL_AWB_MODE_AUTO
    var isAwbLocked = false
        private set

    // Focus/Exposure lock
    private var meteringRegions: Array<MeteringRectangle>? = null
    private var focusLocked = false

    // Device region capabilities (set on session open)
    private var deviceMaxAfRegions = 0
    private var deviceMaxAeRegions = 0

    // Manual exposure
    var isManualExposure = false
        private set
    private var currentManualIso = 400
    private var currentManualExposureNs = 33_333_333L
    private var captureImageLatch = CountDownLatch(1)
    var availableIsoValues: IntArray = intArrayOf()
    var availableShutterSpeedsNs: LongArray = longArrayOf()

    // Auto exposure readout
    private var lastAutoIso = 200
    private var lastAutoShutterNs = 33_333_333L
    private var lastAeReadoutTime = 0L

    val currentFlashModeForExif: FlashMode get() = currentFlashMode

    // Zoom (SCALER_CROP_REGION)
    private var currentZoomFactor = 1.0f
    private var currentZoomCenterX = 0.5f
    private var currentZoomCenterY = 0.5f
    private var pendingZoomUpdate = false

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
    fun openCamera(lens: LensInfo, previewSize: Size, targetCaptureWidth: Int = 0, targetCaptureHeight: Int = 0, onError: ((String) -> Unit)? = null) {
        currentLens = lens
        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager

        Log.d(TAG, "Opening camera ${lens.cameraId}, preview size: ${previewSize.width}x${previewSize.height}, target capture: ${targetCaptureWidth}x${targetCaptureHeight}")
        CrashLogger.log(TAG, "openCamera id=${lens.cameraId} preview=${previewSize.width}x${previewSize.height} level=${lens.hardwareLevel} raw=${lens.hasRawSensor}")

        try {
            previewReader = ImageReader.newInstance(
                previewSize.width, previewSize.height,
                ImageFormat.YUV_420_888, 3
            ).apply {
                setOnImageAvailableListener(previewListener, backgroundHandler)
            }

            val chars = cameraManager.getCameraCharacteristics(lens.cameraId)
            cameraCharacteristics = chars
            availableAfModes = chars.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: intArrayOf()
            availableAeModes = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES) ?: intArrayOf()
            CrashLogger.log(TAG, "AF modes: ${availableAfModes.toList()}, AE modes: ${availableAeModes.toList()}")

            val maxAfRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0
            val maxAeRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0
            deviceMaxAfRegions = maxAfRegions
            deviceMaxAeRegions = maxAeRegions
            CrashLogger.log(TAG, "max AF regions=$maxAfRegions, max AE regions=$maxAeRegions")

            populateAeExposureRange(chars)

            val maxZoom = chars.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1.0f
            CrashLogger.log(TAG, "maxDigitalZoom=$maxZoom")
            onMaxZoomReady?.invoke(maxZoom)

            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            
            // Determine capture size: use target if specified and valid, otherwise max YUV size
            val yuvSizes = map?.getOutputSizes(ImageFormat.YUV_420_888) ?: emptyArray()
            if (targetCaptureWidth > 0 && targetCaptureHeight > 0) {
                // Find the closest YUV size that matches or exceeds the target aspect ratio
                val targetAspect = targetCaptureWidth.toFloat() / targetCaptureHeight
                val exactMatch = yuvSizes.firstOrNull { it.width == targetCaptureWidth && it.height == targetCaptureHeight }
                if (exactMatch != null) {
                    captureSize = Size(exactMatch.width, exactMatch.height)
                } else {
                    // Find closest by aspect ratio, preferring larger sizes
                    captureSize = yuvSizes
                        .filter { it.width >= targetCaptureWidth && it.height >= targetCaptureHeight }
                        .minByOrNull { size ->
                            val sizeAspect = size.width.toFloat() / size.height
                            kotlin.math.abs(sizeAspect - targetAspect)
                        }
                        ?: yuvSizes.maxByOrNull { it.width.toLong() * it.height.toLong() }
                        ?: previewSize
                }
            } else {
                // Fallback: maximum YUV size
                val maxSize = yuvSizes
                    ?.maxByOrNull { it.width.toLong() * it.height.toLong() }
                    ?: previewSize
                captureSize = Size(maxSize.width, maxSize.height)
            }

            Log.d(TAG, "Capture size: ${captureSize.width}x${captureSize.height} (target: ${targetCaptureWidth}x${targetCaptureHeight})")
            CrashLogger.log(TAG, "Capture size: ${captureSize.width}x${captureSize.height} (target: ${targetCaptureWidth}x${targetCaptureHeight})")

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

    private fun populateAeExposureRange(chars: CameraCharacteristics) {
        val keys = chars.keys
        var minVal: Int? = null
        var maxVal: Int? = null
        var stepNum: Int = 1
        var stepDen: Int = 1

        for (key in keys) {
            val name = key.name
            if (name == "android.control.aeExposureCompensationRange") {
                val range = chars.get(key)
                if (range != null) {
                    try {
                        val field = range.javaClass.getDeclaredField("min")
                        field.isAccessible = true
                        minVal = field.get(range) as Int
                        val field2 = range.javaClass.getDeclaredField("max")
                        field2.isAccessible = true
                        maxVal = field2.get(range) as Int
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to read aeExposureCompensationRange: ${e.message}")
                    }
                }
            } else if (name == "android.control.aeExposureCompensationStep") {
                val step = chars.get(key)
                if (step != null) {
                    try {
                        val stepStr = step.toString()
                        if (stepStr.contains("/")) {
                            val parts = stepStr.split("/")
                            stepNum = parts[0].toIntOrNull() ?: 1
                            stepDen = parts[1].toIntOrNull() ?: 1
                        } else {
                            stepNum = stepStr.toIntOrNull() ?: 1
                            stepDen = 1
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to parse aeExposureCompensationStep: ${e.message}")
                    }
                }
            }
        }

        if (minVal != null && maxVal != null) {
            aeExposureCompRange = IntRange(minVal!!, maxVal!!)
            aeExposureStep = if (stepDen != 0) stepNum.toFloat() / stepDen else 1.0f
            Log.d(TAG, "AE exposure compensation range: $aeExposureCompRange, step: $aeExposureStep")
        } else {
            aeExposureCompRange = IntRange(-6, 6)
            aeExposureStep = 1.0f
            Log.w(TAG, "Using fallback AE compensation range")
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

    private fun startPreview() = applyPreviewRequest()

    private fun startPreviewWithEv() = applyPreviewRequest()

    fun setFlashMode(mode: FlashMode) {
        currentFlashMode = mode
        if (isManualExposure) return
        applyPreviewRequest()
    }

    fun setWhiteBalanceMode(mode: Int) {
        currentAwbMode = mode
        CrashLogger.log(TAG, "setWhiteBalanceMode: mode=$mode (${awbModeName(mode)}) isManualExposure=$isManualExposure")
        if (isManualExposure) return
        applyPreviewRequest()
    }

    fun lockAwb() {
        isAwbLocked = true
        applyPreviewRequest()
    }

    fun unlockAwb() {
        isAwbLocked = false
        applyPreviewRequest()
    }

    fun setExposureCompensation(ev: Int) {
        currentExposureComp = ev
        if (isManualExposure) return
        applyPreviewRequest()
    }

    fun setManualExposure(iso: Int, exposureTimeNs: Long) {
        isManualExposure = true
        currentManualIso = iso
        currentManualExposureNs = exposureTimeNs
        val camera = cameraDevice ?: return
        val session = captureSession ?: return

        CrashLogger.log(TAG, "setManualExposure iso=$iso exposureTimeNs=$exposureTimeNs")

        // Focus behaves the same in manual mode: continuous + regions when unlocked, hold when locked
        val afMode = safeAfMode(
            if (focusLocked) CaptureRequest.CONTROL_AF_MODE_AUTO
            else CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
        )

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
            set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            val awbToSend = if (currentAwbMode == CaptureRequest.CONTROL_AWB_MODE_OFF) {
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            } else {
                currentAwbMode
            }
            set(CaptureRequest.CONTROL_AWB_MODE, awbToSend)
            meteringRegions?.let { set(CaptureRequest.CONTROL_AF_REGIONS, it) }
            if (isAwbLocked) set(CaptureRequest.CONTROL_AWB_LOCK, true)
            set(CaptureRequest.SENSOR_SENSITIVITY, iso)
            set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeNs)
            set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
            applyCropRegion()
        }

        try {
            session.setRepeatingRequest(request.build(), aeReadoutCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "setManualExposure setRepeatingRequest failed: ${e.message}", e)
        }
    }

    // --- SCALER_CROP_REGION zoom ---

    fun computeCropRegion(): Rect? {
        val chars = cameraCharacteristics ?: return null
        val sensorRect = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: return null
        val maxZoom = chars.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1.0f
        val zoom = currentZoomFactor.coerceIn(1.0f, maxZoom)
        if (zoom <= 1.0f) return sensorRect

        val cropWidth = (sensorRect.width() / zoom).toInt()
        val cropHeight = (sensorRect.height() / zoom).toInt()
        val centerX = sensorRect.left + (currentZoomCenterX * sensorRect.width()).toInt()
        val centerY = sensorRect.top + (currentZoomCenterY * sensorRect.height()).toInt()
        val left = (centerX - cropWidth / 2).coerceIn(sensorRect.left, sensorRect.right - cropWidth)
        val top = (centerY - cropHeight / 2).coerceIn(sensorRect.top, sensorRect.bottom - cropHeight)
        return Rect(left, top, left + cropWidth, top + cropHeight)
    }

    private fun CaptureRequest.Builder.applyCropRegion() {
        val crop = computeCropRegion() ?: return
        set(CaptureRequest.SCALER_CROP_REGION, crop)
    }

    fun updateZoom(factor: Float, centerX: Float, centerY: Float) {
        currentZoomFactor = factor
        currentZoomCenterX = centerX
        currentZoomCenterY = centerY
        if (!pendingZoomUpdate && backgroundHandler != null) {
            pendingZoomUpdate = true
            backgroundHandler!!.postDelayed({
                pendingZoomUpdate = false
                applyPreviewRequest()
            }, 16)
        }
    }

    fun setAutoExposure() {
        isManualExposure = false
        currentExposureComp = 0
        applyPreviewRequest()
    }

    private var isAfScanning = false
    private var pendingMeteringRegions: Array<MeteringRectangle>? = null

    fun setMeteringRegion(rect: MeteringRectangle?) {
        if (rect == null) {
            pendingMeteringRegions = null
            meteringRegions = null
            isAfScanning = false
            applyPreviewRequest()
            return
        }

        val chars = cameraCharacteristics ?: return
        val maxAfRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0
        val maxAeRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0

        if (maxAfRegions == 0 && maxAeRegions == 0) {
            CrashLogger.log(TAG, "setMeteringRegion: device has 0 regions, ignoring")
            return
        }

        pendingMeteringRegions = arrayOf(rect)
        triggerRegionFocus(rect)
    }

    private fun triggerRegionFocus(rect: MeteringRectangle) {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return
        if (isAfScanning) {
            cancelAfTrigger()
        }

        meteringRegions = arrayOf(rect)

        if (deviceMaxAfRegions == 0) {
            if (deviceMaxAeRegions > 0) holdRegionFocus()
            return
        }

        isAfScanning = true

        val afMode = safeAfMode(CaptureRequest.CONTROL_AF_MODE_AUTO)

        fun buildPreviewRequest(withTrigger: Boolean): CaptureRequest.Builder {
            return camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(previewReader!!.surface)
                set(CaptureRequest.CONTROL_AF_MODE, afMode)
                setMeteringRegions(meteringRegions)
                if (withTrigger) {
                    set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
                }
                if (isManualExposure) {
                    set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                    set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                    set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
                } else {
                    currentFlashMode.applyToRequest(this, availableAeModes)
                }
                applyCropRegion()
            }
        }

        val afCallback = object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                result: TotalCaptureResult
            ) {
                val afState = result.get(CaptureResult.CONTROL_AF_STATE) ?: return
                Log.d(TAG, "AF scan state: $afState")

                if (afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED ||
                    afState == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED
                ) {
                    try {
                        session.stopRepeating()
                    } catch (e: CameraAccessException) {
                        Log.e(TAG, "triggerRegionFocus stopRepeating failed", e)
                    }
                    holdRegionFocus()
                }
            }
        }

        try {
            session.setRepeatingRequest(buildPreviewRequest(false).build(), afCallback, backgroundHandler)
            session.capture(buildPreviewRequest(true).build(), null, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "triggerRegionFocus failed", e)
            isAfScanning = false
        }
    }

    private fun holdRegionFocus() {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return
        isAfScanning = false

        // Switch to repeating request with AF_TRIGGER_CANCEL, no AE_LOCK
        val holdRequest = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
            if (deviceMaxAfRegions > 0) {
                set(CaptureRequest.CONTROL_AF_MODE, safeAfMode(CaptureRequest.CONTROL_AF_MODE_AUTO))
                set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_CANCEL)
            }
            setMeteringRegions(meteringRegions)
            if (isManualExposure) {
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
            } else {
                currentFlashMode.applyToRequest(this, availableAeModes)
            }
            applyCropRegion()
        }

        try {
            session.setRepeatingRequest(holdRequest.build(), aeReadoutCallback, backgroundHandler)
            Log.d(TAG, "holdRegionFocus: locked with regions=${meteringRegions?.contentToString()}")
        } catch (e: CameraAccessException) {
            Log.e(TAG, "holdRegionFocus failed", e)
        }
    }

    private fun cancelAfTrigger() {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return

        try {
            session.stopRepeating()
        } catch (e: CameraAccessException) {
            Log.e(TAG, "cancelAfTrigger stopRepeating failed", e)
        }

        val cancelRequest = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
            set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_CANCEL)
            if (isManualExposure) {
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
            } else {
                currentFlashMode.applyToRequest(this, availableAeModes)
            }
            applyCropRegion()
        }

        try {
            session.setRepeatingRequest(cancelRequest.build(), aeReadoutCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "cancelAfTrigger failed", e)
        }
        isAfScanning = false
    }

    // Call this from your unlock button or when switching lenses
    fun unlockFocus() {
        meteringRegions = null
        pendingMeteringRegions = null
        isAfScanning = false
        focusLocked = false
        applyPreviewRequest() // back to continuous picture, no regions
    }

    // Lock focus and exposure at current metering region
    fun lockFocusAndExposure() {
        if (meteringRegions == null) return
        focusLocked = true
        applyPreviewRequest()
    }

    // Unlock focus and exposure, resume continuous AF
    fun unlockFocusAndExposure() {
        focusLocked = false
        applyPreviewRequest()
    }

    private fun applyPreviewRequest() {
        if (isManualExposure) {
            setManualExposure(currentManualIso, currentManualExposureNs)
            return
        }
        val camera = cameraDevice ?: run { CrashLogger.log(TAG, "applyPreviewRequest: cameraDevice null"); return }
        val session = captureSession ?: run { CrashLogger.log(TAG, "applyPreviewRequest: session null"); return }

        val inHoldState = !focusLocked && meteringRegions != null && !isAfScanning

        val afMode = if (deviceMaxAfRegions > 0) {
            safeAfMode(
                if (focusLocked || inHoldState) CaptureRequest.CONTROL_AF_MODE_AUTO
                else CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
            )
        } else {
            CaptureRequest.CONTROL_AF_MODE_OFF
        }

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            val awbToSend = if (currentAwbMode == CaptureRequest.CONTROL_AWB_MODE_OFF) {
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            } else {
                currentAwbMode
            }
            set(CaptureRequest.CONTROL_AWB_MODE, awbToSend)
            meteringRegions?.let { setMeteringRegions(it) }
            if (inHoldState) {
                set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_CANCEL)
            }
            if (currentExposureComp != 0) {
                set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, currentExposureComp)
            }
            if (focusLocked) {
                set(CaptureRequest.CONTROL_AE_LOCK, true)
            }
            if (isAwbLocked) {
                set(CaptureRequest.CONTROL_AWB_LOCK, true)
            }
            currentFlashMode.applyToRequest(this, availableAeModes)
            applyCropRegion()
        }

        CrashLogger.log(TAG, "applyPreviewRequest: awb=${awbModeName(currentAwbMode)} af=$afMode hold=$inHoldState locked=$focusLocked awbLocked=$isAwbLocked")

        try {
            session.setRepeatingRequest(request.build(), aeReadoutCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "applyPreviewRequest failed: ${e.message}", e)
            CrashLogger.logException(TAG, e)
        }
    }

    fun updateManualControlRanges(chars: CameraCharacteristics) {
        val sensRange = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val expRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)

        val sensMin = sensRange?.lower ?: 100
        val sensMax = sensRange?.upper ?: 6400
        val expMinNs = expRange?.lower ?: 1_000_000L
        val expMaxNs = expRange?.upper ?: 1_000_000_000L

        Log.d(TAG, "Sensor sensitivity range: $sensMin - $sensMax")
        Log.d(TAG, "Sensor exposure range: ${expMinNs/1_000_000.0}ms - ${expMaxNs/1_000_000.0}ms")

        val isoVals = mutableListOf<Int>()
        var iso = sensMin.coerceAtLeast(50)
        while (iso <= sensMax) {
            isoVals.add(iso)
            iso = (iso * 1.26).toInt().coerceAtLeast(iso + 1)
        }
        if (isoVals.lastOrNull() != sensMax) isoVals.add(sensMax)
        availableIsoValues = isoVals.toIntArray()

        val shutterVals = mutableListOf<Long>()
        var exp = expMinNs.coerceAtLeast(10_000L)
        while (exp <= expMaxNs) {
            shutterVals.add(exp)
            exp = (exp * 1.26).toLong().coerceAtLeast(exp + 1)
        }
        if (shutterVals.lastOrNull() != expMaxNs) shutterVals.add(expMaxNs)
        availableShutterSpeedsNs = shutterVals.toLongArray()

        Log.d(TAG, "Manual ISO values (${availableIsoValues.size}): ${availableIsoValues.toList()}")
        Log.d(TAG, "Manual shutter speeds (${availableShutterSpeedsNs.size}): ${availableShutterSpeedsNs.map { formatNs(it) }}")
    }

    private fun formatNs(ns: Long): String {
        val sec = ns / 1_000_000_000.0
        return if (sec >= 1.0) String.format("%.2fs", sec) else String.format("1/%.0fs", 1.0 / sec)
    }

    fun startContinuousAf() {
        if (isManualExposure) return
        focusLocked = false
        applyPreviewRequest()
    }

    fun startAfAeLock(
        meteringRect: MeteringRectangle,
        handler: Handler,
        onCaptureStarted: () -> Unit,
        onCaptureCompleted: (TotalCaptureResult) -> Unit
    ) {
        val camera = cameraDevice ?: run { CrashLogger.log(TAG, "startAfAeLock: cameraDevice is null"); return }
        val session = captureSession ?: run { CrashLogger.log(TAG, "startAfAeLock: captureSession is null"); return }

        val afMode = if (deviceMaxAfRegions > 0) safeAfMode(CaptureRequest.CONTROL_AF_MODE_AUTO) else CaptureRequest.CONTROL_AF_MODE_OFF
        CrashLogger.log(TAG, "startAfAeLock afMode=$afMode rect=(${meteringRect.x},${meteringRect.y},${meteringRect.width},${meteringRect.height})")

        val triggerRequest = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            setMeteringRegions(arrayOf(meteringRect))
            if (deviceMaxAfRegions > 0) {
                set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER_START)
                set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
            }
            if (isManualExposure) {
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
                set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
            } else {
                currentFlashMode.applyToRequest(this, availableAeModes)
            }
            applyCropRegion()
        }

        try {
            session.setRepeatingRequest(triggerRequest.build(), object : CameraCaptureSession.CaptureCallback() {
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
                            if (deviceMaxAfRegions > 0) {
                                set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(meteringRect))
                                set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_CANCEL)
                            }
                            setMeteringRegions(arrayOf(meteringRect))
                            set(CaptureRequest.CONTROL_AE_LOCK, true)
                            if (isManualExposure) {
                                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                                set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                                set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
                                set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
                            } else {
                                currentFlashMode.applyToRequest(this, availableAeModes)
                            }
                            applyCropRegion()
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
            onCaptureStarted()
        } catch (e: CameraAccessException) {
            Log.e(TAG, "startAfAeLock setRepeatingRequest failed: ${e.message}", e)
            CrashLogger.logException(TAG, e)
        }
    }

    fun lockAeAf(meteringRect: MeteringRectangle, handler: Handler, onLocked: () -> Unit) {
        val camera = cameraDevice ?: run { CrashLogger.log(TAG, "lockAeAf: cameraDevice is null"); return }
        val session = captureSession ?: run { CrashLogger.log(TAG, "lockAeAf: captureSession is null"); return }

        val afMode = if (deviceMaxAfRegions > 0) safeAfMode(CaptureRequest.CONTROL_AF_MODE_AUTO) else CaptureRequest.CONTROL_AF_MODE_OFF
        CrashLogger.log(TAG, "lockAeAf afMode=$afMode rect=(${meteringRect.x},${meteringRect.y},${meteringRect.width},${meteringRect.height})")

        val triggerRequest = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewReader!!.surface)
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            setMeteringRegions(arrayOf(meteringRect))
            if (deviceMaxAfRegions > 0) {
                set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER_START)
                set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
            }
            if (isManualExposure) {
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
                set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
            } else {
                currentFlashMode.applyToRequest(this, availableAeModes)
            }
            applyCropRegion()
        }

        try {
            session.setRepeatingRequest(triggerRequest.build(), object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                    val afState = result.get(CaptureResult.CONTROL_AF_STATE)
                    val aeState = result.get(CaptureResult.CONTROL_AE_STATE)
                    if (afState != null && afState != CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN &&
                        aeState != null && aeState != CaptureResult.CONTROL_AE_STATE_SEARCHING) {
                        try {
                            session.stopRepeating()
                        } catch (e: CameraAccessException) {
                            Log.e(TAG, "stopRepeating failed: ${e.message}", e)
                        }
                        val lockRequest = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                            addTarget(previewReader!!.surface)
                            set(CaptureRequest.CONTROL_AF_MODE, afMode)
                            if (deviceMaxAfRegions > 0) {
                                set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(meteringRect))
                                set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_CANCEL)
                            }
                            setMeteringRegions(arrayOf(meteringRect))
                            set(CaptureRequest.CONTROL_AE_LOCK, true)
                            if (isManualExposure) {
                                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                                set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                                set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
                                set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
                            } else {
                                currentFlashMode.applyToRequest(this, availableAeModes)
                            }
                            applyCropRegion()
                        }
                        try {
                            session.setRepeatingRequest(lockRequest.build(), null, backgroundHandler)
                            onLocked()
                        } catch (e: CameraAccessException) {
                            Log.e(TAG, "lock setRepeatingRequest failed: ${e.message}", e)
                        }
                    }
                }
            }, handler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "lockAeAf setRepeatingRequest failed: ${e.message}", e)
            CrashLogger.logException(TAG, e)
        }
    }

    fun unlockAeAwb() {
        if (isManualExposure) {
            setManualExposure(currentManualIso, currentManualExposureNs)
        } else {
            startPreview()
        }
    }

    fun captureStill(onCaptureAvailable: (Image, TotalCaptureResult) -> Unit, onCaptureFailed: () -> Unit) {
        val camera = cameraDevice ?: run { CrashLogger.log(TAG, "captureStill: cameraDevice is null"); onCaptureFailed(); return }
        val session = captureSession ?: run { CrashLogger.log(TAG, "captureStill: captureSession is null"); onCaptureFailed(); return }
        val reader = captureReader ?: run { CrashLogger.log(TAG, "captureStill: captureReader is null"); onCaptureFailed(); return }

        val afMode = safeAfMode(CaptureRequest.CONTROL_AF_MODE_AUTO)
        val aeMode = safeAeMode(if (isManualExposure) CaptureRequest.CONTROL_AE_MODE_OFF else CaptureRequest.CONTROL_AE_MODE_ON)
        CrashLogger.log(TAG, "captureStill afMode=$afMode aeMode=$aeMode manual=$isManualExposure")

        captureImageLatch = CountDownLatch(1)
        reader.setOnImageAvailableListener({ r ->
            CrashLogger.log(TAG, "stillCapture onImageAvailable")
            captureImageLatch.countDown()
        }, backgroundHandler)

        val preview = previewReader?.surface
        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(reader.surface)
            if (preview != null) addTarget(preview)
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            set(CaptureRequest.CONTROL_AE_MODE, aeMode)
            if (isManualExposure) {
                set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
                set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
            } else {
                currentFlashMode.applyToRequest(this, availableAeModes)
            }
            applyCropRegion()
        }

        try {
            session.capture(request.build(), object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureStarted(session: CameraCaptureSession, request: CaptureRequest, timestamp: Long, frameNumber: Long) {
                    CrashLogger.log(TAG, "stillCapture onCaptureStarted frame=$frameNumber")
                }

                override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                    CrashLogger.log(TAG, "stillCapture onCaptureCompleted frame=${result.frameNumber}")
                    val imageReady = captureImageLatch.await(5, TimeUnit.SECONDS)
                    CrashLogger.log(TAG, "stillCapture latch waited imageReady=$imageReady")
                    val image = captureReader?.acquireLatestImage()
                    reader.setOnImageAvailableListener(null, null)
                    if (image != null) {
                        CrashLogger.log(TAG, "stillCapture image acquired ${image.width}x${image.height}")
                        onCaptureAvailable(image, result)
                    } else {
                        CrashLogger.log(TAG, "stillCapture image is null after latch")
                        Log.e(TAG, "Capture image not available after onImageAvailable latch")
                        onCaptureFailed()
                    }
                }

                override fun onCaptureFailed(session: CameraCaptureSession, request: CaptureRequest, failure: CaptureFailure) {
                    CrashLogger.log(TAG, "stillCapture onCaptureFailed reason=${failure.reason} wasImageCaptured=${failure.wasImageCaptured()}")
                    captureImageLatch.countDown()
                    reader.setOnImageAvailableListener(null, null)
                    Log.e(TAG, "Capture failed: ${failure.reason}")
                    onCaptureFailed()
                }
            }, backgroundHandler)
        } catch (e: CameraAccessException) {
            CrashLogger.log(TAG, "stillCapture exception: ${e.message}")
            reader.setOnImageAvailableListener(null, null)
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

    private fun awbModeName(mode: Int): String = when (mode) {
        CaptureRequest.CONTROL_AWB_MODE_AUTO -> "AUTO"
        CaptureRequest.CONTROL_AWB_MODE_OFF -> "OFF"
        CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT -> "DAYLIGHT"
        CaptureRequest.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT -> "CLOUDY"
        CaptureRequest.CONTROL_AWB_MODE_INCANDESCENT -> "INCANDESCENT"
        CaptureRequest.CONTROL_AWB_MODE_FLUORESCENT -> "FLUORESCENT"
        CaptureRequest.CONTROL_AWB_MODE_TWILIGHT -> "TWILIGHT"
        CaptureRequest.CONTROL_AWB_MODE_SHADE -> "SHADE"
        else -> "UNKNOWN($mode)"
    }

    private fun CaptureRequest.Builder.setMeteringRegions(regions: Array<MeteringRectangle>?) {
        if (deviceMaxAfRegions > 0) set(CaptureRequest.CONTROL_AF_REGIONS, regions)
        if (deviceMaxAeRegions > 0) set(CaptureRequest.CONTROL_AE_REGIONS, regions)
    }

    private val aeReadoutCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            val now = SystemClock.elapsedRealtime()
            if (now - lastAeReadoutTime < 500) return
            lastAeReadoutTime = now

            if (isManualExposure) return

            val iso = result.get(CaptureResult.SENSOR_SENSITIVITY)
            val exposureTime = result.get(CaptureResult.SENSOR_EXPOSURE_TIME)
            if (iso != null && exposureTime != null) {
                lastAutoIso = iso
                lastAutoShutterNs = exposureTime
                onAutoExposureReadout?.invoke(iso, exposureTime)
            }

            val afState = result.get(CaptureResult.CONTROL_AF_STATE)
            val afMode = result.get(CaptureResult.CONTROL_AF_MODE)
            val aeState = result.get(CaptureResult.CONTROL_AE_STATE)
            val reportedAfRegions = result.get(CaptureResult.CONTROL_AF_REGIONS)
            val reportedAeRegions = result.get(CaptureResult.CONTROL_AE_REGIONS)

            CrashLogger.log(TAG,
                "onCaptureCompleted: afState=$afState afMode=$afMode aeState=$aeState " +
                "afRegions=${reportedAfRegions?.contentToString()} " +
                "aeRegions=${reportedAeRegions?.contentToString()}"
            )
        }
    }

    companion object {
        private const val TAG = "Camera2Manager"
    }
}