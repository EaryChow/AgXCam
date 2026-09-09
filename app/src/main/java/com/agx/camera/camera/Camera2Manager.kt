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
import android.view.Surface
import com.agx.camera.CrashLogger
import java.lang.reflect.Field
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class Camera2Manager(private val context: Context) {

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var previewReader: ImageReader? = null
    private var captureReader: ImageReader? = null
    private var rawReader: ImageReader? = null

    // When RAW mode is active the capture stream is omitted from the session
    // (still captures come from the RAW reader); the YUV preview surface is
    // kept so the HAL has a valid stream combination and AF behaves normally.
    // The HAL's concurrent full-res YUV capture stream wedged frame delivery on
    // the main lens of Xiaomi 15U when a close object was in front of the
    // camera, so only the full-res YUV capture is dropped.
    private var sessionRawOnly = false
    var rawSize: Size = Size(0, 0)
        private set

    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    private var currentFlashMode: FlashMode = FlashMode.OFF
    private var currentLens: LensInfo? = null
    private var cameraCharacteristics: CameraCharacteristics? = null
    var captureSize: Size = Size(640, 480)
        private set

    private var availableAfModes: IntArray = intArrayOf()
    private var availableAeModes: IntArray = intArrayOf()
    private var availableAwbModes: Set<Int> = emptySet()

    var onFrameAvailable: ((Image) -> Unit)? = null
    var onRawFrameAvailable: ((Image) -> Unit)? = null
    var onSessionReady: ((Int, Int) -> Unit)? = null
    var onError: ((String) -> Unit)? = null
    var onDisconnected: (() -> Unit)? = null
    var onAutoExposureReadout: ((Int, Long) -> Unit)? = null
    var onMaxZoomReady: ((Float) -> Unit)? = null

    @Volatile
    var latestColorCorrectionMatrix: FloatArray? = null

    @Volatile
    var latestColorCorrectionGains: FloatArray? = null

    private var ccLogCount = 0

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

    private val rawListener = ImageReader.OnImageAvailableListener { reader ->
        val image = reader.acquireLatestImage() ?: return@OnImageAvailableListener
        try {
            onRawFrameAvailable?.invoke(image)
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

    // Kelvin never sends AWB-OFF: several vendor HALs (incl. Xiaomi) silently
    // ignore OFF and keep running adaptive AWB. Instead applyAwb holds the HAL
    // at its fixed D65 preset (DAYLIGHT) with CONTROL_AWB_LOCK so physical D65
    // light maps to D65, and the app's relative Kelvin CAT does the shift.
    var isAwbLocked = false
        private set

    // Focus/Exposure lock
    private var meteringRegions: Array<MeteringRectangle>? = null
    // Independent AE metering region, set separately from the AF region once the
    // user drags the auto-exposure indicator away from the focus indicator. When
    // null, AE meters the focus (tap) region in auto mode.
    private var aeRegions: Array<MeteringRectangle>? = null
    private var focusLocked = false
    // Latest auto-focus lens distance (diopters), used to freeze focus on lock
    private var lastAutoFocusDistance: Float? = null

    // Manual focus (AF/MF toggle): when enabled, AF is driven to AF_MODE_OFF +
    // LENS_FOCUS_DISTANCE from the focus roller instead of any controller.
    var isManualFocus = false
        private set
    private var currentManualFocusDistance = 0f

    /** Device minimum focus distance in diopters (0 = no MF support, fixed lens). */
    val minFocusDistance: Float get() = deviceMinFocusDistance

    /** Last auto-focus lens position in diopters, used as the MF roller start. */
    val lastAutoFocusDistanceDiopters: Float? get() = lastAutoFocusDistance

    // Device region capabilities (set on session open)
    private var deviceMaxAfRegions = 0
    private var deviceMaxAeRegions = 0
    private var deviceMinFocusDistance = 0f

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

    fun lastExposureForExif(): Pair<Int, Long> =
        if (isManualExposure) Pair(currentManualIso, currentManualExposureNs)
        else Pair(lastAutoIso, lastAutoShutterNs)

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
    fun openCamera(lens: LensInfo, previewSize: Size, targetCaptureWidth: Int = 0, targetCaptureHeight: Int = 0, useRaw: Boolean = false, onError: ((String) -> Unit)? = null) {
        currentLens = lens
        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager

        Log.d(TAG, "Opening camera ${lens.cameraId}, preview size: ${previewSize.width}x${previewSize.height}, target capture: ${targetCaptureWidth}x${targetCaptureHeight}, useRaw: $useRaw")
        CrashLogger.log(TAG, "openCamera id=${lens.cameraId} preview=${previewSize.width}x${previewSize.height} level=${lens.hardwareLevel} raw=${lens.hasRawSensor} useRawStream=$useRaw")

        rawSize = Size(0, 0)

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
            availableAwbModes = chars.get(CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES)?.toSet() ?: emptySet()
            CrashLogger.log(TAG, "AF modes: ${availableAfModes.toList()}, AE modes: ${availableAeModes.toList()}, AWB modes: $availableAwbModes")

            val maxAfRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0
            val maxAeRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0
            deviceMaxAfRegions = maxAfRegions
            deviceMaxAeRegions = maxAeRegions
            deviceMinFocusDistance = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
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

            if (useRaw && lens.hasRawSensor) {
                val rawSizes = map?.getOutputSizes(ImageFormat.RAW_SENSOR) ?: emptyArray()
                val selected = rawSizes.maxByOrNull { it.width.toLong() * it.height.toLong() }
                if (selected != null) {
                    rawSize = selected
                    rawReader = ImageReader.newInstance(
                        rawSize.width, rawSize.height,
                        ImageFormat.RAW_SENSOR, 4
                    ).apply {
                        setOnImageAvailableListener(rawListener, backgroundHandler)
                    }
                    Log.d(TAG, "RAW stream created: ${rawSize.width}x${rawSize.height}")
                    CrashLogger.log(TAG, "RAW stream ${rawSize.width}x${rawSize.height}")
                } else {
                    Log.w(TAG, "RAW preview requested but no RAW_SENSOR output sizes available")
                    CrashLogger.log(TAG, "RAW requested but no RAW_SENSOR sizes")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set up camera streams: ${e.message}", e)
            CrashLogger.logException(TAG, e)
            resetHard()
            onError?.invoke("Camera setup failed: ${e.message}")
            return
        }

        sessionRawOnly = useRaw && rawReader != null
        CrashLogger.log(TAG, "session raw-only=$sessionRawOnly")

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
        val surfaces = mutableListOf<Surface>()
        if (sessionRawOnly) {
            val rawSurface = rawReader?.surface
            if (rawSurface == null) {
                sessionRawOnly = false
            } else {
                // HALs (especially aux lenses) require at least one YUV preview
                // stream alongside the RAW reader; a RAW-only session causes
                // endConfigure failures on some lenses.  The preview content is
                // still served from the RAW buffer on the GPU side.
                surfaces.add(previewReader!!.surface)
                surfaces.add(rawSurface)
            }
        }
        if (surfaces.isEmpty()) {
            surfaces.add(previewReader!!.surface)
            surfaces.add(captureReader!!.surface)
            rawReader?.let { surfaces.add(it.surface) }
        }
        CrashLogger.log(TAG, "createSession surfaces=${surfaces.size} raw=${rawReader != null} rawOnly=$sessionRawOnly")

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

    fun resolveRawSize(cameraId: String): Size? {
        val cm = context.getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager
        return try {
            val map = cm.getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            map?.getOutputSizes(ImageFormat.RAW_SENSOR)?.maxByOrNull { it.width.toLong() * it.height.toLong() }
        } catch (e: Exception) {
            Log.w(TAG, "resolveRawSize failed: ${e.message}")
            null
        }
    }

    private fun CaptureRequest.Builder.addPreviewTargets() {
        addTarget(previewReader!!.surface)
        rawReader?.let { addTarget(it.surface) }
    }

    fun setFlashMode(mode: FlashMode) {
        currentFlashMode = mode
        if (isManualExposure) return
        applyPreviewRequest()
    }

    fun setWhiteBalanceMode(mode: Int) {
        currentAwbMode = mode
        CrashLogger.log(TAG, "setWhiteBalanceMode: mode=$mode (${awbModeName(mode)}) isManualExposure=$isManualExposure")
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

    // Fixed (non-adaptive) AWB presets are required for Kelvin. Prefer the one
    // closest to D65 that this device supports; returns null only when the
    // device exposes no fixed preset at all (then AUTO is the only option).
    private fun pendingHoldMode(): Int? {
        return listOf(
            CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT,
            CaptureRequest.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT,
            CaptureRequest.CONTROL_AWB_MODE_SHADE,
            CaptureRequest.CONTROL_AWB_MODE_TWILIGHT,
            CaptureRequest.CONTROL_AWB_MODE_INCANDESCENT,
            CaptureRequest.CONTROL_AWB_MODE_FLUORESCENT
        ).firstOrNull { it in availableAwbModes }
    }

    // The AWB mode actually sent to the HAL (Kelvin's OFF is substituted with
    // the fixed D65 preset because vendor HALs ignore AWB-OFF).
    private fun effectiveAwbMode(): Int {
        return if (currentAwbMode == CaptureRequest.CONTROL_AWB_MODE_OFF) {
            pendingHoldMode() ?: CaptureRequest.CONTROL_AWB_MODE_AUTO
        } else {
            currentAwbMode
        }
    }

    private fun CaptureRequest.Builder.applyAwb() {
        if (currentAwbMode == CaptureRequest.CONTROL_AWB_MODE_OFF) {
            // Kelvin: hold the HAL at its fixed D65 preset. A fixed preset is
            // non-adaptive, so no AWB lock is needed; explicitly release any
            // stray latch so the HAL recomputes at the new mode (otherwise the
            // previous mode's converged gains persist into Kelvin). The app's
            // relative Bradford CAT then shifts D65 -> user Kelvin on top of
            // balanced input, keeping Kelvin fully manual but device-accurate.
            set(CaptureRequest.CONTROL_AWB_MODE, effectiveAwbMode())
            set(CaptureRequest.CONTROL_AWB_LOCK, effectiveAwbMode() == CaptureRequest.CONTROL_AWB_MODE_AUTO)
        } else {
            set(CaptureRequest.CONTROL_AWB_MODE, currentAwbMode)
            // AWB_LOCK is sticky in some HALs: if a Kelvin session left it
            // latched, the next mode would never recompute. Explicitly set it
            // every frame so switching modes always re-runs AWB unless the
            // user locked it in AUTO.
            set(CaptureRequest.CONTROL_AWB_LOCK, isAwbLocked)
        }
    }

    // Manual-exposure requests drive AE/ISO directly; the only flash mode that
    // still makes sense is TORCH, which must survive the request switch so the
    // flashlight doesn't die when the user flips to manual exposure.
    private fun CaptureRequest.Builder.applyFlashForManualExposure() {
        set(
            CaptureRequest.FLASH_MODE,
            if (currentFlashMode == FlashMode.TORCH) CaptureRequest.FLASH_MODE_TORCH
            else CaptureRequest.FLASH_MODE_OFF
        )
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
        tapFocusHeld = false
        val camera = cameraDevice ?: return
        val session = captureSession ?: return

        // Focus behaves the same in manual mode: auto when unlocked, frozen when locked,
        // and user-driven (LENS_FOCUS_DISTANCE) when the AF/MF toggle is in manual focus.
        val frozenLens = if (deviceMinFocusDistance > 0f) {
            val distance = if (isManualFocus) currentManualFocusDistance else lastAutoFocusDistance
            if (isManualFocus || focusLocked) distance?.coerceIn(0f, deviceMinFocusDistance) else null
        } else null
        val useManualHold = frozenLens != null
        val afMode = safeAfMode(
            if (useManualHold) CaptureRequest.CONTROL_AF_MODE_OFF
            else CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
        )

        CrashLogger.log(TAG, "setManualExposure iso=$iso exposureTimeNs=$exposureTimeNs frozenLens=${frozenLens ?: "no"} af=$afMode")

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addPreviewTargets()
            set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            applyAwb()
            if (useManualHold) {
                set(CaptureRequest.LENS_FOCUS_DISTANCE, frozenLens)
            }
            set(CaptureRequest.SENSOR_SENSITIVITY, iso)
            set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeNs)
            applyFlashForManualExposure()
            applyPreviewCrop()
        }

        try {
            session.setRepeatingRequest(request.build(), aeReadoutCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "setManualExposure setRepeatingRequest failed: ${e.message}", e)
        }
    }

    // --- SCALER_CROP_REGION zoom ---
    //
    // The HAL never receives SCALER_CROP_REGION on the repeating preview
    // request — zoom is applied purely in the RAW/DNG pipeline while the HAL
    // always stays full-frame. On this device, an active crop on the repeating
    // request makes the HAL ignore tap AF regions and lock the full-frame
    // dominant (near) subject. Our GL preview already zooms the RAW itself, so
    // during preview the HAL stays full-frame (halPreviewCrop=false); only the
    // still-capture request keeps the crop so recorded files match the zoomed
    // view.
    private var halPreviewCrop = false

    private fun CaptureRequest.Builder.applyPreviewCrop() {
        if (halPreviewCrop) {
            applyCropRegion()
        }
    }

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
    private var focusGeneration = 0
    private var activeScanGeneration = 0
    // The current scan's TRIGGER_START / AE pre-capture START have been emitted
    // once; later re-applications of the same scan omit the trigger keys (a
    // trigger acts on the first frame only and re-firing would restart the scan).
    private var scanTriggerFired = false
    // A settled tap-to-focus lock carried on the repeating request. Once a tap
    // converges, applyPreviewRequest() re-applies CONTROL_AF_MODE_AUTO + the
    // tapped region + CONTROL_AF_TRIGGER_IDLE instead of reverting to plain
    // CONTINUOUS_PICTURE, so the continuous stream no longer re-scans over the
    // tap (this mirrors the reference camera app's single-request AF hold).
    private var tapFocusHeld = false

    fun setMeteringRegion(rect: MeteringRectangle?) {
        if (rect == null) {
            meteringRegions = null
            aeRegions = null
            isAfScanning = false
            tapFocusHeld = false
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

        // A fresh tap re-merges AE with the focus region: both indicators
        // reappear at the same spot until the user drags AE away again.
        aeRegions = null
        meteringRegions = arrayOf(rect)
        triggerRegionFocus(rect)
    }

    /**
     * Retarget the stored metering region (e.g. while zooming) for the next
     * tap-trigger. Digital zoom keeps the same focus distance, so no scan is
     * started and the repeating request is never touched.
     */
    fun updateMeteringRegion(rect: MeteringRectangle) {
        meteringRegions = arrayOf(rect)
    }

    /**
     * Live-drag: retarget the active metering region without restarting the
     * scan. Unlike updateMeteringRegion (zoom, storage-only), the repeating
     * preview request is re-applied so the box follows on the HAL. No-op while
     * no scan or hold is active (e.g. when focus is locked).
     */
    fun moveMeteringRegion(rect: MeteringRectangle) {
        if (meteringRegions == null) return
        meteringRegions = arrayOf(rect)
        if (isAfScanning || tapFocusHeld) applyPreviewRequest(log = false)
    }

    /**
     * Set the independent auto-exposure metering region (auto exposure mode
     * only). When the user taps to focus, [setMeteringRegion] clears this so AE
     * re-merges with the tapped focus region; dragging the AE indicator sets it
     * separately.
     */
    fun setAeRegion(rect: MeteringRectangle) {
        if (isManualExposure || focusLocked) return
        val maxAeRegions = cameraCharacteristics?.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0
        if (maxAeRegions <= 0) {
            CrashLogger.log(TAG, "setAeRegion: device has 0 AE regions, ignoring")
            return
        }
        aeRegions = arrayOf(rect)
        applyPreviewRequest()
    }

    /** Live-drag: retarget the independent AE region without restarting any scan. */
    fun moveAeRegion(rect: MeteringRectangle) {
        if (isManualExposure || focusLocked) return
        // Outside MF a live AE drag needs a parked focus region to build on; in MF the
        // AE region can float freely since focus is frozen.
        if (meteringRegions == null && !isManualFocus) return
        aeRegions = arrayOf(rect)
        applyPreviewRequest(log = false)
    }

    /**
     * Tap-to-focus, Camera2Basic style: the repeating preview
     * request is re-applied as AF_MODE_AUTO + the tapped region + a one-time
     * AF_TRIGGER_START, and the repeating stream itself drives the scan at full
     * frame rate to a terminal AF state. When CONTROL_AF_STATE reports
     * FOCUSED_LOCKED / NOT_FOCUSED_LOCKED the same request is re-applied with
     * AF_TRIGGER_IDLE so the tapped focus is held.
     *
     * No stopRepeating and no one-shot continuation chain: those serialized
     * captures dropped preview frames (~130 ms gaps, choppy) and this HAL never
     * reported a terminal state inside the one-shot chain — it stayed
     * ACTIVE_SCAN through all continuations, so every tap hit the timeout path.
     * The in-flight scan only settles once the AUTO + region + TRIGGER_IDLE
     * repeating request is re-armed, so the repeating stream is the right home
     * for the whole scan.
     *
     * Stall prevention: an AUTO + region repeating request only scans when
     * TRIGGER_START is present (one-shot, first frame only), so it cannot
     * autonomously re-sweep like CONTINUOUS + regions did (the frame-stall
     * regression). TRIGGER_IDLE holds the settle without aborting an in-flight
     * scan — the HAL finishes the current sweep and then locks.
     */
    private fun triggerRegionFocus(rect: MeteringRectangle) {
        if (isManualExposure || focusLocked) return
        val camera = cameraDevice ?: return
        val session = captureSession ?: return

        meteringRegions = arrayOf(rect)
        tapFocusHeld = false
        isAfScanning = true
        scanTriggerFired = false
        val generation = ++focusGeneration
        activeScanGeneration = generation

        CrashLogger.log(TAG,
            "triggerRegionFocus: repeating request -> AUTO+region+AF_TRIGGER_START rawOnly=$sessionRawOnly")

        // Re-apply the repeating request with the AF trigger. A trigger in a
        // repeating request only acts on the first frame that carries it.
        applyPreviewRequest()

        // Safety net: if the HAL never reports a terminal AF state, settle the
        // hold via TRIGGER_IDLE so the stream is stable and not mid-scan.
        backgroundHandler?.postDelayed({
            if (generation == activeScanGeneration && isAfScanning) {
                CrashLogger.log(TAG, "triggerRegionFocus: scan timeout, settling via TRIGGER_IDLE hold")
                finishScan()
            }
        }, 3000)
    }

    /**
     * Called when the scan reaches a terminal AF state (or the safety-net
     * timeout): stop treating the stream as mid-scan and flip the repeating
     * request to the AUTO + region + TRIGGER_IDLE hold.
     */
    private fun finishScan() {
        if (!isAfScanning) return
        isAfScanning = false
        holdFocus()
    }

    /**
     * Marks the tapped region as a settled hold and re-applies the repeating
     * request. applyPreviewRequest() is the single builder for the repeating
     * request, and once tapFocusHeld is set it keeps emitting AF_MODE_AUTO +
     * the tapped region + AF_TRIGGER_IDLE, so no later re-apply (zoom, EV, WB,
     * flash, lens metadata refresh) can revert the stream to CONTINUOUS_PICTURE
     * and let it re-scan over the tap.
     */
    private fun holdFocus() {
        if (focusLocked || isManualExposure) return
        if (meteringRegions == null) return
        tapFocusHeld = true
        CrashLogger.log(TAG, "holdFocus: held tap, re-applying repeating request as AUTO+region+TRIGGER_IDLE")
        applyPreviewRequest()
    }

    // Clear any tap region and return to continuous AF.
    fun clearFocusRegion() {
        meteringRegions = null
        aeRegions = null
        isAfScanning = false
        focusLocked = false
        tapFocusHeld = false
        applyPreviewRequest() // back to continuous picture, no regions
    }

    // Lock focus (manual focus hold) only. The HAL reports afState
    // FOCUSED_LOCKED while still moving the lens on this device, so a
    // trigger-based lock is not reliable: freeze the last auto-focus lens
    // distance by switching to AF_MODE_OFF + LENS_FOCUS_DISTANCE, and switch
    // back to auto focus on unlock. Exposure keeps auto-metering normally.
    fun lockFocus() {
        if (meteringRegions == null) return
        focusLocked = true
        tapFocusHeld = false
        // A tap-scan may still be mid-flight. Invalidate its trigger callback so
        // the manual-focus lock request below is the one that survives.
        ++focusGeneration
        isAfScanning = false
        applyPreviewRequest()
    }

    // Unlock focus, resume auto/region AF
    fun unlockFocus() {
        focusLocked = false
        tapFocusHeld = false
        applyPreviewRequest()
    }

    // Drive the lens to an explicit focus distance (diopters). Switches to
    // AF_MODE_OFF + LENS_FOCUS_DISTANCE and cancels any in-flight tap scan or
    // focus lock; independent AE metering regions are preserved so the AE
    // indicator keeps working in manual focus.
    fun setManualFocus(distance: Float) {
        isManualFocus = true
        currentManualFocusDistance = distance
        tapFocusHeld = false
        focusLocked = false
        isAfScanning = false
        scanTriggerFired = false
        // The AE indicator stays visible in manual focus: keep metering the point
        // it was last parked on, even if that was the tapped focus region.
        if (aeRegions == null && meteringRegions != null) {
            aeRegions = meteringRegions
        }
        meteringRegions = null
        ++focusGeneration
        CrashLogger.log(TAG, "setManualFocus distance=$distance minDistance=$deviceMinFocusDistance")
        applyPreviewRequest()
    }

    // Return to auto focus (CONTINUOUS_PICTURE unless a tap region is parked).
    fun resetAutoFocus() {
        if (!isManualFocus) return
        isManualFocus = false
        CrashLogger.log(TAG, "resetAutoFocus")
        applyPreviewRequest()
    }

    private fun applyPreviewRequest(log: Boolean = true) {
        if (isManualExposure) {
            setManualExposure(currentManualIso, currentManualExposureNs)
            return
        }
        val camera = cameraDevice ?: run { CrashLogger.log(TAG, "applyPreviewRequest: cameraDevice null"); return }
        val session = captureSession ?: run { CrashLogger.log(TAG, "applyPreviewRequest: session null"); return }

        val frozenLens = if ((isManualFocus || focusLocked) && deviceMinFocusDistance > 0f) {
            // Manual focus pins the roller distance; focus-lock freezes the last
            // auto-focus position. Both switch the HAL to AF_MODE_OFF + LENS_FOCUS_DISTANCE.
            val distance = if (isManualFocus) currentManualFocusDistance else lastAutoFocusDistance
            distance?.coerceIn(0f, deviceMinFocusDistance)
        } else null
        // Only drop into manual focus when we can actually pin a distance.
        val useManualHold = frozenLens != null
        // Active tap-trigger scan: keep emitting AUTO + region on the repeating
        // stream so the scan is driven at full frame rate. The AF trigger START
        // is carried once (first frame); later re-applications omit the trigger.
        val scanning = isAfScanning && meteringRegions != null && !useManualHold && deviceMaxAfRegions > 0
        // A settled tap-to-focus hold: keep emitting AUTO + region + TRIGGER_IDLE
        // so the continuous background scan cannot re-sweep over the tap.
        val useTapHold = !scanning && tapFocusHeld && meteringRegions != null && !useManualHold && deviceMaxAfRegions > 0
        val carryRegion = scanning || useTapHold
        val tapRegions = if (carryRegion) meteringRegions else null

        val afMode = if (deviceMaxAfRegions > 0) {
            safeAfMode(
                when {
                    useManualHold -> CaptureRequest.CONTROL_AF_MODE_OFF
                    carryRegion -> CaptureRequest.CONTROL_AF_MODE_AUTO
                    else -> CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                }
            )
        } else {
            CaptureRequest.CONTROL_AF_MODE_OFF
        }

        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addPreviewTargets()
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            if (useManualHold) {
                // Manual focus hold: copy the last auto-focus lens position so the
                // HAL has no reason to move the lens. Clamp to the device's range.
                set(CaptureRequest.LENS_FOCUS_DISTANCE, frozenLens)
            } else if (carryRegion) {
                set(CaptureRequest.CONTROL_AF_REGIONS, tapRegions)
                if (scanning && !scanTriggerFired) {
                    // Kick the scan: START acts on the first frame of the request.
                    set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
                } else if (useTapHold) {
                    // Settled hold: TRIGGER_IDLE per spec does not re-scan.
                    set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_IDLE)
                }
            }
            applyAwb()
            if (isManualExposure) {
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
            } else {
                if (currentExposureComp != 0) {
                    set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, currentExposureComp)
                }
                // AE regions: independent region if the user parked one, otherwise
                // follow the tapped focus (scan/hold) region.
                if (deviceMaxAeRegions > 0) {
                    val aeRegionToUse = aeRegions ?: (if (carryRegion) tapRegions else null)
                    if (aeRegionToUse != null) {
                        set(CaptureRequest.CONTROL_AE_REGIONS, aeRegionToUse)
                    }
                }
                if (carryRegion) {
                    if (scanning && !scanTriggerFired) {
                        set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER_START)
                    } else if (useTapHold) {
                        set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER_IDLE)
                    }
                }
                currentFlashMode.applyToRequest(this, availableAeModes)
            }
            applyPreviewCrop()
        }

        if (scanning) scanTriggerFired = true

        if (log) CrashLogger.log(TAG, "applyPreviewRequest: awb=${awbModeName(effectiveAwbMode())} af=$afMode locked=$focusLocked tapHold=$useTapHold scan=$scanning frozenLens=${frozenLens ?: "no"} awbLocked=$isAwbLocked manual=$isManualExposure iso=$currentManualIso shutter=${currentManualExposureNs}")

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
        tapFocusHeld = false
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
            addPreviewTargets()
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            applyAwb()
            setMeteringRegions(arrayOf(meteringRect))
            if (deviceMaxAfRegions > 0) {
                set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER_START)
                set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
            }
            if (isManualExposure) {
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
                applyFlashForManualExposure()
            } else {
                currentFlashMode.applyToRequest(this, availableAeModes)
            }
            applyPreviewCrop()
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
                            addPreviewTargets()
                            set(CaptureRequest.CONTROL_AF_MODE, afMode)
                            applyAwb()
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
                                applyFlashForManualExposure()
                            } else {
                                currentFlashMode.applyToRequest(this, availableAeModes)
                            }
                            applyPreviewCrop()
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
            addPreviewTargets()
            set(CaptureRequest.CONTROL_AF_MODE, afMode)
            applyAwb()
            setMeteringRegions(arrayOf(meteringRect))
            if (deviceMaxAfRegions > 0) {
                set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER_START)
                set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
            }
            if (isManualExposure) {
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
                applyFlashForManualExposure()
            } else {
                currentFlashMode.applyToRequest(this, availableAeModes)
            }
            applyPreviewCrop()
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
                            addPreviewTargets()
                            set(CaptureRequest.CONTROL_AF_MODE, afMode)
                            applyAwb()
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
                                applyFlashForManualExposure()
                            } else {
                                currentFlashMode.applyToRequest(this, availableAeModes)
                            }
                            applyPreviewCrop()
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
            applyAwb()
            set(CaptureRequest.CONTROL_AE_MODE, aeMode)
            if (isManualExposure) {
                set(CaptureRequest.SENSOR_SENSITIVITY, currentManualIso)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, currentManualExposureNs)
                applyFlashForManualExposure()
            } else {
                currentFlashMode.applyToRequest(this, availableAeModes)
                // ON must fire regardless of AE flash-mode support: override
                // with a single-shot flash on the still request (the most
                // device-agnostic way to force one flash at capture).
                if (currentFlashMode == FlashMode.ON) {
                    set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                    set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_SINGLE)
                }
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

    private fun resetAfScanState() {
        isAfScanning = false
        tapFocusHeld = false
        scanTriggerFired = false
    }

    fun resetHard() {
        resetAfScanState()
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

        try {
            rawReader?.close()
        } catch (_: Exception) {}
        rawReader = null

        availableAfModes = intArrayOf()
        availableAeModes = intArrayOf()
        availableAwbModes = emptySet()
    }

    fun close() {
        resetAfScanState()
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

        try {
            rawReader?.close()
        } catch (_: Exception) {}
        rawReader = null

        availableAfModes = intArrayOf()
        availableAeModes = intArrayOf()
        availableAwbModes = emptySet()
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
        else -> "UNKNOWN"
    }

    private fun awbStateName(state: Int?): String {
        return when (state) {
            CaptureResult.CONTROL_AWB_STATE_INACTIVE -> "INACTIVE"
            CaptureResult.CONTROL_AWB_STATE_SEARCHING -> "SEARCHING"
            CaptureResult.CONTROL_AWB_STATE_CONVERGED -> "CONVERGED"
            CaptureResult.CONTROL_AWB_STATE_LOCKED -> "LOCKED"
            else -> "UNKNOWN"
        }
    }

    private fun CaptureRequest.Builder.setMeteringRegions(regions: Array<MeteringRectangle>?) {
        if (deviceMaxAfRegions > 0) set(CaptureRequest.CONTROL_AF_REGIONS, regions)
        if (deviceMaxAeRegions > 0) set(CaptureRequest.CONTROL_AE_REGIONS, regions)
    }

    private val aeReadoutCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            val afState = result.get(CaptureResult.CONTROL_AF_STATE)
            val lensFocusD = result.get(CaptureResult.LENS_FOCUS_DISTANCE)
            if (lensFocusD != null && !isManualFocus && afState != CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN &&
                afState != CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN) {
                lastAutoFocusDistance = lensFocusD
            }
            // A tap-trigger scan reaches a terminal state: settle the hold by
            // flipping the repeating request to AUTO + region + TRIGGER_IDLE.
            if (isAfScanning && (afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED ||
                afState == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED)
            ) {
                CrashLogger.log(TAG, "aeReadoutCallback: afState=$afState terminal, scan settled -> holding")
                finishScan()
            }
            val cct = result.get(CaptureResult.COLOR_CORRECTION_TRANSFORM)
            latestColorCorrectionMatrix = cct?.let { colorSpaceToRowMajor(it) }
            val ccg = result.get(CaptureResult.COLOR_CORRECTION_GAINS)
            latestColorCorrectionGains = ccg?.let {
                floatArrayOf(it.red, it.greenEven, it.greenOdd, it.blue)
            }
            ccLogCount++
            if (ccLogCount % 60 == 0) {
                CrashLogger.log(TAG,
                    "cc: awbState=${awbStateName(result.get(CaptureResult.CONTROL_AWB_STATE))} " +
                    "awbMode=${awbModeName(result.get(CaptureResult.CONTROL_AWB_MODE) ?: -1)} " +
                    "gains=[${latestColorCorrectionGains?.joinToString { String.format("%.3f", it) }}] " +
                    "mat=[${latestColorCorrectionMatrix?.joinToString { String.format("%.4f", it) }}]"
                )
            }

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

            val afStateAtTop = afState
            val afMode = result.get(CaptureResult.CONTROL_AF_MODE)
            val aeState = result.get(CaptureResult.CONTROL_AE_STATE)
            val reportedAfRegions = result.get(CaptureResult.CONTROL_AF_REGIONS)
            val reportedAeRegions = result.get(CaptureResult.CONTROL_AE_REGIONS)

            CrashLogger.log(TAG,
                "onCaptureCompleted: afState=$afStateAtTop afMode=$afMode aeState=$aeState lensFocus=$lensFocusD " +
                "afRegions=${reportedAfRegions?.contentToString()} " +
                "aeRegions=${reportedAeRegions?.contentToString()}"
            )
        }
    }

    companion object {
        private const val TAG = "Camera2Manager"

        private val COLOR_IDENTITY_9 = floatArrayOf(
            1f, 0f, 0f,
            0f, 1f, 0f,
            0f, 0f, 1f
        )

        private fun colorSpaceToRowMajor(cst: android.hardware.camera2.params.ColorSpaceTransform): FloatArray {
            val matrix = FloatArray(9)
            for (row in 0 until 3) {
                for (col in 0 until 3) {
                    val r = cst.getElement(col, row)
                    if (r.denominator == 0) return COLOR_IDENTITY_9.clone()
                    matrix[row * 3 + col] = r.numerator.toFloat() / r.denominator.toFloat()
                }
            }
            return matrix
        }
    }
}