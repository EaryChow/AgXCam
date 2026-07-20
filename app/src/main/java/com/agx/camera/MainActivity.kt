package com.agx.camera

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.OrientationEventListener
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.TextureView
import android.view.View
import android.widget.*
import android.view.ViewGroup
import android.view.Gravity
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.agx.camera.camera.*
import com.agx.camera.color.AgxParams
import com.agx.camera.color.AgxPrecomputer
import com.agx.camera.color.ColorMatrix
import com.agx.camera.color.WhiteBalanceMath
import com.agx.camera.gpu.DegradedPreviewManager
import com.agx.camera.gpu.PreviewRenderer
import com.agx.camera.io.CaptureMetadata
import com.agx.camera.io.ExifWriter
import com.agx.camera.io.JpegEncoder
import com.agx.camera.io.MediaStoreSaver
import com.agx.camera.thermal.ThermalManager
import com.agx.camera.CrashLogger
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {

    private lateinit var camera2Manager: Camera2Manager
    private lateinit var lensManager: LensManager
    private lateinit var previewRenderer: PreviewRenderer
    private lateinit var presetManager: PresetManager
    private lateinit var thermalManager: ThermalManager
    private lateinit var shutterController: ShutterController
    private lateinit var autofocusController: AutofocusController
    private lateinit var grayCardSampler: GrayCardSampler
    private lateinit var mediaStoreSaver: MediaStoreSaver
    private lateinit var developerSwitch: DeveloperSwitch
    private val mainHandler = Handler(Looper.getMainLooper())

    private var currentFlashMode = FlashMode.OFF
    private var currentWbMode = WhiteBalanceMode.AUTO
    private var kelvinState = KelvinState()
    private var agxParams = AgxParams()
    private var photoOutput = PhotoOutputSettings()
    private var cameraReady = false
    private var openingCamera = false
    private var settingsPanelOpen = false
    private var isCapturing = false
    private var pendingPauseCleanup = false
    private var errorDialogShowing = false
    private var currentDeviceOrientation = 0

    private lateinit var orientationListener: OrientationEventListener

    private lateinit var textureView: TextureView
    private lateinit var devBanner: TextView
    private lateinit var disconnectBanner: TextView
    private lateinit var flashButton: ImageView
    private lateinit var wbButton: TextView
    private lateinit var settingsButton: ImageView
    private lateinit var frontRearToggle: ImageView
    private lateinit var zoomLabel: TextView
    private lateinit var zoomSlider: SeekBar
    private lateinit var zoomRow: View
    private lateinit var lensSelector: LinearLayout
    private lateinit var settingsPanel: ScrollView
    private lateinit var finishingCaptureOverlay: TextView
    private lateinit var lensSwitchOverlay: TextView

    // Shutter / Thermal
    private lateinit var shutterButton: ImageView
    private lateinit var thumbnailButton: ImageView
    private lateinit var cooldownText: TextView
    private lateinit var shutterStateLabel: TextView
    private lateinit var thermalIndicator: TextView

    // Focus / WB
    private lateinit var focusIndicator: ImageView
    private lateinit var aeAfLockButton: ImageView
    private lateinit var wbPopup: LinearLayout

    // Manual Exposure
    private lateinit var amToggleButton: TextView
    private lateinit var isoOverlay: TextView
    private lateinit var isoPopup: View
    private lateinit var isoSeekBar: SeekBar
    private lateinit var shutterOverlay: TextView
    private lateinit var shutterPopup: View
    private lateinit var shutterSeekBar: SeekBar
    private lateinit var evPopup: LinearLayout
    private lateinit var evSeekBar: SeekBar
    private lateinit var evSliderContainer: FrameLayout
    private lateinit var evSeekBarVertical: SeekBar

    private var isManualMode = false
    private var lastAutoIso = 200
    private var lastAutoShutterNs = 33_333_333L
    private var lastIsoTapTime = 0L
    private var lastShutterTapTime = 0L
    private var isoPopupShowing = false
    private var shutterPopupShowing = false
    private var evPopupShowing = false

    // Pinch-to-zoom
    private var scaleGestureDetector: ScaleGestureDetector? = null
    private var isScaling = false
    private var aeLocked = false

    // Tap-to-focus capability for current lens
    private var currentLensCanTapToFocus = true

    // Warning toast
    private lateinit var warningContainer: FrameLayout
    private var warningDismissRunnable: Runnable? = null

    private var presetInitialLoad = true
    private var lastSelectedResolution = -1
    private var currentResolutionOptions = emptyList<LensManager.ResolutionOption>()
    private var currentResolutionIndex = 0

    // Preview size cap
    private lateinit var maxPreviewDimensions: Pair<Int, Int>

    // Preset
    private lateinit var presetSpinner: Spinner
    private lateinit var presetAddBtn: TextView
    private lateinit var presetSaveBtn: TextView
    private lateinit var presetDeleteBtn: TextView
    private lateinit var presetRenameBtn: TextView
    private lateinit var presetResetBtn: TextView

    // Curve
    private lateinit var contrastLabel: TextView
    private lateinit var contrastSlider: SeekBar
    private lateinit var toeLabel: TextView
    private lateinit var toeSlider: SeekBar
    private lateinit var shoulderLabel: TextView
    private lateinit var shoulderSlider: SeekBar

    // Inset
    private lateinit var usePreForPostCb: CheckBox
    private lateinit var insetRotRLabel: TextView; private lateinit var insetRotRSlider: SeekBar
    private lateinit var insetRotGLabel: TextView; private lateinit var insetRotGSlider: SeekBar
    private lateinit var insetRotBLabel: TextView; private lateinit var insetRotBSlider: SeekBar
    private lateinit var insetPurRLabel: TextView; private lateinit var insetPurRSlider: SeekBar
    private lateinit var insetPurGLabel: TextView; private lateinit var insetPurGSlider: SeekBar
    private lateinit var insetPurBLabel: TextView; private lateinit var insetPurBSlider: SeekBar

    // Outset
    private lateinit var outsetSection: LinearLayout
    private lateinit var outsetRotRLabel: TextView; private lateinit var outsetRotRSlider: SeekBar
    private lateinit var outsetRotGLabel: TextView; private lateinit var outsetRotGSlider: SeekBar
    private lateinit var outsetRotBLabel: TextView; private lateinit var outsetRotBSlider: SeekBar
    private lateinit var outsetPurRLabel: TextView; private lateinit var outsetPurRSlider: SeekBar
    private lateinit var outsetPurGLabel: TextView; private lateinit var outsetPurGSlider: SeekBar
    private lateinit var outsetPurBLabel: TextView; private lateinit var outsetPurBSlider: SeekBar
    private lateinit var copyInsetBtn: TextView

    // Tinting
    private lateinit var tintingScaleLabel: TextView; private lateinit var tintingScaleSlider: SeekBar
    private lateinit var tintingHueLabel: TextView; private lateinit var tintingHueSlider: SeekBar

    // NR
    private lateinit var nrLabel: TextView; private lateinit var nrSlider: SeekBar

    // WB
    private lateinit var kelvinLabel: TextView; private lateinit var kelvinSlider: SeekBar
    private lateinit var tintLabel: TextView; private lateinit var tintSlider: SeekBar

    // Output
    private lateinit var jpegLabel: TextView; private lateinit var jpegSlider: SeekBar
    private lateinit var resolutionSpinner: Spinner

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        CrashLogger.log(TAG, "onCreate: begin")
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(0, (systemBars.top * 0.7f).toInt(), 0, (systemBars.bottom * 0.7f).toInt())
            insets
        }

        textureView = findViewById(R.id.preview_texture)
        devBanner = findViewById(R.id.dev_banner)
        disconnectBanner = findViewById(R.id.disconnect_banner)
        warningContainer = findViewById(R.id.warningContainer)
        flashButton = findViewById(R.id.flash_button)
        wbButton = findViewById(R.id.wb_button)
        settingsButton = findViewById(R.id.settings_button)
        frontRearToggle = findViewById(R.id.front_rear_toggle)
        zoomLabel = findViewById(R.id.zoom_label)
        zoomSlider = findViewById(R.id.zoom_slider)
        zoomRow = findViewById(R.id.zoom_row)
        lensSelector = findViewById(R.id.lens_selector)
        settingsPanel = findViewById(R.id.settings_panel)
        finishingCaptureOverlay = findViewById(R.id.finishing_capture_overlay)
        lensSwitchOverlay = findViewById(R.id.lens_switch_overlay)

        presetSpinner = findViewById(R.id.preset_spinner)
        presetAddBtn = findViewById(R.id.preset_add_btn)
        presetSaveBtn = findViewById(R.id.preset_save_btn)
        presetDeleteBtn = findViewById(R.id.preset_delete_btn)
        presetRenameBtn = findViewById(R.id.preset_rename_btn)
        presetResetBtn = findViewById(R.id.preset_reset_btn)

        contrastLabel = findViewById(R.id.contrast_label); contrastSlider = findViewById(R.id.contrast_slider)
        toeLabel = findViewById(R.id.toe_label); toeSlider = findViewById(R.id.toe_slider)
        shoulderLabel = findViewById(R.id.shoulder_label); shoulderSlider = findViewById(R.id.shoulder_slider)

        usePreForPostCb = findViewById(R.id.use_pre_for_post_cb)
        insetRotRLabel = findViewById(R.id.inset_rot_r_label); insetRotRSlider = findViewById(R.id.inset_rot_r_slider)
        insetRotGLabel = findViewById(R.id.inset_rot_g_label); insetRotGSlider = findViewById(R.id.inset_rot_g_slider)
        insetRotBLabel = findViewById(R.id.inset_rot_b_label); insetRotBSlider = findViewById(R.id.inset_rot_b_slider)
        insetPurRLabel = findViewById(R.id.inset_pur_r_label); insetPurRSlider = findViewById(R.id.inset_pur_r_slider)
        insetPurGLabel = findViewById(R.id.inset_pur_g_label); insetPurGSlider = findViewById(R.id.inset_pur_g_slider)
        insetPurBLabel = findViewById(R.id.inset_pur_b_label); insetPurBSlider = findViewById(R.id.inset_pur_b_slider)

        outsetSection = findViewById(R.id.outset_section)
        outsetRotRLabel = findViewById(R.id.outset_rot_r_label); outsetRotRSlider = findViewById(R.id.outset_rot_r_slider)
        outsetRotGLabel = findViewById(R.id.outset_rot_g_label); outsetRotGSlider = findViewById(R.id.outset_rot_g_slider)
        outsetRotBLabel = findViewById(R.id.outset_rot_b_label); outsetRotBSlider = findViewById(R.id.outset_rot_b_slider)
        outsetPurRLabel = findViewById(R.id.outset_pur_r_label); outsetPurRSlider = findViewById(R.id.outset_pur_r_slider)
        outsetPurGLabel = findViewById(R.id.outset_pur_g_label); outsetPurGSlider = findViewById(R.id.outset_pur_g_slider)
        outsetPurBLabel = findViewById(R.id.outset_pur_b_label); outsetPurBSlider = findViewById(R.id.outset_pur_b_slider)
        copyInsetBtn = findViewById(R.id.copy_inset_to_outset_btn)

        tintingScaleLabel = findViewById(R.id.tinting_scale_label); tintingScaleSlider = findViewById(R.id.tinting_scale_slider)
        tintingHueLabel = findViewById(R.id.tinting_hue_label); tintingHueSlider = findViewById(R.id.tinting_hue_slider)

        nrLabel = findViewById(R.id.nr_label); nrSlider = findViewById(R.id.nr_slider)

        kelvinLabel = findViewById(R.id.kelvin_label); kelvinSlider = findViewById(R.id.kelvin_slider)
        tintLabel = findViewById(R.id.tint_label); tintSlider = findViewById(R.id.tint_slider)

        jpegLabel = findViewById(R.id.jpeg_label);         jpegSlider = findViewById(R.id.jpeg_slider)
        resolutionSpinner = findViewById(R.id.resolution_spinner)

        shutterButton = findViewById(R.id.shutter_button)
        thumbnailButton = findViewById(R.id.thumbnail_button)
        cooldownText = findViewById(R.id.cooldown_text)
        shutterStateLabel = findViewById(R.id.shutter_state_label)
        thermalIndicator = findViewById(R.id.thermal_indicator)

        focusIndicator = findViewById(R.id.focus_indicator)
        aeAfLockButton = findViewById(R.id.ae_af_lock_button)
        wbPopup = findViewById(R.id.wb_popup)

        amToggleButton = findViewById(R.id.am_toggle_button)
        isoOverlay = findViewById(R.id.iso_overlay)
        isoPopup = findViewById(R.id.iso_popup)
        isoSeekBar = findViewById(R.id.iso_seekbar)
        shutterOverlay = findViewById(R.id.shutter_overlay)
        shutterPopup = findViewById(R.id.shutter_popup)
        shutterSeekBar = findViewById(R.id.shutter_seekbar)
        evPopup = findViewById(R.id.ev_popup)
        evSeekBar = findViewById(R.id.ev_seekbar)
        evSliderContainer = findViewById(R.id.ev_slider_container)
        evSeekBarVertical = findViewById(R.id.ev_seekbar_vertical)

        if (BuildConfig.AGX_ENABLE_YUV_FALLBACK) {
            devBanner.visibility = View.VISIBLE
        }

        lensManager = LensManager(this)
        camera2Manager = Camera2Manager(this)
        presetManager = PresetManager(this)
        presetManager.ensureDefault()

        thermalManager = ThermalManager(this)
        shutterController = ShutterController(mainHandler)
        autofocusController = AutofocusController(camera2Manager, mainHandler)
        grayCardSampler = GrayCardSampler()
        mediaStoreSaver = MediaStoreSaver(this)

        // Compute max preview dimensions based on screen resolution
        maxPreviewDimensions = getMaxPreviewDimensions()

        developerSwitch = DeveloperSwitch(this) { useRaw ->
            CrashLogger.log(TAG, "Developer switch toggled: useRaw=$useRaw")
            Log.d(TAG, "Developer switch toggled: useRaw=$useRaw")

            // Update front/rear toggle visibility based on RAW front lens support
            updateFrontRearToggleVisibility(useRaw)

            // If enabling RAW and current lens doesn't support RAW, switch to closest RAW lens
            if (useRaw) {
                val currentLens = lensManager.activeLens
                if (currentLens != null && !currentLens.hasRawSensor) {
                    val rawLens = lensManager.getClosestRawLens(currentLens)
                    if (rawLens != null) {
                        Log.d(TAG, "RAW mode: switching from non-RAW lens ${currentLens.cameraId} to RAW lens ${rawLens.cameraId}")
                        switchToLens(rawLens)
                        developerSwitch.updateBannerForRawMode(true)
                        return@DeveloperSwitch
                    } else {
                        developerSwitch.revertToggle()
                        developerSwitch.showErrorBanner("No RAW-capable lens found")
                        updateFrontRearToggleVisibility(false)
                        buildLensSelectorUI()
                        return@DeveloperSwitch
                    }
                }
            }

            val lens = lensManager.activeLens ?: return@DeveloperSwitch
            val previewSize = lensManager.getBestPreviewSize(lens, maxPreviewDimensions.first, maxPreviewDimensions.second)
            camera2Manager.close()
            camera2Manager.stopBackgroundThread()
            previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
            previewRenderer.targetAspectRatio = 0f
            if (useRaw) {
                previewRenderer.enableBayerMode(previewSize.width, previewSize.height)
            } else {
                previewRenderer.disableBayerMode()
            }
            previewRenderer.start()
            camera2Manager.startBackgroundThread()
            camera2Manager.openCamera(lens, previewSize, 0, 0)
            developerSwitch.updateBannerForRawMode(useRaw)
            buildLensSelectorUI()
        }
        developerSwitch.init(
            findViewById(R.id.developer_raw_toggle_container),
            findViewById(R.id.developer_raw_toggle),
            findViewById(R.id.developer_banner)
        )
        findViewById<TextView>(R.id.save_debug_log_btn).setOnClickListener {
            CrashLogger.saveDebugLogToDownloads(this)
            Toast.makeText(this, "Debug log saved to Downloads", Toast.LENGTH_SHORT).show()
        }

        thermalManager.onStateChanged = { state ->
            mainHandler.post { updateThermalUI(state) }
        }

        orientationListener = object : OrientationEventListener(this) {
            override fun onOrientationChanged(orientation: Int) {
                if (orientation != ORIENTATION_UNKNOWN) {
                    val degrees = when {
                        orientation in 315..360 || orientation < 45 -> 0
                        orientation in 45..135 -> 90
                        orientation in 135..225 -> 180
                        else -> 270
                    }
                    if (degrees != currentDeviceOrientation) {
                        currentDeviceOrientation = degrees
                    }
                }
            }
        }

        shutterController.onStateChanged = { state ->
            mainHandler.post { updateShutterUI(state) }
        }

        shutterController.onCooldownTick = { remaining ->
            mainHandler.post {
                if (remaining > 0) {
                    cooldownText.text = "${remaining}s"
                    cooldownText.visibility = View.VISIBLE
                } else {
                    cooldownText.visibility = View.GONE
                }
            }
        }

        grayCardSampler.onSampleComplete = { gains ->
            mainHandler.post {
                val grayCardMatrix = ColorMatrix.diagonal(gains.gainR, gains.gainG, gains.gainB)
                val sceneLinearTo709 = WhiteBalanceMath.buildGrayCardSceneLinearTo709(grayCardMatrix)
                previewRenderer.agxSceneLinearTo709 = sceneLinearTo709.m
                Toast.makeText(this, "Gray card WB applied", Toast.LENGTH_SHORT).show()
            }
        }

        previewRenderer = PreviewRenderer(textureView).apply {
            onFirstFrameRendered = { Log.d(TAG, "First frame rendered") }
            onFrameRendered = { ms -> thermalManager.onFrameRendered(ms.toFloat()) }
            onDegradedModeChanged = { banner ->
                mainHandler.post {
                    if (banner != null) {
                        devBanner.text = banner
                        devBanner.visibility = View.VISIBLE
                    } else {
                        devBanner.visibility = View.GONE
                    }
                }
            }
        }
        previewRenderer.degradedManager = DegradedPreviewManager(this)

        textureView.surfaceTextureListener = previewRenderer

        setupUI()
        loadPreset(PresetManager.PRESET_DEFAULT)
        checkPermissions()
    }

    private fun setupUI() {
        flashButton.setOnClickListener {
            currentFlashMode = currentFlashMode.cycle()
            updateFlashUI()
            thermalManager.isTorchActive = (currentFlashMode == FlashMode.TORCH)
            if (cameraReady) camera2Manager.setFlashMode(currentFlashMode)
        }

        wbButton.setOnClickListener {
            showWbPopup()
        }

        settingsButton.setOnClickListener {
            settingsPanelOpen = !settingsPanelOpen
            settingsPanel.visibility = if (settingsPanelOpen) View.VISIBLE else View.GONE
        }

        frontRearToggle.setOnClickListener {
            if (!cameraReady) return@setOnClickListener
            val current = lensManager.activeLens ?: return@setOnClickListener
            val rawOnly = developerSwitch.useRawSensor
            val target = if (current.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_BACK) {
                lensManager.getFrontLens(rawOnly)
            } else {
                val rearId = lensManager.lastUsedRearLensId
                lensManager.lenses.firstOrNull { it.cameraId == rearId && (!rawOnly || it.hasRawSensor) }
                    ?: lensManager.getRearLenses(rawOnly).firstOrNull()
            }
            if (target != null && target.cameraId != current.cameraId) {
                switchToLens(target)
            }
        }

        thumbnailButton.setOnClickListener {
            val uri = mediaStoreSaver.lastSavedUri
            if (uri != null) {
                try {
                    startActivity(Intent(Intent.ACTION_VIEW, uri))
                } catch (e: Exception) {
                    val galleryIntent = Intent(Intent.ACTION_VIEW).apply {
                        type = "image/*"
                    }
                    startActivity(galleryIntent)
                }
            } else {
                val galleryIntent = Intent(Intent.ACTION_VIEW).apply {
                    type = "image/*"
                }
                startActivity(galleryIntent)
            }
        }

        shutterButton.setOnClickListener {
            if (shutterController.state != ShutterController.State.IDLE) return@setOnClickListener
            if (thermalManager.isCaptureBlocked) {
                Toast.makeText(this, "Device too hot — wait for cooldown", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (currentWbMode == WhiteBalanceMode.GRAY_CARD && grayCardSampler.isActive) {
                Toast.makeText(this, "Tap the preview to sample gray card", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            shutterController.onCaptureSubmitted()
            isCapturing = true

            val session = CaptureSession(
                flashMode = camera2Manager.currentFlashModeForExif,
                jpegQuality = photoOutput.jpegQuality,
                resolutionWidth = photoOutput.resolutionWidth,
                resolutionHeight = photoOutput.resolutionHeight,
                deviceOrientation = currentDeviceOrientation,
                sensorOrientation = lensManager.activeLens?.let { lensManager.getSensorOrientation(it) } ?: 0,
                focalLengthMm = lensManager.activeLens?.let {
                    lensManager.getCharacteristicsForLens(it)
                        ?.get(android.hardware.camera2.CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                        ?.getOrNull(0)
                } ?: 4.0f,
                zoomFactor = previewRenderer.zoomController.zoomFactor,
                zoomCenterX = previewRenderer.zoomController.zoomCenterX,
                zoomCenterY = previewRenderer.zoomController.zoomCenterY,
                agxSceneLinearTo709 = previewRenderer.agxSceneLinearTo709.copyOf(),
                agxInsetMat = previewRenderer.agxInsetMat.copyOf(),
                agxOutsetMat = previewRenderer.agxOutsetMat.copyOf(),
                agxToRec2020 = previewRenderer.agxToRec2020.copyOf(),
                agxWhiteLevel = previewRenderer.agxWhiteLevel,
                agxBlackLevel = previewRenderer.agxBlackLevel,
                agxLogMin = previewRenderer.agxLogMin,
                agxLogMax = previewRenderer.agxLogMax,
                agxLogMidgray = previewRenderer.agxLogMidgray,
                agxDisplayMidgray = previewRenderer.agxDisplayMidgray,
                agxContrast = previewRenderer.agxContrast,
                agxToe = previewRenderer.agxToe,
                agxShoulder = previewRenderer.agxShoulder,
                isFrontCamera = previewRenderer.isFrontCamera
            )

            val thumbnailLatch = CountDownLatch(1)
            val thumbnailRef = java.util.concurrent.atomic.AtomicReference<Bitmap?>(null)
            previewRenderer.pendingFboReadback = { fboTexId, w, h ->
                thumbnailRef.set(JpegEncoder.readFboToBitmapFlipped(fboTexId, w, h))
                thumbnailLatch.countDown()
            }

            camera2Manager.captureStill(
                onCaptureAvailable = { image, result ->
                    processCapture(image, result, session, thumbnailLatch, thumbnailRef)
                    mainHandler.post {
                        isCapturing = false
                        shutterController.onCaptureComplete()
                        finishingCaptureOverlay.visibility = View.GONE
                        if (pendingPauseCleanup) {
                            pendingPauseCleanup = false
                            performCleanup()
                        }
                    }
                },
                onCaptureFailed = {
                    previewRenderer.pendingFboReadback = null
                    mainHandler.post {
                        isCapturing = false
                        shutterController.onCaptureFailed()
                        finishingCaptureOverlay.visibility = View.GONE
                        if (pendingPauseCleanup) {
                            pendingPauseCleanup = false
                            performCleanup()
                        }
                    }
                }
            )
        }

        textureView.setOnTouchListener { _, event ->
            scaleGestureDetector?.onTouchEvent(event) // always feed; never veto on its return value

            if (event.action == MotionEvent.ACTION_DOWN && cameraReady) {
                if (isoPopupShowing || shutterPopupShowing || evPopupShowing) {
                    dismissAllPopups()
                    return@setOnTouchListener true
                }
                if (settingsPanelOpen) {
                    settingsPanelOpen = false
                    settingsPanel.visibility = View.GONE
                    return@setOnTouchListener true
                }
                if (currentWbMode == WhiteBalanceMode.GRAY_CARD && grayCardSampler.isActive) {
                    grayCardSampler.sample(
                        previewRenderer.currentYPlane ?: return@setOnTouchListener false,
                        previewRenderer.currentUPlane ?: return@setOnTouchListener false,
                        previewRenderer.currentVPlane ?: return@setOnTouchListener false,
                        previewRenderer.currentYuvWidth, previewRenderer.currentYuvHeight,
                        event.x, event.y, textureView.width, textureView.height
                    )
                } else if (!autofocusController.isLocked && currentLensCanTapToFocus) {
                    showFocusIndicator(event.x, event.y)
                    applyFocusPoint(event.x, event.y)
                }
                return@setOnTouchListener true
            }
            if (event.action == MotionEvent.ACTION_UP || event.action == MotionEvent.ACTION_CANCEL) {
                isScaling = false
            }
            true
        }

        wbButton.setOnClickListener {
            if (cameraReady) showWbPopup()
        }

        wbButton.setOnLongClickListener {
            if (currentWbMode == WhiteBalanceMode.GRAY_CARD) {
                grayCardSampler.activate()
                Toast.makeText(this, "Tap the preview to sample gray card", Toast.LENGTH_SHORT).show()
            }
            true
        }

        aeAfLockButton.setOnClickListener {
            if (!cameraReady) return@setOnClickListener
            if (autofocusController.isLocked) {
                autofocusController.unlock()
                aeAfLockButton.setImageResource(R.drawable.ic_lock_open)
                aeAfLockButton.alpha = 0.6f
            } else {
                autofocusController.lock()
                aeAfLockButton.setImageResource(R.drawable.ic_lock_closed)
                aeAfLockButton.alpha = 1.0f
            }
        }

        zoomSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            val zoom = ZoomController.MIN_ZOOM + (v / 400f) * (ZoomController.MAX_ZOOM - ZoomController.MIN_ZOOM)
            previewRenderer.zoomController.setZoom(zoom)
            zoomLabel.text = String.format("%.1fx", zoom)
        })

        previewRenderer.zoomController.listener = { zoom: Float, _: Float, _: Float ->
            val progress = ((zoom - ZoomController.MIN_ZOOM) / (ZoomController.MAX_ZOOM - ZoomController.MIN_ZOOM) * 400).toInt()
            if (zoomSlider.progress != progress) zoomSlider.progress = progress
            zoomLabel.text = String.format("%.1fx", zoom)
            // Auto-hide zoom controls at 1.0x
            val show = zoom > 1.01f
            zoomRow.visibility = if (show) View.VISIBLE else View.GONE
            zoomSlider.visibility = if (show) View.VISIBLE else View.GONE
            zoomLabel.visibility = if (show) View.VISIBLE else View.GONE
            // Keep focus region glued to indicator across zoom changes
            if (!autofocusController.isLocked && focusIndicator.visibility == View.VISIBLE) {
                applyFocusPoint(
                    focusIndicator.x + focusIndicator.width / 2f,
                    focusIndicator.y + focusIndicator.height / 2f
                )
            }
        }

        setupPinchZoom()
        setupSettingsPanel()
        updateFlashUI()
        updateWbUI()
    }

    private var lastClickTime = 0L
    private var clickCount = 0

    private fun setupSettingsPanel() {
        // Double-click reset helper
        fun setupDoubleClickReset(label: TextView, resetAction: () -> Unit) {
            label.setOnClickListener {
                val now = System.currentTimeMillis()
                if (now - lastClickTime < 300) {
                    clickCount++
                    if (clickCount >= 2) {
                        resetAction()
                        clickCount = 0
                    }
                } else {
                    clickCount = 1
                }
                lastClickTime = now
            }
        }

        // Double-tap on SeekBar to reset to default
        fun setupSliderDoubleClickReset(seekBar: SeekBar, defaultProgress: Int, resetAction: () -> Unit) {
            var lastTapTime = 0L
            var consumed = false
            seekBar.setOnTouchListener { _, event ->
                if (event.action == MotionEvent.ACTION_DOWN) {
                    consumed = false
                    val now = System.currentTimeMillis()
                    if (now - lastTapTime < 300) {
                        seekBar.progress = defaultProgress
                        resetAction()
                        lastTapTime = 0L
                        consumed = true
                        return@setOnTouchListener true
                    }
                    lastTapTime = now
                }
                consumed
            }
        }

        contrastSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(contrast = 1.4f + v * 0.1f)
            contrastLabel.text = String.format("Contrast  %.1f", agxParams.contrast)
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(contrastSlider, 10) {
            agxParams = agxParams.copy(contrast = 2.4f)
            contrastLabel.text = "Contrast  2.4"
            uploadAgxUniforms()
        }
        toeSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(toe = 0.7f + v * 0.1f)
            toeLabel.text = String.format("Toe  %.1f", agxParams.toe)
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(toeSlider, 8) {
            agxParams = agxParams.copy(toe = 1.5f)
            toeLabel.text = "Toe  1.5"
            uploadAgxUniforms()
        }
        shoulderSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(shoulder = 0.7f + v * 0.1f)
            shoulderLabel.text = String.format("Shoulder  %.1f", agxParams.shoulder)
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(shoulderSlider, 8) {
            agxParams = agxParams.copy(shoulder = 1.5f)
            shoulderLabel.text = "Shoulder  1.5"
            uploadAgxUniforms()
        }

        usePreForPostCb.setOnCheckedChangeListener { _, checked ->
            agxParams = agxParams.copy(usePreForPost = checked)
            outsetSection.visibility = if (checked) View.GONE else View.VISIBLE
            uploadAgxUniforms()
        }

        val rotRange = 0.5236f
        fun rotToProgress(v: Float) = ((v + rotRange) / (rotRange * 2) * 524).toInt().coerceIn(0, 524)
        fun progressToRot(p: Float) = (p / 524f * rotRange * 2) - rotRange

        insetRotRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(rgbRotation = floatArrayOf(progressToRot(v.toFloat()), agxParams.rgbRotation[1], agxParams.rgbRotation[2]))
            insetRotRLabel.text = String.format("RGB Rot R  %.3f", agxParams.rgbRotation[0])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetRotRSlider, rotToProgress(AgxParams().rgbRotation[0])) {
            agxParams = agxParams.copy(rgbRotation = floatArrayOf(AgxParams().rgbRotation[0], agxParams.rgbRotation[1], agxParams.rgbRotation[2]))
            insetRotRLabel.text = String.format("RGB Rot R  %.3f", agxParams.rgbRotation[0])
            uploadAgxUniforms()
        }
        insetRotGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(rgbRotation = floatArrayOf(agxParams.rgbRotation[0], progressToRot(v.toFloat()), agxParams.rgbRotation[2]))
            insetRotGLabel.text = String.format("RGB Rot G  %.3f", agxParams.rgbRotation[1])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetRotGSlider, rotToProgress(AgxParams().rgbRotation[1])) {
            agxParams = agxParams.copy(rgbRotation = floatArrayOf(agxParams.rgbRotation[0], AgxParams().rgbRotation[1], agxParams.rgbRotation[2]))
            insetRotGLabel.text = String.format("RGB Rot G  %.3f", agxParams.rgbRotation[1])
            uploadAgxUniforms()
        }
        insetRotBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(rgbRotation = floatArrayOf(agxParams.rgbRotation[0], agxParams.rgbRotation[1], progressToRot(v.toFloat())))
            insetRotBLabel.text = String.format("RGB Rot B  %.3f", agxParams.rgbRotation[2])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetRotBSlider, rotToProgress(AgxParams().rgbRotation[2])) {
            agxParams = agxParams.copy(rgbRotation = floatArrayOf(agxParams.rgbRotation[0], agxParams.rgbRotation[1], AgxParams().rgbRotation[2]))
            insetRotBLabel.text = String.format("RGB Rot B  %.3f", agxParams.rgbRotation[2])
            uploadAgxUniforms()
        }

        insetPurRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(purityAttenuation = floatArrayOf(v.toFloat(), agxParams.purityAttenuation[1], agxParams.purityAttenuation[2]))
            insetPurRLabel.text = String.format("Purity R  %.1f", agxParams.purityAttenuation[0])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetPurRSlider, AgxParams().purityAttenuation[0].toInt()) {
            agxParams = agxParams.copy(purityAttenuation = floatArrayOf(AgxParams().purityAttenuation[0], agxParams.purityAttenuation[1], agxParams.purityAttenuation[2]))
            insetPurRLabel.text = String.format("Purity R  %.1f", agxParams.purityAttenuation[0])
            uploadAgxUniforms()
        }
        insetPurGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(purityAttenuation = floatArrayOf(agxParams.purityAttenuation[0], v.toFloat(), agxParams.purityAttenuation[2]))
            insetPurGLabel.text = String.format("Purity G  %.1f", agxParams.purityAttenuation[1])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetPurGSlider, AgxParams().purityAttenuation[1].toInt()) {
            agxParams = agxParams.copy(purityAttenuation = floatArrayOf(agxParams.purityAttenuation[0], AgxParams().purityAttenuation[1], agxParams.purityAttenuation[2]))
            insetPurGLabel.text = String.format("Purity G  %.1f", agxParams.purityAttenuation[1])
            uploadAgxUniforms()
        }
        insetPurBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(purityAttenuation = floatArrayOf(agxParams.purityAttenuation[0], agxParams.purityAttenuation[1], v.toFloat()))
            insetPurBLabel.text = String.format("Purity B  %.1f", agxParams.purityAttenuation[2])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetPurBSlider, AgxParams().purityAttenuation[2].toInt()) {
            agxParams = agxParams.copy(purityAttenuation = floatArrayOf(agxParams.purityAttenuation[0], agxParams.purityAttenuation[1], AgxParams().purityAttenuation[2]))
            insetPurBLabel.text = String.format("Purity B  %.1f", agxParams.purityAttenuation[2])
            uploadAgxUniforms()
        }

        outsetRotRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(reverseRgbRotation = floatArrayOf(progressToRot(v.toFloat()), agxParams.reverseRgbRotation[1], agxParams.reverseRgbRotation[2]))
            outsetRotRLabel.text = String.format("Rev Rot R  %.3f", agxParams.reverseRgbRotation[0])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetRotRSlider, rotToProgress(AgxParams().reverseRgbRotation[0])) {
            agxParams = agxParams.copy(reverseRgbRotation = floatArrayOf(AgxParams().reverseRgbRotation[0], agxParams.reverseRgbRotation[1], agxParams.reverseRgbRotation[2]))
            outsetRotRLabel.text = String.format("Rev Rot R  %.3f", agxParams.reverseRgbRotation[0])
            uploadAgxUniforms()
        }
        outsetRotGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(reverseRgbRotation = floatArrayOf(agxParams.reverseRgbRotation[0], progressToRot(v.toFloat()), agxParams.reverseRgbRotation[2]))
            outsetRotGLabel.text = String.format("Rev Rot G  %.3f", agxParams.reverseRgbRotation[1])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetRotGSlider, rotToProgress(AgxParams().reverseRgbRotation[1])) {
            agxParams = agxParams.copy(reverseRgbRotation = floatArrayOf(agxParams.reverseRgbRotation[0], AgxParams().reverseRgbRotation[1], agxParams.reverseRgbRotation[2]))
            outsetRotGLabel.text = String.format("Rev Rot G  %.3f", agxParams.reverseRgbRotation[1])
            uploadAgxUniforms()
        }
        outsetRotBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(reverseRgbRotation = floatArrayOf(agxParams.reverseRgbRotation[0], agxParams.reverseRgbRotation[1], progressToRot(v.toFloat())))
            outsetRotBLabel.text = String.format("Rev Rot B  %.3f", agxParams.reverseRgbRotation[2])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetRotBSlider, rotToProgress(AgxParams().reverseRgbRotation[2])) {
            agxParams = agxParams.copy(reverseRgbRotation = floatArrayOf(agxParams.reverseRgbRotation[0], agxParams.reverseRgbRotation[1], AgxParams().reverseRgbRotation[2]))
            outsetRotBLabel.text = String.format("Rev Rot B  %.3f", agxParams.reverseRgbRotation[2])
            uploadAgxUniforms()
        }

        outsetPurRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(restorePurity = floatArrayOf(v.toFloat(), agxParams.restorePurity[1], agxParams.restorePurity[2]))
            outsetPurRLabel.text = String.format("Restore R  %.1f", agxParams.restorePurity[0])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetPurRSlider, AgxParams().restorePurity[0].toInt()) {
            agxParams = agxParams.copy(restorePurity = floatArrayOf(AgxParams().restorePurity[0], agxParams.restorePurity[1], agxParams.restorePurity[2]))
            outsetPurRLabel.text = String.format("Restore R  %.1f", agxParams.restorePurity[0])
            uploadAgxUniforms()
        }
        outsetPurGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(restorePurity = floatArrayOf(agxParams.restorePurity[0], v.toFloat(), agxParams.restorePurity[2]))
            outsetPurGLabel.text = String.format("Restore G  %.1f", agxParams.restorePurity[1])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetPurGSlider, AgxParams().restorePurity[1].toInt()) {
            agxParams = agxParams.copy(restorePurity = floatArrayOf(agxParams.restorePurity[0], AgxParams().restorePurity[1], agxParams.restorePurity[2]))
            outsetPurGLabel.text = String.format("Restore G  %.1f", agxParams.restorePurity[1])
            uploadAgxUniforms()
        }
        outsetPurBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(restorePurity = floatArrayOf(agxParams.restorePurity[0], agxParams.restorePurity[1], v.toFloat()))
            outsetPurBLabel.text = String.format("Restore B  %.1f", agxParams.restorePurity[2])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetPurBSlider, AgxParams().restorePurity[2].toInt()) {
            agxParams = agxParams.copy(restorePurity = floatArrayOf(agxParams.restorePurity[0], agxParams.restorePurity[1], AgxParams().restorePurity[2]))
            outsetPurBLabel.text = String.format("Restore B  %.1f", agxParams.restorePurity[2])
            uploadAgxUniforms()
        }

        copyInsetBtn.setOnClickListener {
            agxParams = agxParams.copy(
                reverseRgbRotation = agxParams.rgbRotation.copyOf(),
                restorePurity = agxParams.purityAttenuation.copyOf()
            )
            syncOutsetSliders()
            uploadAgxUniforms()
        }

        tintingScaleSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(tintingScale = (v - 200) * 0.001f)
            tintingScaleLabel.text = String.format("Scale  %.3f", agxParams.tintingScale)
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(tintingScaleSlider, 200) {
            agxParams = agxParams.copy(tintingScale = 0f)
            tintingScaleLabel.text = "Scale  0.000"
            uploadAgxUniforms()
        }
        tintingHueSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(tintingHue = (v - 314) * 0.01f)
            tintingHueLabel.text = String.format("Hue  %.2f", agxParams.tintingHue)
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(tintingHueSlider, 314) {
            agxParams = agxParams.copy(tintingHue = 0f)
            tintingHueLabel.text = "Hue  0.00"
            uploadAgxUniforms()
        }

        nrSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(nrStrength = v / 100f)
            nrLabel.text = String.format("NR Strength  %.1f", agxParams.nrStrength)
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(nrSlider, 0) {
            agxParams = agxParams.copy(nrStrength = 0f)
            nrLabel.text = "NR Strength  0.0"
            uploadAgxUniforms()
        }

        kelvinSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            kelvinState = kelvinState.copy(kelvin = 2000f + v * 100f)
            kelvinLabel.text = String.format("Kelvin  %.0fK", kelvinState.kelvin)
            if (currentWbMode == WhiteBalanceMode.KELVIN) uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(kelvinSlider, 35) {
            kelvinState = kelvinState.copy(kelvin = 5500f)
            kelvinLabel.text = "Kelvin  5500K"
            if (currentWbMode == WhiteBalanceMode.KELVIN) uploadAgxUniforms()
        }
        tintSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            kelvinState = kelvinState.copy(tint = (v - 100).toFloat())
            tintLabel.text = String.format("Tint  %.0f", kelvinState.tint)
            if (currentWbMode == WhiteBalanceMode.KELVIN) uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(tintSlider, 100) {
            kelvinState = kelvinState.copy(tint = 0f)
            tintLabel.text = "Tint  0"
            if (currentWbMode == WhiteBalanceMode.KELVIN) uploadAgxUniforms()
        }

        jpegSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            photoOutput = photoOutput.copy(jpegQuality = v)
            jpegLabel.text = String.format("JPEG Quality  %d", v)
        })
        setupSliderDoubleClickReset(jpegSlider, 85) {
            photoOutput = photoOutput.copy(jpegQuality = 85)
            jpegLabel.text = "JPEG Quality  85"
        }

        resolutionSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (presetInitialLoad || position == lastSelectedResolution) return
                lastSelectedResolution = position
                currentResolutionIndex = position
                restartCameraWithResolution(position)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        val presetAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, mutableListOf<String>())
        presetAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        presetSpinner.adapter = presetAdapter
        refreshPresetSpinner()

        presetSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val name = parent?.getItemAtPosition(position) as? String ?: return
                loadPreset(name)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        presetAddBtn.setOnClickListener {
            val input = EditText(this).apply { hint = "New preset name" }
            AlertDialog.Builder(this)
                .setTitle("Add Preset")
                .setView(input)
                .setPositiveButton("Add") { _, _ ->
                    val name = input.text.toString().trim()
                    if (name.isNotEmpty()) {
                        presetManager.save(PresetManager.Preset(name = name, agxParams = agxParams))
                        refreshPresetSpinner()
                        selectPreset(name)
                        Toast.makeText(this, "Created: $name", Toast.LENGTH_SHORT).show()
                    }
                }
                .setNegativeButton("Cancel", null)
                .show()
        }

        presetSaveBtn.setOnClickListener {
            val name = presetSpinner.selectedItem as? String ?: return@setOnClickListener
            if (name == PresetManager.PRESET_DEFAULT) return@setOnClickListener
            presetManager.save(PresetManager.Preset(name = name, agxParams = agxParams))
            Toast.makeText(this, "Saved: $name", Toast.LENGTH_SHORT).show()
        }

        presetDeleteBtn.setOnClickListener {
            val name = presetSpinner.selectedItem as? String ?: return@setOnClickListener
            if (name == PresetManager.PRESET_DEFAULT) {
                Toast.makeText(this, "Cannot delete Default", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            presetManager.delete(name)
            refreshPresetSpinner()
            Toast.makeText(this, "Deleted: $name", Toast.LENGTH_SHORT).show()
        }

        presetRenameBtn.setOnClickListener {
            val oldName = presetSpinner.selectedItem as? String ?: return@setOnClickListener
            if (oldName == PresetManager.PRESET_DEFAULT) {
                Toast.makeText(this, "Cannot rename Default", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val input = EditText(this).apply { setText(oldName); selectAll() }
            AlertDialog.Builder(this)
                .setTitle("Rename Preset")
                .setView(input)
                .setPositiveButton("Rename") { _, _ ->
                    val newName = input.text.toString().trim()
                    if (newName.isNotEmpty() && newName != oldName) {
                        presetManager.rename(oldName, newName)
                        refreshPresetSpinner()
                        selectPreset(newName)
                    }
                }
                .setNegativeButton("Cancel", null)
                .show()
        }

        presetResetBtn.setOnClickListener {
            loadPreset(PresetManager.PRESET_DEFAULT)
            selectPreset(PresetManager.PRESET_DEFAULT)
            Toast.makeText(this, "Reset to Default", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadPreset(name: String) {
        val preset = presetManager.load(name) ?: return
        agxParams = preset.agxParams
        syncAllSliders()
        uploadAgxUniforms()
    }

    private fun syncAllSliders() {
        contrastSlider.progress = ((agxParams.contrast - 1.4f) / 0.1f).toInt().coerceIn(0, 26)
        contrastLabel.text = String.format("Contrast  %.1f", agxParams.contrast)
        toeSlider.progress = ((agxParams.toe - 0.7f) / 0.1f).toInt().coerceIn(0, 93)
        toeLabel.text = String.format("Toe  %.1f", agxParams.toe)
        shoulderSlider.progress = ((agxParams.shoulder - 0.7f) / 0.1f).toInt().coerceIn(0, 93)
        shoulderLabel.text = String.format("Shoulder  %.1f", agxParams.shoulder)

        usePreForPostCb.isChecked = agxParams.usePreForPost
        outsetSection.visibility = if (agxParams.usePreForPost) View.GONE else View.VISIBLE
        syncInsetSliders()
        syncOutsetSliders()

        tintingScaleSlider.progress = (agxParams.tintingScale / 0.001f + 200).toInt().coerceIn(0, 400)
        tintingScaleLabel.text = String.format("Scale  %.3f", agxParams.tintingScale)
        tintingHueSlider.progress = (agxParams.tintingHue / 0.01f + 314).toInt().coerceIn(0, 628)
        tintingHueLabel.text = String.format("Hue  %.2f", agxParams.tintingHue)

        nrSlider.progress = (agxParams.nrStrength * 100).toInt().coerceIn(0, 100)
        nrLabel.text = String.format("NR Strength  %.1f", agxParams.nrStrength)

        kelvinSlider.progress = ((kelvinState.kelvin - 2000f) / 100f).toInt().coerceIn(0, 80)
        kelvinLabel.text = String.format("Kelvin  %.0fK", kelvinState.kelvin)
        tintSlider.progress = (kelvinState.tint + 100).toInt().coerceIn(0, 200)
        tintLabel.text = String.format("Tint  %.0f", kelvinState.tint)

        jpegSlider.progress = photoOutput.jpegQuality
        jpegLabel.text = String.format("JPEG Quality  %d", photoOutput.jpegQuality)

        syncWbSliders()
    }

    private fun syncInsetSliders() {
        val rotRange = 0.5236f
        fun rotToProgress(v: Float) = ((v + rotRange) / (rotRange * 2) * 524).toInt().coerceIn(0, 524)
        insetRotRSlider.progress = rotToProgress(agxParams.rgbRotation[0])
        insetRotRLabel.text = String.format("RGB Rot R  %.3f", agxParams.rgbRotation[0])
        insetRotGSlider.progress = rotToProgress(agxParams.rgbRotation[1])
        insetRotGLabel.text = String.format("RGB Rot G  %.3f", agxParams.rgbRotation[1])
        insetRotBSlider.progress = rotToProgress(agxParams.rgbRotation[2])
        insetRotBLabel.text = String.format("RGB Rot B  %.3f", agxParams.rgbRotation[2])

        insetPurRSlider.progress = agxParams.purityAttenuation[0].toInt().coerceIn(0, 60)
        insetPurRLabel.text = String.format("Purity R  %.1f", agxParams.purityAttenuation[0])
        insetPurGSlider.progress = agxParams.purityAttenuation[1].toInt().coerceIn(0, 60)
        insetPurGLabel.text = String.format("Purity G  %.1f", agxParams.purityAttenuation[1])
        insetPurBSlider.progress = agxParams.purityAttenuation[2].toInt().coerceIn(0, 60)
        insetPurBLabel.text = String.format("Purity B  %.1f", agxParams.purityAttenuation[2])
    }

    private fun syncOutsetSliders() {
        val rotRange = 0.5236f
        fun rotToProgress(v: Float) = ((v + rotRange) / (rotRange * 2) * 524).toInt().coerceIn(0, 524)
        outsetRotRSlider.progress = rotToProgress(agxParams.reverseRgbRotation[0])
        outsetRotRLabel.text = String.format("Rev Rot R  %.3f", agxParams.reverseRgbRotation[0])
        outsetRotGSlider.progress = rotToProgress(agxParams.reverseRgbRotation[1])
        outsetRotGLabel.text = String.format("Rev Rot G  %.3f", agxParams.reverseRgbRotation[1])
        outsetRotBSlider.progress = rotToProgress(agxParams.reverseRgbRotation[2])
        outsetRotBLabel.text = String.format("Rev Rot B  %.3f", agxParams.reverseRgbRotation[2])

        outsetPurRSlider.progress = agxParams.restorePurity[0].toInt().coerceIn(0, 60)
        outsetPurRLabel.text = String.format("Restore R  %.1f", agxParams.restorePurity[0])
        outsetPurGSlider.progress = agxParams.restorePurity[1].toInt().coerceIn(0, 60)
        outsetPurGLabel.text = String.format("Restore G  %.1f", agxParams.restorePurity[1])
        outsetPurBSlider.progress = agxParams.restorePurity[2].toInt().coerceIn(0, 60)
        outsetPurBLabel.text = String.format("Restore B  %.1f", agxParams.restorePurity[2])
    }

    private fun syncWbSliders() {
        val kelvinEnabled = currentWbMode == WhiteBalanceMode.KELVIN
        kelvinSlider.isEnabled = kelvinEnabled
        tintSlider.isEnabled = kelvinEnabled
        kelvinLabel.alpha = if (kelvinEnabled) 1.0f else 0.4f
        tintLabel.alpha = if (kelvinEnabled) 1.0f else 0.4f
    }

    private fun refreshPresetSpinner() {
        val names = presetManager.getNames()
        @Suppress("UNCHECKED_CAST")
        val adapter = presetSpinner.adapter as? ArrayAdapter<String> ?: return
        adapter.clear()
        adapter.addAll(names)
        adapter.notifyDataSetChanged()
    }

    private fun selectPreset(name: String) {
        val names = presetManager.getNames()
        val idx = names.indexOf(name)
        if (idx >= 0) presetSpinner.setSelection(idx)
    }

    private fun simpleSeekBar(onProgress: (Int) -> Unit) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
            if (fromUser) onProgress(progress)
        }
        override fun onStartTrackingTouch(seekBar: SeekBar) {}
        override fun onStopTrackingTouch(seekBar: SeekBar) {}
    }

    private fun updateFlashUI() {
        flashButton.setImageResource(when (currentFlashMode) {
            FlashMode.OFF -> R.drawable.ic_flash_off
            FlashMode.AUTO -> R.drawable.ic_flash_auto
            FlashMode.ON -> R.drawable.ic_flash_on
            FlashMode.TORCH -> R.drawable.ic_flash_torch
        })
    }

    private fun updateWbUI() {
        wbButton.text = when (currentWbMode) {
            WhiteBalanceMode.AUTO -> "WB"
            WhiteBalanceMode.KELVIN -> "WB\u00B0K"
            WhiteBalanceMode.GRAY_CARD -> "WB\u25A1"
        }
    }

    private fun showFocusIndicator(x: Float, y: Float) {
        val size = (80 * resources.displayMetrics.density).toFloat()
        focusIndicator.x = x - size / 2f
        focusIndicator.y = y - size / 2f
        focusIndicator.setImageResource(R.drawable.focus_circle)
        focusIndicator.visibility = View.VISIBLE
        focusIndicator.alpha = 0f
        focusIndicator.animate().cancel()
        focusIndicator.animate()
            .alpha(1f)
            .setDuration(150)
            .start()
        val dp44 = 44 * resources.displayMetrics.density
        aeAfLockButton.visibility = View.VISIBLE
        aeAfLockButton.setImageResource(if (autofocusController.isLocked) R.drawable.ic_lock_closed else R.drawable.ic_lock_open)
        aeAfLockButton.alpha = if (autofocusController.isLocked) 1.0f else 0.6f
        aeAfLockButton.x = focusIndicator.x - dp44
        aeAfLockButton.y = focusIndicator.y + size / 2f - 18 * resources.displayMetrics.density
        
        // Show EV slider in auto exposure mode
        if (!isManualMode) {
            showEvSlider(x, y)
        }
    }

    private fun hideFocusIndicator() {
        focusIndicator.animate().cancel()
        focusIndicator.animate()
            .alpha(0f)
            .setDuration(200)
            .withEndAction { 
                focusIndicator.visibility = View.GONE
                focusIndicator.alpha = 1f // reset for next show
            }
            .start()
        aeAfLockButton.animate().cancel()
        aeAfLockButton.animate()
            .alpha(0f)
            .setDuration(200)
            .withEndAction { 
                aeAfLockButton.visibility = View.GONE
                aeAfLockButton.alpha = 1f
            }
            .start()
        hideEvSlider()
    }

    private fun showEvSlider(x: Float, y: Float) {
        if (isManualMode) return
        
        evSliderContainer?.let { container ->
            container.visibility = View.VISIBLE
            evSeekBarVertical?.apply {
                progress = 50
                setExposureCompFromProgress(50)
            }
            
            // Position: center vertically on tap point, offset right of focus indicator
            val density = resources.displayMetrics.density
            val halfContainerH = 70 * density
            container.x = x + 48 * density
            container.y = y - halfContainerH
            
            evSeekBarVertical?.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                    if (fromUser) {
                        val range = camera2Manager.aeExposureCompRange
                        val step = camera2Manager.aeExposureStep
                        if (range != null) {
                            val min = range.start.toInt()
                            val max = range.endInclusive.toInt()
                            val steps = ((max - min) / step).toInt()
                            val value = min + Math.round(progress / 100.0 * steps).toInt()
                            camera2Manager.setExposureCompensation(value)
                        }
                    }
                }
                override fun onStartTrackingTouch(seekBar: SeekBar) {}
                override fun onStopTrackingTouch(seekBar: SeekBar) {}
            })
        }
    }

    // Track last EV tap time for double-tap reset
    private var lastEvTapTime = 0L

    /** Maps view coordinates to normalized (0..1) coordinates in the camera frame. */
    private fun viewToFrameCoords(x: Float, y: Float): FloatArray {
        val fw = previewRenderer.currentYuvWidth.toFloat()
        val fh = previewRenderer.currentYuvHeight.toFloat()
        val vw = textureView.width.toFloat()
        val vh = textureView.height.toFloat()

        CrashLogger.log(TAG, "viewToFrameCoords: x=$x y=$y fw=$fw fh=$fh vw=$vw vh=$vh")

        // Undo the renderer's CENTER_INSIDE viewport (letterbox/pillarbox)
        val contentAspect = fw / fh
        val viewAspect = vw / vh
        var vpW: Float
        var vpH: Float
        var vpX: Float
        var vpY: Float
        if (contentAspect > viewAspect) {
            vpW = vw
            vpH = vw / contentAspect
            vpX = 0f
            vpY = (vh - vpH) / 2f
        } else {
            vpH = vh
            vpW = vh * contentAspect
            vpX = (vw - vpW) / 2f
            vpY = 0f
        }
        var u = ((x - vpX) / vpW).coerceIn(0f, 1f)
        var v = ((y - vpY) / vpH).coerceIn(0f, 1f)

        CrashLogger.log(TAG, "viewToFrameCoords: vpX=$vpX vpY=$vpY vpW=$vpW vpH=$vpH u=$u v=$v")

        // Undo zoom (shader pivots at 1 - zoomCenter: out = (in - pivot) * z + pivot)
        val zc = previewRenderer.zoomController
        u = (u - (1f - zc.zoomCenterX)) / zc.zoomFactor + zc.zoomCenterX
        v = (v - (1f - zc.zoomCenterY)) / zc.zoomFactor + zc.zoomCenterY

        CrashLogger.log(TAG, "viewToFrameCoords after zoom: u=$u v=$v zoomFactor=${zc.zoomFactor} zoomCenter=(${zc.zoomCenterX},${zc.zoomCenterY})")

        return floatArrayOf(u.coerceIn(0f, 1f), v.coerceIn(0f, 1f))
    }

    private fun applyFocusPoint(viewX: Float, viewY: Float) {
        val lens = lensManager.activeLens ?: return
        val uv = viewToFrameCoords(viewX, viewY)
        autofocusController.setFocusPoint(
            uv[0], uv[1],
            lensManager.getSensorActiveArraySize(lens),
            lensManager.getSensorOrientation(lens),
            previewRenderer.isFrontCamera
        )
    }

    private fun showLensWarning(message: String) {
        warningContainer?.let { container ->
            warningDismissRunnable?.let { container.removeCallbacks(it) }
            container.findViewById<TextView>(R.id.tvWarning).text = message
            container.visibility = View.VISIBLE
            container.alpha = 1f

            warningDismissRunnable = Runnable {
                container.animate()
                    .alpha(0f)
                    .setDuration(300)
                    .withEndAction { container.visibility = View.GONE }
                    .start()
            }

            container.postDelayed(warningDismissRunnable, 3000)
        }
    }

    private fun disableTapToFocus() {
        currentLensCanTapToFocus = false
    }

    private fun enableTapToFocus() {
        currentLensCanTapToFocus = true
    }

    private fun showWbPopup() {
        val chars = lensManager.activeLens?.let { lensManager.getCharacteristicsForLens(it) }
        val awbModes = chars?.get(android.hardware.camera2.CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES) ?: intArrayOf()
        
        wbPopup.visibility = View.VISIBLE
        val autoBtn = wbPopup.findViewById<TextView>(R.id.wb_auto)
        val daylightBtn = wbPopup.findViewById<TextView>(R.id.wb_daylight)
        val cloudyBtn = wbPopup.findViewById<TextView>(R.id.wb_cloudy)
        val tungstenBtn = wbPopup.findViewById<TextView>(R.id.wb_tungsten)
        val fluorescentBtn = wbPopup.findViewById<TextView>(R.id.wb_fluorescent)
        val twilightBtn = wbPopup.findViewById<TextView>(R.id.wb_twilight)
        val shadeBtn = wbPopup.findViewById<TextView>(R.id.wb_shade)

        val clickListener = View.OnClickListener { v ->
            val mode = when (v.id) {
                R.id.wb_auto -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_AUTO
                R.id.wb_daylight -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
                R.id.wb_cloudy -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT
                R.id.wb_tungsten -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_INCANDESCENT
                R.id.wb_fluorescent -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_FLUORESCENT
                R.id.wb_twilight -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_TWILIGHT
                R.id.wb_shade -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_SHADE
                else -> return@OnClickListener
            }
            if (mode in awbModes) {
                camera2Manager.setWhiteBalanceMode(mode)
                currentWbMode = WhiteBalanceMode.AUTO
                updateWbUI()
                syncWbSliders()
                uploadAgxUniforms()
            }
            wbPopup.visibility = View.GONE
        }

        autoBtn.setOnClickListener(clickListener)
        daylightBtn.setOnClickListener(clickListener)
        cloudyBtn.setOnClickListener(clickListener)
        tungstenBtn.setOnClickListener(clickListener)
        fluorescentBtn.setOnClickListener(clickListener)
        twilightBtn.setOnClickListener(clickListener)
        shadeBtn.setOnClickListener(clickListener)

        // Hide buttons not supported by this camera
        val supported = awbModes.toSet()
        listOf(
            R.id.wb_auto to android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_AUTO,
            R.id.wb_daylight to android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT,
            R.id.wb_cloudy to android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT,
            R.id.wb_tungsten to android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_INCANDESCENT,
            R.id.wb_fluorescent to android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_FLUORESCENT,
            R.id.wb_twilight to android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_TWILIGHT,
            R.id.wb_shade to android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_SHADE
        ).forEach { (id, mode) ->
            wbPopup.findViewById<TextView>(id).visibility = if (mode in supported) View.VISIBLE else View.GONE
        }
    }

    private fun setupPinchZoom() {
        scaleGestureDetector = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScaleBegin(detector: ScaleGestureDetector): Boolean {
                isScaling = true
                return true
            }

            override fun onScale(detector: ScaleGestureDetector): Boolean {
                val currentZoom = previewRenderer.zoomController.zoomFactor
                val newZoom = (currentZoom * detector.scaleFactor).coerceIn(ZoomController.MIN_ZOOM, previewRenderer.zoomController.maxZoom)
                previewRenderer.zoomController.setZoom(newZoom)
                return true
            }

            override fun onScaleEnd(detector: ScaleGestureDetector) {
                isScaling = false
            }
        })
    }

    private fun populateResolutionSpinner(lens: LensInfo) {
        currentResolutionOptions = lensManager.getResolutionOptions(lens)
        val labels = currentResolutionOptions.map { it.label }
        resolutionSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, labels).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        val restoreIndex = currentResolutionIndex.coerceIn(0, currentResolutionOptions.size - 1)
        lastSelectedResolution = -1
        presetInitialLoad = true
        resolutionSpinner.setSelection(restoreIndex)
        presetInitialLoad = false
        lastSelectedResolution = restoreIndex
    }

    private fun restartCameraWithResolution(resIndex: Int) {
        if (!cameraReady || openingCamera) return
        val lens = lensManager.activeLens ?: return

        val option = currentResolutionOptions.getOrNull(resIndex) ?: return

        val targetW: Int
        val targetH: Int
        if (option.aspectW == 0 || option.aspectH == 0) {
            targetW = lens.sensorActiveWidth
            targetH = lens.sensorActiveHeight
        } else {
            targetW = option.width
            targetH = option.height
        }

        val targetAspect = if (targetW > 0 && targetH > 0) targetW.toFloat() / targetH else 0f

        photoOutput = photoOutput.copy(resolutionWidth = targetW, resolutionHeight = targetH)

        CrashLogger.log(TAG, "restartCameraWithResolution: index=$resIndex target=${targetW}x${targetH} aspect=$targetAspect")

        camera2Manager.close()

        val previewSize = if (targetAspect > 0f) {
            lensManager.getPreviewSizeForAspectRatio(lens, targetAspect, maxPreviewDimensions.first, maxPreviewDimensions.second)
        } else {
            lensManager.getBestPreviewSize(lens, maxPreviewDimensions.first, maxPreviewDimensions.second)
        }

        // Stop render thread to recreate FBO with new preview size
        previewRenderer.stop()

        camera2Manager.stopBackgroundThread()
        camera2Manager.startBackgroundThread()
        previewRenderer.sensorOrientation = lensManager.getSensorOrientation(lens)
        previewRenderer.isFrontCamera = lens.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
        previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
        previewRenderer.targetAspectRatio = targetAspect
        previewRenderer.start()
        camera2Manager.openCamera(lens, previewSize, targetW, targetH)
        previewRenderer.setCaptureSize(camera2Manager.captureSize.width, camera2Manager.captureSize.height)
    }

    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            initCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initCamera()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    private fun initCamera() {
        CrashLogger.log(TAG, "initCamera: start")
        openingCamera = true
        val hasSupported = lensManager.enumerate()
        if (!hasSupported) {
            CrashLogger.log(TAG, "initCamera: no supported camera found")
            openingCamera = false
            Toast.makeText(this, "No supported camera found (requires hardware level LIMITED+)", Toast.LENGTH_LONG).show()
            return
        }

        val primary = lensManager.selectPrimary() ?: run {
            CrashLogger.log(TAG, "initCamera: selectPrimary returned null")
            openingCamera = false
            Log.e(TAG, "No primary lens found")
            return
        }

        CrashLogger.log(TAG, "initCamera: primary=${primary.cameraId} ${primary.label} level=${primary.hardwareLevel}")
        val previewSize = lensManager.getBestPreviewSize(primary, maxPreviewDimensions.first, maxPreviewDimensions.second)
        buildLensSelectorUI()

        previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
        previewRenderer.targetAspectRatio = 0f
        previewRenderer.start()

        uploadAgxUniforms()

        var previewFrameCount = 0
        camera2Manager.onFrameAvailable = frameHandler@{ image ->
            if (!cameraReady) return@frameHandler
            previewFrameCount++
            if (previewFrameCount == 1) {
                CrashLogger.log(TAG, "onFrameAvailable: first frame ${image.width}x${image.height}")
            } else if (previewFrameCount % 30 == 0) {
                CrashLogger.log(TAG, "onFrameAvailable: frame #$previewFrameCount")
            }
            val planes = image.planes
            val w = image.width
            val h = image.height

            val yCopy = extractPlane(planes[0].buffer, w, h, planes[0].rowStride, 1)
            val uvWidth = w / 2
            val uvHeight = h / 2
            val uCopy = extractPlane(planes[1].buffer, uvWidth, uvHeight, planes[1].rowStride, planes[1].pixelStride)
            val vCopy = extractPlane(planes[2].buffer, uvWidth, uvHeight, planes[2].rowStride, planes[2].pixelStride)

            previewRenderer.setYuvFrame(yCopy, uCopy, vCopy, w, h)
        }

        camera2Manager.onSessionReady = { width, height ->
            CrashLogger.log(TAG, "onSessionReady: ${width}x${height}")
            Log.d(TAG, "Camera session ready: ${width}x${height}")
            cameraReady = true
            openingCamera = false
            mainHandler.post {
                lensSwitchOverlay.visibility = View.GONE
                val lens = lensManager.activeLens
                if (lens != null) {
                    populateResolutionSpinner(lens)
                }
                setupManualControls()
                updateManualControlRanges()
                previewRenderer.requestRender()
                // Initial tap-to-focus capability
                val profile = lensManager.getLensProfile(lensManager.activeLens?.cameraId ?: "")
                if (profile == null || !profile.canTapToAdjust()) {
                    disableTapToFocus()
                } else {
                    enableTapToFocus()
                }
                // Initial focus indicator at center
                val cx = textureView.width / 2f
                val cy = textureView.height / 2f
                showFocusIndicator(cx, cy)
                applyFocusPoint(cx, cy)
            }
        }

        camera2Manager.onDisconnected = {
            mainHandler.post {
                CrashLogger.log(TAG, "onDisconnected")
                cameraReady = false
                disconnectBanner.visibility = View.VISIBLE
                Log.w(TAG, "Camera disconnected — banner shown")
            }
        }

        camera2Manager.onError = { msg ->
            mainHandler.post {
                CrashLogger.log(TAG, "onError: $msg")
                cameraReady = false
                openingCamera = false
                camera2Manager.resetHard()
                previewRenderer.stop()
                if (!errorDialogShowing) {
                    errorDialogShowing = true
                    AlertDialog.Builder(this)
                        .setTitle("Camera Error")
                        .setMessage("Camera error. Tap to retry.")
                        .setCancelable(false)
                        .setPositiveButton("Retry") { _, _ ->
                            errorDialogShowing = false
                            restartCamera()
                        }
                        .show()
                }
            }
        }

        camera2Manager.onAutoExposureReadout = { iso, shutterNs ->
            mainHandler.post { updateAutoExposureReadout(iso, shutterNs) }
        }

        camera2Manager.startBackgroundThread()

        try {
            previewRenderer.sensorOrientation = lensManager.getSensorOrientation(primary)
            previewRenderer.isFrontCamera = primary.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
            CrashLogger.log(TAG, "initCamera: calling openCamera sensorOrientation=${previewRenderer.sensorOrientation} isFront=${previewRenderer.isFrontCamera}")
            camera2Manager.openCamera(primary, previewSize, 0, 0)
            previewRenderer.setCaptureSize(camera2Manager.captureSize.width, camera2Manager.captureSize.height)
            CrashLogger.log(TAG, "initCamera: openCamera returned, captureSize=${camera2Manager.captureSize.width}x${camera2Manager.captureSize.height}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open camera: ${e.message}", e)
            CrashLogger.logException(TAG, e)
            openingCamera = false
            cameraReady = false
            mainHandler.post {
                Toast.makeText(this, "Camera open failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
            return
        }

        developerSwitch.setRawSensorAvailable(lensManager.hasAnyRawLens())
    }

    private fun restartCamera() {
        CrashLogger.log(TAG, "restartCamera")
        val lens = lensManager.activeLens ?: lensManager.selectPrimary() ?: return
        camera2Manager.close()
        camera2Manager.stopBackgroundThread()

        val option = currentResolutionOptions.getOrNull(currentResolutionIndex)
        val targetAspect = if (option != null && option.aspectW > 0 && option.aspectH > 0) {
            option.aspectW.toFloat() / option.aspectH
        } else 0f

        val previewSize = if (targetAspect > 0f) {
            lensManager.getPreviewSizeForAspectRatio(lens, targetAspect, maxPreviewDimensions.first, maxPreviewDimensions.second)
        } else {
            lensManager.getBestPreviewSize(lens, maxPreviewDimensions.first, maxPreviewDimensions.second)
        }

        // Stop render thread to recreate FBO with new preview size
        previewRenderer.stop()

        previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
        previewRenderer.targetAspectRatio = targetAspect
        previewRenderer.start()
        camera2Manager.startBackgroundThread()
        previewRenderer.sensorOrientation = lensManager.getSensorOrientation(lens)
        previewRenderer.isFrontCamera = lens.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
        camera2Manager.openCamera(lens, previewSize, photoOutput.resolutionWidth, photoOutput.resolutionHeight)
        previewRenderer.setCaptureSize(camera2Manager.captureSize.width, camera2Manager.captureSize.height)
    }

    private fun switchToLens(targetLens: LensInfo) {
        CrashLogger.log(TAG, "switchToLens: target=${targetLens.cameraId} ${targetLens.label}")
        val currentLens = lensManager.activeLens ?: return

        lensSwitchOverlay.text = "Switching to ${targetLens.label}\u2026"
        lensSwitchOverlay.visibility = View.VISIBLE

        lensManager.saveCurrentState(
            currentLens.cameraId,
            previewRenderer.zoomController.zoomFactor,
            previewRenderer.zoomController.zoomCenterX,
            previewRenderer.zoomController.zoomCenterY,
            currentWbMode.ordinal,
            kelvinState.kelvin,
            kelvinState.tint,
            currentFlashMode.ordinal
        )

        camera2Manager.close()

        lensManager.switchLens(targetLens) { /* save handled above */ }

        val restored = lensManager.getRestoredState(targetLens.cameraId)
        previewRenderer.zoomController.setZoom(restored.zoomFactor)
        previewRenderer.zoomController.pan(restored.zoomCenterX - previewRenderer.zoomController.zoomCenterX, restored.zoomCenterY - previewRenderer.zoomController.zoomCenterY)

        currentWbMode = WhiteBalanceMode.entries[restored.wbModeOrdinal.coerceIn(0, WhiteBalanceMode.entries.size - 1)]
        kelvinState = KelvinState(restored.kelvin, restored.kelvinTint)
        currentFlashMode = FlashMode.entries[restored.flashModeOrdinal.coerceIn(0, FlashMode.entries.size - 1)]

        updateFlashUI()
        updateWbUI()
        syncWbSliders()
        thermalManager.isTorchActive = (currentFlashMode == FlashMode.TORCH)
        syncAllSliders()

        buildLensSelectorUI()
        frontRearToggle.rotation = if (targetLens.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT) 0f else 0f

        uploadAgxUniforms()

        // Check if new lens supports tap-to-focus
        onLensSwitched(targetLens.cameraId, lensManager.getLensProfile(targetLens.cameraId))

        val option = currentResolutionOptions.getOrNull(currentResolutionIndex)
        val targetAspect = if (option != null && option.aspectW > 0 && option.aspectH > 0) {
            option.aspectW.toFloat() / option.aspectH
        } else {
            0f
        }

        val previewSize = if (targetAspect > 0f) {
            lensManager.getPreviewSizeForAspectRatio(targetLens, targetAspect, maxPreviewDimensions.first, maxPreviewDimensions.second)
        } else {
            lensManager.getBestPreviewSize(targetLens, maxPreviewDimensions.first, maxPreviewDimensions.second)
        }

        // Stop render thread to recreate FBO with new preview size
        previewRenderer.stop()

        camera2Manager.stopBackgroundThread()
        camera2Manager.startBackgroundThread()
        previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
        previewRenderer.targetAspectRatio = targetAspect
        previewRenderer.start()
        previewRenderer.sensorOrientation = lensManager.getSensorOrientation(targetLens)
        previewRenderer.isFrontCamera = targetLens.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
        camera2Manager.openCamera(targetLens, previewSize, photoOutput.resolutionWidth, photoOutput.resolutionHeight)
        previewRenderer.setCaptureSize(camera2Manager.captureSize.width, camera2Manager.captureSize.height)

        onLensSwitched(targetLens.cameraId, lensManager.getLensProfile(targetLens.cameraId))
    }

    private fun onLensSwitched(lensId: String, profile: LensProfile?) {
        if (profile == null || !profile.canTapToAdjust()) {
            val msg = when {
                profile?.facing == CameraCharacteristics.LENS_FACING_FRONT ->
                    "Front camera: fixed focus, tap-to-adjust disabled"
                LensClassifier.isAuxiliaryBackCamera(profile!!) ->
                    "Auxiliary lens: tap-to-adjust not supported"
                else -> "This lens does not support tap-to-adjust"
            }
            showLensWarning(msg)
            disableTapToFocus()
        } else {
            if (!profile.canTapToFocus()) {
                showLensWarning("Lens lacks autofocus — tap to adjust exposure only")
            }
            enableTapToFocus()
        }
    }

    private fun createLensButton(lens: LensInfo, isActive: Boolean): TextView {
        return TextView(this).apply {
            text = lensManager.getLensLabel(lens.cameraId)
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 11f
            setPadding(16, 8, 16, 8)
            setBackgroundColor(if (isActive) 0xFF4488FF.toInt() else 0x66444444.toInt())
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { marginStart = 4 }
            isClickable = true
            isFocusable = true
            setOnClickListener { if (lens.cameraId != lensManager.activeLens?.cameraId) switchToLens(lens) }
            setOnLongClickListener {
                Toast.makeText(this@MainActivity, "${lens.label}: ${lens.focalLengthMm}mm, HW level ${lens.hardwareLevel}", Toast.LENGTH_SHORT).show()
                true
            }
        }
    }

    private fun updateFrontRearToggleVisibility(rawMode: Boolean) {
        val current = lensManager.activeLens
        val hasAlternative = when (current?.facing) {
            CameraCharacteristics.LENS_FACING_BACK ->
                lensManager.getFrontLens(rawMode) != null
            CameraCharacteristics.LENS_FACING_FRONT ->
                lensManager.getRearLenses(rawMode).isNotEmpty()
            else -> false
        }
        frontRearToggle.visibility = if (hasAlternative) View.VISIBLE else View.GONE
    }

    private fun buildLensSelectorUI() {
        lensSelector.removeAllViews()
        val active = lensManager.activeLens ?: return
        val currentFacing = active.facing
        val lensesForFacing = if (developerSwitch.useRawSensor) {
            lensManager.getLensesForFacing(currentFacing).filter { it.hasRawSensor }
        } else {
            lensManager.getLensesForFacing(currentFacing)
        }

        // Defensive: if RAW mode is on but active lens is non-RAW, or no RAW lenses exist, revert
        if (developerSwitch.useRawSensor && (!active.hasRawSensor || lensesForFacing.isEmpty())) {
            Log.e(TAG, "buildLensSelectorUI: invariant violated — RAW active but active lens ${active.cameraId} hasRaw=${active.hasRawSensor}, rawLenses=${lensesForFacing.size}")
            developerSwitch.revertToggle()
            updateFrontRearToggleVisibility(false)
            // Camera pipeline may still be in RAW mode — force back to YUV and restart
            previewRenderer.disableBayerMode()
            val fallback = lensManager.getLensesForFacing(currentFacing)
            if (fallback.isEmpty()) return
            for (lens in fallback) {
                lensSelector.addView(createLensButton(lens, lens.cameraId == active.cameraId))
            }
            // Restart camera to ensure YUV pipeline
            val previewSize = lensManager.getBestPreviewSize(active, maxPreviewDimensions.first, maxPreviewDimensions.second)
            camera2Manager.close()
            camera2Manager.stopBackgroundThread()
            previewRenderer.stop()
            previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
            previewRenderer.targetAspectRatio = 0f
            previewRenderer.start()
            camera2Manager.startBackgroundThread()
            camera2Manager.openCamera(active, previewSize, photoOutput.resolutionWidth, photoOutput.resolutionHeight)
            previewRenderer.setCaptureSize(camera2Manager.captureSize.width, camera2Manager.captureSize.height)
            return
        }

        for (lens in lensesForFacing) {
            lensSelector.addView(createLensButton(lens, lens.cameraId == active.cameraId))
        }
    }

    private fun uploadAgxUniforms() {
        val sceneLinearTo709 = when (currentWbMode) {
            WhiteBalanceMode.AUTO -> WhiteBalanceMath.buildAutoSceneLinearTo709(null)
            WhiteBalanceMode.KELVIN -> WhiteBalanceMath.buildSceneLinearTo709(kelvinState.kelvin, kelvinState.tint)
            WhiteBalanceMode.GRAY_CARD -> WhiteBalanceMath.buildAutoSceneLinearTo709(null)
        }

        val insetParams = agxParams.toInsetParams()
        val agx = AgxPrecomputer.compute(insetParams, sceneLinearTo709, whiteLevel = 1023f, blackLevel = 64f)

        previewRenderer.agxSceneLinearTo709 = agx.sceneLinearTo709
        previewRenderer.agxInsetMat = agx.insetMat
        previewRenderer.agxOutsetMat = agx.outsetMat
        previewRenderer.agxToRec2020 = agx.toRec2020
        previewRenderer.agxWhiteLevel = agx.whiteLevel
        previewRenderer.agxBlackLevel = agx.blackLevel
        previewRenderer.agxLogMin = -10f
        previewRenderer.agxLogMax = 6.5f
        previewRenderer.agxLogMidgray = agx.logMidgray
        previewRenderer.agxDisplayMidgray = agx.displayMidgray
        previewRenderer.agxContrast = agxParams.contrast
        previewRenderer.agxToe = agxParams.toe
        previewRenderer.agxShoulder = agxParams.shoulder
    }

    private fun extractPlane(src: java.nio.ByteBuffer, width: Int, height: Int, rowStride: Int, pixelStride: Int): java.nio.ByteBuffer {
        val dst = java.nio.ByteBuffer.allocateDirect(width * height).order(java.nio.ByteOrder.nativeOrder())
        src.position(0)
        if (pixelStride == 1) {
            for (row in 0 until height) {
                val srcOffset = row * rowStride
                src.position(srcOffset)
                src.limit(srcOffset + width)
                dst.position(row * width)
                dst.put(src)
            }
        } else {
            for (row in 0 until height) {
                val srcRowStart = row * rowStride
                val dstRowStart = row * width
                src.position(srcRowStart)
                for (col in 0 until width) {
                    val srcIdx = srcRowStart + col * pixelStride
                    if (srcIdx < src.capacity()) dst.put(dstRowStart + col, src.get(srcIdx))
                }
            }
        }
        dst.position(0)
        src.position(0)
        src.limit(src.capacity())
        return dst
    }

override fun onResume() {
        super.onResume()
        CrashLogger.log(TAG, "onResume: cameraReady=$cameraReady openingCamera=$openingCamera")
        orientationListener.enable()
        thermalManager.reset()
        disconnectBanner.visibility = View.GONE

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED && !cameraReady && !openingCamera) {
            val lens = lensManager.activeLens ?: lensManager.selectPrimary()
            if (lens != null) {
                camera2Manager.close()
                camera2Manager.stopBackgroundThread()
                
                // Respect current resolution selection
                val option = currentResolutionOptions.getOrNull(currentResolutionIndex)
                val targetAspect = if (option != null && option.aspectW > 0 && option.aspectH > 0) {
                    option.aspectW.toFloat() / option.aspectH
                } else 0f

                val previewSize = if (targetAspect > 0f) {
                    lensManager.getPreviewSizeForAspectRatio(lens, targetAspect, maxPreviewDimensions.first, maxPreviewDimensions.second)
                } else {
                    lensManager.getBestPreviewSize(lens, maxPreviewDimensions.first, maxPreviewDimensions.second)
                }

                // Stop render thread to recreate FBO with new preview size
                previewRenderer.stop()

                previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
                previewRenderer.targetAspectRatio = targetAspect
                previewRenderer.start()
                camera2Manager.startBackgroundThread()
                previewRenderer.sensorOrientation = lensManager.getSensorOrientation(lens)
                previewRenderer.isFrontCamera = lens.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
                camera2Manager.openCamera(lens, previewSize, photoOutput.resolutionWidth, photoOutput.resolutionHeight)
                previewRenderer.setCaptureSize(camera2Manager.captureSize.width, camera2Manager.captureSize.height)
            }
        }
    }

    override fun onPause() {
        super.onPause()
        orientationListener.disable()
        CrashLogger.log(TAG, "onPause: isCapturing=$isCapturing cameraReady=$cameraReady")
        if (isCapturing) {
            finishingCaptureOverlay.visibility = View.VISIBLE
            pendingPauseCleanup = true
        } else {
            performCleanup()
        }
    }

    private fun performCleanup() {
        camera2Manager.close()
        camera2Manager.stopBackgroundThread()
        cameraReady = false
        previewRenderer.stop()
        shutterController.cancelCooldown()
    }

    override fun onDestroy() {
        super.onDestroy()
        performCleanup()
    }

    private fun updateThermalUI(state: ThermalManager.State) {

        when (state) {
            ThermalManager.State.NORMAL -> {
                thermalIndicator.visibility = View.GONE
            }
            ThermalManager.State.WARM -> {
                thermalIndicator.text = "WARM"
                thermalIndicator.setTextColor(0xFFFFAA00.toInt())
                thermalIndicator.visibility = View.VISIBLE
            }
            ThermalManager.State.HOT -> {
                thermalIndicator.text = "HOT \u2014 Cooldown"
                thermalIndicator.setTextColor(0xFFFF4444.toInt())
                thermalIndicator.visibility = View.VISIBLE
            }
            ThermalManager.State.CRITICAL -> {
                thermalIndicator.text = "CRITICAL"
                thermalIndicator.setTextColor(0xFFFF0000.toInt())
                thermalIndicator.visibility = View.VISIBLE
            }
        }
    }

    private fun updateShutterUI(state: ShutterController.State) {
        when (state) {
            ShutterController.State.IDLE -> {
                shutterButton.isEnabled = true
                shutterButton.alpha = 1.0f
                cooldownText.visibility = View.GONE
                shutterStateLabel.text = ""
            }
            ShutterController.State.CAPTURING -> {
                shutterButton.isEnabled = false
                shutterButton.alpha = 0.5f
                shutterStateLabel.text = "Capturing..."
            }
            ShutterController.State.COOLDOWN -> {
                shutterButton.isEnabled = false
                shutterButton.alpha = 0.3f
                shutterStateLabel.text = "Cooling..."
            }
            else -> {}
        }
    }

    private fun processCapture(
        image: android.media.Image,
        result: android.hardware.camera2.TotalCaptureResult,
        session: CaptureSession,
        thumbnailLatch: CountDownLatch,
        thumbnailRef: java.util.concurrent.atomic.AtomicReference<Bitmap?>
    ) {
        Thread {
            var tempFile: java.io.File? = null
            try {
                val planes = image.planes
                val w = image.width
                val h = image.height
                val yBuffer = extractPlane(planes[0].buffer, w, h, planes[0].rowStride, 1)
                val uBuffer = extractPlane(planes[1].buffer, w / 2, h / 2, planes[1].rowStride, planes[1].pixelStride)
                val vBuffer = extractPlane(planes[2].buffer, w / 2, h / 2, planes[2].rowStride, planes[2].pixelStride)

                val gpuBitmapRef = java.util.concurrent.atomic.AtomicReference<Bitmap?>(null)
                val gpuLatch = java.util.concurrent.CountDownLatch(1)
                val targetW = if (session.resolutionWidth > 0) session.resolutionWidth else w
                val targetH = if (session.resolutionHeight > 0) session.resolutionHeight else h
                previewRenderer.submitCaptureFrame(yBuffer, uBuffer, vBuffer, w, h, targetW, targetH, session, gpuBitmapRef, gpuLatch)
                val gpuReady = gpuLatch.await(5000, TimeUnit.MILLISECONDS)
                val bitmap = if (gpuReady) gpuBitmapRef.get() else null
                if (bitmap == null) {
                    Log.e(TAG, "GPU capture render timed out")
                    mainHandler.post {
                        Toast.makeText(this, "Capture failed: GPU render timeout", Toast.LENGTH_SHORT).show()
                    }
                    return@Thread
                }

                val iso = result.get(CaptureResult.SENSOR_SENSITIVITY) ?: 100
                val exposureNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L
                val aeState = result.get(CaptureResult.CONTROL_AE_STATE)

                val activeLens = lensManager.activeLens ?: lensManager.selectPrimary()
                val chars = activeLens?.let { lensManager.getCharacteristicsForLens(it) }

                val wallClockOffsetMs = System.currentTimeMillis() - (SystemClock.elapsedRealtimeNanos() / 1_000_000)

                val timestampSource = chars?.get(
                    android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE
                ) ?: android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN

                val sensorTimestampNs = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: SystemClock.elapsedRealtimeNanos()
                val captureWallClockMs = if (timestampSource == android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) {
                    sensorTimestampNs / 1_000_000 + wallClockOffsetMs
                } else {
                    System.currentTimeMillis()
                }

                val focalLengthMm = session.focalLengthMm

                val sensorOrientation = session.sensorOrientation

                // EXIF orientation: rear=(sensor+device)%360, front=(sensor-device+360)%360
                val isFront = session.isFrontCamera
                val exifRotation = if (isFront) {
                    (sensorOrientation - session.deviceOrientation + 360) % 360
                } else {
                    (sensorOrientation + session.deviceOrientation) % 360
                }
                val exifOrientation = when (exifRotation) {
                    90 -> 6
                    180 -> 3
                    270 -> 8
                    else -> 1
                }

                val metadata = CaptureMetadata(
                    sensorOrientation = sensorOrientation,
                    exifOrientation = exifOrientation,
                    focalLengthMm = focalLengthMm,
                    iso = iso,
                    exposureTimeNs = exposureNs,
                    flashMode = session.flashMode,
                    aeState = aeState,
                    captureWallClockMs = captureWallClockMs
                )

                val jpegData = JpegEncoder.encodeToJpeg(bitmap, session.jpegQuality)
                bitmap.recycle()

                val thumbnailReady = thumbnailLatch.await(2000, TimeUnit.MILLISECONDS)
                val thumbnailBitmap = if (thumbnailReady) thumbnailRef.get() else null
                    val thumbnailJpeg = if (thumbnailBitmap != null) {
                        ExifWriter.generateThumbnailJpeg(thumbnailBitmap, 0).also {
                            thumbnailBitmap.recycle()
                        }
                    } else null

                tempFile = java.io.File(cacheDir, "capture_${System.nanoTime()}.jpg")
                tempFile.writeBytes(jpegData)

                ExifWriter.writeExif(tempFile, metadata, thumbnailJpeg)

                val finalJpegData = tempFile.readBytes()

                val savedUri = mediaStoreSaver.saveJpeg(finalJpegData, metadata)
                mainHandler.post {
                    if (savedUri != null) {
                        Toast.makeText(this, "Photo saved", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(this, "Failed to save photo", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Capture processing failed", e)
                mainHandler.post {
                    Toast.makeText(this, "Capture failed: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            } finally {
                image.close()
                tempFile?.delete()
            }
        }.start()
    }

    internal data class CaptureSession(
        val flashMode: com.agx.camera.camera.FlashMode,
        val jpegQuality: Int,
        val resolutionWidth: Int,
        val resolutionHeight: Int,
        val deviceOrientation: Int,
        val sensorOrientation: Int,
        val focalLengthMm: Float,
        val zoomFactor: Float,
        val zoomCenterX: Float,
        val zoomCenterY: Float,
        val agxSceneLinearTo709: FloatArray,
        val agxInsetMat: FloatArray,
        val agxOutsetMat: FloatArray,
        val agxToRec2020: FloatArray,
        val agxWhiteLevel: Float,
        val agxBlackLevel: Float,
        val agxLogMin: Float,
        val agxLogMax: Float,
        val agxLogMidgray: Float,
        val agxDisplayMidgray: Float,
        val agxContrast: Float,
        val agxToe: Float,
        val agxShoulder: Float,
        val isFrontCamera: Boolean = false
    ) {
        override fun equals(other: Any?) = this === other
        override fun hashCode() = System.identityHashCode(this)
    }

    private fun setupManualControls() {
        amToggleButton = findViewById(R.id.am_toggle_button)
        isoOverlay = findViewById(R.id.iso_overlay)
        isoPopup = findViewById(R.id.iso_popup)
        isoSeekBar = findViewById(R.id.iso_seekbar)
        shutterOverlay = findViewById(R.id.shutter_overlay)
        shutterPopup = findViewById(R.id.shutter_popup)
        shutterSeekBar = findViewById(R.id.shutter_seekbar)
        evPopup = findViewById(R.id.ev_popup)
        evSeekBar = findViewById(R.id.ev_seekbar)

        amToggleButton.setOnClickListener {
            isManualMode = !isManualMode
            updateManualModeUI()
        }

        isoOverlay.setOnClickListener { togglePopup(isoPopup, isoOverlay, isoPopupShowing) { isoPopupShowing = it } }
        shutterOverlay.setOnClickListener { togglePopup(shutterPopup, shutterOverlay, shutterPopupShowing) { shutterPopupShowing = it } }

        // Double-tap to reset to auto values
        isoOverlay.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                val now = System.currentTimeMillis()
                if (now - lastIsoTapTime < 300) {
                    isoSeekBar.progress = 50
                    updateManualExposure()
                }
                lastIsoTapTime = now
            }
            false
        }
        shutterOverlay.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                val now = System.currentTimeMillis()
                if (now - lastShutterTapTime < 300) {
                    shutterSeekBar.progress = 50
                    updateManualExposure()
                }
                lastShutterTapTime = now
            }
            false
        }
        evSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                if (fromUser) setExposureCompFromProgress(progress)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })
        // Double-tap on EV slider to reset
        var lastEvTapTime = 0L
        evSeekBar.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                val now = System.currentTimeMillis()
                if (now - lastEvTapTime < 300) {
                    evSeekBar.progress = 50
                    setExposureCompFromProgress(50)
                }
                lastEvTapTime = now
            }
            false
        }

        isoSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                if (fromUser) updateManualExposure()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })
        // Double-tap on ISO slider to reset
        var lastIsoSliderTapTime = 0L
        isoSeekBar.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                val now = System.currentTimeMillis()
                if (now - lastIsoSliderTapTime < 300) {
                    isoSeekBar.progress = 50
                    updateManualExposure()
                }
                lastIsoSliderTapTime = now
            }
            false
        }

        shutterSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                if (fromUser) updateManualExposure()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })
        // Double-tap on Shutter slider to reset
        var lastShutterSliderTapTime = 0L
        shutterSeekBar.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                val now = System.currentTimeMillis()
                if (now - lastShutterSliderTapTime < 300) {
                    shutterSeekBar.progress = 50
                    updateManualExposure()
                }
                lastShutterSliderTapTime = now
            }
            false
        }

        updateManualModeUI()
    }

    private fun updateManualModeUI() {
        amToggleButton.text = if (isManualMode) "M" else "A"
        isoSeekBar.isEnabled = isManualMode
        shutterSeekBar.isEnabled = isManualMode
        isoSeekBar.alpha = if (isManualMode) 1.0f else 0.4f
        shutterSeekBar.alpha = if (isManualMode) 1.0f else 0.4f
        if (isManualMode) {
            // sync sliders to last auto values
            syncSlidersToAutoValues()
            val iso = isoFromProgress(isoSeekBar.progress)
            val expNs = shutterNsFromProgress(shutterSeekBar.progress)
            camera2Manager.setManualExposure(iso, expNs)
        } else {
            camera2Manager.setAutoExposure()
        }
        dismissAllPopups()
        // Hide EV slider in manual mode
        if (isManualMode) hideEvSlider()
    }

    private fun syncSlidersToAutoValues() {
        val isoVals = camera2Manager.availableIsoValues
        val shutterVals = camera2Manager.availableShutterSpeedsNs
        if (isoVals.isEmpty() || shutterVals.isEmpty()) return

        val targetIso = lastAutoIso
        val targetShutterNs = lastAutoShutterNs

        val isoIdx = isoVals.indices.minByOrNull { i -> Math.abs(isoVals[i] - targetIso) } ?: 0
        val shutterIdx = shutterVals.indices.minByOrNull { i -> Math.abs(shutterVals[i] - targetShutterNs) } ?: 0

        isoSeekBar.progress = isoIdx * 100 / (isoVals.size - 1).coerceAtLeast(1)
        shutterSeekBar.progress = shutterIdx * 100 / (shutterVals.size - 1).coerceAtLeast(1)
    }

    private fun updateAutoExposureReadout(iso: Int, shutterNs: Long) {
        lastAutoIso = iso
        lastAutoShutterNs = shutterNs
        if (!isManualMode) {
            isoOverlay.text = "ISO $iso"
            shutterOverlay.text = formatShutterSpeed(shutterNs)
        }
    }

    private fun updateManualExposure() {
        val iso = isoFromProgress(isoSeekBar.progress)
        val expNs = shutterNsFromProgress(shutterSeekBar.progress)
        isoOverlay.text = "ISO $iso"
        shutterOverlay.text = formatShutterSpeed(expNs)
        camera2Manager.setManualExposure(iso, expNs)
    }

    private fun isoFromProgress(progress: Int): Int {
        val vals = camera2Manager.availableIsoValues
        if (vals.isEmpty()) return 400
        val idx = progress * (vals.size - 1) / 100
        return vals[idx.coerceIn(0, vals.size - 1)]
    }

    private fun shutterNsFromProgress(progress: Int): Long {
        val vals = camera2Manager.availableShutterSpeedsNs
        if (vals.isEmpty()) return 33_333_333L
        val idx = progress * (vals.size - 1) / 100
        return vals[idx.coerceIn(0, vals.size - 1)]
    }

    private fun formatShutterSpeed(ns: Long): String {
        val sec = ns / 1_000_000_000.0
        return if (sec >= 1.0) String.format("%.1fs", sec)
        else if (sec >= 0.1) String.format("1/%d", (1.0 / sec).toInt())
        else "1/${(1.0 / sec).toInt()}"
    }

    private fun togglePopup(popup: View, anchor: View, isShowing: Boolean, onChange: (Boolean) -> Unit) {
        if (isShowing) {
            popup.visibility = View.GONE
            onChange(false)
        } else {
            dismissAllPopups()
            popup.visibility = View.VISIBLE
            popup.post {
                val anchorLoc = IntArray(2)
                anchor.getLocationOnScreen(anchorLoc)
                val popupW = popup.width
                val popupH = popup.height
                popup.x = anchorLoc[0].toFloat() + anchor.width / 2f - popupW / 2f
                popup.y = anchorLoc[1].toFloat() - popupH - 8
            }
            onChange(true)
        }
    }

    private fun dismissAllPopups() {
        isoPopup.visibility = View.GONE
        shutterPopup.visibility = View.GONE
        evPopup.visibility = View.GONE
        isoPopupShowing = false
        shutterPopupShowing = false
        evPopupShowing = false
    }

    private fun showEvSlider() {
        if (!isManualMode) {
            evPopup.visibility = View.VISIBLE
            evSeekBar.progress = 50
            evPopupShowing = true
        }
    }

    private fun hideEvSlider() {
        evPopup.visibility = View.GONE
        evPopupShowing = false
        evSliderContainer?.visibility = View.GONE
    }

    private fun updateManualControlRanges() {
        val chars = lensManager.activeLens?.let { lensManager.getCharacteristicsForLens(it) }
        chars?.let { camera2Manager.updateManualControlRanges(it) }
    }

    private fun setExposureCompFromProgress(progress: Int) {
        val range = camera2Manager.aeExposureCompRange
        val step = camera2Manager.aeExposureStep
        if (range == null) return
        val min = range.start.toInt()
        val max = range.endInclusive.toInt()
        val steps = ((max - min) / step).toInt()
        val value = min + Math.round(progress / 100.0 * steps).toInt()
        camera2Manager.setExposureCompensation(value)
    }

    private fun getMaxPreviewDimensions(): Pair<Int, Int> {
        val display = windowManager.defaultDisplay
        val metrics = android.util.DisplayMetrics()
        display.getRealMetrics(metrics)
        val screenWidth = metrics.widthPixels
        val screenHeight = metrics.heightPixels
        // Cap preview at 720p max dimension for performance
        val maxDim = 1280
        return Pair(maxDim, maxDim * screenHeight / screenWidth)
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CAMERA = 100
    }
}
