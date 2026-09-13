package com.agx.camera

import android.Manifest
import android.content.Context
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
import android.util.Size
import android.view.OrientationEventListener
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.TextureView
import android.view.View
import android.view.ViewConfiguration
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
import com.agx.camera.ui.ScrollingIndexBar
import com.agx.camera.CrashLogger
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.io.File
import java.io.FileOutputStream
import org.json.JSONArray
import org.json.JSONObject

// Which map the lens-shading correction consumes.
private enum class LensShadingSourceMode { STOCK, SAMPLED }

// Whether the device HAL ever emitted a usable (non-identity) lens shading map.
private enum class HalMapStatus { UNKNOWN, AVAILABLE, MISSING }

class MainActivity : AppCompatActivity() {

    private enum class IndicatorDragMode { FOCUS, AE }

    private lateinit var camera2Manager: Camera2Manager
    private lateinit var lensManager: LensManager
    private lateinit var previewRenderer: PreviewRenderer
    private lateinit var presetManager: PresetManager
    private lateinit var thermalManager: ThermalManager
    private lateinit var shutterController: ShutterController
    private lateinit var autofocusController: AutofocusController
    private lateinit var mediaStoreSaver: MediaStoreSaver
    private lateinit var developerSwitch: DeveloperSwitch
    private val mainHandler = Handler(Looper.getMainLooper())

    private var currentFlashMode = FlashMode.OFF
    private var currentWbMode = WhiteBalanceMode.AUTO
    private var kelvinState = KelvinState()
    private var agxParams = AgxParams()
    private var photoOutput = PhotoOutputSettings()
    // Leading factor of the demosaic clipping-neutralization exponent (factor * 5).
    private var clipAttenFactor = 0.1f
    private var cameraReady = false
    private var openingCamera = false
    private var settingsPanelOpen = false
    private var isCapturing = false
    private var pendingPauseCleanup = false
    private var errorDialogShowing = false
    private var currentDeviceOrientation = 0

    private var lastFrameArrivalTime = 0L
    private var lastStallRestartTime = 0L
    private val stallWatchdogRunnable = object : Runnable {
        override fun run() {
            val now = SystemClock.elapsedRealtime()
            if (cameraReady && !openingCamera && !isCapturing &&
                lastFrameArrivalTime > 0 && now - lastFrameArrivalTime > 2000 &&
                now - lastStallRestartTime > 4000
            ) {
                CrashLogger.log(TAG, "STALL watchdog: no frames for ${now - lastFrameArrivalTime}ms, restarting camera")
                lastStallRestartTime = now
                lastFrameArrivalTime = 0L
                restartCamera()
            }
            mainHandler.postDelayed(this, 1000)
        }
    }

    @Volatile private var rawFrameDelivered = false
    private var rawFrameLogCount = 0
    private var rawFallbackRunnable: Runnable? = null

    // Lens-shading correction: HAL/driver "stock" maps vs the live sampled
    // estimator map. The source is user-selectable (Auto/Stock/Sampled), but
    // a device that can't emit a non-identity HAL map is pinned to Sampled.
    private lateinit var lensShadingEstimator: LensShadingEstimator
    @Volatile private var lensShadingSourceMode: LensShadingSourceMode? = null
    @Volatile private var halMapStatus = HalMapStatus.UNKNOWN
    @Volatile private var lensShadingStrength = 1f
    @Volatile private var lastHalMapPixels: ShortArray? = null
    @Volatile private var lastHalMapWidth = 0
    @Volatile private var lastHalMapHeight = 0
    private var lastEstimatorMap: LensShadingData? = null
    private var lensShadingUiFrameCount = 0
    // One-shot startup notice that live sampling exists: shown only once per
    // launch when no HAL map is available and no learned map exists yet, never
    // on lens switch or manual sampling.
    private var lensShadingNoticeShown = false
    private var rawSessionCountSinceLaunch = 0
    // The learned map was already persisted this run; avoids re-serializing on
    // every converged frame.
    private var lensShadingMapPersisted = false
    private var rawSensorWidth = 0
    private var rawSensorHeight = 0
    private var rawBayerPattern = BayerPattern.RGGB
    private var rawWhiteLevel = 1023
    private var rawBlackLevel = 64f

    private lateinit var lensShadingStatusPill: TextView
    private lateinit var lensShadingStartBtn: TextView
    private lateinit var lensShadingPauseBtn: TextView
    private lateinit var lensShadingResumeBtn: TextView
    private lateinit var lensShadingResetBtn: TextView
    private lateinit var lensShadingStrengthLabel: TextView
    private lateinit var lensShadingStrengthSlider: SeekBar
    private lateinit var lensShadingStockBtn: TextView
    private lateinit var lensShadingSampledBtn: TextView
    private lateinit var lensShadingNoticeOverlay: TextView

    private lateinit var orientationListener: OrientationEventListener

    private lateinit var textureView: TextureView
    private lateinit var devBanner: TextView
    private lateinit var rawUnsupportedWarning: TextView
    private lateinit var disconnectBanner: TextView
    private lateinit var flashButton: ImageView
    private lateinit var modeLabel: TextView
    private lateinit var wbButton: TextView
    private lateinit var awbLockButton: ImageView
    private lateinit var settingsButton: ImageView
    private lateinit var frontRearToggle: ImageView
    private lateinit var zoomLabel: TextView
    private lateinit var zoomSlider: SeekBar
    private lateinit var zoomRow: View
    private lateinit var lensSelector: LinearLayout
    private lateinit var settingsPanel: ScrollView
    private lateinit var finishingCaptureOverlay: TextView
    private lateinit var lensSwitchOverlay: TextView
    private lateinit var lensInfoOverlay: TextView
    // Only show the transient lens-info message when trigger by an actual lens
    // switch (set in switchToLens, consumed on the new session being ready).
    private var pendingLensInfo = false

    // Shutter / Thermal
    private lateinit var shutterButton: ImageView
    private lateinit var thumbnailButton: ImageView
    private lateinit var shutterStateLabel: TextView
    private lateinit var thermalIndicator: TextView

    // Focus / WB
    private lateinit var focusIndicator: ImageView
    private lateinit var aeAfLockButton: ImageView
    private lateinit var aeIndicator: ImageView
    private lateinit var aeBulb: ImageView
    private lateinit var wbPopup: LinearLayout

    // Manual Exposure
    private lateinit var amToggleButton: TextView
    private lateinit var isoOverlay: TextView
    private lateinit var isoPopup: View
    private lateinit var isoRoller: ScrollingIndexBar
    private lateinit var shutterOverlay: TextView
    private lateinit var shutterPopup: View
    private lateinit var shutterRoller: ScrollingIndexBar
    private lateinit var evPpOverlay: TextView
    private lateinit var evPpPopup: View
    private lateinit var evPpRoller: ScrollingIndexBar
    private lateinit var evSliderContainer: FrameLayout
    private lateinit var evSeekBarVertical: SeekBar

    // Manual Focus (AF/MF toggle)
    private lateinit var focusControls: View
    private lateinit var focusModeButton: TextView
    private lateinit var focusModeCircle: TextView
    private lateinit var focusPopup: View
    private lateinit var focusRoller: ScrollingIndexBar
    // Panel open = the circular A toggle + focus roller popup are shown; toggling A/M
    // switches auto focus (A) vs manual focus (M) while the panel stays visible.
    private var focusPanelOpen = false
    private var isManualFocus = false
    private var focusRollerConfigured = false

    private var postProcessingEv = 1.5f

    private var evPpRollerConfigured = false

    private var isManualMode = false
    private var lastAutoIso = 200
    private var lastAutoShutterNs = 33_333_333L
    private var lastIsoTapTime = 0L
    private var lastShutterTapTime = 0L
    private var isoPopupShowing = false
    private var shutterPopupShowing = false
    private var evPpPopupShowing = false
    private val focusIndicatorHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private val focusIndicatorHideRunnable = Runnable { hideFocusIndicator() }
    private var focusDragging = false
    private var focusDragStartX = 0f
    private var focusDragStartY = 0f
    // Indicator centers (in view pixels), used for drag hit-testing and the EV
    // slider anchor that follows the auto-exposure indicator.
    private var focusCenterX = 0f
    private var focusCenterY = 0f
    private var aeCenterX = 0f
    private var aeCenterY = 0f
    private var aeDragging = false
    // Grab intent captured at DOWN against the current (pre-tap) indicator
    // geometry, before showFocusIndicator re-centers them; used so a drag that
    // starts on the AE-only strip/bulb moves only the AE indicator.
    private var pendingGrabMode: IndicatorDragMode? = null
    // Deferred tap on empty preview, confirmed only on ACTION_UP so a two-finger
    // pinch is never first registered as a tap.
    private var pendingTap = false

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
    private lateinit var previewResPrefs: android.content.SharedPreferences
    private lateinit var previewResSpinner: Spinner
    private lateinit var focusTimeoutSpinner: Spinner
    private var previewResCapMaxDim = 1280 // default 720p
    private var focusIndicatorTimeoutMs = 0L // 0 = never hide

    // Preset
    private lateinit var presetSpinner: Spinner
    private lateinit var presetAddBtn: TextView
    private lateinit var presetSaveBtn: TextView
    private lateinit var presetDeleteBtn: TextView
    private lateinit var presetRenameBtn: TextView
    private lateinit var presetResetBtn: TextView
    private lateinit var presetStartupBtn: TextView

    // Curve
    private lateinit var contrastLabel: TextView
    private lateinit var contrastSlider: SeekBar
    private lateinit var toeLabel: TextView
    private lateinit var toeSlider: SeekBar
    private lateinit var shoulderLabel: TextView
    private lateinit var shoulderSlider: SeekBar
    private lateinit var middleGrayLabel: TextView
    private lateinit var middleGraySlider: SeekBar
    private lateinit var vibranceLabel: TextView
    private lateinit var vibranceSlider: SeekBar

    // Inset
    private lateinit var useRotationForReverseCb: CheckBox
    private lateinit var useAttenuationForBoostCb: CheckBox
    private lateinit var insetRotRLabel: TextView; private lateinit var insetRotRSlider: SeekBar
    private lateinit var insetRotGLabel: TextView; private lateinit var insetRotGSlider: SeekBar
    private lateinit var insetRotBLabel: TextView; private lateinit var insetRotBSlider: SeekBar
    private lateinit var insetPurRLabel: TextView; private lateinit var insetPurRSlider: SeekBar
    private lateinit var insetPurGLabel: TextView; private lateinit var insetPurGSlider: SeekBar
    private lateinit var insetPurBLabel: TextView; private lateinit var insetPurBSlider: SeekBar

    // Outset
    private lateinit var outsetRotationSection: LinearLayout
    private lateinit var outsetBoostSection: LinearLayout
    private lateinit var outsetRotRLabel: TextView; private lateinit var outsetRotRSlider: SeekBar
    private lateinit var outsetRotGLabel: TextView; private lateinit var outsetRotGSlider: SeekBar
    private lateinit var outsetRotBLabel: TextView; private lateinit var outsetRotBSlider: SeekBar
    private lateinit var outsetPurRLabel: TextView; private lateinit var outsetPurRSlider: SeekBar
    private lateinit var outsetPurGLabel: TextView; private lateinit var outsetPurGSlider: SeekBar
    private lateinit var outsetPurBLabel: TextView; private lateinit var outsetPurBSlider: SeekBar
    private lateinit var copyRotationToReverseBtn: TextView
    private lateinit var copyAttenuationToBoostBtn: TextView

    // Tinting
    private lateinit var tintingScaleLabel: TextView; private lateinit var tintingScaleSlider: SeekBar
    private lateinit var tintingHueLabel: TextView; private lateinit var tintingHueSlider: SeekBar

    // NR
    private lateinit var nrLabel: TextView; private lateinit var nrSlider: SeekBar

    // Sensor clip neutralization
    private lateinit var clipAttenLabel: TextView; private lateinit var clipAttenSlider: SeekBar

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
        window.addFlags(
            android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
            android.view.WindowManager.LayoutParams.FLAG_FULLSCREEN or
            android.view.WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
        )
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
            View.SYSTEM_UI_FLAG_FULLSCREEN or
            View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
            View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
            View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE
        )
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(0, (systemBars.top * 0.7f).toInt(), 0, (systemBars.bottom * 0.7f).toInt())
            insets
        }

        textureView = findViewById(R.id.preview_texture)
        devBanner = findViewById(R.id.dev_banner)
        rawUnsupportedWarning = findViewById(R.id.raw_unsupported_warning)
        disconnectBanner = findViewById(R.id.disconnect_banner)
        warningContainer = findViewById(R.id.warningContainer)
        flashButton = findViewById(R.id.flash_button)
        modeLabel = findViewById(R.id.mode_label)
        wbButton = findViewById(R.id.wb_button)
        awbLockButton = findViewById(R.id.awb_lock_button)
        settingsButton = findViewById(R.id.settings_button)
        frontRearToggle = findViewById(R.id.front_rear_toggle)
        zoomLabel = findViewById(R.id.zoom_label)
        zoomSlider = findViewById(R.id.zoom_slider)
        zoomRow = findViewById(R.id.zoom_row)
        lensSelector = findViewById(R.id.lens_selector)
        settingsPanel = findViewById(R.id.settings_panel)
        lensShadingStatusPill = findViewById(R.id.lens_shading_status_pill)
        lensShadingStartBtn = findViewById(R.id.lens_shading_start_btn)
        lensShadingPauseBtn = findViewById(R.id.lens_shading_pause_btn)
        lensShadingResumeBtn = findViewById(R.id.lens_shading_resume_btn)
        lensShadingResetBtn = findViewById(R.id.lens_shading_reset_btn)
        lensShadingStrengthLabel = findViewById(R.id.lens_shading_strength_label)
        lensShadingStrengthSlider = findViewById(R.id.lens_shading_strength_slider)
        lensShadingStockBtn = findViewById(R.id.lens_shading_stock_btn)
        lensShadingSampledBtn = findViewById(R.id.lens_shading_sampled_btn)
        lensShadingNoticeOverlay = findViewById(R.id.lens_shading_notice_overlay)
        finishingCaptureOverlay = findViewById(R.id.finishing_capture_overlay)
        lensSwitchOverlay = findViewById(R.id.lens_switch_overlay)
        lensInfoOverlay = findViewById(R.id.lens_info_overlay)

        presetSpinner = findViewById(R.id.preset_spinner)
        presetAddBtn = findViewById(R.id.preset_add_btn)
        presetSaveBtn = findViewById(R.id.preset_save_btn)
        presetDeleteBtn = findViewById(R.id.preset_delete_btn)
        presetRenameBtn = findViewById(R.id.preset_rename_btn)
        presetResetBtn = findViewById(R.id.preset_reset_btn)
        presetStartupBtn = findViewById(R.id.preset_startup_btn)

        contrastLabel = findViewById(R.id.contrast_label); contrastSlider = findViewById(R.id.contrast_slider)
        toeLabel = findViewById(R.id.toe_label); toeSlider = findViewById(R.id.toe_slider)
        shoulderLabel = findViewById(R.id.shoulder_label); shoulderSlider = findViewById(R.id.shoulder_slider)
        middleGrayLabel = findViewById(R.id.middle_gray_label); middleGraySlider = findViewById(R.id.middle_gray_slider)
        vibranceLabel = findViewById(R.id.vibrance_label); vibranceSlider = findViewById(R.id.vibrance_slider)

        useRotationForReverseCb = findViewById(R.id.use_rotation_for_reverse_cb)
        useAttenuationForBoostCb = findViewById(R.id.use_attenuation_for_boost_cb)
        insetRotRLabel = findViewById(R.id.inset_rot_r_label); insetRotRSlider = findViewById(R.id.inset_rot_r_slider)
        insetRotGLabel = findViewById(R.id.inset_rot_g_label); insetRotGSlider = findViewById(R.id.inset_rot_g_slider)
        insetRotBLabel = findViewById(R.id.inset_rot_b_label); insetRotBSlider = findViewById(R.id.inset_rot_b_slider)
        insetPurRLabel = findViewById(R.id.inset_pur_r_label); insetPurRSlider = findViewById(R.id.inset_pur_r_slider)
        insetPurGLabel = findViewById(R.id.inset_pur_g_label); insetPurGSlider = findViewById(R.id.inset_pur_g_slider)
        insetPurBLabel = findViewById(R.id.inset_pur_b_label); insetPurBSlider = findViewById(R.id.inset_pur_b_slider)

        outsetRotationSection = findViewById(R.id.outset_rotation_section)
        outsetBoostSection = findViewById(R.id.outset_boost_section)
        outsetRotRLabel = findViewById(R.id.outset_rot_r_label); outsetRotRSlider = findViewById(R.id.outset_rot_r_slider)
        outsetRotGLabel = findViewById(R.id.outset_rot_g_label); outsetRotGSlider = findViewById(R.id.outset_rot_g_slider)
        outsetRotBLabel = findViewById(R.id.outset_rot_b_label); outsetRotBSlider = findViewById(R.id.outset_rot_b_slider)
        outsetPurRLabel = findViewById(R.id.outset_pur_r_label); outsetPurRSlider = findViewById(R.id.outset_pur_r_slider)
        outsetPurGLabel = findViewById(R.id.outset_pur_g_label); outsetPurGSlider = findViewById(R.id.outset_pur_g_slider)
        outsetPurBLabel = findViewById(R.id.outset_pur_b_label); outsetPurBSlider = findViewById(R.id.outset_pur_b_slider)
        copyRotationToReverseBtn = findViewById(R.id.copy_rotation_to_reverse_btn)
        copyAttenuationToBoostBtn = findViewById(R.id.copy_attenuation_to_boost_btn)

        tintingScaleLabel = findViewById(R.id.tinting_scale_label); tintingScaleSlider = findViewById(R.id.tinting_scale_slider)
        tintingHueLabel = findViewById(R.id.tinting_hue_label); tintingHueSlider = findViewById(R.id.tinting_hue_slider)

        nrLabel = findViewById(R.id.nr_label); nrSlider = findViewById(R.id.nr_slider)

        clipAttenLabel = findViewById(R.id.clip_atten_label); clipAttenSlider = findViewById(R.id.clip_atten_slider)

        kelvinLabel = findViewById(R.id.kelvin_label); kelvinSlider = findViewById(R.id.kelvin_slider)
        tintLabel = findViewById(R.id.tint_label); tintSlider = findViewById(R.id.tint_slider)

        jpegLabel = findViewById(R.id.jpeg_label);         jpegSlider = findViewById(R.id.jpeg_slider)
        resolutionSpinner = findViewById(R.id.resolution_spinner)
        previewResSpinner = findViewById(R.id.preview_res_spinner)
        focusTimeoutSpinner = findViewById(R.id.focus_timeout_spinner)

        shutterButton = findViewById(R.id.shutter_button)
        thumbnailButton = findViewById(R.id.thumbnail_button)
        shutterStateLabel = findViewById(R.id.shutter_state_label)
        thermalIndicator = findViewById(R.id.thermal_indicator)

        focusIndicator = findViewById(R.id.focus_indicator)
        aeAfLockButton = findViewById(R.id.ae_af_lock_button)
        aeIndicator = findViewById(R.id.ae_indicator)
        aeBulb = findViewById(R.id.ae_bulb)
        wbPopup = findViewById(R.id.wb_popup)

        amToggleButton = findViewById(R.id.am_toggle_button)
        isoOverlay = findViewById(R.id.iso_overlay)
        isoPopup = findViewById(R.id.iso_popup)
        isoRoller = findViewById(R.id.iso_roller)
        shutterOverlay = findViewById(R.id.shutter_overlay)
        shutterPopup = findViewById(R.id.shutter_popup)
        shutterRoller = findViewById(R.id.shutter_roller)
        evPpOverlay = findViewById(R.id.ev_pp_overlay)
        evPpPopup = findViewById(R.id.ev_pp_popup)
        evPpRoller = findViewById(R.id.ev_pp_roller)
        evSliderContainer = findViewById(R.id.ev_slider_container)
        evSeekBarVertical = findViewById(R.id.ev_seekbar_vertical)

        focusControls = findViewById(R.id.focus_controls)
        focusModeButton = findViewById(R.id.focus_mode_button)
        focusModeCircle = findViewById(R.id.focus_mode_circle)
        focusPopup = findViewById(R.id.focus_popup)
        focusRoller = findViewById(R.id.focus_roller)

        if (BuildConfig.AGX_ENABLE_YUV_FALLBACK) {
            devBanner.visibility = View.VISIBLE
        }

        lensManager = LensManager(this)
        camera2Manager = Camera2Manager(this)
        presetManager = PresetManager(this)
        presetManager.ensureDefault()

        thermalManager = ThermalManager(this)
        shutterController = ShutterController()
        autofocusController = AutofocusController(camera2Manager, mainHandler)
        mediaStoreSaver = MediaStoreSaver(this)

        // Compute max preview dimensions based on screen resolution
        previewResPrefs = getSharedPreferences("agxcam_settings", Context.MODE_PRIVATE)
        previewResCapMaxDim = previewResPrefs.getInt(PREF_PREVIEW_RES_CAP, 1280)
        focusIndicatorTimeoutMs = previewResPrefs.getLong(PREF_FOCUS_TIMEOUT, 0L)
        maxPreviewDimensions = getMaxPreviewDimensions()
        clipAttenFactor = previewResPrefs.getFloat(PREF_CLIP_ATTEN, 0.1f).coerceIn(0f, 1f)

        developerSwitch = DeveloperSwitch(this) { useRaw ->
            CrashLogger.log(TAG, "Developer switch toggled: useRaw=$useRaw")
            Log.d(TAG, "Developer switch toggled: useRaw=$useRaw")

            rawFallbackRunnable?.let { mainHandler.removeCallbacks(it) }
            rawFallbackRunnable = null

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
                        updateRawModeWarning()
                        return@DeveloperSwitch
                    } else {
                        developerSwitch.revertToggle()
                        developerSwitch.showErrorBanner("No RAW-capable lens found")
                        updateRawModeWarning()
                        updateFrontRearToggleVisibility(false)
                        buildLensSelectorUI()
                        return@DeveloperSwitch
                    }
                }
            }

            val lens = lensManager.activeLens ?: return@DeveloperSwitch
            val previewSize = lensManager.getBestPreviewSize(lens, maxPreviewDimensions.first, maxPreviewDimensions.second)

            // Stop render thread before recreating it (must be fully stopped before start
            // to avoid two threads sharing the same EGL display/context/surface).
            previewRenderer.stop()

            camera2Manager.close()
            camera2Manager.stopBackgroundThread()
            camera2Manager.startBackgroundThread()

            camera2Manager.openCamera(lens, previewSize, 0, 0, useRaw = useRaw)
            preparePreviewPipeline(previewSize, 0f)
            previewRenderer.start()
            developerSwitch.updateBannerForRawMode(useRaw)
            updateRawModeWarning()
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

        previewRenderer = PreviewRenderer(textureView).apply {
            clipAttenFactor = this@MainActivity.clipAttenFactor
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
        val startupPreset = previewResPrefs.getString(PREF_STARTUP_PRESET, null)
        if (startupPreset != null && presetManager.getNames().contains(startupPreset)) {
            loadPreset(startupPreset)
            selectPreset(startupPreset)
        } else {
            loadPreset(PresetManager.PRESET_DEFAULT)
        }
        checkPermissions()
    }

    private fun setupUI() {
        loadWbModePrefs()
        flashButton.setOnClickListener {
            currentFlashMode = currentFlashMode.cycle()
            updateFlashUI()
            showModeLabel(flashModeDisplayName(currentFlashMode))
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
                focalLength35mm = lensManager.activeLens?.let { Math.round(it.focalLength35mmEq) } ?: 0,
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
                agxVibrance = previewRenderer.agxVibrance,
                isFrontCamera = previewRenderer.isFrontCamera
            )

            val thumbnailLatch = CountDownLatch(1)
            val thumbnailRef = java.util.concurrent.atomic.AtomicReference<Bitmap?>(null)
            previewRenderer.pendingFboReadback = { fboTexId, w, h ->
                thumbnailRef.set(JpegEncoder.readFboToBitmapFlipped(fboTexId, w, h))
                thumbnailLatch.countDown()
            }

            val rawFrame = if (previewRenderer.useBayerPath) previewRenderer.pullBayerCopy() else null
            if (rawFrame != null) {
                CrashLogger.log(TAG, "shutter: RAW snapshot ${rawFrame.width}x${rawFrame.height}")
                processRawSnapshot(rawFrame, session, thumbnailLatch, thumbnailRef)
                mainHandler.post {
                    isCapturing = false
                    shutterController.onCaptureComplete()
                    finishingCaptureOverlay.visibility = View.GONE
                    if (pendingPauseCleanup) {
                        pendingPauseCleanup = false
                        performCleanup()
                    }
                }
            } else {
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
        }

        textureView.setOnTouchListener { _, event ->
            scaleGestureDetector?.onTouchEvent(event) // always feed; never veto on its return value

            // Any second finger (pinch) cancels a pending tap, however early.
            if (event.actionMasked == MotionEvent.ACTION_POINTER_DOWN ||
                event.actionMasked == MotionEvent.ACTION_POINTER_UP) {
                pendingTap = false
            }

            if (event.action == MotionEvent.ACTION_DOWN && cameraReady) {
                if (isoPopupShowing || shutterPopupShowing) {
                    dismissAllPopups()
                    return@setOnTouchListener true
                }
                if (settingsPanelOpen) {
                    settingsPanelOpen = false
                    settingsPanel.visibility = View.GONE
                    return@setOnTouchListener true
                }
                pendingTap = false
                if (!isManualFocus && !autofocusController.isLocked && currentLensCanTapToFocus) {
                    focusDragging = false
                    aeDragging = false
                    focusDragStartX = event.x
                    focusDragStartY = event.y
                    // A DOWN on any visible indicator part is a drag grab, never
                    // a tap-to-focus: grabbing the AE-only strip/bulb drags
                    // exposure, everything else drags focus. A DOWN on empty
                    // preview is a tap candidate, confirmed on UP so a pinch is
                    // not registered as a tap.
                    val grab = if (focusIndicator.visibility == View.VISIBLE &&
                        aeIndicator.visibility == View.VISIBLE) {
                        dragGrabMode(event.x, event.y)
                    } else null
                    pendingGrabMode = grab
                    if (grab == null) {
                        pendingTap = true
                    }
                } else if (isManualFocus) {
                    // MF: focus is frozen, but the auto-exposure metering region can
                    // still be re-positioned (tap = move it, grab the AE strip/bulb =
                    // drag it). Manual exposure has no AE region to move.
                    pendingGrabMode = null
                    if (!isManualMode) {
                        focusDragStartX = event.x
                        focusDragStartY = event.y
                        if (aeIndicator.visibility == View.VISIBLE && aeGrabArea(event.x, event.y)) {
                            pendingGrabMode = IndicatorDragMode.AE
                        } else {
                            pendingTap = true
                        }
                    }
                }
                return@setOnTouchListener true
            }
            if (event.action == MotionEvent.ACTION_MOVE) {
                // Start a drag once the finger moves past touch slop; pause the
                // auto-hide timeout while either indicator is being dragged. The
                // grabbed part decides which one moves: the overlap / circle area
                // drags focus, the lower AE strip or light bulb drags exposure.
                if (!isScaling && !autofocusController.isLocked &&
                    !focusDragging && !aeDragging &&
                    (focusIndicator.visibility == View.VISIBLE || aeIndicator.visibility == View.VISIBLE)) {
                    val slop = ViewConfiguration.get(this).scaledTouchSlop
                    if (Math.abs(event.x - focusDragStartX) > slop ||
                        Math.abs(event.y - focusDragStartY) > slop) {
                        if (isManualFocus) {
                            // MF: only the AE strip/bulb is draggable.
                            if (pendingGrabMode == IndicatorDragMode.AE) {
                                aeDragging = true
                                focusIndicatorHandler.removeCallbacks(focusIndicatorHideRunnable)
                            }
                        } else {
                            when (pendingGrabMode ?: dragGrabMode(focusDragStartX, focusDragStartY)) {
                                IndicatorDragMode.FOCUS -> focusDragging = true
                                IndicatorDragMode.AE -> aeDragging = true
                                null -> {}
                            }
                            if (focusDragging || aeDragging) {
                                focusIndicatorHandler.removeCallbacks(focusIndicatorHideRunnable)
                            }
                        }
                    }
                }
                if (focusDragging) {
                    moveFocusIndicator(event.x, event.y)
                }
                if (aeDragging) {
                    moveAeIndicator(event.x, event.y)
                }
                return@setOnTouchListener true
            }
            if (event.action == MotionEvent.ACTION_UP || event.action == MotionEvent.ACTION_CANCEL) {
                isScaling = false
                val dragJustEnded = focusDragging || aeDragging
                if (focusDragging) {
                    focusDragging = false
                    // Re-scan at the dropped position so the move takes effect.
                    // The AE region stays where its own indicator sits.
                    rescanFocusPoint(event.x, event.y)
                }
                if (aeDragging) {
                    aeDragging = false
                    // Drop the AE metering region at the final position.
                    applyAePoint(event.x, event.y)
                }
                // A confirmed tap happens only on ACTION_UP: a grab that never
                // became a drag, or an empty-preview tap that no pinch cancelled.
                if (event.action == MotionEvent.ACTION_UP && !dragJustEnded && !isScaling) {
                    if (pendingTap) {
                        if (isManualFocus && !isManualMode) {
                            showAeIndicator(event.x, event.y)
                            applyAePoint(event.x, event.y)
                            // Keep the EV slider anchored to the AE indicator.
                            if (evSliderContainer?.visibility == View.VISIBLE) {
                                positionEvSliderAt()
                            }
                        } else if (!isManualFocus && !autofocusController.isLocked &&
                            currentLensCanTapToFocus) {
                            showFocusIndicator(event.x, event.y)
                            applyFocusPoint(event.x, event.y)
                        }
                    } else if (pendingGrabMode != null && !isManualFocus &&
                        !autofocusController.isLocked && currentLensCanTapToFocus) {
                        // A grab that never became a drag is a tap on the indicator:
                        // re-center both and focus there.
                        showFocusIndicator(event.x, event.y)
                        applyFocusPoint(event.x, event.y)
                    }
                }
                pendingGrabMode = null
                pendingTap = false
                // Timeout restarts counting from 0 after the drag (never in MF: the AE
                // indicator stays while focus is manual).
                if (dragJustEnded && focusIndicatorTimeoutMs > 0 && !autofocusController.isLocked && !isManualFocus) {
                    focusIndicatorHandler.postDelayed(focusIndicatorHideRunnable, focusIndicatorTimeoutMs)
                }
            }
            true
        }

        wbButton.setOnClickListener {
            if (cameraReady) showWbPopup()
        }

        awbLockButton.setOnClickListener {
            if (!cameraReady) return@setOnClickListener
            if (camera2Manager.isAwbLocked) {
                camera2Manager.unlockAwb()
                awbLockButton.setImageResource(R.drawable.ic_lock_open)
                awbLockButton.alpha = 0.6f
            } else {
                camera2Manager.lockAwb()
                awbLockButton.setImageResource(R.drawable.ic_lock_closed)
                awbLockButton.alpha = 1.0f
            }
        }

        aeAfLockButton.setOnClickListener {
            if (!cameraReady) return@setOnClickListener
            if (autofocusController.isLocked) {
                autofocusController.unlock()
                aeAfLockButton.setImageResource(R.drawable.ic_lock_open)
                aeAfLockButton.alpha = 0.6f
                if (focusIndicatorTimeoutMs > 0 && focusIndicator.visibility == View.VISIBLE) {
                    focusIndicatorHandler.postDelayed(focusIndicatorHideRunnable, focusIndicatorTimeoutMs)
                }
            } else {
                autofocusController.lock()
                aeAfLockButton.setImageResource(R.drawable.ic_lock_closed)
                aeAfLockButton.alpha = 1.0f
                focusIndicatorHandler.removeCallbacks(focusIndicatorHideRunnable)
            }
        }

        zoomSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            val zoom = ZoomController.MIN_ZOOM + (v / 400f) * (ZoomController.MAX_ZOOM - ZoomController.MIN_ZOOM)
            previewRenderer.zoomController.setZoom(zoom)
            zoomLabel.text = String.format("%.1fx", zoom)
        })

        previewRenderer.zoomController.listener = { zoom: Float, centerX: Float, centerY: Float ->
            camera2Manager.updateZoom(zoom, centerX, centerY)
            val progress = ((zoom - ZoomController.MIN_ZOOM) / (ZoomController.MAX_ZOOM - ZoomController.MIN_ZOOM) * 400).toInt()
            if (zoomSlider.progress != progress) zoomSlider.progress = progress
            zoomLabel.text = String.format("%.1fx", zoom)
            // Auto-hide zoom controls at 1.0x
            val show = zoom > 1.01f
            zoomRow.visibility = if (show) View.VISIBLE else View.GONE
            zoomSlider.visibility = if (show) View.VISIBLE else View.GONE
            zoomLabel.visibility = if (show) View.VISIBLE else View.GONE
            // Keep focus region glued to indicator across zoom changes without
            // restarting the AF scan (digital zoom keeps the focus distance)
            if (!autofocusController.isLocked && focusIndicator.visibility == View.VISIBLE) {
                applyFocusPoint(
                    focusIndicator.x + focusIndicator.width / 2f,
                    focusIndicator.y + focusIndicator.height / 2f,
                    triggerScan = false
                )
            }
            // Similarly keep the independent AE region glued to its indicator
            if (!autofocusController.isLocked && aeIndicator.visibility == View.VISIBLE) {
                applyAePoint(aeCenterX, aeCenterY)
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

        // --- Lens Shading Correction controls (user-facing) ---
        lensShadingEstimator = LensShadingEstimator()

        // null source mode = auto: stock map when the device provides one, sampled otherwise.
        lensShadingSourceMode = when (previewResPrefs.getInt(PREF_LS_SOURCE_MODE, -1)) {
            LensShadingSourceMode.STOCK.ordinal -> LensShadingSourceMode.STOCK
            LensShadingSourceMode.SAMPLED.ordinal -> LensShadingSourceMode.SAMPLED
            else -> null
        }
        // Remember whether this device ever produced a usable (non-identity)
        // HAL map; if we know it can't, the toggle is pinned to Sampled.
        halMapStatus = if (previewResPrefs.contains(PREF_LS_HAL_MAP)) {
            if (previewResPrefs.getBoolean(PREF_LS_HAL_MAP, false)) HalMapStatus.AVAILABLE else HalMapStatus.MISSING
        } else {
            HalMapStatus.UNKNOWN
        }
        lensShadingStrength = previewResPrefs.getFloat(PREF_LS_STRENGTH, 1f).coerceIn(0f, 1f)
        lensShadingStrengthSlider.progress = (lensShadingStrength * 100f).toInt()
        lensShadingStrengthLabel.text = String.format("Strength  %d%%", (lensShadingStrength * 100f).toInt())

        fun wireSourceModeBtn(btn: TextView, mode: LensShadingSourceMode) {
            btn.setOnClickListener { selectLensShadingSourceMode(mode) }
        }
        wireSourceModeBtn(lensShadingStockBtn, LensShadingSourceMode.STOCK)
        wireSourceModeBtn(lensShadingSampledBtn, LensShadingSourceMode.SAMPLED)

        lensShadingStartBtn.setOnClickListener {
            if (useStockMap) {
                Toast.makeText(this, "Stock HAL map active - estimator disabled", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val lens = lensManager.activeLens
            val w = rawSensorWidth
            val h = rawSensorHeight
            if (lens == null || w <= 0 || h <= 0) {
                Toast.makeText(this, "Open a RAW session first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            lastEstimatorMap = null
            lensShadingUiFrameCount = 0
            lensShadingMapPersisted = false
            CrashLogger.log(
                TAG, "lensShading: start sampling ${w}x${h} lens=${lens.cameraId} " +
                    "pattern=${rawBayerPattern.label} white=$rawWhiteLevel"
            )
            lensShadingEstimator.start(w, h, rawBayerPattern, rawWhiteLevel, rawBlackLevel)
            if (!useStockMap) previewRenderer.resetLensShading()
            updateLensShadingUI()
        }

        lensShadingPauseBtn.setOnClickListener {
            lensShadingEstimator.pause()
            saveLearnedLensShading()
            updateLensShadingUI()
        }

        lensShadingResumeBtn.setOnClickListener {
            lensShadingEstimator.resume()
            updateLensShadingUI()
        }

        lensShadingResetBtn.setOnClickListener {
            val lens = lensManager.activeLens
            lastEstimatorMap = null
            lensShadingMapPersisted = false
            lensShadingEstimator.reset()
            if (lens != null) deleteLensShadingPersistence(lens.cameraId)
            if (!useStockMap) previewRenderer.resetLensShading()
            CrashLogger.log(TAG, "lensShading: reset")
            updateLensShadingUI()
        }

        lensShadingStrengthSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            lensShadingStrength = v / 100f
            lensShadingStrengthLabel.text = String.format("Strength  %d%%", v)
            previewResPrefs.edit().putFloat(PREF_LS_STRENGTH, lensShadingStrength).apply()
            applyLensShadingStrengthLive()
        })
        setupSliderDoubleClickReset(lensShadingStrengthSlider, 100) {
            lensShadingStrength = 1f
            lensShadingStrengthLabel.text = "Strength  100%"
            previewResPrefs.edit().putFloat(PREF_LS_STRENGTH, 1f).apply()
            applyLensShadingStrengthLive()
        }

        contrastSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(contrast = 1.4f + v * 0.1f)
            contrastLabel.text = String.format("Contrast  %.1f", agxParams.contrast)
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(contrastSlider, ((AgxParams().contrast - 1.4f) / 0.1f).toInt().coerceIn(0, 26)) {
            agxParams = agxParams.copy(contrast = AgxParams().contrast)
            contrastLabel.text = String.format("Contrast  %.1f", agxParams.contrast)
            uploadAgxUniforms()
        }
        toeSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(toe = 0.7f + v * 0.1f)
            toeLabel.text = String.format("Toe  %.1f", agxParams.toe)
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(toeSlider, ((AgxParams().toe - 0.7f) / 0.1f).toInt().coerceIn(0, 93)) {
            agxParams = agxParams.copy(toe = AgxParams().toe)
            toeLabel.text = String.format("Toe  %.1f", agxParams.toe)
            uploadAgxUniforms()
        }
        shoulderSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(shoulder = 0.7f + v * 0.1f)
            shoulderLabel.text = String.format("Shoulder  %.1f", agxParams.shoulder)
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(shoulderSlider, ((AgxParams().shoulder - 0.7f) / 0.1f).toInt().coerceIn(0, 93)) {
            agxParams = agxParams.copy(shoulder = AgxParams().shoulder)
            shoulderLabel.text = String.format("Shoulder  %.1f", agxParams.shoulder)
            uploadAgxUniforms()
        }
        middleGraySlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(middleGray = 10f + v * 0.15f)
            middleGrayLabel.text = String.format("Middle Gray  %.1f", agxParams.middleGray)
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(middleGraySlider, ((AgxParams().middleGray - 10f) / 0.15f).toInt().coerceIn(0, 100)) {
            agxParams = agxParams.copy(middleGray = AgxParams().middleGray)
            middleGrayLabel.text = String.format("Middle Gray  %.1f", agxParams.middleGray)
            uploadAgxUniforms()
        }
        vibranceSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(vibrance = -1f + v * 0.01f)
            vibranceLabel.text = String.format("Vibrance  %.3f", agxParams.vibrance)
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(vibranceSlider, ((AgxParams().vibrance + 1f) / 0.01f).toInt().coerceIn(0, 200)) {
            agxParams = agxParams.copy(vibrance = AgxParams().vibrance)
            vibranceLabel.text = String.format("Vibrance  %.3f", agxParams.vibrance)
            uploadAgxUniforms()
        }

        useRotationForReverseCb.setOnCheckedChangeListener { _, checked ->
            agxParams = agxParams.copy(useRotationForReverse = checked)
            outsetRotationSection.visibility = if (checked) View.GONE else View.VISIBLE
            uploadAgxUniforms()
        }

        useAttenuationForBoostCb.setOnCheckedChangeListener { _, checked ->
            agxParams = agxParams.copy(useAttenuationForBoost = checked)
            outsetBoostSection.visibility = if (checked) View.GONE else View.VISIBLE
            uploadAgxUniforms()
        }

        val rotRange = 0.5236f
        fun rotToProgress(v: Float) = ((v + rotRange) / (rotRange * 2) * 524).toInt().coerceIn(0, 524)
        fun progressToRot(p: Float) = (p / 524f * rotRange * 2) - rotRange

        insetRotRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(rotation = floatArrayOf(progressToRot(v.toFloat()), agxParams.rotation[1], agxParams.rotation[2]))
            insetRotRLabel.text = String.format("Red Rotation  %.2f", Math.toDegrees(agxParams.rotation[0].toDouble()))
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetRotRSlider, rotToProgress(AgxParams().rotation[0])) {
            agxParams = agxParams.copy(rotation = floatArrayOf(AgxParams().rotation[0], agxParams.rotation[1], agxParams.rotation[2]))
            insetRotRLabel.text = String.format("Red Rotation  %.2f", Math.toDegrees(agxParams.rotation[0].toDouble()))
            uploadAgxUniforms()
        }
        insetRotGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(rotation = floatArrayOf(agxParams.rotation[0], progressToRot(v.toFloat()), agxParams.rotation[2]))
            insetRotGLabel.text = String.format("Green Rotation  %.2f", Math.toDegrees(agxParams.rotation[1].toDouble()))
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetRotGSlider, rotToProgress(AgxParams().rotation[1])) {
            agxParams = agxParams.copy(rotation = floatArrayOf(agxParams.rotation[0], AgxParams().rotation[1], agxParams.rotation[2]))
            insetRotGLabel.text = String.format("Green Rotation  %.2f", Math.toDegrees(agxParams.rotation[1].toDouble()))
            uploadAgxUniforms()
        }
        insetRotBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(rotation = floatArrayOf(agxParams.rotation[0], agxParams.rotation[1], progressToRot(v.toFloat())))
            insetRotBLabel.text = String.format("Blue Rotation  %.2f", Math.toDegrees(agxParams.rotation[2].toDouble()))
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetRotBSlider, rotToProgress(AgxParams().rotation[2])) {
            agxParams = agxParams.copy(rotation = floatArrayOf(agxParams.rotation[0], agxParams.rotation[1], AgxParams().rotation[2]))
            insetRotBLabel.text = String.format("Blue Rotation  %.2f", Math.toDegrees(agxParams.rotation[2].toDouble()))
            uploadAgxUniforms()
        }

        insetPurRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(attenuation = floatArrayOf(v.toFloat(), agxParams.attenuation[1], agxParams.attenuation[2]))
            insetPurRLabel.text = String.format("Attenuation R  %.1f", agxParams.attenuation[0])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetPurRSlider, AgxParams().attenuation[0].toInt()) {
            agxParams = agxParams.copy(attenuation = floatArrayOf(AgxParams().attenuation[0], agxParams.attenuation[1], agxParams.attenuation[2]))
            insetPurRLabel.text = String.format("Attenuation R  %.1f", agxParams.attenuation[0])
            uploadAgxUniforms()
        }
        insetPurGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(attenuation = floatArrayOf(agxParams.attenuation[0], v.toFloat(), agxParams.attenuation[2]))
            insetPurGLabel.text = String.format("Attenuation G  %.1f", agxParams.attenuation[1])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetPurGSlider, AgxParams().attenuation[1].toInt()) {
            agxParams = agxParams.copy(attenuation = floatArrayOf(agxParams.attenuation[0], AgxParams().attenuation[1], agxParams.attenuation[2]))
            insetPurGLabel.text = String.format("Attenuation G  %.1f", agxParams.attenuation[1])
            uploadAgxUniforms()
        }
        insetPurBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(attenuation = floatArrayOf(agxParams.attenuation[0], agxParams.attenuation[1], v.toFloat()))
            insetPurBLabel.text = String.format("Attenuation B  %.1f", agxParams.attenuation[2])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(insetPurBSlider, AgxParams().attenuation[2].toInt()) {
            agxParams = agxParams.copy(attenuation = floatArrayOf(agxParams.attenuation[0], agxParams.attenuation[1], AgxParams().attenuation[2]))
            insetPurBLabel.text = String.format("Attenuation B  %.1f", agxParams.attenuation[2])
            uploadAgxUniforms()
        }

        outsetRotRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(reverseRotation = floatArrayOf(progressToRot(v.toFloat()), agxParams.reverseRotation[1], agxParams.reverseRotation[2]))
            outsetRotRLabel.text = String.format("Reverse R  %.2f", Math.toDegrees(agxParams.reverseRotation[0].toDouble()))
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetRotRSlider, rotToProgress(AgxParams().reverseRotation[0])) {
            agxParams = agxParams.copy(reverseRotation = floatArrayOf(AgxParams().reverseRotation[0], agxParams.reverseRotation[1], agxParams.reverseRotation[2]))
            outsetRotRLabel.text = String.format("Reverse R  %.2f", Math.toDegrees(agxParams.reverseRotation[0].toDouble()))
            uploadAgxUniforms()
        }
        outsetRotGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(reverseRotation = floatArrayOf(agxParams.reverseRotation[0], progressToRot(v.toFloat()), agxParams.reverseRotation[2]))
            outsetRotGLabel.text = String.format("Reverse G  %.2f", Math.toDegrees(agxParams.reverseRotation[1].toDouble()))
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetRotGSlider, rotToProgress(AgxParams().reverseRotation[1])) {
            agxParams = agxParams.copy(reverseRotation = floatArrayOf(agxParams.reverseRotation[0], AgxParams().reverseRotation[1], agxParams.reverseRotation[2]))
            outsetRotGLabel.text = String.format("Reverse G  %.2f", Math.toDegrees(agxParams.reverseRotation[1].toDouble()))
            uploadAgxUniforms()
        }
        outsetRotBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(reverseRotation = floatArrayOf(agxParams.reverseRotation[0], agxParams.reverseRotation[1], progressToRot(v.toFloat())))
            outsetRotBLabel.text = String.format("Reverse B  %.2f", Math.toDegrees(agxParams.reverseRotation[2].toDouble()))
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetRotBSlider, rotToProgress(AgxParams().reverseRotation[2])) {
            agxParams = agxParams.copy(reverseRotation = floatArrayOf(agxParams.reverseRotation[0], agxParams.reverseRotation[1], AgxParams().reverseRotation[2]))
            outsetRotBLabel.text = String.format("Reverse B  %.2f", Math.toDegrees(agxParams.reverseRotation[2].toDouble()))
            uploadAgxUniforms()
        }

        outsetPurRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(purityBoost = floatArrayOf(v.toFloat(), agxParams.purityBoost[1], agxParams.purityBoost[2]))
            outsetPurRLabel.text = String.format("Purity Boost R  %.1f", agxParams.purityBoost[0])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetPurRSlider, AgxParams().purityBoost[0].toInt()) {
            agxParams = agxParams.copy(purityBoost = floatArrayOf(AgxParams().purityBoost[0], agxParams.purityBoost[1], agxParams.purityBoost[2]))
            outsetPurRLabel.text = String.format("Purity Boost R  %.1f", agxParams.purityBoost[0])
            uploadAgxUniforms()
        }
        outsetPurGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(purityBoost = floatArrayOf(agxParams.purityBoost[0], v.toFloat(), agxParams.purityBoost[2]))
            outsetPurGLabel.text = String.format("Purity Boost G  %.1f", agxParams.purityBoost[1])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetPurGSlider, AgxParams().purityBoost[1].toInt()) {
            agxParams = agxParams.copy(purityBoost = floatArrayOf(agxParams.purityBoost[0], AgxParams().purityBoost[1], agxParams.purityBoost[2]))
            outsetPurGLabel.text = String.format("Purity Boost G  %.1f", agxParams.purityBoost[1])
            uploadAgxUniforms()
        }
        outsetPurBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(purityBoost = floatArrayOf(agxParams.purityBoost[0], agxParams.purityBoost[1], v.toFloat()))
            outsetPurBLabel.text = String.format("Purity Boost B  %.1f", agxParams.purityBoost[2])
            uploadAgxUniforms()
        })
        setupSliderDoubleClickReset(outsetPurBSlider, AgxParams().purityBoost[2].toInt()) {
            agxParams = agxParams.copy(purityBoost = floatArrayOf(agxParams.purityBoost[0], agxParams.purityBoost[1], AgxParams().purityBoost[2]))
            outsetPurBLabel.text = String.format("Purity Boost B  %.1f", agxParams.purityBoost[2])
            uploadAgxUniforms()
        }

        copyRotationToReverseBtn.setOnClickListener {
            agxParams = agxParams.copy(reverseRotation = agxParams.rotation.copyOf())
            syncOutsetSliders()
            uploadAgxUniforms()
        }

        copyAttenuationToBoostBtn.setOnClickListener {
            agxParams = agxParams.copy(purityBoost = agxParams.attenuation.copyOf())
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

        clipAttenSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            clipAttenFactor = v / 100f
            clipAttenLabel.text = String.format("Neutralize  %.2f", clipAttenFactor)
            previewRenderer.clipAttenFactor = clipAttenFactor
            previewResPrefs.edit().putFloat(PREF_CLIP_ATTEN, clipAttenFactor).apply()
        })
        setupSliderDoubleClickReset(clipAttenSlider, 10) {
            clipAttenFactor = 0.1f
            clipAttenLabel.text = "Neutralize  0.10"
            previewRenderer.clipAttenFactor = clipAttenFactor
            previewResPrefs.edit().putFloat(PREF_CLIP_ATTEN, 0.1f).apply()
        }

        kelvinSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            kelvinState = kelvinState.copy(kelvin = 2000f + v * 100f)
            kelvinLabel.text = String.format("Kelvin  %.0fK", kelvinState.kelvin)
            if (currentWbMode == WhiteBalanceMode.KELVIN) {
                uploadAgxUniforms()
                persistWbMode()
            }
        })
        setupSliderDoubleClickReset(kelvinSlider, 43) {
            kelvinState = kelvinState.copy(kelvin = 6300f)
            kelvinLabel.text = "Kelvin  6300K"
            if (currentWbMode == WhiteBalanceMode.KELVIN) {
                uploadAgxUniforms()
                persistWbMode()
            }
        }
        tintSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            kelvinState = kelvinState.copy(tint = (v - 100).toFloat())
            tintLabel.text = String.format("Tint  %.0f", kelvinState.tint)
            if (currentWbMode == WhiteBalanceMode.KELVIN) {
                uploadAgxUniforms()
                persistWbMode()
            }
        })
        setupSliderDoubleClickReset(tintSlider, 86) {
            kelvinState = kelvinState.copy(tint = -14f)
            tintLabel.text = "Tint  -14"
            if (currentWbMode == WhiteBalanceMode.KELVIN) {
                uploadAgxUniforms()
                persistWbMode()
            }
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

        // Preview resolution cap spinner
        val previewResOptions = listOf("480p", "720p", "1080p")
        val previewResMaxDims = listOf(854, 1280, 1920)
        previewResSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, previewResOptions).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        val savedCapIndex = previewResMaxDims.indexOf(previewResCapMaxDim).coerceIn(0, previewResOptions.size - 1)
        previewResSpinner.setSelection(savedCapIndex, false)
        previewResSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (!cameraReady) return
                val newCap = previewResMaxDims[position]
                if (newCap == previewResCapMaxDim) return
                previewResCapMaxDim = newCap
                previewResPrefs.edit().putInt(PREF_PREVIEW_RES_CAP, newCap).apply()
                maxPreviewDimensions = getMaxPreviewDimensions()
                val lens = lensManager.activeLens ?: return
                val option = currentResolutionOptions.getOrNull(currentResolutionIndex)
                val targetAspect = if (option != null && option.aspectW > 0 && option.aspectH > 0) {
                    option.aspectW.toFloat() / option.aspectH
                } else 0f
                val previewSize = if (targetAspect > 0f) {
                    lensManager.getPreviewSizeForAspectRatio(lens, targetAspect, maxPreviewDimensions.first, maxPreviewDimensions.second)
                } else {
                    lensManager.getBestPreviewSize(lens, maxPreviewDimensions.first, maxPreviewDimensions.second)
                }
                previewRenderer.stop()
                camera2Manager.close()
                camera2Manager.stopBackgroundThread()
                camera2Manager.startBackgroundThread()
                previewRenderer.sensorOrientation = lensManager.getSensorOrientation(lens)
                previewRenderer.isFrontCamera = lens.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
                preparePreviewPipeline(previewSize, targetAspect)
                previewRenderer.start()
                camera2Manager.openCamera(lens, previewSize, photoOutput.resolutionWidth, photoOutput.resolutionHeight, useRaw = developerSwitch.useRawSensor)
                previewRenderer.setCaptureSize(camera2Manager.captureSize.width, camera2Manager.captureSize.height)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // Focus indicator timeout spinner
        val focusTimeoutOptions = listOf("Never", "1s", "3s", "10s")
        val focusTimeoutValues = listOf(0L, 1000L, 3000L, 10000L)
        focusTimeoutSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, focusTimeoutOptions).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        val savedTimeoutIndex = focusTimeoutValues.indexOf(focusIndicatorTimeoutMs).coerceIn(0, focusTimeoutOptions.size - 1)
        focusTimeoutSpinner.setSelection(savedTimeoutIndex, false)
        focusTimeoutSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val newTimeout = focusTimeoutValues[position]
                if (newTimeout == focusIndicatorTimeoutMs) return
                focusIndicatorTimeoutMs = newTimeout
                previewResPrefs.edit().putLong(PREF_FOCUS_TIMEOUT, newTimeout).apply()
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
            val name = presetSpinner.selectedItem as? String ?: return@setOnClickListener
            loadPreset(name)
            selectPreset(name)
            Toast.makeText(this, "Reset: $name", Toast.LENGTH_SHORT).show()
        }

        presetStartupBtn.setOnClickListener {
            val name = presetSpinner.selectedItem as? String ?: return@setOnClickListener
            previewResPrefs.edit().putString(PREF_STARTUP_PRESET, name).apply()
            Toast.makeText(this, "Startup preset: $name", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadPreset(name: String) {
        val preset = presetManager.load(name) ?: return
        agxParams = preset.agxParams.copy(nrStrength = agxParams.nrStrength)
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
        middleGraySlider.progress = ((agxParams.middleGray - 10f) / 0.15f).toInt().coerceIn(0, 100)
        middleGrayLabel.text = String.format("Middle Gray  %.1f", agxParams.middleGray)
        vibranceSlider.progress = ((agxParams.vibrance + 1f) / 0.01f).toInt().coerceIn(0, 200)
        vibranceLabel.text = String.format("Vibrance  %.3f", agxParams.vibrance)

        useRotationForReverseCb.isChecked = agxParams.useRotationForReverse
        outsetRotationSection.visibility = if (agxParams.useRotationForReverse) View.GONE else View.VISIBLE
        useAttenuationForBoostCb.isChecked = agxParams.useAttenuationForBoost
        outsetBoostSection.visibility = if (agxParams.useAttenuationForBoost) View.GONE else View.VISIBLE
        syncInsetSliders()
        syncOutsetSliders()

        tintingScaleSlider.progress = (agxParams.tintingScale / 0.001f + 200).toInt().coerceIn(0, 400)
        tintingScaleLabel.text = String.format("Scale  %.3f", agxParams.tintingScale)
        tintingHueSlider.progress = (agxParams.tintingHue / 0.01f + 314).toInt().coerceIn(0, 628)
        tintingHueLabel.text = String.format("Hue  %.2f", agxParams.tintingHue)

        nrSlider.progress = (agxParams.nrStrength * 100).toInt().coerceIn(0, 100)
        nrLabel.text = String.format("NR Strength  %.1f", agxParams.nrStrength)

        clipAttenSlider.progress = (clipAttenFactor * 100).toInt().coerceIn(0, 100)
        clipAttenLabel.text = String.format("Neutralize  %.2f", clipAttenFactor)

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
        insetRotRSlider.progress = rotToProgress(agxParams.rotation[0])
        insetRotRLabel.text = String.format("Red Rotation  %.2f", Math.toDegrees(agxParams.rotation[0].toDouble()))
        insetRotGSlider.progress = rotToProgress(agxParams.rotation[1])
        insetRotGLabel.text = String.format("Green Rotation  %.2f", Math.toDegrees(agxParams.rotation[1].toDouble()))
        insetRotBSlider.progress = rotToProgress(agxParams.rotation[2])
        insetRotBLabel.text = String.format("Blue Rotation  %.2f", Math.toDegrees(agxParams.rotation[2].toDouble()))

        insetPurRSlider.progress = agxParams.attenuation[0].toInt().coerceIn(0, 60)
        insetPurRLabel.text = String.format("Attenuation R  %.1f", agxParams.attenuation[0])
        insetPurGSlider.progress = agxParams.attenuation[1].toInt().coerceIn(0, 60)
        insetPurGLabel.text = String.format("Attenuation G  %.1f", agxParams.attenuation[1])
        insetPurBSlider.progress = agxParams.attenuation[2].toInt().coerceIn(0, 60)
        insetPurBLabel.text = String.format("Attenuation B  %.1f", agxParams.attenuation[2])
    }

    private fun syncOutsetSliders() {
        val rotRange = 0.5236f
        fun rotToProgress(v: Float) = ((v + rotRange) / (rotRange * 2) * 524).toInt().coerceIn(0, 524)
        outsetRotRSlider.progress = rotToProgress(agxParams.reverseRotation[0])
        outsetRotRLabel.text = String.format("Reverse R  %.2f", Math.toDegrees(agxParams.reverseRotation[0].toDouble()))
        outsetRotGSlider.progress = rotToProgress(agxParams.reverseRotation[1])
        outsetRotGLabel.text = String.format("Reverse G  %.2f", Math.toDegrees(agxParams.reverseRotation[1].toDouble()))
        outsetRotBSlider.progress = rotToProgress(agxParams.reverseRotation[2])
        outsetRotBLabel.text = String.format("Reverse B  %.2f", Math.toDegrees(agxParams.reverseRotation[2].toDouble()))

        outsetPurRSlider.progress = agxParams.purityBoost[0].toInt().coerceIn(0, 60)
        outsetPurRLabel.text = String.format("Purity Boost R  %.1f", agxParams.purityBoost[0])
        outsetPurGSlider.progress = agxParams.purityBoost[1].toInt().coerceIn(0, 60)
        outsetPurGLabel.text = String.format("Purity Boost G  %.1f", agxParams.purityBoost[1])
        outsetPurBSlider.progress = agxParams.purityBoost[2].toInt().coerceIn(0, 60)
        outsetPurBLabel.text = String.format("Purity Boost B  %.1f", agxParams.purityBoost[2])
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

    private fun flashModeDisplayName(mode: FlashMode): String = when (mode) {
        FlashMode.OFF -> "Flash Off"
        FlashMode.AUTO -> "Flash Auto"
        FlashMode.ON -> "Flash On"
        FlashMode.TORCH -> "Torch"
    }

    private fun whiteBalanceDisplayName(mode: WhiteBalanceMode): String = when (mode) {
        WhiteBalanceMode.AUTO -> "WB Auto"
        WhiteBalanceMode.KELVIN -> "WB Kelvin"
        WhiteBalanceMode.DAYLIGHT -> "WB Daylight"
        WhiteBalanceMode.CLOUDY -> "WB Cloudy"
        WhiteBalanceMode.INCANDESCENT -> "WB Tungsten"
        WhiteBalanceMode.FLUORESCENT -> "WB Fluorescent"
        WhiteBalanceMode.TWILIGHT -> "WB Twilight"
        WhiteBalanceMode.SHADE -> "WB Shade"
    }

    private val modeLabelHideRunnable = Runnable {
        modeLabel.animate().cancel()
        modeLabel.animate().alpha(0f).setDuration(300).withEndAction {
            modeLabel.visibility = View.GONE
        }.start()
    }

    // Transient mode-name pill: fade in, hold, fade out.
    private fun showModeLabel(text: String) {
        if (!::modeLabel.isInitialized) return
        modeLabel.removeCallbacks(modeLabelHideRunnable)
        modeLabel.text = text
        modeLabel.visibility = View.VISIBLE
        modeLabel.animate().cancel()
        modeLabel.alpha = 0f
        modeLabel.animate().alpha(1f).setDuration(150).start()
        modeLabel.postDelayed(modeLabelHideRunnable, 2000)
    }

    private fun wbModeToCameraMode(mode: WhiteBalanceMode): Int = when (mode) {
        WhiteBalanceMode.AUTO -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_AUTO
        WhiteBalanceMode.KELVIN -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_OFF
        WhiteBalanceMode.DAYLIGHT -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
        WhiteBalanceMode.CLOUDY -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT
        WhiteBalanceMode.INCANDESCENT -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_INCANDESCENT
        WhiteBalanceMode.FLUORESCENT -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_FLUORESCENT
        WhiteBalanceMode.TWILIGHT -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_TWILIGHT
        WhiteBalanceMode.SHADE -> android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_SHADE
    }

    // White balance is global (survives lens switches and restarts), like
    // flash mode. The user said the mode keeps getting reset on switch; the
    // per-lens LensState snapshot must never drive WB again.
    private fun persistWbMode() {
        try {
            previewResPrefs.edit()
                .putInt(PREF_WB_MODE, currentWbMode.ordinal)
                .putFloat(PREF_WB_KELVIN, kelvinState.kelvin)
                .putFloat(PREF_WB_TINT, kelvinState.tint)
                .apply()
        } catch (e: Exception) {
            CrashLogger.log(TAG, "persistWbMode failed: ${e.message}")
        }
    }

    private fun loadWbModePrefs() {
        try {
            val ordinal = previewResPrefs.getInt(PREF_WB_MODE, WhiteBalanceMode.AUTO.ordinal)
            currentWbMode = WhiteBalanceMode.entries[ordinal.coerceIn(0, WhiteBalanceMode.entries.size - 1)]
            kelvinState = kelvinState.copy(
                kelvin = previewResPrefs.getFloat(PREF_WB_KELVIN, kelvinState.kelvin),
                tint = previewResPrefs.getFloat(PREF_WB_TINT, kelvinState.tint)
            )
        } catch (e: Exception) {
            CrashLogger.log(TAG, "loadWbModePrefs failed: ${e.message}")
        }
    }

    private fun updateWbUI() {
        wbButton.text = when (currentWbMode) {
            WhiteBalanceMode.AUTO -> "WB"
            WhiteBalanceMode.KELVIN -> "\u00B0K"
            WhiteBalanceMode.DAYLIGHT -> "DAY"
            WhiteBalanceMode.CLOUDY -> "Cld"
            WhiteBalanceMode.INCANDESCENT -> "Tng"
            WhiteBalanceMode.FLUORESCENT -> "Flr"
            WhiteBalanceMode.TWILIGHT -> "Dsk"
            WhiteBalanceMode.SHADE -> "Shd"
        }
        if (currentWbMode == WhiteBalanceMode.AUTO) {
            awbLockButton.visibility = View.VISIBLE
            awbLockButton.setImageResource(
                if (camera2Manager.isAwbLocked) R.drawable.ic_lock_closed else R.drawable.ic_lock_open
            )
            awbLockButton.alpha = if (camera2Manager.isAwbLocked) 1.0f else 0.6f
        } else {
            awbLockButton.visibility = View.GONE
            if (camera2Manager.isAwbLocked) {
                camera2Manager.unlockAwb()
            }
        }
    }

    private fun showFocusIndicator(x: Float, y: Float) {
        focusIndicatorHandler.removeCallbacks(focusIndicatorHideRunnable)
        positionFocusIndicatorAt(x, y)
        focusIndicator.setImageResource(R.drawable.focus_circle)
        focusIndicator.visibility = View.VISIBLE
        focusIndicator.alpha = 0f
        focusIndicator.animate().cancel()
        focusIndicator.animate()
            .alpha(1f)
            .setDuration(150)
            .start()
        aeAfLockButton.visibility = View.VISIBLE
        aeAfLockButton.setImageResource(if (autofocusController.isLocked) R.drawable.ic_lock_closed else R.drawable.ic_lock_open)
        aeAfLockButton.alpha = if (autofocusController.isLocked) 1.0f else 0.6f

        // Show AE indicator + EV slider in auto exposure mode
        if (!isManualMode) {
            showAeIndicator(x, y)
            showEvSlider()
        } else {
            hideAeIndicator()
        }
        // Auto-hide after timeout (never if focus is locked)
        if (focusIndicatorTimeoutMs > 0 && !autofocusController.isLocked) {
            focusIndicatorHandler.postDelayed(focusIndicatorHideRunnable, focusIndicatorTimeoutMs)
        }
    }

    private fun positionFocusIndicatorAt(x: Float, y: Float) {
        focusCenterX = x
        focusCenterY = y
        val size = (80 * resources.displayMetrics.density).toFloat()
        val lockBtnSize = 24 * resources.displayMetrics.density
        focusIndicator.x = x - size / 2f
        focusIndicator.y = y - size / 2f
        aeAfLockButton.x = focusIndicator.x - lockBtnSize - 8 * resources.displayMetrics.density
        aeAfLockButton.y = focusIndicator.y + size / 2f - lockBtnSize / 2f
    }

    private fun moveFocusIndicator(x: Float, y: Float) {
        positionFocusIndicatorAt(x, y)
        moveFocusPoint(x, y)
    }

    // AE indicator geometry (dp): a tall rectangle the same width run as the
    // focus circle, top-aligned with the circle and extending 32dp below it so
    // there is a draggable strip that does not overlap the focus indicator.
    private fun aeRectWidthPx(): Float = 64 * resources.displayMetrics.density
    private fun aeRectHeightPx(): Float = 112 * resources.displayMetrics.density
    private fun aeExtensionPx(): Float = aeRectHeightPx() - 80 * resources.displayMetrics.density

    private fun positionAeIndicatorAt(x: Float, y: Float) {
        aeCenterX = x
        aeCenterY = y
        val w = aeRectWidthPx()
        val h = aeRectHeightPx()
        aeIndicator.x = x - w / 2f
        aeIndicator.y = y - h / 2f
        // Solid light bulb right below the rectangle; dragging it drags AE too.
        val bulbSize = 20 * resources.displayMetrics.density
        val bulbGap = 2 * resources.displayMetrics.density
        aeBulb.x = x - bulbSize / 2f
        aeBulb.y = aeIndicator.y + h + bulbGap
    }

    /** Show the AE indicator overlapping the focus one (top-aligned, wider overlap). */
    private fun showAeIndicator(focusX: Float, focusY: Float) {
        positionAeIndicatorAt(focusX, focusY + aeExtensionPx() / 2f)
        aeIndicator.visibility = View.VISIBLE
        aeBulb.visibility = View.VISIBLE
    }

    private fun hideAeIndicator() {
        aeIndicator.visibility = View.GONE
        aeBulb.visibility = View.GONE
    }

    private fun moveAeIndicator(x: Float, y: Float) {
        positionAeIndicatorAt(x, y)
        if (evSliderContainer?.visibility == View.VISIBLE) {
            positionEvSliderAt()
        }
        moveAePoint(x, y)
    }

    /**
     * Decide which indicator a drag grabbed, based on the DOWN point:
     * the whole AE rectangle/bulb (with extra grab padding) drags AE, except
     * the small core of the focus circle, which drags focus.
     */
    private fun dragGrabMode(touchX: Float, touchY: Float): IndicatorDragMode? {
        val density = resources.displayMetrics.density
        val focusHalf = 40f * density
        val focusCore = focusHalf * 0.65f
        val aeHalfW = aeRectWidthPx() / 2f
        val aeHalfH = aeRectHeightPx() / 2f
        val grabPadding = 10 * density
        val bulbSize = 20f * density
        val bulbGap = 2f * density

        val aeVisible = aeIndicator.visibility == View.VISIBLE

        // Padded AE rectangle so the thin strip and bulb are easy to hit.
        val inAeRect = aeVisible &&
            touchX >= aeCenterX - aeHalfW - grabPadding && touchX <= aeCenterX + aeHalfW + grabPadding &&
            touchY >= aeCenterY - aeHalfH - grabPadding && touchY <= aeCenterY + aeHalfH + grabPadding

        val bulbCx = aeCenterX
        val bulbCy = aeCenterY + aeHalfH + bulbGap + bulbSize / 2f
        val bulbHalf = bulbSize / 2f + grabPadding * 0.5f
        val inBulb = aeVisible &&
            touchX >= bulbCx - bulbHalf && touchX <= bulbCx + bulbHalf &&
            touchY >= bulbCy - bulbHalf && touchY <= bulbCy + bulbHalf

        val inFocusSquare = touchX >= focusCenterX - focusHalf && touchX <= focusCenterX + focusHalf &&
            touchY >= focusCenterY - focusHalf && touchY <= focusCenterY + focusHalf

        val dx = touchX - focusCenterX
        val dy = touchY - focusCenterY
        val nearFocusCenter = dx * dx + dy * dy <= focusCore * focusCore

        return when {
            inBulb -> IndicatorDragMode.AE
            // Anywhere on the AE rect that isn't the focus centre grabs AE.
            inAeRect && (!inFocusSquare || !nearFocusCenter) -> IndicatorDragMode.AE
            inFocusSquare -> IndicatorDragMode.FOCUS
            else -> null
        }
    }

    /** MF-mode AE hit test: the AE rectangle or its bulb (no focus indicator present). */
    private fun aeGrabArea(touchX: Float, touchY: Float): Boolean {
        val density = resources.displayMetrics.density
        val aeHalfW = aeRectWidthPx() / 2f
        val aeHalfH = aeRectHeightPx() / 2f
        val grabPadding = 10 * density
        val bulbSize = 20f * density
        val bulbGap = 2f * density
        val inRect = touchX >= aeCenterX - aeHalfW - grabPadding && touchX <= aeCenterX + aeHalfW + grabPadding &&
            touchY >= aeCenterY - aeHalfH - grabPadding && touchY <= aeCenterY + aeHalfH + grabPadding
        val bulbCx = aeCenterX
        val bulbCy = aeCenterY + aeHalfH + bulbGap + bulbSize / 2f
        val bulbHalf = bulbSize / 2f + grabPadding * 0.5f
        val inBulb = touchX >= bulbCx - bulbHalf && touchX <= bulbCx + bulbHalf &&
            touchY >= bulbCy - bulbHalf && touchY <= bulbCy + bulbHalf
        return inRect || inBulb
    }

    private fun positionEvSliderAt() {
        // Anchor to the auto-exposure indicator: center vertically on the AE
        // center, offset right of it.
        evSliderContainer?.let { container ->
            val density = resources.displayMetrics.density
            val halfContainerH = 70 * density
            container.x = aeCenterX + 48 * density
            container.y = aeCenterY - halfContainerH
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
        hideAeIndicator()
        hideEvSlider()
    }

    private fun showEvSlider() {
        if (isManualMode) return
        
        evSliderContainer?.let { container ->
            container.visibility = View.VISIBLE
            evSeekBarVertical?.apply {
                progress = 50
                setExposureCompFromProgress(50)
            }
            
            positionEvSliderAt()
            
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

    /** Maps view coordinates to normalized (0..1) coordinates in the camera frame. */
    private fun viewToFrameCoords(x: Float, y: Float): FloatArray? {
        val fw: Float
        val fh: Float
        val contentAspect: Float
        if (previewRenderer.useBayerPath) {
            // RAW display is the full sensor frame center-cropped to the preview
            // aspect by the GPU; the displayed strip aspect is the FBO aspect.
            fw = previewRenderer.currentBayerWidth.toFloat()
            fh = previewRenderer.currentBayerHeight.toFloat()
            contentAspect = previewRenderer.currentContentAspect
        } else {
            fw = previewRenderer.currentYuvWidth.toFloat()
            fh = previewRenderer.currentYuvHeight.toFloat()
            contentAspect = fw / fh
        }
        val vw = textureView.width.toFloat()
        val vh = textureView.height.toFloat()

        CrashLogger.log(TAG, "viewToFrameCoords: x=$x y=$y fw=$fw fh=$fh vw=$vw vh=$vh")

        if (fw <= 0f || fh <= 0f || vw <= 0f || vh <= 0f) return null

        // Undo the renderer's CENTER_INSIDE viewport (letterbox/pillarbox) over
        // the DISPLAYED content aspect
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

        // In bayer mode the GPU center-crops the taller/wider sensor to the
        // preview aspect (scale X or Y < 1). Reverse that crop so u,v span the
        // full sensor frame before the zoom-crop dilution below.
        if (previewRenderer.useBayerPath && contentAspect > 0f) {
            val sensorAspect = fw / fh
            if (sensorAspect > contentAspect) {
                val sx = contentAspect / sensorAspect
                u = 0.5f + (u - 0.5f) * sx
            } else if (sensorAspect < contentAspect) {
                val sy = sensorAspect / contentAspect
                v = 0.5f + (v - 0.5f) * sy
            }
        }

        // Map viewport coords to full sensor coordinates via crop region
        val crop = camera2Manager.computeCropRegion()
        val lens = lensManager.activeLens
        if (crop != null && lens != null) {
            val activeArray = lensManager.getSensorActiveArraySize(lens)
            val sensorW = activeArray.width().toFloat()
            val sensorH = activeArray.height().toFloat()
            u = (crop.left + u * crop.width()) / sensorW
            v = (crop.top + v * crop.height()) / sensorH
            CrashLogger.log(TAG, "viewToFrameCoords crop: crop=(${crop.left},${crop.top},${crop.width()},${crop.height()}) sensor=${sensorW}x${sensorH} u=$u v=$v")
        }

        return floatArrayOf(u.coerceIn(0f, 1f), v.coerceIn(0f, 1f))
    }

    private fun applyFocusPoint(viewX: Float, viewY: Float, triggerScan: Boolean = true) {
        val lens = lensManager.activeLens ?: return
        val uv = viewToFrameCoords(viewX, viewY) ?: return
        autofocusController.setFocusPoint(
            uv[0], uv[1],
            lensManager.getSensorActiveArraySize(lens),
            previewRenderer.isFrontCamera,
            triggerScan
        )
    }

    private fun moveFocusPoint(viewX: Float, viewY: Float) {
        val lens = lensManager.activeLens ?: return
        val uv = viewToFrameCoords(viewX, viewY) ?: return
        autofocusController.moveFocusPoint(
            uv[0], uv[1],
            lensManager.getSensorActiveArraySize(lens),
            previewRenderer.isFrontCamera
        )
    }

    /** Drag-drop: re-scan focus at the drop point without moving the AE region. */
    private fun rescanFocusPoint(viewX: Float, viewY: Float) {
        val lens = lensManager.activeLens ?: return
        val uv = viewToFrameCoords(viewX, viewY) ?: return
        autofocusController.rescanFocusPoint(
            uv[0], uv[1],
            lensManager.getSensorActiveArraySize(lens),
            previewRenderer.isFrontCamera
        )
    }

    private fun applyAePoint(viewX: Float, viewY: Float) {
        val lens = lensManager.activeLens ?: return
        val uv = viewToFrameCoords(viewX, viewY) ?: return
        autofocusController.setExposurePoint(
            uv[0], uv[1],
            lensManager.getSensorActiveArraySize(lens),
            previewRenderer.isFrontCamera
        )
    }

    private fun moveAePoint(viewX: Float, viewY: Float) {
        val lens = lensManager.activeLens ?: return
        val uv = viewToFrameCoords(viewX, viewY) ?: return
        autofocusController.moveExposurePoint(
            uv[0], uv[1],
            lensManager.getSensorActiveArraySize(lens),
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

    private val lensInfoHideRunnable = Runnable {
        lensInfoOverlay.animate().cancel()
        lensInfoOverlay.animate().alpha(0f).setDuration(300).withEndAction {
            lensInfoOverlay.visibility = View.GONE
        }.start()
    }

    // Transient details of the newly switched lens: fade in, hold, fade out.
    private fun showLensInfoMessage(text: String) {
        if (!::lensInfoOverlay.isInitialized) return
        lensInfoOverlay.removeCallbacks(lensInfoHideRunnable)
        lensInfoOverlay.text = text
        lensInfoOverlay.visibility = View.VISIBLE
        lensInfoOverlay.animate().cancel()
        lensInfoOverlay.alpha = 0f
        lensInfoOverlay.animate().alpha(1f).setDuration(150).start()
        lensInfoOverlay.postDelayed(lensInfoHideRunnable, 3000)
    }

    private fun lensDetailSummary(lens: LensInfo): String {
        val facing = if (lens.facing == CameraCharacteristics.LENS_FACING_BACK) "Rear" else "Front"
        val sensor = if (lens.sensorActiveWidth > 0 && lens.sensorActiveHeight > 0) {
            "${lens.sensorActiveWidth}x${lens.sensorActiveHeight}"
        } else {
            lens.jpegOutputSizes.maxByOrNull { it.width.toLong() * it.height }
                ?.let { "${it.width}x${it.height}" } ?: "?"
        }
        val hw = when (lens.hardwareLevel) {
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL 3"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
            else -> "HW ${lens.hardwareLevel}"
        }
        return buildString {
            append(facing).append(" · ").append(lens.label)
            if (lens.focalLengthMm > 0f) {
                val eq35 = lens.focalLength35mmEq
                if (eq35 > 0f) {
                    append(String.format(" · %dmm", Math.round(eq35)))
                } else {
                    append(String.format(" · %.1fmm", lens.focalLengthMm))
                }
            }
            append('\n')
            append(sensor).append(" · ").append(if (lens.hasRawSensor) "RAW" else "YUV")
            append(" · ").append(hw)
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
        val kelvinBtn = wbPopup.findViewById<TextView>(R.id.wb_kelvin)

        val clickListener = View.OnClickListener { v ->
            val (mode, wbEnum) = when (v.id) {
                R.id.wb_auto -> Pair(android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_AUTO, WhiteBalanceMode.AUTO)
                R.id.wb_daylight -> Pair(android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT, WhiteBalanceMode.DAYLIGHT)
                R.id.wb_cloudy -> Pair(android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT, WhiteBalanceMode.CLOUDY)
                R.id.wb_tungsten -> Pair(android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_INCANDESCENT, WhiteBalanceMode.INCANDESCENT)
                R.id.wb_fluorescent -> Pair(android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_FLUORESCENT, WhiteBalanceMode.FLUORESCENT)
                R.id.wb_twilight -> Pair(android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_TWILIGHT, WhiteBalanceMode.TWILIGHT)
                R.id.wb_shade -> Pair(android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_SHADE, WhiteBalanceMode.SHADE)
                else -> return@OnClickListener
            }
            if (mode in awbModes) {
                camera2Manager.setWhiteBalanceMode(mode)
                currentWbMode = wbEnum
                persistWbMode()
                updateWbUI()
                syncWbSliders()
                showModeLabel(whiteBalanceDisplayName(wbEnum))
                uploadAgxUniforms()
            }
            wbPopup.visibility = View.GONE
        }

        kelvinBtn.setOnClickListener {
            currentWbMode = WhiteBalanceMode.KELVIN
            camera2Manager.setWhiteBalanceMode(android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_OFF)
            persistWbMode()
            updateWbUI()
            syncWbSliders()
            showModeLabel(whiteBalanceDisplayName(WhiteBalanceMode.KELVIN))
            uploadAgxUniforms()
            // Pin HAL WB to the device's fixed D65 reference (DAYLIGHT + AWB_LOCK);
            // applyAwb substitutes it for OFF because vendor HALs ignore OFF.
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
        // Kelvin is always available (software WB, no camera mode needed)
        kelvinBtn.visibility = View.VISIBLE
    }

    private fun setupPinchZoom() {
        scaleGestureDetector = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScaleBegin(detector: ScaleGestureDetector): Boolean {
                isScaling = true
                pendingTap = false
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
        preparePreviewPipeline(previewSize, targetAspect)
        previewRenderer.start()
        camera2Manager.openCamera(lens, previewSize, targetW, targetH, useRaw = developerSwitch.useRawSensor)
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

        var primary = lensManager.selectPrimary() ?: run {
            CrashLogger.log(TAG, "initCamera: selectPrimary returned null")
            openingCamera = false
            Log.e(TAG, "No primary lens found")
            return
        }

        CrashLogger.log(TAG, "initCamera: primary=${primary.cameraId} ${primary.label} level=${primary.hardwareLevel} hasRaw=${primary.hasRawSensor}")

        val rawAvailable = lensManager.hasAnyRawLens()
        developerSwitch.setRawSensorAvailable(rawAvailable)
        if (!rawAvailable) {
            CrashLogger.log(TAG, "initCamera: RAW sensor stream not supported by this device, using YUV fallback mode")
        } else if (!developerSwitch.useRawSensor) {
            CrashLogger.log(TAG, "initCamera: RAW sensor stream available, auto-enabling RAW mode")
            developerSwitch.forceEnableRaw()
        }
        if (developerSwitch.useRawSensor && !primary.hasRawSensor) {
            val rawLens = lensManager.getClosestRawLens(primary)
            if (rawLens != null) {
                CrashLogger.log(TAG, "initCamera: primary lacks RAW, switching to RAW lens ${rawLens.cameraId}")
                lensManager.switchLens(rawLens) {}
                primary = rawLens
            } else {
                CrashLogger.log(TAG, "initCamera: no RAW lens usable, falling back to YUV")
                developerSwitch.forceDisableRaw()
                developerSwitch.showErrorBanner("No RAW-capable lens found")
            }
        }
        updateRawModeWarning()

        val previewSize = lensManager.getBestPreviewSize(primary, maxPreviewDimensions.first, maxPreviewDimensions.second)
        buildLensSelectorUI()

        previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
        previewRenderer.targetAspectRatio = 0f
        uploadAgxUniforms()

        preparePreviewPipeline(previewSize, 0f)
        previewRenderer.start()

        var previewFrameCount = 0
        camera2Manager.onFrameAvailable = frameHandler@{ image ->
            if (!cameraReady) return@frameHandler
            lastFrameArrivalTime = SystemClock.elapsedRealtime()
            if (previewRenderer.useBayerPath) return@frameHandler
            previewFrameCount++
            if (previewFrameCount == 1) {
                CrashLogger.log(TAG, "onFrameAvailable: first frame ${image.width}x${image.height}")
            } else if (previewFrameCount % 30 == 0) {
                CrashLogger.log(TAG, "onFrameAvailable: frame #$previewFrameCount")
            }
            val planes = image.planes
            val w = image.width
            val h = image.height

            if (previewFrameCount == 1) {
                CrashLogger.log(TAG, "YUV planes: Y stride=${planes[0].rowStride} cap=${planes[0].buffer.capacity()} " +
                    "U stride=${planes[1].rowStride} pixel=${planes[1].pixelStride} cap=${planes[1].buffer.capacity()} " +
                    "V stride=${planes[2].rowStride} pixel=${planes[2].pixelStride} cap=${planes[2].buffer.capacity()} " +
                    "size=${w}x${h}")
            }

            val yCopy = extractPlane(planes[0], w, h)
            val uvWidth = w / 2
            val uvHeight = h / 2
            val uCopy = extractPlane(planes[1], uvWidth, uvHeight)
            val vCopy = extractPlane(planes[2], uvWidth, uvHeight)

            previewRenderer.setYuvFrame(yCopy, uCopy, vCopy, w, h)
        }

        camera2Manager.onRawFrameAvailable = rawHandler@{ image ->
            if (!cameraReady) return@rawHandler
            lastFrameArrivalTime = SystemClock.elapsedRealtime()
            if (!previewRenderer.useBayerPath) return@rawHandler
            val rawW = image.width
            val rawH = image.height
            if (rawW <= 0 || rawH <= 0) return@rawHandler
            val plane = image.planes.getOrNull(0) ?: return@rawHandler
            if (plane.pixelStride != 2) return@rawHandler

            rawFrameDelivered = true
            rawFallbackRunnable?.let { mainHandler.removeCallbacks(it) }
            rawFallbackRunnable = null

            val stride = plane.rowStride
            val dest = acquireRawBuffer(rawW, rawH)
            val src = plane.buffer

            // The raw copy (below) can race the camera teardown: onPause closes
            // the ImageReader while a frame is being imported, freeing the
            // plane buffer mid-read. Drop such frames instead of crashing.
            if (!runCatching {
                    fun pixelValue(row: Int, col: Int): Int {
                        val p = row * stride + col * 2
                        return (src.get(p).toInt() and 0xFF) or ((src.get(p + 1).toInt() and 0xFF) shl 8)
                    }

                    if (rawFrameLogCount == 0 || rawFrameLogCount % 150 == 0) {
                        var sampleMin = 0xFFFF
                        var sampleMax = 0
                        var sampleCount = 0
                        var r = 0
                        while (r < rawH) {
                            var c = 0
                            while (c < rawW) {
                                val v = pixelValue(r, c)
                                if (v < sampleMin) sampleMin = v
                                if (v > sampleMax) sampleMax = v
                                sampleCount++
                                c += 64
                            }
                            r += 64
                        }
                        CrashLogger.log(
                            TAG, "onRawFrameAvailable: #$rawFrameLogCount ${rawW}x${rawH} " +
                                "format=${image.format} stride=$stride pixelStride=${plane.pixelStride} " +
                                "samples=$sampleCount min=$sampleMin max=$sampleMax"
                        )
                        Log.d(TAG, "onRawFrameAvailable: ${rawW}x${rawH} min=$sampleMin max=$sampleMax")
                    }
                    rawFrameLogCount++

                    var offset = 0
                    for (row in 0 until rawH) {
                        src.position(row * stride)
                        src.limit(row * stride + rawW * 2)
                        dest.position(offset)
                        dest.put(src)
                        offset += rawW * 2
                    }
                    dest.position(0)
                }.isSuccess
            ) {
                return@rawHandler
            }

            feedLensShadingEstimator(dest, rawW, rawH)

            val ccGains = camera2Manager.latestColorCorrectionGains
            val ccMat = camera2Manager.latestColorCorrectionMatrix
            val gainsOk = ccGains != null &&
                ccGains.size >= 4 &&
                ccGains.all { it.isFinite() && it > 0f }

            // Grey-world fallback for devices that report no HAL gains. It
            // feeds a sensor neutral into the profile path; the legacy gain
            // path below keeps consuming its smoothed channel gains directly.
            val useEstimator = !gainsOk && currentWbMode != WhiteBalanceMode.KELVIN
            if (useEstimator && wbEstimateFrame % 15 == 0) {
                estimateAutoWhiteBalance(dest, rawW, rawH)
            }

            wbEstimateFrame++

val neutral: FloatArray? = if (gainsOk) {
            // The as-shot neutral is the inverse of the HAL's
            // COLOR_CORRECTION_GAINS in every WB mode. In KELVIN the HAL is
            // pinned to the D65 DAYLIGHT preset, so these gains describe the
            // *scene* neutral under that fixed illuminant -- the profile must
            // whiten it, then the app's relative Bradford CAT shifts D65 ->
            // user Kelvin on top. Feeding SENSOR_NEUTRAL_COLOR_POINT here
            // instead would white-bias by the D50(NCP)->D65(DAYLIGHT) gap and
            // overcast the whole image green.
            floatArrayOf(
                1f / ccGains[0],
                1f / ((ccGains[1] + ccGains[2]) * 0.5f),
                1f / ccGains[3]
            )
        } else if (currentWbMode != WhiteBalanceMode.KELVIN) {
            estimatorNeutral ?: rawNeutralSeed
        } else {
            rawNeutralSeed
        }

            val profileTransform = neutral?.let { rawColorProfile?.neutralTransformForNeutral(it) }

            if (profileTransform != null) {
                // Profile path: split into sensor-space white-balance gains and
                // the WB-removed native->sRGB matrix so the demosaic shader can
                // neutralize clipped regions between the two stages. The
                // matrix no longer folds in the WB, and its Y row provides the
                // camera-native luminance coefficients for the neutralization.
                previewRenderer.ccMatrix = profileTransform.colorMatrix.m
                previewRenderer.wbGainR = profileTransform.wbGains[0]
                previewRenderer.wbGainG = profileTransform.wbGains[1]
                previewRenderer.wbGainB = profileTransform.wbGains[2]
                previewRenderer.nativeLumaCoeffs = profileTransform.lumaCoeffs
                if (wbEstimateFrame <= 3 || wbEstimateFrame % 60 == 0) {
                    val temp = rawColorProfile?.temperatureForNeutral(neutral)
                    CrashLogger.log(
                        TAG, "dcp cc: frame=$wbEstimateFrame " +
                            "temp=${temp?.toInt() ?: -1} " +
                            "neutral=[${neutral.joinToString { String.format("%.3f", it) }}] " +
                            "wb=[${previewRenderer.wbGainR}, ${previewRenderer.wbGainG}, ${previewRenderer.wbGainB}] " +
                            "mat=[${profileTransform.colorMatrix.m.joinToString { String.format("%.4f", it) }}] " +
                            "luma=[${previewRenderer.nativeLumaCoeffs.joinToString { String.format("%.4f", it) }}]"
                    )
                }
            } else if (gainsOk) {
                val gMean = (ccGains[1] + ccGains[2]) * 0.5f
                if (gMean > 0f) {
                    previewRenderer.wbGainR = (ccGains[0] / gMean).coerceIn(0.3f, 8f)
                    previewRenderer.wbGainG = 1f
                    previewRenderer.wbGainB = (ccGains[3] / gMean).coerceIn(0.3f, 8f)
                }
                previewRenderer.ccMatrix = ccMat
                previewRenderer.nativeLumaCoeffs = luminanceFromSrgbMatrix(ccMat)
                if (wbEstimateFrame % 60 == 0) {
                    CrashLogger.log(
                        TAG, "hal cc: frame=$wbEstimateFrame " +
                            "gainsR=${String.format("%.3f", previewRenderer.wbGainR)} " +
                            "gainsB=${String.format("%.3f", previewRenderer.wbGainB)} " +
                            "raw=[${ccGains.joinToString { String.format("%.3f", it) }}] " +
                            "mat=[${ccMat?.joinToString { String.format("%.4f", it) }}]"
                    )
                }
            } else if (currentWbMode == WhiteBalanceMode.KELVIN) {
                // Manual Kelvin without a profile or HAL gains: stay neutral —
                // the illumination comes from the app's Kelvin CAT. Never run
                // the scene-adaptive estimator while in manual WB.
                previewRenderer.wbGainR = 1f
                previewRenderer.wbGainG = 1f
                previewRenderer.wbGainB = 1f
                previewRenderer.ccMatrix = ccMat
                previewRenderer.nativeLumaCoeffs = luminanceFromSrgbMatrix(ccMat)
            } else {
                previewRenderer.ccMatrix = null
                previewRenderer.nativeLumaCoeffs = DEFAULT_LUMA_COEFFS.copyOf()
            }
            previewRenderer.setBayerFrame(dest, rawW, rawH, rawW)
        }

        camera2Manager.onSessionReady = { width, height ->
            CrashLogger.log(TAG, "onSessionReady: ${width}x${height}")
            Log.d(TAG, "Camera session ready: ${width}x${height}")
            cameraReady = true
            openingCamera = false

            mainHandler.removeCallbacks(stallWatchdogRunnable)
            mainHandler.postDelayed(stallWatchdogRunnable, 1000)

            if (previewRenderer.useBayerPath) {
                rawFrameDelivered = false
                rawFallbackRunnable?.let { mainHandler.removeCallbacks(it) }
                rawFallbackRunnable = Runnable {
                    rawFallbackRunnable = null
                    if (previewRenderer.useBayerPath && cameraReady && !rawFrameDelivered) {
                        CrashLogger.log(TAG, "RAW fallback: no RAW frames within timeout, reverting to YUV")
                        Log.w(TAG, "No RAW frames received, reverting to YUV fallback")
                        developerSwitch.revertToggle()
                    }
                }
                mainHandler.postDelayed(rawFallbackRunnable!!, 2500)
            }

            mainHandler.post {
                lensSwitchOverlay.visibility = View.GONE
                if (pendingLensInfo) {
                    pendingLensInfo = false
                    lensManager.activeLens?.let { showLensInfoMessage(lensDetailSummary(it)) }
                }
                // Re-apply the active WB mode to the fresh session. The HAL mode
                // (Camera2Manager.currentAwbMode) does not survive a lens switch,
                // but the restored per-lens WB state lives in currentWbMode;
                // without this the image would keep the previous session's mode
                // while the UI shows the restored one.
                camera2Manager.setWhiteBalanceMode(wbModeToCameraMode(currentWbMode))
                // Same for flash: the restored per-lens mode drives the icon
                // but the HAL needs the mode re-applied to this new session.
                camera2Manager.setFlashMode(currentFlashMode)
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
                // Re-apply manual focus to the fresh session. MF persists whether
                // or not the focus panel is open: the AF/MF button never changes the
                // mode itself, only the circular A does.
                if (isManualFocus) {
                    camera2Manager.setManualFocus(focusRollerDistance())
                    updateManualFocusUI()
                } else {
                    camera2Manager.resetAutoFocus()
                    val cx = textureView.width / 2f
                    val cy = textureView.height / 2f
                    showFocusIndicator(cx, cy)
                    applyFocusPoint(cx, cy)
                }
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

        camera2Manager.onMaxZoomReady = { maxZoom ->
            mainHandler.post {
                previewRenderer.zoomController.setMaxZoom(maxZoom)
                CrashLogger.log(TAG, "onMaxZoomReady: maxZoom=$maxZoom")
            }
        }

        camera2Manager.onLensShadingMapAvailable = { pixels, w, h ->
            // A usable (non-identity) HAL map exists on this device: record it,
            // cancel the "no map" grace timer, and in AUTO mode let the stock
            // map take over. If the user has explicitly chosen the sampled map,
            // the estimator stays in charge (their choice wins).
            halMapStatus = HalMapStatus.AVAILABLE
            previewResPrefs.edit().putBoolean(PREF_LS_HAL_MAP, true).apply()
            mainHandler.removeCallbacks(lensShadingGraceRunnable)
            lastHalMapPixels = pixels
            lastHalMapWidth = w
            lastHalMapHeight = h
            if (useStockMap) {
                if (::lensShadingEstimator.isInitialized) lensShadingEstimator.pause()
                previewRenderer.submitLensShadingMap(pixels, w, h)
            }
            mainHandler.post { updateLensShadingUI() }
        }

        camera2Manager.startBackgroundThread()

        try {
            previewRenderer.sensorOrientation = lensManager.getSensorOrientation(primary)
            previewRenderer.isFrontCamera = primary.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
            CrashLogger.log(TAG, "initCamera: calling openCamera sensorOrientation=${previewRenderer.sensorOrientation} isFront=${previewRenderer.isFrontCamera}")
            camera2Manager.openCamera(primary, previewSize, 0, 0, useRaw = developerSwitch.useRawSensor)
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
    }

    private fun rawSizeForLens(lens: LensInfo): Size? =
        if (lens.hasRawSensor) camera2Manager.resolveRawSize(lens.cameraId) else null

    private fun applyRawMetadata(lens: LensInfo) {
        try {
            rawSessionCountSinceLaunch++
            val cm = getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager
            val chars = cm.getCameraCharacteristics(lens.cameraId)
            val meta = RawMetadataParser(chars)
            val profile = RawColorProfile(chars)
            previewRenderer.bayerColorMap = meta.bayerColorMap
            previewRenderer.bayerBlackLevelPattern = meta.blackLevelPattern
            previewRenderer.agxWhiteLevel = meta.whiteLevel.toFloat()
            previewRenderer.agxBlackLevel = meta.blackLevelAverage
            rawNeutralSeed = meta.neutralColorPoint
            estimatorNeutral = meta.neutralColorPoint
            rawColorProfile = profile
            if (profile.available) {
                val tr = profile.neutralTransformForNeutral(meta.neutralColorPoint)
                if (tr != null) {
                    // Split white balance (sensor-space gains) from the
                    // WB-removed native->sRGB matrix; the demosaic shader
                    // neutralizes clipped regions in between. The matrix
                    // no longer folds in the WB.
                    previewRenderer.ccMatrix = tr.colorMatrix.m
                    previewRenderer.wbGainR = tr.wbGains[0]
                    previewRenderer.wbGainG = tr.wbGains[1]
                    previewRenderer.wbGainB = tr.wbGains[2]
                    previewRenderer.nativeLumaCoeffs = tr.lumaCoeffs
                } else {
                    // Degenerate neutral: fall back to gains-neutral and the
                    // plain matrix, keeping the pre-split behaviour.
                    previewRenderer.wbGainR = 1f
                    previewRenderer.wbGainG = 1f
                    previewRenderer.wbGainB = 1f
                    previewRenderer.ccMatrix = profile.srgbMatrixForNeutral(meta.neutralColorPoint)?.m
                    previewRenderer.nativeLumaCoeffs = DEFAULT_LUMA_COEFFS.copyOf()
                }
            } else {
                rawColorProfile = null
                val sensorGains = meta.sensorWhiteBalanceGains
                previewRenderer.wbGainR = sensorGains[0]
                previewRenderer.wbGainG = sensorGains[1]
                previewRenderer.wbGainB = sensorGains[2]
                previewRenderer.ccMatrix = null
                previewRenderer.nativeLumaCoeffs = DEFAULT_LUMA_COEFFS.copyOf()
            }
            wbEstimateFrame = 0
            // A new lens on the same renderer: start from the identity no-op
            // until its own shading map arrives with the first CaptureResults.
            previewRenderer.resetLensShading()

            // Lens shading: fresh estimator per lens; restore a previously learned
            // map from disk. In AUTO/STOCK on a device that emits a usable HAL
            // map, the HAL map takes over when its first CaptureResult arrives;
            // otherwise the sampled estimator auto-starts if correction is
            // enabled and no map exists yet.
            rawBayerPattern = meta.bayerPattern
            rawWhiteLevel = meta.whiteLevel
            rawBlackLevel = meta.blackLevelAverage
            rawSensorWidth = meta.sensorWidth
            rawSensorHeight = meta.sensorHeight
            armHalMapGraceCheck()
            lastEstimatorMap = null
            lensShadingMapPersisted = false
            if (::lensShadingEstimator.isInitialized) {
                lensShadingEstimator.reset()
                val saved = loadLensShadingMap(lens.cameraId)
                if (saved != null) {
                    lastEstimatorMap = saved
                    if (!useStockMap) submitSampledMap(saved)
                    CrashLogger.log(
                        TAG, "lensShading: restored saved map for lens ${lens.cameraId} " +
                            "${saved.width}x${saved.height}"
                    )
                }
                autoStartSamplingIfNeeded()
                maybeShowLensShadingNotice()
                mainHandler.post { updateLensShadingUI() }
            }
            CrashLogger.log(
                TAG, "applyRawMetadata: ${meta.sensorWidth}x${meta.sensorHeight} white=${meta.whiteLevel} " +
                    "blackAvg=${meta.blackLevelAverage} pattern=${meta.bayerPattern.label} " +
                    "profile=${if (profile.available) "yes" else "no"} " +
                    (if (profile.available)
                        "illum=${profile.colorTemperature1.toInt()}K/${profile.colorTemperature2.toInt()}K fwd=${profile.hasForwardMatrix} " +
                            "neutral=[${meta.neutralColorPoint.joinToString { String.format("%.3f", it) }}] " +
                            "mat=[${previewRenderer.ccMatrix?.joinToString { String.format("%.4f", it) } ?: "null"}]"
                    else
                        "wbGains=[${String.format("%.2f", previewRenderer.wbGainR)}, " +
                            "${String.format("%.2f", previewRenderer.wbGainG)}, " +
                            "${String.format("%.2f", previewRenderer.wbGainB)}]")
            )
            Log.d(TAG, "applyRawMetadata: ${meta.sensorWidth}x${meta.sensorHeight} white=${meta.whiteLevel} black=${meta.blackLevelAverage}")
        } catch (e: Exception) {
            Log.w(TAG, "applyRawMetadata failed: ${e.message}", e)
            resetRawMetadata()
        }
    }

    private fun resetRawMetadata() {
        previewRenderer.bayerColorMap = intArrayOf(0, 1, 1, 2)
        previewRenderer.bayerBlackLevelPattern = intArrayOf(64, 64, 64, 64)
        previewRenderer.agxWhiteLevel = 1023f
        previewRenderer.agxBlackLevel = 64f
        previewRenderer.wbGainR = 1f
        previewRenderer.wbGainG = 1f
        previewRenderer.wbGainB = 1f
        previewRenderer.nativeLumaCoeffs = DEFAULT_LUMA_COEFFS.copyOf()
        rawColorProfile = null
        rawNeutralSeed = floatArrayOf(1f, 1f, 1f)
        estimatorNeutral = null
        mainHandler.removeCallbacks(lensShadingGraceRunnable)
        lastEstimatorMap = null
        lensShadingMapPersisted = false
        if (::lensShadingEstimator.isInitialized) {
            lensShadingEstimator.reset()
            rawSensorWidth = 0
            rawSensorHeight = 0
            mainHandler.post { updateLensShadingUI() }
        }
    }

    // Luminance (Y) coefficients of the camera-native RGB space, derived from a
    // camera-native -> linear-sRGB matrix (the HAL COLOR_CORRECTION_TRANSFORM).
    // The Y row of rgbToXYZ(REC709) * matrix maps the camera-space signal onto
    // the D65 XYZ luminance axis used by the demosaic clipping neutralization.
    private fun luminanceFromSrgbMatrix(mat: FloatArray?): FloatArray {
        if (mat == null) return DEFAULT_LUMA_COEFFS.copyOf()
        val camToXyz = ColorMatrix.multiply(ColorMatrix.rgbToXYZ(ColorMatrix.REC709), ColorMatrix.Mat3(mat))
        return floatArrayOf(camToXyz.m[3], camToXyz.m[4], camToXyz.m[5])
    }

    private fun preparePreviewPipeline(previewSize: Size, targetAspect: Float) {
        previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
        previewRenderer.targetAspectRatio = targetAspect
        val lens = lensManager.activeLens
        val raw = lens?.let { rawSizeForLens(it) }
        if (developerSwitch.useRawSensor && raw != null) {
            applyRawMetadata(lens!!)
            previewRenderer.enableBayerMode(raw.width, raw.height)
            rawFrameDelivered = false
            rawFrameLogCount = 0
        } else {
            resetRawMetadata()
            previewRenderer.disableBayerMode()
        }
    }

    private val rawBuffers = mutableListOf<ByteBuffer>()
    private var rawBufferFrame = 0
    private var wbEstimateFrame = 0

    // Profile-derived RAW color path (static per camera; only the as-shot
    // neutral varies frame to frame). Null when the device reports no usable
    // sensor color transforms -- then the HAL COLOR_CORRECTION_* state is used.
    private var rawColorProfile: RawColorProfile? = null
    private var rawNeutralSeed = floatArrayOf(1f, 1f, 1f)
    private var estimatorNeutral: FloatArray? = null

    private fun acquireRawBuffer(width: Int, height: Int): ByteBuffer {
        val needed = width * height * 2
        if (rawBuffers.isNotEmpty() && rawBuffers[0].capacity() != needed) {
            rawBuffers.clear()
        }
        while (rawBuffers.size < RAW_BUFFER_POOL) {
            rawBuffers.add(ByteBuffer.allocateDirect(needed))
        }
        val dest = rawBuffers[rawBufferFrame % RAW_BUFFER_POOL]
        rawBufferFrame++
        dest.clear()
        return dest
    }

    private fun estimateAutoWhiteBalance(buffer: java.nio.ByteBuffer, w: Int, h: Int) {
        val step = 32
        val colorMap = previewRenderer.bayerColorMap
        val black = previewRenderer.bayerBlackLevelPattern
        val cnt = IntArray(4)
        val sum = DoubleArray(4)
        for (p in 0 until 4) {
            val startR = p / 2
            val startC = p % 2
            var r = startR
            while (r < h) {
                var c = startC
                while (c < w) {
                    val idx = (r * w + c) * 2
                    val v = (buffer.get(idx).toInt() and 0xFF) or
                        ((buffer.get(idx + 1).toInt() and 0xFF) shl 8)
                    sum[p] += v
                    cnt[p]++
                    c += step
                }
                r += step
            }
        }

        var rPhase = 0
        var bPhase = 3
        for (p in 0 until 4) {
            when (colorMap.getOrElse(p) { 1 }) {
                0 -> rPhase = p
                2 -> bPhase = p
            }
        }

        val avg = DoubleArray(4)
        for (i in 0 until 4) {
            avg[i] = if (cnt[i] > 0) sum[i] / cnt[i] - black[i] else 0.0
        }
        var gSum = 0.0
        var gCount = 0
        for (p in 0 until 4) {
            if (colorMap.getOrElse(p) { 1 } == 1) {
                gSum += avg[p]
                gCount++
            }
        }
        val gAvg = if (gCount > 0) gSum / gCount else (avg[1] + avg[2]) / 2.0
        val logNow = wbEstimateFrame % 90 == 0 || wbEstimateFrame <= 3
        if (gAvg <= 1.0 || avg[rPhase] <= 1.0 || avg[bPhase] <= 1.0) {
            if (logNow) {
                CrashLogger.log(
                    TAG, "wb estimate: skip frame=$wbEstimateFrame " +
                        "gAvg=${String.format("%.1f", gAvg)} rAvg=${String.format("%.1f", avg[rPhase])} " +
                        "bAvg=${String.format("%.1f", avg[bPhase])} " +
                        "cnt=[${cnt[0]},${cnt[1]},${cnt[2]},${cnt[3]}]"
                )
            }
            return
        }

        val targetR = (gAvg / avg[rPhase]).toFloat().coerceIn(0.5f, 8f)
        val targetB = (gAvg / avg[bPhase]).toFloat().coerceIn(0.5f, 8f)

        val a = 0.4f
        previewRenderer.wbGainR = previewRenderer.wbGainR + a * (targetR - previewRenderer.wbGainR)
        previewRenderer.wbGainB = previewRenderer.wbGainB + a * (targetB - previewRenderer.wbGainB)

        // Feed the smoothed grey-world result as a sensor-space neutral
        // (the inverse of the gains) for the profile-derived matrix path.
        val prevNeutral = estimatorNeutral ?: floatArrayOf(1f, 1f, 1f)
        estimatorNeutral = floatArrayOf(
            prevNeutral[0] + a * (1f / targetR - prevNeutral[0]),
            1f,
            prevNeutral[2] + a * (1f / targetB - prevNeutral[2])
        )

        if (logNow) {
            CrashLogger.log(
                TAG, "wb estimate: frame=$wbEstimateFrame " +
                    "rAvg=${String.format("%.1f", avg[rPhase])} gAvg=${String.format("%.1f", gAvg)} " +
                    "bAvg=${String.format("%.1f", avg[bPhase])} " +
                    "gains=${String.format("%.2f", previewRenderer.wbGainR)}," +
                    "${String.format("%.2f", previewRenderer.wbGainG)}," +
                    "${String.format("%.2f", previewRenderer.wbGainB)}" +
                    "neutral=[${estimatorNeutral?.joinToString { String.format("%.3f", it) } ?: "null"}]"
            )
        }
    }

    // --- Lens Shading estimator plumbing ---

    private fun feedLensShadingEstimator(buffer: ByteBuffer, w: Int, h: Int) {
        if (useStockMap || !::lensShadingEstimator.isInitialized) return
        // Accumulate only for the lens geometry we started with.
        if (rawSensorWidth <= 0 || rawSensorWidth != w || rawSensorHeight != h) return
        val map = lensShadingEstimator.addFrame(buffer) ?: return
        lastEstimatorMap = map
        if (lensShadingEstimator.mapDelta > 0.005f || lensShadingEstimator.sampledFrames <= 2) {
            submitSampledMap(map)
        }
        if (lensShadingUiFrameCount % 12 == 0) {
            mainHandler.post { updateLensShadingUI() }
        }
        lensShadingUiFrameCount++
        if (lensShadingEstimator.currentState == LensShadingState.CONVERGED && !lensShadingMapPersisted) {
            saveLearnedLensShading()
        }
    }

    private fun submitSampledMap(map: LensShadingData) {
        if (useStockMap) return
        val scaled = scaleLensShadingGains(map, lensShadingStrength)
        previewRenderer.submitLensShadingMap(
            scaled.toRgba16fFlipped(rawBayerPattern), scaled.width, scaled.height
        )
    }

    private fun applyLensShadingStrengthLive() {
        if (useStockMap) return
        lastEstimatorMap?.let { submitSampledMap(it) }
    }

    private fun saveLearnedLensShading() {
        if (useStockMap) return
        val lens = lensManager.activeLens ?: return
        val map = lastEstimatorMap ?: return
        if (lensShadingEstimator.sampledFrames < 5 || !map.available) return
        if (persistLensShadingMap(lens.cameraId, map)) lensShadingMapPersisted = true
    }

    private fun lensShadingFile(lensId: String): File = File(filesDir, "lens_shading_$lensId.json")

    private fun persistLensShadingMap(lensId: String, map: LensShadingData): Boolean {
        return try {
            val obj = JSONObject()
            obj.put("w", map.width)
            obj.put("h", map.height)
            obj.put("r", flattenGains(map.rGains))
            obj.put("gr", flattenGains(map.grGains))
            obj.put("gb", flattenGains(map.gbGains))
            obj.put("b", flattenGains(map.bGains))
            val f = lensShadingFile(lensId)
            FileOutputStream(f).use { it.write(obj.toString().toByteArray(Charsets.UTF_8)) }
            CrashLogger.log(TAG, "lensShading: saved map for lens $lensId ${map.width}x${map.height}")
            true
        } catch (e: Exception) {
            CrashLogger.log(TAG, "lensShading: save failed ${e.message}")
            false
        }
    }

    private fun loadLensShadingMap(lensId: String): LensShadingData? {
        val f = lensShadingFile(lensId)
        if (!f.exists()) return null
        return try {
            val obj = JSONObject(f.readText(Charsets.UTF_8))
            val w = obj.getInt("w")
            val h = obj.getInt("h")
            if (w <= 0 || h <= 0) return null
            fun gains(key: String): Array<FloatArray> {
                val arr = obj.getJSONArray(key)
                return Array(h) { r ->
                    FloatArray(w) { c -> arr.getDouble(r * w + c).toFloat() }
                }
            }
            val r = gains("r")
            val gr = gains("gr")
            val gb = gains("gb")
            val b = gains("b")
            val maxGain = (r + gr + gb + b).maxOf { row -> row.maxOrNull() ?: 1f }
            LensShadingData(r, gr, gb, b, w, h, maxGain > 1.02f)
        } catch (e: Exception) {
            CrashLogger.log(TAG, "lensShading: load failed ${e.message}")
            null
        }
    }

    private fun deleteLensShadingPersistence(lensId: String) {
        try {
            val f = lensShadingFile(lensId)
            if (f.exists()) f.delete()
        } catch (_: Exception) {
        }
    }

    private fun flattenGains(rows: Array<FloatArray>): JSONArray {
        val arr = JSONArray()
        for (row in rows) for (v in row) arr.put(v.toDouble())
        return arr
    }

    private fun updateLensShadingUI() {
        val frames = lensShadingEstimator.sampledFrames
        var text: String
        var color: Int
        when {
            useStockMap -> {
                text = "Stock map (HAL)"; color = 0xFF26A69A.toInt()
            }
            lensShadingEstimator.currentState == LensShadingState.SAMPLING -> {
                text = "Learning"; color = 0xFFFFA726.toInt()
            }
            lensShadingEstimator.currentState == LensShadingState.PAUSED -> {
                text = "Paused"; color = 0xFF42A5F5.toInt()
            }
            lensShadingEstimator.currentState == LensShadingState.CONVERGED -> {
                text = "Applied"; color = 0xFF7CB342.toInt()
            }
            lensShadingEstimator.currentState == LensShadingState.IDLE && lastEstimatorMap != null -> {
                text = "Applied (saved)"; color = 0xFF7CB342.toInt()
            }
            else -> {
                text = "Idle"; color = 0xFF757575.toInt()
            }
        }
        if (!useStockMap) {
            if (frames > 0) text += " · $frames"
            val elapsedS = lensShadingEstimator.elapsedMillis / 1000
            if (elapsedS >= 1 && lensShadingEstimator.currentState != LensShadingState.IDLE) text += " · ${elapsedS}s"
        }
        lensShadingStatusPill.text = text
        lensShadingStatusPill.setBackgroundColor(color)
        lensShadingStartBtn.isEnabled = !useStockMap
        updateLensShadingToggle()
    }

    // Stock map source: only devices that emit a usable (non-identity) HAL map
    // and are not pinned to the sampled map use the HAL map.
    private val useStockMap: Boolean
        get() = halMapStatus == HalMapStatus.AVAILABLE && lensShadingSourceMode != LensShadingSourceMode.SAMPLED

    private fun updateLensShadingToggle() {
        val forcedSampled = halMapStatus == HalMapStatus.MISSING
        val selected = if (useStockMap) LensShadingSourceMode.STOCK else LensShadingSourceMode.SAMPLED
        val enabled = !forcedSampled
        lensShadingStockBtn.isEnabled = enabled
        lensShadingSampledBtn.isEnabled = enabled
        lensShadingStockBtn.alpha = if (enabled) 1f else 0.5f
        lensShadingSampledBtn.alpha = if (enabled) 1f else 0.5f
        fun style(btn: TextView, active: Boolean) {
            btn.background = android.graphics.drawable.ColorDrawable(if (active) 0xFFFFA726.toInt() else 0xFF333333.toInt())
            btn.setTextColor(if (active) 0xFF000000.toInt() else 0xFFCCCCCC.toInt())
        }
        style(lensShadingStockBtn, selected == LensShadingSourceMode.STOCK)
        style(lensShadingSampledBtn, selected == LensShadingSourceMode.SAMPLED)
    }

    private val lensShadingNoticeHideRunnable = Runnable {
        lensShadingNoticeOverlay.animate().cancel()
        lensShadingNoticeOverlay.animate().alpha(0f).setDuration(300).withEndAction {
            lensShadingNoticeOverlay.visibility = View.GONE
        }.start()
    }

    private fun maybeShowLensShadingNotice() {
        if (lensShadingNoticeShown) return
        if (rawSessionCountSinceLaunch != 1) return
        if (halMapStatus != HalMapStatus.MISSING) return
        if (useStockMap) return
        if (lastEstimatorMap != null) return
        if (lensShadingEstimator.currentState != LensShadingState.SAMPLING) return
        lensShadingNoticeShown = true
        lensShadingNoticeOverlay.removeCallbacks(lensShadingNoticeHideRunnable)
        lensShadingNoticeOverlay.animate().cancel()
        lensShadingNoticeOverlay.text =
            "This device can't provide a lens shading map - auto-sampling the lens\n" +
                "shading now. Check the Lens Shading settings to adjust."
        lensShadingNoticeOverlay.alpha = 0f
        lensShadingNoticeOverlay.visibility = View.VISIBLE
        lensShadingNoticeOverlay.animate().alpha(1f).setDuration(200).withEndAction {
            lensShadingNoticeOverlay.postDelayed(lensShadingNoticeHideRunnable, 10_000)
        }.start()
    }

    private fun selectLensShadingSourceMode(mode: LensShadingSourceMode) {
        if (halMapStatus == HalMapStatus.MISSING) return
        if (mode == lensShadingSourceMode) {
            updateLensShadingUI()
            return
        }
        lensShadingSourceMode = mode
        previewResPrefs.edit().putInt(PREF_LS_SOURCE_MODE, mode.ordinal).apply()
        CrashLogger.log(
            TAG, "lensShading: source mode=$mode stockPinned=${halMapStatus == HalMapStatus.MISSING} " +
                "halMapStatus=$halMapStatus"
        )
        if (useStockMap) {
            lensShadingEstimator.pause()
            val pixels = lastHalMapPixels
            if (pixels != null) {
                previewRenderer.submitLensShadingMap(pixels, lastHalMapWidth, lastHalMapHeight)
            } else {
                previewRenderer.resetLensShading()
            }
        } else {
            lensShadingEstimator.resume()
            autoStartSamplingIfNeeded()
            val m = lastEstimatorMap
            if (m != null) submitSampledMap(m) else previewRenderer.resetLensShading()
        }
        updateLensShadingUI()
    }

    // Grace window: if a freshly opened RAW session never emits a usable HAL map,
    // the device is pinned to the sampled map. Covers both "no map at all" and
    // "identity-only" HALs (the parser drops identity maps upstream).
    private val lensShadingGraceRunnable = Runnable {
        if (halMapStatus != HalMapStatus.UNKNOWN) return@Runnable
        halMapStatus = HalMapStatus.MISSING
        previewResPrefs.edit().putBoolean(PREF_LS_HAL_MAP, false).apply()
        CrashLogger.log(
            TAG, "lensShading: no non-identity HAL map within ${HAL_MAP_GRACE_MS}ms - " +
                "pinning to sampled map"
        )
        if (::lensShadingEstimator.isInitialized) {
            autoStartSamplingIfNeeded()
            maybeShowLensShadingNotice()
            updateLensShadingUI()
        }
    }

    private fun armHalMapGraceCheck() {
        if (halMapStatus != HalMapStatus.UNKNOWN) return
        mainHandler.removeCallbacks(lensShadingGraceRunnable)
        mainHandler.postDelayed(lensShadingGraceRunnable, HAL_MAP_GRACE_MS)
    }

    // Auto-start the sampled estimator when correction is enabled, no map exists
    // yet for this lens, and we are not in stock mode. Strength 0 means "user
    // turned correction off" — never force it back on at startup.
    private fun autoStartSamplingIfNeeded() {
        if (useStockMap) return
        if (lensShadingStrength <= 0f) return
        if (lastEstimatorMap != null) return
        if (rawSensorWidth <= 0 || rawSensorHeight <= 0) return
        if (lensShadingEstimator.currentState == LensShadingState.SAMPLING) return
        lensShadingEstimator.start(rawSensorWidth, rawSensorHeight, rawBayerPattern, rawWhiteLevel, rawBlackLevel)
        lensShadingUiFrameCount = 0
        lastEstimatorMap = null
        lensShadingMapPersisted = false
        if (!useStockMap) previewRenderer.resetLensShading()
        CrashLogger.log(
            TAG, "lensShading: auto-start sampling ${rawSensorWidth}x${rawSensorHeight} " +
                "pattern=${rawBayerPattern.label} strength=${lensShadingStrength}"
        )
    }

    private fun restartCamera() {
        CrashLogger.log(TAG, "restartCamera")
        mainHandler.removeCallbacks(stallWatchdogRunnable)
        mainHandler.removeCallbacks(lensShadingGraceRunnable)
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

        camera2Manager.startBackgroundThread()
        previewRenderer.sensorOrientation = lensManager.getSensorOrientation(lens)
        previewRenderer.isFrontCamera = lens.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
        preparePreviewPipeline(previewSize, targetAspect)
        previewRenderer.start()
        camera2Manager.openCamera(lens, previewSize, photoOutput.resolutionWidth, photoOutput.resolutionHeight, useRaw = developerSwitch.useRawSensor)
        previewRenderer.setCaptureSize(camera2Manager.captureSize.width, camera2Manager.captureSize.height)
    }

    private fun switchToLens(targetLens: LensInfo) {
        CrashLogger.log(TAG, "switchToLens: target=${targetLens.cameraId} ${targetLens.label}")
        val currentLens = lensManager.activeLens ?: return

        lensSwitchOverlay.text = "Switching to ${targetLens.label}\u2026"
        lensSwitchOverlay.visibility = View.VISIBLE
        pendingLensInfo = true

        mainHandler.removeCallbacks(stallWatchdogRunnable)
        mainHandler.removeCallbacks(lensShadingGraceRunnable)

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

        // WB stays global across lens switches: never restore the target lens's
        // saved snapshot, or the user's selected mode (e.g. Kelvin) is silently
        // replaced by the other lens's stale AUTO. Push the current global mode
        // to the HAL AWB request, or the new session inherits the *previous*
        // lens's AWB mode (e.g. Kelvin's OFF) because Camera2Manager survives
        // the close(). This must happen before openCamera builds the first
        // request on the new device.
        camera2Manager.setWhiteBalanceMode(wbModeToCameraMode(currentWbMode))

        // Flash mode is global, not per-lens: keep the current mode so the
        // torch does not get dropped when switching to a lens whose saved
        // state is OFF. The active session re-applies it below.

        updateFlashUI()
        updateWbUI()
        syncWbSliders()
        thermalManager.isTorchActive = (currentFlashMode == FlashMode.TORCH)
        // The fresh session inherits the *previous* session's flash state, so
        // push the current mode before openCamera builds its first request
        // (the request itself no-ops until a session exists, but the value
        // must be set so icon and torch stay in sync).
        camera2Manager.setFlashMode(currentFlashMode)
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
        previewRenderer.sensorOrientation = lensManager.getSensorOrientation(targetLens)
        previewRenderer.isFrontCamera = targetLens.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
        preparePreviewPipeline(previewSize, targetAspect)
        previewRenderer.start()
        camera2Manager.openCamera(targetLens, previewSize, photoOutput.resolutionWidth, photoOutput.resolutionHeight, useRaw = developerSwitch.useRawSensor)
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
                val focal = if (lens.focalLength35mmEq > 0f) {
                    "${Math.round(lens.focalLength35mmEq)}mm (${String.format("%.1f", lens.focalLengthMm)} physical)"
                } else {
                    "${String.format("%.1f", lens.focalLengthMm)}mm"
                }
                Toast.makeText(this@MainActivity, "${lens.label}: $focal, HW level ${lens.hardwareLevel}", Toast.LENGTH_SHORT).show()
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
            previewRenderer.sensorOrientation = lensManager.getSensorOrientation(active)
            previewRenderer.isFrontCamera = active.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
            preparePreviewPipeline(previewSize, 0f)
            previewRenderer.start()
            camera2Manager.startBackgroundThread()
            camera2Manager.openCamera(active, previewSize, photoOutput.resolutionWidth, photoOutput.resolutionHeight, useRaw = developerSwitch.useRawSensor)
            previewRenderer.setCaptureSize(camera2Manager.captureSize.width, camera2Manager.captureSize.height)
            return
        }

        for (lens in lensesForFacing) {
            lensSelector.addView(createLensButton(lens, lens.cameraId == active.cameraId))
        }
    }

    private fun uploadAgxUniforms() {
        // For YUV path: camera already handles CCM/AWB, so AUTO is identity.
        // KELVIN applies only the relative chromatic adaptation (user illuminant -> D65),
        // not the absolute xyzToRGB conversion which is for raw sensor data.
        val sceneLinearTo709 = when (currentWbMode) {
            WhiteBalanceMode.AUTO -> ColorMatrix.identity()
            WhiteBalanceMode.KELVIN -> {
                val d65xy = Pair(ColorMatrix.D65_X, ColorMatrix.D65_Y)
                val userXY = WhiteBalanceMath.kelvinToXy(kelvinState.kelvin, kelvinState.tint)
                // Kelvin is fully manual: the HAL white-balance is pinned to the
                // same fixed DAYLIGHT reference the Sun preset uses and locked
                // (see Camera2Manager.applyAwb), and the raw path mirrors the
                // HAL's WB gains, so input is already balanced exactly like the
                // preset. Here we only apply the relative chromatic adaptation
                // (user illuminant -> D65); the default 6300K/-14 is chosen so
                // userXY == D65, making this matrix identity — which must
                // therefore not change the Sun/DAYLIGHT visual.
                val bradford = WhiteBalanceMath.chromaticAdaptationBradford(userXY, d65xy)
                val m = bradford.m
                CrashLogger.log(TAG, "uploadAgxUniforms KELVIN: kelvin=${kelvinState.kelvin} tint=${kelvinState.tint} " +
                    "userXY=(${String.format("%.6f", userXY.first)}, ${String.format("%.6f", userXY.second)}) " +
                    "d65XY=(${String.format("%.6f", d65xy.first)}, ${String.format("%.6f", d65xy.second)}) " +
                    "bradford=[${String.format("%.6f", m[0])},${String.format("%.6f", m[1])},${String.format("%.6f", m[2])}, " +
                    "${String.format("%.6f", m[3])},${String.format("%.6f", m[4])},${String.format("%.6f", m[5])}, " +
                    "${String.format("%.6f", m[6])},${String.format("%.6f", m[7])},${String.format("%.6f", m[8])}] " +
                    "awbMode=$currentWbMode")
                bradford
            }
            WhiteBalanceMode.DAYLIGHT -> ColorMatrix.identity()
            WhiteBalanceMode.CLOUDY -> ColorMatrix.identity()
            WhiteBalanceMode.INCANDESCENT -> ColorMatrix.identity()
            WhiteBalanceMode.FLUORESCENT -> ColorMatrix.identity()
            WhiteBalanceMode.TWILIGHT -> ColorMatrix.identity()
            WhiteBalanceMode.SHADE -> ColorMatrix.identity()
        }

        val insetParams = agxParams.toInsetParams()
        val agx = AgxPrecomputer.compute(insetParams, sceneLinearTo709, whiteLevel = 1023f, blackLevel = 64f, middleGrayPercent = agxParams.middleGray)

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
        previewRenderer.agxVibrance = agxParams.vibrance
        previewRenderer.bayerNrStrength = agxParams.nrStrength
    }

    private fun extractPlane(plane: android.media.Image.Plane, width: Int, height: Int): java.nio.ByteBuffer {
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride

        if (rowStride != width || buffer.capacity() != width * height) {
            Log.d(TAG, "extractPlane: size=${width}x${height} rowStride=$rowStride " +
                "pixelStride=$pixelStride capacity=${buffer.capacity()}")
        }

        // Use absolute buffer.get(index) to bypass Huawei's broken buffer.limit().
        // The limit is wrong on some devices, but capacity is always correct.
        val out = ByteArray(width * height)

        for (row in 0 until height) {
            val rowStart = row * rowStride
            val lastIndex = rowStart + (width - 1) * pixelStride

            if (lastIndex >= buffer.capacity()) break

            for (col in 0 until width) {
                out[row * width + col] = buffer.get(rowStart + col * pixelStride)
            }
        }

        val dst = java.nio.ByteBuffer.allocateDirect(width * height).order(java.nio.ByteOrder.nativeOrder())
        dst.put(out)
        dst.position(0)
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

                preparePreviewPipeline(previewSize, targetAspect)
                previewRenderer.start()
                camera2Manager.startBackgroundThread()
                previewRenderer.sensorOrientation = lensManager.getSensorOrientation(lens)
                previewRenderer.isFrontCamera = lens.facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
                camera2Manager.openCamera(lens, previewSize, photoOutput.resolutionWidth, photoOutput.resolutionHeight, useRaw = developerSwitch.useRawSensor)
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
        mainHandler.removeCallbacks(stallWatchdogRunnable)
        mainHandler.removeCallbacks(lensShadingGraceRunnable)
        camera2Manager.close()
        camera2Manager.stopBackgroundThread()
        cameraReady = false
        previewRenderer.stop()
    }

    override fun onDestroy() {
        focusIndicatorHandler.removeCallbacks(focusIndicatorHideRunnable)
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
                shutterStateLabel.text = ""
            }
            ShutterController.State.CAPTURING -> {
                shutterButton.isEnabled = false
                shutterButton.alpha = 0.5f
                shutterStateLabel.text = "Capturing..."
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
                val yBuffer = extractPlane(planes[0], w, h)
                val uBuffer = extractPlane(planes[1], w / 2, h / 2)
                val vBuffer = extractPlane(planes[2], w / 2, h / 2)

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
                    focalLength35mm = session.focalLength35mm,
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

                saveCaptureJpeg(jpegData, thumbnailJpeg, session, metadata)
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

    /** Encodes the already-processed bitmap to JPEG and builds capture EXIF metadata. */
    private fun encodeCaptureBitmap(bitmap: Bitmap, session: CaptureSession): Pair<ByteArray, CaptureMetadata> {
        val activeLens = lensManager.activeLens ?: lensManager.selectPrimary()
        val chars = activeLens?.let { lensManager.getCharacteristicsForLens(it) }
        val wallClockOffsetMs = System.currentTimeMillis() - (SystemClock.elapsedRealtimeNanos() / 1_000_000)
        val timestampSource = chars?.get(
            android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE
        ) ?: android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN
        val captureWallClockMs = if (timestampSource == android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) {
            SystemClock.elapsedRealtimeNanos() / 1_000_000 + wallClockOffsetMs
        } else {
            System.currentTimeMillis()
        }
        val (rawIso, rawShutterNs) = camera2Manager.lastExposureForExif()
        val isFront = session.isFrontCamera
        val exifRotation = if (isFront) {
            (session.sensorOrientation - session.deviceOrientation + 360) % 360
        } else {
            (session.sensorOrientation + session.deviceOrientation) % 360
        }
        val exifOrientation = when (exifRotation) {
            90 -> 6
            180 -> 3
            270 -> 8
            else -> 1
        }
        val metadata = CaptureMetadata(
            sensorOrientation = session.sensorOrientation,
            exifOrientation = exifOrientation,
            focalLengthMm = session.focalLengthMm,
            focalLength35mm = session.focalLength35mm,
            iso = rawIso,
            exposureTimeNs = rawShutterNs,
            flashMode = session.flashMode,
            aeState = null,
            captureWallClockMs = captureWallClockMs
        )
        return Pair(JpegEncoder.encodeToJpeg(bitmap, session.jpegQuality), metadata)
    }

    private fun saveCaptureJpeg(jpegData: ByteArray, thumbnailJpeg: ByteArray?, session: CaptureSession, metadata: CaptureMetadata) {
        var tempFile: java.io.File? = null
        try {
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
            Log.e(TAG, "Save capture failed", e)
            mainHandler.post {
                Toast.makeText(this, "Capture failed: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        } finally {
            tempFile?.delete()
        }
    }

    private fun processRawSnapshot(
        frame: com.agx.camera.gpu.PreviewRenderer.RawFrameCopy,
        session: CaptureSession,
        thumbnailLatch: CountDownLatch,
        thumbnailRef: java.util.concurrent.atomic.AtomicReference<Bitmap?>
    ) {
        val gpuBitmapRef = java.util.concurrent.atomic.AtomicReference<Bitmap?>(null)
        val gpuLatch = java.util.concurrent.CountDownLatch(1)
        val targetW = if (session.resolutionWidth > 0) session.resolutionWidth else frame.width
        val targetH = if (session.resolutionHeight > 0) session.resolutionHeight else frame.height
        Thread {
            try {
                previewRenderer.submitRawCaptureFrame(
                    frame.buffer, frame.width, frame.height, frame.width,
                    targetW, targetH, session, gpuBitmapRef, gpuLatch
                )
                val gpuReady = gpuLatch.await(8000, TimeUnit.MILLISECONDS)
                val bitmap = if (gpuReady) gpuBitmapRef.get() else null
                if (bitmap == null) {
                    Log.e(TAG, "RAW capture render failed or timed out (ready=$gpuReady)")
                    mainHandler.post {
                        Toast.makeText(this, "RAW capture failed: render timeout", Toast.LENGTH_SHORT).show()
                    }
                    return@Thread
                }
                val (jpegData, metadata) = encodeCaptureBitmap(bitmap, session)
                bitmap.recycle()

                val thumbnailReady = thumbnailLatch.await(2000, TimeUnit.MILLISECONDS)
                val thumbnailBitmap = if (thumbnailReady) thumbnailRef.get() else null
                val thumbnailJpeg = if (thumbnailBitmap != null) {
                    ExifWriter.generateThumbnailJpeg(thumbnailBitmap, 0).also {
                        thumbnailBitmap.recycle()
                    }
                } else null

                saveCaptureJpeg(jpegData, thumbnailJpeg, session, metadata)
            } catch (e: Exception) {
                Log.e(TAG, "RAW capture processing failed", e)
                mainHandler.post {
                    Toast.makeText(this, "RAW capture failed: ${e.message}", Toast.LENGTH_SHORT).show()
                }
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
        val focalLength35mm: Int,
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
        val agxVibrance: Float,
        val isFrontCamera: Boolean = false
    ) {
        override fun equals(other: Any?) = this === other
        override fun hashCode() = System.identityHashCode(this)
    }

    private fun setupManualControls() {
        amToggleButton = findViewById(R.id.am_toggle_button)
        isoOverlay = findViewById(R.id.iso_overlay)
        isoPopup = findViewById(R.id.iso_popup)
        isoRoller = findViewById(R.id.iso_roller)
        shutterOverlay = findViewById(R.id.shutter_overlay)
        shutterPopup = findViewById(R.id.shutter_popup)
        shutterRoller = findViewById(R.id.shutter_roller)
        evPpOverlay = findViewById(R.id.ev_pp_overlay)
        evPpPopup = findViewById(R.id.ev_pp_popup)
        evPpRoller = findViewById(R.id.ev_pp_roller)

        focusControls = findViewById(R.id.focus_controls)
        focusModeButton = findViewById(R.id.focus_mode_button)
        focusModeCircle = findViewById(R.id.focus_mode_circle)
        focusPopup = findViewById(R.id.focus_popup)
        focusRoller = findViewById(R.id.focus_roller)

        amToggleButton.setOnClickListener {
            isManualMode = !isManualMode
            updateManualModeUI(isManualMode)
        }

        isoOverlay.setOnClickListener {
            val opening = !isoPopupShowing
            togglePopup(isoPopup, isoOverlay, isoPopupShowing) { isoPopupShowing = it }
            if (opening) isoRoller.recenter()
        }
        shutterOverlay.setOnClickListener {
            val opening = !shutterPopupShowing
            togglePopup(shutterPopup, shutterOverlay, shutterPopupShowing) { shutterPopupShowing = it }
            if (opening) shutterRoller.recenter()
        }
        evPpOverlay.setOnClickListener {
            val opening = !evPpPopupShowing
            togglePopup(evPpPopup, evPpOverlay, evPpPopupShowing) { evPpPopupShowing = it }
            if (opening) evPpRoller.recenter()
        }
        // Double-tap on EV overlay to reset to default (+1.5)
        var lastEvPpOverlayTapTime = 0L
        evPpOverlay.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                val now = System.currentTimeMillis()
                if (now - lastEvPpOverlayTapTime < 300) {
                    evPpRoller.setIndex(EV_PP_DEFAULT_INDEX)
                    setPostProcessingEv(EV_PP_DEFAULT_INDEX)
                }
                lastEvPpOverlayTapTime = now
            }
            false
        }

        // Post-processing EV: ±10 EV in 0.5 EV steps (sensitive scrolling for big
        // adjustments), default +1.5. Configure once; never re-apply on session reopens.
        evPpRoller.maxIndex = EV_PP_MAX_INDEX
        evPpRoller.resetIndex = EV_PP_DEFAULT_INDEX
        evPpRoller.spacingPx = 14f * resources.displayMetrics.density
        evPpRoller.hapticEnabled = false
        if (!evPpRollerConfigured) {
            evPpRoller.setIndex(EV_PP_DEFAULT_INDEX)
            evPpRollerConfigured = true
        }
        evPpRoller.labelFormatter = { i -> String.format("%+.1f", (i - EV_PP_MID_INDEX) * 0.5f) }
        evPpRoller.onIndexChange = { setPostProcessingEv(it) }

        // Double-tap to reset to auto values
        isoOverlay.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                val now = System.currentTimeMillis()
                if (now - lastIsoTapTime < 300) {
                    isoRoller.setIndex(isoRoller.maxIndex / 2)
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
                    shutterRoller.setIndex(shutterRoller.maxIndex / 2)
                    updateManualExposure()
                }
                lastShutterTapTime = now
            }
            false
        }

        isoRoller.labelFormatter = { i ->
            val vals = camera2Manager.availableIsoValues
            if (i in vals.indices) "${vals[i]}" else ""
        }
        isoRoller.onIndexChange = { updateManualExposure() }
        shutterRoller.labelFormatter = { i ->
            val vals = camera2Manager.availableShutterSpeedsNs
            if (i in vals.indices) formatShutterSpeed(vals[i]) else ""
        }
        shutterRoller.onIndexChange = { updateManualExposure() }

        // Manual-focus (AF/MF) panel. Tapping the AF/MF button only opens/closes the
        // panel (circular A switch above + focus roller popup above that); it never
        // changes the focus mode itself. The circular A is the switch: tapping it
        // toggles auto (A, roller disabled) vs manual (M, dimmed A, roller enabled)
        // and the button text reads AF/MF accordingly. Closing the panel keeps the
        // current mode (e.g. MF stays manual focus).
        focusModeButton.setOnClickListener {
            if (!cameraReady) return@setOnClickListener
            if (focusPanelOpen) {
                focusPanelOpen = false
            } else {
                val minFocus = camera2Manager.minFocusDistance
                if (minFocus <= 0f) {
                    showLensWarning("Manual focus not supported on this lens")
                    return@setOnClickListener
                }
                // A locked AE/AF hold would fight the manual-focus override; release
                // the lock so MF owns the lens and auto focus resumes on exit.
                if (autofocusController.isLocked) autofocusController.unlock()
                val startDistance = (camera2Manager.lastAutoFocusDistanceDiopters ?: 0f)
                    .coerceIn(0f, minFocus)
                focusRoller.setIndex(distanceToRollerIndex(startDistance))
                focusPanelOpen = true
            }
            updateManualFocusUI()
        }

        // Inside the panel: switch between auto focus (A, full opacity, roller
        // disabled) and manual focus (M, dimmed A, roller enabled).
        focusModeCircle.setOnClickListener {
            if (!cameraReady || !focusPanelOpen) return@setOnClickListener
            isManualFocus = !isManualFocus
            if (isManualFocus) {
                // Enter manual focus at the last auto focus value: re-seed the roller
                // from the most recent auto focus distance before holding the lens.
                val minFocus = camera2Manager.minFocusDistance
                if (minFocus > 0f) {
                    val lastDistance = (camera2Manager.lastAutoFocusDistanceDiopters ?: 0f)
                        .coerceIn(0f, minFocus)
                    focusRoller.setIndex(distanceToRollerIndex(lastDistance))
                }
                camera2Manager.setManualFocus(focusRollerDistance())
            } else {
                camera2Manager.resetAutoFocus()
            }
            updateManualFocusUI()
        }

        focusRoller.spacingPx = 14f * resources.displayMetrics.density
        focusRoller.sensitivity = 7f
        focusRoller.hapticEnabled = false
        focusRoller.isHapticFeedbackEnabled = false
        focusRoller.maxIndex = FOCUS_ROLLER_MAX_INDEX
        // Double-tap the roller to reset to the middle of the focus range.
        focusRoller.resetIndex = FOCUS_ROLLER_MAX_INDEX / 2
        if (!focusRollerConfigured) {
            focusRoller.setIndex(FOCUS_ROLLER_MAX_INDEX / 2)
            focusRollerConfigured = true
        }
        focusRoller.labelFormatter = { i -> focusLabelForIndex(i) }
        focusRoller.onIndexChange = { applyManualFocusRoller(it) }

        updateManualModeUI()
        updateManualFocusUI()
        positionFocusControls()
    }

    private fun updateManualModeUI(syncToAuto: Boolean = false) {
        amToggleButton.text = if (isManualMode) "ME" else "AE"
        isoRoller.isEnabled = isManualMode
        shutterRoller.isEnabled = isManualMode
        isoRoller.alpha = if (isManualMode) 1.0f else 0.4f
        shutterRoller.alpha = if (isManualMode) 1.0f else 0.4f
        if (isManualMode) {
            // Only snap sliders to last auto values when the user first enters manual mode;
            // on session reopens (RAW/YUV toggle, resolution change, resume) keep the
            // user's manual settings.
            if (syncToAuto) {
                syncSlidersToAutoValues()
            }
            val iso = isoFromIndex(isoRoller.index)
            val expNs = shutterNsFromIndex(shutterRoller.index)
            camera2Manager.setManualExposure(iso, expNs)
        } else {
            camera2Manager.setAutoExposure()
        }
        dismissAllPopups()
        // Hide EV slider + AE indicator in manual mode (focus indicator only)
        if (isManualMode) {
            hideEvSlider()
            hideAeIndicator()
        }
    }

    private fun updateManualFocusUI() {
        focusModeButton.text = if (isManualFocus) "MF" else "AF"
        focusModeCircle.alpha = if (isManualFocus) 0.4f else 1.0f
        focusRoller.isEnabled = isManualFocus
        focusRoller.alpha = if (isManualFocus) 1.0f else 0.4f
        focusModeCircle.visibility = if (focusPanelOpen) View.VISIBLE else View.GONE
        focusPopup.visibility = if (focusPanelOpen) View.VISIBLE else View.GONE
        // Green focus-peak overlay: on while the focus roller is visible and editable
        // (panel open + manual focus), so the in-focus edges track the scroller.
        previewRenderer.focusPeakEnabled = focusPanelOpen && isManualFocus
        previewRenderer.requestRender()
        positionFocusControls()
        // In MF the focus indicator is hidden; the AE indicator stays so exposure
        // metering readout keeps working.
        if (isManualFocus) {
            focusIndicatorHandler.removeCallbacks(focusIndicatorHideRunnable)
            focusIndicator.animate().cancel()
            focusIndicator.visibility = View.GONE
            focusIndicator.alpha = 1f
            aeAfLockButton.animate().cancel()
            aeAfLockButton.visibility = View.GONE
            aeAfLockButton.alpha = 1f
        }
    }

    /** Open the focus roller popup above the AF/MF button (same geometry as ISO). */
    /**
     * Mirror the AF/MF button across the shutter center from the EV PP overlay, so
     * the focus button sits at the same distance from the shutter as the EV PP
     * button, and (when the panel is open) stack the circular A switch and the
     * focus roller popup above it. Runs after layout since the left-side overlays'
     * widths (ISO/shutter readouts) move the EV PP center.
     */
    private fun positionFocusControls() {
        focusControls.post {
            val row = focusControls.parent as? View ?: return@post
            if (row.width <= 0 || evPpOverlay.width <= 0) return@post
            val rowCenter = row.width / 2f
            val evCenter = evPpOverlay.x + evPpOverlay.width / 2f
            val mirrorCenter = 2f * rowCenter - evCenter
            focusControls.translationX = mirrorCenter - focusControls.width / 2f - focusControls.left
            if (!focusPanelOpen) return@post
            if (focusModeCircle.width <= 0 || focusPopup.width <= 0) {
                focusControls.post { positionFocusControls() }
                return@post
            }
            val root = focusModeCircle.parent as? View ?: return@post
            val rowLoc = IntArray(2)
            row.getLocationOnScreen(rowLoc)
            val rootLoc = IntArray(2)
            root.getLocationOnScreen(rootLoc)
            val density = focusControls.resources.displayMetrics.density
            val gapPx = 4 * density
            val lineBottomOffsetPx = 180 * density
            // Button center/top in row coordinates (translationX already applied).
            val btnCenterRow = focusControls.left + focusControls.width / 2f + focusControls.translationX
            val btnTopRow = focusControls.top.toFloat() + focusControls.translationY
            val btnCenterX = rowLoc[0] + btnCenterRow - rootLoc[0]
            val btnTopY = rowLoc[1] + btnTopRow - rootLoc[1]
            val circleW = focusModeCircle.width.toFloat()
            val circleH = focusModeCircle.height.toFloat()
            // Circular A switch centered over the button, just above its top.
            focusModeCircle.x = btnCenterX - circleW / 2f
            focusModeCircle.y = btnTopY - circleH - gapPx
            // Roller popup centered over the button, with the roller (bottom 180dp of
            // the strip) ending just above the circle.
            focusPopup.x = btnCenterX - focusPopup.width / 2f
            focusPopup.y = btnTopY - circleH - gapPx - lineBottomOffsetPx - gapPx
        }
    }

    private fun distanceToRollerIndex(distance: Float): Int {
        val minFocus = camera2Manager.minFocusDistance
        if (minFocus <= 0f) return 0
        return Math.round(distance / minFocus * FOCUS_ROLLER_MAX_INDEX)
            .coerceIn(0, FOCUS_ROLLER_MAX_INDEX)
    }

    private fun focusRollerDistance(): Float {
        val minFocus = camera2Manager.minFocusDistance
        if (minFocus <= 0f) return 0f
        return focusRoller.index.toFloat() / FOCUS_ROLLER_MAX_INDEX * minFocus
    }

    private fun focusLabelForIndex(index: Int): String {
        val minFocus = camera2Manager.minFocusDistance
        if (minFocus <= 0f) return ""
        return String.format("%.1f", index.toFloat() / FOCUS_ROLLER_MAX_INDEX * minFocus)
    }

    private fun applyManualFocusRoller(index: Int) {
        if (!isManualFocus) return
        camera2Manager.setManualFocus(focusRollerDistance())
    }

    private fun syncSlidersToAutoValues() {
        val isoVals = camera2Manager.availableIsoValues
        val shutterVals = camera2Manager.availableShutterSpeedsNs
        if (isoVals.isEmpty() || shutterVals.isEmpty()) return

        val targetIso = lastAutoIso
        val targetShutterNs = lastAutoShutterNs

        val isoIdx = isoVals.indices.minByOrNull { i -> Math.abs(isoVals[i] - targetIso) } ?: 0
        val shutterIdx = shutterVals.indices.minByOrNull { i -> Math.abs(shutterVals[i] - targetShutterNs) } ?: 0

        isoRoller.setIndex(isoIdx)
        shutterRoller.setIndex(shutterIdx)
    }

    private fun updateAutoExposureReadout(iso: Int, shutterNs: Long) {
        lastAutoIso = iso
        lastAutoShutterNs = shutterNs
        if (!isManualMode) {
            isoOverlay.text = "ISO $iso"
            shutterOverlay.text = formatShutterSpeed(shutterNs)
        }
        positionFocusControls()
    }

    private fun updateManualExposure() {
        val iso = isoFromIndex(isoRoller.index)
        val expNs = shutterNsFromIndex(shutterRoller.index)
        isoOverlay.text = "ISO $iso"
        shutterOverlay.text = formatShutterSpeed(expNs)
        camera2Manager.setManualExposure(iso, expNs)
        positionFocusControls()
    }

    private fun isoFromIndex(index: Int): Int {
        val vals = camera2Manager.availableIsoValues
        if (vals.isEmpty()) return 400
        return vals[index.coerceIn(0, vals.size - 1)]
    }

    private fun shutterNsFromIndex(index: Int): Long {
        val vals = camera2Manager.availableShutterSpeedsNs
        if (vals.isEmpty()) return 33_333_333L
        return vals[index.coerceIn(0, vals.size - 1)]
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
                val parentLoc = IntArray(2)
                (popup.parent as View).getLocationOnScreen(parentLoc)
                val popupW = popup.width
                val popupH = popup.height
                val density = popup.resources.displayMetrics.density
                val lineBottomOffsetPx = 180 * density
                val gapPx = 4 * density
                popup.x = (anchorLoc[0] - parentLoc[0]).toFloat() + anchor.width / 2f - popupW / 2f
                popup.y = (anchorLoc[1] - parentLoc[1]).toFloat() - lineBottomOffsetPx - gapPx
            }
            onChange(true)
        }
    }

    private fun dismissAllPopups() {
        isoPopup.visibility = View.GONE
        shutterPopup.visibility = View.GONE
        evPpPopup.visibility = View.GONE
        isoPopupShowing = false
        shutterPopupShowing = false
        evPpPopupShowing = false
    }

    private fun hideEvSlider() {
        evSliderContainer?.visibility = View.GONE
    }

    private fun updateManualControlRanges() {
        val chars = lensManager.activeLens?.let { lensManager.getCharacteristicsForLens(it) }
        chars?.let { camera2Manager.updateManualControlRanges(it) }
        syncSeekBarMax()
    }

    private fun syncSeekBarMax() {
        val isoMax = (camera2Manager.availableIsoValues.size - 1).coerceAtLeast(0)
        val shutterMax = (camera2Manager.availableShutterSpeedsNs.size - 1).coerceAtLeast(0)
        isoRoller.maxIndex = isoMax
        shutterRoller.maxIndex = shutterMax
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

    private fun setPostProcessingEv(index: Int) {
        postProcessingEv = (index - EV_PP_MID_INDEX) * 0.5f
        previewRenderer.exposureEv = postProcessingEv
        evPpOverlay.text = String.format("EV %+.1f", postProcessingEv)
        previewRenderer.requestRender()
        positionFocusControls()
    }

    // Permanent red warning: RAW-unsupported devices already get one; RAW-capable
    // devices running YUV instead of the RAW sensor stream also get a red "debug only"
    // warning so it is never silently confused with a production YUV path.
    private fun updateRawModeWarning() {
        when {
            !developerSwitch.rawSensorAvailable -> {
                rawUnsupportedWarning.text = "RAW sensor stream not supported by this device - YUV fallback mode"
                rawUnsupportedWarning.setTextColor(0xFFFF4444.toInt())
                rawUnsupportedWarning.visibility = View.VISIBLE
            }
            !developerSwitch.useRawSensor -> {
                rawUnsupportedWarning.text = "Currently DEBUG ONLY YUV MODE - please use RAW_SENSOR stream"
                rawUnsupportedWarning.setTextColor(0xFFFF4444.toInt())
                rawUnsupportedWarning.visibility = View.VISIBLE
            }
            else -> rawUnsupportedWarning.visibility = View.GONE
        }
    }

    private fun getMaxPreviewDimensions(): Pair<Int, Int> {
        val display = windowManager.defaultDisplay
        val metrics = android.util.DisplayMetrics()
        display.getRealMetrics(metrics)
        val screenWidth = metrics.widthPixels
        val screenHeight = metrics.heightPixels
        val maxDim = previewResCapMaxDim
        return Pair(maxDim, maxDim * screenHeight / screenWidth)
    }

    companion object {
        private const val TAG = "MainActivity"
        // Rec.709 Y row of RGB->XYZ(D65); fallback camera-native luminance
        // coefficients when no native->XYZ map is available.
        private val DEFAULT_LUMA_COEFFS = floatArrayOf(0.2126f, 0.7152f, 0.0722f)
        private const val REQUEST_CAMERA = 100
        private const val FOCUS_ROLLER_MAX_INDEX = 100
        private const val PREF_PREVIEW_RES_CAP = "preview_res_cap"
        private const val PREF_FOCUS_TIMEOUT = "focus_indicator_timeout"
        private const val PREF_STARTUP_PRESET = "startup_preset"
        // White balance is a user preference, not a per-lens one: it persists
        // across lens switches and app restarts (global, like flash mode).
        private const val PREF_WB_MODE = "wb_mode_global"
        private const val PREF_WB_KELVIN = "wb_kelvin_global"
        private const val PREF_WB_TINT = "wb_tint_global"
        private const val RAW_BUFFER_POOL = 3
        // Leading factor of the demosaic clipping-neutralization exponent.
        private const val PREF_CLIP_ATTEN = "clip_atten_factor"
        // Lens shading correction strength (0.0..1.0) of the estimated map.
        private const val PREF_LS_STRENGTH = "lens_shading_strength"
        // Lens shading source policy, persisted so the user's explicit choice is
        // not overridden by the device type on the next launch.
        private const val PREF_LS_SOURCE_MODE = "lens_shading_source_mode"
        // Whether this device ever produced a usable (non-identity) HAL map.
        private const val PREF_LS_HAL_MAP = "lens_shading_hal_map"
        // After a RAW session opens with no HAL map yet, wait this long before
        // declaring the device incapable of a usable lens shading map.
        private const val HAL_MAP_GRACE_MS = 5_000L
        // Post-processing EV roller: 0.5 EV per step over the ±10 EV range.
        private const val EV_PP_MAX_INDEX = 40
        private const val EV_PP_MID_INDEX = 20
        private const val EV_PP_DEFAULT_INDEX = 23
    }
}
