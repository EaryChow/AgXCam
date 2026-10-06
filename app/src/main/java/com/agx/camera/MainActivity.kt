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
import com.agx.camera.thermal.CriticalBlinkController
import com.agx.camera.thermal.ThermalManager
import com.agx.camera.ui.ScrollingIndexBar
import com.agx.camera.CrashLogger
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
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
    // Limited auto white balance: gate AUTO's correction by how much
    // adaptation the estimated light earns. The scene luminance estimate and
    // the low-passed degree of adaptation feed it. Both are written on the
    // raw thread and reset from the preview pipeline, like the estimator's
    // smoothed gains.
    @Volatile
    private var limitedAutoWb = false
    @Volatile
    private var smoothedAdaptation = 1f
    @Volatile
    private var lastSceneGreenLevel = -1f
    @Volatile
    private var aeConvergedOnce = false
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
    // Daylight anchor sampler: reduces the HAL's fixed-mode daylight gains to
    // the diagonal bridge RawColorProfile needs between its reported color
    // matrices and the live RAW gain space. Persisted per lens like the lens
    // shading map; the correction stays identity until a sample arrives.
    private lateinit var daylightAnchorEstimator: DaylightAnchorEstimator
    private var daylightAnchorGains: FloatArray? = null
    private var daylightAnchorApplied: FloatArray? = null
    private var daylightAnchorPersisted = false
    private var daylightAnchorLastState = DaylightAnchorEstimator.State.IDLE
    private var daylightBootstrapActive = false
    private var daylightBootstrapFrames = 0
    private var daylightAnchorNoticeShown = false
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
    private lateinit var daylightAnchorStatusPill: TextView
    private lateinit var daylightAnchorStartBtn: TextView
    private lateinit var daylightAnchorResetBtn: TextView
    private lateinit var limitedAwbCheckBox: CheckBox

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
    private lateinit var criticalThermalOverlay: TextView
    private lateinit var thermalProtectionSwitch: android.widget.Switch
    // Measurement switches. All default off; nothing here touches the pipeline
    // until it is explicitly armed, and the off state is bit-identical.
    @Volatile private var measurementEnabled = false
    private lateinit var criticalBlinkWarning: TextView
    private lateinit var criticalBlinkController: CriticalBlinkController
    @Volatile private var lastThermalTempC = 0f
    @Volatile private var lastThermalState = ThermalManager.State.NORMAL
    @Volatile private var thermalProtectionEnabled = true
    // True while the confirm-disable dialog is alive. During that window the
    // switch is owned by the dialog: extra taps are consumed (and the switch
    // restored to the still-applied state) instead of applying anything, so the
    // tracked state and the switch position can never drift apart.
    @Volatile private var thermalProtectionDialogShowing = false

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
    // Multi-stage denoise strengths (0..1, 0 = bypass): S1 DPC, S3 RAW
    // green-guided GF, S5 output-domain SWGF. Persisted in agxcam_settings.
    private var dpcStrength = 0f
    private var rawNrStrength = 0f
    private var outNrStrength = 0f
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

    // Multi-stage denoise: independent S1/S3/S5 sliders.
    private lateinit var s1Label: TextView; private lateinit var s1Slider: SeekBar
    private lateinit var s3Label: TextView; private lateinit var s3Slider: SeekBar
    private lateinit var s5Label: TextView; private lateinit var s5Slider: SeekBar

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
        daylightAnchorStatusPill = findViewById(R.id.daylight_anchor_status_pill)
        daylightAnchorStartBtn = findViewById(R.id.daylight_anchor_start_btn)
        daylightAnchorResetBtn = findViewById(R.id.daylight_anchor_reset_btn)
        limitedAwbCheckBox = findViewById(R.id.limited_awb_checkbox)
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

        s1Label = findViewById(R.id.s1_label); s1Slider = findViewById(R.id.s1_slider)
        s3Label = findViewById(R.id.s3_label); s3Slider = findViewById(R.id.s3_slider)
        s5Label = findViewById(R.id.s5_label); s5Slider = findViewById(R.id.s5_slider)

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
        criticalThermalOverlay = findViewById(R.id.critical_thermal_overlay)
        thermalProtectionSwitch = findViewById(R.id.thermal_protection_switch)
        criticalBlinkWarning = findViewById(R.id.critical_blink_warning)
        criticalBlinkController = CriticalBlinkController(mainHandler) { visible ->
            criticalBlinkWarning.visibility = if (visible) View.VISIBLE else View.GONE
        }

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

        thermalManager = ThermalManager(this).also { tm ->
            tm.onStateChanged = { state -> mainHandler.post { onThermalStateChanged(state) } }
            tm.onTemperatureUpdate = { displayC, _ -> mainHandler.post { onThermalTempUpdate(displayC) } }
            tm.onTorchForcedOff = { reason -> mainHandler.post { onThermalTorchForcedOff(reason) } }
            // Provider reflection is read at actuator time (never cached), so a
            // toggle takes effect immediately. Default true covers the brief
            // window before previewResPrefs is read below.
            tm.thermalProtectionEnabledProvider = { thermalProtectionEnabled }
            tm.start()
        }
        shutterController = ShutterController()
        autofocusController = AutofocusController(camera2Manager, mainHandler)
        mediaStoreSaver = MediaStoreSaver(this)

        // Compute max preview dimensions based on screen resolution
        previewResPrefs = getSharedPreferences("agxcam_settings", Context.MODE_PRIVATE)
        previewResCapMaxDim = previewResPrefs.getInt(PREF_PREVIEW_RES_CAP, 1280)
        focusIndicatorTimeoutMs = previewResPrefs.getLong(PREF_FOCUS_TIMEOUT, 0L)
        maxPreviewDimensions = getMaxPreviewDimensions()
        thermalProtectionEnabled = previewResPrefs.getBoolean(PREF_THERMAL_PROTECTION_ENABLED, true)
        limitedAutoWb = previewResPrefs.getBoolean(PREF_LIMITED_WB, true)
        clipAttenFactor = previewResPrefs.getFloat(PREF_CLIP_ATTEN, 0.1f).coerceIn(0f, 1f)
        dpcStrength = previewResPrefs.getFloat(PREF_S1_DPC, 0f).coerceIn(0f, 1f)
        rawNrStrength = previewResPrefs.getFloat(PREF_S3_RAW, 0f).coerceIn(0f, 1f)
        outNrStrength = previewResPrefs.getFloat(PREF_S5_OUT, 0f).coerceIn(0f, 1f)

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
            // The one export action. It writes the log with the measurement
            // report folded in as a section, and carries the one-shot raw frame
            // capture alongside it when one was requested and has landed.
            CrashLogger.log(TAG, "debug log export requested")
            // The raw frame is written first so the measurement bundle can name
            // the file it landed in. The renderer keeps the grab's own summary,
            // so consuming the bytes here does not blank the section.
            val attached = ::previewRenderer.isInitialized && writePendingFrameGrab()
            saveMeasurementBundle()
            CrashLogger.saveDebugLogToDownloads(this)

            // Clear the one-shot request either way, so the next export does not
            // silently carry a stale frame the user no longer means to send.
            findViewById<android.widget.CheckBox>(R.id.measurement_raw_frame_check)?.isChecked = false

            Toast.makeText(
                this,
                if (attached) "Debug log and raw frame saved to Downloads" else "Debug log saved to Downloads",
                Toast.LENGTH_SHORT
            ).show()
        }
        setupMeasurementSwitches()

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
            dpcStrength = this@MainActivity.dpcStrength
            rawNrStrength = this@MainActivity.rawNrStrength
            outNrStrength = this@MainActivity.outNrStrength
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
        // The switches are read before the renderer exists, so the persisted
        // flags are applied here once there is something to apply them to.
        applyMeasurementFlags()

        textureView.surfaceTextureListener = previewRenderer

        setupUI()
        val startupPreset = previewResPrefs.getString(PREF_STARTUP_PRESET, null)
        if (startupPreset != null && presetManager.getNames().contains(startupPreset)) {
            loadPreset(startupPreset)
            selectPreset(startupPreset)
        } else {
            loadPreset(PresetManager.PRESET_DEFAULT)
        }
        thermalProtectionSwitch.isChecked = thermalProtectionEnabled
        // Cold start with protection already off and the battery already warm:
        // derive the tier + render the corner strip immediately instead of
        // waiting for the next temperature callback (which may be seconds away).
        if (!thermalProtectionEnabled) {
            thermalManager.forceEvaluation()
            handleThermalTier(thermalManager.currentState)
        }
        checkPermissions()
    }

    private fun setupUI() {
        loadWbModePrefs()
        flashButton.setOnClickListener {
            val thermalState = thermalManager.currentState
            if (thermalProtectionEnabled &&
                (thermalState == ThermalManager.State.HOT || thermalState == ThermalManager.State.CRITICAL)) {
                showWarningForDuration("Flash disabled \u2014 battery too hot", 10_000)
                return@setOnClickListener
            }
            currentFlashMode = currentFlashMode.cycle()
            updateFlashUI()
            showModeLabel(flashModeDisplayName(currentFlashMode))
            thermalManager.isTorchActive = (currentFlashMode == FlashMode.TORCH)
            if (cameraReady) camera2Manager.setFlashMode(currentFlashMode)
        }

        // Thermal protection toggle. Click-driven (not a checked-change
        // listener) so programmatic setChecked leaves it untouched; disabling
        // asks for confirmation every single time.
        thermalProtectionSwitch.setOnClickListener {
            onThermalProtectionToggleRequested(thermalProtectionSwitch.isChecked)
        }

        limitedAwbCheckBox.isChecked = limitedAutoWb
        limitedAwbCheckBox.setOnClickListener {
            limitedAutoWb = limitedAwbCheckBox.isChecked
            previewResPrefs.edit().putBoolean(PREF_LIMITED_WB, limitedAutoWb).apply()
            CrashLogger.log(TAG, "limitedAWB: enabled=$limitedAutoWb")
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
            if (thermalProtectionEnabled && thermalManager.isCaptureBlocked) {
                // Critical: the RAW stream is off, so the capture is a black frame.
                captureBlackStill()
                return@setOnClickListener
            }
            shutterController.onCaptureSubmitted()
            isCapturing = true

            val session = buildCaptureSession()

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

    private fun kelvinSliderIndex(kelvin: Float): Int =
        KelvinState.kelvinToSliderIndex(kelvin).coerceIn(0, kelvinSlider.max)

    private fun tintSliderIndex(tint: Float): Int =
        KelvinState.tintToSliderIndex(tint).coerceIn(0, tintSlider.max)

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
        daylightAnchorEstimator = DaylightAnchorEstimator()

        daylightAnchorStartBtn.setOnClickListener {
            if (!cameraReady) {
                Toast.makeText(this, "Open the camera first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (rawColorProfile == null) {
                Toast.makeText(this, "No sensor color profile on this lens", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (!camera2Manager.supportsDaylightMode) {
                Toast.makeText(this, "This device has no fixed daylight mode to sample", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            restartDaylightAnchorSampling()
        }

        daylightAnchorResetBtn.setOnClickListener {
            resetDaylightAnchor()
        }

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

        val rotRange = 0.26179938779f // +/-15 deg = +/-pi/12 rad, matches Blender reference
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

        s1Slider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            dpcStrength = v / 100f
            s1Label.text = "S1 DPC  $v"
            previewRenderer.dpcStrength = dpcStrength
            previewResPrefs.edit().putFloat(PREF_S1_DPC, dpcStrength).apply()
        })
        setupSliderDoubleClickReset(s1Slider, 0) {
            dpcStrength = 0f
            s1Label.text = "S1 DPC  0"
            previewRenderer.dpcStrength = 0f
            previewResPrefs.edit().putFloat(PREF_S1_DPC, 0f).apply()
        }

        s3Slider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            rawNrStrength = v / 100f
            s3Label.text = "S3 RAW  $v"
            previewRenderer.rawNrStrength = rawNrStrength
            previewResPrefs.edit().putFloat(PREF_S3_RAW, rawNrStrength).apply()
        })
        setupSliderDoubleClickReset(s3Slider, 0) {
            rawNrStrength = 0f
            s3Label.text = "S3 RAW  0"
            previewRenderer.rawNrStrength = 0f
            previewResPrefs.edit().putFloat(PREF_S3_RAW, 0f).apply()
        }

        s5Slider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            outNrStrength = v / 100f
            s5Label.text = "S5 OUT  $v"
            previewRenderer.outNrStrength = outNrStrength
            previewResPrefs.edit().putFloat(PREF_S5_OUT, outNrStrength).apply()
        })
        setupSliderDoubleClickReset(s5Slider, 0) {
            outNrStrength = 0f
            s5Label.text = "S5 OUT  0"
            previewRenderer.outNrStrength = 0f
            previewResPrefs.edit().putFloat(PREF_S5_OUT, 0f).apply()
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
            kelvinState = kelvinState.copy(
                kelvin = KelvinState.sliderIndexToKelvin(v)
            )
            kelvinLabel.text = String.format("Kelvin  %.0fK", kelvinState.kelvin)
            if (currentWbMode == WhiteBalanceMode.KELVIN) {
                uploadAgxUniforms()
                persistWbMode()
            }
        })
        setupSliderDoubleClickReset(kelvinSlider, kelvinSliderIndex(DEFAULT_WB_KELVIN)) {
            kelvinState = kelvinState.copy(kelvin = DEFAULT_WB_KELVIN)
            kelvinLabel.text = String.format("Kelvin  %.0fK", kelvinState.kelvin)
            if (currentWbMode == WhiteBalanceMode.KELVIN) {
                uploadAgxUniforms()
                persistWbMode()
            }
        }
        tintSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            kelvinState = kelvinState.copy(tint = KelvinState.sliderIndexToTint(v))
            tintLabel.text = String.format("Tint  %.0f", kelvinState.tint)
            if (currentWbMode == WhiteBalanceMode.KELVIN) {
                uploadAgxUniforms()
                persistWbMode()
            }
        })
        setupSliderDoubleClickReset(tintSlider, tintSliderIndex(DEFAULT_WB_TINT)) {
            kelvinState = kelvinState.copy(tint = DEFAULT_WB_TINT)
            tintLabel.text = String.format("Tint  %.0f", kelvinState.tint)
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

        s1Slider.progress = (dpcStrength * 100).toInt().coerceIn(0, 100)
        s1Label.text = "S1 DPC  ${s1Slider.progress}"
        s3Slider.progress = (rawNrStrength * 100).toInt().coerceIn(0, 100)
        s3Label.text = "S3 RAW  ${s3Slider.progress}"
        s5Slider.progress = (outNrStrength * 100).toInt().coerceIn(0, 100)
        s5Label.text = "S5 OUT  ${s5Slider.progress}"

        clipAttenSlider.progress = (clipAttenFactor * 100).toInt().coerceIn(0, 100)
        clipAttenLabel.text = String.format("Neutralize  %.2f", clipAttenFactor)

        kelvinSlider.progress = kelvinSliderIndex(kelvinState.kelvin)
        kelvinLabel.text = String.format("Kelvin  %.0fK", kelvinState.kelvin)
        tintSlider.progress = tintSliderIndex(kelvinState.tint)
        tintLabel.text = String.format("Tint  %.0f", kelvinState.tint)

        jpegSlider.progress = photoOutput.jpegQuality
        jpegLabel.text = String.format("JPEG Quality  %d", photoOutput.jpegQuality)

        syncWbSliders()
    }

    private fun syncInsetSliders() {
        val rotRange = 0.26179938779f // +/-15 deg = +/-pi/12 rad, matches Blender reference
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
        val rotRange = 0.26179938779f // +/-15 deg = +/-pi/12 rad, matches Blender reference
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
        showWarningForDuration(message, 3_000)
    }

    private fun showWarningForDuration(message: String, durationMs: Long) {
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

            container.postDelayed(warningDismissRunnable, durationMs)
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
            append(facing).append(" | ").append(lens.label)
            if (lens.focalLengthMm > 0f) {
                val eq35 = lens.focalLength35mmEq
                if (eq35 > 0f) {
                    append(String.format(" | %dmm", Math.round(eq35)))
                } else {
                    append(String.format(" | %.1fmm", lens.focalLengthMm))
                }
            }
            append('\n')
            append(sensor).append(" | ").append(if (lens.hasRawSensor) "RAW" else "YUV")
            append(" | ").append(hw)
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
            // Cached-state read (no evaluateState): Cheap volatile only. State
            // transitions are poll-driven (~1s), so a frame in the gaps is fine.
            if (thermalProtectionEnabled && thermalManager.currentState == ThermalManager.State.CRITICAL) return@frameHandler
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
            // Cached-state read (no evaluateState): Cheap volatile only. State
            // transitions are poll-driven (~1s), so a frame in the gaps is fine.
            if (thermalProtectionEnabled && thermalManager.currentState == ThermalManager.State.CRITICAL) return@rawHandler
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
            feedDaylightAnchorEstimator()

            val ccGains = camera2Manager.latestColorCorrectionGains
            val ccMat = camera2Manager.latestColorCorrectionMatrix
            val gainsOk = ccGains != null &&
                ccGains.size >= 4 &&
                ccGains.all { it.isFinite() && it > 0f }

            // AUTO is the app's own estimate: it works on every CFA phase separately,
            // which no HAL or profile answer can express, so it is preferred over
            // the HAL. The fixed presets leave the HAL in charge - it is already
            // pinned to the chosen illuminant - and KELVIN is a preset like the
            // rest, parameterized by the slider pair.
            val useEstimator = currentWbMode == WhiteBalanceMode.AUTO || !gainsOk
            val estimatorStep = useEstimator && wbEstimateFrame % 15 == 0
            // The estimate is only genuine once AE has converged: while AE
            // hunts, overexposed frames clip the green channel first, the
            // clipped cells are dropped, and the surviving means tilt the
            // estimate magenta. The latch sticks so later AE hunting does not
            // freeze AUTO, and a timeout keeps AUTO alive if AE never
            // converges at all. When no daylight transform exists the
            // estimator still runs: the estimate falls back to the raw
            // signal, its documented no-profile behavior.
            val aeState = camera2Manager.latestAeState
            if (aeState == CaptureResult.CONTROL_AE_STATE_CONVERGED ||
                aeState == CaptureResult.CONTROL_AE_STATE_LOCKED
            ) {
                aeConvergedOnce = true
            }
            val awbInputReady =
                aeConvergedOnce || wbEstimateFrame > AWB_INPUT_TIMEOUT_FRAMES
            if (estimatorStep && awbInputReady) {
                estimateAutoWhiteBalance(dest, rawW, rawH)
            }

            wbEstimateFrame++

            // Once the estimator has an answer it keeps the gains until the lens
            // changes; it only steps every 15th frame, so the gains outlive the
            // call that produced them.
            val estimatorOwnsGains = currentWbMode == WhiteBalanceMode.AUTO && estimatorReady

            // The one thing every mode shares: gains that take a neutral object to level
            // codes, and a matrix built from that same neutral to put the level
            // codes on D65. The matrix is the human-vision adaptation from the
            // illuminant the gains were solved against, so it has to be solved
            // from the illuminant and not held fixed - a daylight matrix against
            // non-daylight gains would render every scene neutral and throw the
            // adaptation away.
            //
            // AUTO's illuminant is the estimate, so the daylight transform is
            // only the first stage of the estimate (it removes the CFA's green bias
            // before grey world sees the frame); the daylight *matrix* it also
            // carries is not used, because the illuminant the matrix has to adapt
            // from is the estimated one, not D65.
            //
            // Always defined: every branch falls back to the seeded neutral rather than
            // to nothing, so there is a sensor neutral to read the CCT from even
            // before the estimator or the HAL has produced one.
            //
            // A preset names the light, so it takes the profile's daylight white as
            // its neutral: the gains solved from that are the sensor's own daylight
            // whitening, which is a property of the sensor and not of the frame, and
            // the illuminant the preset asserts is left entirely to the matrix built
            // below from the mode's table. KELVIN is a preset with a parameterized
            // table entry, so it takes this branch too; AUTO measures this number
            // instead of asserting it, so AUTO alone falls through.
            val presetDaylightWhite =
                if (currentWbMode != WhiteBalanceMode.AUTO) {
                    rawColorProfile?.daylightWhite()
                } else {
                    null
                }

            val neutral: FloatArray = presetDaylightWhite ?: if (useEstimator) {
                // The estimator's smoothed gains inverted: the sensor neutral the
                // profile path needs, so it whitens what the app measured rather
                // than what the HAL measured.
                estimatorNeutral ?: rawNeutralSeed
            } else if (gainsOk) {
                // No profile to name the sensor's daylight response, and the
                // estimator not standing in: the as-shot neutral is the
                // inverse of the HAL's COLOR_CORRECTION_GAINS.
                floatArrayOf(
                    1f / ccGains[0],
                    1f / ((ccGains[1] + ccGains[2]) * 0.5f),
                    1f / ccGains[3]
                )
            } else {
                rawNeutralSeed
            }

            // Every mode reduces to one scene chromaticity, and that number is the
            // only thing that differs between them: AUTO measures it, a preset reads
            // it off the mode's table, KELVIN parametrizes it. The gains whiten the
            // sensor's response to that light and the matrix adapts it onto D65, so
            // the two halves are always built from one answer and cannot disagree.
            //
            // The chromaticity follows the *mode*, not the gain source. useEstimator
            // is also true for a preset whose HAL reports no COLOR_CORRECTION_GAINS,
            // where the estimator is standing in for the gains - but the user picking
            // "tungsten" is still asserting the light is tungsten, so that preset's
            // illuminant is what the matrix adapts from, measured or not.
            //
            // AUTO measures by pushing the estimated neutral through the D65
            // camera->XYZ map. The reference matrices stay read at D65 on this path,
            // so that map is the sensor's response and not a function of the light,
            // and no temperature is involved anywhere. KELVIN parametrizes the same
            // number: the slider pair maps to a chromaticity on the Planckian locus,
            // tint walking perpendicular to it, which makes KELVIN exactly a preset
            // whose table entry is computed instead of constant.
            val sceneXy: FloatArray? = when (currentWbMode) {
                WhiteBalanceMode.AUTO -> rawColorProfile?.sceneXyForNeutral(neutral)
                WhiteBalanceMode.KELVIN -> kelvinSceneXy()
                else -> currentWbMode.sceneXy()
            }

            // Limited auto white balance: gate the scene->D65 correction by the
            // degree of adaptation the estimated light earns. Highly chromatic
            // light (party LEDs) sits far off the Planckian locus and earns
            // zero, so the scene keeps its light untouched; near-locus light
            // partially adapts. Presets and the plain mode keep full correction.
            val adaptation: Float = when {
                !limitedAutoWb -> 1f
                currentWbMode != WhiteBalanceMode.AUTO -> 1f
                sceneXy == null -> 1f
                // The locus scan and the low-pass only run on estimator steps:
                // the estimate, and the degree of adaptation derived from it,
                // cannot change faster than the estimator updates.
                !estimatorStep -> smoothedAdaptation
                else -> {
                    // First trusted estimate: adopt its degree of adaptation
                    // exactly, like the illuminant snap; the EMA tracks from
                    // there.
                    if (awbSmoother.consumeSnapAdaptation()) {
                        smoothedAdaptation = limitedAdaptationTarget(sceneXy)
                    }
                    smoothedLimitedAdaptation(sceneXy)
                }
            }

            // The white balance is ours, so the matrix is the WB-removed one and the
            // gains the transform hands back are unused. A scene chromaticity that
            // cannot be resolved is a miss, not a reason to invent a temperature:
            // the profile's own solve is tried instead.
            val profileTransform = if (sceneXy != null) {
                rawColorProfile?.neutralTransformForSceneXy(neutral, sceneXy, adaptation)
            } else {
                rawColorProfile?.neutralTransformForNeutral(neutral)
            }

            if (profileTransform != null) {
                previewRenderer.ccMatrix = profileTransform.colorMatrix.m
                previewRenderer.nativeLumaCoeffs = profileTransform.lumaCoeffs
            } else if (ccMat != null) {
                previewRenderer.ccMatrix = ccMat
                previewRenderer.nativeLumaCoeffs = luminanceFromSrgbMatrix(ccMat)
            } else {
                previewRenderer.ccMatrix = null
                previewRenderer.nativeLumaCoeffs = DEFAULT_LUMA_COEFFS.copyOf()
            }

            if (!estimatorOwnsGains) {
                if (profileTransform != null) {
                    // Profile path: split into sensor-space white-balance gains
                    // and the WB-removed native->sRGB matrix so the demosaic
                    // shader can neutralize clipped regions between the two
                    // stages. The matrix no longer folds in the WB, and its Y row
                    // provides the camera-native luminance coefficients for the
                    // neutralization.
                    previewRenderer.wbGainR = profileTransform.wbGains[0]
                    previewRenderer.wbGainG = profileTransform.wbGains[1]
                    previewRenderer.wbGainB = profileTransform.wbGains[2]
                    if (wbEstimateFrame <= 3 || wbEstimateFrame % 60 == 0) {
                        val temp = rawColorProfile?.temperatureForNeutral(neutral)
                        CrashLogger.log(
                            TAG, "dcp cc: frame=$wbEstimateFrame " +
                                "auto=${if (currentWbMode == WhiteBalanceMode.AUTO) "yes" else "no"} " +
                                "daylight=${if (rawDaylightGains != null) "yes" else "no"} " +
                                "temp=${temp?.toInt() ?: -1} " +
                                "neutral=[${neutral.joinToString { String.format("%.3f", it) }}] " +
                                "wb=[${previewRenderer.wbGainR}, ${previewRenderer.wbGainG}, " +
                                "${previewRenderer.wbGainB}] " +
                                "mat=[${profileTransform.colorMatrix.m.joinToString { String.format("%.4f", it) }}]"
                        )
                    }
                } else if (gainsOk) {
                    val gMean = (ccGains[1] + ccGains[2]) * 0.5f
                    if (gMean > 0f) {
                        previewRenderer.wbGainR = (ccGains[0] / gMean).coerceIn(0.3f, 8f)
                        previewRenderer.wbGainG = 1f
                        previewRenderer.wbGainB = (ccGains[3] / gMean).coerceIn(0.3f, 8f)
                    }
                    if (wbEstimateFrame % 60 == 0) {
                        CrashLogger.log(
                            TAG, "hal cc: frame=$wbEstimateFrame " +
                                "gainsR=${String.format("%.3f", previewRenderer.wbGainR)} " +
                                "gainsB=${String.format("%.3f", previewRenderer.wbGainB)} " +
                                "raw=[${ccGains.joinToString { String.format("%.3f", it) }}] " +
                                "mat=[${ccMat?.joinToString { String.format("%.4f", it) }}]"
                        )
                    }
                } else {
                    // No usable source at all: stay neutral.
                    previewRenderer.wbGainR = 1f
                    previewRenderer.wbGainG = 1f
                    previewRenderer.wbGainB = 1f
                }
                // The per-phase solution belongs to the estimator alone.
                previewRenderer.wbPhaseGains = null
            } else if (wbEstimateFrame % 60 == 0) {
                // The estimator wrote them; log which source owns the gains, so a
                // capture can be traced back to one or the other.
                CrashLogger.log(
                    TAG, "estimator cc: frame=$wbEstimateFrame " +
                        "wb=[${previewRenderer.wbGainR}, ${previewRenderer.wbGainG}, " +
                        "${previewRenderer.wbGainB}] " +
                        "phase=[${previewRenderer.wbPhaseGains?.joinToString { String.format("%.3f", it) }}]" +
                        (if (currentWbMode == WhiteBalanceMode.AUTO) " adapt=$adaptation" else "")
                )
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
                Log.w(TAG, "Camera disconnected - banner shown")
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
            // Fresh session for the estimator and its smoother: no estimate
            // is trusted yet, and stale smoothing state must not cross lenses.
            awbSmoother.reset()
            aeConvergedOnce = false
            rawColorProfile = profile
            // The daylight gain is a sensor property, so it is resolved once per lens
            // here rather than per frame. AUTO's estimator takes it off the raw
            // frame first, so grey world reads the illuminant rather than the
            // CFA's green bias. Only the gains are kept: the matrix from the same
            // call adapts D65 to D65, and what the renderer needs is the CAT from
            // the estimated illuminant, which is built per frame from that estimate.
            rawDaylightGains = profile.daylightTransform()?.wbGains
            if (rawDaylightGains != null) {
                CrashLogger.log(
                    TAG, "daylight cc: frame=$wbEstimateFrame " +
                        "wb=[${rawDaylightGains!!.joinToString { String.format("%.3f", it) }}]"
                )
            }
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
            // The green sites belong to this sensor, so the smoothed illuminant
            // starts over with the lens (AwbIlluminantSmoother.reset above)
            // rather than carrying the previous one's balance. The renderer
            // goes to identity with it.
            previewRenderer.wbPhaseGains = null
            estimatorReady = false
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
            initDaylightAnchorForLens(lens)
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
        previewRenderer.agxWhiteLevel = rawWhiteLevel.toFloat()
        previewRenderer.agxBlackLevel = rawBlackLevel
        previewRenderer.wbGainR = 1f
        previewRenderer.wbGainG = 1f
        previewRenderer.wbGainB = 1f
        awbSmoother.reset()
        previewRenderer.wbPhaseGains = null
        estimatorReady = false
        previewRenderer.nativeLumaCoeffs = DEFAULT_LUMA_COEFFS.copyOf()
        rawColorProfile = null
        rawDaylightGains = null
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
        if (::daylightAnchorEstimator.isInitialized) {
            stopDaylightAnchorBootstrap()
            daylightAnchorEstimator.reset()
            daylightAnchorLastState = DaylightAnchorEstimator.State.IDLE
            daylightAnchorGains = null
            daylightAnchorApplied = null
            daylightAnchorPersisted = false
            mainHandler.post { updateDaylightAnchorUI() }
        }
        lastSceneGreenLevel = -1f
        smoothedAdaptation = 1f
        aeConvergedOnce = false
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
    // The fixed daylight gain, green bias removal for the estimator. Only its
    // wbGains are used: the color matrix the same call also produces would adapt
    // D65 to D65, which is the identity and not the CAT the render needs.
    private var rawDaylightGains: FloatArray? = null
    private var rawNeutralSeed = floatArrayOf(1f, 1f, 1f)
    private var estimatorNeutral: FloatArray? = null

    // Smoothed per-CFA-phase grey world gains, owned by the dual-rate
    // smoother: only the two green sites reach the demosaic from here; red
    // and blue leave as the post-merge colorGains. The illuminant stage is
    // what gets smoothed: it is the part that varies with the scene. The
    // daylight reference is fixed per lens and is reapplied on top, so it
    // never enters the smoothing history and cannot drift.
    private val awbSmoother = AwbIlluminantSmoother()

    // Sticky: the estimator steps every 15th frame, so its gains have to outlive
    // the call that produced them. Cleared with the lens, and ignored whenever
    // the mode is not AUTO.
    private var estimatorReady = false

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
        // 16 cells is a 32-pixel step, matching the stride this walk used before
        // it started dropping clipped cells: the sample density is unchanged, only
        // the grouping is.
        val cellStride = 16
        val colorMap = previewRenderer.bayerColorMap
        val scan = GreyWorldEstimate.scan(
            buffer, w, h, previewRenderer.bayerBlackLevelPattern,
            cellStride, rawWhiteLevel
        )
        val phaseMeans = scan.means
        // Scene luminance proxy for the limited auto white balance: the green
        // level normalized to the sensor white, mapped to an adapting
        // luminance estimate by a tunable gain.
        lastSceneGreenLevel =
            ((phaseMeans[1] + phaseMeans[2]) * 0.5f / rawWhiteLevel.coerceAtLeast(1)).toFloat()
        // The sensor's daylight response comes off before the estimate, so what
        // the estimator reports is the light's departure from D65 rather than the
        // CFA's own green bias, which every D65 scene shares. Without a profile
        // there is no response to take off and this falls back to estimating
        // straight off the raw signal.
        val daylight = rawDaylightGains
        val estimate = GreyWorldEstimate.estimate(phaseMeans, colorMap, daylight)
        val logNow = wbEstimateFrame % 90 == 0 || wbEstimateFrame <= 3
        if (estimate == null) {
            if (logNow) {
                CrashLogger.log(
                    TAG, "wb estimate: skip frame=$wbEstimateFrame " +
                        "cells=${scan.cellsUsed} clipped=${scan.cellsClipped} " +
                        "means=[${phaseMeans.joinToString { String.format("%.1f", it) }}]"
                )
            }
            return
        }

        // Smoothed on the illuminant stage alone, not the product: the
        // daylight reference is fixed per lens and smoothing it would only
        // let a stale sensor response drift in on top of a scene that has
        // not changed. The dual-rate state machine (acquire, handover,
        // slow pole, fast re-acquire) lives in AwbIlluminantSmoother.
        val now = SystemClock.elapsedRealtimeNanos()
        awbSmoother.step(estimate.illuminant, now)
        // Clamped as the product, through the estimator's own combine, so the array
        // the renderer holds is the one Estimate.phaseGains would describe for these
        // two stages. Multiplying here instead would leave the product unclamped
        // while estimate() clamps it, and the two would disagree. Derived every
        // frame rather than carried as state: combine returns a fresh array, so
        // this thread can go on smoothing while the render thread holds this one.
        val phaseGains = GreyWorldEstimate.combine(awbSmoother.gains, estimate.daylight)
        previewRenderer.wbPhaseGains = phaseGains
        estimatorReady = true

        // The renderer splits this per phase on its own: the two green sites
        // before the demosaic merges them, red and blue after.
        val colorGains = GreyWorldEstimate.colorGains(phaseGains, colorMap)
        previewRenderer.wbGainR = colorGains[0]
        previewRenderer.wbGainG = colorGains[1]
        previewRenderer.wbGainB = colorGains[2]

        // The sensor response to a neutral object under the estimated illuminant, which
        // is what the profile path needs. AUTO measures the illuminant's chromaticity
        // off this directly, through the D65 camera->XYZ map; the presets instead
        // hand it to neutralTransformForNeutral to solve for a temperature, and
        // temperatureForNeutral reads the CCT off that same solve. Either way it is
        // read as a full sensor response, so this is the inverse of the whole product
        // rather than of the illuminant stage alone - the daylight stage is part of
        // what the sensor actually answered.
        //
        // Reachable outside AUTO too: a fixed preset on a HAL reporting no
        // COLOR_CORRECTION_GAINS still runs the estimator, does not own the gains,
        // and has its profile transform built from this value.
        //
        // Smoothed on its own fixed rate, not the illuminant stage's dual-rate
        // dynamics: the color gains are already smoothed, so this is a
        // derivation of a settled signal, and inheriting the slow pole would
        // leave the profile's neutral crawling for tens of seconds after any
        // change - which is a visible cast while limited AWB blends from it.
        // The first trusted estimate is adopted exactly, like the illuminant
        // snap.
        if (awbSmoother.consumeSnapNeutral()) {
            estimatorNeutral = floatArrayOf(1f / colorGains[0], 1f, 1f / colorGains[2])
        } else {
            val prevNeutral = estimatorNeutral ?: floatArrayOf(1f, 1f, 1f)
            estimatorNeutral = floatArrayOf(
                prevNeutral[0] + ESTIMATOR_NEUTRAL_SMOOTHING * (1f / colorGains[0] - prevNeutral[0]),
                1f,
                prevNeutral[2] + ESTIMATOR_NEUTRAL_SMOOTHING * (1f / colorGains[2] - prevNeutral[2])
            )
        }

        if (logNow) {
            val greens = GreyWorldEstimate.greenPhases(colorMap)
            CrashLogger.log(
                TAG, "wb estimate: frame=$wbEstimateFrame " +
                        "cells=${scan.cellsUsed} clipped=${scan.cellsClipped} " +
                        "daylight=${if (daylight != null) "yes" else "no"} " +
                        "means=[${phaseMeans.joinToString { String.format("%.1f", it) }}] " +
                        "illuminant=[${awbSmoother.gains.joinToString { String.format("%.3f", it) }}] " +
                        "phaseGains=[${phaseGains.joinToString { String.format("%.3f", it) }}] " +
                    "greenSites=[${greens[0]},${greens[1]}] " +
                    "colorGains=[${String.format("%.2f", colorGains[0])}," +
                    "${String.format("%.2f", colorGains[1])}," +
                    "${String.format("%.2f", colorGains[2])}] " +
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
            mainHandler.post { updateDaylightAnchorUI() }
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
            if (frames > 0) text += " | $frames"
            val elapsedS = lensShadingEstimator.elapsedMillis / 1000
            if (elapsedS >= 1 && lensShadingEstimator.currentState != LensShadingState.IDLE) text += " | ${elapsedS}s"
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
        lensShadingNoticeOverlay.removeCallbacks(daylightAnchorNoticeHideRunnable)
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
    // turned correction off" - never force it back on at startup.
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

    // Per-lens daylight anchor setup: restore a saved anchor, or start sampling
    // (bootstrapping the HAL into its fixed DAYLIGHT mode if the current white
    // balance mode does not already pin it there).
    private fun initDaylightAnchorForLens(lens: LensInfo) {
        if (!::daylightAnchorEstimator.isInitialized) return
        daylightAnchorGains = null
        daylightAnchorApplied = null
        daylightAnchorPersisted = false
        daylightAnchorEstimator.reset()
        daylightAnchorLastState = DaylightAnchorEstimator.State.IDLE
        val saved = loadDaylightAnchor(lens.cameraId)
        if (saved != null) {
            daylightAnchorGains = saved
            applyDaylightAnchor()
            daylightAnchorPersisted = true
            CrashLogger.log(
                TAG, "daylightAnchor: restored saved anchor for lens ${lens.cameraId} " +
                    "d=[${daylightAnchorApplied?.joinToString { String.format("%.4f", it) }}]"
            )
        } else if (rawColorProfile != null) {
            restartDaylightAnchorSampling()
        }
        mainHandler.post { updateDaylightAnchorUI() }
    }

    private fun restartDaylightAnchorSampling() {
        val lens = lensManager.activeLens ?: return
        if (!camera2Manager.supportsDaylightMode) {
            CrashLogger.log(TAG, "daylightAnchor: no fixed daylight mode on this device")
            return
        }
        daylightAnchorGains = null
        daylightAnchorApplied = null
        daylightAnchorPersisted = false
        deleteDaylightAnchorPersistence(lens.cameraId)
        daylightAnchorEstimator.start()
        // The HAL only reports daylight gains while pinned to its fixed
        // DAYLIGHT mode. If it already is, readbacks flow on their own;
        // otherwise hold the override until convergence. KELVIN resolves to
        // CLOUDY/SHADE on devices without DAYLIGHT, so only the reported HAL
        // mode is trusted here, never the app's mode.
        val pinnedToDaylight = camera2Manager.latestAwbMode ==
            android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
        daylightBootstrapFrames = 0
        if (!pinnedToDaylight) {
            daylightBootstrapActive = true
            camera2Manager.daylightCalibrationOverride = true
            maybeShowDaylightAnchorNotice()
        }
        CrashLogger.log(
            TAG, "daylightAnchor: sampling started lens=${lens.cameraId} " +
                "override=$daylightBootstrapActive"
        )
        mainHandler.post { updateDaylightAnchorUI() }
    }

    private fun resetDaylightAnchor() {
        val lens = lensManager.activeLens
        stopDaylightAnchorBootstrap()
        daylightAnchorEstimator.reset()
        daylightAnchorGains = null
        daylightAnchorApplied = null
        daylightAnchorPersisted = false
        rawColorProfile?.daylightCorrection = floatArrayOf(1f, 1f, 1f)
        val oldDaylight = rawDaylightGains
        rawDaylightGains = rawColorProfile?.daylightTransform()?.wbGains
        // The renderer holds a product of both stages; rescale the smoothed
        // stage so it stays put across the swap.
        awbSmoother.retarget(oldDaylight, rawDaylightGains, previewRenderer.bayerColorMap)
        if (lens != null) deleteDaylightAnchorPersistence(lens.cameraId)
        CrashLogger.log(TAG, "daylightAnchor: reset")
        mainHandler.post { updateDaylightAnchorUI() }
    }

    private fun stopDaylightAnchorBootstrap() {
        if (!daylightBootstrapActive) return
        daylightBootstrapActive = false
        camera2Manager.daylightCalibrationOverride = false
        CrashLogger.log(
            TAG, "daylightAnchor: bootstrap done state=${daylightAnchorEstimator.state} " +
                "frames=$daylightBootstrapFrames"
        )
    }

    // Feeds the anchor sampler from the per-frame HAL readback. Only converged
    // readbacks produced while the HAL is pinned to its fixed DAYLIGHT mode
    // describe the sensor's daylight response; scene-adaptive (AUTO) readbacks
    // are a property of the scene, not the sensor, and are ignored here.
    private fun feedDaylightAnchorEstimator() {
        if (!::daylightAnchorEstimator.isInitialized || rawColorProfile == null) return
        val est = daylightAnchorEstimator
        // The pill follows the estimator's state, not only the apply path:
        // convergence can arrive with the correction unchanged (small median
        // moves skip applyDaylightAnchor's hysteresis entirely), and the
        // periodic lens-shading-driven refresh does not run on HAL-map
        // devices. State transitions are rare, so posting per change is free.
        if (est.state != daylightAnchorLastState) {
            daylightAnchorLastState = est.state
            mainHandler.post { updateDaylightAnchorUI() }
        }
        if (est.state == DaylightAnchorEstimator.State.IDLE &&
            camera2Manager.latestAwbMode == android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
        ) {
            est.start()
        }
        if (est.state == DaylightAnchorEstimator.State.SAMPLING ||
            est.state == DaylightAnchorEstimator.State.CONVERGED
        ) {
            if (camera2Manager.latestAwbMode == android.hardware.camera2.CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT &&
                camera2Manager.latestAwbState == CaptureResult.CONTROL_AWB_STATE_CONVERGED
            ) {
                est.addSample(camera2Manager.latestColorCorrectionGains)
            }
        }
        if (daylightBootstrapActive) {
            daylightBootstrapFrames++
            if (est.state == DaylightAnchorEstimator.State.CONVERGED || daylightBootstrapFrames > 300) {
                stopDaylightAnchorBootstrap()
                // Anything but convergence means the HAL never reported stable
                // daylight (no fixed mode, an ignored override, or gains that
                // never settled): leave the pill Idle rather than Learning
                // forever. Convergence keeps the anchor it sampled.
                if (est.state != DaylightAnchorEstimator.State.CONVERGED) {
                    est.reset()
                }
            }
        }
        val anchor = est.currentAnchor ?: return
        val gains = floatArrayOf(anchor.rGain, 1f, anchor.bGain)
        val prev = daylightAnchorGains
        if (prev == null ||
            kotlin.math.abs(gains[0] - prev[0]) > 0.01f ||
            kotlin.math.abs(gains[2] - prev[2]) > 0.01f
        ) {
            daylightAnchorGains = gains
            applyDaylightAnchor()
        }
        if (est.state == DaylightAnchorEstimator.State.CONVERGED && !daylightAnchorPersisted) {
            val lens = lensManager.activeLens
            if (lens != null && persistDaylightAnchor(lens.cameraId, gains)) {
                daylightAnchorPersisted = true
            }
        }
    }

    private fun applyDaylightAnchor() {
        val profile = rawColorProfile ?: return
        val gains = daylightAnchorGains ?: return
        val d = profile.daylightCorrectionFor(floatArrayOf(gains[0], 1f, 1f, gains[2])) ?: return
        val applied = daylightAnchorApplied
        if (applied != null &&
            kotlin.math.abs(applied[0] - d[0]) < 0.005f &&
            kotlin.math.abs(applied[2] - d[2]) < 0.005f
        ) {
            return
        }
        daylightAnchorApplied = d
        profile.daylightCorrection = d
        // The estimator's daylight reference is cached per lens; rebuild it
        // against the corrected profile and carry the smoothing history over
        // so the applied product stays put.
        val oldDaylight = rawDaylightGains
        rawDaylightGains = profile.daylightTransform()?.wbGains
        // The renderer holds a product of both stages; rescale the smoothed
        // stage so it stays put across the swap.
        awbSmoother.retarget(oldDaylight, rawDaylightGains, previewRenderer.bayerColorMap)
        CrashLogger.log(
            TAG, "daylightAnchor: applied d=[${d.joinToString { String.format("%.4f", it) }}] " +
                "daylight wb=[${rawDaylightGains?.joinToString { String.format("%.3f", it) }}]"
        )
        mainHandler.post { updateDaylightAnchorUI() }
    }

    private fun daylightAnchorFile(lensId: String): File = File(filesDir, "daylight_anchor_$lensId.json")

    private fun persistDaylightAnchor(lensId: String, gains: FloatArray): Boolean {
        return try {
            val obj = JSONObject()
            obj.put("r", gains[0].toDouble())
            obj.put("b", gains[2].toDouble())
            FileOutputStream(daylightAnchorFile(lensId)).use {
                it.write(obj.toString().toByteArray(Charsets.UTF_8))
            }
            CrashLogger.log(TAG, "daylightAnchor: saved anchor for lens $lensId")
            true
        } catch (e: Exception) {
            CrashLogger.log(TAG, "daylightAnchor: save failed ${e.message}")
            false
        }
    }

    private fun loadDaylightAnchor(lensId: String): FloatArray? {
        val f = daylightAnchorFile(lensId)
        if (!f.exists()) return null
        return try {
            val obj = JSONObject(f.readText(Charsets.UTF_8))
            val r = obj.getDouble("r").toFloat()
            val b = obj.getDouble("b").toFloat()
            if (r.isFinite() && b.isFinite() && r in 0.3f..8f && b in 0.3f..8f) {
                floatArrayOf(r, 1f, b)
            } else {
                null
            }
        } catch (e: Exception) {
            CrashLogger.log(TAG, "daylightAnchor: load failed ${e.message}")
            null
        }
    }

    private fun deleteDaylightAnchorPersistence(lensId: String) {
        try {
            daylightAnchorFile(lensId).delete()
        } catch (_: Exception) {
        }
    }

    private fun updateDaylightAnchorUI() {
        if (!::daylightAnchorEstimator.isInitialized) return
        val frames = daylightAnchorEstimator.sampledFrames
        var text: String
        var color: Int
        when {
            daylightAnchorEstimator.state == DaylightAnchorEstimator.State.SAMPLING -> {
                text = "Learning"; color = 0xFFFFA726.toInt()
            }
            daylightAnchorEstimator.state == DaylightAnchorEstimator.State.CONVERGED -> {
                text = "Applied"; color = 0xFF7CB342.toInt()
            }
            daylightAnchorApplied != null -> {
                text = "Applied (saved)"; color = 0xFF7CB342.toInt()
            }
            else -> {
                text = "Idle"; color = 0xFF757575.toInt()
            }
        }
        if (frames > 0 && daylightAnchorEstimator.state != DaylightAnchorEstimator.State.IDLE) {
            text += " | $frames"
        }
        daylightAnchorStatusPill.text = text
        daylightAnchorStatusPill.setBackgroundColor(color)
        val canSample = cameraReady && camera2Manager.supportsDaylightMode
        daylightAnchorStartBtn.isEnabled = canSample
        daylightAnchorStartBtn.alpha = if (canSample) 1f else 0.5f
    }

    private val daylightAnchorNoticeHideRunnable = Runnable {
        lensShadingNoticeOverlay.animate().cancel()
        lensShadingNoticeOverlay.animate().alpha(0f).setDuration(300).withEndAction {
            lensShadingNoticeOverlay.visibility = View.GONE
        }.start()
    }

    private fun maybeShowDaylightAnchorNotice() {
        if (daylightAnchorNoticeShown) return
        daylightAnchorNoticeShown = true
        lensShadingNoticeOverlay.removeCallbacks(daylightAnchorNoticeHideRunnable)
        lensShadingNoticeOverlay.removeCallbacks(lensShadingNoticeHideRunnable)
        lensShadingNoticeOverlay.animate().cancel()
        lensShadingNoticeOverlay.text =
            "Sampling the sensor's daylight response to calibrate the white balance presets.\n" +
                "Check the Daylight Anchor settings to restart or clear it."
        lensShadingNoticeOverlay.alpha = 0f
        lensShadingNoticeOverlay.visibility = View.VISIBLE
        lensShadingNoticeOverlay.animate().alpha(1f).setDuration(200).withEndAction {
            lensShadingNoticeOverlay.postDelayed(daylightAnchorNoticeHideRunnable, 10_000)
        }.start()
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
                showLensWarning("Lens lacks autofocus - tap to adjust exposure only")
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
            Log.e(TAG, "buildLensSelectorUI: invariant violated - RAW active but active lens ${active.cameraId} hasRaw=${active.hasRawSensor}, rawLenses=${lensesForFacing.size}")
            developerSwitch.revertToggle()
            updateFrontRearToggleVisibility(false)
            // Camera pipeline may still be in RAW mode - force back to YUV and restart
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

    // KELVIN as a parameterized preset: the slider pair names an illuminant
    // chromaticity (kelvin picks the CCT on the Planckian locus, tint walks
    // perpendicular to it), and the profile path adapts from it exactly like
    // from a table preset's constant.
    private fun kelvinSceneXy(): FloatArray {
        val (x, y) = WhiteBalanceMath.kelvinToXy(kelvinState.kelvin, kelvinState.tint)
        return floatArrayOf(x, y)
    }

    // Raw degree of adaptation for a scene chromaticity, from the current
    // estimate and scene luminance proxy.
    private fun limitedAdaptationTarget(sceneXy: FloatArray): Float {
        val (distanceUv, nearestCct) = WhiteBalanceMath.locusDistanceUv(sceneXy[0], sceneXy[1])
        val la = if (lastSceneGreenLevel > 0f) {
            lastSceneGreenLevel.toDouble() * LIMITED_WB_LA_GAIN
        } else {
            200.0
        }
        return WhiteBalanceMath.limitedAdaptation(la, nearestCct, distanceUv)
    }

    // Degree of adaptation for the limited auto white balance, low-passed so
    // the correction eases in and out over about a second and a half instead
    // of stepping with each estimator update.
    private fun smoothedLimitedAdaptation(sceneXy: FloatArray): Float {
        smoothedAdaptation += LIMITED_WB_SMOOTHING * (limitedAdaptationTarget(sceneXy) - smoothedAdaptation)
        return smoothedAdaptation.coerceIn(0f, 1f)
    }

    private fun uploadAgxUniforms() {
        // All white balance, KELVIN included, lives in the profile path: gains
        // plus the WB-removed matrix through the Bayer pipeline, with the
        // illuminant adapted in the camera-native domain. KELVIN is a preset
        // whose table entry is computed from the sliders, so nothing is
        // adapted here after sRGB anymore; this stays identity.
        val sceneLinearTo709 = ColorMatrix.identity()

        val insetParams = agxParams.toInsetParams()
        val agx = AgxPrecomputer.compute(insetParams, sceneLinearTo709, whiteLevel = rawWhiteLevel.toFloat(), blackLevel = rawBlackLevel, middleGrayPercent = agxParams.middleGray)

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
        // Stop the blink chain first: its tick runnable lives on the main
        // handler and would keep posting while holding this Activity (and the
        // whole GL/preview reference chain) alive after destroy.
        criticalBlinkController.stop()
        super.onDestroy()
        thermalManager.stop()
        performCleanup()
    }

    private fun onThermalStateChanged(state: ThermalManager.State) {
        if (!::previewRenderer.isInitialized) return
        lastThermalState = state
        handleThermalTier(state)
    }

    /**
     * Applies the tier for [state]. Actuators only run when thermal protection
     * is enabled (the protected-mode behavior). With protection off the state
     * machine + UI keep reporting, but every actuator is a no-op: full
     * fps/resolution stay (restored immediately on disable mid-throttle), the
     * torch is never forced off, no RAW shutdown, no blackout, and at CRITICAL
     * only the flashing warning is shown. Called from state transitions, the
     * protection toggle, and cold start alike.
     */
    private fun handleThermalTier(state: ThermalManager.State) {
        updateThermalIndicatorFor(state)
        // Recorded before the early return, so the report names the tier even
        // when protection is off and nothing was actually throttled. A timing
        // segment whose preview size moved needs to say whether the app's own
        // controller moved it; otherwise a self-imposed resolution drop reads
        // as a hardware downclock.
        previewRenderer.setAppThermalTier(state.name)

        if (!thermalProtectionEnabled) {
            // Protection off: restore full throughput immediately (the same
            // restore path used on the state step-down) and never blackout,
            // stop the stream, or block capture.
            previewRenderer.setPreviewSuppressed(false)
            previewRenderer.setPreviewResolutionCap(0)
            camera2Manager.setThermalFpsLimit(0)
            hideCriticalThermalOverlay()
            updateBlinkWarning(protectionEnabled = false, state)
            return
        }

        when (state) {
            ThermalManager.State.NORMAL -> {
                hideCriticalThermalOverlay()
                previewRenderer.setPreviewSuppressed(false)
                previewRenderer.setPreviewResolutionCap(0)
                camera2Manager.setThermalFpsLimit(0)
            }
            ThermalManager.State.WARM -> {
                previewRenderer.setPreviewSuppressed(false)
                applyThermalThrottle()
            }
            ThermalManager.State.HOT -> {
                // Flash is prohibited entirely while hot: torch on -> off, and
                // other modes forced back to OFF.
                forceFlashOffForThermal()
                previewRenderer.setPreviewSuppressed(false)
                applyThermalThrottle()
            }
            ThermalManager.State.CRITICAL -> {
                // Torch is torn down FIRST, then the RAW stream shuts down:
                // frame handlers drop incoming frames and the renderer paints
                // black, keeping the viewfinder fully dark until cooldown.
                forceFlashOffForThermal()
                showCriticalThermalOverlay()
                previewRenderer.setPreviewSuppressed(true)
            }
        }
        updateBlinkWarning(protectionEnabled = true, state)
    }

    /** WARM/HOT throttle: sensor fps cap + demosaic/output resolution cap <=480p. */
    private fun applyThermalThrottle() {
        camera2Manager.setThermalFpsLimit(thermalManager.throttledPreviewFps)
        previewRenderer.setPreviewResolutionCap(thermalManager.maxPreviewResolutionDim)
    }

    private fun onThermalTempUpdate(displayC: Float) {
        // Debounce: temperature backs the on-screen label/overlay numbers, so only
        // repaint when the reading actually moved or a label still needs showing.
        val changed = kotlin.math.abs(displayC - lastThermalTempC) >= 0.05f
        lastThermalTempC = displayC
        if (!changed && thermalIndicator.visibility != View.VISIBLE && criticalThermalOverlay.visibility != View.VISIBLE) {
            return
        }
        if (lastThermalState != ThermalManager.State.NORMAL && changed) {
            updateThermalIndicatorFor(lastThermalState)
        }
        if (criticalThermalOverlay.visibility == View.VISIBLE) {
            criticalThermalOverlay.text =
                String.format("BATTERY TOO HOT \u2014 %.1f\u00B0C\nCapture disabled until battery cools down", displayC)
        }
    }

    private fun updateThermalIndicatorFor(state: ThermalManager.State) {
        if (!thermalProtectionEnabled) {
            // Persistent corner strip while protection is off.
            thermalIndicator.text = String.format("THERMAL PROTECTION OFF, current state: %s", state.name)
            thermalIndicator.setTextColor(0xFFFFAA00.toInt())
            thermalIndicator.visibility = View.VISIBLE
            return
        }
        when (state) {
            ThermalManager.State.NORMAL -> {
                thermalIndicator.visibility = View.GONE
            }
            ThermalManager.State.WARM -> {
                thermalIndicator.text = String.format("BATTERY WARM: %.1f\u00B0C", lastThermalTempC)
                thermalIndicator.setTextColor(0xFFFFAA00.toInt())
                thermalIndicator.visibility = View.VISIBLE
            }
            ThermalManager.State.HOT -> {
                thermalIndicator.text = String.format("BATTERY HOT: %.1f\u00B0C", lastThermalTempC)
                thermalIndicator.setTextColor(0xFFFF4444.toInt())
                thermalIndicator.visibility = View.VISIBLE
            }
            ThermalManager.State.CRITICAL -> {
                thermalIndicator.text = String.format("BATTERY CRITICAL: %.1f\u00B0C", lastThermalTempC)
                thermalIndicator.setTextColor(0xFFFF0000.toInt())
                thermalIndicator.visibility = View.VISIBLE
            }
        }
    }

    private fun showCriticalThermalOverlay() {
        criticalThermalOverlay.text = String.format(
            "BATTERY TOO HOT \u2014 %.1f\u00B0C\nCapture disabled until battery cools down",
            lastThermalTempC
        )
        criticalThermalOverlay.visibility = View.VISIBLE
    }

    private fun hideCriticalThermalOverlay() {
        criticalThermalOverlay.visibility = View.GONE
    }

    /**
     * User pressed the protection switch. Enabling applies immediately; each
     * disabling requires fresh confirmation, and the switch reverts on cancel.
     * While the confirmation is pending every further tap is swallowed (no
     * second dialog, no apply) so the applied state and the switch position
     * always agree.
     */
    private fun onThermalProtectionToggleRequested(checked: Boolean) {
        if (thermalProtectionDialogShowing) {
            // The tap already flipped the switch while the dialog owns the
            // decision: put the switch back on the still-applied state (ON)
            // and ignore it.
            restoreThermalProtectionSwitch()
            return
        }
        if (checked) {
            applyThermalProtection(true)
            return
        }
        thermalProtectionDialogShowing = true
        AlertDialog.Builder(this)
            .setMessage("Disabling thermal protection might cause permanent damage to the device upon heavy use, are you sure?")
            .setPositiveButton("Turn off protection") { _, _ -> applyThermalProtection(false) }
            .setNegativeButton("Keep enabled") { _, _ -> restoreThermalProtectionSwitch() }
            .setOnCancelListener { restoreThermalProtectionSwitch() }
            .setOnDismissListener { thermalProtectionDialogShowing = false }
            .show()
    }

    /**
     * Wires the two controls that feed the debug log export.
     *
     * "Record processing time" persists: turning it on keeps accumulating until
     * it is turned off. "Attach raw frame" covers the next export only, so it
     * is not persisted.
     */
    private fun setupMeasurementSwitches() {
        val timingSwitch = findViewById<android.widget.Switch>(R.id.measurement_timing_switch)
        val rawFrameCheck = findViewById<android.widget.CheckBox>(R.id.measurement_raw_frame_check)

        measurementEnabled = previewResPrefs.getBoolean(PREF_MEASUREMENT, false)
        timingSwitch.isChecked = measurementEnabled

        timingSwitch.setOnClickListener {
            measurementEnabled = timingSwitch.isChecked
            previewResPrefs.edit().putBoolean(PREF_MEASUREMENT, measurementEnabled).apply()
            applyMeasurementFlags()
            CrashLogger.log(TAG, "record processing time: $measurementEnabled")
        }

        rawFrameCheck.setOnClickListener {
            if (rawFrameCheck.isChecked && ::previewRenderer.isInitialized) {
                previewRenderer.armFrameGrabOnce()
                Toast.makeText(
                    this,
                    "Raw frame will be attached to the next export",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
        applyMeasurementFlags()
    }

    private fun applyMeasurementFlags() {
        if (!::previewRenderer.isInitialized) return
        previewRenderer.setMeasurementEnabled(measurementEnabled)
    }

    /**
     * Folds the renderer-side measurement state into the log as a section. The
     * renderer owns the histogram and the profile, so this only snapshots it;
     * the section is cleared by the next save, matching the existing dump-guard
     * behavior.
     */
    private fun saveMeasurementBundle() {
        if (!::previewRenderer.isInitialized) return
        CrashLogger.setSection("Measurement", previewRenderer.measurementBundle())
        // The grab is one-shot, so its report lines are retired here. Without
        // this the next export would replay a capture that was already reported.
        previewRenderer.clearFrameGrabReport()
    }

    /**
     * Writes the pending one-shot raw frame next to the log. Returns true only
     * when a file was actually written, so the toast cannot claim an
     * attachment that never arrived.
     */
    private fun writePendingFrameGrab(): Boolean {
        if (!::previewRenderer.isInitialized) return false
        val bytes = previewRenderer.takeFrameGrabBytes() ?: return false
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val ok = CrashLogger.writeDownloadsBytes(
            this, "agxframe_$stamp.framegrab", bytes, "application/octet-stream"
        )
        // After a successful write only: rotation deletes older grabs, and
        // doing that when this write failed would lose evidence without adding
        // a replacement.
        if (ok) CrashLogger.rotateDownloadsBlobs(this, "agxframe_", FRAME_GRAB_KEEP)
        // Handed back to the renderer so the bundle names the attachment.
        previewRenderer.setFrameGrabSavedPath(if (ok) CrashLogger.lastDownloadsLocation() else null)
        if (!ok) {
            CrashLogger.log(TAG, "raw frame attachment failed to write")
        }
        return ok
    }

    private fun restoreThermalProtectionSwitch() {
        thermalProtectionSwitch.isChecked = true
    }

    /** Persist the new protection state, then immediately re-resolve the tier. */
    private fun applyThermalProtection(enabled: Boolean) {
        thermalProtectionEnabled = enabled
        previewResPrefs.edit().putBoolean(PREF_THERMAL_PROTECTION_ENABLED, enabled).apply()
        // Single source of truth for the switch: whatever decision the dialog
        // produced, the toggle comes to rest matching the applied state.
        thermalProtectionSwitch.isChecked = enabled
        // Force one evaluation now so the toggle takes effect immediately
        // rather than on the next ~1s poll tick.
        thermalManager.forceEvaluation()
        handleThermalTier(thermalManager.currentState)
    }

    /**
     * The flashing full-screen warning only ever appears with protection off
     * AND critical. start() is idempotent, so repeated transitions and toggle
     * spam can never double-pace or leak a second tick chain.
     */
    private fun updateBlinkWarning(protectionEnabled: Boolean, state: ThermalManager.State) {
        if (!protectionEnabled && state == ThermalManager.State.CRITICAL) {
            criticalBlinkController.start()
        } else {
            criticalBlinkController.stop()
            criticalBlinkWarning.visibility = View.GONE
        }
    }

    private fun forceFlashOffForThermal() {
        if (currentFlashMode == FlashMode.OFF && !thermalManager.isTorchActive) return
        currentFlashMode = FlashMode.OFF
        thermalManager.isTorchActive = false
        updateFlashUI()
        if (cameraReady) camera2Manager.setFlashMode(FlashMode.OFF)
    }

    private fun onThermalTorchForcedOff(reason: ThermalManager.TorchShutdownReason) {
        // Torch off FIRST (before any RAW stream shutdown on critical).
        forceFlashOffForThermal()
        val message = when (reason) {
            ThermalManager.TorchShutdownReason.DURATION_LIMIT ->
                "Battery warm \u2014 torch duration is limited, torch off"
            ThermalManager.TorchShutdownReason.HOT ->
                "Battery hot \u2014 torch off"
            ThermalManager.TorchShutdownReason.CRITICAL ->
                "Battery too hot \u2014 torch off"
        }
        showWarningForDuration(message, 10_000)
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

    /** Photo-session snapshot used for every capture path (RAW, GPU, black). */
    private fun buildCaptureSession(): CaptureSession {
        return CaptureSession(
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
    }

    /**
     * Critical-mode capture: the RAW stream is shut down (viewfinder is dark),
     * so a shutter press still completes but encodes a fully black frame. The
     * sensor is never engaged - nothing is exposed.
     */
    private fun captureBlackStill() {
        shutterController.onCaptureSubmitted()
        isCapturing = true
        finishingCaptureOverlay.visibility = View.VISIBLE

        val session = buildCaptureSession()
        val targetW = if (session.resolutionWidth > 0) session.resolutionWidth else 640
        val targetH = if (session.resolutionHeight > 0) session.resolutionHeight else 480

        Thread {
            try {
                val bitmap = android.graphics.Bitmap.createBitmap(targetW, targetH, android.graphics.Bitmap.Config.ARGB_8888)
                bitmap.eraseColor(android.graphics.Color.rgb(0, 0, 0))
                val (jpegData, metadata) = encodeCaptureBitmap(bitmap, session)
                bitmap.recycle()
                saveCaptureJpeg(jpegData, null, session, metadata)
            } catch (e: Exception) {
                Log.e(TAG, "Black capture failed", e)
                mainHandler.post {
                    Toast.makeText(this, "Capture failed: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            } finally {
                mainHandler.post {
                    isCapturing = false
                    shutterController.onCaptureComplete()
                    finishingCaptureOverlay.visibility = View.GONE
                    if (pendingPauseCleanup) {
                        pendingPauseCleanup = false
                        performCleanup()
                    }
                }
            }
        }.start()
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

        // Post-processing EV: +-10 EV in 0.5 EV steps (sensitive scrolling for big
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
        previewRenderer.isoForDenoise = iso
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
        previewRenderer.isoForDenoise = iso
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
        val bounds = Pair(maxDim, maxDim * screenHeight / screenWidth)
        // The height bound is cap * screenAspect. On a tall screen that lands
        // below the sensor's own 4:3 ladder (cap 854 on 20:9 -> 854x384), which
        // filters out every size except the smallest and pins the preview at
        // 320x240 / zoomK 12.8. Log the resolved numbers so that failure mode is
        // visible instead of having to be inferred from the chosen size.
        CrashLogger.log(
            TAG,
            "preview cap=$maxDim screen=${screenWidth}x$screenHeight -> bounds=" +
                "${bounds.first}x${bounds.second}; 4:3 needs maxHeight>=480 -> cap>=${(480 * screenWidth + screenHeight - 1) / screenHeight}"
        )
        return bounds
    }

    companion object {
        private const val TAG = "MainActivity"
        const val DEFAULT_WB_KELVIN = KelvinState.DEFAULT_KELVIN
        const val DEFAULT_WB_TINT = KelvinState.DEFAULT_TINT
        // Rec.709 Y row of RGB->XYZ(D65); fallback camera-native luminance
        // coefficients when no native->XYZ map is available.
        private val DEFAULT_LUMA_COEFFS = floatArrayOf(0.2126f, 0.7152f, 0.0722f)
        private const val REQUEST_CAMERA = 100
        private const val FOCUS_ROLLER_MAX_INDEX = 100
        private const val PREF_PREVIEW_RES_CAP = "preview_res_cap"
        private const val PREF_FOCUS_TIMEOUT = "focus_indicator_timeout"
        private const val PREF_STARTUP_PRESET = "startup_preset"
        private const val PREF_THERMAL_PROTECTION_ENABLED = "thermal_protection_enabled"

    /**
     * Captured-frame attachments kept in Downloads. Three is enough to hold the
     * last build's grab alongside the current one and a known-good reference,
     * at about 15 MB each.
     */
    private const val FRAME_GRAB_KEEP = 3
        // White balance is a user preference, not a per-lens one: it persists
        // across lens switches and app restarts (global, like flash mode).
        private const val PREF_WB_MODE = "wb_mode_global"
        private const val PREF_WB_KELVIN = "wb_kelvin_global"
        private const val PREF_WB_TINT = "wb_tint_global"
        // Limited auto white balance: gate the scene->D65 correction by how
        // much adaptation the estimated light earns (chromatic LEDs earn none).
        private const val PREF_LIMITED_WB = "limited_auto_wb"
        // Normalized scene luminance -> adapting luminance estimate in cd/m2.
        private const val LIMITED_WB_LA_GAIN = 1000.0
        // Low-pass on the degree of adaptation: ~1.5 s at the estimator's
        // ~0.5 s update cadence.
        private const val LIMITED_WB_SMOOTHING = 0.28f
        private const val RAW_BUFFER_POOL = 3
        // Leading factor of the demosaic clipping-neutralization exponent.
        private const val PREF_CLIP_ATTEN = "clip_atten_factor"
        // If AE has not converged within this many frames, run the estimator
        // anyway so a permanently hunting AE cannot disable AUTO.
        private const val AWB_INPUT_TIMEOUT_FRAMES = 90
        // Fixed per-estimator-step rate for the profile-path neutral derived
        // from the color gains. Deliberately independent of the illuminant
        // stage's dual-rate alpha.
        private const val ESTIMATOR_NEUTRAL_SMOOTHING = 0.4f
        // Multi-stage denoise strengths, 0..1 each.
        private const val PREF_S1_DPC = "stage1_dpc_strength"
        private const val PREF_S3_RAW = "stage3_raw_strength"
        private const val PREF_S5_OUT = "stage5_out_strength"
private const val PREF_MEASUREMENT = "measurement_timing"
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
        // Post-processing EV roller: 0.5 EV per step over the +-10 EV range.
        private const val EV_PP_MAX_INDEX = 40
        private const val EV_PP_MID_INDEX = 20
        private const val EV_PP_DEFAULT_INDEX = 23
    }
}
