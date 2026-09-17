package com.agx.camera.camera

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.util.Log
import android.util.Size
import com.agx.camera.CrashLogger
import org.json.JSONObject
import java.io.File

data class LensInfo(
    val cameraId: String,
    val facing: Int,
    val focalLengthMm: Float,
    val hasRawSensor: Boolean,
    val hardwareLevel: Int,
    val label: String,
    val jpegOutputSizes: Array<android.util.Size> = emptyArray(),
    val sensorActiveWidth: Int = 0,
    val sensorActiveHeight: Int = 0,
    val maxDigitalZoom: Float = 1.0f,
    val sensorWidthMm: Float = 0f,
    val sensorHeightMm: Float = 0f
) {
    val focalLength35mmEq: Float
        get() {
            if (focalLengthMm <= 0f) return 0f
            val diagonal = kotlin.math.sqrt(
                sensorWidthMm * sensorWidthMm + sensorHeightMm * sensorHeightMm
            )
            if (diagonal <= 0f) return 0f
            return focalLengthMm * 43.27f / diagonal
        }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is LensInfo) return false
        return cameraId == other.cameraId
    }
    override fun hashCode(): Int = cameraId.hashCode()
}

data class LensState(
    val zoomFactor: Float = 1.0f,
    val zoomCenterX: Float = 0.5f,
    val zoomCenterY: Float = 0.5f,
    val wbModeOrdinal: Int = 0,
    val kelvin: Float = 5500f,
    val kelvinTint: Float = 0f,
    val flashModeOrdinal: Int = 0
)

class LensManager(private val context: Context) {
    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val _lenses = mutableListOf<LensInfo>()
    val lenses: List<LensInfo> get() = _lenses

    // Lens classification
    var lensOrganization: LensOrganization? = null
        private set
    var lensLabels: Map<String, String> = emptyMap()
        private set

    var activeLens: LensInfo? = null
        private set

    var lastUsedRearLensId: String? = null
        private set

    var listener: ((LensInfo) -> Unit)? = null

    private val lensStates = mutableMapOf<String, LensState>()

    fun enumerate(): Boolean {
        _lenses.clear()
        var hasSupportedLens = false
        CrashLogger.log(TAG, "enumerate: starting")

        val cameraIds = try {
            cameraManager.cameraIdList
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get camera ID list: ${e.message}", e)
            return false
        }

        CrashLogger.log(TAG, "enumerate: raw cameraIdList=${cameraIds.joinToString()}")

        val logicalMultiCameraId = findLogicalMultiCamera()

        for (id in cameraIds) {
            val chars = try {
                cameraManager.getCameraCharacteristics(id)
            } catch (e: Exception) {
                CrashLogger.log(TAG, "enumerate: id=$id characteristics FAILED: ${e.message}")
                Log.w(TAG, "Failed to get characteristics for camera $id: ${e.message}")
                continue
            }

            val facing = chars.get(CameraCharacteristics.LENS_FACING)
            val level = chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
                ?: CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY

            if (facing == null) {
                CrashLogger.log(TAG, "enumerate: id=$id skipping: LENS_FACING == null")
                continue
            }
            if (level < CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED) {
                CrashLogger.log(TAG, "enumerate: id=$id skipping: level=$level < LIMITED")
                Log.w(TAG, "Camera $id hardware level $level < LIMITED, skipping")
                continue
            }

            val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val allFocalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList() ?: emptyList()
            val focal = allFocalLengths.firstOrNull() ?: 0.0f
            val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()
            val capNames = caps.joinToString(",") {
                when (it) {
                    0 -> "BACKWARD_COMPATIBLE"; 1 -> "MANUAL_SENSOR"; 2 -> "MANUAL_POST_PROCESSING"
                    3 -> "RAW"; 4 -> "PRIVATE_REPROCESSING"; 5 -> "READ_SENSOR_SETTINGS"
                    6 -> "BURST_CAPTURE"; 7 -> "DEPTH_OUTPUT"; 8 -> "CONSTRAINED_HIGH_SPEED_VIDEO"
                    9 -> "MOTION_TRACKING"; 10 -> "LOGICAL_MULTI_CAMERA"; 11 -> "MONOCHROME"; 12 -> "SECURE_IMAGE_DATA"
                    13 -> "SYSTEM_CAMERA"; 14 -> "OFFLINE_PROCESSING"; 15 -> "ULTRA_HIGH_RESOLUTION_SENSOR"
                    16 -> "BASIC_MANUAL"; 17 -> "CROSS_STREAM"; 18 -> "FULL_QUALITY"; 19 -> "FRONT_MONOCHROME"
                    20 -> "ALGORITHM"; 21 -> "PRIVATE_REPROCESSING_STREAM_SUPPORT"; 22 -> "FLASH"; else -> "UNKNOWN($it)"
                }
            }
            val hasRawFromCap = caps.contains(17)
            val activeArray = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            val streamMap = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)

            val logicalPhysicalIds = try {
                chars.get(
                    CameraCharacteristics.Key("android.logicalMultiCameraPhysicalIds", Array<String>::class.java)
                )
            } catch (e: Exception) {
                CrashLogger.log(TAG, "enumerate: id=$id logicalPhysicalIds read failed: ${e.message}")
                null
            }

            val physicalIds = try {
                // Public API since API 28 (minSdk 30); previously reflected into the
                // hidden method (SoonBlockedPrivateApi lint + runtime greylist).
                chars.getPhysicalCameraIds().toString()
            } catch (e: Exception) {
                "err"
            }

            // Diagnostic list of the multi-camera-related key names.  Previously
            // reflected into the private CameraCharacteristics.mProperties field
            // (SoonBlockedPrivateApi lint + runtime greylist); the public getKeys()
            // surface exposes the same SDK-key names (hidden OEM keys may be
            // omitted from the dump as a result).
            val multiKeys = try {
                chars.keys
                    .filter {
                        it.name.contains("logical", ignoreCase = true) ||
                            it.name.contains("physical", ignoreCase = true) ||
                            it.name.contains("multi", ignoreCase = true)
                    }
                    .map { it.name }
                    .toMutableList()
            } catch (e: Exception) {
                mutableListOf()
            }

            // Fallback: some OEM HALs (e.g. Xiaomi) don't advertise RAW in capabilities
            // but do support RAW_SENSOR output via stream configuration
            val hasRaw = hasRawFromCap || (
                streamMap?.getOutputSizes(android.graphics.ImageFormat.RAW_SENSOR)?.isNotEmpty() == true
            )
            
            // New detailed characteristics for lens classification
            val maxAfRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0
            val maxAeRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0
            val afModes = chars.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: intArrayOf()
            val minFocusDist = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
            val hasFlash = chars.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) ?: false

            val rawDetectMethod = when {
                hasRawFromCap -> "capability"
                hasRaw -> "streamConfig"
                else -> "none"
            }
            CrashLogger.log(TAG, "enumerate: id=$id facing=$facing level=$level focal=$focal focalLens=$allFocalLengths sensor=${chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)} logicalPhysical=${logicalPhysicalIds?.joinToString() ?: "-"} physicalCameraIds=$physicalIds caps=$capNames")
            CrashLogger.log(TAG, "enumerate: id=$id hasRaw=$hasRaw rawDetect=$rawDetectMethod maxAfRegions=$maxAfRegions maxAeRegions=$maxAeRegions afModes=${afModes.toList()} minFocusDist=$minFocusDist hasFlash=$hasFlash multiKeys=$multiKeys")

            _lenses.add(LensInfo(id, facing, focal, hasRaw, level, "",
                jpegOutputSizes = streamMap?.getOutputSizes(android.graphics.ImageFormat.JPEG) ?: emptyArray(),
                sensorActiveWidth = activeArray?.width() ?: 0,
                sensorActiveHeight = activeArray?.height() ?: 0,
                maxDigitalZoom = (chars.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1.0f),
                sensorWidthMm = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.width ?: 0f,
                sensorHeightMm = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.height ?: 0f
            ))
            hasSupportedLens = true
        }

        // Discover hidden cameras: physical sub-cameras of any logical camera
        // (e.g. ids 2,3,4,5 under back logical id 0 on Xiaomi) plus brute-probed
        // numeric ids that answer getCameraCharacteristics but are absent from
        // cameraIdList.
        val discovered = mutableListOf<String>()
        for (lens in _lenses.toList()) {
            try {
                val chars = cameraManager.getCameraCharacteristics(lens.cameraId)
                discovered += chars.physicalCameraIds
            } catch (e: Exception) { /* not a logical multi-camera */ }
        }
        val knownSet = cameraIds.toSet()
        for (probe in 0..12) {
            if (probe.toString() in knownSet) continue
            discovered += probe.toString()
        }
        for (id in discovered.distinct()) {
            addDiscoveredLens(id)
        }

        assignLabels(logicalMultiCameraId)

        // Classify lenses using LensClassifier
        val profiles = _lenses.map { lens ->
            val chars = cameraManager.getCameraCharacteristics(lens.cameraId)
            val maxAfRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0
            val maxAeRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0
            val afModes = chars.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: intArrayOf()
            val hasContinuousAf = afModes.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            val hasAutoAf = afModes.contains(CaptureRequest.CONTROL_AF_MODE_AUTO)
            val minFocusDist = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
            val hasFlash = chars.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) ?: false
            val maxSize = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?.getOutputSizes(ImageFormat.YUV_420_888)
                ?.maxByOrNull { it.width * it.height } ?: Size(0, 0)

            LensProfile(
                id = lens.cameraId,
                facing = lens.facing,
                hardwareLevel = lens.hardwareLevel,
                focalLength = lens.focalLengthMm,
                maxAfRegions = maxAfRegions,
                maxAeRegions = maxAeRegions,
                hasContinuousAf = hasContinuousAf,
                hasAutoAf = hasAutoAf,
                maxResolution = maxSize,
                hasFlash = hasFlash,
                minFocusDistance = minFocusDist
            )
        }

        lensOrganization = LensClassifier.organize(profiles)
        lensLabels = LensClassifier.computeLabels(lensOrganization!!)

        CrashLogger.log(TAG, "enumerate: found ${_lenses.size} lenses, hasSupported=$hasSupportedLens")
        CrashLogger.log(TAG, "  primaryBack=${lensOrganization?.primaryBack?.id} usableBack=${lensOrganization?.usableBack?.map { it.id }} front=${lensOrganization?.front?.map { it.id }}")
        CrashLogger.log(TAG, "  labels=$lensLabels")

        return hasSupportedLens
    }

    private fun addDiscoveredLens(id: String): Boolean {
        if (_lenses.any { it.cameraId == id }) {
            CrashLogger.log(TAG, "enumerate: discovered id=$id already present, skipping")
            return false
        }
        val chars = try {
            cameraManager.getCameraCharacteristics(id)
        } catch (e: Exception) {
            CrashLogger.log(TAG, "enumerate: discovered id=$id not present: ${e.message}")
            return false
        }
        val facing = chars.get(CameraCharacteristics.LENS_FACING)
        val level = chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
            ?: CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY
        if (facing == null) {
            CrashLogger.log(TAG, "enumerate: discovered id=$id skipping: LENS_FACING == null")
            return false
        }
        if (level < CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED) {
            CrashLogger.log(TAG, "enumerate: discovered id=$id skipping: level=$level < LIMITED")
            return false
        }
        val focal = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: 0.0f
        val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()
        val streamMap = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val hasRaw = caps.contains(17) || (
            streamMap?.getOutputSizes(ImageFormat.RAW_SENSOR)?.isNotEmpty() == true
        )
        val activeArray = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
        val activeW = activeArray?.width() ?: 0
        val activeH = activeArray?.height() ?: 0

        // Dedupe: OEMs (e.g. Xiaomi) expose the same physical sensor multiple times
        // (logical camera + physical sub-camera + hidden clones). Same facing, focal
        // length and active-array size => same lens.
        val duplicateOf = _lenses.firstOrNull { existing ->
            existing.facing == facing &&
            Math.abs(existing.focalLengthMm - focal) < 0.05f &&
            existing.sensorActiveWidth == activeW &&
            existing.sensorActiveHeight == activeH
        }
        if (duplicateOf != null) {
            CrashLogger.log(TAG, "enumerate: discovered id=$id duplicates existing id=${duplicateOf.cameraId} (focal=$focal size=${activeW}x$activeH), skipping")
            return false
        }

        CrashLogger.log(TAG, "enumerate: add discovered id=$id facing=$facing level=$level focal=$focal sensor=${chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)} hasRaw=$hasRaw")
        _lenses.add(LensInfo(id, facing, focal, hasRaw, level, "",
            jpegOutputSizes = streamMap?.getOutputSizes(ImageFormat.JPEG) ?: emptyArray(),
            sensorActiveWidth = activeW,
            sensorActiveHeight = activeH,
            maxDigitalZoom = chars.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1.0f,
            sensorWidthMm = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.width ?: 0f,
            sensorHeightMm = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.height ?: 0f
        ))
        return true
    }

    private fun findLogicalMultiCamera(): String? {
        val cameraIds = try {
            cameraManager.cameraIdList
        } catch (e: Exception) {
            return null
        }

        for (id in cameraIds) {
            val chars = try {
                cameraManager.getCameraCharacteristics(id)
            } catch (e: Exception) {
                continue
            }
            val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()
            if (caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA)) {
                return id
            }
        }
        return null
    }

    private fun assignLabels(logicalMultiCameraId: String?) {
        val rearLenses = _lenses.filter { it.facing == CameraCharacteristics.LENS_FACING_BACK }.sortedBy { it.focalLengthMm }
        val frontLenses = _lenses.filter { it.facing == CameraCharacteristics.LENS_FACING_FRONT }

        if (rearLenses.size == 1) {
            rearLenses[0].let { lens ->
                val idx = _lenses.indexOf(lens)
                _lenses[idx] = lens.copy(label = "Wide")
            }
        } else if (rearLenses.size == 2) {
            rearLenses[0].let { lens ->
                val idx = _lenses.indexOf(lens)
                _lenses[idx] = lens.copy(label = "Wide")
            }
            rearLenses[1].let { lens ->
                val idx = _lenses.indexOf(lens)
                _lenses[idx] = lens.copy(label = "Tele")
            }
        } else if (rearLenses.size >= 3) {
            rearLenses.forEachIndexed { i, lens ->
                val label = when (i) {
                    0 -> "Wide"
                    rearLenses.lastIndex -> "Super Tele"
                    else -> "Tele"
                }
                val idx = _lenses.indexOf(lens)
                _lenses[idx] = lens.copy(label = label)
            }
        }

        for (lens in frontLenses) {
            val idx = _lenses.indexOf(lens)
            _lenses[idx] = lens.copy(label = "Front")
        }
    }

    fun selectPrimary(): LensInfo? {
        // Use LensClassifier's primary back camera if available
        val primary = lensOrganization?.primaryBack?.let { profile ->
            _lenses.firstOrNull { it.cameraId == profile.id }
        } ?: _lenses.filter { it.facing == CameraCharacteristics.LENS_FACING_BACK }
            .minByOrNull { it.focalLengthMm } ?: _lenses.firstOrNull { it.facing == CameraCharacteristics.LENS_FACING_BACK }

        if (primary != null) {
            activeLens = primary
            lastUsedRearLensId = primary.cameraId
            Log.d(TAG, "Selected primary lens: ${primary.cameraId} ${primary.label}")
        }
        return primary
    }

    fun saveCurrentState(
        lensId: String,
        zoomFactor: Float,
        zoomCenterX: Float,
        zoomCenterY: Float,
        wbModeOrdinal: Int,
        kelvin: Float,
        kelvinTint: Float,
        flashModeOrdinal: Int
    ) {
        val state = LensState(zoomFactor, zoomCenterX, zoomCenterY, wbModeOrdinal, kelvin, kelvinTint, flashModeOrdinal)
        lensStates[lensId] = state
        persistState(lensId, state)
    }

    fun getRestoredState(lensId: String): LensState {
        val cached = lensStates[lensId]
        if (cached != null) return cached
        val loaded = loadState(lensId)
        lensStates[lensId] = loaded
        return loaded
    }

    fun switchLens(lens: LensInfo, saveCurrentStateFn: (String) -> Unit) {
        val prev = activeLens
        if (prev != null) {
            saveCurrentStateFn(prev.cameraId)
        }
        activeLens = lens
        if (lens.facing == CameraCharacteristics.LENS_FACING_BACK) {
            lastUsedRearLensId = lens.cameraId
        }
        Log.d(TAG, "Switched to lens: ${lens.cameraId} ${lens.label}")
        listener?.invoke(lens)
    }

    fun getFrontLens(rawOnly: Boolean = false): LensInfo? {
        val front = _lenses.filter { it.facing == CameraCharacteristics.LENS_FACING_FRONT }
        return if (rawOnly) front.firstOrNull { it.hasRawSensor } else front.firstOrNull()
    }

    fun getRearLenses(rawOnly: Boolean = false): List<LensInfo> {
        val rear = lensOrganization?.usableBack?.mapNotNull { profile ->
            _lenses.firstOrNull { it.cameraId == profile.id }
        }?.sortedBy { it.focalLengthMm }
        ?: _lenses.filter { it.facing == CameraCharacteristics.LENS_FACING_BACK }
            .sortedBy { it.focalLengthMm }
        return if (rawOnly) rear.filter { it.hasRawSensor } else rear
    }

    fun getLensesForFacing(facing: Int): List<LensInfo> {
        return if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
            lensOrganization?.front?.mapNotNull { profile ->
                _lenses.firstOrNull { it.cameraId == profile.id }
            } ?: _lenses.filter { it.facing == CameraCharacteristics.LENS_FACING_FRONT }
        } else {
            getRearLenses()
        }
    }

    fun hasAnyRawLens(facing: Int): Boolean {
        return getLensesForFacing(facing).any { it.hasRawSensor }
    }

    fun hasAnyRawLens(): Boolean {
        return _lenses.any { it.hasRawSensor }
    }

    fun getClosestRawLens(referenceLens: LensInfo): LensInfo? {
        val rawLenses = _lenses.filter { it.hasRawSensor && it.facing == referenceLens.facing }
        if (rawLenses.isEmpty()) return null
        return rawLenses.minByOrNull { kotlin.math.abs(it.focalLengthMm - referenceLens.focalLengthMm) }
    }

    fun getSensorActiveArraySize(lens: LensInfo): Rect {
        return try {
            val chars = cameraManager.getCameraCharacteristics(lens.cameraId)
            chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                ?: Rect(0, 0, 1, 1)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get active array size for ${lens.cameraId}: ${e.message}")
            Rect(0, 0, 1, 1)
        }
    }

    fun getSensorOrientation(lens: LensInfo): Int {
        return try {
            val chars = cameraManager.getCameraCharacteristics(lens.cameraId)
            chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get sensor orientation for ${lens.cameraId}: ${e.message}")
            0
        }
    }

    fun getLensLabel(lensId: String): String = lensLabels[lensId] ?: "Unknown"

    fun canLensTapToFocus(lensId: String): Boolean {
        return getLensProfile(lensId)?.canTapToAdjust() == true
    }

    fun getLensProfile(lensId: String): LensProfile? {
        return lensOrganization?.allBack?.firstOrNull { it.id == lensId }
            ?: lensOrganization?.front?.firstOrNull { it.id == lensId }
    }

    fun getPreviewSizes(lens: LensInfo): Array<Size> {
        return try {
            val chars = cameraManager.getCameraCharacteristics(lens.cameraId)
            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return emptyArray()
            map.getOutputSizes(android.graphics.ImageFormat.YUV_420_888)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get preview sizes for ${lens.cameraId}: ${e.message}")
            emptyArray()
        }
    }

    fun getBestPreviewSize(lens: LensInfo, maxWidth: Int = 1280, maxHeight: Int = 720): Size {
        val sizes = getPreviewSizes(lens)
        if (sizes.isEmpty()) return Size(640, 480)

        val activeArray = getSensorActiveArraySize(lens)
        val sensorAspect = activeArray.width().toFloat() / activeArray.height()

        return sizes
            .filter { it.width <= maxWidth && it.height <= maxHeight }
            .minByOrNull { size ->
                val sizeAspect = size.width.toFloat() / size.height
                kotlin.math.abs(sizeAspect - sensorAspect)
            }
            ?: sizes.lastOrNull()
            ?: Size(640, 480)
    }

    fun getPreviewSizeForAspectRatio(lens: LensInfo, targetAspect: Float, maxWidth: Int, maxHeight: Int): Size {
        val sizes = getPreviewSizes(lens)
        if (sizes.isEmpty()) return Size(640, 480)

        return sizes
            .filter { it.width <= maxWidth && it.height <= maxHeight }
            .minByOrNull { size ->
                val sizeAspect = size.width.toFloat() / size.height
                kotlin.math.abs(sizeAspect - targetAspect)
            }
            ?: getBestPreviewSize(lens, maxWidth, maxHeight)
    }

    data class ResolutionOption(
        val label: String,
        val aspectW: Int,
        val aspectH: Int,
        val width: Int,
        val height: Int
    )

    fun getResolutionOptions(lens: LensInfo): List<ResolutionOption> {
        val jpegSizes = lens.jpegOutputSizes
        if (jpegSizes.isEmpty()) return listOf(ResolutionOption("Full Sensor", 0, 0, 0, 0))

        val options = mutableListOf<ResolutionOption>()
        options.add(ResolutionOption("Full Sensor", 0, 0, 0, 0))

        val grouped = mutableMapOf<String, MutableList<android.util.Size>>()
        for (size in jpegSizes) {
            val gcd = gcd(size.width, size.height)
            val key = "${size.width / gcd}:${size.height / gcd}"
            grouped.getOrPut(key) { mutableListOf() }.add(size)
        }

        for ((ratio, sizes) in grouped) {
            val sortedSizes = sizes.sortedByDescending { it.width.toLong() * it.height.toLong() }
            val parts = ratio.split(":")
            val aw = parts[0].toIntOrNull() ?: continue
            val ah = parts[1].toIntOrNull() ?: continue
            for (size in sortedSizes) {
                val mp = size.width.toLong() * size.height / 1_000_000.0
                options.add(ResolutionOption(
                    "$ratio (${size.width}x${size.height}, ${String.format("%.1f", mp)}MP)",
                    aw, ah, size.width, size.height
                ))
            }
        }

        return options
    }

    private fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

    fun getCharacteristicsForLens(lens: LensInfo): CameraCharacteristics? {
        return try {
            cameraManager.getCameraCharacteristics(lens.cameraId)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get characteristics for ${lens.cameraId}: ${e.message}")
            null
        }
    }

    fun getCameraManager(): CameraManager = cameraManager

    private fun persistState(lensId: String, state: LensState) {
        try {
            val json = JSONObject().apply {
                put("zoomFactor", state.zoomFactor.toDouble())
                put("zoomCenterX", state.zoomCenterX.toDouble())
                put("zoomCenterY", state.zoomCenterY.toDouble())
                put("wbModeOrdinal", state.wbModeOrdinal)
                put("kelvin", state.kelvin.toDouble())
                put("kelvinTint", state.kelvinTint.toDouble())
                put("flashModeOrdinal", state.flashModeOrdinal)
            }
            getLensStateFile(lensId).writeText(json.toString())
        } catch (e: Exception) {
            Log.w(TAG, "Failed to persist lens state for $lensId", e)
        }
    }

    private fun loadState(lensId: String): LensState {
        return try {
            val file = getLensStateFile(lensId)
            if (!file.exists()) return LensState()
            val json = JSONObject(file.readText())
            LensState(
                zoomFactor = json.optDouble("zoomFactor", 1.0).toFloat(),
                zoomCenterX = json.optDouble("zoomCenterX", 0.5).toFloat(),
                zoomCenterY = json.optDouble("zoomCenterY", 0.5).toFloat(),
                wbModeOrdinal = json.optInt("wbModeOrdinal", 0),
                kelvin = json.optDouble("kelvin", 6300.0).toFloat(),
                kelvinTint = json.optDouble("kelvinTint", -14.0).toFloat(),
                flashModeOrdinal = json.optInt("flashModeOrdinal", 0)
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load lens state for $lensId", e)
            LensState()
        }
    }

    private fun getLensStateFile(lensId: String): File =
        File(context.filesDir, "lens_state_${lensId}.json")

    companion object {
        private const val TAG = "LensManager"
    }
}
