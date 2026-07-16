package com.agx.camera.camera

import android.content.Context
import android.graphics.Rect
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
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
    val label: String
)

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

        val logicalMultiCameraId = findLogicalMultiCamera()

        for (id in cameraIds) {
            val chars = try {
                cameraManager.getCameraCharacteristics(id)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to get characteristics for camera $id: ${e.message}")
                continue
            }

            val facing = chars.get(CameraCharacteristics.LENS_FACING) ?: continue
            val level = chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
                ?: CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY

            if (level < CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED) {
                Log.w(TAG, "Camera $id hardware level $level < LIMITED, skipping")
                continue
            }

            val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val focal = focalLengths?.firstOrNull() ?: 0.0f
            val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()
            val hasRaw = caps.contains(17) // REQUEST_AVAILABLE_CAPABILITIES_RAW

            _lenses.add(LensInfo(id, facing, focal, hasRaw, level, ""))
            hasSupportedLens = true
        }

        assignLabels(logicalMultiCameraId)

        CrashLogger.log(TAG, "enumerate: found ${_lenses.size} lenses, hasSupported=$hasSupportedLens")
        for (lens in _lenses) {
            Log.d(TAG, "  ${lens.cameraId}: ${lens.label} ${lens.focalLengthMm}mm raw=${lens.hasRawSensor} level=${lens.hardwareLevel}")
        }

        return hasSupportedLens
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
        val rear = _lenses.filter { it.facing == CameraCharacteristics.LENS_FACING_BACK }
        val primary = rear.minByOrNull { it.focalLengthMm } ?: rear.firstOrNull()
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

    fun getFrontLens(): LensInfo? =
        _lenses.firstOrNull { it.facing == CameraCharacteristics.LENS_FACING_FRONT }

    fun getRearLenses(): List<LensInfo> =
        _lenses.filter { it.facing == CameraCharacteristics.LENS_FACING_BACK }
            .sortedBy { it.focalLengthMm }

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
                kelvin = json.optDouble("kelvin", 5500.0).toFloat(),
                kelvinTint = json.optDouble("kelvinTint", 0.0).toFloat(),
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
