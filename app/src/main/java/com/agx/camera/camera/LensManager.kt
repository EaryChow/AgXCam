package com.agx.camera.camera

import android.content.Context
import android.graphics.Rect
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Log
import android.util.Size

data class LensInfo(
    val cameraId: String,
    val facing: Int,
    val focalLengthMm: Float,
    val hasRawSensor: Boolean,
    val hardwareLevel: Int,
    val label: String
)

class LensManager(context: Context) {
    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val _lenses = mutableListOf<LensInfo>()
    val lenses: List<LensInfo> get() = _lenses

    var activeLens: LensInfo? = null
        private set

    var listener: ((LensInfo) -> Unit)? = null

    fun enumerate(): Boolean {
        _lenses.clear()
        var hasSupportedLens = false

        for (id in cameraManager.cameraIdList) {
            val chars = cameraManager.getCameraCharacteristics(id)
            val facing = chars.get(CameraCharacteristics.LENS_FACING) ?: continue
            val level = chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
                ?: CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY

            if (level < CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL) {
                Log.w(TAG, "Camera $id hardware level $level < FULL, skipping")
                continue
            }

            val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val focal = focalLengths?.firstOrNull() ?: 0.0f
            val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()
            val hasRaw = caps.contains(17) // REQUEST_AVAILABLE_CAPABILITIES_RAW
            val label = when (facing) {
                CameraCharacteristics.LENS_FACING_BACK -> "Wide"
                CameraCharacteristics.LENS_FACING_FRONT -> "Front"
                else -> "Unknown"
            }
            _lenses.add(LensInfo(id, facing, focal, hasRaw, level, label))
            hasSupportedLens = true
        }

        Log.d(TAG, "Enumerated ${_lenses.size} supported lenses")
        for (lens in _lenses) {
            Log.d(TAG, "  ${lens.cameraId}: ${lens.label} ${lens.focalLengthMm}mm raw=${lens.hasRawSensor} level=${lens.hardwareLevel}")
        }

        return hasSupportedLens
    }

    fun selectPrimary(): LensInfo? {
        val rear = _lenses.filter { it.facing == CameraCharacteristics.LENS_FACING_BACK }
        val primary = rear.minByOrNull { it.focalLengthMm } ?: rear.firstOrNull()
        if (primary != null) {
            activeLens = primary
            Log.d(TAG, "Selected primary lens: ${primary.cameraId} ${primary.label}")
        }
        return primary
    }

    fun selectLens(lens: LensInfo) {
        activeLens = lens
        Log.d(TAG, "Selected lens: ${lens.cameraId} ${lens.label}")
        listener?.invoke(lens)
    }

    fun getSensorActiveArraySize(lens: LensInfo): Rect {
        val chars = cameraManager.getCameraCharacteristics(lens.cameraId)
        return chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            ?: Rect(0, 0, 1, 1)
    }

    fun getSensorOrientation(lens: LensInfo): Int {
        val chars = cameraManager.getCameraCharacteristics(lens.cameraId)
        return chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
    }

    fun getPreviewSizes(lens: LensInfo): Array<Size> {
        val chars = cameraManager.getCameraCharacteristics(lens.cameraId)
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return emptyArray()
        return map.getOutputSizes(android.graphics.ImageFormat.YUV_420_888)
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

    fun getCharacteristicsForLens(lens: LensInfo): CameraCharacteristics =
        cameraManager.getCameraCharacteristics(lens.cameraId)

    fun getCameraManager(): CameraManager = cameraManager

    companion object {
        private const val TAG = "LensManager"
    }
}
