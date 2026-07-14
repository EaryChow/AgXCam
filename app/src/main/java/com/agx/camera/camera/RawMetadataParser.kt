package com.agx.camera.camera

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.graphics.Matrix
import android.util.Log
import android.util.Rational
import android.util.Size

class RawMetadataParser(
    private val characteristics: CameraCharacteristics
) {

    val sensorWidth: Int
    val sensorHeight: Int
    val whiteLevel: Int
    val blackLevelPattern: IntArray
    val bayerPattern: BayerPattern
    val sensorOrientation: Int

    val blackLevelAverage: Float

    val bayerColorMap: IntArray

    val bitDepth: Int
    val maxSensorDimension: Int

    val calibrationTransform1: FloatArray
    val calibrationTransform2: FloatArray

    init {
        val activeArray = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            ?: throw IllegalStateException("SENSOR_INFO_ACTIVE_ARRAY_SIZE unavailable")
        sensorWidth = activeArray.width()
        sensorHeight = activeArray.height()
        maxSensorDimension = maxOf(sensorWidth, sensorHeight)

        whiteLevel = characteristics.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 1023

        val pattern = characteristics.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)
            ?: throw IllegalStateException("SENSOR_BLACK_LEVEL_PATTERN unavailable")
        // phase = (x%2)+(y%2)*2 maps directly to spatial indices [TL,TR,BL,BR],
        // matching the Android API's row-major order. No CFA reordering needed (§6).
        blackLevelPattern = IntArray(4) { pattern.getOffsetForIndex(it % 2, it / 2) }

        val cfa = characteristics.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT)
            ?: throw IllegalStateException("SENSOR_INFO_COLOR_FILTER_ARRANGEMENT unavailable")
        bayerPattern = when (cfa) {
            CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB -> BayerPattern.RGGB
            CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GRBG -> BayerPattern.GRBG
            CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GBRG -> BayerPattern.GBRG
            CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_BGGR -> BayerPattern.BGGR
            else -> throw IllegalStateException("Unsupported CFA: $cfa")
        }

        sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0

        blackLevelAverage = blackLevelPattern.average().toFloat()

        bayerColorMap = bayerPattern.colorMap()

        bitDepth = inferBitDepth(whiteLevel)

        calibrationTransform1 = parseCalibrationTransform(
            characteristics, "SENSOR_CALIBRATION_TRANSFORM1", 1
        )
        calibrationTransform2 = parseCalibrationTransform(
            characteristics, "SENSOR_CALIBRATION_TRANSFORM2", 2
        )

        Log.d(TAG, "RawMetadata: ${sensorWidth}x${sensorHeight}, white=$whiteLevel, " +
                "pattern=$bayerPattern, bitDepth=$bitDepth, " +
                "blackLevels=[${blackLevelPattern.joinToString()}], " +
                "avg=${String.format("%.1f", blackLevelAverage)}, " +
                "cal1 available=${!calibrationTransform1.contentEquals(FLOAT_IDENTITY_9)}, " +
                "cal2 available=${!calibrationTransform2.contentEquals(FLOAT_IDENTITY_9)}")
    }

    fun parseLensShadingMap(result: CaptureResult): LensShadingData? {
        val map = try {
            val key = CaptureResult::class.java.getField("STATISTICS_LENS_SHADING_MAP")
                .get(null) as android.hardware.camera2.CaptureResult.Key<*>
            result.get(key) as? android.hardware.camera2.params.LensShadingMap
        } catch (e: Exception) {
            Log.w(TAG, "STATISTICS_LENS_SHADING_MAP not available: ${e.message}")
            null
        } ?: return null

        val mapSize = try {
            val key = CameraCharacteristics::class.java.getField("LENS_INFO_SHADING_MAP_SIZE")
                .get(null) as android.hardware.camera2.CameraCharacteristics.Key<*>
            characteristics.get(key) as? android.util.Size
        } catch (e: Exception) {
            Log.w(TAG, "LENS_INFO_SHADING_MAP_SIZE not available: ${e.message}")
            null
        } ?: return null

        if (mapSize.width <= 0 || mapSize.height <= 0) return null

        val rGains = Array(mapSize.height) { FloatArray(mapSize.width) }
        val grGains = Array(mapSize.height) { FloatArray(mapSize.width) }
        val gbGains = Array(mapSize.height) { FloatArray(mapSize.width) }
        val bGains = Array(mapSize.height) { FloatArray(mapSize.width) }

        for (row in 0 until mapSize.height) {
            for (col in 0 until mapSize.width) {
                rGains[row][col] = map.getGainFactor(0, row, col)
                grGains[row][col] = map.getGainFactor(1, row, col)
                gbGains[row][col] = map.getGainFactor(2, row, col)
                bGains[row][col] = map.getGainFactor(3, row, col)
            }
        }

        val allIdentity = rGains.all { row -> row.all { Math.abs(it - 1.0f) < 0.001f } } &&
                grGains.all { row -> row.all { Math.abs(it - 1.0f) < 0.001f } } &&
                gbGains.all { row -> row.all { Math.abs(it - 1.0f) < 0.001f } } &&
                bGains.all { row -> row.all { Math.abs(it - 1.0f) < 0.001f } }

        return LensShadingData(
            rGains = rGains,
            grGains = grGains,
            gbGains = gbGains,
            bGains = bGains,
            width = mapSize.width,
            height = mapSize.height,
            available = !allIdentity
        )
    }

    fun hasRawSensorCapability(): Boolean {
        return try {
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            map?.getOutputSizes(android.graphics.ImageFormat.RAW_SENSOR)?.isNotEmpty() == true
        } catch (e: Exception) {
            Log.w(TAG, "RAW_SENSOR capability check failed: ${e.message}")
            false
        }
    }

    companion object {
        private const val TAG = "RawMetadataParser"

        private val FLOAT_IDENTITY_9 = floatArrayOf(
            1f, 0f, 0f,
            0f, 1f, 0f,
            0f, 0f, 1f
        )

        private fun inferBitDepth(whiteLevel: Int): Int {
            return when {
                whiteLevel <= 1023 -> 10
                whiteLevel <= 4095 -> 12
                whiteLevel <= 16383 -> 14
                else -> 16
            }
        }

        private fun parseCalibrationTransform(
            chars: CameraCharacteristics,
            fieldName: String,
            index: Int
        ): FloatArray {
            return try {
                val keyField = CameraCharacteristics::class.java.getField(fieldName)
                @Suppress("UNCHECKED_CAST")
                val key = keyField.get(null) as CameraCharacteristics.Key<android.hardware.camera2.params.ColorSpaceTransform>
                val cst = chars.get(key) ?: return FLOAT_IDENTITY_9.clone()
                val matrix = FloatArray(9)
                for (row in 0 until 3) {
                    for (col in 0 until 3) {
                        val r = cst.getElement(col, row)
                        val denom = r.denominator
                        if (denom == 0) {
                            Log.w(TAG, "SENSOR_CALIBRATION_TRANSFORM$index zero denominator at [$row][$col], using identity")
                            return FLOAT_IDENTITY_9.clone()
                        }
                        matrix[row * 3 + col] = r.numerator.toFloat() / denom.toFloat()
                    }
                }
                Log.d(TAG, "Loaded SENSOR_CALIBRATION_TRANSFORM$index: " +
                        "[${matrix.joinToString(", ") { String.format("%.4f", it) }}]")
                matrix
            } catch (e: Exception) {
                Log.w(TAG, "SENSOR_CALIBRATION_TRANSFORM$index not available: ${e.message}")
                FLOAT_IDENTITY_9.clone()
            }
        }
    }
}

enum class BayerPattern(val label: String) {
    RGGB("bayer_rggb"),
    GRBG("bayer_grbg"),
    GBRG("bayer_gbrg"),
    BGGR("bayer_bggr");

    fun phaseIndex(x: Int, y: Int): Int = (x % 2) + (y % 2) * 2

    fun colorMap(): IntArray = when (this) {
        RGGB -> intArrayOf(0, 1, 1, 2)
        GRBG -> intArrayOf(1, 0, 2, 1)
        GBRG -> intArrayOf(1, 2, 0, 1)
        BGGR -> intArrayOf(2, 1, 1, 0)
    }

    companion object {
        fun fromString(s: String): BayerPattern = when (s) {
            "bayer_rggb" -> RGGB
            "bayer_grbg" -> GRBG
            "bayer_gbrg" -> GBRG
            "bayer_bggr" -> BGGR
            else -> throw IllegalArgumentException("Unknown Bayer pattern: $s")
        }

        fun fromAndroidConstant(cfa: Int): BayerPattern = when (cfa) {
            0 -> RGGB
            1 -> GRBG
            2 -> GBRG
            3 -> BGGR
            else -> throw IllegalArgumentException("Unknown CFA constant: $cfa")
        }
    }
}

data class LensShadingData(
    val rGains: Array<FloatArray>,
    val grGains: Array<FloatArray>,
    val gbGains: Array<FloatArray>,
    val bGains: Array<FloatArray>,
    val width: Int,
    val height: Int,
    val available: Boolean
) {
    fun toRgba16fFlipped(bayerPattern: BayerPattern): ShortArray {
        val perm = channelPermutation(bayerPattern)
        val pixels = ShortArray(width * height * 4)
        for (row in 0 until height) {
            val flippedRow = height - 1 - row
            for (col in 0 until width) {
                val srcIdx = row * width + col
                val dstIdx = (flippedRow * width + col) * 4
                val gains = arrayOf(rGains, grGains, gbGains, bGains)
                for (ch in 0 until 4) {
                    pixels[dstIdx + ch] = floatToHalf(gains[perm[ch]][row][col])
                }
            }
        }
        return pixels
    }

    private fun channelPermutation(pattern: BayerPattern): IntArray = when (pattern) {
        BayerPattern.RGGB -> intArrayOf(0, 1, 2, 3)
        BayerPattern.GRBG -> intArrayOf(1, 0, 3, 2)
        BayerPattern.GBRG -> intArrayOf(1, 3, 0, 2)
        BayerPattern.BGGR -> intArrayOf(3, 2, 1, 0)
    }

    private fun floatToHalf(f: Float): Short {
        val bits = java.lang.Float.floatToRawIntBits(f)
        val sign = (bits ushr 16) and 0x8000
        var exp = ((bits ushr 23) and 0xFF) - 127 + 15
        var mantissa = bits and 0x7FFFFF
        if (exp <= 0) {
            exp = 0
            mantissa = 0
        } else if (exp >= 31) {
            exp = 31
            mantissa = 0
        } else {
            mantissa = mantissa shr 13
        }
        return (sign or (exp shl 10) or mantissa).toShort()
    }
}
