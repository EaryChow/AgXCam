package com.agx.camera.camera

import android.hardware.camera2.CameraCharacteristics
import android.util.Log
import com.agx.camera.color.ColorMatrix
import com.agx.camera.color.ColorMatrix.Mat3
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Builds a camera-RGB -> linear-sRGB matrix from the camera2 RAW sensor
 * profile. The returned matrix is applied row-major as
 *   out[r] = SUM_c M[r][c] * in[c]   (column-vector convention).
 *
 * Pipeline (all standard CIE 1931 / DNG-style camera-profile math):
 *   1. take the as-shot neutral (sensor RGB of a neutral scene patch),
 *      normalize it, and find the scene-white chromaticity that maps to it
 *      through the interpolated XYZ->camera matrix (fixed-point solve).
 *   2. corridor not needed: the chromaticity is converted to a correlated
 *      color temperature using the CIE Robertson table (Wyszecki & Stiles,
 *      "Color Science", 2nd ed., p. 228).
 *   3. interpolate SENSOR_COLOR_TRANSFORM1/2, SENSOR_CALIBRATION_TRANSFORM1/2
 *      and SENSOR_FORWARD_MATRIX1/2 at that temperature with a reciprocal-
 *      temperature weight between the two reference illuminants.
 *   4. adapt the XYZ frame from D50 (PCS) to the scene white using a linear
 *      Bradford von-Kries transform.
 *   5. derive the camera-space white from ColorMatrix * XYZ(scene white),
 *      normalize it, and build the camera->XYZ map either through the forward
 *      matrices (normalized so FWD maps the reference neutral to the PCS white,
 *      then FORWARD_MATRIX * diag(1/refWhite) * inv(CALIBRATION)) or, when no
 *      forward matrices are reported, by inverting the scaled PCS->camera
 *      matrix.
 *   6. convert PCS (D50) XYZ to linear sRGB by inverting the standard
 *      sRGB primaries matrix expressed in the PCS (sRGB->XYZ(D50), Bradford
 *      adapted), with a per-row white-point scale to pin the PCS white.
 *
 * The result makes the physical as-shot neutral map to sRGB white: scene
 * whites become numerically equal after the matrix, independent of the
 * sensor's channel sensitivities or the capture illuminant. No HAL color
 * correction state (COLOR_CORRECTION_TRANSFORM / COLOR_CORRECTION_GAINS) is
 * used -- the profile is static per camera and the only frame-varying input
 * is the as-shot neutral.
 */
class RawColorProfile(chars: CameraCharacteristics) {

    private val colorMatrix1: Mat3
    private val colorMatrix2: Mat3
    private val calibration1: Mat3
    private val calibration2: Mat3
    private val forwardMatrix1: Mat3?
    private val forwardMatrix2: Mat3?

    /** Correlated color temperature of reference illuminant 1 (Kelvin). */
    val colorTemperature1: Float

    /** Correlated color temperature of reference illuminant 2 (Kelvin). */
    val colorTemperature2: Float

    /** True when a usable XYZ->camera transform was found in characteristics. */
    val available: Boolean

    /** True when at least one forward camera->XYZ matrix was reported. */
    val hasForwardMatrix: Boolean

    init {
        val cm1 = readMatrix(chars, "SENSOR_COLOR_TRANSFORM1")
        val cm2 = readMatrix(chars, "SENSOR_COLOR_TRANSFORM2")
        val cal1 = readMatrix(chars, "SENSOR_CALIBRATION_TRANSFORM1")
        val cal2 = readMatrix(chars, "SENSOR_CALIBRATION_TRANSFORM2")
        val fm1 = readMatrix(chars, "SENSOR_FORWARD_MATRIX1")
        val fm2 = readMatrix(chars, "SENSOR_FORWARD_MATRIX2")

        colorMatrix1 = cm1 ?: Mat3()
        colorMatrix2 = cm2 ?: Mat3()
        calibration1 = cal1 ?: Mat3()
        calibration2 = cal2 ?: Mat3()
        forwardMatrix1 = fm1
        forwardMatrix2 = fm2

        available = cm1 != null || cm2 != null
        hasForwardMatrix = fm1 != null || fm2 != null

        var t1 = readReferenceIlluminantKelvin(chars, "SENSOR_REFERENCE_ILLUMINANT1") ?: 2856f
        var t2 = readReferenceIlluminantKelvin(chars, "SENSOR_REFERENCE_ILLUMINANT2") ?: 6504f
        if (t1 > t2) {
            val swap = t1
            t1 = t2
            t2 = swap
        }
        colorTemperature1 = t1
        colorTemperature2 = t2

        Log.d(
            TAG, "RawColorProfile: available=$available fwd=$hasForwardMatrix " +
                "illuminants=${t1.toInt()}K/${t2.toInt()}K " +
                "cm1=${matStr(cm1)}" +
                "cal1=${matStr(cal1)}" +
                "fwd1=${matStr(fm1)}"
        )
    }

    /**
     * Correlated color temperature of the scene described by [neutralSensorRgb]
     * (sensor RGB values of a neutral patch, any scale). Null when the neutral
     * is degenerate.
     */
    fun temperatureForNeutral(neutralSensorRgb: FloatArray): Float? {
        val neutral = RawColorMath.normalize(neutralSensorRgb) ?: return null
        val xy = RawColorMath.sceneWhiteXy(neutral, colorMatrix1, colorMatrix2, colorTemperature1, colorTemperature2) ?: return null
        return RawColorMath.xyToTemperature(xy)
    }

    /**
     * Returns the camera-RGB -> linear-sRGB matrix (row-major) for the given
     * as-shot neutral, or null when the profile has no usable transform.
     */
    fun srgbMatrixForNeutral(neutralSensorRgb: FloatArray): Mat3? {
        if (!available) return null
        return RawColorMath.srgbMatrix(
            colorMatrix1, colorMatrix2,
            calibration1, calibration2,
            forwardMatrix1, forwardMatrix2,
            colorTemperature1, colorTemperature2,
            neutralSensorRgb
        )
    }

    /**
     * White-balance gains, the WB-removed camera-native -> linear-sRGB matrix,
     * and camera-native luminance coefficients for the given as-shot neutral.
     * Null when the profile has no usable transform.
     */
    fun neutralTransformForNeutral(neutralSensorRgb: FloatArray): RawColorMath.NeutralTransform? {
        if (!available) return null
        return RawColorMath.neutralTransform(
            colorMatrix1, colorMatrix2,
            calibration1, calibration2,
            forwardMatrix1, forwardMatrix2,
            colorTemperature1, colorTemperature2,
            neutralSensorRgb
        )
    }

    companion object {
        private const val TAG = "RawColorProfile"

        private fun matStr(m: Mat3?): String =
            if (m == null) "none"
            else String.format("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f", m.m[0], m.m[1], m.m[2], m.m[3], m.m[4], m.m[5], m.m[6], m.m[7], m.m[8])

        private fun readMatrix(chars: CameraCharacteristics, keyField: String): Mat3? {
            return try {
                val field = CameraCharacteristics::class.java.getField(keyField)
                @Suppress("UNCHECKED_CAST")
                val key = field.get(null) as CameraCharacteristics.Key<android.hardware.camera2.params.ColorSpaceTransform>
                val cst = chars.get(key) ?: return null
                val m = FloatArray(9)
                var degenerate = false
                for (row in 0 until 3) {
                    for (col in 0 until 3) {
                        val r = cst.getElement(col, row)
                        if (r.denominator == 0) {
                            degenerate = true
                            break
                        }
                        m[row * 3 + col] = r.numerator.toFloat() / r.denominator.toFloat()
                    }
                }
                if (degenerate) {
                    Log.w(TAG, "$keyField has a zero denominator, ignoring")
                    return null
                }
                val mat = Mat3(m)
                if (abs(ColorMatrix.determinant(mat)) < 1e-6f) {
                    Log.w(TAG, "$keyField is degenerate, ignoring")
                    return null
                }
                mat
            } catch (e: Exception) {
                Log.w(TAG, "$keyField not available: ${e.message}")
                null
            }
        }

        private fun readReferenceIlluminantKelvin(chars: CameraCharacteristics, keyField: String): Float? {
            return try {
                val field = CameraCharacteristics::class.java.getField(keyField)
                @Suppress("UNCHECKED_CAST")
                val key = field.get(null) as CameraCharacteristics.Key<Byte>
                val v = chars.get(key)?.toInt() ?: return null
                standardIlluminantKelvin(v)
            } catch (e: Exception) {
                Log.w(TAG, "$keyField not available: ${e.message}")
                null
            }
        }

        // Correlated color temperatures of the CIE standard illuminants used by
        // SENSOR_REFERENCE_ILLUMINANT1/2 (values not listed fall back to D65).
        private fun standardIlluminantKelvin(v: Int): Float = when (v) {
            3, 17 -> 2856f   // tungsten / standard illuminant A
            18 -> 4874f      // standard illuminant B
            19 -> 6774f      // standard illuminant C
            4 -> 5500f       // flash
            20 -> 5503f      // D55
            22 -> 7504f      // D75
            23 -> 5003f      // D50
            else -> 6504f    // daylight / fine weather / cloud / shade / D65
        }
    }
}

/**
 * Pure matrix math behind [RawColorProfile]; unit-testable without a camera.
 */
object RawColorMath {

    private const val MAX_NEUTRAL_ITERS = 30
    private const val NEUTRAL_EPS = 1e-7

    // CIE 1931 D50 reference white (PCS white).
    private val D50_XY = floatArrayOf(0.3457f, 0.3585f)
    private val D50_XYZ = floatArrayOf(0.9642f, 1.0f, 0.8249f)
    private val ONE = floatArrayOf(1f, 1f, 1f)

    // Standard sRGB (IEC 61966-2-1) primaries expressed in the PCS (D50):
    // the linear-Bradford-adapted sRGB->XYZ(D50) matrix. Row sums are the
    // PCS white because sRGB primaries red/green/blue map to PCS white when
    // summed; [pcsToSrgb] re-normalizes them to the exact PCS white below.
    private val SRGB_TO_XYZ = Mat3(
        floatArrayOf(
            0.4360747f, 0.3850649f, 0.1430804f,
            0.2225045f, 0.7168786f, 0.0606169f,
            0.0139322f, 0.0971045f, 0.7141733f
        )
    )

    // Linear Bradford cone-response matrix.
    private val BRADFORD = Mat3(
        floatArrayOf(
            0.8951f, 0.2664f, -0.1614f,
            -0.7502f, 1.7135f, 0.0367f,
            0.0389f, -0.0685f, 1.0296f
        )
    )
    private val BRADFORD_INV = ColorMatrix.inverse(BRADFORD)

    /**
     * CIE Robertson iso-temperature table (Wyszecki & Stiles, "Color Science",
     * 2nd ed., page 228): for each anchor, r = 1e6 / temperature (K),
     * u/v = CIE 1960 uv coordinates, t = line slope in uv space.
     */
    private val ROBERTSON = arrayOf(
        doubleArrayOf(0.0, 0.18006, 0.26352, -0.24341),
        doubleArrayOf(10.0, 0.18066, 0.26589, -0.25479),
        doubleArrayOf(20.0, 0.18133, 0.26846, -0.26876),
        doubleArrayOf(30.0, 0.18208, 0.27119, -0.28539),
        doubleArrayOf(40.0, 0.18293, 0.27407, -0.30470),
        doubleArrayOf(50.0, 0.18388, 0.27709, -0.32675),
        doubleArrayOf(60.0, 0.18494, 0.28021, -0.35156),
        doubleArrayOf(70.0, 0.18611, 0.28342, -0.37915),
        doubleArrayOf(80.0, 0.18740, 0.28668, -0.40955),
        doubleArrayOf(90.0, 0.18880, 0.28997, -0.44278),
        doubleArrayOf(100.0, 0.19032, 0.29326, -0.47888),
        doubleArrayOf(125.0, 0.19462, 0.30141, -0.58204),
        doubleArrayOf(150.0, 0.19962, 0.30921, -0.70471),
        doubleArrayOf(175.0, 0.20525, 0.31647, -0.84901),
        doubleArrayOf(200.0, 0.21142, 0.32312, -1.0182),
        doubleArrayOf(225.0, 0.21807, 0.32909, -1.2168),
        doubleArrayOf(250.0, 0.22511, 0.33439, -1.4512),
        doubleArrayOf(275.0, 0.23247, 0.33904, -1.7298),
        doubleArrayOf(300.0, 0.24010, 0.34308, -2.0637),
        doubleArrayOf(325.0, 0.24702, 0.34655, -2.4681),
        doubleArrayOf(350.0, 0.25591, 0.34951, -2.9641),
        doubleArrayOf(375.0, 0.26400, 0.35200, -3.5814),
        doubleArrayOf(400.0, 0.27218, 0.35407, -4.3633),
        doubleArrayOf(425.0, 0.28039, 0.35577, -5.3762),
        doubleArrayOf(450.0, 0.28863, 0.35714, -6.7262),
        doubleArrayOf(475.0, 0.29685, 0.35823, -8.5955),
        doubleArrayOf(500.0, 0.30505, 0.35907, -11.324),
        doubleArrayOf(525.0, 0.31320, 0.35968, -15.628),
        doubleArrayOf(550.0, 0.32129, 0.36011, -23.325),
        doubleArrayOf(575.0, 0.32931, 0.36038, -40.770),
        doubleArrayOf(600.0, 0.33724, 0.36051, -116.45)
    )

    // The DNG forward matrices are defined in the PCS: a camera neutral under
    // the reference illuminant (mapped to [1,1,1] by the calibration step) must
    // land on the D50 PCS white. Vendors ship them with the reference
    // illuminant's own white instead (e.g. XYZ of illuminant A), so force
    // FWD * [1,1,1] == PCS white with a per-row scale, matching the reference
    // implementation. Without this the as-shot neutral renders at the reference
    // illuminant's chromaticity, adding a tint that tracks the warm/cool pair.
    fun normalizeForwardMatrix(m: Mat3): Mat3? {
        val rows = floatArrayOf(
            m.m[0] + m.m[1] + m.m[2],
            m.m[3] + m.m[4] + m.m[5],
            m.m[6] + m.m[7] + m.m[8]
        )
        if (!rows.all { it.isFinite() && abs(it) >= 1e-6f }) return null
        val s = ColorMatrix.diagonal(
            D50_XYZ[0] / rows[0],
            D50_XYZ[1] / rows[1],
            D50_XYZ[2] / rows[2]
        )
        return ColorMatrix.multiply(s, m)
    }

    /**
     * Camera-RGB -> linear-sRGB for the given as-shot neutral, with the white
     * balance folded in. See [RawColorProfile] for the pipeline description.
     * Returns null for a degenerate neutral.
     */
    fun srgbMatrix(
        colorMatrix1: Mat3,
        colorMatrix2: Mat3,
        calibration1: Mat3,
        calibration2: Mat3,
        forwardMatrix1: Mat3?,
        forwardMatrix2: Mat3?,
        temperature1: Float,
        temperature2: Float,
        neutralSensorRgb: FloatArray
    ): Mat3? = neutralCore(
        colorMatrix1, colorMatrix2, calibration1, calibration2,
        forwardMatrix1, forwardMatrix2, temperature1, temperature2,
        neutralSensorRgb
    )?.colorMatrix

    /**
     * White balance gains (green-normalized), the camera-native -> linear-sRGB
     * matrix with that white balance REMOVED, and the luminance (Y) row of the
     * camera-native -> XYZ map. Splitting the folded srgb matrix lets a
     * pipeline insert processing between the white-balance stage and the
     * native-to-sRGB color matrix. Applying [wbGains] first and then
     * [colorMatrix] reproduces [srgbMatrix] exactly. [lumaCoeffs] is the Y row
     * of the camera-native -> XYZ matrix, which dots with the white-balanced
     * camera-native signal to give CIE luminance of the gray axis.
     */
    data class NeutralTransform(
        val wbGains: FloatArray,
        val colorMatrix: Mat3,
        val lumaCoeffs: FloatArray
    )

    fun neutralTransform(
        colorMatrix1: Mat3,
        colorMatrix2: Mat3,
        calibration1: Mat3,
        calibration2: Mat3,
        forwardMatrix1: Mat3?,
        forwardMatrix2: Mat3?,
        temperature1: Float,
        temperature2: Float,
        neutralSensorRgb: FloatArray
    ): NeutralTransform? {
        val core = neutralCore(
            colorMatrix1, colorMatrix2, calibration1, calibration2,
            forwardMatrix1, forwardMatrix2, temperature1, temperature2,
            neutralSensorRgb
        ) ?: return null
        val neutral = core.neutral
        // Green-normalized gains that whiten the as-shot neutral in camera
        // native space; g * neutral == (1,1,1).
        val wbGains = floatArrayOf(neutral[1] / neutral[0], 1f, neutral[1] / neutral[2])
        // Undo those gains on the folded matrix column-wise so that
        // colorMatrix * diag(wbGains) == folded matrix.
        val gInv = ColorMatrix.diagonal(1f / wbGains[0], 1f, 1f / wbGains[2])
        return NeutralTransform(
            wbGains,
            ColorMatrix.multiply(core.colorMatrix, gInv),
            floatArrayOf(
                core.cameraToXyz.m[3],
                core.cameraToXyz.m[4],
                core.cameraToXyz.m[5]
            )
        )
    }

    private data class NeutralCore(
        val colorMatrix: Mat3,
        val cameraToXyz: Mat3,
        val neutral: FloatArray
    )

    private fun neutralCore(
        colorMatrix1: Mat3,
        colorMatrix2: Mat3,
        calibration1: Mat3,
        calibration2: Mat3,
        forwardMatrix1: Mat3?,
        forwardMatrix2: Mat3?,
        temperature1: Float,
        temperature2: Float,
        neutralSensorRgb: FloatArray
    ): NeutralCore? {
        val neutral = normalize(neutralSensorRgb) ?: return null

        val lowColor = normalizeColorMatrix(colorMatrix1)
        val highColor = normalizeColorMatrix(colorMatrix2)

        val xy = sceneWhiteXy(neutral, colorMatrix1, colorMatrix2, temperature1, temperature2) ?: return null
        val temp = xyToTemperature(xy)

        val cm = interpolate(temp, lowColor, highColor, temperature1, temperature2)
        val cal = interpolate(temp, calibration1, calibration2, temperature1, temperature2)
        // Enforce the PCS convention on the (possibly interpolated) forward
        // matrix: FWD * [1,1,1] must equal the D50 PCS white. Vendors ship
        // forward matrices in the reference-illuminant convention, so without
        // this the as-shot neutral renders at the reference illuminant's
        // chromaticity and overcasts the image.
        val fwd = interpolateOpt(temp, forwardMatrix1, forwardMatrix2, temperature1, temperature2)
            ?.let { normalizeForwardMatrix(it) }

        // Camera-space white for the scene chromaticity.
        val cameraWhite = ColorMatrix.mulMatVec(cm, xyToXyz(xy))
        val whiteScale = 1.0f / max(0f, max(cameraWhite[0], max(cameraWhite[1], cameraWhite[2])))
        val cw = floatArrayOf(
            clamp02(cameraWhite[0] * whiteScale),
            clamp02(cameraWhite[1] * whiteScale),
            clamp02(cameraWhite[2] * whiteScale)
        )

        // PCS (D50) -> camera, normalized so the PCS white maps to ~1.
        val adapted = ColorMatrix.multiply(cm, mapWhite(D50_XY, xy))
        val scale = max(
            0f,
            max(
                ColorMatrix.mulMatVec(adapted, D50_XYZ)[0],
                max(
                    ColorMatrix.mulMatVec(adapted, D50_XYZ)[1],
                    ColorMatrix.mulMatVec(adapted, D50_XYZ)[2]
                )
            )
        )
        if (scale <= 0f) return null
        val pcsToCamera = ColorMatrix.Mat3(
            adapted.m.map { it / scale }.toFloatArray()
        )

        val cameraToXyz: Mat3 = if (fwd != null) {
            val individualToReference = ColorMatrix.inverse(cal)
            val refCameraWhite = ColorMatrix.mulMatVec(individualToReference, cw)
            val d = ColorMatrix.diagonal(
                1.0f / max(1e-6f, refCameraWhite[0]),
                1.0f / max(1e-6f, refCameraWhite[1]),
                1.0f / max(1e-6f, refCameraWhite[2])
            )
            ColorMatrix.multiply(
                ColorMatrix.multiply(fwd, d),
                individualToReference
            )
        } else {
            ColorMatrix.inverse(pcsToCamera)
        }

        return NeutralCore(
            ColorMatrix.multiply(pcsToSrgb(), cameraToXyz),
            cameraToXyz,
            neutral
        )
    }

    /**
     * Scene-white chromaticity from an as-shot neutral. Solves xy such that
     * ColorMatrix(xy-interp) * XYZ(xy) == neutral by fixed-point iteration.
     */
    fun sceneWhiteXy(
        neutral: FloatArray,
        colorMatrix1: Mat3,
        colorMatrix2: Mat3,
        temperature1: Float,
        temperature2: Float
    ): FloatArray? {
        var lastXy = D50_XY.copyOf()
        for (i in 0 until MAX_NEUTRAL_ITERS) {
            val lowColor = normalizeColorMatrix(colorMatrix1)
            val highColor = normalizeColorMatrix(colorMatrix2)
            val interpColor = interpolate(
                xyToTemperature(lastXy), lowColor, highColor, temperature1, temperature2
            )
            val inv = ColorMatrix.inverse(interpColor)
            val xyz = ColorMatrix.mulMatVec(inv, neutral)
            val next = xyzToXy(xyz)

            if (abs(next[0] - lastXy[0]) + abs(next[1] - lastXy[1]) < NEUTRAL_EPS) {
                return next
            }
            if (i == MAX_NEUTRAL_ITERS - 1) {
                next[0] = (lastXy[0] + next[0]) * 0.5f
                next[1] = (lastXy[1] + next[1]) * 0.5f
            }
            lastXy = next
        }
        return null
    }

    /** CIE Robertson correlated color temperature (Kelvin) from chromaticity. */
    fun xyToTemperature(xy: FloatArray): Float {
        val denom = 1.5 - xy[0] + 6.0 * xy[1]
        if (denom <= 0.0) return D65_TEMP
        val u = 2.0 * xy[0] / denom
        val v = 3.0 * xy[1] / denom

        var lastDt = 0.0
        var lastDu = 0.0
        var lastDv = 0.0

        for (index in 1 until ROBERTSON.size) {
            var du = 1.0
            var dv = ROBERTSON[index][3]
            val len = sqrt(1.0 + dv * dv)
            du /= len
            dv /= len

            val uu = u - ROBERTSON[index][1]
            val vv = v - ROBERTSON[index][2]
            val dt = -uu * dv + vv * du

            if (dt <= 0.0 || index == ROBERTSON.size - 1) {
                var d = if (dt > 0.0) 0.0 else -dt
                val f = if (index == 1) 0.0 else d / (lastDt + d)
                val invTemp = ROBERTSON[index - 1][0] * f + ROBERTSON[index][0] * (1.0 - f)
                if (invTemp <= 0.0) return D65_TEMP
                return (1.0E6 / invTemp).toFloat()
            }
            lastDt = dt
            lastDu = du
            lastDv = dv
        }
        return D65_TEMP
    }

    /** RGB-preserving scale of an XYZ->camera matrix so its PCS white hits ~1. */
    private fun normalizeColorMatrix(m: Mat3): Mat3 {
        val coord = ColorMatrix.mulMatVec(m, D50_XYZ)
        val maxC = max(0f, max(coord[0], max(coord[1], coord[2])))
        if (maxC > 0.0f && (maxC < 0.99f || maxC > 1.01f)) {
            return ColorMatrix.Mat3(m.m.map { it / maxC }.toFloatArray())
        }
        return m
    }

    // Linear Bradford von-Kries adaptation that maps srcXy white into dstXy white.
    private fun mapWhite(srcXy: FloatArray, dstXy: FloatArray): Mat3 {
        val src = ColorMatrix.mulMatVec(BRADFORD, xyToXyz(srcXy))
        val dst = ColorMatrix.mulMatVec(BRADFORD, xyToXyz(dstXy))
        val ratio = ColorMatrix.diagonal(
            clampScale(dst[0], src[0]),
            clampScale(dst[1], src[1]),
            clampScale(dst[2], src[2])
        )
        return ColorMatrix.multiply(
            ColorMatrix.multiply(BRADFORD_INV, ratio),
            BRADFORD
        )
    }

    // PCS (D50) -> linear sRGB with white-point row scaling of the sRGB matrix.
    private fun pcsToSrgb(): Mat3 {
        val w1 = ColorMatrix.mulMatVec(SRGB_TO_XYZ, ONE)
        val w2 = D50_XY // for scale compute PCS white XYZ
        val pcsWhite = xyToXyz(w2)
        val s = ColorMatrix.diagonal(
            pcsWhite[0] / w1[0],
            pcsWhite[1] / w1[1],
            pcsWhite[2] / w1[2]
        )
        return ColorMatrix.inverse(ColorMatrix.multiply(s, SRGB_TO_XYZ))
    }

    private fun interpolate(
        temp: Float,
        m1: Mat3,
        m2: Mat3,
        t1: Float,
        t2: Float
    ): Mat3 {
        val g = weight(temp, t1, t2)
        return ColorMatrix.Mat3(
            FloatArray(9) { i -> g * m1.m[i] + (1.0f - g) * m2.m[i] }
        )
    }

    private fun interpolateOpt(
        temp: Float,
        a: Mat3?,
        b: Mat3?,
        t1: Float,
        t2: Float
    ): Mat3? {
        if (a == null && b == null) return null
        if (a == null) return b
        if (b == null) return a
        return interpolate(temp, a, b, t1, t2)
    }

    // Reciprocal-temperature weight: 1 at t1 (warm), 0 at t2 (cool).
    private fun weight(temp: Float, t1: Float, t2: Float): Float {
        if (temp <= t1) return 1.0f
        if (temp >= t2) return 0.0f
        val invT = 1.0f / temp
        val inv1 = 1.0f / max(t1, 1f)
        val inv2 = 1.0f / max(t2, 1f)
        return (invT - inv2) / (inv1 - inv2)
    }

    private fun clamp02(v: Float): Float = max(0.001f, min2(1.0f, v))

    private fun clampScale(n: Float, d: Float): Float {
        val ratio = if (d > 0.0f) n / d else 10.0f
        return max(0.1f, min2(10.0f, ratio))
    }

    private fun min2(a: Float, b: Float): Float = if (a < b) a else b

    fun normalize(v: FloatArray): FloatArray? {
        if (v.size < 3) return null
        for (i in 0..2) {
            val c = v[i]
            if (!c.isFinite() || c <= 0.0f) return null
        }
        val mx = v[0].coerceAtLeast(v[1]).coerceAtLeast(v[2])
        if (mx <= 0f) return null
        return floatArrayOf(v[0] / mx, v[1] / mx, v[2] / mx)
    }

    private fun xyToXyz(xy: FloatArray): FloatArray {
        val x = max(1e-6f, xy[0])
        val y = max(1e-6f, xy[1])
        val total = (x + y).coerceAtMost(0.999999f)
        val scale = if (x + y > 0.999999f) total / (x + y) else 1.0f
        val sx = x * scale
        val sy = y * scale
        return floatArrayOf(sx / sy, 1.0f, (1.0f - sx - sy) / sy)
    }

    private fun xyzToXy(xyz: FloatArray): FloatArray {
        val total = xyz[0] + xyz[1] + xyz[2]
        if (total <= 0.0f) return D50_XY.copyOf()
        return floatArrayOf(xyz[0] / total, xyz[1] / total)
    }

    private const val D65_TEMP = 6504f
}