package com.agx.camera.color

import kotlin.math.*

object WhiteBalanceMath {

    data class KelvinResult(
        val xy: Pair<Float, Float>,
        val sceneLinearTo709: ColorMatrix.Mat3
    )

    // CIE 1931 2-deg standard observer colour matching functions (x_bar, y_bar, z_bar),
    // sampled every 5 nm from 380 nm to 780 nm (81 entries). Table I(3.3.1) of
    // Wyszecki & Stiles, "Color Science" (2nd ed., 1982); values per CIE 018:2019.
    private val CIE1931_2DEG_XBAR = doubleArrayOf(
        0.0013680, 0.0022360, 0.0042430, 0.0076500, 0.0143100, 0.0231900, 0.0435100, 0.0776300,
        0.1343800, 0.2147700, 0.2839000, 0.3285000, 0.3482800, 0.3480600, 0.3362000, 0.3187000,
        0.2908000, 0.2511000, 0.1953600, 0.1421000, 0.0956400, 0.0579500, 0.0320100, 0.0147000,
        0.0049000, 0.0024000, 0.0093000, 0.0291000, 0.0632700, 0.1096000, 0.1655000, 0.2257500,
        0.2904000, 0.3597000, 0.4334500, 0.5120500, 0.5945000, 0.6784000, 0.7621000, 0.8425000,
        0.9163000, 0.9786000, 1.0263000, 1.0567000, 1.0622000, 1.0456000, 1.0026000, 0.9384000,
        0.8544500, 0.7514000, 0.6424000, 0.5419000, 0.4479000, 0.3608000, 0.2835000, 0.2187000,
        0.1649000, 0.1212000, 0.0874000, 0.0636000, 0.0467700, 0.0329000, 0.0227000, 0.0158400,
        0.0113600, 0.0081100, 0.0057900, 0.0041100, 0.0028990, 0.0020490, 0.0014400, 0.0010000,
        0.0006901, 0.0004760, 0.0003323, 0.0002348, 0.0001662, 0.0001174, 0.0000831, 0.0000587,
        0.0000415
    )
    private val CIE1931_2DEG_YBAR = doubleArrayOf(
        0.0000390, 0.0000640, 0.0001200, 0.0002170, 0.0003960, 0.0006400, 0.0012100, 0.0021800,
        0.0040000, 0.0073000, 0.0116000, 0.0168400, 0.0230000, 0.0298000, 0.0380000, 0.0480000,
        0.0600000, 0.0739000, 0.0909800, 0.1126000, 0.1390200, 0.1693000, 0.2080200, 0.2586000,
        0.3230000, 0.4073000, 0.5030000, 0.6082000, 0.7100000, 0.7932000, 0.8620000, 0.9148500,
        0.9540000, 0.9803000, 0.9949500, 1.0000000, 0.9950000, 0.9786000, 0.9520000, 0.9154000,
        0.8700000, 0.8163000, 0.7570000, 0.6949000, 0.6310000, 0.5668000, 0.5030000, 0.4412000,
        0.3810000, 0.3210000, 0.2650000, 0.2170000, 0.1750000, 0.1382000, 0.1070000, 0.0816000,
        0.0610000, 0.0445800, 0.0320000, 0.0232000, 0.0170000, 0.0119200, 0.0082100, 0.0057230,
        0.0041020, 0.0029290, 0.0020910, 0.0014840, 0.0010470, 0.0007400, 0.0005200, 0.0003611,
        0.0002492, 0.0001719, 0.0001200, 0.0000848, 0.0000600, 0.0000424, 0.0000300, 0.0000212,
        0.0000150
    )
    private val CIE1931_2DEG_ZBAR = doubleArrayOf(
        0.0064500, 0.0105500, 0.0200500, 0.0362100, 0.0678500, 0.1102000, 0.2074000, 0.3713000,
        0.6456000, 1.0390500, 1.3856000, 1.6229600, 1.7470600, 1.7826000, 1.7721100, 1.7441000,
        1.6692000, 1.5281000, 1.2876400, 1.0419000, 0.8129500, 0.6162000, 0.4651800, 0.3533000,
        0.2720000, 0.2123000, 0.1582000, 0.1117000, 0.0782500, 0.0572500, 0.0421600, 0.0298400,
        0.0203000, 0.0134000, 0.0087500, 0.0057500, 0.0039000, 0.0027500, 0.0021000, 0.0018000,
        0.0016500, 0.0014000, 0.0011000, 0.0010000, 0.0008000, 0.0006000, 0.0003400, 0.0002400,
        0.0001900, 0.0001000, 0.0000500, 0.0000300, 0.0000200, 0.0000100, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000
    )
    private const val CIE_CMF_START_NM = 380.0
    private const val CIE_CMF_STEP_NM = 5.0
    private const val PLANCK_C2_KELVIN_METERS = 1.438776877e-2

    fun kelvinToXy(kelvin: Float, tint: Float = 0f): Pair<Float, Float> {
        val k = kelvin.coerceIn(2000f, 10000f).toDouble()
        val (x, y) = planckianLocusXy(k)

        val t = tint / 1000.0f
        val xShift = x.toFloat() + t * 0.3f
        val yShift = (y.toFloat() - t * 0.15f).coerceIn(0f, 1f)
        return Pair(xShift, yShift)
    }

    // Planck's law (2*h*c^2 factor dropped; only relative SPD matters):
    //   L(lambda, T) ~ lambda^-5 / (exp(c2 / (lambda * T)) - 1),  c2 = h*c/k = 1.438776877e-2 m K.
    // Integrated over the CIE 1931 2-deg observer with Simpson's rule (5 nm steps, 380-780 nm)
    // equivalent to a blackbody radiator's Planckian-locus chromaticity (x, y).
    private fun planckianLocusXy(tempK: Double): Pair<Double, Double> {
        var xSum = 0.0
        var ySum = 0.0
        var zSum = 0.0
        for (i in CIE1931_2DEG_XBAR.indices) {
            val lambdaM = (CIE_CMF_START_NM + i * CIE_CMF_STEP_NM) * 1.0e-9
            val e = exp(-PLANCK_C2_KELVIN_METERS / (lambdaM * tempK))
            val sp = lambdaM.pow(-5.0) * e / (1.0 - e)
            val weight = if (i == 0 || i == CIE1931_2DEG_XBAR.lastIndex) 1.0 else if (i % 2 == 1) 4.0 else 2.0
            val f = sp * weight * CIE_CMF_STEP_NM / 3.0
            xSum += f * CIE1931_2DEG_XBAR[i]
            ySum += f * CIE1931_2DEG_YBAR[i]
            zSum += f * CIE1931_2DEG_ZBAR[i]
        }
        val total = xSum + ySum + zSum
        if (total <= 0.0) return Pair(ColorMatrix.D65_X.toDouble(), ColorMatrix.D65_Y.toDouble())
        return Pair(xSum / total, ySum / total)
    }

    fun chromaticAdaptationBradford(
        srcWhiteXY: Pair<Float, Float>,
        dstWhiteXY: Pair<Float, Float>
    ): ColorMatrix.Mat3 {
        val srcXyzVec = xyToXyzVector(srcWhiteXY.first, srcWhiteXY.second)
        val dstXyzVec = xyToXyzVector(dstWhiteXY.first, dstWhiteXY.second)

        val mBradford = ColorMatrix.Mat3(
            floatArrayOf(
                0.8951f, 0.2664f, -0.1614f,
                -0.7502f, 1.7135f, 0.0367f,
                0.0389f, -0.0685f, 1.0296f
            )
        )
        val mBradfordInv = ColorMatrix.inverse(mBradford)

        val srcLms = ColorMatrix.mulMatVec(mBradford, srcXyzVec)
        val dstLms = ColorMatrix.mulMatVec(mBradford, dstXyzVec)

        val ratio = ColorMatrix.diagonal(
            dstLms[0] / srcLms[0],
            dstLms[1] / srcLms[1],
            dstLms[2] / srcLms[2]
        )

        return ColorMatrix.multiply(
            ColorMatrix.multiply(mBradfordInv, ratio),
            mBradford
        )
    }

    fun buildSceneLinearTo709(
        kelvin: Float,
        tint: Float = 0f,
        calibrationMatrix: ColorMatrix.Mat3 = ColorMatrix.identity(),
        referenceToXyz: ColorMatrix.Mat3 = ColorMatrix.identity()
    ): ColorMatrix.Mat3 {
        val d65xy = Pair(ColorMatrix.D65_X, ColorMatrix.D65_Y)
        val userXY = kelvinToXy(kelvin, tint)

        val catMatrix = chromaticAdaptationBradford(userXY, d65xy)

        return ColorMatrix.multiply(
            ColorMatrix.multiply(
                ColorMatrix.multiply(
                    ColorMatrix.xyzToRGB(ColorMatrix.REC709),
                    catMatrix
                ),
                referenceToXyz
            ),
            calibrationMatrix
        )
    }

    fun buildAutoSceneLinearTo709(
        colorCorrectionTransform: android.hardware.camera2.params.ColorSpaceTransform?,
        calibrationMatrix: ColorMatrix.Mat3 = ColorMatrix.identity()
    ): ColorMatrix.Mat3 {
        val refToXyz = ColorMatrix.colorSpaceTransformToMatrix(colorCorrectionTransform)
            ?: ColorMatrix.identity()

        return ColorMatrix.multiply(
            ColorMatrix.multiply(
                ColorMatrix.xyzToRGB(ColorMatrix.REC709),
                refToXyz
            ),
            calibrationMatrix
        )
    }

    private fun xyToXyzVector(x: Float, y: Float): FloatArray {
        return floatArrayOf(x / y, 1.0f, (1 - x - y) / y)
    }

    private fun xyToXyz(x: Float, y: Float): ColorMatrix.Mat3 {
        val xyz = xyToXyzVector(x, y)
        return ColorMatrix.Mat3(floatArrayOf(
            xyz[0], 0f, 0f,
            0f, xyz[1], 0f,
            0f, 0f, xyz[2]
        ))
    }
}
