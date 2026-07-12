package com.agx.camera.color

import kotlin.math.*

object WhiteBalanceMath {

    data class KelvinResult(
        val xy: Pair<Float, Float>,
        val sceneLinearTo709: ColorMatrix.Mat3
    )

    fun kelvinToXy(kelvin: Float, tint: Float = 0f): Pair<Float, Float> {
        val k = kelvin.coerceIn(2000f, 10000f)
        val t = tint / 1000.0f

        val x = if (k < 4000) {
            -0.2661239e9f / k.pow(3) - 0.2343580e6f / k.pow(2) + 0.8776956e3f / k + 0.179910f
        } else {
            -3.0258469e9f / k.pow(3) + 2.1070379e6f / k.pow(2) + 0.2226347e3f / k + 0.240390f
        }
        val y = if (k < 4000) {
            -1.1063814e9f / k.pow(3) - 1.34811020e6f / k.pow(2) + 1.7275600e3f / k + 0.179910f
        } else {
            -2.0903677e9f / k.pow(3) + 1.6435273e6f / k.pow(2) + 0.1372315e3f / k + 0.240390f
        }

        val xShift = x + t * 0.3f
        val yShift = y - t * 0.15f
        return Pair(xShift, yShift.coerceIn(0.0f, 1.0f))
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

        val catMatrix = chromaticAdaptationBradford(d65xy, userXY)

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

    fun buildGrayCardSceneLinearTo709(
        grayCardMatrix: ColorMatrix.Mat3,
        calibrationMatrix: ColorMatrix.Mat3 = ColorMatrix.identity()
    ): ColorMatrix.Mat3 {
        return ColorMatrix.multiply(
            ColorMatrix.multiply(
                ColorMatrix.xyzToRGB(ColorMatrix.REC709),
                grayCardMatrix
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
