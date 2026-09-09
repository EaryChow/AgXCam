package com.agx.camera.color

import kotlin.math.*

object AgxPrecomputer {

    data class AgxUniforms(
        val insetMat: FloatArray,
        val outsetMat: FloatArray,
        val sceneLinearTo709: FloatArray,
        val toRec2020: FloatArray,
        val logMidgray: Float,
        val displayMidgray: Float,
        val whiteLevel: Float,
        val blackLevel: Float
    )

    data class InsetParams(
        val rotation: FloatArray = floatArrayOf(0.0373f, -0.0214f, -0.0532f),
        val attenuation: FloatArray = floatArrayOf(32.9652f, 28.0513f, 12.4754f),
        val useRotationForReverse: Boolean = true,
        val useAttenuationForBoost: Boolean = false,
        val reverseRotation: FloatArray = floatArrayOf(0f, 0f, 0f),
        val purityBoost: FloatArray = floatArrayOf(32.3174f, 28.3256f, 3.7433f),
        val tintingScale: Float = 0f,
        val tintingHue: Float = 0f
    )

    private const val LOG_MIN = -10.0f
    private const val LOG_MAX = 6.5f

    fun compute(
        params: InsetParams,
        sceneLinearTo709: ColorMatrix.Mat3,
        whiteLevel: Float,
        blackLevel: Float,
        middleGrayPercent: Float = 18f
    ): AgxUniforms {
        val insetMat = computeInsetMatrix(params)
        val outsetMat = computeOutsetMatrix(params)
        val logMidgray = lin2logScalar(0.18f)
        val displayMidgray = (middleGrayPercent / 100f).pow(1.0f / 2.4f)
        val toRec2020 = ColorMatrix.rgbToRGB(ColorMatrix.REC709, ColorMatrix.REC2020)

        return AgxUniforms(
            insetMat = insetMat.m,
            outsetMat = outsetMat.m,
            sceneLinearTo709 = sceneLinearTo709.m,
            toRec2020 = toRec2020.m,
            logMidgray = logMidgray,
            displayMidgray = displayMidgray,
            whiteLevel = whiteLevel,
            blackLevel = blackLevel
        )
    }

    private fun lin2logScalar(linear: Float): Float {
        val logFloor = 0.18f * 2f.pow(LOG_MIN)
        val clamped = max(linear, logFloor)
        val logVal = log2(clamped / 0.18f)
        val clampedLog = logVal.coerceIn(LOG_MIN, LOG_MAX)
        return (clampedLog + abs(LOG_MIN)) / (abs(LOG_MIN) + abs(LOG_MAX))
    }

    private fun computeInsetMatrix(params: InsetParams): ColorMatrix.Mat3 {
        val insetCh = insetPrimaries(
            ColorMatrix.REC709,
            params.attenuation[0], params.attenuation[1], params.attenuation[2],
            params.rotation[0], params.rotation[1], params.rotation[2],
            0f, 0f
        )
        return ColorMatrix.rgbToRGB(insetCh, ColorMatrix.REC709)
    }

    private fun computeOutsetMatrix(params: InsetParams): ColorMatrix.Mat3 {
        val rot = if (params.useRotationForReverse) params.rotation else params.reverseRotation
        val boost = if (params.useAttenuationForBoost) params.attenuation else params.purityBoost
        val ch = insetPrimaries(
            ColorMatrix.REC709,
            boost[0], boost[1], boost[2],
            rot[0], rot[1], rot[2],
            params.tintingHue + PI.toFloat(), params.tintingScale
        )
        return ColorMatrix.inverse(ColorMatrix.rgbToRGB(ch, ColorMatrix.REC709))
    }

    private fun insetPrimaries(
        origN: Chromaticities,
        redScale: Float, greenScale: Float, blueScale: Float,
        redRotate: Float, greenRotate: Float, blueRotate: Float,
        achromaticRotate: Float, achromaticOutset: Float
    ): Chromaticities {
        var n = origN
        val originalN = origN

        val scaledN = scalePrim(origN, 4f, 4f, 4f)
        n = rotatePrimary(scaledN, redRotate, greenRotate, blueRotate)

        val m = polygon(origN)

        val redLine = if (redRotate > 0) m.red else m.green
        val greenLine = if (greenRotate > 0) m.blue else m.red
        val blueLine = if (blueRotate > 0) m.green else m.blue

        val wp = Pair(originalN.whiteX, originalN.whiteY)
        val lr = lineEquation(Pair(n.redX, n.redY), wp)
        val lg = lineEquation(Pair(n.greenX, n.greenY), wp)
        val lb = lineEquation(Pair(n.blueX, n.blueY), wp)

        val pr = lineIntersection(lr, redLine)
        val pg = lineIntersection(lg, greenLine)
        val pb = lineIntersection(lb, blueLine)

        n = n.copy(redX = pr.first, redY = pr.second,
                    greenX = pg.first, greenY = pg.second,
                    blueX = pb.first, blueY = pb.second)

        val rs = (100 - redScale) / 100
        val gs = (100 - greenScale) / 100
        val bs = (100 - blueScale) / 100
        n = scalePrim(n, rs, gs, bs)

        val polyOrig = polygon(originalN)
        val origWhite = Pair(polyOrig.whiteX, polyOrig.whiteY)
        val arbScale = 4f

        val scaledAch = Pair(origWhite.first, origWhite.second * arbScale)
        val dx = scaledAch.first - origWhite.first
        val dy = scaledAch.second - origWhite.second
        val rotatedAch = Pair(
            origWhite.first + dx * cos(achromaticRotate) - dy * sin(achromaticRotate),
            origWhite.second + dx * sin(achromaticRotate) + dy * cos(achromaticRotate)
        )

        val la = lineEquation(rotatedAch, origWhite)
        val e1 = lineEquation(
            Pair(originalN.redX, originalN.redY),
            Pair(originalN.greenX, originalN.greenY)
        )
        val e2 = lineEquation(
            Pair(originalN.greenX, originalN.greenY),
            Pair(originalN.blueX, originalN.blueY)
        )
        val e3 = lineEquation(
            Pair(originalN.blueX, originalN.blueY),
            Pair(originalN.redX, originalN.redY)
        )

        val i1 = lineIntersection(la, e1)
        val i2 = lineIntersection(la, e2)
        val i3 = lineIntersection(la, e3)

        var hullAch = origWhite
        if (onSegment(i1, Pair(originalN.redX, originalN.redY), Pair(originalN.greenX, originalN.greenY), rotatedAch, origWhite))
            hullAch = i1
        else if (onSegment(i2, Pair(originalN.greenX, originalN.greenY), Pair(originalN.blueX, originalN.blueY), rotatedAch, origWhite))
            hullAch = i2
        else if (onSegment(i3, Pair(originalN.blueX, originalN.blueY), Pair(originalN.redX, originalN.redY), rotatedAch, origWhite))
            hullAch = i3

        val interpX = (origWhite.first - hullAch.first) * (1 - achromaticOutset)
        val interpY = (origWhite.second - hullAch.second) * (1 - achromaticOutset)
        val newWhiteX = hullAch.first + interpX
        val newWhiteY = hullAch.second + interpY

        return n.copy(whiteX = newWhiteX, whiteY = newWhiteY)
    }

    private fun centerPrimaries(ch: Chromaticities): Chromaticities = ch.copy(
        redX = ch.redX - ch.whiteX, redY = ch.redY - ch.whiteY,
        greenX = ch.greenX - ch.whiteX, greenY = ch.greenY - ch.whiteY,
        blueX = ch.blueX - ch.whiteX, blueY = ch.blueY - ch.whiteY
    )

    private fun decenterPrimaries(ch: Chromaticities): Chromaticities = ch.copy(
        redX = ch.redX + ch.whiteX, redY = ch.redY + ch.whiteY,
        greenX = ch.greenX + ch.whiteX, greenY = ch.greenY + ch.whiteY,
        blueX = ch.blueX + ch.whiteX, blueY = ch.blueY + ch.whiteY
    )

    private fun cartesianToPolar(x: Float, y: Float): Pair<Float, Float> =
        Pair(sqrt(x * x + y * y), atan2(y, x))

    private fun polarToCartesian(r: Float, theta: Float): Pair<Float, Float> =
        Pair(r * cos(theta), r * sin(theta))

    private fun rotatePrimary(ch: Chromaticities, rrot: Float, grot: Float, brot: Float): Chromaticities {
        var c = centerPrimaries(ch)
        var (rr, rt) = cartesianToPolar(c.redX, c.redY)
        var (gr, gt) = cartesianToPolar(c.greenX, c.greenY)
        var (br, bt) = cartesianToPolar(c.blueX, c.blueY)
        rt += rrot; gt += grot; bt += brot
        val (rx, ry) = polarToCartesian(rr, rt)
        val (gx, gy) = polarToCartesian(gr, gt)
        val (bx, by) = polarToCartesian(br, bt)
        c = c.copy(redX = rx, redY = ry, greenX = gx, greenY = gy, blueX = bx, blueY = by)
        return decenterPrimaries(c)
    }

    private fun scalePrim(ch: Chromaticities, rs: Float, gs: Float, bs: Float): Chromaticities {
        var c = centerPrimaries(ch)
        c = c.copy(redX = c.redX * rs, redY = c.redY * rs)
        c = c.copy(greenX = c.greenX * gs, greenY = c.greenY * gs)
        c = c.copy(blueX = c.blueX * bs, blueY = c.blueY * bs)
        return decenterPrimaries(c)
    }

    private fun lineEquation(a: Pair<Float, Float>, b: Pair<Float, Float>): Pair<Float, Float> {
        val dx = b.first - a.first
        if (abs(dx) < 1e-6f) return Pair(Float.POSITIVE_INFINITY, a.first)
        val m = (b.second - a.second) / dx
        val c = a.second - m * a.first
        return Pair(m, c)
    }

    private data class EdgeLines(
        val red: Pair<Float, Float>,
        val green: Pair<Float, Float>,
        val blue: Pair<Float, Float>,
        val whiteX: Float,
        val whiteY: Float
    )

    private fun polygon(ch: Chromaticities): EdgeLines {
        return EdgeLines(
            red = lineEquation(Pair(ch.redX, ch.redY), Pair(ch.greenX, ch.greenY)),
            green = lineEquation(Pair(ch.redX, ch.redY), Pair(ch.blueX, ch.blueY)),
            blue = lineEquation(Pair(ch.blueX, ch.blueY), Pair(ch.greenX, ch.greenY)),
            whiteX = ch.whiteX,
            whiteY = ch.whiteY
        )
    }

    private fun lineIntersection(l1: Pair<Float, Float>, l2: Pair<Float, Float>): Pair<Float, Float> {
        val m1 = l1.first; val c1 = l1.second
        val m2 = l2.first; val c2 = l2.second
        val inf1 = !m1.isFinite()
        val inf2 = !m2.isFinite()
        return when {
            inf1 && inf2 -> Pair(0f, 0f)
            inf1 -> Pair(c1, m2 * c1 + c2)
            inf2 -> Pair(c2, m1 * c2 + c1)
            else -> {
                val x = (c2 - c1) / (m1 - m2)
                Pair(x, m1 * x + c1)
            }
        }
    }

    private fun onSegment(
        p: Pair<Float, Float>,
        c: Pair<Float, Float>, d: Pair<Float, Float>,
        a: Pair<Float, Float>, b: Pair<Float, Float>
    ): Boolean {
        val abX = b.first - a.first; val abY = b.second - a.second
        val cdX = d.first - c.first; val cdY = d.second - c.second
        val t = if (abs(abX) > abs(abY)) (p.first - a.first) / abX else (p.second - a.second) / abY
        val u = if (abs(cdX) > abs(cdY)) (p.first - c.first) / cdX else (p.second - c.second) / cdY
        return t in 0f..1f && u in 0f..1f
    }
}
