package com.agx.camera.camera

/**
 * Stage 0 - ISO-calibrated noise model sigma^2(x) = a*x + b (units: 10-bit DN above black level).
 *
 * The model follows a three-segment labeling convention:
 *  - MEASURED     : ISO <= 1600   (placeholder values pending field calibration)
 *  - INTERPOLATED : 3200 ~ 6400
 *  - ASSUMED      : >= 12800      (extreme segment, incl. the alpha -> 0.1~0.2 hypothesis range)
 *
 *  a(iso) folds photon shot noise (grows with ISO gain), b(iso) folds read noise
 *  (grows as gain^2 in DN). Current anchors are a physically-plausible starting point
 *  marked ASSUMED until device calibration replaces them.
 */
object NoiseModel {

    const val SEG_MEASURED = "measured"
    const val SEG_INTERPOLATED = "interpolated"
    const val SEG_ASSUMED = "assumed"

    /** Photon-noise coefficient a(iso) in sigma^2 = a*x + b. Linear in ISO gain. */
    fun photonCoeff(iso: Int): Float = 0.0067f * iso.coerceAtLeast(1) / 100f

    /** Read-noise variance b(iso) in DN^2. Read noise in DN grows with ISO gain. */
    fun readNoiseVariance(iso: Int): Float {
        val g = 0.33f * iso.coerceAtLeast(1) / 100f
        return g * g
    }

    /** sigma^2(x) at signal level x (black-level-subtracted DN). */
    fun sigmaSq(signal: Float, iso: Int): Float =
        photonCoeff(iso) * signal.coerceAtLeast(0f) + readNoiseVariance(iso)

    /** sigma(x) at signal level x. */
    fun sigma(signal: Float, iso: Int): Float = kotlin.math.sqrt(sigmaSq(signal, iso))

    fun segmentForIso(iso: Int): String = when {
        iso <= 1600 -> SEG_MEASURED
        iso <= 6400 -> SEG_INTERPOLATED
        else -> SEG_ASSUMED
    }

    /** Stage 3 blend alpha(ISO) per the Stage 3 table. 1.0 = keep RAW (off), 0.3/0.1~0.2 = strong. */
    fun alphaRaw(iso: Int): Float {
        val i = iso.coerceIn(50, 200000).toFloat()
        val anchors = listOf(
            Pair(0f, 1f),        // <=1600  (assumed)
            Pair(1600f, 1f),     // boundary
            Pair(3200f, 0.5f),   // measured (2410 mismatch aside)
            Pair(6400f, 0.4f),   // interpolated
            Pair(12800f, 0.3f),  // measured side
            Pair(40000f, 0.25f), // assumed extreme shoulder
            Pair(200000f, 0.1f)  // assumed extreme floor
        )
        if (i <= anchors.first().first) return anchors.first().second
        for (k in 1 until anchors.size) {
            val (x0, y0) = anchors[k - 1]
            val (x1, y1) = anchors[k]
            if (i <= x1) {
                val t = if (x1 > x0) (i - x0) / (x1 - x0) else 0f
                return y0 + (y1 - y0) * t
            }
        }
        return anchors.last().second
    }

    /** Debug string for a given ISO. */
    fun describe(iso: Int): String {
        val seg = segmentForIso(iso)
        val a = photonCoeff(iso)
        val b = readNoiseVariance(iso)
        val sMid = sigma(300f, iso)
        val sBright = sigma(700f, iso)
        return "iso=$iso $seg a=$a b=$b sig(300)=${String.format("%.2f", sMid)} sig(700)=${String.format("%.2f", sBright)} alpha=${String.format("%.2f", alphaRaw(iso))}"
    }
}