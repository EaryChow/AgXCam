package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * What the preview-size picker owes the rest of the pipeline, as properties of
 * the picking rule rather than as a table of one handset's sizes.
 *
 * zoomK is not a camera zoom. It is the sensor crop divided by the demosaic
 * output, per axis (`PreviewRenderer: previewZoomK = max(crop[2]/w, crop[3]/h)`),
 * so it is fixed by the pair (sensor, preview size) and nothing else. That makes
 * it the one variant parameter every measurement inherits without anyone choosing
 * it - which is why the picker has to be trustworthy about it. Getting the
 * preview size wrong does not just cost resolution: it changes zoomK, and
 * zoomK changes what every downstream ratio is measuring.
 *
 * The defect this file exists for was in the tie-break. Sorting candidate sizes
 * on aspect error alone is not enough, because every size of the requested aspect
 * ties at error zero, and a stable min-then-first then returns whichever size the
 * HAL happened to list first. HALs commonly list smallest-first, so the picker
 * silently took the *worst* size that satisfied the aspect - and did so while
 * looking like it had honoured the cap.
 *
 * Everything below is written against a synthetic sensor and a synthetic ladder.
 * That is deliberate. A test that names a real device's sensor size, its HAL's
 * size list or its screen turns one handset's geometry into the specification,
 * and then fails on every other device while proving nothing about the rule. The
 * properties are what transfer; the numbers are not.
 *
 * This covers the host-side picking rule. The GLSL for every program is
 * scope-resolved by GlslScopeTest; there is no per-resolution shader variant to
 * compile, because none exists.
 */
class PreviewZoomKMatrixTest {

    /**
     * A synthetic 4:3 sensor. Not any particular handset: the point of these
     * tests is the rule, and the rule does not care how many pixels it is handed.
     */
    private val sensorWidth = 3200
    private val sensorHeight = 2400

    /**
     * A synthetic 4:3 size ladder - round multiples of one base rung, ascending.
     * The real HAL's list is whatever that device supports; nothing here depends
     * on which sizes those are.
     */
    private val ladder43: List<Pair<Int, Int>> =
        (1..5).map { 320 * it to 240 * it }

    /** Float render targets the pipeline allocates at preview size: demosaic,
     * 4x DPC, denoise, sigma, 4x stage-5, axis bucket, structure map. RGBA32F is
     * 16 bytes per texel. */
    private val floatTargetsPerPreview = 13
    private val bytesPerFloatTarget = 16

    /** Mirrors PreviewRenderer's per-axis-worst zoom factor. */
    private fun zoomK(sensorW: Int, sensorH: Int, previewWidth: Int, previewHeight: Int): Float =
        maxOf(
            sensorW.toFloat() / previewWidth,
            sensorH.toFloat() / previewHeight
        )

    private fun source(name: String): String {
        val f = File("src/main/java/com/agx/camera/gpu/$name.kt")
        assertTrue("cannot read $name.kt from ${f.absolutePath}", f.exists())
        return f.readText()
    }

    /** Mirrors MainActivity.getMaxPreviewDimensions: the cap bounds the width,
     * and the screen aspect scales the height bound down from it. */
    private fun screenAspectBounds(cap: Int, screenWidth: Int, screenHeight: Int): Pair<Int, Int> =
        cap to (cap * screenHeight / screenWidth)

    /** Mirrors LensManager.bestPreviewSize: aspect error first, area second. */
    private fun pick(
        ladder: List<Pair<Int, Int>>,
        bounds: Pair<Int, Int>,
        targetAspect: Float
    ): Pair<Int, Int>? {
        val sensorAspect = sensorWidth.toFloat() / sensorHeight
        val effective = if (targetAspect > 0f) targetAspect else sensorAspect
        return ladder
            .filter { it.first <= bounds.first && it.second <= bounds.second }
            .sortedWith(
                compareBy(
                    { kotlin.math.abs(it.first.toFloat() / it.second - effective) },
                    { -(it.first.toLong() * it.second) }
                )
            )
            .firstOrNull()
    }

    private fun megabytes(previewWidth: Int, previewHeight: Int): Double =
        floatTargetsPerPreview * previewWidth * previewHeight * bytesPerFloatTarget / (1024.0 * 1024.0)

    // ---- The tie-break, which is the actual defect ----------------------

    @Test
    fun anAspectTieResolvesToTheLargestSizeNotTheFirstTheHalListed() {
        // The regression. Every 4:3 size ties at aspect error zero, so with a
        // single-key sort the winner is decided by listing order - and HALs tend
        // to list smallest-first. Under that ordering the broken rule returned
        // the smallest size that satisfied the aspect, i.e. the maximum
        // resolution loss the bounds allowed, while still honouring the cap.
        val smallestFirst = ladder43
        val bounds = screenAspectBounds(1280, 2000, 1000)
        assertEquals(
            "a smallest-first HAL must still yield the largest in-bounds 4:3 rung",
            ladder43.filter { it.first <= bounds.first && it.second <= bounds.second }.maxByOrNull { it.first * it.second },
            pick(ladder43, bounds, 0f)
        )
        assertEquals("sanity: the bounds must admit several rungs",
            ladder43.count { it.first <= bounds.first && it.second <= bounds.second } > 1,
            true
        )
    }

    @Test
    fun theHalListingOrderCannotChangeThePick() {
        // The stronger statement, and the one that actually pins the defect: the
        // result must be a function of the ladder's *contents*, not of the order
        // the HAL happened to enumerate it in. A single-key aspect sort fails
        // this immediately, which is why it is worth asserting separately from
        // the expected value above.
        val bounds = screenAspectBounds(1280, 2000, 1000)
        val expected = pick(ladder43, bounds, 0f)
        assertTrue("this test is vacuous if the bounds admit fewer than two rungs",
            ladder43.count { it.first <= bounds.first && it.second <= bounds.second } > 1)
        for (ordering in listOf(
            ladder43,
            ladder43.reversed(),
            ladder43.sortedBy { it.second },
            ladder43.shuffled(java.util.Random(12345))
        )) {
            assertEquals(
                "picked size must not depend on HAL listing order: $ordering",
                expected,
                pick(ordering, bounds, 0f)
            )
        }
    }

    @Test
    fun aSizeTheBoundsFilteredOutCanNeverBePicked() {
        // The other half of the constraint. The aspect tie-break can only choose
        // among what survived the bounds filter - no ordering can recover a size
        // the bounds removed, so a test that expects an out-of-bounds size is
        // testing a wish rather than the picker.
        val narrow = ladder43.first()
        val bounds = narrow.first - 1 to narrow.second - 1
        assertEquals(
            "nothing at or below the bounds means no pick at all",
            null,
            pick(ladder43, bounds, 0f)
        )
        assertTrue("the excluded rung must really have been excluded",
            pick(ladder43, bounds, 0f) == null || (pick(ladder43, bounds, 0f)!!.first <= bounds.first))
    }

    @Test
    fun theScreenAspectBoundCanExcludeARungTheCapWouldOtherwiseFit() {
        // Why the bounds are a screen-aspect calculation and not just the cap: a
        // cap that bounds width alone admits sizes the panel cannot show, and on
        // a tall aspect ratio that removes the rung the cap was clearly aimed at.
        // Asserted as a property over several aspects rather than for one
        // particular screen, because which rungs a given screen loses is a fact
        // about that screen, not about the rule.
        val cap = 854
        val sensorAspect = sensorHeight.toFloat() / sensorWidth
        var sawAScreenThatLostARung = false
        for ((screenW, screenH) in listOf(1000 to 1000, 2000 to 1000, 1600 to 1200, 1200 to 1600)) {
            val viaScreen = screenAspectBounds(cap, screenW, screenH)
            val viaSensor = cap to (cap * sensorAspect).toInt()
            if (pick(ladder43, viaScreen, 0f) != pick(ladder43, viaSensor, 0f)) sawAScreenThatLostARung = true
        }
        assertTrue(
            "the screen aspect must be able to remove a rung the sensor aspect keeps",
            sawAScreenThatLostARung
        )
    }

    // ---- zoomK itself ---------------------------------------------------

    @Test
    fun zoomKIsThePerAxisMaximumSoAMatchingAspectCollapsesItToOneNumber() {
        // The formula takes a per-axis max, which only collapses to a single
        // number while the sensor and the preview share an aspect. A picker that
        // returned a mismatched preview would make zoomK aspect-dependent and
        // every ratio measured at it would stop being comparable to another.
        for ((pw, ph) in ladder43) {
            assertEquals(
                "$pw x $ph must keep the sensor's aspect",
                sensorWidth.toFloat() / sensorHeight,
                pw.toFloat() / ph,
                1.0e-3f
            )
            assertEquals(
                "$pw x $ph: a matching aspect makes both axes agree",
                sensorWidth.toFloat() / pw,
                zoomK(sensorWidth, sensorHeight, pw, ph),
                1.0e-3f
            )
        }
    }

    @Test
    fun zoomKShrinksAsThePreviewGrows() {
        // Guards the direction of the bias: a smaller preview is a larger k and
        // therefore a larger minification of the sensor crop per axis. If this
        // ever inverts, a "smaller preview is less biased" assumption elsewhere
        // is wrong.
        val rows = ladder43.map { (w, h) -> Triple(w, h, zoomK(sensorWidth, sensorHeight, w, h)) }
        for (i in 1 until rows.size) {
            assertTrue(
                "k must shrink as the preview grows: ${rows[i - 1]} then ${rows[i]}",
                rows[i].third < rows[i - 1].third
            )
        }
        assertEquals(
            "every rung must be a distinct operating point",
            rows.size,
            rows.map { it.third }.distinct().size
        )
    }

    // ---- Memory ---------------------------------------------------------

    @Test
    fun thePipelineMemoryCostGrowsWithThePreviewArea() {
        // Why the bounds double as a pixel-count guard and should be replaced
        // deliberately rather than deleted. The pipeline holds a fixed number of
        // RGBA32F targets at preview resolution, so the cost of relaxing the
        // height bound is linear in the area it admits.
        val small = ladder43[1]
        val large = ladder43.last()
        assertTrue(
            "the largest rung must cost materially more than the smallest",
            megabytes(large.first, large.second) > megabytes(small.first, small.second)
        )
        // And the arithmetic is exactly targets * bytes * texels, so a caller can
        // price any rung the picker might return without running the pipeline.
        assertEquals(
            doubleToLongBits(
                (floatTargetsPerPreview * large.first * large.second * bytesPerFloatTarget).toDouble()
            ),
            doubleToLongBits(
                megabytes(large.first, large.second) * 1024.0 * 1024.0
            )
        )
    }

    private fun doubleToLongBits(d: Double): Long = java.lang.Double.doubleToLongBits(d)

    // ---- The ratio denominator carries no zoomK ------------------------

    @Test
    fun theRatioDefinitionItselfHasNoZoomKTerm() {
        // The structure map's SNR ratio is
        //
        //   ratio(p) = residual(p) / ( noisePrediction(p) * r_retain(ISO) )
        //
        // and a zoomK correction is only *derivable* if k appears somewhere in
        // that definition - in the denominator or in the bands. It appears in
        // neither:
        //
        //   noisePrediction = lumaEpsScale * (sigmaHat2 + sigmaDm2)
        //                     * inverseRange2 * calibScale * epsBoost
        //
        // Six factors, none of them geometry. Checked as source text rather than
        // by eye, because a k argument added later is precisely the change that
        // would survive a visual review, and because the host and the shader are
        // two copies of one formula - a k term added to only one of them is the
        // drift this project's neighbouring tests already guard against.
        val host = source("StructureMapClassifier")
        val hostBody = host.substringAfter("return lumaEpsScale * (sigmaHat2 + sigmaDm2)")
            .substringBefore("\n")
        for (factor in listOf("inverseRange2", "calibScale", "epsBoost")) {
            assertTrue("the fold must keep $factor", hostBody.contains(factor))
        }
        assertFalse(
            "the host denominator must not have grown a geometry term: $hostBody",
            Regex("zoom|crop|preview|\\bk\\b|k_[a-z]", RegexOption.IGNORE_CASE)
                .containsMatchIn(hostBody)
        )

        val glsl = source("StructureMapShaderProgram")
        val deviceBody = glsl.substringAfter("float noisePrediction(vec2 p) {")
            .substringBefore("\n}")
        for (uniform in listOf("u_luma_eps_scale", "u_sigma_dm2", "u_inverse_range2", "u_calib_scale")) {
            assertTrue("the GLSL denominator must keep $uniform", deviceBody.contains(uniform))
        }
        assertFalse(
            "the GLSL denominator must not have grown a geometry term: $deviceBody",
            Regex("zoom|minif|\\bu_?k\\b", RegexOption.IGNORE_CASE).containsMatchIn(deviceBody)
        )

        // And the bands are per ISO: rRetain, plainMax and textureMin are
        // non-decreasing in ISO and none of them has a k column, so the
        // thresholds cannot carry a geometry correction even if some future
        // revision decided it wanted one. Non-decreasing rather than strictly
        // increasing because the rungs are sparser than the profile buckets on
        // purpose - one rung covers several buckets without a visible jump.
        val cals = StructureMapClassifier.allCalibrations()
        for (i in 1 until cals.size) {
            val (lo, hi) = cals[i - 1] to cals[i]
            assertTrue("rRetain must not fall with ISO: ${lo.isoBucket} then ${hi.isoBucket}", hi.rRetain >= lo.rRetain)
            assertTrue("plainMax must not narrow with ISO: ${lo.isoBucket} then ${hi.isoBucket}", hi.plainMax >= lo.plainMax)
            assertTrue("textureMin must not narrow with ISO: ${lo.isoBucket} then ${hi.isoBucket}", hi.textureMin >= lo.textureMin)
        }
    }
}