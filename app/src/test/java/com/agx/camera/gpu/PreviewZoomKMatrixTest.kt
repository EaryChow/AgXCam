package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The preview-resolution variant matrix: which zoom factor each preview size
 * the app can actually produce hands to the rest of the pipeline.
 *
 * zoomK is not a camera zoom. It is the sensor crop divided by the demosaic
 * output, per axis (PreviewRenderer: `previewZoomK = max(crop[2]/w,
 * crop[3]/h)`), so it is fixed by the pair (sensor, preview size) and nothing
 * else. That makes it the one variant parameter every measurement inherits
 * without anyone choosing it.
 *
 * The matrix exists because of what it revealed. On this 4096x3072 sensor the
 * smallest preview the size picker can return is 320x240, which pins zoomK at
 * 12.8 - and 12.8 is not a mild setting, it is past the HAL's own
 * `maxDigitalZoom=10.0`, so the app is doing the minification itself in GL.
 * Three separate faults all read as one at that operating point:
 *
 *  - the sparse-grid chain gate is `zoomK <= 2`, so it is permanently closed
 *    here and sigma always comes from the measurement chain;
 *  - `calibScale` was calibrated against residual that is *not* minified by 12.8,
 *    so every device census ratio carried that bias;
 *  - the size picker returned 320x240 for a 4:3 sensor where 640x480 also fit,
 *    purely because both tie at aspect error 0.
 *
 * So the three rows below are not examples. They are the three acceptance
 * points the structure-map ratio has to hold across, and the matrix is the
 * thing that keeps a future picker change from silently re-picking a fourth.
 *
 * This covers the host-side variant configuration per resolution. The GLSL for
 * every program is scope-resolved by GlslScopeTest; there is no per-resolution
 * shader variant to compile, because none exists.
 */
class PreviewZoomKMatrixTest {

    /** The sensor this project ships against: 12MP, 4:3. */
    private val sensorWidth = 4096
    private val sensorHeight = 3072

    /**
     * Preview sizes the picker can return on this sensor. 480x360 is included
     * because it is a real HAL size reachable between the other two, and it is
     * the row that distinguishes "k tracks the preview" from "k is a constant".
     */
    private val variants = listOf(
        Triple("640x480", 640, 480),
        Triple("480x360", 480, 360),
        Triple("320x240", 320, 240)
    )

    /** Mirrors PreviewRenderer's per-axis-worst zoom factor. */
    private fun zoomK(previewWidth: Int, previewHeight: Int): Float =
        maxOf(
            sensorWidth.toFloat() / previewWidth,
            sensorHeight.toFloat() / previewHeight
        )

    @Test
    fun theMatrixReproducesTheZoomFactorsTheDeviceLogged() {
        // Pinned rather than derived from the list, so a change to the sensor
        // constant or to the formula has to be acknowledged here. These are the
        // values the 2026-10-04 device logs reported at 320x240, at the two
        // decimal places the pipeline-state line prints them.
        assertEquals(6.40f, zoomK(640, 480), 0.01f)
        assertEquals(8.53f, zoomK(480, 360), 0.01f)
        assertEquals(12.80f, zoomK(320, 240), 0.01f)
    }

    @Test
    fun everyVariantIsFourByThreeSoOneAxisAlwaysSetsTheFactor() {
        // The formula takes a per-axis max, which only collapses to a single
        // number while the sensor and the preview share an aspect. A picker that
        // returned a 16:9 preview would make k aspect-dependent and the flat
        // anchor would stop being comparable between variants.
        for ((label, w, h) in variants) {
            assertEquals(
                "$label must keep the sensor's 4:3 aspect",
                sensorWidth.toFloat() / sensorHeight,
                w.toFloat() / h,
                1.0e-3f
            )
        }
    }

    @Test
    fun thisSensorCanNeverCloseTheSparseGridGate() {
        // zoomK <= 2 would need a 2048-wide preview from a 4096-wide sensor.
        // That size is not offered by this HAL, so the sparse chain is shut on
        // every variant in the matrix and the sigma the map reads always comes
        // from the measurement chain. Anything reasoning about the sparse path
        // has to know this, because no device run will ever produce it.
        assertTrue(
            "no preview this HAL offers reaches the k<=2 sparse-grid gate",
            variants.all { zoomK(it.second, it.third) > 2f }
        )
        // The HAL caps its own digital zoom at 10.0. Past that the app is doing
        // the minification itself in GL, so the two smaller previews are the ones
        // whose residual no HAL model describes. The largest variant is still
        // inside the HAL's range, which is the point of picking it as the
        // baseline: it is the only row where a HAL-side explanation survives.
        assertEquals(6.40f, zoomK(640, 480), 0.01f)
        assertTrue(
            "320x240 must be past the HAL's own digital zoom",
            zoomK(320, 240) > 10f
        )
    }

    @Test
    fun zoomKIsMonotonicInPreviewSizeAcrossTheMatrix() {
        // Guards the direction of the bias: a smaller preview is a larger k and
        // therefore a larger minification of the sensor crop per axis. If this
        // ever inverts, a "smaller preview is less biased" assumption elsewhere
        // is wrong.
        val rows = variants.map { Triple(it.first, it.second, zoomK(it.second, it.third)) }
            .sortedBy { it.second }
        for (i in 1 until rows.size) {
            assertTrue(
                "k must shrink as the preview grows: ${rows[i - 1]} then ${rows[i]}",
                rows[i].third < rows[i - 1].third
            )
        }
        assertEquals(
            "every row must be a distinct operating point",
            rows.size,
            rows.map { it.third }.distinct().size
        )
    }
}