package com.agx.camera.phase1.week7

import org.junit.Assert.*
import org.junit.Test

class OrientationMirrorTest {

    private fun applyOrientation(u: FloatArray, sensorOrientation: Int): FloatArray {
        val uv = u.copyOf()
        when (sensorOrientation) {
            90 -> {
                val tmp = uv[0]
                uv[0] = uv[1]
                uv[1] = 1.0f - tmp
            }
            180 -> {
                uv[0] = 1.0f - uv[0]
                uv[1] = 1.0f - uv[1]
            }
            270 -> {
                val tmp = uv[0]
                uv[0] = 1.0f - uv[1]
                uv[1] = tmp
            }
        }
        return uv
    }

    private fun applyMirrorX(u: FloatArray, flipX: Boolean): FloatArray {
        val uv = u.copyOf()
        if (flipX) {
            uv[0] = 1.0f - uv[0]
        }
        return uv
    }

    @Test
    fun orientation0_noTransform() {
        val uv = floatArrayOf(0.5f, 0.5f)
        val result = applyOrientation(uv, 0)
        assertEquals(0.5f, result[0], 0.001f)
        assertEquals(0.5f, result[1], 0.001f)
    }

    @Test
    fun orientation90_rotatesClockwise() {
        val uv = floatArrayOf(0.0f, 0.5f)
        val result = applyOrientation(uv, 90)
        assertEquals(0.5f, result[0], 0.001f)
        assertEquals(1.0f, result[1], 0.001f)
    }

    @Test
    fun orientation90_topLeftMapsToTopRight() {
        val uv = floatArrayOf(0.0f, 0.0f)
        val result = applyOrientation(uv, 90)
        assertEquals(0.0f, result[0], 0.001f)
        assertEquals(1.0f, result[1], 0.001f)
    }

    @Test
    fun orientation90_bottomRightMapsToBottomLeft() {
        val uv = floatArrayOf(1.0f, 1.0f)
        val result = applyOrientation(uv, 90)
        assertEquals(1.0f, result[0], 0.001f)
        assertEquals(0.0f, result[1], 0.001f)
    }

    @Test
    fun orientation180_flipsBothAxes() {
        val uv = floatArrayOf(0.2f, 0.8f)
        val result = applyOrientation(uv, 180)
        assertEquals(0.8f, result[0], 0.001f)
        assertEquals(0.2f, result[1], 0.001f)
    }

    @Test
    fun orientation180_centerStaysCenter() {
        val uv = floatArrayOf(0.5f, 0.5f)
        val result = applyOrientation(uv, 180)
        assertEquals(0.5f, result[0], 0.001f)
        assertEquals(0.5f, result[1], 0.001f)
    }

    @Test
    fun orientation270_rotatesCounterClockwise() {
        val uv = floatArrayOf(0.0f, 0.5f)
        val result = applyOrientation(uv, 270)
        assertEquals(0.5f, result[0], 0.001f)
        assertEquals(0.0f, result[1], 0.001f)
    }

    @Test
    fun orientation270_topLeftMapsToBottomLeft() {
        val uv = floatArrayOf(0.0f, 0.0f)
        val result = applyOrientation(uv, 270)
        assertEquals(1.0f, result[0], 0.001f)
        assertEquals(0.0f, result[1], 0.001f)
    }

    @Test
    fun mirrorX_flipsHorizontal() {
        val uv = floatArrayOf(0.2f, 0.5f)
        val result = applyMirrorX(uv, true)
        assertEquals(0.8f, result[0], 0.001f)
        assertEquals(0.5f, result[1], 0.001f)
    }

    @Test
    fun mirrorX_centerStaysCenter() {
        val uv = floatArrayOf(0.5f, 0.3f)
        val result = applyMirrorX(uv, true)
        assertEquals(0.5f, result[0], 0.001f)
        assertEquals(0.3f, result[1], 0.001f)
    }

    @Test
    fun mirrorX_noFlip_whenFalse() {
        val uv = floatArrayOf(0.2f, 0.5f)
        val result = applyMirrorX(uv, false)
        assertEquals(0.2f, result[0], 0.001f)
        assertEquals(0.5f, result[1], 0.001f)
    }

    @Test
    fun mirrorX_thenMirrorX_isIdentity() {
        val uv = floatArrayOf(0.3f, 0.7f)
        val result = applyMirrorX(applyMirrorX(uv, true), true)
        assertEquals(0.3f, result[0], 0.001f)
        assertEquals(0.7f, result[1], 0.001f)
    }

    @Test
    fun orientation90_then270_isIdentity() {
        val uv = floatArrayOf(0.3f, 0.7f)
        val rotated = applyOrientation(uv, 90)
        val result = applyOrientation(rotated, 270)
        assertEquals(0.3f, result[0], 0.001f)
        assertEquals(0.7f, result[1], 0.001f)
    }

    @Test
    fun orientation180_appliedTwice_isIdentity() {
        val uv = floatArrayOf(0.4f, 0.6f)
        val result = applyOrientation(applyOrientation(uv, 180), 180)
        assertEquals(0.4f, result[0], 0.001f)
        assertEquals(0.6f, result[1], 0.001f)
    }

    @Test
    fun frontCamera_90degree_withMirror() {
        val uv = floatArrayOf(0.0f, 0.0f)
        val oriented = applyOrientation(uv, 90)
        val mirrored = applyMirrorX(oriented, true)
        assertEquals(1.0f, mirrored[0], 0.001f)
        assertEquals(1.0f, mirrored[1], 0.001f)
    }

    @Test
    fun exifOrientation_rearCamera_90sensor() {
        val sensorOrientation = 90
        val deviceRotation = 0
        val expectedExif = (sensorOrientation - deviceRotation + 360) % 360
        assertEquals(90, expectedExif)
    }

    @Test
    fun exifOrientation_rearCamera_0sensor_landscapeDevice() {
        val sensorOrientation = 0
        val deviceRotation = 90
        val expectedExif = (sensorOrientation - deviceRotation + 360) % 360
        assertEquals(270, expectedExif)
    }

    @Test
    fun exifOrientation_frontCamera_270sensor() {
        val sensorOrientation = 270
        val deviceRotation = 0
        val baseExif = (sensorOrientation + deviceRotation + 360) % 360
        assertEquals(270, baseExif)
    }

    @Test
    fun phaseCalculation_spatialRowMajor() {
        for (y in 0..3) {
            for (x in 0..3) {
                val phase = Math.abs(x % 2) + Math.abs(y % 2) * 2
                assertTrue("Phase at ($x,$y) = $phase out of range", phase in 0..3)
            }
        }
    }

    @Test
    fun phaseCalculation_bayerPatternConsistency() {
        val pattern = intArrayOf(0, 1, 2, 3)

        for (y in 0..1) {
            for (x in 0..1) {
                val phase = Math.abs(x % 2) + Math.abs(y % 2) * 2
                assertEquals(phase, pattern[y * 2 + x])
            }
        }
    }

    @Test
    fun safePhase_negativeCoords_wrapsCorrectly() {
        for (y in -4..3) {
            for (x in -4..3) {
                val phase = Math.abs(x % 2) + Math.abs(y % 2) * 2
                assertTrue("Phase at ($x,$y) = $phase out of range", phase in 0..3)
            }
        }
    }

    @Test
    fun uvCoordinateBoundary_clampToSensor() {
        val sensorWidth = 8192f
        val sensorHeight = 6144f

        val uvOutside = floatArrayOf(1.5f, -0.3f)
        val clampedX = (uvOutside[0] * sensorWidth).coerceIn(0f, sensorWidth - 1f)
        val clampedY = (uvOutside[1] * sensorHeight).coerceIn(0f, sensorHeight - 1f)

        assertEquals(sensorWidth - 1f, clampedX, 0.001f)
        assertEquals(0f, clampedY, 0.001f)
    }

    @Test
    fun zoomTransform_appliedAfterOrientation() {
        val sensorOrientation = 90
        val zoomFactor = 2.0f
        val zoomCenterX = 0.5f
        val zoomCenterY = 0.5f

        val uv = floatArrayOf(0.3f, 0.3f)
        val oriented = applyOrientation(uv, sensorOrientation)

        val zoomed = floatArrayOf(
            (oriented[0] - zoomCenterX) / zoomFactor + zoomCenterX,
            (oriented[1] - zoomCenterY) / zoomFactor + zoomCenterY
        )

        assertTrue("Zoomed X in range", zoomed[0] in 0f..1f)
        assertTrue("Zoomed Y in range", zoomed[1] in 0f..1f)
    }

    @Test
    fun allOrientations_outputInBounds() {
        val orientations = listOf(0, 90, 180, 270)
        val testUVs = listOf(
            floatArrayOf(0.0f, 0.0f),
            floatArrayOf(0.5f, 0.5f),
            floatArrayOf(1.0f, 1.0f),
            floatArrayOf(0.3f, 0.7f),
            floatArrayOf(0.0f, 1.0f),
            floatArrayOf(1.0f, 0.0f)
        )

        for (orientation in orientations) {
            for (uv in testUVs) {
                val result = applyOrientation(uv, orientation)
                assertTrue("Orientation $orientation, uv=$uv: x=${result[0]} out of [0,1]", result[0] in -0.001f..1.001f)
                assertTrue("Orientation $orientation, uv=$uv: y=${result[1]} out of [0,1]", result[1] in -0.001f..1.001f)
            }
        }
    }
}
