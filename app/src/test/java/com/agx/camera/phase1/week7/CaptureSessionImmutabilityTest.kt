package com.agx.camera.phase1.week7

import com.agx.camera.camera.FlashMode
import com.agx.camera.camera.LensState
import org.junit.Assert.*
import org.junit.Test

class CaptureSessionImmutabilityTest {

    private fun createDefaultSession() = com.agx.camera.MainActivity.CaptureSession(
        flashMode = FlashMode.OFF,
        jpegQuality = 95,
        resolutionWidth = 4080,
        resolutionHeight = 3060,
        deviceRotationDegrees = 0,
        sensorOrientation = 90,
        focalLengthMm = 6.7f,
        zoomFactor = 1.0f,
        zoomCenterX = 0.5f,
        zoomCenterY = 0.5f,
        agxSceneLinearTo709 = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f),
        agxInsetMat = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f),
        agxOutsetMat = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f),
        agxToRec2020 = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f),
        agxWhiteLevel = 1.0f,
        agxBlackLevel = 0.0f,
        agxLogMin = -6.0f,
        agxLogMax = 2.0f,
        agxLogMidgray = -2.0f,
        agxDisplayMidgray = 0.18f,
        agxContrast = 1.0f,
        agxToe = 0.5f,
        agxShoulder = 0.5f
    )

    @Test
    fun captureSession_copy_createsNewInstance() {
        val original = createDefaultSession()
        val copy = original.copy(flashMode = FlashMode.ON, jpegQuality = 85)

        assertNotSame(original, copy)
        assertEquals(FlashMode.ON, copy.flashMode)
        assertEquals(85, copy.jpegQuality)
        assertEquals(FlashMode.OFF, original.flashMode)
        assertEquals(95, original.jpegQuality)
    }

    @Test
    fun captureSession_equals_usesIdentityComparison() {
        val a = createDefaultSession()
        val b = createDefaultSession()

        assertEquals(a, a)
        assertNotEquals(a, b)
        assertNotEquals(a, null)
        assertNotEquals(a, "not a session")
    }

    @Test
    fun captureSession_hashCode_usesIdentityHashCode() {
        val a = createDefaultSession()
        val b = createDefaultSession()

        assertEquals(System.identityHashCode(a), a.hashCode())
        assertEquals(System.identityHashCode(b), b.hashCode())
        assertNotEquals(a.hashCode(), b.hashCode())
    }

    @Test
    fun captureSession_allPropertiesAreImmutable() {
        val session = createDefaultSession()

        assertEquals(FlashMode.OFF, session.flashMode)
        assertEquals(95, session.jpegQuality)
        assertEquals(4080, session.resolutionWidth)
        assertEquals(3060, session.resolutionHeight)
        assertEquals(0, session.deviceRotationDegrees)
        assertEquals(90, session.sensorOrientation)
        assertEquals(6.7f, session.focalLengthMm, 0.001f)
        assertEquals(1.0f, session.zoomFactor, 0.001f)
        assertEquals(0.5f, session.zoomCenterX, 0.001f)
        assertEquals(0.5f, session.zoomCenterY, 0.001f)
        assertEquals(1.0f, session.agxWhiteLevel, 0.001f)
        assertEquals(0.0f, session.agxBlackLevel, 0.001f)
        assertEquals(-6.0f, session.agxLogMin, 0.001f)
        assertEquals(2.0f, session.agxLogMax, 0.001f)
        assertEquals(-2.0f, session.agxLogMidgray, 0.001f)
        assertEquals(0.18f, session.agxDisplayMidgray, 0.001f)
        assertEquals(1.0f, session.agxContrast, 0.001f)
        assertEquals(0.5f, session.agxToe, 0.001f)
        assertEquals(0.5f, session.agxShoulder, 0.001f)
    }

    @Test
    fun captureSession_toString_containsAllFields() {
        val session = createDefaultSession()
        val str = session.toString()

        assertTrue(str.contains("flashMode"))
        assertTrue(str.contains("jpegQuality"))
        assertTrue(str.contains("resolutionWidth"))
        assertTrue(str.contains("resolutionHeight"))
        assertTrue(str.contains("agxContrast"))
    }

    @Test
    fun lensState_defaultValues() {
        val state = LensState()

        assertEquals(1.0f, state.zoomFactor, 0.001f)
        assertEquals(0.5f, state.zoomCenterX, 0.001f)
        assertEquals(0.5f, state.zoomCenterY, 0.001f)
        assertEquals(0, state.wbModeOrdinal)
        assertEquals(5500f, state.kelvin, 0.001f)
        assertEquals(0f, state.kelvinTint, 0.001f)
        assertEquals(0, state.flashModeOrdinal)
    }

    @Test
    fun lensState_copy_createsNewInstance() {
        val original = LensState()
        val copy = original.copy(zoomFactor = 2.0f, kelvin = 3200f)

        assertNotSame(original, copy)
        assertEquals(2.0f, copy.zoomFactor, 0.001f)
        assertEquals(3200f, copy.kelvin, 0.001f)
        assertEquals(1.0f, original.zoomFactor, 0.001f)
        assertEquals(5500f, original.kelvin, 0.001f)
    }

    @Test
    fun lensState_equals_isStructural() {
        val a = LensState(zoomFactor = 2.0f, kelvin = 3200f)
        val b = LensState(zoomFactor = 2.0f, kelvin = 3200f)
        val c = LensState(zoomFactor = 3.0f, kelvin = 3200f)

        assertEquals(a, b)
        assertNotEquals(a, c)
    }

    @Test
    fun lensState_hashCode_isStructural() {
        val a = LensState(zoomFactor = 2.0f, kelvin = 3200f)
        val b = LensState(zoomFactor = 2.0f, kelvin = 3200f)

        assertEquals(a.hashCode(), b.hashCode())
    }

    @Test
    fun lensState_toString_containsAllFields() {
        val state = LensState(zoomFactor = 1.5f, kelvin = 4000f)
        val str = state.toString()

        assertTrue(str.contains("zoomFactor"))
        assertTrue(str.contains("kelvin"))
        assertTrue(str.contains("wbModeOrdinal"))
        assertTrue(str.contains("flashModeOrdinal"))
    }

    @Test
    fun captureSession_withAllFlashModes() {
        for (mode in FlashMode.entries) {
            val session = createDefaultSession().copy(flashMode = mode)
            assertEquals(mode, session.flashMode)
        }
    }
}
