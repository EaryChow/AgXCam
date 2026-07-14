package com.agx.camera.phase1.week7

import android.content.Context
import com.agx.camera.camera.Camera2Manager
import org.junit.Assert.*
import org.junit.Test
import org.mockito.Mockito.*

class ResourceCleanupTest {

    private fun createManager(): Camera2Manager {
        val context = mock(Context::class.java)
        return Camera2Manager(context)
    }

    @Test
    fun close_setsAllFieldsToNull() {
        val manager = createManager()

        manager.close()

        assertNull(getField(manager, "cameraDevice"))
        assertNull(getField(manager, "captureSession"))
        assertNull(getField(manager, "previewReader"))
        assertNull(getField(manager, "captureReader"))
    }

    @Test
    fun close_isIdempotent() {
        val manager = createManager()

        manager.close()
        manager.close()
        manager.close()

        assertNull(getField(manager, "cameraDevice"))
        assertNull(getField(manager, "captureSession"))
    }

    @Test
    fun resetHard_setsAllFieldsToNull() {
        val manager = createManager()

        manager.resetHard()

        assertNull(getField(manager, "cameraDevice"))
        assertNull(getField(manager, "captureSession"))
        assertNull(getField(manager, "previewReader"))
        assertNull(getField(manager, "captureReader"))
    }

    @Test
    fun resetHard_isIdempotent() {
        val manager = createManager()

        manager.resetHard()
        manager.resetHard()

        assertNull(getField(manager, "cameraDevice"))
    }

    @Test
    fun close_afterResetHard_doesNotThrow() {
        val manager = createManager()

        manager.resetHard()
        manager.close()

        assertNull(getField(manager, "cameraDevice"))
    }

    @Test
    fun startStopBackgroundThread_isClean() {
        val manager = createManager()

        manager.startBackgroundThread()
        assertNotNull(getField(manager, "backgroundThread"))
        assertNotNull(getField(manager, "backgroundHandler"))

        manager.stopBackgroundThread()
        assertNull(getField(manager, "backgroundThread"))
        assertNull(getField(manager, "backgroundHandler"))
    }

    @Test
    fun stopBackgroundThread_isIdempotent() {
        val manager = createManager()

        manager.startBackgroundThread()
        manager.stopBackgroundThread()
        manager.stopBackgroundThread()

        assertNull(getField(manager, "backgroundThread"))
    }

    @Test
    fun flashModeDefault_isOff() {
        val manager = createManager()
        assertEquals(com.agx.camera.camera.FlashMode.OFF, manager.currentFlashModeForExif)
    }

    @Test
    fun finallyBlockEnsuresCleanup_close() {
        var closeCompleted = false
        val manager = createManager()

        try {
            manager.startBackgroundThread()
        } finally {
            manager.close()
            manager.stopBackgroundThread()
            closeCompleted = true
        }

        assertTrue(closeCompleted)
        assertNull(getField(manager, "cameraDevice"))
        assertNull(getField(manager, "backgroundThread"))
    }

    @Test
    fun finallyBlockEnsuresCleanup_resetHard() {
        var resetCompleted = false
        val manager = createManager()

        try {
            manager.startBackgroundThread()
        } finally {
            manager.resetHard()
            manager.stopBackgroundThread()
            resetCompleted = true
        }

        assertTrue(resetCompleted)
        assertNull(getField(manager, "cameraDevice"))
        assertNull(getField(manager, "backgroundThread"))
    }

    private fun getField(obj: Any, fieldName: String): Any? {
        val field = obj.javaClass.getDeclaredField(fieldName)
        field.isAccessible = true
        return field.get(obj)
    }
}
