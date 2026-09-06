package com.agx.camera.phase1.week7

import android.content.Context
import android.widget.LinearLayout
import android.widget.Switch
import android.widget.TextView
import com.agx.camera.camera.DeveloperSwitch
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.Mockito.*

class DevSwitchGrayOutTest {

    private lateinit var devSwitch: DeveloperSwitch
    private lateinit var mockContext: Context
    private lateinit var mockContainer: LinearLayout
    private lateinit var mockToggle: Switch
    private lateinit var mockBanner: TextView

    @Before
    fun setUp() {
        mockContext = mock(Context::class.java)
        val mockPrefs = mock(android.content.SharedPreferences::class.java)
        val mockEditor = mock(android.content.SharedPreferences.Editor::class.java)
        `when`(mockContext.getSharedPreferences(anyString(), anyInt())).thenReturn(mockPrefs)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putBoolean(anyString(), anyBoolean())).thenReturn(mockEditor)

        mockContainer = mock(LinearLayout::class.java)
        mockToggle = mock(Switch::class.java)
        mockBanner = mock(TextView::class.java)

        devSwitch = DeveloperSwitch(mockContext) {}
    }

    @Test
    fun initialRawSensorAvailable_isFalse() {
        assertFalse(devSwitch.rawSensorAvailable)
    }

    @Test
    fun initialUseRawSensor_isFalse() {
        assertFalse(devSwitch.useRawSensor)
    }

    @Test
    fun setRawSensorAvailable_true_setsFlag() {
        devSwitch.setRawSensorAvailable(true)
        assertTrue(devSwitch.rawSensorAvailable)
    }

    @Test
    fun setRawSensorAvailable_false_setsFlag() {
        devSwitch.setRawSensorAvailable(true)
        devSwitch.setRawSensorAvailable(false)
        assertFalse(devSwitch.rawSensorAvailable)
    }

    @Test
    fun setRawSensorAvailable_false_disablesToggle() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)
        devSwitch.setRawSensorAvailable(true)
        devSwitch.setRawSensorAvailable(false)

        verify(mockToggle).isEnabled = false
    }

    @Test
    fun setRawSensorAvailable_false_unchecksToggle() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)
        clearInvocations(mockToggle)
        devSwitch.setRawSensorAvailable(true)
        devSwitch.setRawSensorAvailable(false)

        verify(mockToggle).isChecked = false
    }

    @Test
    fun setRawSensorAvailable_true_enablesToggle() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)
        devSwitch.setRawSensorAvailable(true)

        verify(mockToggle).isEnabled = true
    }

    @Test
    fun setRawSensorAvailable_false_showsWarningBanner() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)
        devSwitch.setRawSensorAvailable(false)

        verify(mockBanner).text = "RAW sensor stream not supported by this device, YUV fallback mode active"
        verify(mockBanner).setTextColor(0xFFFFAA00.toInt())
        verify(mockBanner).visibility = TextView.VISIBLE
    }

    @Test
    fun setRawSensorAvailable_true_showsYuvFallbackBanner() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)
        devSwitch.setRawSensorAvailable(true)

        verify(mockBanner).text = "YUV fallback mode"
        verify(mockBanner).setTextColor(0xFF00AAFF.toInt())
        verify(mockBanner).visibility = TextView.VISIBLE
    }

    @Test
    fun setRawSensorAvailable_false_resetsUseRawSensor() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)

        devSwitch.setRawSensorAvailable(true)
        devSwitch.setRawSensorAvailable(false)

        assertFalse(devSwitch.useRawSensor)
    }

    @Test
    fun init_debugBuild_showsContainer() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)
        verify(mockContainer).visibility = LinearLayout.VISIBLE
    }

    @Test
    fun init_setsCheckedChangeListener() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)
        verify(mockToggle).setOnCheckedChangeListener(any())
    }

    @Test
    fun updateBannerForRawMode_active_showsActive() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)
        devSwitch.setRawSensorAvailable(true)
        reset(mockBanner)

        devSwitch.updateBannerForRawMode(true)

        verify(mockBanner).text = "RAW_SENSOR mode active"
        verify(mockBanner).setTextColor(0xFF00CC00.toInt())
        verify(mockBanner).visibility = TextView.VISIBLE
    }

    @Test
    fun updateBannerForRawMode_inactive_showsYuvFallback() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)
        devSwitch.setRawSensorAvailable(true)
        reset(mockBanner)

        devSwitch.updateBannerForRawMode(false)

        verify(mockBanner).text = "YUV fallback mode"
        verify(mockBanner).setTextColor(0xFF00AAFF.toInt())
    }

    @Test
    fun updateBannerForRawMode_notAvailable_doesNotUpdate() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)
        devSwitch.setRawSensorAvailable(false)
        reset(mockBanner)

        devSwitch.updateBannerForRawMode(true)

        verify(mockBanner, never()).text = anyString()
    }

    @Test
    fun showErrorBanner_showsMessage() {
        devSwitch.init(mockContainer, mockToggle, mockBanner)

        devSwitch.showErrorBanner("Test error message")

        verify(mockBanner).text = "Test error message"
        verify(mockBanner).setTextColor(0xFFFF4444.toInt())
        verify(mockBanner).visibility = TextView.VISIBLE
    }

    @Test
    fun setRawSensorAvailable_withoutInit_doesNotThrow() {
        devSwitch.setRawSensorAvailable(true)
        assertTrue(devSwitch.rawSensorAvailable)

        devSwitch.setRawSensorAvailable(false)
        assertFalse(devSwitch.rawSensorAvailable)
    }
}
