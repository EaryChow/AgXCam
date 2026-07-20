package com.agx.camera.camera

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import android.widget.LinearLayout
import android.widget.Switch
import android.widget.TextView
import com.agx.camera.BuildConfig

class DeveloperSwitch(
    private val context: Context,
    private val onToggle: (Boolean) -> Unit
) {

    var useRawSensor: Boolean = false
        private set

    var rawSensorAvailable: Boolean = false
        private set

    private var container: LinearLayout? = null
    private var toggle: Switch? = null
    private var banner: TextView? = null
    private var prefs: SharedPreferences? = null

    fun init(
        containerLayout: LinearLayout,
        toggleSwitch: Switch,
        bannerView: TextView
    ) {
        container = containerLayout
        toggle = toggleSwitch
        banner = bannerView

        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        useRawSensor = prefs?.getBoolean(PREF_RAW_TOGGLE, false) ?: false

        if (BuildConfig.DEBUG) {
            containerLayout.visibility = LinearLayout.VISIBLE
        } else {
            containerLayout.visibility = LinearLayout.VISIBLE
            bannerView.visibility = TextView.GONE
        }

        toggleSwitch.setOnCheckedChangeListener { _, isChecked ->
            useRawSensor = isChecked
            prefs?.edit()?.putBoolean(PREF_RAW_TOGGLE, isChecked)?.apply()
            Log.d(TAG, "RAW_SENSOR toggle: $isChecked")
            onToggle(isChecked)
        }
    }

    fun setRawSensorAvailable(available: Boolean) {
        rawSensorAvailable = available
        toggle?.let { sw ->
            sw.isEnabled = available
            if (!available) {
                sw.isChecked = false
                useRawSensor = false
            }
        }
        banner?.let { tv ->
            if (available) {
                tv.text = if (useRawSensor) "RAW_SENSOR mode active" else "YUV fallback mode"
                tv.setTextColor(0xFF00AAFF.toInt())
            } else {
                tv.text = "RAW sensor access unavailable for this device"
                tv.setTextColor(0xFFFFAA00.toInt())
            }
            tv.visibility = TextView.VISIBLE
        }
    }

    fun updateBannerForRawMode(active: Boolean) {
        banner?.let { tv ->
            if (rawSensorAvailable) {
                tv.text = if (active) "RAW_SENSOR mode active" else "YUV fallback mode"
                tv.setTextColor(if (active) 0xFF00CC00.toInt() else 0xFF00AAFF.toInt())
                tv.visibility = TextView.VISIBLE
            }
        }
    }

    fun showErrorBanner(message: String) {
        banner?.let { tv ->
            tv.text = message
            tv.setTextColor(0xFFFF4444.toInt())
            tv.visibility = TextView.VISIBLE
        }
    }

    companion object {
        private const val TAG = "DeveloperSwitch"
        private const val PREFS_NAME = "agxcam_developer"
        private const val PREF_RAW_TOGGLE = "use_raw_sensor"
    }
}
