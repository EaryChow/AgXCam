package com.agx.camera.thermal

import android.app.Activity
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.agx.camera.BuildConfig
import com.agx.camera.CrashLogger
import com.agx.camera.vulkan.EncoderSurfaceProbe
import java.util.Locale

/**
 * Probe-only thermal runner.
 *
 * This activity never creates a camera session, a Surface, or the GLES preview.
 * It runs the fixed-path baseline at 3840x2160 and 30 fps (demosaic -> AgX ->
 * encoder surface, no S1/S2/S5) in a native loop and logs battery temperature
 * alongside frame cadence. It is reachable only in a Vulkan probe APK
 * (-PenableVulkanProbe).
 */
class ThermalProbeActivity : Activity() {
    private lateinit var thermalManager: ThermalManager
    private lateinit var countdown: TextView
    private lateinit var status: TextView
    private lateinit var plateauButton: Button
    private lateinit var enduranceButton: Button
    private lateinit var saveButton: Button
    private val uiHandler = Handler(Looper.getMainLooper())
    private var lastThermalLogMs = 0L
    private var runStarted = false
    private var runStartMs = 0L
    private var runSeconds = 0
    @Volatile private var finished = false
    private val countdownRunnable = object : Runnable {
        override fun run() {
            if (finished) return
            val elapsed = ((SystemClock.elapsedRealtime() - runStartMs) / 1000L).toInt()
            val remaining = (runSeconds - elapsed).coerceAtLeast(0)
            val h = remaining / 3600
            val m = (remaining % 3600) / 60
            val s = remaining % 60
            countdown.text = String.format(Locale.US, "%02d:%02d:%02d", h, m, s)
            if (remaining > 0) uiHandler.postDelayed(this, 1000L)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!BuildConfig.AGX_VULKAN_PROBE) {
            finish()
            return
        }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(Color.BLACK)
            setPadding(48, 48, 48, 48)
        }
        val title = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 21f
            gravity = Gravity.CENTER
            text = "AgX Thermal Probe\nfixed-path baseline 4K30\nno S1 / S2 / S5"
        }
        countdown = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 34f
            gravity = Gravity.CENTER
            setPadding(0, 14, 0, 4)
            text = "00:00:00"
        }
        status = TextView(this).apply {
            setTextColor(Color.LTGRAY)
            textSize = 16f
            gravity = Gravity.CENTER
            setPadding(0, 4, 0, 14)
            text = "choose a run"
        }
        plateauButton = Button(this).apply {
            text = "20 min baseline"
            setOnClickListener { startBaseline(20 * 60) }
        }
        enduranceButton = Button(this).apply {
            text = "2 h baseline"
            setOnClickListener { startBaseline(2 * 60 * 60) }
        }
        saveButton = Button(this).apply {
            text = "Save log"
            setOnClickListener { saveLog() }
        }
        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        buttonRow.addView(plateauButton)
        buttonRow.addView(enduranceButton)
        buttonRow.addView(saveButton)
        root.addView(title)
        root.addView(countdown)
        root.addView(status)
        root.addView(buttonRow)
        val scroll = ScrollView(this).apply {
            setBackgroundColor(Color.BLACK)
            isFillViewport = true
        }
        scroll.addView(root)
        setContentView(scroll)

        thermalManager = ThermalManager(this).also { tm ->
            tm.thermalProtectionEnabledProvider = { false }
            tm.onStateChanged = { state ->
                CrashLogger.log("Thermal", "probe state=${state.name}")
            }
            tm.onTemperatureUpdate = { displayC, judgedC ->
                val now = SystemClock.elapsedRealtime()
                if (lastThermalLogMs == 0L || now - lastThermalLogMs >= 10_000L) {
                    lastThermalLogMs = now
                    CrashLogger.log(
                        "Thermal",
                        String.format(
                            Locale.US,
                            "battery=%.1fC judged=%.1fC state=%s protection=false",
                            displayC, judgedC, tm.currentState.name
                        )
                    )
                }
                if (!finished) {
                    runOnUiThread {
                        status.text = String.format(
                            Locale.US,
                            "%s\nbattery %.1fC\nstate %s",
                            if (runStarted) "running" else "ready",
                            displayC, tm.currentState.name
                        )
                    }
                }
            }
        }
        thermalManager.start()

        val autoSeconds = intent.getIntExtra("seconds", 0)
        if (autoSeconds > 0) {
            window.decorView.post { startBaseline(autoSeconds) }
        }
    }

    private fun saveLog() {
        status.text = "saving log"
        Thread({
            try {
                CrashLogger.saveDebugLogToDownloads(this)
                runOnUiThread { status.text = "log saved to Downloads" }
            } catch (t: Throwable) {
                CrashLogger.logException("VulkanThermal", t)
                runOnUiThread {
                    status.text = "save failed: ${t.javaClass.simpleName}: ${t.message}"
                }
            }
        }, "VulkanThermalSave").start()
    }

    private fun startBaseline(seconds: Int) {
        startRun(seconds) { EncoderSurfaceProbe.runBaselineThermal(seconds) }
    }

    private fun startRun(seconds: Int, runner: () -> String) {
        if (runStarted) return
        runStarted = true
        plateauButton.isEnabled = false
        enduranceButton.isEnabled = false
        status.text = String.format(Locale.US, "running %d s", seconds)
        runStartMs = SystemClock.elapsedRealtime()
        runSeconds = seconds
        uiHandler.removeCallbacks(countdownRunnable)
        countdownRunnable.run()
        CrashLogger.log("VulkanThermal", "start fixed-path baseline 3840x2160@30 seconds=$seconds")
        ThermalRunGuard.activate()

        Thread({
            try {
                val result = runner()
                CrashLogger.log("VulkanThermal", result)
                CrashLogger.saveDebugLogToDownloads(this)
                finished = true
                runOnUiThread {
                    uiHandler.removeCallbacks(countdownRunnable)
                    countdown.text = "00:00:00"
                    status.text = "finished\n$result\nlog saved to Downloads"
                }
            } catch (t: Throwable) {
                CrashLogger.logException("VulkanThermal", t)
                CrashLogger.saveDebugLogToDownloads(this)
                finished = true
                runOnUiThread {
                    uiHandler.removeCallbacks(countdownRunnable)
                    countdown.text = "00:00:00"
                    status.text = "failed: ${t.javaClass.simpleName}: ${t.message}"
                }
            } finally {
                ThermalRunGuard.deactivate()
            }
        }, "VulkanThermal").start()
    }

    override fun onDestroy() {
        uiHandler.removeCallbacks(countdownRunnable)
        if (::thermalManager.isInitialized) thermalManager.stop()
        super.onDestroy()
    }
}
