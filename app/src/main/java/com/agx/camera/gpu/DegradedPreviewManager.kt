package com.agx.camera.gpu

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.Debug
import android.util.Log

class DegradedPreviewManager(private val context: Context) {

    enum class PreviewMode {
        FULL,
        REDUCED_RESOLUTION,
        THROTTLED_FPS,
        CPU_DEMOSAIC_HYBRID
    }

    var activeMode: PreviewMode = PreviewMode.FULL
        private set

    var reducedWidth: Int = 320
        private set
    var reducedHeight: Int = 240
        private set

    var targetFps: Float = 30f
        private set

    var degradationBanner: String? = null
        private set

    private val frameTimes = mutableListOf<Long>()
    private var consecutiveSlowFrames = 0
    private val thermalThresholdConsecutiveSlow = 5
    private val thermalThresholdFrameMultiplier = 1.5f
    private var baselineFrameTimeMs: Float = 33.3f

    private var zeroCopyPathActive = false
    private var cpuCopyActive = false

    fun setBaselineFrameTime(ms: Float) {
        baselineFrameTimeMs = ms
    }

    fun reportFrameTime(frameTimeMs: Long) {
        synchronized(frameTimes) {
            frameTimes.add(frameTimeMs)
            if (frameTimes.size > 60) {
                frameTimes.removeAt(0)
            }
        }

        if (frameTimeMs > baselineFrameTimeMs * thermalThresholdFrameMultiplier) {
            consecutiveSlowFrames++
        } else {
            consecutiveSlowFrames = 0
        }
    }

    fun setUploadPath(zeroCopy: Boolean, cpuCopy: Boolean) {
        zeroCopyPathActive = zeroCopy
        cpuCopyActive = cpuCopy
    }

    fun evaluateMode(
        sensorWidth: Int,
        sensorHeight: Int,
        previewWidth: Int,
        previewHeight: Int
    ): PreviewMode {
        val avgFrameTime = getAverageFrameTime()
        val memInfo = ActivityManager.MemoryInfo()
        val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        actManager.getMemoryInfo(memInfo)

        val availableMemoryMB = memInfo.availMem / (1024 * 1024)

        if (activeMode == PreviewMode.FULL && cpuCopyActive) {
            if (availableMemoryMB < 256) {
                activeMode = PreviewMode.CPU_DEMOSAIC_HYBRID
                degradationBanner = "Preview performance reduced. Full-resolution capture unaffected."
                Log.w(TAG, "Degraded to CPU demosaic hybrid (memory critical: ${availableMemoryMB}MB)")
                return activeMode
            }

            if (availableMemoryMB < 512) {
                activeMode = PreviewMode.REDUCED_RESOLUTION
                reducedWidth = 320
                reducedHeight = 240
                degradationBanner = "Preview performance reduced. Full-resolution capture unaffected."
                Log.w(TAG, "Degraded to reduced resolution: ${reducedWidth}x${reducedHeight}")
                return activeMode
            }

            if (consecutiveSlowFrames >= thermalThresholdConsecutiveSlow) {
                if (availableMemoryMB < 1024) {
                    activeMode = PreviewMode.REDUCED_RESOLUTION
                    reducedWidth = 320
                    reducedHeight = 240
                } else if (avgFrameTime > baselineFrameTimeMs * 2.5f) {
                    activeMode = PreviewMode.CPU_DEMOSAIC_HYBRID
                    Log.w(TAG, "Degraded to CPU demosaic hybrid (sustained thermal: ${String.format("%.1f", avgFrameTime)}ms)")
                } else {
                    activeMode = PreviewMode.THROTTLED_FPS
                    targetFps = 15f
                }
                degradationBanner = "Preview performance reduced. Full-resolution capture unaffected."
                Log.w(TAG, "Degraded to ${activeMode.name}")
                return activeMode
            }
        }

        if (activeMode == PreviewMode.THROTTLED_FPS) {
            if (consecutiveSlowFrames == 0 && avgFrameTime < baselineFrameTimeMs * 1.2f) {
                activeMode = PreviewMode.FULL
                degradationBanner = null
                targetFps = 30f
                Log.d(TAG, "Recovered to full preview")
            }
        }

        if (activeMode == PreviewMode.REDUCED_RESOLUTION) {
            if (consecutiveSlowFrames == 0 && avgFrameTime < baselineFrameTimeMs * 1.2f) {
                activeMode = PreviewMode.FULL
                degradationBanner = null
                Log.d(TAG, "Recovered to full preview")
            }
        }

        if (activeMode == PreviewMode.CPU_DEMOSAIC_HYBRID) {
            if (consecutiveSlowFrames == 0 && avgFrameTime < baselineFrameTimeMs * 1.2f &&
                availableMemoryMB > 512) {
                activeMode = PreviewMode.FULL
                degradationBanner = null
                Log.d(TAG, "Recovered to full preview from CPU hybrid")
            }
        }

        return activeMode
    }

    fun getPreviewDimensions(
        sensorWidth: Int,
        sensorHeight: Int,
        standardPreviewWidth: Int,
        standardPreviewHeight: Int
    ): Pair<Int, Int> {
        return when (activeMode) {
            PreviewMode.FULL -> Pair(standardPreviewWidth, standardPreviewHeight)
            PreviewMode.REDUCED_RESOLUTION -> Pair(reducedWidth, reducedHeight)
            PreviewMode.THROTTLED_FPS -> Pair(standardPreviewWidth, standardPreviewHeight)
            PreviewMode.CPU_DEMOSAIC_HYBRID -> Pair(standardPreviewWidth, standardPreviewHeight)
        }
    }

    fun shouldThrottleFrame(): Boolean {
        if (activeMode != PreviewMode.THROTTLED_FPS) return false
        val avgFrameTime = getAverageFrameTime()
        return avgFrameTime > (1000f / targetFps)
    }

    fun getCpuDemosaicEnabled(): Boolean {
        return activeMode == PreviewMode.CPU_DEMOSAIC_HYBRID
    }

    fun reset() {
        activeMode = PreviewMode.FULL
        degradationBanner = null
        consecutiveSlowFrames = 0
        synchronized(frameTimes) {
            frameTimes.clear()
        }
        targetFps = 30f
    }

    private fun getAverageFrameTime(): Float {
        synchronized(frameTimes) {
            if (frameTimes.isEmpty()) return 33.3f
            return frameTimes.average().toFloat()
        }
    }

    companion object {
        private const val TAG = "DegradedPreviewManager"
    }
}
