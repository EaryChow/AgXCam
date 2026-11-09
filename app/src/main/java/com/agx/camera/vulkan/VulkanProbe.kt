package com.agx.camera.vulkan

/**
 * Vulkan native probe. The library is built only with
 * -PenableVulkanProbe; AGX_VULKAN_PROBE is false in normal builds.
 */
object VulkanProbe {
    init {
        System.loadLibrary("vulkan_probe")
    }

    external fun runProbe(width: Int, height: Int, frames: Int): String

    external fun runS5StatsProbe(width: Int, height: Int, frames: Int): String

    external fun probeSurface(surface: android.view.Surface): String

    external fun runEncoderProbe(
        surface: android.view.Surface,
        frames: Int,
        timestamps: Boolean
    ): String
}
