package com.agx.camera.vulkan

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Gate-2 encoder-surface probe. Creates a 4K AVC encoder input Surface and
 * asks Vulkan whether it can be used as a VkSurfaceKHR. Temporary.
 */
object EncoderSurfaceProbe {
    fun run(): String {
        val lines = StringBuilder()
        probe(3840, 2160, lines, uhd = false)
        return lines.toString()
    }

    fun runUhd(): String {
        val lines = StringBuilder()
        probe(3840, 2160, lines, uhd = true)
        return lines.toString()
    }

    private fun probe(width: Int, height: Int, lines: StringBuilder, uhd: Boolean) {
        var codec: MediaCodec? = null
        var surface: android.view.Surface? = null
        try {
            codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            val format = MediaFormat.createVideoFormat(
                MediaFormat.MIMETYPE_VIDEO_AVC, width, height
            )
            format.setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface
            )
            format.setInteger(MediaFormat.KEY_BIT_RATE, 30_000_000)
            format.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            val input = codec.createInputSurface()
            surface = input
            codec.start()
            val c = codec
            lines.append("encoder=${width}x${height} surface=ok\n")
            lines.append(VulkanProbe.probeSurface(input))
            val stop = AtomicBoolean(false)
            val drain = Thread({
                val info = MediaCodec.BufferInfo()
                try {
                    while (!stop.get()) {
                        val index = c.dequeueOutputBuffer(info, 10_000)
                        if (index >= 0) {
                            c.releaseOutputBuffer(index, false)
                        }
                    }
                } catch (_: Throwable) {
                    // The codec can be torn down while the loop is waiting.
                }
            }, "vulkan-encoder-drain")
            drain.start()
            try {
                lines.append(
                    if (uhd) VulkanProbe.runUhdProbe(input, 30, true)
                    else VulkanProbe.runEncoderProbe(input, 30, false)
                )
            } finally {
                stop.set(true)
                drain.join(2000)
            }
        } catch (t: Throwable) {
            lines.append(
                "encoder=${width}x${height} failed: " +
                    "${t.javaClass.simpleName}: ${t.message}\n"
            )
        } finally {
            runCatching { codec?.stop() }
            surface?.release()
            runCatching { codec?.release() }
        }
    }
}
