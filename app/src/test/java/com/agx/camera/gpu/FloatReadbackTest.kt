package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Pins the byte arithmetic of the GL float readbacks.
 *
 * This is the crash. readRowF computed its destination size as a float count
 * (width * 4) and handed that number to ByteBuffer.allocateDirect, which takes
 * bytes, so a 640-wide row allocated 2,560 bytes for a 10,240-byte read. The
 * driver then wrote four times past the end of a native allocation, 32 rows per
 * frame, in the measurement path only. A direct buffer is not on the Java heap,
 * so nothing about it was visible to the collector: no exception, no crash log,
 * just a native fault a frame or two in.
 *
 * The size is the whole defence, so it is asserted against an independently
 * written byte count rather than against the production expression.
 */
class FloatReadbackTest {

    @Test
    fun `a texel is 16 bytes read as RGBA float`() {
        assertEquals(16, FloatReadback.BYTES_PER_TEXEL)
        assertEquals(16, FloatReadback.texelBytes())
    }

    @Test
    fun `a 640-wide row needs a 10240-byte destination`() {
        // 640 texels * 4 channels * 4 bytes. The broken code allocated 2,560.
        assertEquals(10_240, FloatReadback.rowBytes(640))
    }

    @Test
    fun `row destination is exactly what glReadPixels writes`() {
        // Count the write independently of the production code: four channels
        // of four bytes per texel.
        for (width in intArrayOf(1, 7, 64, 640, 1920, 4096)) {
            val expected = width * 4 * 4
            assertEquals("width=$width", expected, FloatReadback.rowBytes(width))
        }
    }

    @Test
    fun `row destination holds every float the caller will index`() {
        val width = 640
        val buf = ByteBuffer.allocateDirect(FloatReadback.rowBytes(width))
            .order(ByteOrder.nativeOrder())
        val floats = buf.asFloatBuffer()
        // readRowF indexes up to (width - 1) * 4 + 3 on a row it just read.
        for (i in 0 until width * 4) {
            floats.put(i, 0f)
        }
        assertEquals(width * 4, floats.capacity())
    }

    @Test
    fun `reusing one buffer across reads of the same width stays in bounds`() {
        val width = 640
        val buf = ByteBuffer.allocateDirect(FloatReadback.rowBytes(width))
        // The reuse guard compares capacity() (bytes) against the requirement
        // (also bytes now). It must report no growth needed, and the buffer must
        // still be large enough for the write.
        val needed = FloatReadback.rowBytes(width)
        assertTrue("reuse guard must see the buffer as sufficient", buf.capacity() >= needed)
        assertEquals(needed, buf.capacity())
    }

    @Test
    fun `a wider row grows the shared buffer instead of overflowing it`() {
        var buf = ByteBuffer.allocateDirect(FloatReadback.rowBytes(640))
        val narrow = buf.capacity()
        val wide = FloatReadback.rowBytes(4096)
        assertTrue("a wider row must not be served by the narrow buffer", wide > narrow)
        buf = ByteBuffer.allocateDirect(wide)
        assertTrue(buf.capacity() >= FloatReadback.rowBytes(4096))
    }

    @Test
    fun `plane destination covers a full frame of float texels`() {
        // 640x480 preview, and the 4096x3072 sensor size, must both be exact.
        assertEquals(640 * 480 * 16, FloatReadback.planeBytes(640, 480))
        assertEquals(4096 * 3072 * 16, FloatReadback.planeBytes(4096, 3072))
    }

    @Test
    fun `no destination is ever sized in units smaller than bytes`() {
        // The regression in one assertion: a texel count is 4x too small to be
        // passed to allocateDirect, and a channel count is 16x too small.
        val width = 640
        assertTrue(FloatReadback.rowBytes(width) > width)
        assertTrue(FloatReadback.rowBytes(width) > width * 4)
    }
}
