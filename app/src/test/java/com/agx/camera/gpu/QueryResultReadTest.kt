package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the shape of the GL timer query read.
 *
 * The read is 32-bit because that is all android.opengl binds: the generated JNI
 * for GLES30.glGetQueryObjectuiv passes the Java int[] to
 * glGetQueryObjectuiv(GLuint, GLenum, GLuint*), one 32-bit value per element.
 * There is no long[] overload and no bufSize overload to prefer.
 *
 * This test exists because the obvious-looking "fix" is to allocate two words and
 * claim the 64-bit result, which would be wrong twice over: it would read a
 * second word the driver never writes, and it would have papered over a real
 * crash that had nothing to do with this call. If a future SDK adds a 64-bit
 * binding, this fails and the read can be widened on evidence.
 */
class QueryResultReadTest {

    @Test
    fun `one word per read, matching the 32-bit platform binding`() {
        assertEquals(1, QueryResultRead.WORDS)
        assertEquals(1, QueryResultRead.allocate().size)
    }

    @Test
    fun `result is zero-extended, never read as a negative duration`() {
        val words = QueryResultRead.allocate()
        words[0] = -1
        assertEquals(4_294_967_295L, QueryResultRead.read(words))
    }

    @Test
    fun `a fresh buffer reads as zero`() {
        assertEquals(0L, QueryResultRead.read(QueryResultRead.allocate()))
    }

    @Test
    fun `sub-microsecond results survive the round trip`() {
        val words = QueryResultRead.allocate()
        for (nanos in longArrayOf(1L, 1_000L, 1_000_000L, 33_000_000L, 1_500_000_000L)) {
            words[0] = (nanos and 0xFFFFFFFFL).toInt()
            assertEquals(nanos, QueryResultRead.read(words))
        }
    }

    @Test
    fun `the wrap past 2^32 ns is documented rather than silently absorbed`() {
        // 4.295 s is where a 32-bit nanosecond count rolls over. The value
        // genuinely cannot be recovered through this binding, so the test records
        // the boundary instead of pretending it is handled.
        val wrapPoint = 1L shl 32
        assertEquals(4_294_967_296L, wrapPoint)
        assertTrue("wrap is at about 4.295 s", wrapPoint / 1_000_000_000.0 < 4.3)
    }

    @Test
    fun `platform still offers no 64-bit overload to switch to`() {
        val overloads = Class.forName("android.opengl.GLES30")
            .declaredMethods
            .filter { it.name == "glGetQueryObjectuiv" }
        assertTrue(
            "expected the int[] form to exist, found $overloads",
            overloads.any { it.parameterTypes.contains(IntArray::class.java) }
        )
        assertTrue(
            "a long[] or bufSize overload appeared; widen QueryResultRead: $overloads",
            overloads.none {
                it.parameterTypes.contains(LongArray::class.java) || it.parameterCount == 5
            }
        )
    }
}
