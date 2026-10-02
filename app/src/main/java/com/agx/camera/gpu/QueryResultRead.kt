package com.agx.camera.gpu

/**
 * Reads one GL_TIME_ELAPSED result.
 *
 * GL_TIME_ELAPSED is a 64-bit nanosecond count, but the only query-result entry
 * point android.opengl exposes is GLES30.glGetQueryObjectuiv(int, int, int[],
 * int), and the platform's generated JNI passes that array to the 32-bit
 * glGetQueryObjectuiv(GLuint, GLenum, GLuint*). There is no long[] overload, no
 * bufSize overload and no 64-bit binding to prefer, so the high word is simply
 * not reachable from here.
 *
 * The consequence is that a pass longer than 2^32 ns, about 4.295 s, reads back
 * as its own wrapped value and is indistinguishable from a very short one. A
 * single pass cannot realistically take that long, and a segment of them still
 * lands in the right place, so the wrap is reported rather than worked around.
 * Reading two words would not help: the driver writes one word per element, so
 * the second word would always read back as zero.
 */
object QueryResultRead {

    /** Words per read: the platform binding writes one 32-bit value per element. */
    const val WORDS = 1

    /** A zeroed buffer of the width the driver writes. */
    fun allocate(): IntArray = IntArray(WORDS)

    /**
     * The result as an unsigned 32-bit count of nanoseconds, zero-extended into
     * a Long. See the class comment for the wrap this cannot see past.
     */
    fun read(words: IntArray): Long {
        require(words.size >= WORDS) {
            "query read needs $WORDS word, got ${words.size}"
        }
        return words[0].toLong() and 0xFFFFFFFFL
    }
}
