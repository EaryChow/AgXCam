package com.agx.camera.gpu

/**
 * Byte sizes for GL float readbacks.
 *
 * Every glReadPixels destination in the renderer is sized from here. That is not
 * tidiness: the two sides of a readback speak different units. allocateDirect
 * takes bytes, ByteBuffer.capacity() reports bytes, and glReadPixels writes
 * bytes, while the shape of a readback is naturally described in texels and
 * channels. Mixing the two silently under-allocates the destination by a factor
 * of four per float, and the driver then writes past the end of a native
 * allocation.
 *
 * A direct ByteBuffer does not live on the Java heap, so an overread of one
 * corrupts native memory without the collector ever seeing it. There is no
 * exception to catch and nothing for a crash handler to report: the process
 * simply dies of a native fault one or two frames later. The sizing therefore
 * lives behind named functions with unit-explicit names, and a test pins the
 * arithmetic.
 */
object FloatReadback {

    /** Bytes one RGBA texel occupies read back as GL_FLOAT: 4 channels x 4 bytes. */
    const val BYTES_PER_TEXEL = 16

    /**
     * Destination size for a readback of one RGBA32F row [width] texels wide.
     *
     * The row samplers read a full row and then stride across it on the CPU,
     * because one glReadPixels per sample point is far worse.
     */
    fun rowBytes(width: Int): Int = width * BYTES_PER_TEXEL

    /** Destination size for a full [width] x [height] RGBA32F plane. */
    fun planeBytes(width: Int, height: Int): Int = width * height * BYTES_PER_TEXEL

    /** Destination size for a single RGBA32F texel. */
    fun texelBytes(): Int = BYTES_PER_TEXEL
}
