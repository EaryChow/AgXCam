package com.agx.camera.color

object ColorMatrix {

    val REC709 = Chromaticities(
        0.64f, 0.33f,
        0.30f, 0.60f,
        0.15f, 0.06f,
        0.3127f, 0.3290f
    )

    val REC2020 = Chromaticities(
        0.708f, 0.292f,
        0.170f, 0.797f,
        0.131f, 0.046f,
        0.3127f, 0.3290f
    )

    val D65_X = 0.3127f
    val D65_Y = 0.3290f

    data class Mat3(
        val m: FloatArray = floatArrayOf(
            1f, 0f, 0f,
            0f, 1f, 0f,
            0f, 0f, 1f
        )
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Mat3) return false
            return m.contentEquals(other.m)
        }
        override fun hashCode(): Int = m.contentHashCode()
    }

    fun identity(): Mat3 = Mat3()

    fun multiply(a: Mat3, b: Mat3): Mat3 {
        val r = FloatArray(9)
        for (row in 0..2) {
            for (col in 0..2) {
                r[row * 3 + col] =
                    a.m[row * 3] * b.m[col] +
                    a.m[row * 3 + 1] * b.m[3 + col] +
                    a.m[row * 3 + 2] * b.m[6 + col]
            }
        }
        return Mat3(r)
    }

    fun mulMatVec(m: Mat3, v: FloatArray): FloatArray = floatArrayOf(
        m.m[0] * v[0] + m.m[1] * v[1] + m.m[2] * v[2],
        m.m[3] * v[0] + m.m[4] * v[1] + m.m[5] * v[2],
        m.m[6] * v[0] + m.m[7] * v[1] + m.m[8] * v[2]
    )

    fun diagonal(r: Float, g: Float, b: Float): Mat3 = Mat3(
        floatArrayOf(r, 0f, 0f, 0f, g, 0f, 0f, 0f, b)
    )

    fun transpose(m: Mat3): Mat3 = Mat3(
        floatArrayOf(
            m.m[0], m.m[3], m.m[6],
            m.m[1], m.m[4], m.m[7],
            m.m[2], m.m[5], m.m[8]
        )
    )

    fun inverse(m: Mat3): Mat3 {
        val a = m.m[0]; val b = m.m[1]; val c = m.m[2]
        val d = m.m[3]; val e = m.m[4]; val f = m.m[5]
        val g = m.m[6]; val h = m.m[7]; val i = m.m[8]

        val det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if (Math.abs(det) < 1e-10f) return identity()
        val invDet = 1.0f / det

        return Mat3(
            floatArrayOf(
                (e * i - f * h) * invDet,
                (c * h - b * i) * invDet,
                (b * f - c * e) * invDet,
                (f * g - d * i) * invDet,
                (a * i - c * g) * invDet,
                (c * d - a * f) * invDet,
                (d * h - e * g) * invDet,
                (b * g - a * h) * invDet,
                (a * e - b * d) * invDet
            )
        )
    }

    fun determinant(m: Mat3): Float {
        return m.m[0] * (m.m[4] * m.m[8] - m.m[5] * m.m[7]) -
               m.m[1] * (m.m[3] * m.m[8] - m.m[5] * m.m[6]) +
               m.m[2] * (m.m[3] * m.m[7] - m.m[4] * m.m[6])
    }

    fun rgbToXYZ(ch: Chromaticities): Mat3 {
        val m = floatArrayOf(
            ch.redX / ch.redY, ch.greenX / ch.greenY, ch.blueX / ch.blueY,
            1.0f, 1.0f, 1.0f,
            (1 - ch.redX - ch.redY) / ch.redY,
            (1 - ch.greenX - ch.greenY) / ch.greenY,
            (1 - ch.blueX - ch.blueY) / ch.blueY
        )
        val wx = ch.whiteX / ch.whiteY
        val wy = 1.0f
        val wz = (1 - ch.whiteX - ch.whiteY) / ch.whiteY

        val invM = inverse(Mat3(m))
        val sx = invM.m[0] * wx + invM.m[1] * wy + invM.m[2] * wz
        val sy = invM.m[3] * wx + invM.m[4] * wy + invM.m[5] * wz
        val sz = invM.m[6] * wx + invM.m[7] * wy + invM.m[8] * wz

        return Mat3(
            floatArrayOf(
                m[0] * sx, m[1] * sy, m[2] * sz,
                m[3] * sx, m[4] * sy, m[5] * sz,
                m[6] * sx, m[7] * sy, m[8] * sz
            )
        )
    }

    fun xyzToRGB(ch: Chromaticities): Mat3 = inverse(rgbToXYZ(ch))

    fun rgbToRGB(from: Chromaticities, to: Chromaticities): Mat3 {
        return multiply(xyzToRGB(to), rgbToXYZ(from))
    }

    fun colorSpaceTransformToMatrix(
        transform: android.hardware.camera2.params.ColorSpaceTransform?
    ): Mat3? {
        if (transform == null) return null
        return try {
            val method = transform.javaClass.getMethod("get", Int::class.javaPrimitiveType, Int::class.javaPrimitiveType)
            val m = FloatArray(9)
            for (row in 0 until 3) {
                for (col in 0 until 3) {
                    val r = method.invoke(transform, col, row) as android.util.Rational
                    m[row * 3 + col] = r.numerator.toFloat() / r.denominator.toFloat()
                }
            }
            val mat = Mat3(m)
            if (Math.abs(determinant(mat)) < 1e-6f) null else mat
        } catch (e: Exception) {
            null
        }
    }
}
