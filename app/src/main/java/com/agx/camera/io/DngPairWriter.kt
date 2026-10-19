package com.agx.camera.io

import com.agx.camera.CrashLogger
import com.agx.camera.camera.LensShadingData

/**
 * Writes the DNG half of a capture pair from the exact Bayer buffer the
 * pipeline processed.
 *
 * The file serves two consumers: standard readers (folded AsShotNeutral,
 * AsShotWhiteXY, the daylight-anchor corrected ColorMatrix1/2,
 * BaselineExposure, the GainMap opcode) and offline reproduction (private
 * tag 55001, the render recipe: the exact transform, AgX params, and NR
 * strengths the pipeline consumed).
 *
 * Every value written here is snapshot from the renderer's consumed state
 * at capture time, so when the pipeline's WB, color, AgX, NR, or shading
 * stages change, the DNG stays correct without touching this file. The one
 * case that needs a hand: a NEW stage consuming state this file does not
 * record - add it to the snapshots and to the recipe.
 *
 * The tag template is a DngCreator serialization of a neighboring raw
 * frame (the only on-device source of the OEM calibration matrices); the
 * pixel layout is replaced with an uncompressed single-strip mosaic of the
 * Bayer buffer.
 */
object DngPairWriter {

    private const val TAG = "DngPairWriter"

    fun write(
        bayer: java.nio.ByteBuffer,
        w: Int,
        h: Int,
        templateRaw: android.media.Image?,
        chars: android.hardware.camera2.CameraCharacteristics?,
        result: android.hardware.camera2.TotalCaptureResult?,
        exifOrientation: Int,
        baselineExposureEv: Float,
        thumbnailJpeg: ByteArray?,
        foldedNeutral: FloatArray?,
        sceneXy: FloatArray?,
        correctedCm: FloatArray?,
        renderRecipeJson: String?,
        shadingMap: LensShadingData?
    ): ByteArray? {
        if (templateRaw == null || chars == null || result == null) return null
        val template: ByteArray = try {
            val bos = java.io.ByteArrayOutputStream()
            android.hardware.camera2.DngCreator(chars, result).use { it.writeImage(bos, templateRaw) }
            bos.toByteArray()
        } catch (e: Exception) {
            android.util.Log.e(TAG, "DNG template failed", e)
            return null
        }
        val typeSize = intArrayOf(1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8)
        fun tu16(b: ByteArray, o: Int) =
            (b[o].toInt() and 0xff) or ((b[o + 1].toInt() and 0xff) shl 8)
        fun tu32(b: ByteArray, o: Int): Int {
            var v = 0
            for (k in 3 downTo 0) v = (v shl 8) or (b[o + k].toInt() and 0xff)
            return v
        }
        fun le16(v: Int) = byteArrayOf(v.toByte(), (v shr 8).toByte())
        fun le32(v: Int) = byteArrayOf(
            v.toByte(), (v shr 8).toByte(), (v shr 16).toByte(), (v shr 24).toByte()
        )
        fun rat(num: Long, den: Long) = le32(num.toInt()) + le32(den.toInt())
        fun w16(b: ByteArray, o: Int, v: Int) { b[o] = v.toByte(); b[o + 1] = (v shr 8).toByte() }
        fun w32(b: ByteArray, o: Int, v: Int) {
            for (k in 0..3) b[o + k] = (v shr (8 * k)).toByte()
        }
        class Tlv(val tag: Int, val type: Int, val count: Int, val data: ByteArray)
        val tlvs = ArrayList<Tlv>()
        return try {
            if (!(template[0] == 'I'.toByte() && template[1] == 'I'.toByte())) return null
            val ifd0 = tu32(template, 4)
            val n = tu16(template, ifd0)
            if (n <= 0 || ifd0 + 2 + n * 12 + 4 > template.size) return null
            // Pixel layout, orientation, white balance tags, calibration
            // matrices, and ForwardMatrix come from this writer, not the
            // template: the calibration is replaced with the daylight-anchor
            // corrected pair (the render whitens against the corrected
            // device truth, so the file must describe that same sensor), and
            // ForwardMatrix was calibrated against the uncorrected pair.
            val drop = setOf(
                256, 257, 258, 259, 262, 270, 273, 274, 277, 278, 279, 284, 317,
                322, 323, 324, 325, 330, 34665, 34853, 34855, 33434, 50728, 50730,
                50721, 50722, 50964, 50965
            )
            var i = 0
            while (i < n) {
                val e = ifd0 + 2 + i * 12
                val tag = tu16(template, e)
                val ty = tu16(template, e + 2)
                val cnt = tu32(template, e + 4)
                if (!drop.contains(tag) && ty in 1..12) {
                    val len = typeSize[ty - 1] * cnt
                    val data = if (len <= 4) {
                        template.copyOfRange(e + 8, e + 12)
                    } else {
                        val vo = tu32(template, e + 8)
                        if (vo < 0 || vo + len > template.size) return null
                        template.copyOfRange(vo, vo + len)
                    }
                    tlvs.add(Tlv(tag, ty, cnt, data))
                }
                i++
            }
            if (tlvs.none { it.tag == 254 }) tlvs.add(Tlv(254, 4, 1, le32(0)))
            val iso = result.get(android.hardware.camera2.CaptureResult.SENSOR_SENSITIVITY) ?: 100
            val expNs = result.get(android.hardware.camera2.CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L
            tlvs.add(Tlv(34855, 3, 1, le16(iso)))
            tlvs.add(Tlv(33434, 5, 1, rat(expNs, 1_000_000_000L)))
            if (correctedCm != null && correctedCm.size >= 18) {
                fun sr(v: Double): ByteArray {
                    val num = Math.round(v * 100000.0).toInt()
                    return le32(num) + le32(100000)
                }
                var cmData = byteArrayOf()
                for (v in correctedCm) cmData += sr(v.toDouble())
                tlvs.add(Tlv(50721, 10, 9, cmData.copyOfRange(0, 72)))
                tlvs.add(Tlv(50722, 10, 9, cmData.copyOfRange(72, 144)))
            }
            // AsShotNeutral claims the white the render actually folded
            // (gains -> Bradford -> primaries transform inverted onto the sRGB white), so
            // a standard converter reproduces the app's look including the
            // adaptation gate's deliberate residual color; AsShotWhiteXY
            // records the same white's chromaticity through the shipped
            // ColorMatrix1, keeping the two tags consistent with the
            // calibration set for readers that prefer 50729.
            if (foldedNeutral != null && foldedNeutral.size >= 3) {
                val nr = Math.round(foldedNeutral[0] * 100000f).toLong()
                val nb = Math.round(foldedNeutral[2] * 100000f).toLong()
                tlvs.add(Tlv(50728, 5, 3, rat(nr, 100000L) + rat(1, 1L) + rat(nb, 100000L)))
            }
            if (sceneXy != null && sceneXy.size >= 2) {
                val nx = Math.round(sceneXy[0] * 100000f).toLong()
                val ny = Math.round(sceneXy[1] * 100000f).toLong()
                tlvs.add(Tlv(50729, 5, 2, rat(nx, 100000L) + rat(ny, 100000L)))
            }
            if (Math.abs(baselineExposureEv) >= 0.005f) {
                tlvs.add(
                    Tlv(50730, 10, 1, rat(Math.round(baselineExposureEv * 10000f).toLong(), 10000L))
                )
            }
            // Lens shading: embed the active gain map as a DNG GainMap
            // opcode so a standard decode de-vignettes the raw with the same
            // correction the pipeline applied - the file's bayer is
            // uncorrected, and without the map every reader sees the
            // vignette the app removed. The learned map is achromatic (one
            // gain for every CFA phase), which is exactly what a
            // single-plane GainMap expresses.
            val shading = shadingMap
            if (shading != null && shading.available &&
                shading.width > 1 && shading.height > 1
            ) {
                fun be32(v: Int) = byteArrayOf(
                    (v shr 24).toByte(), (v shr 16).toByte(),
                    (v shr 8).toByte(), v.toByte()
                )
                fun beD(v: Double): ByteArray {
                    val bits = java.lang.Double.doubleToRawLongBits(v)
                    return ByteArray(8) { k ->
                        ((bits shr (56 - 8 * k)) and 0xffL).toByte()
                    }
                }
                fun beU16(v: Int) = byteArrayOf((v shr 8).toByte(), v.toByte())
                val gh = shading.height
                val gw = shading.width
                var payload = byteArrayOf()
                payload += be32(0) + be32(0) + be32(h) + be32(w)
                payload += be32(0) + be32(1)
                payload += be32(1) + be32(1)
                payload += be32(gh) + be32(gw)
                payload += beD(1.0 / (gh - 1)) + beD(1.0 / (gw - 1))
                payload += beD(0.0) + beD(0.0)
                payload += be32(1)
                for (r in 0 until gh) {
                    for (c in 0 until gw) {
                        val q = Math.round(shading.rGains[r][c] * 16384.0)
                            .toInt().coerceIn(0, 61440)
                        payload += beU16(q)
                    }
                }
                val blob = be32(1) + be32(9) + be32(0x01030000) + be32(1) +
                    be32(payload.size) + payload
                tlvs.add(Tlv(51044, 7, blob.size, blob))
            }
            if (renderRecipeJson != null) {
                val rb = renderRecipeJson.toByteArray(Charsets.UTF_8)
                tlvs.add(Tlv(55001, 7, rb.size, rb))
            }
            tlvs.add(Tlv(274, 3, 1, le16(exifOrientation)))
            tlvs.add(Tlv(256, 4, 1, le32(w)))
            tlvs.add(Tlv(257, 4, 1, le32(h)))
            tlvs.add(Tlv(258, 3, 1, le16(16)))
            tlvs.add(Tlv(259, 3, 1, le16(1)))
            tlvs.add(Tlv(262, 3, 1, le16(32803)))
            tlvs.add(Tlv(273, 4, 1, le32(0)))
            tlvs.add(Tlv(277, 3, 1, le16(1)))
            tlvs.add(Tlv(278, 4, 1, le32(h)))
            tlvs.add(Tlv(279, 4, 1, le32(0)))
            tlvs.add(Tlv(284, 3, 1, le16(1)))
            tlvs.sortBy { it.tag }
            val ne = tlvs.size
            val ifdLen = 2 + ne * 12 + 4
            fun align4(v: Int) = (v + 3) and 3.inv()
            var cursor = align4(8 + ifdLen)
            val offs = IntArray(ne)
            i = 0
            while (i < ne) {
                val tlv = tlvs[i]
                if (tlv.data.size > 4) {
                    offs[i] = cursor
                    cursor = align4(cursor + tlv.data.size)
                }
                i++
            }
            val stripOffset = cursor
            val pixelBytes = w * h * 2
            // Embedded thumbnail (the pipeline's own JPEG) as a second IFD
            // per the TIFF convention, so the DNG is a self-contained pair.
            val thumb = thumbnailJpeg
            val ifd1Offset = if (thumb != null) align4(stripOffset + pixelBytes) else 0
            val ifd1Len = if (thumb != null) 2 + 9 * 12 + 4 else 0
            var thumbJpegOffset = 0
            if (thumb != null) {
                var c = align4(ifd1Offset + ifd1Len)
                c = align4(c + 6)
                thumbJpegOffset = c
            }
            val total = if (thumb != null) thumbJpegOffset + thumb.size else stripOffset + pixelBytes
            val out = ByteArray(total)
            out[0] = 'I'.toByte(); out[1] = 'I'.toByte(); out[2] = 42; out[3] = 0
            w32(out, 4, 8)
            w16(out, 8, ne)
            i = 0
            while (i < ne) {
                val tlv = tlvs[i]
                val e = 10 + i * 12
                w16(out, e, tlv.tag)
                w16(out, e + 2, tlv.type)
                w32(out, e + 4, tlv.count)
                when (tlv.tag) {
                    273 -> w32(out, e + 8, stripOffset)
                    279 -> w32(out, e + 8, pixelBytes)
                    else -> if (tlv.data.size <= 4) {
                        System.arraycopy(tlv.data, 0, out, e + 8, tlv.data.size)
                    } else {
                        w32(out, e + 8, offs[i])
                        System.arraycopy(tlv.data, 0, out, offs[i], tlv.data.size)
                    }
                }
                i++
            }
            w32(out, 8 + ifdLen - 4, if (thumb != null) ifd1Offset else 0)
            val bv = bayer.asReadOnlyBuffer()
            bv.position(0)
            val row = ByteArray(w * 2)
            var y = 0
            while (y < h) {
                bv.get(row)
                System.arraycopy(row, 0, out, stripOffset + y * w * 2, row.size)
                y++
            }
            if (thumb != null) {
                var e1 = ifd1Offset + 2
                fun ent(tag: Int, ty: Int, cnt: Int, v: Int) {
                    w16(out, e1, tag); w16(out, e1 + 2, ty); w32(out, e1 + 4, cnt)
                    w32(out, e1 + 8, v); e1 += 12
                }
                w16(out, ifd1Offset, 9)
                ent(254, 4, 1, 1)
                ent(256, 4, 1, w)
                ent(257, 4, 1, h)
                ent(258, 3, 3, align4(ifd1Offset + ifd1Len))
                ent(259, 3, 1, 6)
                ent(262, 3, 1, 2)
                ent(277, 3, 1, 3)
                ent(513, 4, 1, thumbJpegOffset)
                ent(514, 4, 1, thumb.size)
                w32(out, e1, 0)
                val bpsOff = align4(ifd1Offset + ifd1Len)
                w16(out, bpsOff, 8); w16(out, bpsOff + 2, 8); w16(out, bpsOff + 4, 8)
                System.arraycopy(thumb, 0, out, thumbJpegOffset, thumb.size)
            }
            CrashLogger.log(
                TAG,
                "bayer dng tags=$ne strip=$stripOffset pixels=$pixelBytes thumb=${thumb?.size ?: 0}"
            )
            out
        } catch (err: Throwable) {
            CrashLogger.log(TAG, "bayer dng failed: " + err.message)
            null
        }
    }
}
