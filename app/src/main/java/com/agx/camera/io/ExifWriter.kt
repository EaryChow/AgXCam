package com.agx.camera.io

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import com.agx.camera.camera.FlashMode
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class CaptureMetadata(
    val sensorOrientation: Int,
    val exifOrientation: Int,
    val focalLengthMm: Float,
    val focalLength35mm: Int = 0,
    val iso: Int,
    val exposureTimeNs: Long,
    val flashMode: FlashMode,
    val aeState: Int?,
    val captureWallClockMs: Long,
    val make: String = android.os.Build.MANUFACTURER,
    val model: String = android.os.Build.MODEL
)

object ExifWriter {

    private const val TAG = "ExifWriter"

    private const val TYPE_BYTE = 1
    private const val TYPE_ASCII = 2
    private const val TYPE_SHORT = 3
    private const val TYPE_LONG = 4
    private const val TYPE_RATIONAL = 5
    private const val AE_STATE_FLASH_REQUIRED = 4

    fun writeExif(file: File, metadata: CaptureMetadata, thumbnailJpeg: ByteArray? = null) {
        val jpegData = file.readBytes()
        if (jpegData.size < 2 ||
            (jpegData[0].toInt() and 0xFF) != 0xFF ||
            (jpegData[1].toInt() and 0xFF) != 0xD8) {
            Log.e(TAG, "Not a valid JPEG — cannot write EXIF")
            return
        }

        val exifApp1 = buildExifApp1(metadata, thumbnailJpeg)
        val result = ByteArray(2 + exifApp1.size + jpegData.size - 2)
        result[0] = 0xFF.toByte()
        result[1] = 0xD8.toByte()
        System.arraycopy(exifApp1, 0, result, 2, exifApp1.size)
        System.arraycopy(jpegData, 2, result, 2 + exifApp1.size, jpegData.size - 2)

        file.writeBytes(result)
    }

    private fun buildExifApp1(metadata: CaptureMetadata, thumbnailJpeg: ByteArray?): ByteArray {
        val ifd0Tags = buildIfd0Tags(metadata).sortedBy { it.tag }

        val hasThumbnail = thumbnailJpeg != null && thumbnailJpeg.isNotEmpty()

        val tiffHeaderSize = 8
        val ifd0EntrySize = 2 + ifd0Tags.size * 12 + 4

        var valuesArea = ByteArray(0)
        var currentOffset = tiffHeaderSize + ifd0EntrySize

        for (tag in ifd0Tags) {
            val raw = tag.externalData()
            if (raw != null) {
                tag.externalOffset = currentOffset
                val data = if (raw.size % 2 != 0) raw + 0x00.toByte() else raw
                valuesArea = valuesArea + data
                currentOffset += data.size
            }
        }

        var ifd1Bytes = ByteArray(0)
        var thumbnailBytes = ByteArray(0)
        var thumbnailDataOffset = 0
        var thumbnailDataLength = 0

        val ifd1StartOffset = currentOffset

        if (hasThumbnail) {
            thumbnailDataLength = thumbnailJpeg!!.size
            thumbnailDataOffset = currentOffset + 2 + ifd1EntryCount() * 12 + 4
            currentOffset += 2 + ifd1EntryCount() * 12 + 4

            thumbnailBytes = thumbnailJpeg.copyOf()
            if (thumbnailBytes.size % 2 != 0) {
                thumbnailBytes = thumbnailBytes + 0x00.toByte()
            }

            val ifd1Tags = buildIfd1Tags(thumbnailDataOffset, thumbnailDataLength)
            ifd1Bytes = writeIfd(ifd1Tags, 0)
        }

        val ifd0NextIfdOffset = if (hasThumbnail) ifd1StartOffset else 0
        val ifd0Bytes = writeIfd(ifd0Tags, ifd0NextIfdOffset)

        val tiffDataSize = tiffHeaderSize + ifd0Bytes.size + valuesArea.size + ifd1Bytes.size + thumbnailBytes.size
        val totalPayload = 2 + 6 + tiffDataSize
        val buf = ByteBuffer.allocate(2 + totalPayload)

        buf.put(0xFF.toByte())
        buf.put(0xE1.toByte())
        buf.put((totalPayload shr 8 and 0xFF).toByte())
        buf.put((totalPayload and 0xFF).toByte())

        buf.put("Exif".toByteArray(Charsets.US_ASCII))
        buf.put(0x00.toByte())
        buf.put(0x00.toByte())

        buf.order(ByteOrder.LITTLE_ENDIAN)
        buf.put(0x49.toByte())
        buf.put(0x49.toByte())
        buf.putShort(42.toShort())
        buf.putInt(8)

        buf.put(ifd0Bytes)
        buf.put(valuesArea)
        buf.put(ifd1Bytes)
        buf.put(thumbnailBytes)

        return buf.array().copyOf(buf.position())
    }

    private fun buildIfd0Tags(metadata: CaptureMetadata): List<ExifTag> {
        val tags = mutableListOf<ExifTag>()

        tags.add(ExifTag(0x0112, TYPE_SHORT, longArrayOf(metadata.exifOrientation.toLong())))
        tags.add(ExifTag(0x9209, TYPE_SHORT, longArrayOf(computeFlashTag(metadata).toLong())))
        tags.add(ExifTag(0xA001, TYPE_SHORT, longArrayOf(1L)))
        tags.add(ExifTag(0x8827, TYPE_SHORT, longArrayOf(metadata.iso.toLong())))

        var expNum = metadata.exposureTimeNs
        var expDen = 1_000_000_000L
        while (expNum > Int.MAX_VALUE || expDen > Int.MAX_VALUE) {
            expNum = expNum shr 1
            expDen = expDen shr 1
            if (expDen == 0L) { expDen = 1; expNum = expNum.coerceAtMost(Int.MAX_VALUE.toLong()); break }
        }
        tags.add(ExifTag(0x829A, TYPE_RATIONAL, longArrayOf(expNum, expDen)))
        val flNum = (metadata.focalLengthMm * 100).toInt()
        tags.add(ExifTag(0x920A, TYPE_RATIONAL, longArrayOf(flNum.toLong(), 100L)))
        if (metadata.focalLength35mm > 0) {
            tags.add(ExifTag(0xA405, TYPE_SHORT, longArrayOf(metadata.focalLength35mm.toLong())))
        }

        tags.add(ExifTag(0x010F, TYPE_ASCII, stringData = metadata.make.toByteArray(Charsets.US_ASCII) + 0x00.toByte()))
        tags.add(ExifTag(0x0110, TYPE_ASCII, stringData = metadata.model.toByteArray(Charsets.US_ASCII) + 0x00.toByte()))

        val formatter = java.text.SimpleDateFormat("yyyy:MM:dd HH:mm:ss", java.util.Locale.US)
        val dateStr = formatter.format(java.util.Date(metadata.captureWallClockMs))
        tags.add(ExifTag(0x9003, TYPE_ASCII, stringData = dateStr.toByteArray(Charsets.US_ASCII) + 0x00.toByte()))

        return tags
    }

    private fun buildIfd1Tags(thumbnailOffset: Int, thumbnailLength: Int): List<ExifTag> {
        return listOf(
            ExifTag(0x0103, TYPE_SHORT, longArrayOf(6)),
            ExifTag(0x0112, TYPE_SHORT, longArrayOf(1)),
            ExifTag(0x0201, TYPE_LONG, longArrayOf(thumbnailOffset.toLong())),
            ExifTag(0x0202, TYPE_LONG, longArrayOf(thumbnailLength.toLong()))
        )
    }

    private fun ifd1EntryCount(): Int = 4

    private fun writeIfd(tags: List<ExifTag>, nextIfdOffset: Int): ByteArray {
        val buf = ByteBuffer.allocate(2 + tags.size * 12 + 4).order(ByteOrder.LITTLE_ENDIAN)
        buf.putShort(tags.size.toShort())

        for (tag in tags) {
            buf.putShort(tag.tag.toShort())
            buf.putShort(tag.type.toShort())
            buf.putInt(tag.count())

            if (tag.needsExternal()) {
                buf.putInt(tag.externalOffset)
            } else {
                buf.put(tag.inlineValue())
            }
        }

        buf.putInt(nextIfdOffset)
        return buf.array().copyOf(buf.position())
    }

    fun generateThumbnailJpeg(sourceBitmap: Bitmap, rotationDegrees: Int): ByteArray {
        val targetWidth = 160
        val targetHeight = (sourceBitmap.height.toFloat() / sourceBitmap.width * targetWidth).toInt().coerceAtLeast(1)
        val scaled = Bitmap.createScaledBitmap(sourceBitmap, targetWidth, targetHeight, true)

        val rotated = if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            val r = Bitmap.createBitmap(scaled, 0, 0, scaled.width, scaled.height, matrix, true)
            if (r !== scaled) scaled.recycle()
            r
        } else {
            scaled
        }

        val stream = ByteArrayOutputStream()
        rotated.compress(Bitmap.CompressFormat.JPEG, 85, stream)
        if (rotated !== sourceBitmap && rotated.config != null) rotated.recycle()
        return stream.toByteArray()
    }

    private fun computeFlashTag(metadata: CaptureMetadata): Int {
        return when (metadata.flashMode) {
            FlashMode.OFF -> 0
            FlashMode.ON -> 1
            FlashMode.TORCH -> 1
            FlashMode.AUTO -> {
                if (metadata.aeState == AE_STATE_FLASH_REQUIRED) 1 else 32
            }
        }
    }

    private class ExifTag(
        val tag: Int,
        val type: Int,
        private val longValues: LongArray? = null,
        private val stringData: ByteArray? = null
    ) {
        var externalOffset: Int = 0

        fun count(): Int {
            return when (type) {
                TYPE_ASCII -> (stringData?.size ?: 0)
                TYPE_RATIONAL -> (longValues?.size ?: 0) / 2
                TYPE_SHORT, TYPE_LONG, TYPE_BYTE -> (longValues?.size ?: 0)
                else -> 0
            }
        }

        fun needsExternal(): Boolean {
            val totalBytes = when (type) {
                TYPE_ASCII -> stringData?.size ?: 0
                TYPE_RATIONAL -> (longValues?.size ?: 0) * 4
                TYPE_SHORT -> (longValues?.size ?: 0) * 2
                TYPE_LONG -> (longValues?.size ?: 0) * 4
                TYPE_BYTE -> (longValues?.size ?: 0)
                else -> 0
            }
            return totalBytes > 4
        }

        fun inlineValue(): ByteArray {
            val buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            when (type) {
                TYPE_SHORT -> {
                    val v = longValues?.getOrNull(0) ?: 0L
                    buf.putShort(v.toInt().toShort())
                }
                TYPE_LONG -> {
                    buf.putInt((longValues?.getOrNull(0) ?: 0L).toInt())
                }
                TYPE_ASCII -> {
                    val bytes = stringData ?: return buf.array()
                    val copyLen = minOf(bytes.size, 4)
                    buf.put(bytes, 0, copyLen)
                }
                TYPE_BYTE -> {
                    val v = longValues?.getOrNull(0) ?: 0L
                    buf.put(v.toByte())
                }
            }
            return buf.array()
        }

        fun externalData(): ByteArray? {
            if (!needsExternal()) return null
            return when (type) {
                TYPE_ASCII -> stringData
                TYPE_RATIONAL -> {
                    val buf = ByteBuffer.allocate(longValues!!.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                    for (v in longValues) buf.putInt(v.toInt())
                    buf.array()
                }
                TYPE_SHORT -> {
                    val buf = ByteBuffer.allocate(longValues!!.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                    for (v in longValues) buf.putShort(v.toInt().toShort())
                    buf.array()
                }
                TYPE_LONG -> {
                    val buf = ByteBuffer.allocate(longValues!!.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                    for (v in longValues) buf.putInt(v.toInt())
                    buf.array()
                }
                else -> null
            }
        }
    }
}
