package com.agx.camera.gpu

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

/**
 * Armed forensic capture container (`.framegrab`).
 *
 * One file, self-describing: a text header of key=value metadata followed by
 * named float planes. The point is that a grab taken on a device with no way
 * to attach logs can still be opened offline - the reader needs nothing but
 * this class.
 *
 * Format (little-endian throughout):
 *
 *   magic       8 bytes  "AGXFRMGR"
 *   version     u32      currently 1
 *   metaLen     u32      byte length of the metadata block
 *   meta        metaLen bytes, UTF-8, one "key=value" per line
 *   planeCount  u32
 *   repeated planeCount times:
 *     nameLen   u32
 *     name      nameLen bytes, UTF-8
 *     width     u32
 *     height    u32
 *     channels  u32      1..4
 *     payload   width*height*channels f32
 *
 * The writer is deliberately plain: no compression, no alignment tricks, no
 * shared state. A grab is built once, then handed to whoever saves it.
 *
 * Plane provenance lives in the meta block as `name@frame`, not in the binary
 * layout, because not every plane is computed on the frame the grab names: the
 * structure map and the axis buckets are sampled on a census cadence. The
 * container therefore records, per plane, the render-frame index its contents
 * came from, so a reader can tell a plane that describes the exported frame
 * from one that describes an earlier census frame.
 */
class FrameGrab private constructor(
    private val meta: LinkedHashMap<String, String>,
    private val planes: ArrayList<Plane>
) {

    data class Plane(
        val name: String,
        val width: Int,
        val height: Int,
        val channels: Int,
        val data: FloatArray
    ) {
        fun texelCount(): Int = width * height
    }

    class Builder {
        private val meta = LinkedHashMap<String, String>()
        private val planes = ArrayList<Plane>()

        fun meta(key: String, value: String): Builder {
            meta[key] = value
            return this
        }

        fun meta(key: String, value: Int): Builder = meta(key, value.toString())
        fun meta(key: String, value: Float): Builder = meta(key, String.format("%.6f", value))
        fun meta(key: String, value: Boolean): Builder = meta(key, if (value) "1" else "0")

        /**
         * Adds a plane. A null array or an empty payload is rejected so a
         * half-built grab cannot masquerade as a valid file.
         *
         * [sourceFrame] is the render-frame index the plane's contents were
         * actually computed on, and it exists because several planes are
         * sampled on a census cadence rather than per frame. Without it a
         * container that names frame N while carrying a plane computed on frame
         * N-3 asserts a currency it cannot back up, and a reader has no way to
         * tell which planes are describing the frame they think they are looking
         * at. Pass -1 to mean "same frame as the grab", which is the default for
         * planes that genuinely are.
         *
         * Recorded as `name@frame` in the meta block rather than in the binary
         * layout, so the provenance is visible in the plaintext index without
         * spending a format version on it.
         */
        fun plane(
            name: String,
            width: Int,
            height: Int,
            channels: Int,
            data: FloatArray?,
            sourceFrame: Int = FrameGrabProvenance.SAME_FRAME
        ): Builder {
            if (width <= 0 || height <= 0) return this
            if (channels < 1 || channels > 4) return this
            if (data == null || data.size < width * height * channels) return this
            planes.add(Plane(name, width, height, channels, data.copyOf()))
            meta["$name@frame"] = sourceFrame.toString()
            return this
        }

        fun planeCount(): Int = planes.size

        fun build(): FrameGrab = FrameGrab(LinkedHashMap(meta), ArrayList(planes))
    }

    fun metaSnapshot(): Map<String, String> = LinkedHashMap(meta)

    fun planeNames(): List<String> = planes.map { it.name }

    fun plane(name: String): Plane? = planes.firstOrNull { it.name == name }

    /**
     * The render-frame index a plane's contents were computed on, or null when
     * the plane is absent or declared same-frame.
     *
     * Returns [FrameGrabProvenance.SAME_FRAME] for a plane that claims to be
     * from the grabbed frame, so a caller can assert freshness directly:
     * `planeSourceFrame("s5.structureMap") != grabFrame()` means the map does
     * not describe this frame.
     */
    fun planeSourceFrame(name: String): Int? =
        meta["$name@frame"]?.toIntOrNull()

    /** True when the plane exists and was computed on [grabFrame]. */
    fun planeIsCurrentFor(name: String, grabFrame: Int): Boolean {
        if (plane(name) == null) return false
        val src = planeSourceFrame(name) ?: return true
        // A plane that never declared a frame is same-frame by default, which is
        // why SAME_FRAME has to be handled explicitly here: it is a recorded
        // value, not an absent one.
        return src == FrameGrabProvenance.SAME_FRAME || src == grabFrame
    }

    /** Names of planes whose contents came from a different frame than [grabFrame]. */
    fun planesNotFromFrame(grabFrame: Int): List<String> =
        planes.map { it.name }.filter { !planeIsCurrentFor(it, grabFrame) }

    /**
     * Size [toBytes] would produce, without building it. The bundle reports
     * this rather than serializing just to read the length.
     */
    fun byteCount(): Long {
        val metaBytes = buildString {
            for ((k, v) in meta) {
                append(k).append('=').append(v).append('\n')
            }
        }.toByteArray(StandardCharsets.UTF_8).size
        // magic + version + metaLen + meta + planeCount
        var total = MAGIC.size + 4 + 4 + metaBytes + 4
        for (p in planes) {
            // nameLen + name + width + height + channels + payload
            total += 4 + p.name.toByteArray(StandardCharsets.UTF_8).size + 12 +
                p.data.size * 4
        }
        return total.toLong()
    }

    /** Serializes to the on-disk byte layout. */
    fun toBytes(): ByteArray {
        val metaText = buildString {
            for ((k, v) in meta) {
                append(k).append('=').append(v).append('\n')
            }
        }
        val metaBytes = metaText.toByteArray(StandardCharsets.UTF_8)
        val out = ByteArrayOutputStream(64 + metaBytes.size + 64 * planes.size)
        val bb = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
        out.write(MAGIC)
        bb.clear()
        bb.putInt(VERSION)
        bb.putInt(metaBytes.size)
        out.write(bb.array())
        out.write(metaBytes)

        val header = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
        header.putInt(planes.size)
        out.write(header.array())

        for (p in planes) {
            val nameBytes = p.name.toByteArray(StandardCharsets.UTF_8)
            header.clear()
            header.putInt(nameBytes.size)
            out.write(header.array())
            out.write(nameBytes)
            val dims = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN)
            dims.putInt(p.width)
            dims.putInt(p.height)
            dims.putInt(p.channels)
            out.write(dims.array())
            val payload = ByteBuffer.allocate(p.data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            payload.asFloatBuffer().put(p.data)
            out.write(payload.array())
        }
        return out.toByteArray()
    }

    /** ASCII index of the grab: metadata lines plus one summary line per plane. */
    fun indexText(): String {
        val sb = StringBuilder()
        sb.append("framegrab v").append(VERSION).append(" planes=").append(planes.size).append('\n')
        for ((k, v) in meta) sb.append("  ").append(k).append('=').append(v).append('\n')
        for (p in planes) {
            sb.append("  plane ")
                .append(p.name)
                .append(' ').append(p.width).append('x').append(p.height)
                .append(" c").append(p.channels)
                .append(" texels=").append(p.texelCount())
                .append('\n')
        }
        return sb.toString()
    }

    companion object {
        private val MAGIC = byteArrayOf(
            'A'.code.toByte(), 'G'.code.toByte(), 'X'.code.toByte(), 'F'.code.toByte(),
            'R'.code.toByte(), 'M'.code.toByte(), 'G'.code.toByte(), 'R'.code.toByte()
        )
        const val VERSION = 1

        fun builder(): Builder = Builder()

        /**
         * Whether a frame should produce the plane for [diagnosticArmed].
         *
         * True for a grab whatever the diagnostic state, because a grab is the
         * only route to an offline explanation of a frame and inherits nothing:
         * pressed with the timing switch off it used to emit one float plane
         * and no image, with the meta's `structureMap=0` and `axisBuckets=0`
         * explaining nothing.
         *
         * Widening this does not cost the image path anything. Neither pass
         * feeds S5, and both are bounded by the census an armed grab forces
         * anyway.
         */
        fun collectsPlane(diagnosticArmed: Boolean, grabArmed: Boolean): Boolean =
            diagnosticArmed || grabArmed

        /**
         * Whether the frame may be folded into persisted measurement state -
         * the noise profile, the churn rate, the structure stabilizer, the
         * anchor advisory.
         *
         * Deliberately narrower than [collectsPlane], and deliberately not
         * widened by a grab. A grab is a read-only request; letting it move an
         * EMA or a churn counter would make "capture a frame" a mutation of the
         * very statistics the next report is about. The draw can happen without
         * the readback that writes them.
         */
        fun foldsPersistedState(diagnosticArmed: Boolean, grabArmed: Boolean): Boolean =
            diagnosticArmed

        /** Parses a grab back. Returns null on a bad magic, version or truncated body. */
        fun parse(bytes: ByteArray): FrameGrab? {
            return try {
                if (bytes.size < 16) return null
                for (i in MAGIC.indices) {
                    if (bytes[i] != MAGIC[i]) return null
                }
                val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                bb.position(MAGIC.size)
                val version = bb.int
                if (version != VERSION) return null
                val metaLen = bb.int
                if (metaLen < 0 || bb.position() + metaLen > bytes.size) return null
                val metaText = String(bytes, bb.position(), metaLen, StandardCharsets.UTF_8)
                bb.position(bb.position() + metaLen)
                val meta = LinkedHashMap<String, String>()
                for (line in metaText.split('\n')) {
                    if (line.isEmpty()) continue
                    val eq = line.indexOf('=')
                    if (eq <= 0) continue
                    meta[line.substring(0, eq)] = line.substring(eq + 1)
                }
                if (bb.remaining() < 4) return null
                val planeCount = bb.int
                if (planeCount < 0 || planeCount > MAX_PLANES) return null
                val planes = ArrayList<Plane>(planeCount)
                repeat(planeCount) {
                    if (bb.remaining() < 4) return null
                    val nameLen = bb.int
                    if (nameLen < 0 || nameLen > MAX_NAME || bb.remaining() < nameLen + 12) return null
                    val name = String(bytes, bb.position(), nameLen, StandardCharsets.UTF_8)
                    bb.position(bb.position() + nameLen)
                    val width = bb.int
                    val height = bb.int
                    val channels = bb.int
                    if (width <= 0 || height <= 0 || channels < 1 || channels > 4) return null
                    val count = width.toLong() * height.toLong() * channels.toLong()
                    if (count > MAX_TEXELS || bb.remaining() < count * 4) return null
                    val data = FloatArray(count.toInt())
                    // asFloatBuffer() hands back a view with its own position,
                    // so reading through it would leave bb where it started and
                    // the next plane header would be read from the payload.
                    // Slicing copies the span and lets the parent cursor move.
                    bb.slice().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(data)
                    bb.position(bb.position() + count.toInt() * 4)
                    planes.add(Plane(name, width, height, channels, data))
                }
                FrameGrab(meta, planes)
            } catch (t: Throwable) {
                null
            }
        }

        // A grab is a debug artifact, not a media file; these caps only exist
        // so a corrupt length cannot make the parser allocate wildly.
        private const val MAX_PLANES = 64
        private const val MAX_NAME = 128
        private const val MAX_TEXELS = 64L * 1024L * 1024L
    }
}

/**
 * The sentinel a plane records when it needs no separate source frame.
 *
 * Declared after [FrameGrab] so the container's format documentation stays
 * attached to the container. A KDoc block belongs to the declaration directly
 * beneath it, so an object wedged between the doc and the class silently
 * reassigns the whole format description to the wrong type.
 */
object FrameGrabProvenance {

    /** Passed to [FrameGrab.Builder.plane] when the plane is from the grabbed frame. */
    const val SAME_FRAME = -1
}