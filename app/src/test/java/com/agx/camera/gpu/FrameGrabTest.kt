package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * `.framegrab` round-trip checks.
 *
 * The container exists to be opened offline on a device with no way to attach
 * logs, so the round-trip is the whole product: if a multi-plane grab comes
 * back with the planes transposed or the second plane aliased onto the first,
 * the file is worse than useless because it still parses. Every test here is
 * about that asymmetry.
 */
class FrameGrabTest {

    private fun sampleGrab(): FrameGrab = FrameGrab.builder()
        .meta("iso", 1600)
        .meta("s3Active", true)
        .meta("exposureEv", 1.5f)
        .meta("stage", "capture")
        .plane("rgb", 2, 2, 3, floatArrayOf(
            0.0f, 0.1f, 0.2f,
            0.3f, 0.4f, 0.5f,
            0.6f, 0.7f, 0.8f,
            0.9f, 1.0f, 1.1f
        ))
        .plane("sigma", 2, 2, 1, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
        .build()

    @Test
    fun aGrabCollectsTheForensicSetEvenWithMeasurementOff() {
        // Pressed with the timing switch off, a grab used to emit s2.sigma and
        // nothing else: setMeasurementEnabled(false) leaves both diagnostic
        // flags false and frees both buffers, so the container held one float
        // plane and no image while its meta read structureMap=0 axisBuckets=0.
        // A grab is the only route to an offline explanation of a frame, so it
        // cannot inherit that from an unrelated toggle.
        assertTrue(
            "a grab must collect the plane with no diagnostic armed",
            FrameGrab.collectsPlane(diagnosticArmed = false, grabArmed = true)
        )
        assertTrue(FrameGrab.collectsPlane(diagnosticArmed = true, grabArmed = false))
        assertTrue(FrameGrab.collectsPlane(diagnosticArmed = true, grabArmed = true))
        assertFalse(
            "with neither armed there is nothing to collect",
            FrameGrab.collectsPlane(diagnosticArmed = false, grabArmed = false)
        )
    }

    @Test
    fun aGrabDrawsButDoesNotFoldPersistedMeasurementState() {
        // The draw can widen; the state write must not. Folding a grab frame
        // into the noise profile, the churn rate, the stabilizer or the anchor
        // advisory would make "capture a frame" a mutation of the statistics the
        // next report is about.
        assertFalse(
            "a grab must not fold churn, the stabilizer, the anchor advisory or the noise profile",
            FrameGrab.foldsPersistedState(diagnosticArmed = false, grabArmed = true)
        )
        assertTrue(FrameGrab.foldsPersistedState(diagnosticArmed = true, grabArmed = false))
        assertTrue(FrameGrab.foldsPersistedState(diagnosticArmed = true, grabArmed = true))

        // Folding state implies drawing the plane - you cannot fold a readback of
        // something that was never drawn. The reverse does not hold, and must
        // not: the grab-only case is exactly the plane-without-state-folding
        // asymmetry that keeps a grab read-only.
        for (armed in listOf(true, false)) {
            for (grab in listOf(true, false)) {
                if (FrameGrab.foldsPersistedState(armed, grab)) {
                    assertTrue(
                        "folding state without drawing the plane would read a stale target",
                        FrameGrab.collectsPlane(armed, grab)
                    )
                }
            }
        }
        assertTrue(
            "the deliberate asymmetry: a grab draws without folding",
            FrameGrab.collectsPlane(diagnosticArmed = false, grabArmed = true) &&
                !FrameGrab.foldsPersistedState(diagnosticArmed = false, grabArmed = true)
        )
    }

    @Test
    fun aPlaneCanDeclareTheFrameItWasComputedOn() {
        // The structure map and the axis buckets are drawn on census frames, so
        // between draws their framebuffers still hold the previous census
        // frame's data. A grab naming frame N while carrying a plane computed
        // on frame N-3 asserts a currency the container cannot back up, and the
        // reader has no way to tell which planes describe the exported frame.
        val grab = FrameGrab.builder()
            .meta("frame", 4120)
            .plane("s2.sigma", 1, 1, 1, floatArrayOf(1.0f), sourceFrame = 4120)
            .plane("s5.structureMap", 1, 1, 1, floatArrayOf(0.5f), sourceFrame = 4101)
            .build()

        assertEquals(4120, grab.planeSourceFrame("s2.sigma"))
        assertEquals(4101, grab.planeSourceFrame("s5.structureMap"))
        assertTrue(
            "the fresh plane describes this export",
            grab.planeIsCurrentFor("s2.sigma", 4120)
        )
        assertFalse(
            "the structure map is nineteen census frames stale and must say so",
            grab.planeIsCurrentFor("s5.structureMap", 4120)
        )
        assertEquals(
            listOf("s5.structureMap"),
            grab.planesNotFromFrame(4120)
        )
    }

    @Test
    fun provenanceSurvivesTheRoundTrip() {
        // The stamps live in the meta block, so they have to come back out of
        // the serialized form. If they did not, the offline reader - the entire
        // reason this container exists - would have no way to date a plane.
        val parsed = FrameGrab.parse(
            FrameGrab.builder()
                .meta("frame", 77)
                .plane("s5.structureMap", 1, 1, 1, floatArrayOf(0.25f), sourceFrame = 57)
                .build()
                .toBytes()
        )
        assertNotNull(parsed)
        assertEquals("77", parsed!!.metaSnapshot()["frame"])
        assertEquals(57, parsed.planeSourceFrame("s5.structureMap"))
        assertEquals(listOf("s5.structureMap"), parsed.planesNotFromFrame(77))
    }

    @Test
    fun aRejectedPlaneIsNotStamped() {
        // plane() drops malformed payloads. If the provenance stamp were written
        // before that check, a rejected plane would leave behind a name@frame
        // entry describing a plane the file does not contain.
        val grab = FrameGrab.builder()
            .plane("short", 4, 4, 4, floatArrayOf(1.0f), sourceFrame = 12)
            .plane("good", 1, 1, 1, floatArrayOf(1.0f), sourceFrame = 12)
            .build()

        assertEquals(listOf("good"), grab.planeNames())
        assertNull(grab.planeSourceFrame("short"))
        assertEquals(12, grab.planeSourceFrame("good"))
    }

    @Test
    fun sameFramePlanesDeclareNoSeparateFrame() {
        val grab = FrameGrab.builder()
            .plane("demosaic", 1, 1, 4, floatArrayOf(1f, 2f, 3f, 4f))
            .build()
        assertEquals(
            FrameGrabProvenance.SAME_FRAME,
            grab.planeSourceFrame("demosaic")
        )
        assertTrue(
            "an unstated plane is same-frame and must count as current",
            grab.planeIsCurrentFor("demosaic", 9999)
        )
    }

    @Test
    fun aPlaneAbsentFromTheGrabIsNeverCurrent() {
        // Otherwise a reader checking freshness would get a vacuous true for a
        // plane that is simply not in the file.
        val grab = FrameGrab.builder()
            .plane("demosaic", 1, 1, 4, floatArrayOf(1f, 2f, 3f, 4f))
            .build()
        assertFalse(grab.planeIsCurrentFor("s5.structureMap", 1))
        assertNull(grab.planeSourceFrame("s5.structureMap"))
    }

    @Test
    fun roundTripPreservesMetaAndPlaneData() {
        val original = sampleGrab()
        val parsed = FrameGrab.parse(original.toBytes())
        assertNotNull("round trip returned null", parsed)

        val meta = parsed!!.metaSnapshot()
        assertEquals("1600", meta["iso"])
        assertEquals("1", meta["s3Active"])
        assertEquals("1.500000", meta["exposureEv"])
        assertEquals("capture", meta["stage"])

        val rgb = parsed.plane("rgb")
        assertNotNull("rgb plane missing", rgb)
        assertEquals(2, rgb!!.width)
        assertEquals(2, rgb.height)
        assertEquals(3, rgb.channels)
        for (i in original.plane("rgb")!!.data.indices) {
            assertEquals(
                "rgb texel $i",
                original.plane("rgb")!!.data[i],
                rgb.data[i],
                0f
            )
        }
    }

    @Test
    fun byteCountMatchesSerializedSize() {
        for (grab in listOf(sampleGrab(), FrameGrab.builder().build())) {
            assertEquals(
                "byteCount drifted from toBytes",
                grab.toBytes().size.toLong(),
                grab.byteCount()
            )
        }
    }

    @Test
    fun multiPlanePayloadsDoNotAliasEachOther() {
        // The failure this guards is subtle: reading a plane payload through a
        // float view without advancing the byte cursor leaves the parent buffer
        // pointing at the first payload, so plane 2 is parsed out of plane 1's
        // bytes and the file parses "successfully" with wrong data.
        val parsed = FrameGrab.parse(sampleGrab().toBytes())!!
        val sigma = parsed.plane("sigma")
        assertNotNull("second plane missing after round trip", sigma)
        assertEquals(4, sigma!!.data.size)
        for (i in sigma.data.indices) {
            assertEquals("sigma texel $i", (i + 1).toFloat(), sigma.data[i], 0f)
        }
        // And the first plane must not have been overwritten by the second.
        val rgb = parsed.plane("rgb")!!
        assertEquals(0.9f, rgb.data[9], 0f)
        assertEquals(1.1f, rgb.data[11], 0f)
    }

    @Test
    fun planeNamesAndOrderSurvive() {
        val parsed = FrameGrab.parse(sampleGrab().toBytes())!!
        assertEquals(listOf("rgb", "sigma"), parsed.planeNames())
    }

    @Test
    fun indexTextIsAsciiAndListsPlanes() {
        val index = sampleGrab().indexText()
        assertTrue(index, index.all { it.code < 128 })
        assertTrue(index, index.contains("plane rgb 2x2 c3"))
        assertTrue(index, index.contains("plane sigma 2x2 c1"))
    }

    @Test
    fun badMagicIsRejected() {
        val bytes = sampleGrab().toBytes()
        bytes[0] = 'X'.code.toByte()
        assertNull(FrameGrab.parse(bytes))
    }

    @Test
    fun wrongVersionIsRejected() {
        val bytes = sampleGrab().toBytes()
        // version is the first little-endian int after the 8-byte magic.
        bytes[8] = 0x63
        assertNull("version mismatch must be rejected, not parsed with a guess", FrameGrab.parse(bytes))
    }

    @Test
    fun truncatedFileIsRejectedRatherThanPartiallyParsed() {
        val bytes = sampleGrab().toBytes()
        val cut = bytes.copyOf(bytes.size / 2)
        assertNull("truncated grab must not parse", FrameGrab.parse(cut))
    }

    @Test
    fun emptyInputIsRejected() {
        assertNull(FrameGrab.parse(ByteArray(0)))
        assertNull(FrameGrab.parse(ByteArray(4)))
    }

    @Test
    fun builderRejectsIncompletePlanes() {
        val grab = FrameGrab.builder()
            .plane("tooSmall", 4, 4, 3, floatArrayOf(1.0f, 2.0f))
            .plane("badChannels", 2, 2, 7, FloatArray(2 * 2 * 7))
            .plane("zeroWidth", 0, 2, 1, FloatArray(0))
            .plane("null", 2, 2, 1, null)
            .plane("good", 2, 2, 1, floatArrayOf(3.0f, 4.0f, 5.0f, 6.0f))
            .build()
        assertEquals(1, grab.planeNames().size)
        assertEquals(listOf("good"), grab.planeNames())
    }

    @Test
    fun builderDoesNotAliasTheCallersArray() {
        val src = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val grab = FrameGrab.builder().plane("p", 2, 2, 1, src).build()
        src[0] = 99f
        assertEquals(
            "grab must own its payload",
            1.0f,
            grab.plane("p")!!.data[0],
            0f
        )
    }

    @Test
    fun singleChannelLargePlaneRoundTrips() {
        // The realistic shape: one big mono sigma plane plus a small rgb one.
        val sigma = FloatArray(64 * 64) { it * 0.25f }
        val grab = FrameGrab.builder()
            .plane("sigma", 64, 64, 1, sigma)
            .plane("rgb", 2, 2, 3, FloatArray(12) { it.toFloat() })
            .build()
        val parsed = FrameGrab.parse(grab.toBytes())!!
        assertEquals(64 * 64, parsed.plane("sigma")!!.data.size)
        for (i in sigma.indices) {
            assertEquals("sigma $i", sigma[i], parsed.plane("sigma")!!.data[i], 0f)
        }
    }
}