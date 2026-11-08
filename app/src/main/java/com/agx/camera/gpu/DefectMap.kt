package com.agx.camera.gpu

import com.agx.camera.camera.NoiseModel
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Shared S1 candidate list.
 *
 * The list is built once from a high-exposure frame with the mean-excess
 * 12-tap alpha-trim plus the direction-specific four-tap veto. This is the
 * capture list/S1-mask operator, not the demosaic's inline max-neighbour
 * isolation gate. Every frame then checks only the listed coordinates with
 * the same mean-excess gate; only coordinates that trigger are repaired. The
 * list is a candidate set, not an unconditional repair map.
 */
object DefectMap {
    // dp=0.6 mean-excess band. The centre is compared with the trimmed mean,
    // not the maximum neighbour: adjacent same-colour hot pixels otherwise
    // mask each other. The four-tap veto below is unchanged.
    private const val BAND_M1 = 0.28f
    private const val BAND_M2 = 3.2f

    fun buildList(
        raw: ByteBuffer,
        width: Int,
        height: Int,
        stridePixels: Int,
        iso: Int,
        blackLevelPattern: IntArray
    ): IntArray {
        val mask = build(raw, width, height, stridePixels, iso, blackLevelPattern)
        val out = ArrayList<Int>(256)
        for (i in mask.indices) if (mask[i].toInt() != 0) out.add(i)
        val arr = IntArray(out.size)
        for (i in arr.indices) arr[i] = out[i]
        return arr
    }

    /**
     * Re-check every listed coordinate against the current frame and repair
     * only the ones that still pass the gate. The four replacement neighbours
     * are read from the pre-repair mosaic; triggered neighbours are excluded,
     * matching the sim's repair_from_map.
     */
    fun repairInPlace(
        raw: ByteBuffer,
        width: Int,
        height: Int,
        stridePixels: Int,
        iso: Int,
        blackLevelPattern: IntArray,
        list: IntArray
    ): Int {
        if (list.isEmpty()) return 0
        val src = raw.duplicate().order(ByteOrder.nativeOrder())
        val dst = raw.duplicate().order(ByteOrder.nativeOrder())
        val isoA = NoiseModel.photonCoeff(iso.coerceAtLeast(1))
        val isoB = NoiseModel.readNoiseVariance(iso.coerceAtLeast(1))

        fun sample(x: Int, y: Int): Float {
            val xx = x.coerceIn(0, width - 1)
            val yy = y.coerceIn(0, height - 1)
            val phase = (xx and 1) + ((yy and 1) shl 1)
            val v = (src.getShort((yy * stridePixels + xx) * 2).toInt() and 0xffff) -
                blackLevelPattern[phase]
            return max(v, 0).toFloat()
        }

        fun gate(x: Int, y: Int): Boolean {
            val c = sample(x, y)
            val e = sample(x + 2, y)
            val w = sample(x - 2, y)
            val n = sample(x, y - 2)
            val s = sample(x, y + 2)
            val ne = sample(x + 2, y - 2)
            val nw = sample(x - 2, y - 2)
            val se = sample(x + 2, y + 2)
            val sw = sample(x - 2, y + 2)
            val ee = sample(x + 4, y)
            val ww = sample(x - 4, y)
            val nn = sample(x, y - 4)
            val ss = sample(x, y + 4)

            val sum = e + w + n + s + ne + nw + se + sw + ee + ww + nn + ss
            val mn = minOf(minOf(minOf(minOf(minOf(minOf(e, w), n), s), ne),
                minOf(minOf(nw, se), sw)), minOf(minOf(ee, ww), minOf(nn, ss)))
            val mx = maxOf(maxOf(maxOf(maxOf(maxOf(maxOf(e, w), n), s), ne),
                maxOf(maxOf(nw, se), sw)), maxOf(maxOf(ee, ww), maxOf(nn, ss)))
            val iavg = (sum - mn - mx) * 0.1f
            val sigma = sqrt(max(isoA * max(iavg, 0f) + isoB, 1f))
            val band = max(BAND_M1 * max(iavg, 0f), BAND_M2 * sigma)
            val hot = c - iavg > band
            val cold = iavg - c > band
            if (!hot && !cold) return false

            val ewMin = minOf(e, w)
            val ewMax = maxOf(e, w)
            val nsMin = minOf(n, s)
            val nsMax = maxOf(n, s)
            val secondSmall = minOf(maxOf(ewMin, nsMin), minOf(ewMax, nsMax))
            val secondLarge = maxOf(minOf(ewMax, nsMax), maxOf(ewMin, nsMin))
            // Use the hot/cold direction that raised the mean-excess test.
            // An OR here can accept the opposite direction's veto.
            return if (hot) c - secondLarge > band
            else secondSmall - c > band
        }

        val triggered = HashSet<Int>(list.size * 2)
        for (idx in list) {
            val y = idx / width
            val x = idx - y * width
            if (gate(x, y)) triggered.add(idx)
        }
        if (triggered.isEmpty()) return 0

        for (idx in triggered) {
            val y = idx / width
            val x = idx - y * width
            val vals = FloatArray(4)
            var count = 0
            val neighbours = intArrayOf(
                y * width + (x + 2).coerceIn(0, width - 1),
                y * width + (x - 2).coerceIn(0, width - 1),
                (y - 2).coerceIn(0, height - 1) * width + x,
                (y + 2).coerceIn(0, height - 1) * width + x
            )
            for (nIdx in neighbours) {
                if (triggered.contains(nIdx)) continue
                val ny = nIdx / width
                val nx = nIdx - ny * width
                vals[count++] = sample(nx, ny)
            }
            if (count == 0) continue
            val used = vals.copyOf(count)
            used.sort()
            val v = 0.5f * (used[(count - 1) / 2] + used[count / 2])
            // Convert back to raw DN for the buffer write.
            val phase = (x and 1) + ((y and 1) shl 1)
            val rawV = (v + blackLevelPattern[phase]).toInt().coerceIn(0, 65535)
            dst.putShort((y * stridePixels + x) * 2, rawV.toShort())
        }
        return triggered.size
    }

    fun build(
        raw: ByteBuffer,
        width: Int,
        height: Int,
        stridePixels: Int,
        iso: Int,
        blackLevelPattern: IntArray
    ): ByteArray {
        val src = raw.duplicate().order(ByteOrder.nativeOrder())
        val out = ByteArray(width * height)
        val isoA = NoiseModel.photonCoeff(iso.coerceAtLeast(1))
        val isoB = NoiseModel.readNoiseVariance(iso.coerceAtLeast(1))

        fun sample(x: Int, y: Int): Float {
            val xx = x.coerceIn(0, width - 1)
            val yy = y.coerceIn(0, height - 1)
            val phase = (xx and 1) + ((yy and 1) shl 1)
            val v = (src.getShort((yy * stridePixels + xx) * 2).toInt() and 0xffff) -
                blackLevelPattern[phase]
            return max(v, 0).toFloat()
        }

        fun scanRow(y: Int) {
            for (x in 0 until width) {
                val c = sample(x, y)
                val e = sample(x + 2, y)
                val w = sample(x - 2, y)
                val n = sample(x, y - 2)
                val s = sample(x, y + 2)
                val ne = sample(x + 2, y - 2)
                val nw = sample(x - 2, y - 2)
                val se = sample(x + 2, y + 2)
                val sw = sample(x - 2, y + 2)
                val ee = sample(x + 4, y)
                val ww = sample(x - 4, y)
                val nn = sample(x, y - 4)
                val ss = sample(x, y + 4)

                val sum = e + w + n + s + ne + nw + se + sw + ee + ww + nn + ss
                val mn = minOf(minOf(minOf(minOf(minOf(minOf(e, w), n), s), ne),
                    minOf(minOf(nw, se), sw)), minOf(minOf(ee, ww), minOf(nn, ss)))
                val mx = maxOf(maxOf(maxOf(maxOf(maxOf(maxOf(e, w), n), s), ne),
                    maxOf(maxOf(nw, se), sw)), maxOf(maxOf(ee, ww), maxOf(nn, ss)))
                val iavg = (sum - mn - mx) * 0.1f
                val sigma = sqrt(max(isoA * max(iavg, 0f) + isoB, 1f))
                val band = max(BAND_M1 * max(iavg, 0f), BAND_M2 * sigma)
                val hot = c - iavg > band
                val cold = iavg - c > band
                if (!hot && !cold) continue

                val ewMin = minOf(e, w)
                val ewMax = maxOf(e, w)
                val nsMin = minOf(n, s)
                val nsMax = maxOf(n, s)
                val secondSmall = minOf(maxOf(ewMin, nsMin), minOf(ewMax, nsMax))
                val secondLarge = maxOf(minOf(ewMax, nsMax), maxOf(ewMin, nsMin))
                val veto = if (hot) c - secondLarge > band
                else secondSmall - c > band
                if (veto) {
                    out[y * width + x] = 255.toByte()
                }
            }
        }

        val workers = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        if (workers == 1) {
            for (y in 0 until height) scanRow(y)
        } else {
            val pool = Executors.newFixedThreadPool(workers)
            try {
                val futures = ArrayList<java.util.concurrent.Future<*>>(workers)
                for (w in 0 until workers) {
                    val y0 = height * w / workers
                    val y1 = height * (w + 1) / workers
                    futures.add(pool.submit {
                        for (y in y0 until y1) scanRow(y)
                    })
                }
                for (f in futures) f.get()
            } finally {
                pool.shutdown()
            }
        }
        return out
    }
}
