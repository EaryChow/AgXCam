package com.agx.camera.gpu

import com.agx.camera.camera.NoiseModel
import com.agx.camera.gpu.SyntheticBayerTest.SENSOR_H
import com.agx.camera.gpu.SyntheticBayerTest.SENSOR_W
import com.agx.camera.gpu.SyntheticBayerTest.SynthParams
import com.agx.camera.gpu.SyntheticBayerTest.cellOriginX
import com.agx.camera.gpu.SyntheticBayerTest.cellOriginY
import com.agx.camera.gpu.SyntheticBayerTest.cellTexelFor
import com.agx.camera.gpu.SyntheticBayerTest.ownerTexelFor
import com.agx.camera.gpu.SyntheticBayerTest.buildSensor
import com.agx.camera.gpu.SyntheticBayerTest.flagCell
import com.agx.camera.gpu.SyntheticBayerTest.avgCell
import com.agx.camera.gpu.SyntheticBayerTest.sensorValue
import com.agx.camera.gpu.SyntheticBayerTest.isoSigma
import com.agx.camera.gpu.SyntheticBayerTest.rawPackCell
import com.agx.camera.gpu.SyntheticBayerTest.correctedCell
import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Host-side desk check of the CPU reference (SyntheticBayerTest) — runs on the
 * JVM before any device install. The oracle's geometry/phase/detector must
 * behave as designed on the fixed-seed scene so later on-device GPU/CPU diffs
 * can only be GPU bugs, not reference bugs.
 */
class SyntheticBayerRefTest {

    private val viewW = 960
    private val viewH = 720
    private val iso = 200
    private val sensor = buildSensor()

    private val params = SynthParams(
        dpEnabled = true,
        m1 = 0.4f, m2 = 10f, theta = 6f,
        isoA = NoiseModel.photonCoeff(iso),
        isoB = NoiseModel.readNoiseVariance(iso),
        corrStrength = 1f,
        s3Active = false, alpha = 1f, eps = 1f,
        viewW = viewW, viewH = viewH
    )

    private fun expectedPhaseValue(value: Float): Float = value - SyntheticBayerTest.BLACK_LEVEL

    @Test
    fun sceneDefectPhaseConvention() {
        // phase() = |x%2| + 2*|y%2|; with this convention (62,61) and (62,65)
        // are BOTH phase2 (x even, y odd) — there is no phase1 defect in the scene.
        assertEquals(0, SyntheticBayerTest.phase(60, 60))
        assertEquals(2, SyntheticBayerTest.phase(62, 61))
        assertEquals(3, SyntheticBayerTest.phase(65, 63))
        assertEquals(2, SyntheticBayerTest.phase(62, 65))
        val phases = probePhases()
        assertEquals(0, phases[60 * SENSOR_W + 60])
        assertEquals(2, phases[61 * SENSOR_W + 62])
        assertEquals(3, phases[63 * SENSOR_W + 65])
        assertEquals(2, phases[65 * SENSOR_W + 62])
    }

    @Test
    fun interiorMarkerCellsHonourPhaseConvention() {
        // Interior marker cell (84,68): the ±2 lattice window stays inside the
        // marker (x∈[80,128), y∈[64,112)), so avg == black-subtracted phase DN.
        val (tx, ty) = cellTexelFor(84, 68, viewW, viewH)
        val avg = avgCell(sensor, tx, ty, viewW, viewH)
        val raw = rawPackCell(sensor, tx, ty, viewW, viewH)
        for (p in 0 until 4) {
            val expected = when (p) { 0 -> expectedPhaseValue(840f); 1 -> expectedPhaseValue(220f); 2 -> expectedPhaseValue(230f); else -> expectedPhaseValue(120f) }
            assertEquals("marker avg phase $p", expected, avg[p], 1.0f)
            assertEquals("marker raw phase $p", expected, raw[p], 1.0f)
        }
    }

    @Test
    fun defectProbesFlaggedAtExpectedPhase() {
        class Spec(val sx: Int, val sy: Int, val phase: Int, val hot: Boolean)
        val expected = listOf(
            Spec(60, 60, 0, true),   // phase0 hot
            Spec(62, 61, 2, false),  // phase2 cold
            Spec(65, 63, 3, true),   // phase3 hot
            Spec(62, 65, 2, true)    // phase2 hot
        )
        for (d in expected) {
            val (tx, ty) = ownerTexelFor(d.sx, d.sy, viewW, viewH)
            val flags = flagCell(sensor, tx, ty, viewW, viewH,
                params.m1, params.m2, params.theta, params.isoA, params.isoB, enabled = true)
            val expectedFlag = if (d.hot) 1f else -1f
            assertEquals("defect (${d.sx},${d.sy}) phase ${d.phase}", expectedFlag, flags[d.phase], 0.0f)
        }
    }

    @Test
    fun defectCorrectionPullsTowardsNeighbours() {
        // Couplet semantics (DpcShaderProgram pass 3 mirrors, incl. the no-op path):
        //  - pass-CORRECT (couplet=false) pulls an isolated defect toward its
        //    neighbouring same-phase cells;
        //  - the final pass-COUPLET that feeds the sparse grid re-runs
        //    correction for texels whose OWN cell or 8-neighbourhood contains a
        //    flagged cell (flagged neighbours excluded from the candidates), so
        //    an isolated defect must come out corrected — NOT raw.  Only truly
        //    clean texels pass through.  (This used to leak single flagged
        //    cells raw; the pass-through was tightened to fix that.)
        class Spec(val sx: Int, val sy: Int, val phase: Int, val hot: Boolean)
        val expected = listOf(
            Spec(60, 60, 0, true),
            Spec(62, 61, 2, false),
            Spec(65, 63, 3, true),
            Spec(62, 65, 2, true)
        )
        for (d in expected) {
            val (tx, ty) = ownerTexelFor(d.sx, d.sy, viewW, viewH)
            val raw = rawPackCell(sensor, tx, ty, viewW, viewH)
            val pulled = correctedCell(sensor, tx, ty, viewW, viewH,
                params.m1, params.m2, params.theta, params.isoA, params.isoB,
                params.corrStrength, enabled = true, couplet = false)
            val coupletGrid = correctedCell(sensor, tx, ty, viewW, viewH,
                params.m1, params.m2, params.theta, params.isoA, params.isoB,
                params.corrStrength, enabled = true, couplet = true)
            assertTrue("couplet must correct isolated defect at (${d.sx},${d.sy}): " +
                "grid ${coupletGrid[d.phase]} not pulled from raw ${raw[d.phase]}",
                abs(coupletGrid[d.phase] - pulled[d.phase]) <= 1f)
            if (d.hot) {
                assertTrue("hot (${d.sx},${d.sy}) not pulled below raw: ${coupletGrid[d.phase]} >= ${raw[d.phase]}",
                    coupletGrid[d.phase] < raw[d.phase] - 50f)
                assertTrue("hot (${d.sx},${d.sy}) lost its flat base: ${coupletGrid[d.phase]}",
                    coupletGrid[d.phase] >= 60f)
            } else {
                assertTrue("cold (${d.sx},${d.sy}) not pulled above raw: ${coupletGrid[d.phase]} <= ${raw[d.phase]}",
                    coupletGrid[d.phase] > raw[d.phase] + 50f)
            }
        }
    }

    @Test
    fun sigmaCellsFiniteAndFloored() {
        val refs = SyntheticBayerTest.CpuRefs(sensor, params).compute()
        for (r in refs) {
            val s2 = r.sigma[0]
            assertTrue("sigma² NaN at (${r.sx},${r.sy})", !s2.isNaN())
            assertTrue("sigma² infinite at (${r.sx},${r.sy})", s2.isFinite())
            assertEquals("sigma² == sqrt channel at (${r.sx},${r.sy})",
                kotlin.math.sqrt(maxOf(s2, 1e-6f)), r.sigma[1], 1e-4f)
            // The floor uses the cell's OWN mean signal, not the probe coords.
            val meanSig = (r.grid[0] + r.grid[1] + r.grid[2] + r.grid[3]) * 0.25f
            val floorAtCell = isoSigma(meanSig, params.isoA, params.isoB)
            assertTrue("sigma² < per-cell floor at (${r.sx},${r.sy}): $s2 < $floorAtCell", s2 >= floorAtCell - 1e-4f)
        }
    }

    @Test
    fun cellOriginsMirrorGpuTransform() {
        // X axis: cellTexelFor -> cellOriginX is an exact bijection on even
        // anchor columns (both are the plain linear map), mirroring the GPU.
        for (sx in 0 until SENSOR_W step 2) {
            val (tx, _) = cellTexelFor(sx, 0, viewW, viewH)
            assertEquals("originX at sensor col $sx", sx, cellOriginX(tx, viewW))
        }
        // Y axis: the GPU feeds previewTransform into every sparse pass —
        // centre-crop to output aspect (scaleY≈0.9 for 192x160 -> 960x720) then
        // a Y flip — so cellOriginY is NOT the inverse of cellTexelFor's row.
        // Pin the exact GPU contract per truncated row.
        val sourceAspect = SENSOR_W.toFloat() / SENSOR_H.toFloat()
        val viewAspect = viewW.toFloat() / viewH.toFloat()
        val scaleY = if (sourceAspect > viewAspect) 1f else sourceAspect / viewAspect
        for (ty in 0 until viewH) {
            val expected = (SENSOR_H.toFloat() *
                (0.5f * (1f + scaleY) - scaleY * (ty + 0.5f) / viewH.toFloat()))
                .toInt().coerceIn(0, SENSOR_H - 1)
            assertEquals("originY at texel row $ty", expected, cellOriginY(ty, viewW, viewH))
        }
    }

    @Test
    fun reportProbeTable() {
        val refs = SyntheticBayerTest.CpuRefs(sensor, params).compute()
        System.out.println("\n== SYNTH CPU REF TABLE (iso=$iso, s1=1.0) ==")
        for (r in refs) {
            val flagsStr = r.flags.joinToString("") { v ->
                if (v > 0.5f) "H" else if (v < -0.5f) "C" else "."
            }
            val avgStr = r.avg.joinToString(", ") { String.format("%7.1f", it) }
            val gridStr = r.grid.joinToString(", ") { String.format("%7.1f", it) }
            val sigStr = String.format("%8.1f", r.sigma[0])
            System.out.println(
                "sensor=(${r.sx},${r.sy}) texel=(${r.tx},${r.ty}) flags=$flagsStr " +
                    "avg[$avgStr] grid[$gridStr] sig2=$sigStr"
            )
        }
        System.out.println("\n== DETECT DIAGNOSTICS (defect probes, s1=1.0) ==")
        for ((sx, sy, phase, hot) in listOf<IntArray>(
                intArrayOf(60, 60, 0, 1), intArrayOf(62, 61, 2, -1), intArrayOf(65, 63, 3, 1), intArrayOf(62, 65, 2, 1))) {
            val (tx, ty) = ownerTexelFor(sx, sy, viewW, viewH)
            val scx = cellOriginX(tx, viewW)
            val scy = cellOriginY(ty, viewW, viewH)
            val px = abs(scx % 2); val py = abs(scy % 2)
            val ccx = scx + (px xor (phase and 1))
            val ccy = scy + (py xor (phase shr 1))
            val i = sensorValue(sensor, ccx, ccy)
            val iavg = avgCell(sensor, tx, ty, viewW, viewH)[phase]
            val sigma = isoSigma(iavg, params.isoA, params.isoB)
            val band = maxOf(params.m1 * maxOf(iavg, 0f), params.theta * sigma)
            val dev = i - iavg
            var maxNb = 0f
            for (k in 0 until 8) {
                val nx = ccx + SyntheticBayerTest.NDX[k]; val ny = ccy + SyntheticBayerTest.NDY[k]
                val nbI = sensorValue(sensor, nx, ny)
                val (ntx, nty) = cellTexelFor(nx, ny, viewW, viewH)
                val nbAvg = avgCell(sensor, ntx, nty, viewW, viewH)[phase]
                maxNb = maxOf(maxNb, abs(nbI - nbAvg))
            }
            System.out.println(
                "defect (${sx},${sy}) phase$phase i=$i iavg=$iavg dev=$dev " +
                    "band=$band m2*maxNb=${params.m2 * maxNb} (maxNb=$maxNb) " +
                    "verdict=${if (abs(dev) > band && abs(dev) > params.m2 * maxNb) "FLAG" else "miss"}"
            )
        }
        assertTrue("probe sensor out of bounds", refs.all { it.sx in 0 until SENSOR_W && it.sy in 0 until SENSOR_H })
    }

    private fun probePhases(): IntArray {
        val p = IntArray(SENSOR_W * SENSOR_H)
        for (y in 0 until SENSOR_H) for (x in 0 until SENSOR_W) p[y * SENSOR_W + x] = SyntheticBayerTest.phase(x, y)
        return p
    }
}