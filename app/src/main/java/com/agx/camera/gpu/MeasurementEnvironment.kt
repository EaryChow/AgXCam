package com.agx.camera.gpu

import android.opengl.GLES20
import android.opengl.GLES30

/**
 * What the GL implementation under the preview actually is.
 *
 * The measurement report has to carry this because every timing and every
 * "unavailable" verdict is a statement about a specific GL implementation. A
 * driver that lacks disjoint timer queries, a context requested at an older
 * version than the driver supports, and a genuinely fast pass all produce the
 * same empty section otherwise, and the reader cannot tell them apart from the
 * report alone.
 */
class GlEnvironment private constructor(
    val version: String,
    val renderer: String,
    val vendor: String,
    val shadingLanguageVersion: String,
    val eglClientVersionRequested: Int,
    val timerRoute: String,
    val timerAvailable: Boolean
) {

    fun report(): String {
        val sb = StringBuilder()
        sb.append("GL environment\n")
        sb.append("  GL_VERSION=").append(version).append('\n')
        sb.append("  GL_RENDERER=").append(renderer).append('\n')
        sb.append("  GL_VENDOR=").append(vendor).append('\n')
        sb.append("  GL_SHADING_LANGUAGE_VERSION=").append(shadingLanguageVersion).append('\n')
        sb.append("  EGL_CONTEXT_CLIENT_VERSION requested=").append(eglClientVersionRequested)
            .append('\n')
        sb.append("  timer queries: ").append(if (timerAvailable) "available via " else "unavailable - ")
            .append(timerRoute).append('\n')
        return sb.toString()
    }

    companion object {

        /**
         * Reads the strings once. Every read is guarded because a vendor driver
         * is exactly the kind of code that returns null from glGetString, and a
         * null here must not take the preview thread down.
         */
        fun probe(eglClientVersionRequested: Int, timerProbe: () -> Boolean): GlEnvironment {
            fun str(name: Int, fallback: String = "(unavailable)"): String = try {
                GLES20.glGetString(name) ?: fallback
            } catch (t: Throwable) {
                fallback
            }

            val version = str(GLES20.GL_VERSION)
            val renderer = str(GLES20.GL_RENDERER)
            val vendor = str(GLES20.GL_VENDOR)
            val glsl = str(GLES20.GL_SHADING_LANGUAGE_VERSION)

            var route = "no disjoint-timer extension and no ES 3.2 version string"
            var available = false
            try {
                if (version.contains("OpenGL ES 3.2")) {
                    route = "GLES 3.2 core (GL_TIME_ELAPSED)"
                    available = true
                } else {
                    val ext = try {
                        GLES20.glGetString(GLES20.GL_EXTENSIONS) ?: ""
                    } catch (t: Throwable) {
                        ""
                    }
                    if (ext.contains("GL_EXT_disjoint_timer_query")) {
                        route = "GL_EXT_disjoint_timer_query"
                        available = true
                    } else if (ext.contains("GL_ARB_timer_query")) {
                        route = "GL_ARB_timer_query"
                        available = true
                    }
                }
            } catch (t: Throwable) {
                route = "extension probe threw: ${t.javaClass.simpleName}"
            }
            if (!available && timerProbe()) {
                // The live probe disagrees with the string scan. Say so rather
                // than picking a winner, so the next decision has both facts.
                route += "; live query probe reported usable"
            }
            return GlEnvironment(
                version = version,
                renderer = renderer,
                vendor = vendor,
                shadingLanguageVersion = glsl,
                eglClientVersionRequested = eglClientVersionRequested,
                timerRoute = route,
                timerAvailable = available
            )
        }
    }
}

/**
 * Compiled size of one GL program, in bytes.
 *
 * GL_PROGRAM_BINARY_LENGTH is the honest number: it is what the driver keeps
 * resident for the program. It arrived in GLES 3.1, so on the 3.0 context this
 * class requests it is unavailable. Rather than pass a different measure off as
 * the same one, the fallback sums GL_SHADER_SOURCE_LENGTH over the program's
 * attached shaders and is labelled a proxy in the report.
 */
object GlProgramSize {

    private const val GL_PROGRAM_BINARY_LENGTH = 0x8741
    private const val GL_SHADER_SOURCE_LENGTH = 0x8B88
    private const val GL_ATTACHED_SHADER = 0x8B85

    /** Binary bytes, or -1 when this context cannot report them. */
    fun binaryBytes(programId: Int): Int {
        if (programId == 0) return -1
        return try {
            val buf = IntArray(1)
            GLES30.glGetProgramiv(programId, GL_PROGRAM_BINARY_LENGTH, buf, 0)
            if (buf[0] > 0) buf[0] else -1
        } catch (t: Throwable) {
            -1
        }
    }

    /**
     * Total GLSL source bytes of every shader attached to [programId], or -1
     * when the program has none attached or the context refuses the query.
     */
    fun sourceBytesOfProgram(programId: Int): Int {
        if (programId == 0) return -1
        return try {
            val count = IntArray(1)
            GLES20.glGetProgramiv(programId, GL_ATTACHED_SHADER, count, 0)
            if (count[0] <= 0) return -1
            val shaders = IntArray(count[0])
            GLES20.glGetAttachedShaders(programId, count[0], shaders, 0, null, 0)
            var total = 0
            for (shaderId in shaders) {
                val buf = IntArray(1)
                GLES20.glGetShaderiv(shaderId, GL_SHADER_SOURCE_LENGTH, buf, 0)
                if (buf[0] > 0) total += buf[0]
            }
            if (total > 0) total else -1
        } catch (t: Throwable) {
            -1
        }
    }
}

/**
 * The per-sample fetch count and resident size of the shaders Phase 1 added.
 *
 * The fetch counts are the expensive part of a pass, so they are what a budget
 * discussion actually needs; the sizes say how much of the driver budget the
 * additions hold. Both are static properties of the GLSL, derived by reading
 * the shaders rather than estimated:
 *
 *  - S2 sigma-hat: eight same-CFA-phase neighbours per phase over four phases,
 *    plus one centre tap for the ISO floor -> 8*4 + 1 = 33 in the sparse grid
 *    domain; three phases in the capture RGB domain -> 8*3 + 1 = 25.
 *  - S3 green-guided GF: one centre tap, one green-guide tap, then a 5x5 window
 *    per CFA phase over four phases -> 1 + 1 + 25*4 = 102.
 *  - Axis-min buckets: eight taps across three phases plus one centre tap
 *    -> 8*3 + 1 = 25.
 *  - Structure map: one centre tap; localRatio costs 1 centre + 9 window + 1
 *    sigma = 11 and is evaluated for this texel and for each of the eight
 *    organization neighbours (8 * 11); gradDir costs 4 taps and is evaluated
 *    for this texel and each of the eight coherence neighbours (8 * 4).
 *    Total 1 + 11 + 4 + 88 + 32 = 136 on a textured texel. A texel whose
 *    gradient vanishes skips the coherence ring, so the flat-scene floor is
 *    1 + 11 + 4 + 88 = 104.
 *
 * ShaderBudgetTest re-derives these from the shader sources, so editing a tap
 * count in the GLSL without updating this table fails the build rather than
 * quietly invalidating the budget.
 */
class ShaderBudget {

    private class Entry(val name: String, val role: String, val fetches: Int, val fetchesNote: String)

    private val entries = ArrayList<Entry>()
    private val sizes = LinkedHashMap<String, Int>()
    private val sizeMeasure = LinkedHashMap<String, String>()

    /** Records one program and how its size was measured. */
    fun record(name: String, bytes: Int, measure: String) {
        if (bytes <= 0) return
        sizes[name] = bytes
        sizeMeasure[name] = measure
    }

    fun sizeOf(name: String): Int = sizes[name] ?: 0

    fun totalBytes(): Int {
        var n = 0
        for (v in sizes.values) n += v
        return n
    }

    fun baselineBytes(): Int {
        var n = 0
        for (name in BASELINE_PROGRAMS) n += sizes[name] ?: 0
        return n
    }

    fun phase1Bytes(): Int {
        var n = 0
        for (name in PHASE1_PROGRAMS) n += sizes[name] ?: 0
        return n
    }

    fun hasSizes(): Boolean = sizes.isNotEmpty()

    /**
     * The budget section. The gate note is part of it on purpose: a regression
     * percentage is meaningless without the segment key it was measured in, so
     * the rule is stated where the numbers are printed.
     */
    fun report(measurementForcedChain: Boolean): String {
        val sb = StringBuilder()
        sb.append("shader budget (fetch counts are per output texel)\n")
        sb.append("  fetch count by shader\n")
        for (e in entries) {
            sb.append("    ").append(e.name).append(" (").append(e.role).append("): ")
                .append(e.fetches).append(" fetches")
            if (e.fetchesNote.isNotEmpty()) sb.append(" [").append(e.fetchesNote).append(']')
            sb.append('\n')
        }
        sb.append("  resident size\n")
        if (sizes.isEmpty()) {
            sb.append("    unavailable: this GL context reports no program sizes\n")
        } else {
            for (name in ALL_PROGRAMS) {
                val bytes = sizes[name] ?: continue
                sb.append("    ").append(name).append(' ').append(bytes).append(" bytes (")
                    .append(sizeMeasure[name] ?: "unrecorded").append(")\n")
            }
            sb.append(String.format(
                "    baseline set (%d programs) = %d bytes%n",
                BASELINE_PROGRAMS.size, baselineBytes()
            ))
            sb.append(String.format(
                "    Phase-1 additions (%d programs) = %d bytes = %+.1f%% of the baseline set%n",
                PHASE1_PROGRAMS.size,
                phase1Bytes(),
                // A share of the baseline, not a delta against it. The
                // additions are a subset of the programs, so they can never
                // equal the baseline set, and subtracting 1 from the ratio
                // reports how far short of it they fall: on device that read
                // "-88.5%" for a set that is really 11.5% of the baseline.
                if (baselineBytes() > 0) {
                    phase1Bytes().toDouble() / baselineBytes().toDouble() * 100.0
                } else {
                    0.0
                }
            ))
            sb.append(String.format(
                "    all %d programs = %d bytes, additions are %.1f%% of the combined total%n",
                ALL_PROGRAMS.size, totalBytes(),
                if (totalBytes() > 0) phase1Bytes().toDouble() / totalBytes().toDouble() * 100.0 else 0.0
            ))
            // The caliber note matters because the two numbers get compared by
            // eye and are not the same quantity.
            sb.append("    caliber: these bytes are GL_PROGRAM_BINARY_LENGTH, i.e. the driver's\n")
            sb.append("    compiled resident binary, which is not comparable to the C-Driver\n")
            sb.append("    ~45 KB fragment budget. That budget is a SOURCE-SIZE limit on the\n")
            sb.append("    fragment sources; this table is a driver-output measure of the same\n")
            sb.append("    programs. A row here cannot be checked against 45 KB, and a rise\n")
            sb.append("    here does not by itself mean the source budget was exceeded.\n")
        }
        if (measurementForcedChain) {
            sb.append("  measurement on: the sigma-hat and green-guided GF passes are forced on\n")
            sb.append("    even when the image path has them gated out, so their cost lands in this\n")
            sb.append("    report and in the timing segments rather than being invisible\n")
        } else {
            sb.append("  measurement off: no pass is forced; the image path alone decides\n")
        }
        sb.append("  regression gate: the 5% budget limit is applied per segment; a baseline and\n")
        sb.append("    a candidate may only be compared when their segment keys match\n")
        return sb.toString()
    }

    companion object {
        /** The shaders Phase 1 introduced. */
        val PHASE1_PROGRAMS = listOf(
            "sigmaHat", "rawDenoise", "sigmaBucket", "structureMap"
        )

        /** The shader set that existed before Phase 1. */
        val BASELINE_PROGRAMS = listOf(
            "bayer", "demosaic", "nr", "spatialNr", "dpc", "outNr"
        )

        private val ALL_PROGRAMS = BASELINE_PROGRAMS + PHASE1_PROGRAMS

        /** Analytically derived from the GLSL; re-derived by ShaderBudgetTest. */
        const val FETCH_SIGMA_HAT_SPARSE = 33
        const val FETCH_SIGMA_HAT_RGB = 25
        const val FETCH_RAW_DENOISE = 102
        const val FETCH_SIGMA_BUCKET = 25
        const val FETCH_STRUCTURE_MAP_TEXTURED = 136
        const val FETCH_STRUCTURE_MAP_FLAT = 104

        fun standard(): ShaderBudget = ShaderBudget().apply {
            entries.add(
                Entry(
                    "S2.sigma_hat", "Stage 2 MAD, sparse grid domain",
                    FETCH_SIGMA_HAT_SPARSE, "capture RGB domain: $FETCH_SIGMA_HAT_RGB"
                )
            )
            entries.add(
                Entry("S3.green_guided_gf", "Stage 3 SWGF core", FETCH_RAW_DENOISE, "")
            )
            entries.add(
                Entry("S2.axis_buckets", "axis-min sigma split", FETCH_SIGMA_BUCKET, "")
            )
            entries.add(
                Entry(
                    "S5.structure_map", "structure classification",
                    FETCH_STRUCTURE_MAP_TEXTURED,
                    "flat texel with no gradient: $FETCH_STRUCTURE_MAP_FLAT"
                )
            )
        }
    }
}