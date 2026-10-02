package com.agx.camera.gpu

/**
 * The pipeline gate truth behind one frame, rendered as the block that starts
 * the measurement report.
 *
 * This exists because an empty measurement section is otherwise undecidable
 * from the report alone. "no samples yet" reads the same whether the pass never
 * ran, the gate that starts it was closed, or the benchmark was off; only the
 * gate truth tells those three apart. Printing the gates next to the data makes
 * a missing section self-diagnosing at the moment of export, with no need to
 * reconstruct the slider settings by hand afterwards.
 *
 * The same block is the plaintext form of the timing segment key, so the state
 * block and the segment list in the report corroborate each other rather than
 * being two independent claims about the run.
 *
 * That guarantee is structural rather than editorial: [segmentKey] arrives as a
 * field built by [segmentKeyFor] before the frame's first timed pass, and this
 * class cannot recompute or disagree with it. An earlier version exposed a
 * `segmentKey()` that rebuilt the key from these fields, which left the renderer
 * free to construct its own copy - the two agreed only because they sat in the
 * same function. Rebuilding here would have reintroduced the exact split the
 * class exists to prevent, so there is deliberately no builder method here.
 */
data class PipelineGates(
    val s1Raw: Int,
    val s3Raw: Int,
    val s5Raw: Int,
    val s1Strength: Float,
    val s3Strength: Float,
    val s5Strength: Float,
    val outDenoiseActive: Boolean,
    val needSparseGrid: Boolean,
    val sparseGridRendered: Boolean,
    val zoomK: Float,
    val viewWidth: Int,
    val viewHeight: Int,
    val passSigma: Boolean,
    val passSwgf: Boolean,
    val passOutNr: Boolean,
    val passBucket: Boolean,
    val bucketDrawn: Boolean,
    val passStructureMap: Boolean,
    val structureMapDrawn: Boolean,
    val sigmaSource: SigmaSource,
    val lumaSource: LumaSource,
    val thermalStatus: Int,
    val appThermalTier: String,
    val viewfinderFps: Float,
    val segmentKey: MeasurementSegment
) {

    companion object {
        /**
         * Frames between CPU-side census samples, and between structure-map
         * draws.
         *
         * One source for the cadence: the renderer's sampling gate and this
         * report's description of it have to agree, and two copies of a
         * sampling period drift apart the first time one of them is tuned.
         *
         * 20 frames is 1/3 s at preview rate. Both quantities the census
         * produces are slow variables - an EMA that already averages over
         * history, and a churn rate that carries no new information at 60 Hz -
         * so the resolution is not lost, while the structure map's measured
         * 3.5 ms mean / 5.0 ms p95 at 960x720 is amortised across twenty
         * frames instead of paid on every one.
         */
        const val CENSUS_INTERVAL_FRAMES = 20

        /**
         * The one producer of [MeasurementSegment] for a frame.
         *
         * Called once per frame, before the frame's first timed pass, because
         * the segment has to exist before the passes that populate its
         * histograms are entered into it. The resulting key is then handed to
         * [PipelineGates] rather than derived from it, which is what keeps the
         * segment list in the report and this block describing the same
         * segmentation.
         *
         * `thermalPlatformThrottled` is taken as an argument rather than
         * sampled here so that the caller decides which frame's reading is
         * used; this function has no clock and cannot drift from the frame it
         * is labelling.
         */
        fun segmentKeyFor(
            s1Raw: Int,
            s3Raw: Int,
            s5Raw: Int,
            sparseChainRegime: Boolean,
            viewWidth: Int,
            viewHeight: Int,
            appThermalTier: String,
            thermalPlatformThrottled: Boolean
        ): MeasurementSegment = MeasurementSegment(
            s1Raw = s1Raw,
            s3Raw = s3Raw,
            s5Raw = s5Raw,
            sparseChainRegime = sparseChainRegime,
            viewWidth = viewWidth,
            viewHeight = viewHeight,
            appTier = appThermalTier,
            platformThrottled = thermalPlatformThrottled
        )
    }

    /** Where this frame's sigma_hat estimate came from. */
    enum class SigmaSource(val text: String) {
        /** The image path ran the sparse chain; measurement read its output. */
        IMAGE_CHAIN("sparse chain output (image path)"),

        /** Only the measurement consumer ran the chain, into offscreen targets. */
        OFFSCREEN_CHAIN("grid chain output (measurement offscreen)"),

        /** No estimate was produced this frame. */
        NONE("none (sigma pass did not run)")
    }

    /** Which S5 output the structure map classified. */
    enum class LumaSource(val text: String) {
        S5_SPARSE("S5 output, sparse grid chain"),
        S5_INLINE("S5 output, demosaic inline path"),
        DEMOSAIC_NO_S5("demosaic output (S5 inactive)"),
        NOT_USED("structure map did not run")
    }

    /**
     * The discrete gate truth, which is also the change-detection key for the
     * live log line.
     *
     * Only discrete fields may go here. [PipelineStateLog.shouldLog] decides
     * whether to re-emit by comparing this string to the last one, so any
     * continuously-varying value placed in it logs a new line every frame and
     * floods the log with a state that has not changed. The viewfinder rate is
     * exactly that kind of value and is reported through [reportText] instead.
     *
     * Thermal level and the app's own tier belong here because they are
     * discrete, and a line the moment the device heats or the app throttles
     * itself is the whole point of recording them.
     */
    fun stateText(): String = String.format(
        "pipeline state: s1=%d s3=%d s5=%d | derived strengths s1=%.3f s3=%.3f s5=%.3f | " +
            "outDenoiseActive=%s needSparseGrid=%s sparseGridRendered=%s zoomK=%.2f | view=%dx%d | " +
            "thermal=%s%s appTier=%s",
        s1Raw, s3Raw, s5Raw,
        s1Strength, s3Strength, s5Strength,
        outDenoiseActive, needSparseGrid, sparseGridRendered, zoomK,
        viewWidth, viewHeight,
        ThermalStatus.text(thermalStatus),
        if (ThermalStatus.isThrottled(thermalStatus)) " THROTTLED" else "",
        appThermalTier
    )

    /**
     * The viewfinder rate, reported but not part of the change-detection key.
     *
     * Carries its own note that it is a wall-clock observation, not a pass
     * timing, so nobody reads it as a substitute for the GPU query budget.
     */
    fun frameRateText(): String = String.format(
        "viewfinder fps=%.1f (wall clock, presentation rate; not a GPU pass timing)",
        viewfinderFps
    )

    /**
     * Pass truth for the frame, with execution marked distinctly from eligibility.
     *
     * A `*` means the pass drew this frame; a bare `true` means it was eligible
     * but the cadence skipped it. Without the mark, `bucket=true` on nineteen
     * frames in twenty read as "the bucket pass ran", when the draw is
     * census-gated and had not run at all. sigma, swgf and outnr are not
     * cadence-gated, so they carry no mark and are true exactly when they ran.
     */
    fun passesText(): String =
        "passes: sigma=$passSigma swgf=$passSwgf outnr=$passOutNr " +
            "bucket=${drawn(passBucket, bucketDrawn)} smap=${drawn(passStructureMap, structureMapDrawn)}" +
            " | * = drew this frame | sigma source: ${sigmaSource.text}"

    private fun drawn(eligible: Boolean, drew: Boolean): String = when {
        drew -> "$eligible*"
        else -> "$eligible"
    }

    /**
     * The density note. Past the gate the grid chain does not run for the image
     * path, so a measurement-only chain is producing estimates at grid scale;
     * saying so keeps a counterintuitive k value from reading as a bug.
     */
    fun gridDensityNote(): String? {
        if (sparseGridRendered) return null
        return String.format(
            "grid density k=%.2f: measurement at grid scale (sparse chain gate k<=2 closed for the image path)",
            zoomK
        )
    }

/**
     * Which S5 output the structure map classified, and whether it drew.
     *
     * Eligibility and execution are separate facts, and reporting only the
     * first made this note claim a draw on the nineteen frames in every twenty
     * where the census cadence had skipped it.
     */
    fun structureMapPathNote(): String {
        val cadence = "sampled on a 1/$CENSUS_INTERVAL_FRAMES frame census because nothing consumes it"
        val sb = StringBuilder("structure map: luma=${lumaSource.text}, sigma=${sigmaSource.text}")
        return when {
            // Both states reported, because "eligible" and "drew it" are
            // different facts and printing only the first made the note claim a
            // draw on the nineteen frames in every twenty where there was none.
            structureMapDrawn -> sb.append(", drew this frame; it is $cadence").toString()
            passStructureMap -> sb.append(", eligible but not this frame's census frame; it is $cadence")
                .toString()
            else -> sb.append(", not eligible this frame").toString()
        }
    }

    fun reportText(): String {
        val sb = StringBuilder()
        sb.append(stateText()).append('\n')
        sb.append(frameRateText()).append('\n')
        sb.append(passesText()).append('\n')
        gridDensityNote()?.let { sb.append(it).append('\n') }
        sb.append(structureMapPathNote()).append('\n')
        return sb.toString()
    }
}

/**
 * Decides when the state block is worth printing.
 *
 * It prints once when measurement is armed, then again whenever a gate flips
 * or a slider moves, which is exactly when the reader's mental model of the
 * run goes stale. Printing it every frame would bury the transitions in noise,
 * and printing it only on segment switches would miss a gate flip that does not
 * change the segment key, such as the chain shaders finishing compilation.
 */
class PipelineStateLog {

    private var lastState: String? = null
    private var lastPasses: String? = null
    private var loggedAny = false

    fun reset() {
        lastState = null
        lastPasses = null
        loggedAny = false
    }

    /** True when this frame's gates differ from the last ones logged. */
    fun shouldLog(gates: PipelineGates): Boolean {
        if (loggedAny && gates.stateText() == lastState && gates.passesText() == lastPasses) {
            return false
        }
        lastState = gates.stateText()
        lastPasses = gates.passesText()
        loggedAny = true
        return true
    }

    fun hasLogged(): Boolean = loggedAny
}