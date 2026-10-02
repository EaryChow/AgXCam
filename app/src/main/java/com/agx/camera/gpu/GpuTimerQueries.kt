package com.agx.camera.gpu

/**
 * Per-pass GPU timer histograms for the measurement switch, grouped by the
 * pipeline configuration that was live when each sample was opened.
 *
 * Every timed value comes from a GL_TIME_ELAPSED query. A CPU wall-clock
 * bracket would measure submission rather than execution here, since the host
 * thread returns long before the GPU finishes the pass.
 *
 * begin()/end() are called from the S1/S3/S5 sites whether or not the switch is
 * on, so only the branch inside differs between the two states and pass order
 * cannot drift. With the switch off no query object is ever created, and no
 * segment key is computed either, so the off state costs nothing beyond the
 * predicate at each call site.
 *
 * Samples are not pooled into one histogram. The benchmark has to work in any
 * slider combination and has to say which combination each number came from, so
 * a sample is filed under the segment stamped on its query at begin() time.
 *
 * GL_TIME_ELAPSED is a 64-bit result, but android.opengl binds only the 32-bit
 * query read, so a pass longer than 4.295 s wraps. See QueryResultRead.
 */
class GpuTimerQueries {

    private val segments = TimingSegments()
    private val book = TimerQueryBook()

    @Volatile var armed: Boolean = false
        private set

    private var disjointSeen = false

        /** Sequence of the segment current on the GL thread; -1 when untracked. */
        private var lastSeq = TimingSegments.NO_SEGMENT
    private var disjointSupport = DISJOINT_SUPPORT_UNKNOWN
    private var disjointEvents = 0
    private var unsupported = false
    private var glEnvironment: GlEnvironment? = null

    /** Ids awaiting deletion on the GL thread. See disarm(). */
    private val idsPendingDeletion = ArrayList<Int>()

    /**
 * Hands over the GL environment captured when the context came up.
 *
 * Probed on the GL thread at init rather than here, because arm and disarm run
 * on whichever thread flipped the switch (the UI thread in practice) and must
 * not issue GL calls from it. The environment is also wanted in an export taken
 * with the benchmark off, which is the case where a reader most needs to know
 * what implementation produced an empty section.
 */
    fun noteEnvironment(env: GlEnvironment) {
        glEnvironment = env
        if (armed) {
            // Arming may have happened before the context existed, in which case
            // the pending state was set without knowing. Resolve it now.
            unsupported = !env.timerAvailable
        }
    }

    /**
     * Turns the benchmark on. Off is the default and the only state a
     * normal frame ever sees. Call from any thread: no GL call is made here.
     */
    fun arm() {
        if (armed) return
        armed = true
        segments.reset()
        book.clearFree()
        book.drainAll()
        disjointSeen = false
        disjointSupport = DISJOINT_SUPPORT_UNKNOWN
        disjointEvents = 0
        val env = glEnvironment
        if (env == null) {
            // No context has come up yet. Samples are held off rather than
            // decided on a probe that could not have run.
            unsupported = true
            com.agx.camera.CrashLogger.log(
                TAG, "GPU timer queries armed before the GL context existed; probe pending"
            )
        } else {
            unsupported = !env.timerAvailable
            if (unsupported) {
                com.agx.camera.CrashLogger.log(
                    TAG,
                    "GPU timer queries unavailable (${env.timerRoute}) - benchmark stays empty"
                )
            }
        }
    }

    fun disarm() {
        if (!armed) return
        armed = false
        // Query objects can only be deleted with a current context, and disarm
        // arrives on the switching thread. Park the ids and let poll() delete
        // them from the GL thread; objects with an unread result stay allocated
        // for the life of the context, which is harmless because the ids are
        // forgotten either way.
        synchronized(idsPendingDeletion) { idsPendingDeletion.addAll(book.allIds()) }
        book.drainAll()
        book.clearFree()
        segments.reset()
    }

    /** Deletes queries left over from a disarm. GL thread only. */
    private fun drainPendingDeletion() {
        val ids: List<Int>
        synchronized(idsPendingDeletion) {
            if (idsPendingDeletion.isEmpty()) return
            ids = ArrayList(idsPendingDeletion)
            idsPendingDeletion.clear()
        }
        for (id in ids) {
            if (id != 0) android.opengl.GLES30.glDeleteQueries(1, intArrayOf(id), 0)
        }
    }

    /**
     * True while samples are being filed. The renderer consults this before
     * building a segment key, so the key is never computed in the off state.
     */
    fun isTracking(): Boolean = armed && !unsupported

    /**
     * Called once per frame before any begin(), so a segment switch lands on a
     * frame boundary rather than in the middle of a pass sequence.
     */
    fun onFrameBoundary(key: MeasurementSegment) {
        if (!isTracking()) return
        val seg = segments.enterFrame(key)
        lastSeq = seg.seq
        }

        /**
         * Records per-frame context that is reported but does not key the
         * segment. Kept off [onFrameBoundary] so the segment key stays a pure
         * value and callers cannot smuggle a context field into it.
         */
        fun noteFrameContext(thermalStatus: Int, zoomK: Float) {
            if (!isTracking()) return
            segments.noteFrameContext(lastSeq, thermalStatus, zoomK)
        }

    /** Opens a timed region. No-op unless armed and the timer target exists. */
    fun begin(passName: String) {
        if (!isTracking()) return
        val seq = segments.currentSeq()
        // No segment means no frame boundary was opened for this frame. Creating
        // a query now would produce a sample with nowhere correct to file it.
        if (seq == TimingSegments.NO_SEGMENT) return
        // Count finished-but-unread entries too, not just open ones, so a GPU
        // that never reports a result cannot make this pass accumulate objects.
        if (book.countPendingFor(passName) >= MAX_INFLIGHT_PER_PASS) return
        val id = book.takeFree().takeIf { it != 0 } ?: newQuery()
        if (id == 0) return
        android.opengl.GLES30.glBeginQuery(GL_TIME_ELAPSED, id)
        book.begin(passName, id, seq)
        segments.noteBegin(seq)
    }

    /** Closes the innermost timed region of the given pass. */
    fun end(passName: String) {
        if (!isTracking()) return
        if (!book.isOpen(passName)) return
        android.opengl.GLES30.glEndQuery(GL_TIME_ELAPSED)
        // The entry stays pending until poll() has read the result, so the
        // query object cannot be handed back out while its result is unread.
        book.end(passName)
    }

    /**
     * Harvests finished queries into their segments. Called once per frame
     * from the renderer; polling instead of blocking on
     * GL_QUERY_RESULT keeps the benchmark itself off the frame budget.
     *
     * A query object is only returned to the free list here, once its result
     * has been read. GL forbids re-targeting a query whose result is still
     * pending, so recycling any earlier would make the sample unreadable.
     *
     * The sample goes to the segment stamped on the query, not to whichever
     * segment happens to be current now: results arrive late, and filing them
     * by harvest time would push them across a slider-change boundary into a
     * segment whose key says they were captured under other settings.
     */
    fun poll() {
        // Runs before the armed check on purpose: a disarm parks its ids here and
        // the next frame has to release them even though the benchmark is off.
        drainPendingDeletion()
        if (!isTracking()) return
        val available = QueryResultRead.allocate()
        val result = QueryResultRead.allocate()
        for (entry in book.entries()) {
            if (entry.isOpen) {
                // Still inside a begin/end pair for this pass.
                continue
            }
            available[0] = 0
            android.opengl.GLES30.glGetQueryObjectuiv(entry.queryId, GL_QUERY_RESULT_AVAILABLE, available, 0)
            if (available[0] == 0) continue
            result[0] = 0
            android.opengl.GLES30.glGetQueryObjectuiv(entry.queryId, GL_QUERY_RESULT, result, 0)
            val nanos = QueryResultRead.read(result)
            if (isDisjoint(entry.queryId)) {
                // The clock was reset inside this query, so the number is not a
                // measurement of anything. Count the frame as settled and drop
                // the sample rather than filing a fabricated timing.
                disjointSeen = true
                segments.noteSettled(entry.segmentSeq)
            } else if (nanos > 0L) {
                segments.addSample(entry.passName, nanos, entry.segmentSeq)
            } else {
                segments.noteSettled(entry.segmentSeq)
            }
            // Result read: only now may the object go back to the free list.
            book.removePending(entry)
            book.release(entry.queryId)
        }
    }

    /** Samples harvested across every pass in every segment. */
    fun totalSamples(): Long {
        var n = 0L
        for (seg in segments.all()) n += seg.sampleCount()
        return n
    }

    fun hasSegments(): Boolean = segments.segmentCount > 0

    /** GL self-description, or a line saying why it is missing. */
    fun glEnvironmentReport(): String {
        val env = glEnvironment
            ?: return "GL environment\n" +
                "  unavailable: the probe runs when the GL context comes up, so this means\n" +
                "  no frame has been rendered yet\n"
        return env.report()
    }

    /** The block that starts the measurement report. */
    fun report(): String {
        if (!armed) return "benchmark off"
        if (unsupported) {
            return "benchmark on but GPU timer queries unavailable on this GL implementation\n" +
                "  probe: ${glEnvironment?.timerRoute ?: "not probed"}"
        }
        if (totalSamples() == 0L) {
            // Distinguish "the GPU never reported a result" from "every result
            // was discarded", which otherwise look identical in the bundle.
            val why = when {
                disjointSeen ->
                    "every sample dropped as a clock reset (${disjointNote()})"
                disjointSupport == DISJOINT_SUPPORT_UNAVAILABLE ->
                    "no sample survived, and the disjoint check is unavailable on this driver, " +
                        "so a reset cannot be ruled out"
                segments.segmentCount == 0 -> "no frame has been timed yet"
                else -> "the GPU has not returned a result for any query yet"
            }
            return "benchmark on, no samples harvested yet ($why)"
        }
        return "timing samples below; ${disjointNote()}\n" + segments.report()
    }

    /** Segment list on its own, so the report can place it next to the state block. */
    fun segmentReport(): String = segments.report()

    private fun newQuery(): Int {
        val buf = IntArray(1)
        android.opengl.GLES30.glGenQueries(1, buf, 0)
        return buf[0]
    }

    /**
     * Timer queries arrive either as a GLES 3.2 core feature or through an
     * extension string. When neither is present the passes are still called
     * (begin/end no-op) so the call sites stay identical, but no samples are
     * ever recorded.
     */
    fun timerQueriesAvailable(): Boolean {
        return try {
            val version = android.opengl.GLES20.glGetString(GL_VERSION) ?: ""
            if (version.contains("OpenGL ES 3.2")) return true
            val ext = android.opengl.GLES20.glGetString(
                android.opengl.GLES20.GL_EXTENSIONS
            ) ?: ""
            ext.contains("GL_EXT_disjoint_timer_query") || ext.contains("GL_ARB_timer_query")
        } catch (t: Throwable) {
            false
        }
    }

    /**
     * Whether the GPU's clock was reset inside this query, so its result is
     * meaningless.
     *
     * The signal is GL_DISJOINT, read off the query object itself. The previous
     * implementation asked glGetIntegerv for a desktop-GL robustness enum while
     * passing a constant that is not that enum's value: the driver answered with
     * something nonzero on every call, so every sample was thrown away and the
     * section reported an empty benchmark on a GPU that had answered with
     * OpenGL ES 3.2 and core timer queries. Reading a bogus enum is not
     * evidence that the context was reset.
     *
     * GL_DISJOINT is core from desktop GL 4.3 and is not part of ES 3.0, so a
     * driver may legitimately refuse it. Refusing is not evidence of a reset
     * either: the sample is trusted, and the report says the check was
     * unavailable. Collapsing "could not verify" into "a reset happened" throws
     * away every measurement at exactly the moment the driver is being
     * unhelpful about it, which is what left the timing baseline empty.
     */
    private fun isDisjoint(queryId: Int): Boolean {
        if (disjointSupport == DISJOINT_SUPPORT_UNAVAILABLE) return false
        // What is cached is the *availability verdict*, not the disjoint flag.
        // GL_DISJOINT is a property of the individual time query, not a global
        // device state, so a cached "not disjoint" would silently excuse every
        // later query. This therefore costs a glGetError drain plus one
        // glGetQueryObjectuiv on every harvested sample, which is the honest
        // price of the check; only the failure path is short-circuited, and the
        // unsupported log fires once.
        //
        // The drain is needed because the verdict below is read from the same
        // error slot someone else's GL call may have tripped. Without it an
        // unrelated error would be misattributed to GL_DISJOINT and reported as
        // "unsupported", discarding every timing measurement on a driver that
        // merely had a stale error pending.
        while (android.opengl.GLES20.glGetError() != android.opengl.GLES20.GL_NO_ERROR) {
            // drain
        }
        val buf = IntArray(1)
        android.opengl.GLES30.glGetQueryObjectuiv(queryId, GL_DISJOINT, buf, 0)
        val err = android.opengl.GLES20.glGetError()
        if (err != android.opengl.GLES20.GL_NO_ERROR) {
            disjointSupport = DISJOINT_SUPPORT_UNAVAILABLE
            com.agx.camera.CrashLogger.log(
                TAG,
                "GL_DISJOINT unsupported (glError 0x${Integer.toHexString(err)}); " +
                    "timer results trusted without a disjoint check"
            )
            return false
        }
        disjointSupport = DISJOINT_SUPPORT_AVAILABLE
        if (buf[0] != 0) disjointEvents++
        return buf[0] != 0
    }

    private fun disjointNote(): String = when (disjointSupport) {
        DISJOINT_SUPPORT_AVAILABLE ->
            "disjoint checked per query, $disjointEvents sample(s) dropped as reset"
        DISJOINT_SUPPORT_UNAVAILABLE -> "disjoint unverifiable, assumed none"
        else -> "disjoint not yet probed"
    }

    companion object {
        private const val TAG = "GpuTimerQueries"

        // GL_TIME_ELAPSED / GL_GUILTY_CONTEXT_RESET are not surfaced by the
        // android.opengl wrappers on every SDK level, so the enum values are
        // spelled out here (both are core GLES 3.2 / ES 3.0 values).
        private const val GL_VERSION = 0x1F02
        private const val GL_TIME_ELAPSED = 0x88BF
        private const val GL_DISJOINT = 0x8F20
        private const val GL_QUERY_RESULT = 0x8866
        private const val GL_QUERY_RESULT_AVAILABLE = 0x8867

        private const val DISJOINT_SUPPORT_UNKNOWN = 0
        private const val DISJOINT_SUPPORT_AVAILABLE = 1
        private const val DISJOINT_SUPPORT_UNAVAILABLE = -1

        // Two in-flight samples per pass name is enough to keep the histogram
        // fed without holding query objects hostage to a slow GPU.
        private const val MAX_INFLIGHT_PER_PASS = 2
    }
}
