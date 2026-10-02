package com.agx.camera.gpu

/**
 * Bookkeeping for the timer query objects, with no GL calls in it.
 *
 * The lifecycle is three states per query: open (between begin and end),
 * finished (result not yet read), and free. A query object may only go back to
 * the free list once its result has been read, because GL forbids re-targeting
 * a query whose result is still pending. Keeping that ordering in a GL-free
 * type makes it unit-testable, which the inlined version was not.
 *
 * Each entry also carries the timing segment it was opened against. A result is
 * harvested a frame or two after the query opened, by which time the pipeline
 * may have moved to a different slider configuration, so the segment has to be
 * recorded at begin() rather than looked up when the sample arrives.
 *
 * Access is synchronized because arm and disarm arrive on whichever thread
 * flipped the switch, which is the UI thread in practice, while begin, end and
 * poll run on the GL thread. Without the lock a disarm landing mid-harvest
 * mutates the list the harvest is iterating.
 */
internal class TimerQueryBook {

    class Entry(
        val passName: String,
        val queryId: Int,
        val segmentSeq: Int = TimingSegments.NO_SEGMENT
    ) {
        var isOpen: Boolean = true
    }

    private val pending = ArrayList<Entry>()
    private val free = ArrayList<Int>()
    private val openPerPass = HashMap<String, Int>()

    val pendingCount: Int
        @Synchronized get() = pending.size

    val freeCount: Int
        @Synchronized get() = free.size

    @Synchronized
    fun entries(): List<Entry> = pending.toList()

    /** Entries still inside a begin/end pair. */
    @Synchronized
    fun isOpen(passName: String): Boolean = openPerPass.containsKey(passName)

    /** Pending entries for a pass, open or finished. Caps allocation. */
    @Synchronized
    fun countPendingFor(passName: String): Int {
        var n = 0
        for (e in pending) if (e.passName == passName) n++
        return n
    }

    /** Takes a free id if one exists. The caller creates one otherwise. */
    @Synchronized
    fun takeFree(): Int = if (free.isEmpty()) 0 else free.removeAt(free.size - 1)

    @Synchronized
    fun begin(passName: String, queryId: Int, segmentSeq: Int = TimingSegments.NO_SEGMENT) {
        pending.add(Entry(passName, queryId, segmentSeq))
        openPerPass[passName] = (openPerPass[passName] ?: 0) + 1
    }

    /** Returns false when the pass has no open entry to close. */
    @Synchronized
    fun end(passName: String): Boolean {
        for (i in pending.indices.reversed()) {
            val e = pending[i]
            if (e.passName == passName && e.isOpen) {
                e.isOpen = false
                val open = (openPerPass[passName] ?: 1) - 1
                if (open <= 0) openPerPass.remove(passName) else openPerPass[passName] = open
                return true
            }
        }
        return false
    }

    /**
     * Marks a result as read and returns the id for reuse. Call this only after
     * the GL result has actually been read.
     */
    @Synchronized
    fun release(queryId: Int) {
        free.add(queryId)
    }

    /** Drops a harvested entry from the pending list. */
    @Synchronized
    fun removePending(entry: Entry) {
        pending.remove(entry)
    }

    /** Removes and returns every pending entry, for teardown. */
    @Synchronized
    fun drainAll(): List<Entry> {
        val out = pending.toList()
        pending.clear()
        openPerPass.clear()
        return out
    }

    @Synchronized
    fun clearFree() {
        free.clear()
    }

    /** Every id currently held, for deletion. */
    @Synchronized
    fun allIds(): List<Int> {
        val out = ArrayList<Int>(pending.size + free.size)
        for (e in pending) out.add(e.queryId)
        out.addAll(free)
        return out
    }
}