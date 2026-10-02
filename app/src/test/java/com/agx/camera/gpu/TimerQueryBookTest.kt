package com.agx.camera.gpu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The query lifecycle, tested without a GL context.
 *
 * The bug this pins down: recycling a query object at end() instead of after
 * its result was read left poll() nothing to walk, so every sample was dropped
 * and every histogram stayed empty.
 */
class TimerQueryBookTest {

    @Test
    fun endedEntryStaysPendingUntilItsResultIsRead() {
        val book = TimerQueryBook()
        book.begin("S5", 7)
        assertTrue("entry should be open between begin and end", book.isOpen("S5"))

        assertTrue(book.end("S5"))
        assertFalse("entry must be closed after end", book.isOpen("S5"))
        assertEquals("ended entry must remain harvestable", 1, book.pendingCount)
        assertEquals("id must not be reusable before the result is read", 0, book.freeCount)
    }

    @Test
    fun idIsReusableOnlyAfterTheResultIsRead() {
        val book = TimerQueryBook()
        book.begin("S5", 7)
        book.end("S5")

        assertEquals("nothing to reuse yet", 0, book.takeFree())

        book.removePending(book.entries()[0])
        book.release(7)

        assertEquals(1, book.freeCount)
        assertEquals("freed id should come back", 7, book.takeFree())
        assertEquals("id must not be handed out twice", 0, book.freeCount)
    }

    @Test
    fun unfinishedEntriesAreNotHarvested() {
        val book = TimerQueryBook()
        book.begin("S5", 1)
        assertTrue(book.entries().all { it.isOpen })
    }

    @Test
    fun endWithoutBeginIsRejected() {
        val book = TimerQueryBook()
        assertFalse("end() with no open region must not consume anything", book.end("S5"))
        assertEquals(0, book.pendingCount)
    }

    @Test
    fun nestedRegionsOfOnePassCloseIndependently() {
        val book = TimerQueryBook()
        book.begin("S5", 1)
        book.begin("S5", 2)
        assertEquals(2, book.pendingCount)

        assertTrue(book.end("S5"))
        assertTrue("outer region is still open", book.isOpen("S5"))
        assertEquals(1, book.entries().count { !it.isOpen })

        assertTrue(book.end("S5"))
        assertFalse(book.isOpen("S5"))
        assertEquals(2, book.entries().count { !it.isOpen })
    }

    @Test
    fun onePassOpenDoesNotBlockAnother() {
        val book = TimerQueryBook()
        book.begin("S2", 1)
        assertTrue(book.isOpen("S2"))
        assertFalse(book.isOpen("S5"))
        assertTrue(book.end("S5").not())
        assertTrue("S2 entry survives", book.isOpen("S2"))
    }

    @Test
    fun pendingCountIsScopedPerPass() {
        val book = TimerQueryBook()
        book.begin("S2", 1)
        book.begin("S2", 2)
        book.begin("S5", 3)
        assertEquals(2, book.countPendingFor("S2"))
        assertEquals(1, book.countPendingFor("S5"))
    }

    @Test
    fun teardownCollectsOpenFinishedAndFreeIds() {
        val book = TimerQueryBook()
        book.begin("S2", 1)
        book.begin("S5", 2)
        book.end("S5")
        book.removePending(book.entries().first { !it.isOpen })
        book.release(2)

        val ids = book.allIds().sorted()
        assertEquals("open and free ids must both be collected", listOf(1, 2), ids)

        book.drainAll()
        book.clearFree()
        assertTrue("nothing left after teardown", book.allIds().isEmpty())
    }
}