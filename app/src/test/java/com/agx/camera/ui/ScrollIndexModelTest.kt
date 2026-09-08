package com.agx.camera.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ScrollIndexModelTest {

    private val spacing = 24f

    @Test
    fun defaultsToMidpointIndex() {
        val m = ScrollIndexModel(100, spacing)
        assertEquals(50, m.index)
    }

    @Test
    fun defaultsToMidpointForOddMax() {
        val m = ScrollIndexModel(9, spacing)
        assertEquals(4, m.index)
    }

    @Test
    fun setIndexClamps() {
        val m = ScrollIndexModel(100, spacing)
        m.setIndex(-5)
        assertEquals(0, m.index)
        m.setIndex(999)
        assertEquals(100, m.index)
    }

    @Test
    fun noMovementNoChange() {
        val m = ScrollIndexModel(100, spacing)
        m.setIndex(40)
        m.beginGesture()
        assertEquals(40, m.consumeDrag(0f))
        assertEquals(40, m.index)
    }

    @Test
    fun dragUpIncreasesIndex() {
        val m = ScrollIndexModel(100, spacing)
        m.setIndex(40)
        m.beginGesture()
        // finger up one full spacing
        assertEquals(41, m.consumeDrag(-spacing))
    }

    @Test
    fun dragDownDecreasesIndex() {
        val m = ScrollIndexModel(100, spacing)
        m.setIndex(40)
        m.beginGesture()
        assertEquals(39, m.consumeDrag(spacing))
    }

    @Test
    fun cumulativeDragSteps() {
        val m = ScrollIndexModel(100, spacing)
        m.setIndex(40)
        m.beginGesture()
        m.consumeDrag(-spacing)
        m.consumeDrag(-spacing)
        m.consumeDrag(-spacing)
        assertEquals(43, m.index)
    }

    @Test
    fun subHalfSpacingDoesNotStep() {
        val m = ScrollIndexModel(100, spacing)
        m.setIndex(40)
        m.beginGesture()
        assertEquals(40, m.consumeDrag(-spacing * 0.49f))
    }

    @Test
    fun halfSpacingSteps() {
        val m = ScrollIndexModel(100, spacing)
        m.setIndex(40)
        m.beginGesture()
        assertEquals(41, m.consumeDrag(-spacing * 0.5f))
    }

    @Test
    fun clampsAtLowerEndpoint() {
        val m = ScrollIndexModel(5, spacing)
        m.setIndex(0)
        m.beginGesture()
        repeat(10) { m.consumeDrag(spacing) }
        assertEquals(0, m.index)
    }

    @Test
    fun clampsAtUpperEndpoint() {
        val m = ScrollIndexModel(5, spacing)
        m.setIndex(5)
        m.beginGesture()
        repeat(10) { m.consumeDrag(-spacing) }
        assertEquals(5, m.index)
    }

    @Test
    fun maxIndexZeroStaysZero() {
        val m = ScrollIndexModel(0, spacing)
        m.beginGesture()
        assertEquals(0, m.consumeDrag(-spacing * 5f))
    }

    @Test
    fun shiftPxStaysBounded() {
        val m = ScrollIndexModel(100, spacing)
        m.setIndex(40)
        m.beginGesture()
        val deltas = listOf(-47f, 13f, 51f, -120f, 8f)
        for (d in deltas) {
            m.consumeDrag(d)
            val maxAbs = spacing / 2f + 0.001f
            assertTrue("shiftPx out of bounds: ${m.shiftPx}", Math.abs(m.shiftPx) <= maxAbs)
        }
    }

    @Test
    fun gestureRebasesOnEachDetent() {
        val m = ScrollIndexModel(100, spacing)
        m.setIndex(50)
        m.beginGesture()
        // one detent up: index now 51 and strip is re-centered (shiftPx ~ 0)
        m.consumeDrag(-spacing)
        assertEquals(51, m.index)
        assertTrue("shiftPx should re-center after full detent", Math.abs(m.shiftPx) <= 0.001f)
    }
}