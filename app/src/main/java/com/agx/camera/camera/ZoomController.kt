package com.agx.camera.camera

import android.util.Log

class ZoomController {
    var zoomFactor: Float = 1.0f
        private set
    var zoomCenterX: Float = 0.5f
        private set
    var zoomCenterY: Float = 0.5f
        private set

    var maxZoom: Float = MAX_ZOOM
        private set

    var listener: ((Float, Float, Float) -> Unit)? = null

    fun setMaxZoom(max: Float) {
        maxZoom = max.coerceAtLeast(MIN_ZOOM)
    }

    fun setZoom(factor: Float) {
        zoomFactor = factor.coerceIn(MIN_ZOOM, maxZoom)
        listener?.invoke(zoomFactor, zoomCenterX, zoomCenterY)
    }

    fun pan(dx: Float, dy: Float) {
        zoomCenterX = (zoomCenterX + dx).coerceIn(0.0f, 1.0f)
        zoomCenterY = (zoomCenterY + dy).coerceIn(0.0f, 1.0f)
        listener?.invoke(zoomFactor, zoomCenterX, zoomCenterY)
    }

    fun reset() {
        zoomFactor = 1.0f
        zoomCenterX = 0.5f
        zoomCenterY = 0.5f
        listener?.invoke(zoomFactor, zoomCenterX, zoomCenterY)
    }

    companion object {
        const val MIN_ZOOM = 1.0f
        const val MAX_ZOOM = 5.0f
    }
}
