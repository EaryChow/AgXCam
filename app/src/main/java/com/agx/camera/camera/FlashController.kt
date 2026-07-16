package com.agx.camera.camera

import android.hardware.camera2.CaptureRequest

enum class FlashMode {
    OFF,
    AUTO,
    ON,
    TORCH;

    fun cycle(): FlashMode = when (this) {
        OFF -> AUTO
        AUTO -> ON
        ON -> TORCH
        TORCH -> OFF
    }

    fun applyToRequest(request: CaptureRequest.Builder, availableAeModes: IntArray = intArrayOf()) {
        when (this) {
            OFF -> {
                val aeMode = if (CaptureRequest.CONTROL_AE_MODE_ON in availableAeModes)
                    CaptureRequest.CONTROL_AE_MODE_ON else CaptureRequest.CONTROL_AE_MODE_OFF
                request.set(CaptureRequest.CONTROL_AE_MODE, aeMode)
                request.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
            }
            AUTO -> {
                val aeMode = if (CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH in availableAeModes)
                    CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH
                else if (CaptureRequest.CONTROL_AE_MODE_ON in availableAeModes)
                    CaptureRequest.CONTROL_AE_MODE_ON
                else CaptureRequest.CONTROL_AE_MODE_OFF
                request.set(CaptureRequest.CONTROL_AE_MODE, aeMode)
                request.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
            }
            ON -> {
                val aeMode = if (CaptureRequest.CONTROL_AE_MODE_ON_ALWAYS_FLASH in availableAeModes)
                    CaptureRequest.CONTROL_AE_MODE_ON_ALWAYS_FLASH
                else if (CaptureRequest.CONTROL_AE_MODE_ON in availableAeModes)
                    CaptureRequest.CONTROL_AE_MODE_ON
                else CaptureRequest.CONTROL_AE_MODE_OFF
                request.set(CaptureRequest.CONTROL_AE_MODE, aeMode)
                request.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
            }
            TORCH -> {
                request.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                request.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_TORCH)
            }
        }
    }
}
