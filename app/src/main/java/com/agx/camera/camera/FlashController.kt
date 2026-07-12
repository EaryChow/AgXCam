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

    fun applyToRequest(request: CaptureRequest.Builder) {
        when (this) {
            OFF -> {
                request.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                request.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
            }
            AUTO -> {
                request.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH)
                request.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
            }
            ON -> {
                request.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_ALWAYS_FLASH)
                request.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
            }
            TORCH -> {
                request.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                request.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_TORCH)
            }
        }
    }
}
