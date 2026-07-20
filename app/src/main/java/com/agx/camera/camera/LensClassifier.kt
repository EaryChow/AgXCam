package com.agx.camera.camera

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.util.Size

data class LensProfile(
    val id: String,
    val facing: Int,
    val hardwareLevel: Int,
    val focalLength: Float,
    val maxAfRegions: Int,
    val maxAeRegions: Int,
    val hasContinuousAf: Boolean,
    val hasAutoAf: Boolean,
    val maxResolution: Size,
    val hasFlash: Boolean,
    val minFocusDistance: Float?
) {
    fun isFixedFocus(): Boolean = minFocusDistance == 0.0f
    fun canTapToFocus(): Boolean = maxAfRegions > 0 && (hasAutoAf || hasContinuousAf)
    fun canMeterExposure(): Boolean = maxAeRegions > 0
    fun canTapToAdjust(): Boolean = canTapToFocus() || canMeterExposure()
}

object LensClassifier {

    fun classify(context: Context): List<LensProfile> {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val profiles = mutableListOf<LensProfile>()

        for (id in manager.cameraIdList) {
            val chars = manager.getCameraCharacteristics(id)

            val facing = chars.get(CameraCharacteristics.LENS_FACING) ?: continue
            val level = chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL) ?: 0
            val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val focalLength = focalLengths?.firstOrNull() ?: 0f
            val maxAfRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0
            val maxAeRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0
            val afModes = chars.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: intArrayOf()
            val hasContinuousAf = afModes.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            val hasAutoAf = afModes.contains(CaptureRequest.CONTROL_AF_MODE_AUTO)
            val minFocusDist = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
            val hasFlash = chars.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) ?: false

            val streamMap = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val maxSize = streamMap?.getOutputSizes(ImageFormat.YUV_420_888)
                ?.maxByOrNull { it.width * it.height } ?: Size(0, 0)

            profiles.add(LensProfile(
                id, facing, level, focalLength, maxAfRegions, maxAeRegions,
                hasContinuousAf, hasAutoAf, maxSize, hasFlash, minFocusDist
            ))
        }
        return profiles
    }

    private fun primaryScore(p: LensProfile): Int {
        var score = 0
        score += when (p.hardwareLevel) {
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> 40
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> 25
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> 10
            else -> 5
        }
        if (p.hasContinuousAf) score += 30
        else if (p.hasAutoAf) score += 15
        score += p.maxAfRegions * 10
        score += (p.maxResolution.width * p.maxResolution.height) / 500_000
        if (p.hasFlash) score += 10
        if (p.isFixedFocus()) score -= 25
        if (p.facing == CameraCharacteristics.LENS_FACING_FRONT) score -= 100
        return score
    }

    fun isAuxiliaryBackCamera(p: LensProfile): Boolean {
        if (p.facing != CameraCharacteristics.LENS_FACING_BACK) return false
        val legacyNoAf = p.hardwareLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY
            && !p.hasAutoAf && !p.hasContinuousAf && p.maxAfRegions == 0
        return legacyNoAf && p.isFixedFocus()
    }

    fun organize(profiles: List<LensProfile>): LensOrganization {
        val front = deduplicateFrontCameras(profiles.filter { it.facing == CameraCharacteristics.LENS_FACING_FRONT })
        val back = profiles.filter { it.facing == CameraCharacteristics.LENS_FACING_BACK }

        val usableBack = back.filter { !isAuxiliaryBackCamera(it) }
        val primaryBack = usableBack.maxByOrNull { primaryScore(it) }

        val primaryFront = front.maxByOrNull { primaryScore(it) }

        return LensOrganization(primaryBack, usableBack, primaryFront, front, back)
    }

    private fun deduplicateFrontCameras(front: List<LensProfile>): List<LensProfile> {
        // Keep the highest-scoring front camera per unique focal length
        return front.groupBy { it.focalLength }
            .values
            .map { it.maxByOrNull { primaryScore(it) }!! }
            .toList()
    }

    fun computeLabels(organization: LensOrganization): Map<String, String> {
        val labels = mutableMapOf<String, String>()
        val primary = organization.primaryBack ?: return labels
        val primaryFocal = primary.focalLength.takeIf { it > 0 } ?: 4.0f

        for (cam in organization.usableBack) {
            val ratio = if (cam.focalLength > 0 && primaryFocal > 0) {
                cam.focalLength / primaryFocal
            } else 1.0f
            val display = (kotlin.math.round(ratio * 10) / 10.0)
            labels[cam.id] = "${display}x"
        }

        for (cam in organization.front) {
            labels[cam.id] = "Front"
        }

        val allBackIds = organization.allBack.map { it.id }.toSet()
        val usableBackIds = organization.usableBack.map { it.id }.toSet()
        for (id in allBackIds - usableBackIds) {
            labels[id] = "Aux"
        }

        return labels
    }
}

data class LensOrganization(
    val primaryBack: LensProfile?,
    val usableBack: List<LensProfile>,
    val primaryFront: LensProfile?,
    val front: List<LensProfile>,
    val allBack: List<LensProfile>
)