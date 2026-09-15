package com.agx.camera.gpu

import android.opengl.GLES20
import android.opengl.GLES30
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

private const val GL_HALF_FLOAT = 0x140B

class BayerShaderProgram {

    private var programId = 0
    private var demosaicProgramId = 0
    private var bayerTextureId = 0
    private var lensShadingTextureId = 0
    // 1x1 native-RGBA texture bound to the u_denoisedTex unit whenever no real
    // denoised mosaic is provided: the sampler is float-typed so binding the
    // integer R16UI Bayer texture there would raise GL_INVALID_OPERATION (0x502)
    // at draw time even though the inactive branch never samples it.
    private var fallbackFloatTextureId = 0

    private var uBayerTexLoc = 0
    private var uLensShadingMapLoc = 0
    private var uOutputResolutionLoc = 0
    private var uTransformMatrixLoc = 0
    private var uSensorSizeLoc = 0
    private var uCropOriginLoc = 0
    private var uCropSizeLoc = 0

    private var uSceneLinearTo709Loc = 0
    private var uInsetmatLoc = 0
    private var uOutsetmatLoc = 0
    private var u709To2020Loc = 0
    private var uWhiteLevelLoc = 0
    private var uBlackLevelLoc = 0
    private var uLogMinLoc = 0
    private var uLogMaxLoc = 0
    private var uLogMidgrayLoc = 0
    private var uDisplayMidgrayLoc = 0
    private var uContrastLoc = 0
    private var uToeLoc = 0
    private var uShoulderLoc = 0
    private var uVibranceLoc = 0

    private var uBlackLevelPatternLoc = 0
    private var uBayerColorMapLoc = 0
    private var uBitDepthLoc = 0
    private var uNrStrengthLoc = 0
    private var uExposureLoc = 0

    private var dUTextureLoc = 0
    private var dULensShadingMapLoc = 0
    private var dUOutputResolutionLoc = 0
    private var dUTransformMatrixLoc = 0
    private var dUInverseTransformMatrixLoc = 0
    private var dUSensorSizeLoc = 0
    private var dUCropOriginLoc = 0
    private var dUCropSizeLoc = 0
    private var dUBlackLevelPatternLoc = 0
    private var dUBayerColorMapLoc = 0
    private var dUBitDepthLoc = 0
    private var dUBoxAaLoc = 0
    private var dUWbGainsLoc = 0
    private var dUColorMatLoc = 0
    private var dUWhiteLevelLoc = 0
    private var dUBlackLevelLoc = 0
    private var dUDenoisedTexLoc = 0
    private var dUDenoiseActiveLoc = 0
    private var dUScaleFactorLoc = 0
    private var dULumaCoeffsLoc = 0
    private var dUClipAttenLoc = 0
    private var dUDpStrengthLoc = 0
    private var dURawNrStrengthLoc = 0
    private var dUIsoModelALoc = 0
    private var dUIsoModelBLoc = 0

    // S1/S3 same-colour pack fallback program: precomputes the S3 filter once
    // per texel per CFA phase into an RGBA32F denoised mosaic, which the
    // demosaic then reverse-maps. The primary preview path runs the identical
    // 12-tap filter inline per demosaiced sample instead; this precomputed
    // pack is only used when the DPC/GF shaders are not ready (fallback).
    private var s3PackProgramId = 0
    private var pUTextureLoc = 0
    private var pUTransformMatrixLoc = 0
    private var pUSensorSizeLoc = 0
    private var pUCropOriginLoc = 0
    private var pUCropSizeLoc = 0
    private var pUOutputResolutionLoc = 0
    private var pUBlackLevelPatternLoc = 0
    private var pUBitDepthLoc = 0
    private var pUDpStrengthLoc = 0
    private var pURawNrStrengthLoc = 0
    private var pUIsoModelALoc = 0
    private var pUIsoModelBLoc = 0
    private var pUWhiteLevelLoc = 0
    private var pUBlackLevelLoc = 0
    private var pUPackWbGainsLoc = 0

    private var sensorWidth = 0
    private var sensorHeight = 0
    private var cropOriginX = 0f
    private var cropOriginY = 0f
    private var cropSizeX = 1f
    private var cropSizeY = 1f

    private val quadVertices: FloatBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_COORDS).also { it.position(0) }
    private val quadTexCoords: FloatBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().put(QUAD_TEX_COORDS).also { it.position(0) }

    fun create(sensorW: Int, sensorH: Int) {
        sensorWidth = sensorW
        sensorHeight = sensorH
        cropSizeX = sensorW.toFloat()
        cropSizeY = sensorH.toFloat()

        programId = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        if (programId == 0) {
            Log.e(TAG, "Failed to create Bayer shader program")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create Bayer shader program")
            return
        }

        demosaicProgramId = createProgram(VERTEX_SHADER, DEMOSAIC_FRAGMENT_SHADER)
        if (demosaicProgramId == 0) {
            Log.e(TAG, "Failed to create demosaic shader program")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create demosaic shader program")
            return
        }

        uBayerTexLoc = GLES20.glGetUniformLocation(programId, "u_bayerTex")
        uLensShadingMapLoc = GLES20.glGetUniformLocation(programId, "u_lens_shading_map")
        uOutputResolutionLoc = GLES20.glGetUniformLocation(programId, "u_outputResolution")
        uTransformMatrixLoc = GLES20.glGetUniformLocation(programId, "u_transformMatrix")
        uSensorSizeLoc = GLES20.glGetUniformLocation(programId, "u_sensorSize")
        uCropOriginLoc = GLES20.glGetUniformLocation(programId, "u_cropOrigin")
        uCropSizeLoc = GLES20.glGetUniformLocation(programId, "u_cropSize")

        uSceneLinearTo709Loc = GLES20.glGetUniformLocation(programId, "u_scene_linear_to_709")
        uInsetmatLoc = GLES20.glGetUniformLocation(programId, "u_insetmat")
        uOutsetmatLoc = GLES20.glGetUniformLocation(programId, "u_outsetmat")
        u709To2020Loc = GLES20.glGetUniformLocation(programId, "u_709_to_2020")
        uWhiteLevelLoc = GLES20.glGetUniformLocation(programId, "u_white_level")
        uBlackLevelLoc = GLES20.glGetUniformLocation(programId, "u_black_level")
        uLogMinLoc = GLES20.glGetUniformLocation(programId, "u_log_min")
        uLogMaxLoc = GLES20.glGetUniformLocation(programId, "u_log_max")
        uLogMidgrayLoc = GLES20.glGetUniformLocation(programId, "u_log_midgray")
        uDisplayMidgrayLoc = GLES20.glGetUniformLocation(programId, "u_display_midgray")
        uContrastLoc = GLES20.glGetUniformLocation(programId, "u_contrast")
        uToeLoc = GLES20.glGetUniformLocation(programId, "u_toe")
        uShoulderLoc = GLES20.glGetUniformLocation(programId, "u_shoulder")
        uVibranceLoc = GLES20.glGetUniformLocation(programId, "u_vibrance")

        uBlackLevelPatternLoc = GLES20.glGetUniformLocation(programId, "u_black_level_pattern")
        uBayerColorMapLoc = GLES20.glGetUniformLocation(programId, "u_bayer_color_map")
        uBitDepthLoc = GLES20.glGetUniformLocation(programId, "u_bit_depth")
        uNrStrengthLoc = GLES20.glGetUniformLocation(programId, "u_nr_strength")
        uExposureLoc = GLES20.glGetUniformLocation(programId, "u_exposure")

        dUTextureLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_bayerTex")
        dULensShadingMapLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_lens_shading_map")
        dUOutputResolutionLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_outputResolution")
        dUTransformMatrixLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_transformMatrix")
        dUInverseTransformMatrixLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_inverseTransformMatrix")
        dUSensorSizeLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_sensorSize")
        dUCropOriginLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_cropOrigin")
        dUCropSizeLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_cropSize")
        dUBlackLevelPatternLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_black_level_pattern")
        dUBayerColorMapLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_bayer_color_map")
        dUBitDepthLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_bit_depth")
        dUBoxAaLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_box_aa")
        dUWbGainsLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_wb_gains")
        dUColorMatLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_color_mat")
        dUWhiteLevelLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_white_level")
        dUBlackLevelLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_black_level")
        dUDenoisedTexLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_denoisedTex")
        dUDenoiseActiveLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_denoise_active")
        dUScaleFactorLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_scale_factor")
        dULumaCoeffsLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_luma_coeffs")
        dUClipAttenLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_clip_atten_factor")
        dUDpStrengthLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_dp_strength")
        dURawNrStrengthLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_raw_nr_strength")
        dUIsoModelALoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_iso_model_a")
        dUIsoModelBLoc = GLES20.glGetUniformLocation(demosaicProgramId, "u_iso_model_b")

        s3PackProgramId = createProgram(VERTEX_SHADER, S3_PACK_FRAGMENT_SHADER)
        if (s3PackProgramId == 0) {
            Log.e(TAG, "Failed to create S1/S3 pack shader program")
            com.agx.camera.CrashLogger.log(TAG, "Failed to create S1/S3 pack shader program")
            return
        }
        pUTextureLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_bayerTex")
        pUTransformMatrixLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_transformMatrix")
        pUSensorSizeLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_sensorSize")
        pUCropOriginLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_cropOrigin")
        pUCropSizeLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_cropSize")
        pUOutputResolutionLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_outputResolution")
        pUBlackLevelPatternLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_black_level_pattern")
        pUBitDepthLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_bit_depth")
        pUDpStrengthLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_dp_strength")
        pURawNrStrengthLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_raw_nr_strength")
        pUIsoModelALoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_iso_model_a")
        pUIsoModelBLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_iso_model_b")
        pUWhiteLevelLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_white_level")
        pUBlackLevelLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_black_level")
        pUPackWbGainsLoc = GLES20.glGetUniformLocation(s3PackProgramId, "u_pack_wb_gains")

        val textures = IntArray(2)
        GLES20.glGenTextures(2, textures, 0)
        bayerTextureId = textures[0]
        lensShadingTextureId = textures[1]

        val fallback = IntArray(1)
        GLES20.glGenTextures(1, fallback, 0)
        fallbackFloatTextureId = fallback[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fallbackFloatTextureId)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
            1, 1, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE,
            java.nio.ByteBuffer.allocateDirect(4).put(byteArrayOf(0, 0, 0, 255.toByte())).apply { position(0) }
        )
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, bayerTextureId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, lensShadingTextureId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        Log.d(TAG, "Bayer shader program created: $programId, sensor: ${sensorWidth}x${sensorHeight}")
        com.agx.camera.CrashLogger.log(TAG, "Programs created: bayer=$programId demosaic=$demosaicProgramId")
    }

    fun setCropRegion(originX: Float, originY: Float, sizeX: Float, sizeY: Float) {
        cropOriginX = originX
        cropOriginY = originY
        cropSizeX = sizeX
        cropSizeY = sizeY
    }

    fun currentCropRegion(): FloatArray = floatArrayOf(cropOriginX, cropOriginY, cropSizeX, cropSizeY)

    fun uploadBayer(buffer: ByteBuffer, width: Int, height: Int, stridePixels: Int) {
        // u_sensorSize must match the ACTUAL uploaded texture dimensions (it is
        // the clamp bound for every texelFetch on u_bayerTex). The physical
        // sensor size passed to create() only happens to equal the texture size
        // on the real path; the synthetic 192x160 frame would otherwise sample
        // out of bounds (u_sensorSize still 4096x3072) and demosaic emits
        // garbage/denormals.
        sensorWidth = width
        sensorHeight = height

        GLES30.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, bayerTextureId)

        if (stridePixels > width && stridePixels > 0) {
            GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, stridePixels)
        }

        buffer.position(0)
        GLES30.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_R16UI,
            width, height, 0,
            GLES30.GL_RED_INTEGER, GLES20.GL_UNSIGNED_SHORT, buffer
        )

        if (stridePixels > width && stridePixels > 0) {
            GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, 0)
        }
    }

    fun uploadLensShadingMap(data: ShortArray, width: Int, height: Int) {
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, lensShadingTextureId)

        val buf = ByteBuffer.allocateDirect(data.size * 2)
            .order(ByteOrder.nativeOrder())
        buf.asShortBuffer().put(data)
        buf.position(0)

        GLES30.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F,
            width, height, 0,
            GLES30.GL_RGBA, GL_HALF_FLOAT, buf
        )
    }

    fun uploadIdentityLensShading() {
        val identity = ShortArray(4) { floatToHalf(1.0f) }
        uploadLensShadingMap(identity, 1, 1)
    }

    fun draw(
        outputWidth: Int, outputHeight: Int,
        transformMatrix: FloatArray,
        exposure: Float,
        sceneLinearTo709: FloatArray,
        insetMat: FloatArray,
        outsetMat: FloatArray,
        toRec2020: FloatArray,
        whiteLevel: Float, blackLevel: Float,
        logMin: Float, logMax: Float,
        logMidgray: Float, displayMidgray: Float,
        contrast: Float, toe: Float, shoulder: Float,
        vibrance: Float,
        blackLevelPattern: IntArray,
        bayerColorMap: IntArray,
        bitDepth: Int,
        nrStrength: Float
    ) {
        GLES20.glUseProgram(programId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, bayerTextureId)
        GLES20.glUniform1i(uBayerTexLoc, 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, lensShadingTextureId)
        GLES20.glUniform1i(uLensShadingMapLoc, 1)

        GLES20.glUniform2f(uOutputResolutionLoc, outputWidth.toFloat(), outputHeight.toFloat())
        GLES20.glUniformMatrix4fv(uTransformMatrixLoc, 1, false, transformMatrix, 0)
        GLES20.glUniform2f(uSensorSizeLoc, sensorWidth.toFloat(), sensorHeight.toFloat())
        GLES20.glUniform2f(uCropOriginLoc, cropOriginX, cropOriginY)
        GLES20.glUniform2f(uCropSizeLoc, cropSizeX, cropSizeY)

        GLES20.glUniformMatrix3fv(uSceneLinearTo709Loc, 1, true, sceneLinearTo709, 0)
        GLES20.glUniform1f(uExposureLoc, exposure)
        GLES20.glUniformMatrix3fv(uInsetmatLoc, 1, true, insetMat, 0)
        GLES20.glUniformMatrix3fv(uOutsetmatLoc, 1, true, outsetMat, 0)
        GLES20.glUniformMatrix3fv(u709To2020Loc, 1, true, toRec2020, 0)
        GLES20.glUniform1f(uWhiteLevelLoc, whiteLevel)
        GLES20.glUniform1f(uBlackLevelLoc, blackLevel)
        GLES20.glUniform1f(uLogMinLoc, logMin)
        GLES20.glUniform1f(uLogMaxLoc, logMax)
        GLES20.glUniform1f(uLogMidgrayLoc, logMidgray)
        GLES20.glUniform1f(uDisplayMidgrayLoc, displayMidgray)
        GLES20.glUniform1f(uContrastLoc, contrast)
        GLES20.glUniform1f(uToeLoc, toe)
        GLES20.glUniform1f(uShoulderLoc, shoulder)
        GLES20.glUniform1f(uVibranceLoc, vibrance)

        GLES20.glUniform4i(uBlackLevelPatternLoc,
            blackLevelPattern[0], blackLevelPattern[1],
            blackLevelPattern[2], blackLevelPattern[3])
        GLES20.glUniform4i(uBayerColorMapLoc,
            bayerColorMap[0], bayerColorMap[1],
            bayerColorMap[2], bayerColorMap[3])
        GLES20.glUniform1i(uBitDepthLoc, bitDepth)
        GLES20.glUniform1f(uNrStrengthLoc, nrStrength)

        val posHandle = GLES20.glGetAttribLocation(programId, "a_position")
        val texHandle = GLES20.glGetAttribLocation(programId, "a_texCoord")

        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 2, GLES20.GL_FLOAT, false, 0, quadVertices)

        GLES20.glEnableVertexAttribArray(texHandle)
        GLES20.glVertexAttribPointer(texHandle, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(posHandle)
        GLES20.glDisableVertexAttribArray(texHandle)
    }

    fun drawDemosaic(
        outputWidth: Int, outputHeight: Int,
        transformMatrix: FloatArray,
        blackLevelPattern: IntArray,
        bayerColorMap: IntArray,
        bitDepth: Int,
        whiteLevel: Float, blackLevel: Float,
        boxAA: Int = 4,
        wbGains: FloatArray = floatArrayOf(1f, 1f, 1f),
        colorMat: FloatArray? = null,
        denoisedTextureId: Int = 0,
        denoiseActive: Boolean = false,
        scaleFactor: Float = 1f,
        lumaCoeffs: FloatArray = DEFAULT_LUMA_COEFFS,
        clipAttenFactor: Float = CLIP_ATTEN_DEFAULT,
        dpStrength: Float = 0f,
        rawNrStrength: Float = 0f,
        isoModelA: Float = 0f,
        isoModelB: Float = 0f
    ) {
        GLES20.glUseProgram(demosaicProgramId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, bayerTextureId)
        GLES20.glUniform1i(dUTextureLoc, 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, lensShadingTextureId)
        GLES20.glUniform1i(dULensShadingMapLoc, 1)

        // Always bind a float-typed texture on u_denoisedTex's unit — never the
        // integer R16UI Bayer texture — otherwise the float sampler2D is
        // incompatible at draw time (GL_INVALID_OPERATION).
        GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
        val denoisedBindTex = if (denoisedTextureId != 0) denoisedTextureId else fallbackFloatTextureId
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, denoisedBindTex)
        GLES20.glUniform1i(dUDenoisedTexLoc, 2)
        GLES20.glUniform1f(dUDenoiseActiveLoc, if (denoiseActive && denoisedTextureId != 0) 1f else 0f)
        GLES20.glUniform1f(dUScaleFactorLoc, scaleFactor)

        GLES20.glUniform2f(dUOutputResolutionLoc, outputWidth.toFloat(), outputHeight.toFloat())
        GLES20.glUniformMatrix4fv(dUTransformMatrixLoc, 1, false, transformMatrix, 0)
        // Precomputed on the CPU: inverting the transform per-fragment in GLSL
        // accumulates float error that shows up as off-by-one denoised-texel
        // lookups when the demosaic reverse-maps sensor coordinates.
        val invTransform = FloatArray(16)
        android.opengl.Matrix.invertM(invTransform, 0, transformMatrix, 0)
        GLES20.glUniformMatrix4fv(dUInverseTransformMatrixLoc, 1, false, invTransform, 0)
        GLES20.glUniform2f(dUSensorSizeLoc, sensorWidth.toFloat(), sensorHeight.toFloat())
        GLES20.glUniform2f(dUCropOriginLoc, cropOriginX, cropOriginY)
        GLES20.glUniform2f(dUCropSizeLoc, cropSizeX, cropSizeY)

        GLES20.glUniform4i(dUBlackLevelPatternLoc,
            blackLevelPattern[0], blackLevelPattern[1],
            blackLevelPattern[2], blackLevelPattern[3])
        GLES20.glUniform4i(dUBayerColorMapLoc,
            bayerColorMap[0], bayerColorMap[1],
            bayerColorMap[2], bayerColorMap[3])
        GLES20.glUniform1i(dUBitDepthLoc, bitDepth)
        GLES20.glUniform1i(dUBoxAaLoc, boxAA)
        GLES20.glUniform3f(dUWbGainsLoc, wbGains[0], wbGains[1], wbGains[2])
        GLES20.glUniformMatrix3fv(dUColorMatLoc, 1, true, colorMat ?: COLOR_IDENTITY_9, 0)
        GLES20.glUniform1f(dUWhiteLevelLoc, whiteLevel)
        GLES20.glUniform1f(dUBlackLevelLoc, blackLevel)
        GLES20.glUniform3f(dULumaCoeffsLoc, lumaCoeffs[0], lumaCoeffs[1], lumaCoeffs[2])
        GLES20.glUniform1f(dUClipAttenLoc, clipAttenFactor)
        GLES20.glUniform1f(dUDpStrengthLoc, dpStrength)
        GLES20.glUniform1f(dURawNrStrengthLoc, rawNrStrength)
        GLES20.glUniform1f(dUIsoModelALoc, isoModelA)
        GLES20.glUniform1f(dUIsoModelBLoc, isoModelB)

        val posHandle = GLES20.glGetAttribLocation(demosaicProgramId, "a_position")
        val texHandle = GLES20.glGetAttribLocation(demosaicProgramId, "a_texCoord")

        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 2, GLES20.GL_FLOAT, false, 0, quadVertices)

        GLES20.glEnableVertexAttribArray(texHandle)
        GLES20.glVertexAttribPointer(texHandle, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(posHandle)
        GLES20.glDisableVertexAttribArray(texHandle)
    }

    // S1/S3 same-colour pack fallback: renders the S3 filter (12-tap α-trimmed
    // mean + DPC correction) once per texel and CFA phase into the caller-bound
    // RGBA32F FBO. The demosaic then serves as the cheap reverse-map sampler.
    // Only used when the DPC/GF shaders are unavailable; the normal path runs
    // sampleSameColorNR inline inside the demosaic.
    fun drawS3Pack(
        transformMatrix: FloatArray,
        blackLevelPattern: IntArray,
        bitDepth: Int,
        whiteLevel: Float,
        blackLevel: Float,
        dpStrength: Float,
        rawNrStrength: Float,
        isoModelA: Float,
        isoModelB: Float,
        outputWidth: Float,
        outputHeight: Float,
        wbGains: FloatArray = floatArrayOf(1f, 1f, 1f)
    ) {
        if (s3PackProgramId == 0) return
        GLES20.glUseProgram(s3PackProgramId)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES30.glBindTexture(GLES20.GL_TEXTURE_2D, bayerTextureId)
        GLES20.glUniform1i(pUTextureLoc, 0)

        GLES20.glUniformMatrix4fv(pUTransformMatrixLoc, 1, false, transformMatrix, 0)
        GLES20.glUniform2f(pUSensorSizeLoc, sensorWidth.toFloat(), sensorHeight.toFloat())
        GLES20.glUniform2f(pUCropOriginLoc, cropOriginX, cropOriginY)
        GLES20.glUniform2f(pUCropSizeLoc, cropSizeX, cropSizeY)
        GLES20.glUniform2f(pUOutputResolutionLoc, outputWidth, outputHeight)
        GLES20.glUniform4i(pUBlackLevelPatternLoc,
            blackLevelPattern[0], blackLevelPattern[1],
            blackLevelPattern[2], blackLevelPattern[3])
        GLES20.glUniform1i(pUBitDepthLoc, bitDepth)
        GLES20.glUniform1f(pUDpStrengthLoc, dpStrength)
        GLES20.glUniform1f(pURawNrStrengthLoc, rawNrStrength)
        GLES20.glUniform1f(pUIsoModelALoc, isoModelA)
        GLES20.glUniform1f(pUIsoModelBLoc, isoModelB)
        GLES20.glUniform1f(pUWhiteLevelLoc, whiteLevel)
        GLES20.glUniform1f(pUBlackLevelLoc, blackLevel)
        GLES20.glUniform4f(pUPackWbGainsLoc, wbGains[0], wbGains[1], wbGains[1], wbGains[2])

        val posHandle = GLES20.glGetAttribLocation(s3PackProgramId, "a_position")
        val texHandle = GLES20.glGetAttribLocation(s3PackProgramId, "a_texCoord")

        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 2, GLES20.GL_FLOAT, false, 0, quadVertices)

        GLES20.glEnableVertexAttribArray(texHandle)
        GLES20.glVertexAttribPointer(texHandle, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(posHandle)
        GLES20.glDisableVertexAttribArray(texHandle)
    }

    fun isReady(): Boolean = programId != 0 && demosaicProgramId != 0 && s3PackProgramId != 0 && bayerTextureId != 0

    fun rawProgramId(): Int = programId

    fun rawDemosaicProgramId(): Int = demosaicProgramId

    fun bayerTextureHandle(): Int = bayerTextureId

    fun sensorWidth(): Int = sensorWidth

    fun sensorHeight(): Int = sensorHeight

    fun destroy() {
        if (programId != 0) {
            GLES20.glDeleteProgram(programId)
            programId = 0
        }
        if (demosaicProgramId != 0) {
            GLES20.glDeleteProgram(demosaicProgramId)
            demosaicProgramId = 0
        }
        if (s3PackProgramId != 0) {
            GLES20.glDeleteProgram(s3PackProgramId)
            s3PackProgramId = 0
        }
        val textures = intArrayOf(bayerTextureId, lensShadingTextureId, fallbackFloatTextureId)
        GLES20.glDeleteTextures(3, textures, 0)
        bayerTextureId = 0
        lensShadingTextureId = 0
        fallbackFloatTextureId = 0
    }

    companion object {
        private const val TAG = "BayerShaderProgram"

        private val COLOR_IDENTITY_9 = floatArrayOf(
            1f, 0f, 0f,
            0f, 1f, 0f,
            0f, 0f, 1f
        )

        // Rec.709 Y row of RGB->XYZ(D65); fallback when no camera-native ->
        // XYZ map exists (YUV end of the app or unknown sensor).
        private val DEFAULT_LUMA_COEFFS = floatArrayOf(0.2126f, 0.7152f, 0.0722f)

        // Leading factor of the clip-neutralization exponent (u_clip_atten_factor * 5).
        private const val CLIP_ATTEN_DEFAULT = 0.1f

        private val QUAD_COORDS = floatArrayOf(
            -1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f
        )
        private val QUAD_TEX_COORDS = floatArrayOf(
            0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f
        )

        private const val VERTEX_SHADER = """
#version 300 es
in vec2 a_position;
in vec2 a_texCoord;
out vec2 v_texCoord;
uniform mat4 u_transformMatrix;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = (u_transformMatrix * vec4(a_texCoord, 0.0, 1.0)).xy;
}
"""

        private const val FRAGMENT_SHADER = """
#version 300 es
precision highp float;
precision highp usampler2D;
precision highp int;

in vec2 v_texCoord;
out vec4 fragColor;

uniform usampler2D u_bayerTex;
uniform sampler2D u_lens_shading_map;
uniform vec2 u_outputResolution;
uniform vec2 u_sensorSize;
uniform vec2 u_cropOrigin;
uniform vec2 u_cropSize;

uniform mat3 u_scene_linear_to_709;
uniform mat3 u_insetmat;
uniform mat3 u_outsetmat;
uniform mat3 u_709_to_2020;
uniform float u_white_level;
uniform float u_black_level;
uniform float u_exposure;
uniform float u_log_min;
uniform float u_log_max;
uniform float u_log_midgray;
uniform float u_display_midgray;
uniform float u_contrast;
uniform float u_toe;
uniform float u_shoulder;
uniform float u_vibrance;

uniform ivec4 u_black_level_pattern;
uniform ivec4 u_bayer_color_map;
uniform int u_bit_depth;
uniform float u_nr_strength;

${AgxCoreGlsl.CORE_HELPERS}

float unpackRaw(uint rawPacked) {
    uint mask;
    if (u_bit_depth <= 10) {
        mask = 0x3FFu;
    } else if (u_bit_depth <= 12) {
        mask = 0xFFFu;
    } else if (u_bit_depth <= 14) {
        mask = 0x3FFFu;
    } else {
        mask = 0xFFFFu;
    }
    return float(rawPacked & mask);
}

float sampleBayerRaw(ivec2 coord) {
    ivec2 clamped = clamp(coord, ivec2(0), ivec2(u_sensorSize) - ivec2(1));
    uint raw = texelFetch(u_bayerTex, clamped, 0).r;
    float val = unpackRaw(raw);
    int phase = abs(clamped.x % 2) + abs(clamped.y % 2) * 2;
    val -= float(u_black_level_pattern[phase]);
    return max(val, 0.0);
}

int safePhase(int x, int y) {
    return abs(x % 2) + abs(y % 2) * 2;
}

vec2 clampSensor(vec2 v) {
    return clamp(v, vec2(0.0), u_sensorSize - vec2(1.0));
}

float lsGain(vec2 lsNeighborUV) {
    ivec2 c = ivec2(clampSensor(lsNeighborUV));
    return texture(u_lens_shading_map, vec2(c) / u_sensorSize)[safePhase(c.x, c.y)];
}

vec3 demosaicBilinear(usampler2D tex, vec2 sensorUV, vec2 lsSensorUV) {
    ivec2 sensorCoord = ivec2(sensorUV);
    int phase = safePhase(sensorCoord.x, sensorCoord.y);
    int color = u_bayer_color_map[phase];

    float gain = texture(u_lens_shading_map, lsSensorUV / u_sensorSize)[phase];

    ivec2 nN_coord = sensorCoord + ivec2( 0, -1);
    ivec2 nS_coord = sensorCoord + ivec2( 0,  1);
    ivec2 nW_coord = sensorCoord + ivec2(-1,  0);
    ivec2 nE_coord = sensorCoord + ivec2( 1,  0);
    ivec2 nNW_coord = sensorCoord + ivec2(-1, -1);
    ivec2 nNE_coord = sensorCoord + ivec2( 1, -1);
    ivec2 nSW_coord = sensorCoord + ivec2(-1,  1);
    ivec2 nSE_coord = sensorCoord + ivec2( 1,  1);

    vec2 lsN  = lsSensorUV + vec2( 0.0, -1.0);
    vec2 lsS  = lsSensorUV + vec2( 0.0,  1.0);
    vec2 lsW  = lsSensorUV + vec2(-1.0,  0.0);
    vec2 lsE  = lsSensorUV + vec2( 1.0,  0.0);
    vec2 lsNW = lsSensorUV + vec2(-1.0, -1.0);
    vec2 lsNE = lsSensorUV + vec2( 1.0, -1.0);
    vec2 lsSW = lsSensorUV + vec2(-1.0,  1.0);
    vec2 lsSE = lsSensorUV + vec2( 1.0,  1.0);

    float nN  = sampleBayerRaw(nN_coord)  * lsGain(lsN);
    float nS  = sampleBayerRaw(nS_coord)  * lsGain(lsS);
    float nW  = sampleBayerRaw(nW_coord)  * lsGain(lsW);
    float nE  = sampleBayerRaw(nE_coord)  * lsGain(lsE);

    float nNW = sampleBayerRaw(nNW_coord) * lsGain(lsNW);
    float nNE = sampleBayerRaw(nNE_coord) * lsGain(lsNE);
    float nSW = sampleBayerRaw(nSW_coord) * lsGain(lsSW);
    float nSE = sampleBayerRaw(nSE_coord) * lsGain(lsSE);

    float center = sampleBayerRaw(sensorCoord) * gain;
    float r = 0.0, g = 0.0, b = 0.0;

    if (color == 0) {
        r = center;
        float diag = (nNW + nNE + nSW + nSE) * 0.25;
        g = (nW + nE + nN + nS) * 0.25;
        b = diag;
    } else if (color == 2) {
        b = center;
        float diag = (nNW + nNE + nSW + nSE) * 0.25;
        g = (nW + nE + nN + nS) * 0.25;
        r = diag;
    } else {
        g = center;
        int colorNS = u_bayer_color_map[safePhase(sensorCoord.x, sensorCoord.y - 1)];
        if (colorNS == 0) {
            r = (nN + nS) * 0.5;
            b = (nW + nE) * 0.5;
        } else {
            r = (nW + nE) * 0.5;
            b = (nN + nS) * 0.5;
        }
    }

    return vec3(r, g, b);
}

vec3 previewNR(usampler2D tex, vec2 uv, vec2 lsUV, float strength) {
    if (strength <= 0.0) {
        return demosaicBilinear(tex, uv, lsUV);
    }
    vec3 center = demosaicBilinear(tex, uv, lsUV);
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 pixelOff = vec2(float(x), float(y));
            vec3 s = demosaicBilinear(tex, uv + pixelOff, lsUV + pixelOff);
            float w = 1.0 / (1.0 + length(s - center));
            sum += s * w;
            wsum += w;
        }
    }
    vec3 filtered = sum / wsum;
    return mix(center, filtered, strength);
}

${AgxCoreGlsl.AGX_FORMATION}

void main() {
    vec2 uv = v_texCoord;
    vec2 lsSensorUV = u_cropOrigin + uv * u_cropSize;

    vec2 sensorUV = u_cropOrigin + uv * u_cropSize;
    sensorUV = clamp(sensorUV, vec2(0.0), u_sensorSize - vec2(1.0));
    lsSensorUV = clamp(lsSensorUV, vec2(0.0), u_sensorSize - vec2(1.0));

    vec3 linearRGB = previewNR(u_bayerTex, sensorUV, lsSensorUV, u_nr_strength);

    vec3 formed = agxFormation(linearRGB);
    fragColor = vec4(formed, 1.0);
}
"""

        private const val DEMOSAIC_FRAGMENT_SHADER = """
#version 300 es
precision highp float;
precision highp usampler2D;
precision highp int;

in vec2 v_texCoord;
out vec4 fragColor;

uniform usampler2D u_bayerTex;
uniform sampler2D u_lens_shading_map;
uniform sampler2D u_denoisedTex;
uniform float u_denoise_active;
uniform float u_scale_factor;
uniform mat4 u_transformMatrix;
uniform mat4 u_inverseTransformMatrix;
uniform vec2 u_outputResolution;
uniform vec2 u_sensorSize;
uniform vec2 u_cropOrigin;
uniform vec2 u_cropSize;

uniform mat3 u_709_to_2020;

uniform ivec4 u_black_level_pattern;
uniform ivec4 u_bayer_color_map;
uniform int u_bit_depth;
uniform int u_box_aa;
uniform float u_white_level;
uniform float u_black_level;
uniform vec3 u_wb_gains;
uniform mat3 u_color_mat;
uniform vec3 u_luma_coeffs;
uniform float u_clip_atten_factor;
uniform float u_dp_strength;
uniform float u_raw_nr_strength;
uniform float u_iso_model_a;
uniform float u_iso_model_b;

${AgxCoreGlsl.CORE_HELPERS}

float unpackRaw(uint rawPacked) {
    uint mask;
    if (u_bit_depth <= 10) {
        mask = 0x3FFu;
    } else if (u_bit_depth <= 12) {
        mask = 0xFFFu;
    } else if (u_bit_depth <= 14) {
        mask = 0x3FFFu;
    } else {
        mask = 0xFFFFu;
    }
    return float(rawPacked & mask);
}

float sampleBayerRaw(ivec2 coord) {
    ivec2 clamped = clamp(coord, ivec2(0), ivec2(u_sensorSize) - ivec2(1));
    uint raw = texelFetch(u_bayerTex, clamped, 0).r;
    float val = unpackRaw(raw);
    int phase = abs(clamped.x % 2) + abs(clamped.y % 2) * 2;
    val -= float(u_black_level_pattern[phase]);
    return max(val, 0.0);
}

int safePhase(int x, int y) {
    return abs(x % 2) + abs(y % 2) * 2;
}

vec2 clampSensor(vec2 v) {
    return clamp(v, vec2(0.0), u_sensorSize - vec2(1.0));
}

float lsGain(vec2 lsNeighborUV) {
    ivec2 c = ivec2(clampSensor(lsNeighborUV));
    return texture(u_lens_shading_map, vec2(c) / u_sensorSize)[safePhase(c.x, c.y)];
}

// Reverse-map a clamped sensor coordinate onto the denoised output grid.
// At 1:1 (capture) this is a per-pixel bijection; at downscaled preview the
// mapping collapses whole sensor neighbourhoods onto single output texels.
ivec2 mappedOutCoord(ivec2 clampedCoord) {
    vec2 uv = (vec2(clampedCoord) - u_cropOrigin) / u_cropSize;
    vec4 frag = u_inverseTransformMatrix * vec4(uv, 0.0, 1.0);
    ivec2 oc = ivec2(floor(frag.xy * u_outputResolution));
    return clamp(oc, ivec2(0), ivec2(u_outputResolution) - ivec2(1));
}

// Per-sample same-colour neighbourhood filter for the preview S1/S3 sliders.
// Runs inside the demosaic loop, so the CFA phase is implied by each sample's
// own coordinate (no fixed grid, no grid->sensor collapse).  Ring 1 is the
// 8 step-2 neighbours (radius 2); ring 2 adds the 4 axis-neighbours at step 4
// so the S3 mean spans roughly 9x9 sensor pixels -- wider than the box-AA
// average, which is what keeps S3 visually effective on top of it.
//   S1 = defect correction: centre outside the neighbour range AND beyond
//        band -> replaced by the α-trimmed neighbour mean.
//   S3 = blend toward that same trimmed mean.  The trim (drop 1 max & 1 min)
//        is what keeps defective neighbour values from inflating the smoothing
//        average, and the outlier guard below also engages at high S3 so a hot
//        centre can't be re-injected by the blending.
float sampleSameColorNR(ivec2 coord) {
    float c = sampleBayerRaw(coord);
    if (u_dp_strength <= 0.0 && u_raw_nr_strength <= 0.0) return c;

    float nE  = sampleBayerRaw(coord + ivec2( 2, 0));
    float nW  = sampleBayerRaw(coord + ivec2(-2, 0));
    float nN  = sampleBayerRaw(coord + ivec2( 0,-2));
    float nS  = sampleBayerRaw(coord + ivec2( 0, 2));
    float nNE = sampleBayerRaw(coord + ivec2( 2,-2));
    float nNW = sampleBayerRaw(coord + ivec2(-2,-2));
    float nSE = sampleBayerRaw(coord + ivec2( 2, 2));
    float nSW = sampleBayerRaw(coord + ivec2(-2, 2));
    float nEE = sampleBayerRaw(coord + ivec2( 4, 0));
    float nWW = sampleBayerRaw(coord + ivec2(-4, 0));
    float nNN = sampleBayerRaw(coord + ivec2( 0,-4));
    float nSS = sampleBayerRaw(coord + ivec2( 0, 4));

    // Bright-pixel guard (clip-safe but mean-stable): the sensor clips in the
    // Bayer domain, so a genuinely chromatic centre must NOT be dragged down
    // toward the filtered mean — that was the "softening" at hotspots.  The
    // final S3 blend is therefore skipped when the centre itself sits at/above
    // the clip, keeping a clipped plateau/specular at its measured brightness.
    // The α-trim mean itself stays the full 12-member form: excluding clipped
    // members would starve the mean right at the plateau edge, turning it
    // per-pixel noisy (blocky water-stain patches radiating from highlights).
    // S1 (DPC) is unchanged and never skipped.  Non-clip pixels are
    // bit-identical to the baseline filter.
    float clipLo = max(u_white_level - u_black_level, 1.0);

    float sumN = nE + nW + nN + nS + nNE + nNW + nSE + nSW + nEE + nWW + nNN + nSS;
    float mn = min(min(min(min(min(nE, nW), min(nN, nS)), min(nNE, nNW)), min(nSE, nSW)),
                  min(min(nEE, nWW), min(nNN, nSS)));
    float mx = max(max(max(max(max(nE, nW), max(nN, nS)), max(nNE, nNW)), max(nSE, nSW)),
                  max(max(nEE, nWW), max(nNN, nSS)));
    float iavg = (sumN - mn - mx) * (1.0 / 10.0);

    float sigma = sqrt(max(u_iso_model_a * max(iavg, 0.0) + u_iso_model_b, 1.0));
    float band = max((0.1 + 0.3 * u_dp_strength) * max(iavg, 0.0),
                     (2.0 + 2.0 * u_dp_strength) * sigma);

    float corrStrength = max(u_dp_strength, 0.85 * u_raw_nr_strength);
    float center = c;
    if (corrStrength > 0.0) {
        bool hot = (c > mx) && (c - iavg) > band;
        bool cold = (c < mn) && (iavg - c) > band;
        if (hot || cold) {
            center = mix(c, iavg, corrStrength);
        }
    }

    if (c < clipLo) {
        return mix(center, iavg, 0.98 * u_raw_nr_strength);
    }
    return center;
}

// Spatial-NR-aware RAW sample.  When the output-driven denoiser produced a
// denoised mosaiced frame, invert the reverse map (sensor -> output grid) and
// take the denoised value for the requested CFA phase.  The denoiser writes
// one spatial estimate per phase into r/g/b/a of every output texel (which is
// the whole 4-phase mosaic cell), so the colour plane is selected by channel
// index — correct at every zoom AND in 1:1 stills, with no guard arithmetic.
float denoisedSampleRaw(ivec2 sensorCoord) {
    ivec2 clampedCoord = clamp(sensorCoord, ivec2(0), ivec2(u_sensorSize) - ivec2(1));
    if (u_denoise_active > 0.5) {
        // Extreme resize capture (a whole filter footprint << one output texel,
        // k >= 16) cannot be represented by one per-texel mosaic cell, so defer
        // to the inline per-sample filter there; below that the mosaic is the
        // cheap and validated read at every zoom (this guard also covers a
        // stale pack from a previous frame).
        vec2 kk = u_cropSize / u_outputResolution;
        if (max(kk.x, kk.y) < 16.0) {
        // The mosaic holds one filter estimate per output texel + CFA phase.
        // A NEAREST read would collate every sensor coord that maps inside an
        // output texel onto THAT single estimate, quantizing the preview to
        // per-texel steps the instant S1/S3 turn on.  Sample bilinearly
        // instead: at 1:1 the read falls exactly on a texel centre
        // (bit-identical), and at any other zoom it smoothly blends the four
        // surrounding per-phase estimates — continuous like the raw box-AA
        // demosaic (see the boxed plain phase means in the pack at k>=3.5),
        // but lower-noise and artifact-free.
        // (GL_LINEAR itself is illegal on RGBA32F in ES 3.0, so the blend is
        // done with 4 texelFetch + linear weights — matches the model 1:1 and
        // works on any float format including the RGBA16F capture mosaic.)
        vec2 inv = (vec2(clampedCoord) - u_cropOrigin) / u_cropSize;
        vec4 frag = u_inverseTransformMatrix * vec4(inv, 0.0, 1.0);
        // Manual 4-tap bilinear via texelFetch.  GL_LINEAR is ILLEGAL on the
        // RGBA32F mosaic (ES 3.0 float textures don't filter past 16 bits), so
        // the hardware silently degraded the read to NEAREST -> per-texel
        // quantized phase estimates (blocky "magenta mosaic" at 1..3.5x zoom).
        // texelFetch is format-agnostic, matches the model 1:1, and 1:1 zoom
        // still lands exactly on a texel centre (w = 0) for bit-identical
        // capture.
        vec2 tc = clamp(frag.xy * u_outputResolution + 0.5, vec2(0.5), u_outputResolution - 0.5);
        vec2 t0f = floor(tc - 0.5);
        vec2 t0 = clamp(t0f, vec2(0.0), u_outputResolution - 1.0);
        vec2 t1 = min(t0 + 1.0, u_outputResolution - 1.0);
        vec2 wts = clamp(tc - (t0 + 0.5), vec2(0.0), vec2(1.0));
        vec4 d00 = texelFetch(u_denoisedTex, ivec2(t0), 0);
        vec4 d10 = texelFetch(u_denoisedTex, ivec2(int(t1.x), int(t0.y)), 0);
        vec4 d01 = texelFetch(u_denoisedTex, ivec2(int(t0.x), int(t1.y)), 0);
        vec4 d11 = texelFetch(u_denoisedTex, ivec2(t1), 0);
        vec4 den = mix(mix(d00, d10, wts.x), mix(d01, d11, wts.x), wts.y);
        int want = safePhase(clampedCoord.x, clampedCoord.y);
        if (want == 0) return den.r;
        if (want == 1) return den.g;
        if (want == 2) return den.b;
        return den.a;
        }
    }
    return sampleSameColorNR(clampedCoord);
}

vec3 demosaicAt(ivec2 sensorCoord, vec2 lsSensorUV) {
    int phase = safePhase(sensorCoord.x, sensorCoord.y);
    int color = u_bayer_color_map[phase];

    float gain = texture(u_lens_shading_map, lsSensorUV / u_sensorSize)[phase];

    ivec2 nN_coord = sensorCoord + ivec2( 0, -1);
    ivec2 nS_coord = sensorCoord + ivec2( 0,  1);
    ivec2 nW_coord = sensorCoord + ivec2(-1,  0);
    ivec2 nE_coord = sensorCoord + ivec2( 1,  0);
    ivec2 nNW_coord = sensorCoord + ivec2(-1, -1);
    ivec2 nNE_coord = sensorCoord + ivec2( 1, -1);
    ivec2 nSW_coord = sensorCoord + ivec2(-1,  1);
    ivec2 nSE_coord = sensorCoord + ivec2( 1,  1);

    vec2 lsN  = lsSensorUV + vec2( 0.0, -1.0);
    vec2 lsS  = lsSensorUV + vec2( 0.0,  1.0);
    vec2 lsW  = lsSensorUV + vec2(-1.0,  0.0);
    vec2 lsE  = lsSensorUV + vec2( 1.0,  0.0);
    vec2 lsNW = lsSensorUV + vec2(-1.0, -1.0);
    vec2 lsNE = lsSensorUV + vec2( 1.0, -1.0);
    vec2 lsSW = lsSensorUV + vec2(-1.0,  1.0);
    vec2 lsSE = lsSensorUV + vec2( 1.0,  1.0);

    float nN  = denoisedSampleRaw(nN_coord)  * lsGain(lsN);
    float nS  = denoisedSampleRaw(nS_coord)  * lsGain(lsS);
    float nW  = denoisedSampleRaw(nW_coord)  * lsGain(lsW);
    float nE  = denoisedSampleRaw(nE_coord)  * lsGain(lsE);

    float nNW = denoisedSampleRaw(nNW_coord) * lsGain(lsNW);
    float nNE = denoisedSampleRaw(nNE_coord) * lsGain(lsNE);
    float nSW = denoisedSampleRaw(nSW_coord) * lsGain(lsSW);
    float nSE = denoisedSampleRaw(nSE_coord) * lsGain(lsSE);

    float center = denoisedSampleRaw(sensorCoord) * gain;
    float r = 0.0, g = 0.0, b = 0.0;

    if (color == 0) {
        r = center;
        float diag = (nNW + nNE + nSW + nSE) * 0.25;
        g = (nW + nE + nN + nS) * 0.25;
        b = diag;
    } else if (color == 2) {
        b = center;
        float diag = (nNW + nNE + nSW + nSE) * 0.25;
        g = (nW + nE + nN + nS) * 0.25;
        r = diag;
    } else {
        g = center;
        int colorNS = u_bayer_color_map[safePhase(sensorCoord.x, sensorCoord.y - 1)];
        if (colorNS == 0) {
            r = (nN + nS) * 0.5;
            b = (nW + nE) * 0.5;
        } else {
            r = (nW + nE) * 0.5;
            b = (nN + nS) * 0.5;
        }
    }

    return vec3(r, g, b);
}

vec3 demosaicBilinear(usampler2D tex, vec2 sensorUV, vec2 lsSensorUV) {
    if (u_box_aa <= 1) {
        return demosaicAt(ivec2(clampSensor(sensorUV)), lsSensorUV);
    }
    vec2 base = floor(sensorUV) - vec2(1.0);
    vec3 sum = vec3(0.0);
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            ivec2 c = clamp(ivec2(base) + ivec2(dx, dy), ivec2(0), ivec2(u_sensorSize) - ivec2(1));
            vec2 lsC = lsSensorUV + vec2(float(dx) - 1.0, float(dy) - 1.0);
            sum += demosaicAt(c, lsC);
        }
    }
    return sum * (1.0 / 16.0);
}

// Simplified low-side (negative) compensation: lift the
// signal so its lowest channel is zero, then rescale so the peak channel
// (intensity) is unchanged. No-op when nothing is negative.
vec3 compensateNegatives(vec3 rgb) {
    float minimum = min(rgb.r, min(rgb.g, rgb.b));
    if (minimum >= 0.0) {
        return rgb;
    }
    float peak = max(rgb.r, max(rgb.g, rgb.b));
    float liftedPeak = peak - minimum;
    float ratio = (liftedPeak > 0.0) ? (peak / liftedPeak) : 0.0;
    return max((rgb - minimum) * ratio, vec3(0.0));
}

void main() {
    vec2 uv = v_texCoord;
    vec2 lsSensorUV = u_cropOrigin + uv * u_cropSize;

    vec2 sensorUV = u_cropOrigin + uv * u_cropSize;
    sensorUV = clamp(sensorUV, vec2(0.0), u_sensorSize - vec2(1.0));
    lsSensorUV = clamp(lsSensorUV, vec2(0.0), u_sensorSize - vec2(1.0));

    vec3 linearRGB = demosaicBilinear(u_bayerTex, sensorUV, lsSensorUV);
    linearRGB *= u_wb_gains;

    // Clipping neutralization: the sensor clips in the Bayer domain, so after
    // demosaic + white balance the clipped regions pick up a color cast
    // (often magenta). Normalize to the sensor clipping value, collapse the
    // chrominance toward an intensity-preserving neutral (the peak channel,
    // max(r,g,b)) by an attenuation factor that fades in as the luminance
    // approaches the clip, then undo the normalization. Peak-preserving
    // keeps the clipped regions' intensity intact instead of dragging them to
    // the luminance average (which reads as darker). The exponent's leading
    // factor (u_clip_atten_factor) is a slider. At factor == 0 the step is
    // skipped.
    if (u_clip_atten_factor > 0.0) {
        float clipScalar = max(u_white_level - u_black_level, 1.0);
        vec3 norm = linearRGB / clipScalar;
        norm = compensateNegatives(norm);
        float luma = max(dot(u_luma_coeffs, norm), 0.0);
        float peak = max(norm.r, max(norm.g, norm.b));
        float inverted = max(1.0 - luma, 0.0);
        float attenuation = pow(inverted, u_clip_atten_factor * 5.0);
        linearRGB = (norm - vec3(peak)) * attenuation + vec3(peak);
        linearRGB *= clipScalar;
    }

    linearRGB = u_color_mat * linearRGB;
    linearRGB = linearRGB / max(u_white_level - u_black_level, 1.0);
    fragColor = vec4(linearRGB, 1.0);
}
"""

        // S1/S3 same-colour pack fallback: precomputes sampleSameColorNR (12-tap α-
        // trimmed same-colour mean + DPC outlier correction, identical GLSL to
        // the demosaic's inline filter) once per output texel and CFA phase.
        // The phase is derived from the texel's own sensor cell parity so the
        // demosaic's per-phase channel lookup (safePhase of the reverse-mapped
        // coordinate) returns exactly the value computed for that position —
        // phase correctness is guaranteed at pack time, never by rewiring
        // stored channels.  Renders to the caller-bound RGBA32F target, where
        // the demosaic reverse-maps it as a fallback when the DPC/GF chain is
        // unavailable; otherwise the identical filter runs inline per sample.
        private const val S3_PACK_FRAGMENT_SHADER = """
#version 300 es
precision highp float;
precision highp usampler2D;
precision highp int;

in vec2 v_texCoord;
out vec4 outDenoised;

uniform usampler2D u_bayerTex;
uniform vec2 u_sensorSize;
uniform vec2 u_cropOrigin;
uniform vec2 u_cropSize;
uniform vec2 u_outputResolution;
uniform mat4 u_transformMatrix;
uniform ivec4 u_black_level_pattern;
uniform int u_bit_depth;
uniform float u_dp_strength;
uniform float u_raw_nr_strength;
uniform float u_iso_model_a;
uniform float u_iso_model_b;
uniform float u_white_level;
uniform float u_black_level;
uniform vec4 u_pack_wb_gains;

float unpackRaw(uint rawPacked) {
    uint mask;
    if (u_bit_depth <= 10) {
        mask = 0x3FFu;
    } else if (u_bit_depth <= 12) {
        mask = 0xFFFu;
    } else if (u_bit_depth <= 14) {
        mask = 0x3FFFu;
    } else {
        mask = 0xFFFFu;
    }
    return float(rawPacked & mask);
}

float sampleBayerRaw(ivec2 coord) {
    ivec2 clamped = clamp(coord, ivec2(0), ivec2(u_sensorSize) - ivec2(1));
    uint raw = texelFetch(u_bayerTex, clamped, 0).r;
    float val = unpackRaw(raw);
    int phase = abs(clamped.x % 2) + abs(clamped.y % 2) * 2;
    val -= float(u_black_level_pattern[phase]);
    return max(val, 0.0);
}

float sampleSameColorNR(ivec2 coord) {
    float c = sampleBayerRaw(coord);
    if (u_dp_strength <= 0.0 && u_raw_nr_strength <= 0.0) return c;

    float nE  = sampleBayerRaw(coord + ivec2( 2, 0));
    float nW  = sampleBayerRaw(coord + ivec2(-2, 0));
    float nN  = sampleBayerRaw(coord + ivec2( 0,-2));
    float nS  = sampleBayerRaw(coord + ivec2( 0, 2));
    float nNE = sampleBayerRaw(coord + ivec2( 2,-2));
    float nNW = sampleBayerRaw(coord + ivec2(-2,-2));
    float nSE = sampleBayerRaw(coord + ivec2( 2, 2));
    float nSW = sampleBayerRaw(coord + ivec2(-2, 2));
    float nEE = sampleBayerRaw(coord + ivec2( 4, 0));
    float nWW = sampleBayerRaw(coord + ivec2(-4, 0));
    float nNN = sampleBayerRaw(coord + ivec2( 0,-4));
    float nSS = sampleBayerRaw(coord + ivec2( 0, 4));

    // Bright-pixel guard (clip-safe but mean-stable): the sensor clips in the
    // Bayer domain, so a genuinely clipped centre must NOT be dragged down
    // toward the filtered mean — that was the "softening" at hotspots.  The
    // final S3 blend is therefore skipped when the centre itself sits at/above
    // the clip, keeping a clipped plateau/specular at its measured brightness.
    // The α-trim mean itself stays the full 12-member form: excluding clipped
    // members would starve the mean right at the plateau edge, turning it
    // per-pixel noisy (blocky water-stain patches radiating from highlights).
    // S1 (DPC) is unchanged and never skipped.  Non-clip pixels are
    // bit-identical to the baseline filter.
    float clipLo = max(u_white_level - u_black_level, 1.0);

    float sumN = nE + nW + nN + nS + nNE + nNW + nSE + nSW + nEE + nWW + nNN + nSS;
    float mn = min(min(min(min(min(nE, nW), min(nN, nS)), min(nNE, nNW)), min(nSE, nSW)),
                  min(min(nEE, nWW), min(nNN, nSS)));
    float mx = max(max(max(max(max(nE, nW), max(nN, nS)), max(nNE, nNW)), max(nSE, nSW)),
                  max(max(nEE, nWW), max(nNN, nSS)));
    float iavg = (sumN - mn - mx) * (1.0 / 10.0);

    float sigma = sqrt(max(u_iso_model_a * max(iavg, 0.0) + u_iso_model_b, 1.0));
    float band = max((0.1 + 0.3 * u_dp_strength) * max(iavg, 0.0),
                     (2.0 + 2.0 * u_dp_strength) * sigma);

    float corrStrength = max(u_dp_strength, 0.85 * u_raw_nr_strength);
    float center = c;
    if (corrStrength > 0.0) {
        bool hot = (c > mx) && (c - iavg) > band;
        bool cold = (c < mn) && (iavg - c) > band;
        if (hot || cold) {
            center = mix(c, iavg, corrStrength);
        }
    }

    if (c < clipLo) {
        return mix(center, iavg, 0.98 * u_raw_nr_strength);
    }
    return center;
}

// Boxed mosaic estimate: the texel's whole sensor footprint is averaged per
// CFA phase BEFORE any noise correction (box-average-before-filter).  At wide
// zoom a phase box holds many cells and is averaged to a single estimate per
// phase (plain = already phase-targeted noise-free samples, so the mean is
// lower-noise than inline per-sample filtering) — the interpolated read keeps
// it continuous.  At 1:1 the box covers exactly one 2x2 cell and the anchored
// per-phase path is preserved (bit-identical capture).  Only a whole filter
// footprint smaller than one output texel (k>=16, extreme resize capture)
// skips the pack in favour of the inline filter (see denoisedSampleRaw).
vec4 boxedBlend(vec4 b) {
    // Cross-phase trim/drag must operate in photometrically NEUTRAL space: the
    // four per-phase box means sit at each CFA base (R < G ≈ G > B on a real
    // sensor under neutral light), so a raw-space iavg pulls the R/B means
    // toward the green phases and the demosaic WB gains later magnify that
    // residual into a magenta cast.  Normalizing each phase by its WB gain
    // makes the drag a fixed point on any WB-correct neutral field while
    // keeping the DPC hot/cold pull pixel-photometric.
    vec4 bN = b * u_pack_wb_gains;
    float mn = min(min(min(bN.r, bN.g), bN.b), bN.a);
    float mxx = max(max(max(bN.r, bN.g), bN.b), bN.a);
    float iavg = (bN.r + bN.g + bN.b + bN.a - mn - mxx) * 0.5;

    float sigma = sqrt(max(u_iso_model_a * max(iavg, 0.0) + u_iso_model_b, 1.0));
    float band = max((0.1 + 0.3 * u_dp_strength) * max(iavg, 0.0),
                     (2.0 + 2.0 * u_dp_strength) * sigma);
    float corr = max(u_dp_strength, 0.85 * u_raw_nr_strength);
    float clipLo = max(u_white_level - u_black_level, 1.0);

    vec4 omxx = vec4(max(max(bN.g, bN.b), bN.a),
                     max(max(bN.r, bN.b), bN.a),
                     max(max(bN.r, bN.g), bN.a),
                     max(max(bN.r, bN.g), bN.b));
    vec4 omn = vec4(min(min(bN.g, bN.b), bN.a),
                    min(min(bN.r, bN.b), bN.a),
                    min(min(bN.r, bN.g), bN.a),
                    min(min(bN.r, bN.g), bN.b));

    vec4 o;
    for (int p = 0; p < 4; p++) {
        float c = bN[p];
        float center = c;
        if (corr > 0.0) {
            bool hot = (c > omxx[p]) && (c - iavg) > band;
            bool cold = (c < omn[p]) && (iavg - c) > band;
            if (hot || cold) center = c + (iavg - c) * corr;
        }
        float outN = (b[p] < clipLo) ? center + (iavg - center) * (0.98 * u_raw_nr_strength) : center;
        o[p] = outN / u_pack_wb_gains[p];
    }
    return o;
}

void main() {
    // This pass renders into the demosaic-sized FBO, so gl_FragCoord.xy-0.5 is
    // the output texel index.  The footprint must use the SAME mapping the
    // demosaic inverts (u_inverseTransformMatrix on normalized crop coords), so
    // the box is derived through the forward view transform — otherwise any
    // mirror/rotation in the preview matches a horizontally flipped footprint
    // (magenta channel-mix overlay on mirrored/rotated sensors).
    vec2 texelIdx = gl_FragCoord.xy - vec2(0.5);
    vec2 n0 = (u_transformMatrix * vec4(texelIdx / u_outputResolution, 0.0, 1.0)).xy;
    vec2 n1 = (u_transformMatrix * vec4((texelIdx + 1.0) / u_outputResolution, 0.0, 1.0)).xy;
    vec2 loRaw = u_cropOrigin + n0 * u_cropSize;
    vec2 hiRaw = u_cropOrigin + n1 * u_cropSize;
    // Affine view transforms (mirror/flip/rotate) can swap the order of the two
    // footprints; the cell set is the span between them either way.
    vec2 lo = min(loRaw, hiRaw);
    vec2 hi = max(loRaw, hiRaw);
    ivec2 b0 = ivec2(floor(lo));
    ivec2 b1 = max(ivec2(floor(hi)) - 1, b0);
    ivec2 boxN = max(b1 - b0 + 1, ivec2(1));
    int nCells = boxN.x * boxN.y;

    if (nCells <= 1) {
        // 1:1 capture / zoom-in: anchored per-phase filter, exactly the current
        // pack behaviour (bit-identical to the pre-boxed path at identity).
        // Anchor at the texel CENTRE (like the old v_texCoord read) so the cell
        // matches the reverse-map midpoint for fractional zooms.
        vec2 n0c = (u_transformMatrix * vec4((texelIdx + vec2(0.5)) / u_outputResolution, 0.0, 1.0)).xy;
        vec2 sensorUV = u_cropOrigin + n0c * u_cropSize;
        sensorUV = clamp(sensorUV, vec2(0.0), u_sensorSize - vec2(1.0));
        ivec2 sc = ivec2(floor(sensorUV));

        int parityX = abs(sc.x % 2);
        int parityY = abs(sc.y % 2);

        vec4 result = vec4(0.0);
        for (int p = 0; p < 4; p++) {
            int phaseX = p & 1;
            int phaseY = p >> 1;
            // Phase p's sensor pixel within this texel's 2x2 cell: flip the cell
            // origin's parity where it does not match p, so safePhase(cc) == p.
            ivec2 cc = sc + ivec2(parityX ^ phaseX, parityY ^ phaseY);
            cc = clamp(cc, ivec2(0), ivec2(u_sensorSize) - ivec2(1));
            result[p] = sampleSameColorNR(cc);
        }
        outDenoised = result;
        return;
    }

    // Box-average-before-filter per CFA phase (max 16x16 box, matching the
    // host pack range k < 16; beyond that the demosaic read defers to inline).
    float pSum[4];
    int pN[4];
    for (int p = 0; p < 4; p++) { pSum[p] = 0.0; pN[p] = 0; }
    for (int iy = 0; iy < 16; iy++) {
        if (b0.y + iy > b1.y) break;
        int yy = clamp(b0.y + iy, 0, int(u_sensorSize.y) - 1);
        for (int ix = 0; ix < 16; ix++) {
            if (b0.x + ix > b1.x) break;
            int xx = clamp(b0.x + ix, 0, int(u_sensorSize.x) - 1);
            int phase = abs(xx % 2) + abs(yy % 2) * 2;
            pSum[phase] += sampleBayerRaw(ivec2(xx, yy));
            pN[phase]++;
        }
    }
    // A phase with no cells in an odd-shaped footprint borrows the mean of the
    // phases that ARE present, so every channel stays populated.
    float tot = 0.0;
    int tn = 0;
    for (int p = 0; p < 4; p++) {
        if (pN[p] > 0) { tot += pSum[p] / float(pN[p]); tn++; }
    }
    if (tn == 0) { tot = 0.0; tn = 1; }
    float fbk = tot / float(tn);
    for (int p = 0; p < 4; p++) {
        if (pN[p] == 0) { pN[p] = 1; pSum[p] = fbk; }
    }
    vec4 b = vec4(pSum[0] / float(pN[0]), pSum[1] / float(pN[1]),
                  pSum[2] / float(pN[2]), pSum[3] / float(pN[3]));

    if (nCells >= 16) {
        // Strong zoom-out: the box already averages most of the sensor texture;
        // further trim-blending re-introduces phase-subset aliasing.  Store the
        // plain phase area means (colour intact, no added noise).
        outDenoised = b;
    } else {
        outDenoised = boxedBlend(b);
    }
}
"""

        private fun createProgram(vertexSource: String, fragmentSource: String): Int {
            val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource)
            val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource)
            if (vertexShader == 0 || fragmentShader == 0) return 0

            val program = GLES20.glCreateProgram()
            GLES20.glAttachShader(program, vertexShader)
            GLES20.glAttachShader(program, fragmentShader)
            GLES20.glLinkProgram(program)

            val linkStatus = IntArray(1)
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0)
            if (linkStatus[0] != GLES20.GL_TRUE) {
                val info = "Program link failed: ${GLES20.glGetProgramInfoLog(program)}"
                Log.e(TAG, info)
                com.agx.camera.CrashLogger.log(TAG, info)
                GLES20.glDeleteProgram(program)
                return 0
            }

            GLES20.glDeleteShader(vertexShader)
            GLES20.glDeleteShader(fragmentShader)
            return program
        }

        private fun loadShader(type: Int, source: String): Int {
            val shader = GLES20.glCreateShader(type)
            GLES20.glShaderSource(shader, source)
            GLES20.glCompileShader(shader)

            val compiled = IntArray(1)
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0)
            if (compiled[0] != GLES20.GL_TRUE) {
                val info = "Shader compile failed: ${GLES20.glGetShaderInfoLog(shader)}"
                Log.e(TAG, info)
                com.agx.camera.CrashLogger.log(TAG, info)
                GLES20.glDeleteShader(shader)
                return 0
            }
            return shader
        }

        private fun floatToHalf(f: Float): Short {
            val bits = java.lang.Float.floatToRawIntBits(f)
            val sign = (bits ushr 16) and 0x8000
            var exp = ((bits ushr 23) and 0xFF) - 127 + 15
            var mantissa = bits and 0x7FFFFF
            if (exp <= 0) {
                exp = 0
                mantissa = 0
            } else if (exp >= 31) {
                exp = 31
                mantissa = 0
            } else {
                mantissa = mantissa shr 13
            }
            return (sign or (exp shl 10) or mantissa).toShort()
        }
    }
}
