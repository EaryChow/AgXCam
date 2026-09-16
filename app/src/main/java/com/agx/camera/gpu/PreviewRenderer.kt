package com.agx.camera.gpu

import android.graphics.SurfaceTexture
import android.opengl.*
import android.util.Log
import android.view.TextureView
import android.graphics.Bitmap
import com.agx.camera.CrashLogger
import com.agx.camera.camera.ZoomController
import com.agx.camera.io.JpegEncoder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.abs

class PreviewRenderer(private val textureView: TextureView) : TextureView.SurfaceTextureListener {

    @Volatile private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    @Volatile private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    @Volatile private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    @Volatile private var eglConfig: EGLConfig? = null

    private var fboId = 0
    private var fboTextureId = 0
    private var fboWidth = 640
    private var fboHeight = 480

    private var captureFboId = 0
    private var captureFboTextureId = 0
    private var captureFboWidth = 0
    private var captureFboHeight = 0
    private var captureFboAllocatedWidth = 0
    private var captureFboAllocatedHeight = 0

    private var downscaleFboId = 0
    private var downscaleFboTextureId = 0
    private var downscaleFboWidth = 0
    private var downscaleFboHeight = 0

    private var demosaicFboId = 0
    private var demosaicFboTextureId = 0
    private var demosaicFboWidth = 0
    private var demosaicFboHeight = 0

    private var rawDemosaicFboId = 0
    private var rawDemosaicFboTextureId = 0
    private var rawDemosaicFboWidth = 0
    private var rawDemosaicFboHeight = 0

    // Spatial-NR output-driven denoise buffers: single RGBA32F texture
    // holding the denoised 4-phase mosaic (R,G1,G2,B), written every frame
    // and consumed by the demosaic pass in the SAME frame via the reverse
    // map.  Spatial-only — no history, no MRT.
    private var denoiseFboId = 0
    private var denoisedTexId = 0
    private var denoiseFboWidth = 0
    private var denoiseFboHeight = 0

    // S1/S3 same-colour pack buffers (RGBA32F denoised 4-phase mosaic).  The
    // pack pass precomputes the S3 filter once per texel / CFA phase and the
    // demosaic reverse-maps it, replacing the inline 13-tap filter that ran
    // per box-AA sample × demosaic neighbourhood (1872 R16UI fetches per
    // output texel at preview, 117 at 1:1 capture).
    private var s3PackFboId = 0
    private var s3PackTexId = 0
    private var s3PackWidth = 0
    private var s3PackHeight = 0

    // Capture (still) spatial-NR buffers: same spatial-only Bayer-domain pass
    // at the capture output resolution; the denoised mosaic is RGBA16F to
    // halve the memory footprint of a full-res still.
    private var captureDenoiseFboId = 0
    private var captureDenoisedTexId = 0
    private var captureDenoiseFboWidth = 0
    private var captureDenoiseFboHeight = 0

    // Stage 1 (DPC) transient buffers: avg/flag maps + two ping-pong work
    // textures for the pass-CORRECT → pass-COUPLET chain. All RGBA32F at the
    // sparse-grid (demosaic) resolution.
    private var dpcAvgFboId = 0
    private var dpcAvgTexId = 0
    private var dpcFlagFboId = 0
    private var dpcFlagTexId = 0
    private var dpcWorkAFboId = 0
    private var dpcWorkATexId = 0
    private var dpcWorkBFboId = 0
    private var dpcWorkBTexId = 0
    private var dpcBufferWidth = 0
    private var dpcBufferHeight = 0

    // Stage 2 (sigma-hat) output texture: RGBA32F, R=σ̂², G=σ̂.
    private var sigmaFboId = 0
    private var sigmaTexId = 0
    private var sigmaBufferWidth = 0
    private var sigmaBufferHeight = 0

    // Stage 5 (output-domain SWGF) buffers: separable box-stats pair and two
    // iteration outputs. RGBA32F at the demosaic resolution.
    private var statsHFboId = 0
    private var statsHTexId = 0
    private var statsVFboId = 0
    private var statsVTexId = 0
    private var outNr1FboId = 0
    private var outNr1TexId = 0
    private var outNr2FboId = 0
    private var outNr2TexId = 0
    private var outNrBufferWidth = 0
    private var outNrBufferHeight = 0

    // 1x1 RGBA32F zero texture bound to float samplers that have no real data
    // behind them (DPC avg/flag inputs while packing, σ̂ fallback).
    private var fallbackFloatTexId = 0

    private val yuvShader = YuvShaderProgram()
    private val bayerShader = BayerShaderProgram()
    private val nrShader = NrShaderProgram()
    private val spatialNrShader = SpatialNrShaderProgram()
    private val blitShader = BlitShaderProgram()
    private val focusPeakShader = FocusPeakShaderProgram()
    private val dpcShader = DpcShaderProgram()
    private val sigmaHatShader = SigmaHatShaderProgram()
    private val rawDenoiseShader = RawDenoiseShaderProgram()
    private val outNrShader = OutNrShaderProgram()

    /** Green focus-peak overlay on the final blit. Enabled while the MF roller is
     *  actively editable (focus panel open + manual focus). [focusPeakRadius] is the
     *  screen-pixel support radius; sharpness is additionally checked at a source-
     *  anchored distance so peaks stay visible when zoomed or on a tele lens. */
    @Volatile var focusPeakEnabled = false
    @Volatile var focusPeakRadius = 3f
    @Volatile var focusPeakThreshold = 0.55f
    @Volatile var focusPeakStrength = 0.8f

    @Volatile var useBayerPath = false
        private set

    @Volatile private var bayerBuffer: ByteBuffer? = null
    @Volatile private var bayerWidth = 0
    @Volatile private var bayerHeight = 0
    private var bayerStridePixels = 0

    var bayerBlackLevelPattern = intArrayOf(64, 64, 64, 64)
    var bayerColorMap = intArrayOf(0, 1, 1, 2)
    var bayerBitDepth: Int
        get() = when {
            agxWhiteLevel <= 1023f -> 10
            agxWhiteLevel <= 4095f -> 12
            agxWhiteLevel <= 16383f -> 14
            else -> 16
        }
        set(value) { /* no-op, computed from agxWhiteLevel */ }
    var bayerNrStrength = 0f
    // Stage sliders (independent, 0 = bypass): S1 DPC, S3 RAW green-guided
    // GF, S5 output-domain SWGF.
    @Volatile var dpcStrength = 0f
    @Volatile var rawNrStrength = 0f
    @Volatile var outNrStrength = 0f
    @Volatile var syntheticTestEnabled = com.agx.camera.BuildConfig.AGX_SYNTHETIC_BAYER
    private var syntheticSensor: ShortArray? = null
    private var syntheticBuffer: java.nio.ByteBuffer? = null
    // Live ISO used to derive the Stage-0 noise-model uniforms for every
    // denoise stage. Pushed by MainActivity from the auto-exposure readout.
    @Volatile var isoForDenoise = 200
    @Volatile var wbGainR = 1f
    @Volatile var wbGainG = 1f
    @Volatile var wbGainB = 1f
    @Volatile var ccMatrix: FloatArray? = null
    // Luminance (Y) coefficients of the camera-native RGB space; used by the
    // demosaic shader's clipping-neutralization step. Defaults to the Rec.709
    // Y row when no camera-native -> XYZ map is available.
    @Volatile var nativeLumaCoeffs = floatArrayOf(0.2126f, 0.7152f, 0.0722f)
    // S5 SWGF must score windows in the same luma space as yccOf
    // (0.25*R + 0.5*G + 0.25*B); the Rec.709 row above is only for
    // demosaic clipping-neutralization and would bias st.r vs yccIn.x.
    val S5_LUMA_WEIGHTS = floatArrayOf(0.25f, 0.5f, 0.25f)
    // Leading factor of the clipping-neutralization exponent (factor * 5).
    @Volatile var clipAttenFactor = 0.1f
    // Lens shading (vignette) gain map for the Bayer/RAW pipeline: RGBA16F
    // half-float pixels, CFA-permuted and Y-flipped, uploaded to
    // u_lens_shading_map by initGlResources() or the GL thread whenever a new
    // map arrives post-init. Null means identity (no correction).
    @Volatile var bayerLensShadingData: ShortArray? = null
    @Volatile var bayerLensShadingWidth = 1
    @Volatile private var bayerLensShadingHeight = 1
    @Volatile private var lensShadingUploadPending = false

    private var renderThread: Thread? = null
    private val renderLock = ReentrantLock()
    private val frameCondition = renderLock.newCondition()
    @Volatile private var hasNewFrame = false
    @Volatile private var running = false

    val zoomController = ZoomController()

    var agxSceneLinearTo709 = floatArrayOf(1f,0f,0f, 0f,1f,0f, 0f,0f,1f)
    var agxInsetMat = floatArrayOf(1f,0f,0f, 0f,1f,0f, 0f,0f,1f)
    var agxOutsetMat = floatArrayOf(1f,0f,0f, 0f,1f,0f, 0f,0f,1f)
    var agxToRec2020 = floatArrayOf(1f,0f,0f, 0f,1f,0f, 0f,0f,1f)
    var agxWhiteLevel = 1023f
    var agxBlackLevel = 64f
    var agxLogMin = -10f
    var agxLogMax = 6.5f
    var agxLogMidgray = 0.5f
    var agxDisplayMidgray = 0.48f
    var agxContrast = 2.4f
    var agxToe = 1.5f
    var agxShoulder = 1.5f
    var agxVibrance = 0.5f
    var exposureEv = 1.5f

    fun setPreviewSize(width: Int, height: Int) {
        fboWidth = width
        fboHeight = height
    }

    fun setCaptureSize(width: Int, height: Int) {
        captureFboWidth = width
        captureFboHeight = height
    }

    private var yPlane: ByteBuffer? = null
    private var uPlane: ByteBuffer? = null
    private var vPlane: ByteBuffer? = null
    private var yuvWidth = 0
    private var yuvHeight = 0

    val currentYPlane: ByteBuffer? get() = yPlane
    val currentUPlane: ByteBuffer? get() = uPlane
    val currentVPlane: ByteBuffer? get() = vPlane
    val currentYuvWidth: Int get() = yuvWidth
    val currentYuvHeight: Int get() = yuvHeight
    val currentBayerWidth: Int get() = bayerWidth
    val currentBayerHeight: Int get() = bayerHeight

    // Aspect of the content actually displayed (the FBO, which is pre-cropped to
    // the preview aspect). Used to reverse the CENTER_INSIDE viewport + aspect
    // crop when mapping taps back to sensor coordinates.
    val currentContentAspect: Float
        get() = if (fboWidth > 0 && fboHeight > 0) fboWidth.toFloat() / fboHeight.toFloat() else 0f

    var onFirstFrameRendered: (() -> Unit)? = null
    var onFrameRendered: ((Long) -> Unit)? = null
    var onDegradedModeChanged: ((String?) -> Unit)? = null
    private var firstFrameReported = false

    var degradedManager: DegradedPreviewManager? = null
    private var frameCount = 0

    var pendingFboReadback: ((Int, Int, Int) -> Unit)? = null

    @Volatile var sensorOrientation: Int = 0
    @Volatile var isFrontCamera: Boolean = false
    @Volatile var targetAspectRatio: Float = 0f // kept for metadata/debugging; cropping is driven by the FBO preview aspect

    // Center-crop scale so the SOURCE content (sensor/sensor crop for the bayer
    // path, YUV frame for the yuv path) is sampled at the FBO's aspect instead of
    // being stretched to fill it. The FBO aspect is the preview aspect chosen for
    // the requested ratio, so this crops the wider/taller source to that aspect.
    private fun computeCropScale(sourceW: Int, sourceH: Int): Pair<Float, Float> {
        if (sourceW <= 0 || sourceH <= 0 || fboWidth <= 0 || fboHeight <= 0) return Pair(1f, 1f)
        val sourceAspect = sourceW.toFloat() / sourceH.toFloat()
        val fboAspect = fboWidth.toFloat() / fboHeight.toFloat()
        return if (sourceAspect > fboAspect) {
            // Source is wider than the output aspect → crop horizontal (scale X < 1)
            Pair(fboAspect / sourceAspect, 1f)
        } else {
            // Source is taller than the output aspect → crop vertical (scale Y < 1)
            Pair(1f, sourceAspect / fboAspect)
        }
    }

    private fun computePreviewTransform(sourceW: Int, sourceH: Int): FloatArray {
        val (cropScaleX, cropScaleY) = computeCropScale(sourceW, sourceH)
        val matrix = FloatArray(16)
        android.opengl.Matrix.setIdentityM(matrix, 0)

        android.opengl.Matrix.translateM(matrix, 0, 0.5f, 0.5f, 0f)

        // Apply crop scale for target aspect ratio (center crop)
        android.opengl.Matrix.scaleM(matrix, 0, cropScaleX, cropScaleY, 1f)

        if (isFrontCamera) {
            android.opengl.Matrix.scaleM(matrix, 0, -1f, -1f, 1f)
        } else {
            android.opengl.Matrix.scaleM(matrix, 0, 1f, -1f, 1f)
        }

        android.opengl.Matrix.translateM(matrix, 0, -0.5f, -0.5f, 0f)
        return matrix
    }

    // Build the RAW-capture demosaic matrix: the usual Y-flip plus a center crop
    // so the full sensor is framed at the OUTPUT aspect (matching the preview),
    // instead of being stretched into it.
    private fun buildRawCaptureMatrix(sourceW: Int, sourceH: Int, outW: Int, outH: Int): FloatArray {
        val matrix = FloatArray(16).also { android.opengl.Matrix.setIdentityM(it, 0) }
        var cropScaleX = 1f
        var cropScaleY = 1f
        if (sourceW > 0 && sourceH > 0 && outW > 0 && outH > 0) {
            val sourceAspect = sourceW.toFloat() / sourceH.toFloat()
            val outAspect = outW.toFloat() / outH.toFloat()
            if (sourceAspect > outAspect) {
                cropScaleX = outAspect / sourceAspect
            } else {
                cropScaleY = sourceAspect / outAspect
            }
        }
        android.opengl.Matrix.translateM(matrix, 0, 0.5f, 0.5f, 0f)
        android.opengl.Matrix.scaleM(matrix, 0, cropScaleX, cropScaleY, 1f)
        android.opengl.Matrix.scaleM(matrix, 0, 1f, -1f, 1f)
        android.opengl.Matrix.translateM(matrix, 0, -0.5f, -0.5f, 0f)
        return matrix
    }

    private data class CaptureFrame(
        val y: ByteBuffer, val u: ByteBuffer, val v: ByteBuffer,
        val w: Int, val h: Int,
        val targetW: Int, val targetH: Int,
        val agxSceneLinearTo709: FloatArray,
        val agxInsetMat: FloatArray,
        val agxOutsetMat: FloatArray,
        val agxToRec2020: FloatArray,
        val agxWhiteLevel: Float, val agxBlackLevel: Float,
        val agxLogMin: Float, val agxLogMax: Float,
        val agxLogMidgray: Float, val agxDisplayMidgray: Float,
        val agxContrast: Float, val agxToe: Float, val agxShoulder: Float,
        val agxVibrance: Float,
        val sensorOrientation: Int,
        val deviceOrientation: Int,
        val resultRef: AtomicReference<Bitmap?>,
        val latch: CountDownLatch
    )

    @Volatile private var pendingCaptureFrame: CaptureFrame? = null

    data class RawFrameCopy(val buffer: ByteBuffer, val width: Int, val height: Int)

    fun pullBayerCopy(): RawFrameCopy? {
        val src = bayerBuffer ?: return null
        if (bayerWidth <= 0 || bayerHeight <= 0) return null
        val copy = ByteBuffer.allocateDirect(bayerWidth * bayerHeight * 2)
        val view = src.asReadOnlyBuffer()
        view.position(0)
        view.limit(src.capacity())
        copy.put(view)
        copy.position(0)
        return RawFrameCopy(copy, bayerWidth, bayerHeight)
    }

    internal fun submitCaptureFrame(
        y: ByteBuffer, u: ByteBuffer, v: ByteBuffer,
        w: Int, h: Int,
        targetW: Int, targetH: Int,
        session: com.agx.camera.MainActivity.CaptureSession,
        resultRef: AtomicReference<Bitmap?>,
        latch: CountDownLatch
    ) {
        pendingCaptureFrame = CaptureFrame(
            y, u, v, w, h, targetW, targetH,
            session.agxSceneLinearTo709, session.agxInsetMat, session.agxOutsetMat, session.agxToRec2020,
            session.agxWhiteLevel, session.agxBlackLevel,
            session.agxLogMin, session.agxLogMax,
            session.agxLogMidgray, session.agxDisplayMidgray,
            session.agxContrast, session.agxToe, session.agxShoulder,
            session.agxVibrance,
            session.sensorOrientation, session.deviceOrientation,
            resultRef, latch
        )
        hasNewFrame = true
        renderLock.withLock { frameCondition.signal() }
    }

    private data class RawCaptureFrame(
        val buffer: ByteBuffer, val rawW: Int, val rawH: Int, val stridePixels: Int,
        val targetW: Int, val targetH: Int,
        val agxSceneLinearTo709: FloatArray,
        val agxInsetMat: FloatArray,
        val agxOutsetMat: FloatArray,
        val agxToRec2020: FloatArray,
        val agxWhiteLevel: Float, val agxBlackLevel: Float,
        val agxLogMin: Float, val agxLogMax: Float,
        val agxLogMidgray: Float, val agxDisplayMidgray: Float,
        val agxContrast: Float, val agxToe: Float, val agxShoulder: Float,
        val agxVibrance: Float,
        val resultRef: AtomicReference<Bitmap?>,
        val latch: CountDownLatch
    )

    @Volatile private var pendingRawCaptureFrame: RawCaptureFrame? = null

    internal fun submitRawCaptureFrame(
        buffer: ByteBuffer, rawW: Int, rawH: Int, stridePixels: Int,
        targetW: Int, targetH: Int,
        session: com.agx.camera.MainActivity.CaptureSession,
        resultRef: AtomicReference<Bitmap?>,
        latch: CountDownLatch
    ) {
        pendingRawCaptureFrame = RawCaptureFrame(
            buffer, rawW, rawH, stridePixels, targetW, targetH,
            session.agxSceneLinearTo709, session.agxInsetMat, session.agxOutsetMat, session.agxToRec2020,
            session.agxWhiteLevel, session.agxBlackLevel,
            session.agxLogMin, session.agxLogMax,
            session.agxLogMidgray, session.agxDisplayMidgray,
            session.agxContrast, session.agxToe, session.agxShoulder,
            session.agxVibrance,
            resultRef, latch
        )
        hasNewFrame = true
        renderLock.withLock { frameCondition.signal() }
    }

    fun setYuvFrame(y: ByteBuffer, u: ByteBuffer, v: ByteBuffer, width: Int, height: Int) {
        yPlane = y
        uPlane = u
        vPlane = v
        yuvWidth = width
        yuvHeight = height
        hasNewFrame = true
        renderLock.withLock {
            frameCondition.signal()
        }
    }

    fun enableBayerMode(sensorWidth: Int, sensorHeight: Int) {
        bayerWidth = sensorWidth
        bayerHeight = sensorHeight
        useBayerPath = true
        Log.d(TAG, "Bayer mode enabled: ${sensorWidth}x${sensorHeight}")
    }

    fun disableBayerMode() {
        useBayerPath = false
        bayerBuffer = null
        Log.d(TAG, "Bayer mode disabled")
    }

    fun setBayerFrame(buffer: ByteBuffer, width: Int, height: Int, stridePixels: Int) {
        bayerBuffer = buffer
        bayerWidth = width
        bayerHeight = height
        bayerStridePixels = stridePixels
        hasNewFrame = true
        renderLock.withLock {
            frameCondition.signal()
        }
    }

    /** Push a freshly-parsed lens shading gain map (§15.1). [pixelsRgba16f] is
     *  RGBA16F half-float data, already permuted per the active CFA and
     *  Y-flipped for direct glTexImage2D upload. If GL is not initialized yet,
     *  initGlResources() picks it up; otherwise the upload runs on the GL
     *  thread before the next frame. */
    fun submitLensShadingMap(pixelsRgba16f: ShortArray, width: Int, height: Int) {
        if (width <= 0 || height <= 0 || pixelsRgba16f.size < width * height * 4) return
        bayerLensShadingData = pixelsRgba16f
        bayerLensShadingWidth = width
        bayerLensShadingHeight = height
        lensShadingUploadPending = true
        requestRender()
    }

    /** Drop the current lens shading map back to the identity no-op. Used when
     *  a different lens is opened — its own calibration arrives with the first
     *  CaptureResults, and applying the previous lens's map in between would
     *  vignette the preview wrong. */
    fun resetLensShading() {
        bayerLensShadingData = null
        bayerLensShadingWidth = 1
        bayerLensShadingHeight = 1
        lensShadingUploadPending = true
        requestRender()
    }

    fun start() {
        running = true
        renderThread = Thread({ renderLoop() }, "PreviewRenderer").also { it.start() }
    }

    fun stop() {
        running = false
        lensShadingUploadPending = false
        renderLock.withLock {
            frameCondition.signal()
        }
        renderThread?.join(2000)
        renderThread = null
        destroyEgl()
    }

    fun requestRender() {
        hasNewFrame = true
        renderLock.withLock {
            frameCondition.signal()
        }
    }

    private var bayerRenderCount = 0
    private var lastBayerCropLog: String? = null
    private var lastProbeSig = ""
    private var lastDenoiseSliderSig = ""

    private fun renderYuvFrame(viewW: Int, viewH: Int) {
        val y = yPlane
        val u = uPlane
        val v = vPlane
        val w = yuvWidth
        val h = yuvHeight
        if (y == null || u == null || v == null || w <= 0 || h <= 0) {
            clearAndSwap(viewW, viewH)
            return
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId)
        GLES20.glViewport(0, 0, fboWidth, fboHeight)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        yuvShader.uploadY(y.duplicate(), w, h)
        yuvShader.uploadU(u.duplicate(), w / 2, h / 2)
        yuvShader.uploadV(v.duplicate(), w / 2, h / 2)

        yuvShader.draw(
            fboWidth, fboHeight,
            computePreviewTransform(w, h),
            exposureEv,
            agxSceneLinearTo709,
            agxInsetMat,
            agxOutsetMat,
            agxToRec2020,
            agxWhiteLevel, agxBlackLevel,
            agxLogMin, agxLogMax,
            agxLogMidgray, agxDisplayMidgray,
            agxContrast, agxToe, agxShoulder,
            agxVibrance
        )
    }

    private fun renderBayerFrame(viewW: Int, viewH: Int) {
        val synthetic = syntheticTestEnabled
        if (synthetic && syntheticSensor == null) {
            syntheticSensor = SyntheticBayerTest.buildSensor()
            syntheticBuffer = SyntheticBayerTest.buildBayerBuffer(syntheticSensor!!)
            CrashLogger.log(TAG, "synthetic Bayer frame: ${SyntheticBayerTest.SENSOR_W}x${SyntheticBayerTest.SENSOR_H} built")
        }
        val buffer = if (synthetic) syntheticBuffer else bayerBuffer
        val effW = if (synthetic) SyntheticBayerTest.SENSOR_W else bayerWidth
        val effH = if (synthetic) SyntheticBayerTest.SENSOR_H else bayerHeight
        val effStride = if (synthetic) SyntheticBayerTest.SENSOR_W else bayerStridePixels
        val effBlack = if (synthetic) SyntheticBayerTest.BLACK else bayerBlackLevelPattern
        val effColorMap = if (synthetic) SyntheticBayerTest.COLOR_MAP else bayerColorMap
        val effBitDepth = if (synthetic) SyntheticBayerTest.BIT_DEPTH else bayerBitDepth
        val effWhite = if (synthetic) SyntheticBayerTest.WHITE_LEVEL.toFloat() else agxWhiteLevel
        val effBlackLevel = if (synthetic) SyntheticBayerTest.BLACK_LEVEL.toFloat() else agxBlackLevel

        if (buffer == null || effW <= 0 || effH <= 0) {
            clearAndSwap(viewW, viewH)
            return
        }

        if (synthetic) {
            bayerShader.setCropRegion(0f, 0f, effW.toFloat(), effH.toFloat())
        } else {
            applyBayerCrop()
        }

        bayerRenderCount++
        if (bayerRenderCount <= 8 || bayerRenderCount % 120 == 0) {
            val pend = GLES20.glGetError()
            CrashLogger.log(
                TAG, "PENDING_GL_ERROR at frame start #$bayerRenderCount: " +
                    if (pend == GLES20.GL_NO_ERROR) "none" else "0x${Integer.toHexString(pend)}"
            )
        }

        ensureDemosaicFbo(fboWidth, fboHeight)
        // Preview S1/S3 denoising now runs per-sample inside the demosaic
        // (same-colour neighbourhood filter, no fixed grid).  The RAW sparse
        // grid chain (pack/DPC/S3/S2 sigma) is kept ONLY as the synthetic
        // oracle's reference path; on the live device it is skipped entirely,
        // which also removes the non-integer grid->sensor collapse that the
        // old output-res grid caused at this sensor/grid ratio.
        val cellGridW = demosaicFboWidth
        val cellGridH = demosaicFboHeight

        val dpEnabled = dpcStrength > 0f && dpcShader.isReady()
        val rawDenoiseActive = rawNrStrength > 0f && rawDenoiseShader.isReady()
        val outDenoiseActive = outNrStrength > 0f && outNrShader.isReady()
        val needSparseGrid = (dpEnabled || rawDenoiseActive || (synthetic && outDenoiseActive)) && dpcShader.isReady()
        val previewTransform = computePreviewTransform(effW, effH)
        val whiteRange = (effWhite - effBlackLevel).coerceAtLeast(1f)
        val crop = bayerShader.currentCropRegion()
        val iso = isoForDenoise.coerceAtLeast(1)
        val isoModelA = com.agx.camera.camera.NoiseModel.photonCoeff(iso)
        val isoModelB = com.agx.camera.camera.NoiseModel.readNoiseVariance(iso)

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, demosaicFboId)
        GLES20.glViewport(0, 0, demosaicFboWidth, demosaicFboHeight)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        bayerShader.uploadBayer(buffer!!.duplicate(), effW, effH, effStride)
        logGlError("after uploadBayer", bayerRenderCount)

        // Hoisted slider-strength values shared by the live pipeline and the
        // synthetic CPU-reference oracle.
        val s1 = dpcStrength.coerceIn(0f, 1f)
        val m1v = 0.1f + 0.3f * s1
        val m2v = 4f + 6f * s1
        val thetav = 2f + 4f * s1
        val s3 = rawNrStrength.coerceIn(0f, 1f)
        val keepRawAlpha = 1f - (1f - com.agx.camera.camera.NoiseModel.alphaRaw(iso)) * s3
        val s3Eps = 0.02f * whiteRange * whiteRange * (0.5f + iso / 6400f)

        val gridW = cellGridW
        val gridH = cellGridH

        // Zoom factor = sensor crop cells per output texel (per axis, worst).
        val previewZoomK = maxOf(
            crop[2] / demosaicFboWidth.toFloat(),
            crop[3] / demosaicFboHeight.toFloat()
        )

        // ---- RAW-domain stages (S1 DPC + S3 green-guided GF) producing the
        // float sparse Bayer grid that the demosaic (Stage 4) consumes.  On the
        // live device this replaces the fused pack's cross-phase approximation
        // (which violated the spec and caused achromatic washout + mosaic
        // pattern); the pack is kept only as a fallback when the DPC/GF shaders
        // are not ready.
        //
        // The DPC+GF chain is spec-correct only when the sparse grid ≈ sensor
        // resolution (k <= 2, i.e. roughly 3x+ zoom-in at preview sizes).  At
        // wider zoom each grid texel reverse-maps to several sensor pixels, so
        // the GF 5x5 window covers a huge sensor footprint and the DPC operates
        // on sub-sampled texels — both add noise and let hot pixels fall between
        // samples.  There the demosaic's inline sampleSameColorNR (which reads
        // raw sensor values at any zoom and blends within a single CFA channel)
        // is the right path; demosaicDenoisedId stays 0 so drawDemosaic applies
        // dpStrength/rawNrStrength inline.
        var demosaicDenoisedId = 0
        if (needSparseGrid && previewZoomK <= 2.0f) {
            demosaicDenoisedId = runDpcAndGfChain(
                previewTransform, crop, gridW, gridH,
                effW.toFloat(), effH.toFloat(),
                bayerShader.bayerTextureHandle(),
                effBlack, effBitDepth,
                dpEnabled, m1v, m2v, thetav, isoModelA, isoModelB, s1,
                rawDenoiseActive, keepRawAlpha, s3Eps,
                bayerRenderCount
            )
            logGlError("after runDpcAndGfChain", bayerRenderCount)
        } else if (!needSparseGrid && (s1 > 0f || s3 > 0f) && bayerShader.isReady() && previewZoomK <= 2.0f) {
            // Pack fallback when DPC/GF shaders aren't ready.
            ensureS3PackBuffers(cellGridW, cellGridH)
            if (s3PackFboId != 0) {
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, s3PackFboId)
                GLES20.glViewport(0, 0, s3PackWidth, s3PackHeight)
                bayerShader.drawS3Pack(
                    previewTransform, effBlack, effBitDepth, effWhite, effBlackLevel, s1, s3, isoModelA, isoModelB,
                    cellGridW.toFloat(), cellGridH.toFloat(),
                    floatArrayOf(wbGainR, wbGainG, wbGainB)
                )
                demosaicDenoisedId = s3PackTexId
                logGlError("after drawS3Pack", bayerRenderCount)
            }
        }

        // Stage 2: σ̂ re-estimation (MAD) on the sparse residual grid —
        // synthetic oracle only; on the live device S5 uses the ISO model
        // fallback (useIsoSigma = true).
        if (synthetic && outDenoiseActive) {
            ensureSigmaBuffer(gridW, gridH)
            bindTarget(sigmaFboId, gridW, gridH)
            sigmaHatShader.draw(
                transformMatrix = previewTransform,
                cropOriginX = crop[0], cropOriginY = crop[1],
                cropSizeX = crop[2], cropSizeY = crop[3],
                viewWidth = gridW.toFloat(), viewHeight = gridH.toFloat(),
                sensorWidth = effW.toFloat(), sensorHeight = effH.toFloat(),
                sparseTex = if (rawDenoiseActive) denoisedTexId
                    else if (dpEnabled) dpcWorkBTexId else dpcWorkATexId,
                blackLevelPattern = effBlack,
                isoModelA = isoModelA, isoModelB = isoModelB
            )
            logGlError("after stage2 sigma", bayerRenderCount)
        }

        // CPU cross-check against the synthetic oracle whenever the RAW-domain
        // configuration changes (early frames included), so the comparison fires
        // both at startup and when the user moves the S1/S3/S5 sliders mid-run.
        val sig = "dp=$dpEnabled:$m1v:$m2v:$thetav:$s1,s3=$rawDenoiseActive:$keepRawAlpha:$s3Eps,o=$outDenoiseActive"
        if (needSparseGrid && previewZoomK <= 2.0f &&
            (bayerRenderCount <= 3 || (synthetic && sig != lastProbeSig))) {
            lastProbeSig = sig
            if (synthetic) {
                runSyntheticChecks(
                    SyntheticBayerTest.SynthParams(
                        dpEnabled = dpEnabled,
                        m1 = m1v, m2 = m2v, theta = thetav,
                        isoA = isoModelA, isoB = isoModelB,
                        corrStrength = s1,
                        s3Active = rawDenoiseActive, alpha = keepRawAlpha, eps = s3Eps,
                        viewW = gridW, viewH = gridH
                    ),
                    dpcAvgFboId, dpcFlagFboId, dpcWorkAFboId, dpcWorkBFboId,
                    if (rawDenoiseActive) denoiseFboId else if (dpEnabled) dpcWorkBFboId else dpcWorkAFboId,
                    if (synthetic && outDenoiseActive) sigmaFboId else 0, gridW, gridH
                )
            }
        }

        // The RAW-domain passes left an intermediate FBO bound; put back the
        // demosaic target FBO + viewport so drawDemosaic renders correctly.
        bindTarget(demosaicFboId, demosaicFboWidth, demosaicFboHeight)

        // The S1/S3 inline filter must never make the output noisier: two
        // rules bind the demosaic configuration, probed across k=1.5..8 (the
        // S3PreviewGLReproTest monotonicity walk).
        //   (a) baseline bound: the effective averaging must never drop below
        //       the zero-slider 4x4 — any step-down needs the blend to already
        //       be strong enough to keep every band's sigma AT OR BELOW the
        //       previous (weaker) slider value (per-band sigma gate).
        //   (b) with that bound, box3 is strictly dominated: 4x4 + the 4-tap
        //       ring is cheaper AND smoother than 3x3 + the 12-tap ring, so
        //       there is no monotone reason to ever emit box3.  The strong-s3
        //       zone (>= 0.7) drops to 2x2 + the full 12-tap ring, whose
        //       wider trim is what actually pulls sigma down hard up there —
        //       the transition sits late enough that every band (incl. the
        //       G floor, which 2x2 raises) lands below the 4x4 band's level.
        // Baseline (s3=0, or sliders off) stays at 4, unchanged; the pack
        // regime (k<=2) uses its own shader, boxAA=4 throughout.
        val previewBoxAA = when {
            previewZoomK <= 2.0f -> 4
            s3 >= 0.7f -> 2
            else -> 4
        }
        // In the box-AA=4 band the demosaic box supplies the steady averaging, so
        // the α-trim ring only needs the 4 step-2 axis neighbours
        // (u_nr_radius=2).  The strong-s3 2x2 box needs the full 12-tap ring
        // to actually drag sigma down.  Capture never runs this path.
        val previewNrRadius = if (previewBoxAA == 4) 2 else 4

        val sliderSig = "%.4f:%.4f:%d:%d".format(s1, s3, previewBoxAA, previewNrRadius)
        if ((!synthetic && bayerRenderCount <= 3) || (sliderSig != lastDenoiseSliderSig)) {
            lastDenoiseSliderSig = sliderSig
            android.util.Log.i(TAG, "preview sliders: s1=$s1 s3=$s3 boxAA=$previewBoxAA nrRadius=$previewNrRadius")
        }

        bayerShader.drawDemosaic(
            cellGridW, cellGridH, // u_outputResolution: maps sensor coords → denoised grid texels
            previewTransform,
            effBlack,
            effColorMap,
            effBitDepth,
            effWhite, effBlackLevel,
            boxAA = previewBoxAA,
            nrRadius = previewNrRadius,
            wbGains = floatArrayOf(wbGainR, wbGainG, wbGainB),
            colorMat = ccMatrix,
            denoisedTextureId = demosaicDenoisedId,
            denoiseActive = demosaicDenoisedId != 0,
            scaleFactor = previewZoomK,
            lumaCoeffs = nativeLumaCoeffs,
            clipAttenFactor = clipAttenFactor,
            dpStrength = if (demosaicDenoisedId != 0) 0f else s1,
            rawNrStrength = if (demosaicDenoisedId != 0) 0f else s3,
            isoModelA = isoModelA,
            isoModelB = isoModelB
        )
        logGlError("after drawDemosaic", bayerRenderCount)

        var ispInputTex = demosaicFboTextureId
        if (outDenoiseActive) {
            ispInputTex = runStage5(
                demosaicFboWidth, demosaicFboHeight, demosaicFboTextureId,
                sigmaTexId,
                whiteRange, gridW, gridH,
                useIsoSigma = sigmaFboId == 0,
                isoModelA = isoModelA, isoModelB = isoModelB
            )
            logGlError("after stage5 outDenoise", bayerRenderCount)
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId)
        GLES20.glViewport(0, 0, fboWidth, fboHeight)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        nrShader.draw(
            ispInputTex,
            exposureEv,
            agxSceneLinearTo709,
            agxInsetMat,
            agxOutsetMat,
            agxToRec2020,
            agxWhiteLevel, agxBlackLevel,
            agxLogMin, agxLogMax,
            agxLogMidgray, agxDisplayMidgray,
            agxContrast, agxToe, agxShoulder,
            agxVibrance
        )
        logGlError("after nrShader.draw", bayerRenderCount)
    }

    private fun applyBayerCrop() {
        if (bayerWidth <= 0 || bayerHeight <= 0) return
        val zoom = zoomController.zoomFactor.coerceIn(1.0f, zoomController.maxZoom)
        if (zoom <= 1.0f) {
            bayerShader.setCropRegion(0f, 0f, bayerWidth.toFloat(), bayerHeight.toFloat())
            return
        }
        val cropW = (bayerWidth / zoom).toInt().coerceAtLeast(2)
        val cropH = (bayerHeight / zoom).toInt().coerceAtLeast(2)
        val centerX = (zoomController.zoomCenterX * bayerWidth).toInt()
        val centerY = (zoomController.zoomCenterY * bayerHeight).toInt()
        val left = (centerX - cropW / 2).coerceIn(0, bayerWidth - cropW)
        val top = (centerY - cropH / 2).coerceIn(0, bayerHeight - cropH)
        bayerShader.setCropRegion(left.toFloat(), top.toFloat(), cropW.toFloat(), cropH.toFloat())
        val tag = "zoom=$zoom origin=${left},${top} size=${cropW}x$cropH"
        if (tag != lastBayerCropLog) {
            lastBayerCropLog = tag
            CrashLogger.log(TAG, "bayerCrop: $tag")
        }
    }

    private fun logGlError(where: String, frame: Int) {
        val err = GLES20.glGetError()
        if (err != GLES20.GL_NO_ERROR && (frame <= 3 || frame % 300 == 0)) {
            CrashLogger.log(TAG, "glError $where: 0x${Integer.toHexString(err)}")
        }
    }

    private fun bindTarget(fboId: Int, width: Int, height: Int) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId)
        GLES20.glViewport(0, 0, width, height)
    }

    /** Stage 5 — output-domain SWGF over the demosaiced RGB (plan §3 Stage 5).
     *  Two iterations with a β=0.3 noise return and κ×1.4 on round 2. The
     *  returned texture id feeds the ISP (nrShader/draw) instead of the raw
     *  demosaic output.  The S5 geometry lives in output-pixel space, so the
     *  capture path passes winScale = captureRes/previewRes > 1 to widen the
     *  window centres + dense chroma-mean box to the same relative image
     *  footprint, and epsBoost to compensate for its missing boxAA low-pass
     *  (preview residual σ̂²/32 vs 1:1 capture ~16x larger). */
    private fun runStage5(
        w: Int, h: Int, inputTex: Int, sigmaTex: Int, whiteRange: Float, gridW: Int, gridH: Int,
        useIsoSigma: Boolean = false, isoModelA: Float = 0f, isoModelB: Float = 0f,
        winScale: Float = 1f, epsBoost: Float = 1f
    ): Int {
        if (!outNrShader.isReady()) return inputTex
        ensureOutNrBuffers(w, h)
        // sigmaTex holds σ̂ on the 2×2-cell grid (cellGridW/H), which can differ
        // from this pass's output res; drawMain scales the fetch accordingly.
        val sigmaW = gridW.toFloat()
        val sigmaH = gridH.toFloat()
        val s = outNrStrength.coerceIn(0f, 1f)
        // Decoupled luma/chroma response.  eps is fixed at design maximum;
        // the slider controls a linear blend toward the filtered result so
        // strength 0..100 maps to 0%..100% denoise (0% = identity,
        // 100% = full SWGF at epsY≈1.4, epsC≈64).
        val lumaEpsScale = 1.4f
        val chromaEpsScale = 64.0f
        val beta = 0.3f
        // Stage-4 sparse-demosaic residual variance (plan §3: σ_dm ≈ 3~4 DN).
        val sigmaDm2 = 10f
        // ε lives in the pixel domain (0..1); σ̂² and σ_dm² are in raw-DN².
        val inverseRange2 = 1f / (whiteRange * whiteRange)
        // S5D f#1200 crash evidence: σ̂² is the sparse-grid DN² noise floor
        // (readNoiseVariance(3200)=111.5 → sig2≈113), but the filter actually
        // sees the S3 + 4x4-boxAA-denoised demosaic residual (measured var
        // ≈2-6e-6 → σ≈1.5-2.5 DN). Using the DN² floor verbatim makes
        // ε≈30-100× the true residual → aY≈0.02 → output collapses to the
        // window mean (brighten + pixel-art). Scale ε to the real residual.
        val sigmaScale = 1f / 32f
        // EV PP multiplies the linear scene by exp2(EV) AFTER Stage 5 in the
        // formed picture read, so the noise the viewer sees grows with EV exactly as
        // if the sensor ISO had been raised.  Fold the EV gain (σ² ∝ gain²)
        // into the ISO model, anchored at the +1.5 EV default so the baseline
        // the pipeline was tuned around stays unchanged.  Clamp to keep the
        // filter out of the ε≫var collapse regime on either extreme.
        val evGain2 = Math.pow(2.0, 2.0 * (exposureEv - 1.5)).toFloat().coerceIn(1f / 8f, 16f)
        val s5IsoA = isoModelA * evGain2
        val s5IsoB = isoModelB * evGain2
        // Single SWGF iteration; round 2 (κ×1.4 + β noise return) is
        // bypassed here pending further tuning of the double-pass behavior.
        val iterations = 1

        // Round 1 (κ1): demosaic output → outNr1.
        bindTarget(statsHFboId, w, h)
        outNrShader.drawStatsH(inputTex, inputTex, beta, S5_LUMA_WEIGHTS, winScale)
        logGlError("stage5 statsH1", bayerRenderCount)
        bindTarget(statsVFboId, w, h)
        outNrShader.drawStatsV(statsHTexId, winScale)
        logGlError("stage5 statsV1", bayerRenderCount)
        bindTarget(outNr1FboId, w, h)
        outNrShader.drawMain(
            inputTex, inputTex, statsVTexId, sigmaTex,
            sigmaW, sigmaH, w.toFloat(), h.toFloat(),
            beta, lumaEpsScale, chromaEpsScale, sigmaDm2,
            inverseRange2, sigmaScale,
            useIsoSigma, s5IsoA, s5IsoB,
            winScale, epsBoost, strength = s
        )
        logGlError("stage5 main1", bayerRenderCount)

        if (iterations < 2) return outNr1TexId

        // Round 2 (κ×1.4): outNr1 → outNr2, β-return against the original.
        bindTarget(statsHFboId, w, h)
        outNrShader.drawStatsH(outNr1TexId, inputTex, beta, S5_LUMA_WEIGHTS, winScale)
        logGlError("stage5 statsH2", bayerRenderCount)
        bindTarget(statsVFboId, w, h)
        outNrShader.drawStatsV(statsHTexId, winScale)
        logGlError("stage5 statsV2", bayerRenderCount)
        bindTarget(outNr2FboId, w, h)
        outNrShader.drawMain(
            outNr1TexId, inputTex, statsVTexId, sigmaTex,
            sigmaW, sigmaH, w.toFloat(), h.toFloat(),
            beta, 1.96f * lumaEpsScale, 1.96f * chromaEpsScale, sigmaDm2,
            inverseRange2, sigmaScale,
            useIsoSigma, s5IsoA, s5IsoB,
            winScale, epsBoost, strength = s
        )
        logGlError("stage5 main2", bayerRenderCount)

        return outNr2TexId
    }

    // GL writes GL_FLOAT pixels in the platform's native (little-endian) byte
    // order; a default big-endian ByteBuffer would byte-swap every component
    // (0.5f reads as 8.83e-44). All readback helpers MUST use native order.
    // Readbacks use GL_FLOAT regardless of the attachment format: RGBA32F
    // cannot be read as GL_UNSIGNED_BYTE (GL_INVALID_OPERATION, and the buffer
    // stays zeroed → fake black content), while GL_FLOAT is valid for both
    // RGBA8 and RGBA32F attachments.
    private fun readBuffer(capacity: Int): ByteBuffer =
        ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder())

    // Reads the float RGBA value of one texel of an RGBA32F FBO attachment.
    private fun readTexelF(fboId: Int, x: Int, y: Int): FloatArray {
        return try {
            val buf = readBuffer(16)
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId)
            GLES20.glReadPixels(x, y, 1, 1, GLES20.GL_RGBA, GLES20.GL_FLOAT, buf)
            val rb = buf.asFloatBuffer()
            floatArrayOf(rb.get(0), rb.get(1), rb.get(2), rb.get(3))
        } catch (t: Throwable) {
            floatArrayOf(Float.NaN, Float.NaN, Float.NaN, Float.NaN)
        }
    }

    /** Synthetic-frame oracle: compares GPU stage outputs against the CPU
     *  reference (SyntheticBayerTest) at per-region probe texels. Logs one line
     *  per probe with the max channel delta per stage; a large avg/sparse delta
     *  points at indexing/phase bugs, a large sigma delta at the MAD/unit path. */
    private fun runSyntheticChecks(
        params: SyntheticBayerTest.SynthParams,
        avgFbo: Int, flagFbo: Int, workAFbo: Int, workBFbo: Int,
        sparseFbo: Int, sigmaFbo: Int, gridW: Int, gridH: Int
    ) {
        val sensor = syntheticSensor ?: return
        if (avgFbo == 0) return
        val refs = try {
            SyntheticBayerTest.CpuRefs(sensor, params).compute()
        } catch (t: Throwable) {
            CrashLogger.log(TAG, "synth ref failed: ${t.message}")
            return
        }
        val tol = 2.5f
        var worst = 0f
        var worstStage = "none"
        for (r in refs) {
            val avgGpu = readTexelF(avgFbo, r.tx, r.ty)
            val flagGpu = readTexelF(flagFbo, r.tx, r.ty)
            val gridGpu = readTexelF(sparseFbo, r.tx, r.ty)
            val avgD = maxDelta(avgGpu, r.avg)
            val flagD = maxDelta(flagGpu, r.flags)
            val gridD = maxDelta(gridGpu, r.grid)
            val gridRef = if (params.s3Active) r.s3Grid else r.grid
            val gridUsedD = maxDelta(gridGpu, gridRef)
            val sigStr = if (params.s3Active) "s3-active" else "raw/dpc"
            val flagsStr = formatFlags(r.flags)
            var line = "SYNTH (sx,sy)=(${r.sx},${r.sy}) texel=(${r.tx},${r.ty}) flags=$flagsStr " +
                "avgD=$avgD flagD=$flagD gridD=$gridD (stage=$sigStr)"
            if (params.s3Active) {
                // gridD (vs r.grid) shows how far S3 moved the cell; the pass
                // criterion is gridUsedD = |gpu s3 grid − cpu s3 grid|.
                line += " used=$gridUsedD"
            }
            if (sigmaFbo != 0) {
                val sigGpu = readTexelF(sigmaFbo, r.tx, r.ty)
                val sigD = maxDelta(sigGpu, r.sigma)
                line += " sigmaD=$sigD"
                if (sigD > worst) { worst = sigD; worstStage = "sigma" }
            }
            if (gridUsedD > worst) { worst = gridUsedD; worstStage = "grid" }
            if (avgD > worst) { worst = avgD; worstStage = "avg" }
            if (flagD > worst) { worst = flagD; worstStage = "flag" }
            val status = if (maxOf(avgD, gridUsedD, flagD) <= tol) "OK" else "DIFF"
            CrashLogger.log(TAG, "$status $line")
        }
        CrashLogger.log(TAG, "SYNTH worst delta=$worst at $worstStage (tol=$tol)")
    }

    private fun maxDelta(a: FloatArray, b: FloatArray): Float {
        var m = 0f
        for (k in 0 until minOf(a.size, b.size)) {
            val d = abs(a[k] - b[k])
            if (d > m) m = d
        }
        return m
    }

    private fun formatFlags(f: FloatArray): String =
        f.joinToString("") { v ->
            if (v > 0.5f) "H" else if (v < -0.5f) "C" else "."
        }

    private fun clearAndSwap(viewW: Int, viewH: Int) {
        if (viewW > 0 && viewH > 0) {
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
            GLES20.glViewport(0, 0, viewW, viewH)
            GLES20.glClearColor(0f, 0f, 0f, 1f)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
            val swapResult = EGL14.eglSwapBuffers(eglDisplay, eglSurface)
            if (!swapResult) {
                val error = EGL14.eglGetError()
                if (error != EGL14.EGL_SUCCESS) {
                    CrashLogger.log(TAG, "eglSwapBuffers failed: error=0x${Integer.toHexString(error)}, recreating surface")
                    createEglSurface()
                }
            }
        }
    }

    private fun renderLoop() {
        CrashLogger.log(TAG, "renderLoop: starting")
        initEgl()

        while (running) {
            renderLock.withLock {
                while (!hasNewFrame && running) {
                    frameCondition.await(100, java.util.concurrent.TimeUnit.MILLISECONDS)
                }
            }
            if (!running) break
            try {
                hasNewFrame = false

            if (degradedManager?.shouldThrottleFrame() == true) continue

            if (eglDisplay == EGL14.EGL_NO_DISPLAY || eglSurface == EGL14.EGL_NO_SURFACE) {
                CrashLogger.log(TAG, "renderLoop: eglSurface not ready, retrying createEglSurface")
                createEglSurface()
                if (eglSurface == EGL14.EGL_NO_SURFACE) {
                    Thread.sleep(50)
                    continue
                }
            }

            if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
                Log.w(TAG, "eglMakeCurrent failed")
                CrashLogger.log(TAG, "renderLoop: eglMakeCurrent failed")
                continue
            }

            if (!glInitialized) {
                initGlResources()
            }

            // A lens shading map arrived after GL init: upload it on the render
            // thread (textures can only be mutated with the context current).
            if (lensShadingUploadPending) {
                lensShadingUploadPending = false
                val data = bayerLensShadingData
                if (data != null) {
                    bayerShader.uploadLensShadingMap(data, bayerLensShadingWidth, bayerLensShadingHeight)
                    CrashLogger.log(TAG, "lens shading map uploaded ${bayerLensShadingWidth}x${bayerLensShadingHeight}")
                } else {
                    bayerShader.uploadIdentityLensShading()
                    CrashLogger.log(TAG, "lens shading reset to identity")
                }
            }

            val viewW = textureView.width
            val viewH = textureView.height
            if (viewW <= 0 || viewH <= 0) continue

            val captureReq = pendingCaptureFrame?.also { pendingCaptureFrame = null }
            val rawCaptureReq = pendingRawCaptureFrame?.also { pendingRawCaptureFrame = null }
            val frameStartNs = if (captureReq == null && rawCaptureReq == null) System.nanoTime() else 0L
            if (captureReq != null) {
                ensureCaptureFbo()
                if (captureFboId != 0) {
                    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, captureFboId)
                    GLES20.glViewport(0, 0, captureFboWidth, captureFboHeight)
                    GLES20.glClearColor(0f, 0f, 0f, 1f)
                    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

                    yuvShader.uploadY(captureReq.y.duplicate(), captureReq.w, captureReq.h)
                    yuvShader.uploadU(captureReq.u.duplicate(), captureReq.w / 2, captureReq.h / 2)
                    yuvShader.uploadV(captureReq.v.duplicate(), captureReq.w / 2, captureReq.h / 2)

                    val captureMatrix = FloatArray(16).also { android.opengl.Matrix.setIdentityM(it, 0) }
                    // Capture matrix: match preview's Y-flip (sensor -> OpenGL), but NO X-flip for front camera
                    // (saved JPG must NOT be mirrored per spec). Rotation handled via EXIF.
                    // Preview transform: rear=scale(1,-1), front=scale(-1,-1)
                    // Capture transform: both use scale(1,-1) = Y-flip only
                    android.opengl.Matrix.translateM(captureMatrix, 0, 0.5f, 0.5f, 0f)
                    android.opengl.Matrix.scaleM(captureMatrix, 0, 1f, -1f, 1f)
                    android.opengl.Matrix.translateM(captureMatrix, 0, -0.5f, -0.5f, 0f)
                    yuvShader.draw(
                        captureFboWidth, captureFboHeight,
                        captureMatrix,
                        exposureEv,
                        captureReq.agxSceneLinearTo709, captureReq.agxInsetMat, captureReq.agxOutsetMat, captureReq.agxToRec2020,
                        captureReq.agxWhiteLevel, captureReq.agxBlackLevel,
                        captureReq.agxLogMin, captureReq.agxLogMax,
                        captureReq.agxLogMidgray, captureReq.agxDisplayMidgray,
                        captureReq.agxContrast, captureReq.agxToe, captureReq.agxShoulder,
                        captureReq.agxVibrance
                    )

                    val readW: Int
                    val readH: Int
                    val readTexId: Int

                    if (captureReq.targetW < captureFboWidth && captureReq.targetH < captureFboHeight &&
                        captureReq.targetW > 0 && captureReq.targetH > 0) {
                        ensureDownscaleFbo(captureReq.targetW, captureReq.targetH)
                        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, downscaleFboId)
                        GLES20.glViewport(0, 0, downscaleFboWidth, downscaleFboHeight)
                        blitShader.draw(captureFboTextureId)
                        readW = downscaleFboWidth
                        readH = downscaleFboHeight
                        readTexId = downscaleFboTextureId
                    } else {
                        readW = captureFboWidth
                        readH = captureFboHeight
                        readTexId = captureFboTextureId
                    }

                    val bitmap = JpegEncoder.readFboToBitmapFlipped(readTexId, readW, readH)
                    captureReq.resultRef.set(bitmap)
                } else {
                    Log.e(TAG, "Capture FBO not available")
                }
                captureReq.latch.countDown()
            }

            if (rawCaptureReq != null) {
                val outW = rawCaptureReq.targetW.takeIf { it in 1..rawCaptureReq.rawW } ?: rawCaptureReq.rawW
                val outH = rawCaptureReq.targetH.takeIf { it in 1..rawCaptureReq.rawH } ?: rawCaptureReq.rawH

                ensureRawDemosaicFbo(outW, outH)
                ensureDownscaleFbo(outW, outH)
                if (rawDemosaicFboId != 0 && downscaleFboId != 0) {
                    applyBayerCrop()

                    val captureMatrix = buildRawCaptureMatrix(
                        rawCaptureReq.rawW, rawCaptureReq.rawH, outW, outH
                    )

                    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, rawDemosaicFboId)
                    GLES20.glViewport(0, 0, rawDemosaicFboWidth, rawDemosaicFboHeight)
                    GLES20.glClearColor(0f, 0f, 0f, 1f)
                    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

                    bayerShader.uploadBayer(
                        rawCaptureReq.buffer.duplicate(),
                        rawCaptureReq.rawW, rawCaptureReq.rawH, rawCaptureReq.stridePixels
                    )
                    logGlError("raw after uploadBayer", bayerRenderCount)
                    if (bayerNrStrength > 0f && spatialNrShader.isReady()) {
                        ensureCaptureDenoiseBuffers(rawDemosaicFboWidth, rawDemosaicFboHeight)
                    }

                    // Capture-side denoise: run the same Stage-1/3/5 slider
                    // pipeline as the live preview.
                    val iso = isoForDenoise.coerceAtLeast(1)
                    val captureIsoModelA = com.agx.camera.camera.NoiseModel.photonCoeff(iso)
                    val captureIsoModelB = com.agx.camera.camera.NoiseModel.readNoiseVariance(iso)
                    val captureWhiteRange = (rawCaptureReq.agxWhiteLevel - rawCaptureReq.agxBlackLevel).coerceAtLeast(1f)
                    val crop = bayerShader.currentCropRegion()

                    // Legacy placeholder spatial NR: only used when no Stage-1/3
                    // slider is active, so it can't double-process the still.
                    val legacyRawDenoiseActive =
                        bayerNrStrength > 0f &&
                            dpcStrength <= 0f && rawNrStrength <= 0f &&
                            spatialNrShader.isReady()
                    if (legacyRawDenoiseActive) {
                        ensureCaptureDenoiseBuffers(rawDemosaicFboWidth, rawDemosaicFboHeight)
                        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, captureDenoiseFboId)
                        GLES20.glViewport(0, 0, captureDenoiseFboWidth, captureDenoiseFboHeight)
                        spatialNrShader.draw(
                            transformMatrix = captureMatrix,
                            cropOriginX = crop[0],
                            cropOriginY = crop[1],
                            cropSizeX = crop[2],
                            cropSizeY = crop[3],
                            viewWidth = captureDenoiseFboWidth.toFloat(),
                            viewHeight = captureDenoiseFboHeight.toFloat(),
                            sensorWidth = rawCaptureReq.rawW.toFloat(),
                            sensorHeight = rawCaptureReq.rawH.toFloat(),
                            bayerTex = bayerShader.bayerTextureHandle(),
                            blackLevelPattern = bayerBlackLevelPattern,
                            bitDepth = bayerBitDepth,
                            nrStrength = bayerNrStrength.coerceIn(0f, 1f),
                        )
                        logGlError("raw after legacy spatialNr.draw", bayerRenderCount)
                    }

                    // Restore the demosaic target FBO + viewport (the denoise
                    // pass left the capture denoise FBO bound).
                    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, rawDemosaicFboId)
                    GLES20.glViewport(0, 0, rawDemosaicFboWidth, rawDemosaicFboHeight)

                    // Capture stills always use the demosaic's inline
                    // sampleSameColorNR: it reads raw sensor pixels directly at
                    // every sample, blends within one CFA channel, uses a softer
                    // hot/cold band and has no M2 isolation test — so it removes
                    // defect pixels at ANY capture resolution (Full/Half/Quarter).
                    // The spec-accurate DPC grid chain (θ=6 band, M2=10 isolation)
                    // deliberately leaves moderate hot pixels in a one-shot still,
                    // and the fused pack fallback would reintroduce boxedBlend's
                    // cross-phase achromatic drag.  The DPC+GF chain stays
                    // preview-only (zoomed-in, k <= 2).
                    val captureBoxAA = if (outW < rawCaptureReq.rawW || outH < rawCaptureReq.rawH) 4 else 0
                    val captureK = maxOf(
                        rawCaptureReq.rawW.toFloat() / rawDemosaicFboWidth.toFloat(),
                        rawCaptureReq.rawH.toFloat() / rawDemosaicFboHeight.toFloat()
                    )

                    bayerShader.drawDemosaic(
                        rawDemosaicFboWidth, rawDemosaicFboHeight,
                        captureMatrix,
                        bayerBlackLevelPattern,
                        bayerColorMap,
                        bayerBitDepth,
                        rawCaptureReq.agxWhiteLevel, rawCaptureReq.agxBlackLevel,
                        boxAA = captureBoxAA,
                        wbGains = floatArrayOf(wbGainR, wbGainG, wbGainB),
                        colorMat = ccMatrix,
                        denoisedTextureId = if (legacyRawDenoiseActive) captureDenoisedTexId else 0,
                        denoiseActive = legacyRawDenoiseActive,
                        scaleFactor = captureK,
                        lumaCoeffs = nativeLumaCoeffs,
                        clipAttenFactor = clipAttenFactor,
                        dpStrength = dpcStrength.coerceIn(0f, 1f),
                        rawNrStrength = rawNrStrength.coerceIn(0f, 1f),
                        isoModelA = captureIsoModelA,
                        isoModelB = captureIsoModelB
                    )
                    logGlError("raw after drawDemosaic", bayerRenderCount)

                    // Stage-5 output-domain SWGF at capture resolution, fed from
                    // the ISO noise model (the S2 grid stays preview-only).
                    // The S5 windows live in output-pixel space, so at full-res
                    // capture (up to 3-4x the preview FBO) the fixed ±2 px geometry
                    // would cover a 3-4x smaller image footprint than on preview;
                    // scale the SWGF window centres and the dense chroma-mean
                    // half-width by the preview→capture resolution ratio (the
                    // stats' R follows the same rule) so the box hits the same
                    // relative image regions.  (The capture demosaic
                    // FBO is sized to the SETTINGS target resolution, which can be
                    // below the full sensor, so this ratio follows that too.)
                    // The ε residual also depends on the settings resolution: at 1:1
                    // target == sensor there is no boxAA, so the per-pixel residual
                    // is ~16x the preview's boxAA'd σ̂²/32 and ε is boosted by 16;
                    // at a reduced target the demosaic boxAA is ON (boxAA=4) and the
                    // residual is preview-like, so 16 would over-soften the means
                    // and push the output onto the sparse windows (blocky patches).
                    val previewS5W = demosaicFboWidth
                    val capWinScale = maxOf(
                        1f,
                        rawDemosaicFboWidth.toFloat() / maxOf(previewS5W, 1).toFloat()
                    )
                    val capEpsBoost = if (captureBoxAA > 0) 1f else 16f
                    var captureIspInputTex = rawDemosaicFboTextureId
                    if (outNrStrength > 0f && outNrShader.isReady()) {
                        captureIspInputTex = runStage5(
                            rawDemosaicFboWidth, rawDemosaicFboHeight, rawDemosaicFboTextureId,
                            0,
                            captureWhiteRange, rawDemosaicFboWidth, rawDemosaicFboHeight,
                            useIsoSigma = true,
                            isoModelA = captureIsoModelA, isoModelB = captureIsoModelB,
                            winScale = capWinScale, epsBoost = capEpsBoost
                        )
                        logGlError("raw after Stage-5 outDenoise", bayerRenderCount)
                    }

                    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, downscaleFboId)
                    GLES20.glViewport(0, 0, downscaleFboWidth, downscaleFboHeight)
                    GLES20.glClearColor(0f, 0f, 0f, 1f)
                    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

                    nrShader.draw(
                        captureIspInputTex,
                        exposureEv,
                        rawCaptureReq.agxSceneLinearTo709,
                        rawCaptureReq.agxInsetMat,
                        rawCaptureReq.agxOutsetMat,
                        rawCaptureReq.agxToRec2020,
                        rawCaptureReq.agxWhiteLevel, rawCaptureReq.agxBlackLevel,
                        rawCaptureReq.agxLogMin, rawCaptureReq.agxLogMax,
                        rawCaptureReq.agxLogMidgray, rawCaptureReq.agxDisplayMidgray,
                        rawCaptureReq.agxContrast, rawCaptureReq.agxToe, rawCaptureReq.agxShoulder,
                        rawCaptureReq.agxVibrance
                    )
                    logGlError("raw after nrShader.draw", bayerRenderCount)

                    val bitmap = JpegEncoder.readFboToBitmapFlipped(downscaleFboTextureId, outW, outH)
                    rawCaptureReq.resultRef.set(bitmap)
                } else {
                    Log.e(TAG, "RAW render FBO not available")
                }
                rawCaptureReq.latch.countDown()
            }

            if (useBayerPath || syntheticTestEnabled) {
                renderBayerFrame(viewW, viewH)
            } else {
                renderYuvFrame(viewW, viewH)
            }

            pendingFboReadback?.let { callback ->
                callback(fboTextureId, fboWidth, fboHeight)
                pendingFboReadback = null
            }

            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
            GLES20.glClearColor(0f, 0f, 0f, 1f)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

            val contentW = fboWidth
            val contentH = fboHeight
            // The FBO is already cropped to the preview aspect, so the content
            // aspect IS the FBO aspect. Letterbox (CENTER_INSIDE) against that.
            val contentAspect = contentW.toFloat() / contentH.toFloat()
            val viewAspect = viewW.toFloat() / viewH.toFloat()
            val vpW: Int
            val vpH: Int
            val vpX: Int
            val vpY: Int
            // FIT (CENTER_INSIDE) - show full frame with bars for selected aspect ratio
            if (contentAspect > viewAspect) {
                // Content wider than view → fit width, letterbox top/bottom
                vpW = viewW
                vpH = (viewW / contentAspect).toInt()
                vpX = 0
                vpY = (viewH - vpH) / 2
            } else {
                // Content taller than view → fit height, pillarbox left/right
                vpH = viewH
                vpW = (viewH * contentAspect).toInt()
                vpX = (viewW - vpW) / 2
                vpY = 0
            }
            GLES20.glViewport(vpX, vpY, vpW, vpH)

            if (focusPeakEnabled) {
                val zoom = zoomController.zoomFactor.coerceAtLeast(1f)
                val srcW = if (useBayerPath) bayerWidth else yuvWidth
                val srcH = if (useBayerPath) bayerHeight else yuvHeight
                val K_SOURCE_ANCHOR_PX = 2f
                val sharpUvX = if (srcW > 0) K_SOURCE_ANCHOR_PX * zoom / srcW else 0f
                val sharpUvY = if (srcH > 0) K_SOURCE_ANCHOR_PX * zoom / srcH else 0f
                focusPeakShader.draw(
                    fboTextureId,
                    vpW, vpH,
                    focusPeakRadius,
                    sharpUvX, sharpUvY,
                    focusPeakThreshold,
                    focusPeakStrength
                )
            } else {
                blitShader.draw(fboTextureId)
            }

            val swapResult = EGL14.eglSwapBuffers(eglDisplay, eglSurface)
            if (!swapResult) {
                val error = EGL14.eglGetError()
                if (error != EGL14.EGL_SUCCESS) {
                    CrashLogger.log(TAG, "eglSwapBuffers failed: $error, recreating surface")
                    createEglSurface()
                }
            }

            if (captureReq == null && rawCaptureReq == null && frameStartNs > 0) {
                val frameTimeMs = (System.nanoTime() - frameStartNs) / 1_000_000L
                onFrameRendered?.invoke(frameTimeMs)

                degradedManager?.let { dm ->
                    dm.reportFrameTime(frameTimeMs)
                    frameCount++
                    if (frameCount % 30 == 0) {
                        val prevBanner = dm.degradationBanner
                        val prevMode = dm.activeMode
                        dm.evaluateMode(bayerWidth, bayerHeight, fboWidth, fboHeight)
                        if (dm.activeMode != prevMode) {
                            onDegradedModeChanged?.invoke(dm.degradationBanner)
                        }
                    }
                }
            }

            if (!firstFrameReported) {
                firstFrameReported = true
                CrashLogger.log(TAG, "renderLoop: first frame rendered")
                onFirstFrameRendered?.invoke()
            }
            } catch (t: Throwable) {
                CrashLogger.log(TAG, "renderLoop: frame error: ${t.javaClass.simpleName}: ${t.message}")
                CrashLogger.logException(TAG, t)
            }
            }

        yuvShader.destroy()
        bayerShader.destroy()
        nrShader.destroy()
        spatialNrShader.destroy()
        blitShader.destroy()
        focusPeakShader.destroy()
        dpcShader.destroy()
        sigmaHatShader.destroy()
        rawDenoiseShader.destroy()
        outNrShader.destroy()
    }

    fun initEgl() {
        CrashLogger.log(TAG, "initEgl: start")
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        val version = IntArray(2)
        EGL14.eglInitialize(eglDisplay, version, 0, version, 1)

        val configAttribs = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL_RECORDABLE_ANDROID, EGL14.EGL_TRUE,
            EGL14.EGL_NONE
        )

        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)
        eglConfig = configs[0]

        if (eglConfig == null) {
            Log.e(TAG, "No suitable EGL config found")
            return
        }

        val contextAttribs = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
            EGL14.EGL_NONE
        )
        eglContext = EGL14.eglCreateContext(
            eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0
        )

        EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, eglContext)
        EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)

        CrashLogger.log(TAG, "initEgl: context created eglDisplay=$eglDisplay eglContext=$eglContext")
        Log.d(TAG, "EGL initialized: display=$eglDisplay context=$eglContext")
    }

    @Volatile private var glInitialized = false

    private fun initGlResources() {
        createFbo(fboWidth, fboHeight)

        yuvShader.create()
        bayerShader.create(bayerWidth, bayerHeight)
        nrShader.create()
        spatialNrShader.create()
        blitShader.create()
        focusPeakShader.create()
        dpcShader.create()
        sigmaHatShader.create()
        rawDenoiseShader.create()
        outNrShader.create()

        bayerLensShadingData?.let {
            bayerShader.uploadLensShadingMap(it, bayerLensShadingWidth, bayerLensShadingHeight)
        } ?: bayerShader.uploadIdentityLensShading()

        glInitialized = true
        CrashLogger.log(TAG, "initGlResources: done fbo=$fboId")
    }

    fun createEglSurface() {
        val st = textureView.surfaceTexture
        if (st == null) {
            CrashLogger.log(TAG, "createEglSurface: surfaceTexture is null, skipping")
            return
        }
        if (eglSurface != EGL14.EGL_NO_SURFACE) {
            EGL14.eglDestroySurface(eglDisplay, eglSurface)
        }
        val surfaceAttribs = intArrayOf(EGL14.EGL_NONE)
        eglSurface = EGL14.eglCreateWindowSurface(
            eglDisplay, eglConfig, st, surfaceAttribs, 0
        )
        CrashLogger.log(TAG, "createEglSurface: eglSurface=$eglSurface")
        Log.d(TAG, "EGL surface created from TextureView")
    }

    private fun createFbo(width: Int, height: Int) {
        fboWidth = width
        fboHeight = height

        val texBuf = IntArray(1)
        GLES20.glGenTextures(1, texBuf, 0)
        fboTextureId = texBuf[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTextureId)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
            width, height, 0,
            GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null
        )
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        val fboBuf = IntArray(1)
        GLES20.glGenFramebuffers(1, fboBuf, 0)
        fboId = fboBuf[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, fboTextureId, 0
        )

        val status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "FBO incomplete: $status")
            CrashLogger.log(TAG, "FBO incomplete: $status")
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        Log.d(TAG, "FBO created: ${width}x${height}")
    }

    private fun ensureCaptureFbo() {
        if (captureFboWidth <= 0 || captureFboHeight <= 0) return
        if (captureFboId != 0 && captureFboAllocatedWidth == captureFboWidth && captureFboAllocatedHeight == captureFboHeight) return

        if (captureFboId != 0) {
            GLES20.glDeleteFramebuffers(1, intArrayOf(captureFboId), 0)
            GLES20.glDeleteTextures(1, intArrayOf(captureFboTextureId), 0)
            captureFboId = 0
            captureFboTextureId = 0
        }

        val texBuf = IntArray(1)
        GLES20.glGenTextures(1, texBuf, 0)
        captureFboTextureId = texBuf[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, captureFboTextureId)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
            captureFboWidth, captureFboHeight, 0,
            GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null
        )
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        val fboBuf = IntArray(1)
        GLES20.glGenFramebuffers(1, fboBuf, 0)
        captureFboId = fboBuf[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, captureFboId)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, captureFboTextureId, 0
        )

        val status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "Capture FBO incomplete: $status")
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        captureFboAllocatedWidth = captureFboWidth
        captureFboAllocatedHeight = captureFboHeight
        Log.d(TAG, "Capture FBO created: ${captureFboWidth}x${captureFboHeight}")
    }

    private fun ensureDownscaleFbo(width: Int, height: Int) {
        if (downscaleFboId != 0 && downscaleFboWidth == width && downscaleFboHeight == height) return
        if (downscaleFboId != 0) {
            GLES20.glDeleteFramebuffers(1, intArrayOf(downscaleFboId), 0)
            GLES20.glDeleteTextures(1, intArrayOf(downscaleFboTextureId), 0)
            downscaleFboId = 0
            downscaleFboTextureId = 0
        }

        downscaleFboWidth = width
        downscaleFboHeight = height

        val texBuf = IntArray(1)
        GLES20.glGenTextures(1, texBuf, 0)
        downscaleFboTextureId = texBuf[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, downscaleFboTextureId)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
            width, height, 0,
            GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null
        )
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        val fboBuf = IntArray(1)
        GLES20.glGenFramebuffers(1, fboBuf, 0)
        downscaleFboId = fboBuf[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, downscaleFboId)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, downscaleFboTextureId, 0
        )

        val status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "Downscale FBO incomplete: $status")
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        Log.d(TAG, "Downscale FBO created: ${width}x${height}")
    }

    private fun ensureDemosaicFbo(width: Int, height: Int) {
        if (demosaicFboId != 0 && demosaicFboWidth == width && demosaicFboHeight == height) return
        if (demosaicFboId != 0) {
            GLES20.glDeleteFramebuffers(1, intArrayOf(demosaicFboId), 0)
            GLES20.glDeleteTextures(1, intArrayOf(demosaicFboTextureId), 0)
            demosaicFboId = 0
            demosaicFboTextureId = 0
        }

        demosaicFboWidth = width
        demosaicFboHeight = height

        val texBuf = IntArray(1)
        GLES20.glGenTextures(1, texBuf, 0)
        demosaicFboTextureId = texBuf[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, demosaicFboTextureId)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_RGBA32F,
            width, height, 0,
            GLES20.GL_RGBA, GLES20.GL_FLOAT, null
        )
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        val fboBuf = IntArray(1)
        GLES20.glGenFramebuffers(1, fboBuf, 0)
        demosaicFboId = fboBuf[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, demosaicFboId)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, demosaicFboTextureId, 0
        )

        val status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "Demosaic FBO incomplete: $status")
            CrashLogger.log(TAG, "Demosaic FBO incomplete: $status")
        }
        val attBuf = IntArray(1)
        GLES20.glGetFramebufferAttachmentParameteriv(
            GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, attBuf, 0
        )
        CrashLogger.log(
            TAG, "Demosaic FBO attach tex=$demosaicFboTextureId object=${attBuf[0]} " +
                "size=${width}x$height"
        )

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        CrashLogger.log(TAG, "Demosaic FBO created RGBA32F: ${width}x${height}")
    }

    private fun ensureRawDemosaicFbo(width: Int, height: Int) {
        if (rawDemosaicFboId != 0 && rawDemosaicFboWidth == width && rawDemosaicFboHeight == height) return
        if (rawDemosaicFboId != 0) {
            GLES20.glDeleteFramebuffers(1, intArrayOf(rawDemosaicFboId), 0)
            GLES20.glDeleteTextures(1, intArrayOf(rawDemosaicFboTextureId), 0)
            rawDemosaicFboId = 0
            rawDemosaicFboTextureId = 0
        }

        rawDemosaicFboWidth = width
        rawDemosaicFboHeight = height

        val texBuf = IntArray(1)
        GLES20.glGenTextures(1, texBuf, 0)
        rawDemosaicFboTextureId = texBuf[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, rawDemosaicFboTextureId)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_RGBA32F,
            width, height, 0,
            GLES20.GL_RGBA, GLES20.GL_FLOAT, null
        )
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        val fboBuf = IntArray(1)
        GLES20.glGenFramebuffers(1, fboBuf, 0)
        rawDemosaicFboId = fboBuf[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, rawDemosaicFboId)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, rawDemosaicFboTextureId, 0
        )

        val status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "Raw demosaic FBO incomplete: $status")
            CrashLogger.log(TAG, "Raw demosaic FBO incomplete: $status")
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        CrashLogger.log(TAG, "Raw demosaic FBO created RGBA32F: ${width}x${height}")
    }

    private fun allocRgba32fTexture(width: Int, height: Int): Int {
        val texBuf = IntArray(1)
        GLES20.glGenTextures(1, texBuf, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texBuf[0])
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_RGBA32F,
            width, height, 0,
            GLES20.GL_RGBA, GLES20.GL_FLOAT, null
        )
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        return texBuf[0]
    }

    private fun ensureDenoiseBuffers(width: Int, height: Int) {
        if (denoiseFboId != 0 && denoiseFboWidth == width && denoiseFboHeight == height) return
        if (denoiseFboId != 0) {
            GLES20.glDeleteFramebuffers(1, intArrayOf(denoiseFboId), 0)
            denoiseFboId = 0
            GLES20.glDeleteTextures(1, intArrayOf(denoisedTexId), 0)
            denoisedTexId = 0
        }

        denoiseFboWidth = width
        denoiseFboHeight = height

        denoisedTexId = allocRgba32fTexture(width, height)

        val fboBuf = IntArray(1)
        GLES20.glGenFramebuffers(1, fboBuf, 0)
        denoiseFboId = fboBuf[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, denoiseFboId)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, denoisedTexId, 0
        )

        val status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "Denoise FBO incomplete: $status")
            CrashLogger.log(TAG, "Denoise FBO incomplete: $status")
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        CrashLogger.log(TAG, "Denoise FBO created RGBA32F: ${width}x${height}")
    }

    private fun ensureS3PackBuffers(width: Int, height: Int) {
        if (s3PackFboId != 0 && s3PackWidth == width && s3PackHeight == height) return
        deleteFboTex(s3PackFboId, s3PackTexId)
        s3PackFboId = 0
        s3PackTexId = 0
        s3PackWidth = width
        s3PackHeight = height
        val (t, f) = allocRgba32fFbo(width, height)
        // The demosaic reads this mosaic with 4-tap texelFetch bilinear
        // (denoisedSampleRaw): GL_LINEAR is illegal on RGBA32F in ES 3.0, so
        // the shared allocator's NEAREST filter is left in place — texelFetch
        // ignores filtering state entirely anyway.
        s3PackTexId = t
        s3PackFboId = f
        CrashLogger.log(TAG, "S3-pack buffers: ${width}x${height} fbo=$f tex=$t")
    }

    // Capture/still spatial-NR: single spatial pass at the capture output
    // resolution; the denoised mosaic is RGBA16F to halve the memory
    // footprint of a full-res still.
    private fun allocRgba16fTexture(width: Int, height: Int): Int {
        val texBuf = IntArray(1)
        GLES20.glGenTextures(1, texBuf, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texBuf[0])
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F,
            width, height, 0,
            GLES20.GL_RGBA, GLES20.GL_FLOAT, null
        )
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        return texBuf[0]
    }

    private fun ensureCaptureDenoiseBuffers(width: Int, height: Int) {
        if (captureDenoiseFboId != 0 &&
            captureDenoiseFboWidth == width && captureDenoiseFboHeight == height) return
        if (captureDenoiseFboId != 0) {
            GLES20.glDeleteFramebuffers(1, intArrayOf(captureDenoiseFboId), 0)
            GLES20.glDeleteTextures(1, intArrayOf(captureDenoisedTexId), 0)
            captureDenoiseFboId = 0
            captureDenoisedTexId = 0
        }

        captureDenoiseFboWidth = width
        captureDenoiseFboHeight = height
        captureDenoisedTexId = allocRgba16fTexture(width, height)

        val fboBuf = IntArray(1)
        GLES20.glGenFramebuffers(1, fboBuf, 0)
        captureDenoiseFboId = fboBuf[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, captureDenoiseFboId)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, captureDenoisedTexId, 0
        )

        val status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            val msg = "Capture denoise FBO incomplete: $status"
            Log.e(TAG, msg)
            CrashLogger.log(TAG, msg)
            GLES20.glDeleteFramebuffers(1, intArrayOf(captureDenoiseFboId), 0)
            captureDenoiseFboId = 0
        } else {
            GLES20.glClearColor(0f, 0f, 0f, 1f)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
            CrashLogger.log(TAG, "Capture denoise FBO created RGBA16F: ${width}x${height}")
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
    }

    // Returns (textureId, fboId). Bind/cleanup handled by callers.
    private fun allocRgba32fFbo(width: Int, height: Int): Pair<Int, Int> {
        val tex = allocRgba32fTexture(width, height)
        val fboBuf = IntArray(1)
        GLES20.glGenFramebuffers(1, fboBuf, 0)
        val fbo = fboBuf[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, tex, 0
        )
        val status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            val msg = "RGBA32F FBO incomplete: $status"
            Log.e(TAG, msg)
            CrashLogger.log(TAG, msg)
        }
        return Pair(tex, fbo)
    }

    private fun deleteFboTex(fboId: Int, texId: Int) {
        if (fboId != 0) GLES20.glDeleteFramebuffers(1, intArrayOf(fboId), 0)
        if (texId != 0) GLES20.glDeleteTextures(1, intArrayOf(texId), 0)
    }

    private fun ensureFallbackFloatTexture() {
        if (fallbackFloatTexId != 0) return
        val texBuf = IntArray(1)
        GLES20.glGenTextures(1, texBuf, 0)
        fallbackFloatTexId = texBuf[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fallbackFloatTexId)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_RGBA32F,
            1, 1, 0,
            GLES20.GL_RGBA, GLES20.GL_FLOAT,
            java.nio.FloatBuffer.wrap(floatArrayOf(0f, 0f, 0f, 0f))
        )
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
    }

    private fun ensureDpcBuffers(width: Int, height: Int) {
        if (dpcBufferWidth == width && dpcBufferHeight == height && dpcAvgFboId != 0) return
        deleteFboTex(dpcAvgFboId, dpcAvgTexId)
        deleteFboTex(dpcFlagFboId, dpcFlagTexId)
        deleteFboTex(dpcWorkAFboId, dpcWorkATexId)
        deleteFboTex(dpcWorkBFboId, dpcWorkBTexId)
        dpcAvgFboId = 0; dpcAvgTexId = 0
        dpcFlagFboId = 0; dpcFlagTexId = 0
        dpcWorkAFboId = 0; dpcWorkATexId = 0
        dpcWorkBFboId = 0; dpcWorkBTexId = 0
        dpcBufferWidth = width
        dpcBufferHeight = height

        while (true) {
            val (at, af) = allocRgba32fFbo(width, height); dpcAvgTexId = at; dpcAvgFboId = af
            if (dpcAvgFboId == 0) break
            val (ft, ff) = allocRgba32fFbo(width, height); dpcFlagTexId = ft; dpcFlagFboId = ff
            if (dpcFlagFboId == 0) break
            val (wa, waf) = allocRgba32fFbo(width, height); dpcWorkATexId = wa; dpcWorkAFboId = waf
            if (dpcWorkAFboId == 0) break
            val (wb, wbf) = allocRgba32fFbo(width, height); dpcWorkBTexId = wb; dpcWorkBFboId = wbf
            break
        }
        CrashLogger.log(TAG, "Stage-1 DPC buffers: ${width}x${height} avg=$dpcAvgTexId flag=$dpcFlagTexId workA=$dpcWorkATexId workB=$dpcWorkBTexId")
    }

    /**
     * Run the spec-accurate Stage-1 DPC (4-pass) + Stage-3 green-guided GF
     * chain on the sparse Bayer grid.  Returns the output texture ID
     * (dpcWorkBTexId after DPC, denoisedTexId after GF, or dpcWorkATexId
     * for the passthrough path when DPC is disabled).
     */
    private fun runDpcAndGfChain(
        transformMatrix: FloatArray,
        crop: FloatArray,
        gridW: Int, gridH: Int,
        sensorW: Float, sensorH: Float,
        bayerTex: Int,
        blackLevelPattern: IntArray,
        bitDepth: Int,
        dpEnabled: Boolean,
        m1: Float, m2: Float, theta: Float,
        isoModelA: Float, isoModelB: Float,
        corrStrength: Float,
        rawDenoiseActive: Boolean,
        keepRawAlpha: Float,
        s3Eps: Float,
        renderCount: Int
    ): Int {
        ensureFallbackFloatTexture()
        ensureDpcBuffers(gridW, gridH)

        var outputTex = 0
        if (dpEnabled) {
            bindTarget(dpcAvgFboId, gridW, gridH)
            dpcShader.draw(
                DpcShaderProgram.PASS_AVG, transformMatrix,
                crop[0], crop[1], crop[2], crop[3],
                gridW.toFloat(), gridH.toFloat(),
                sensorW, sensorH,
                bayerTex,
                dpcAvgTexId, dpcFlagTexId, fallbackFloatTexId,
                blackLevelPattern, bitDepth,
                dpcEnabled = true, m1 = m1, m2 = m2, theta = theta,
                isoModelA = isoModelA, isoModelB = isoModelB,
                corrStrength = corrStrength
            )
            logGlError("chain-dpc-avg", renderCount)

            bindTarget(dpcFlagFboId, gridW, gridH)
            dpcShader.draw(
                DpcShaderProgram.PASS_DETECT, transformMatrix,
                crop[0], crop[1], crop[2], crop[3],
                gridW.toFloat(), gridH.toFloat(),
                sensorW, sensorH,
                bayerTex,
                dpcAvgTexId, dpcFlagTexId, fallbackFloatTexId,
                blackLevelPattern, bitDepth,
                dpcEnabled = true, m1 = m1, m2 = m2, theta = theta,
                isoModelA = isoModelA, isoModelB = isoModelB,
                corrStrength = corrStrength
            )
            logGlError("chain-dpc-detect", renderCount)

            bindTarget(dpcWorkAFboId, gridW, gridH)
            dpcShader.draw(
                DpcShaderProgram.PASS_CORRECT, transformMatrix,
                crop[0], crop[1], crop[2], crop[3],
                gridW.toFloat(), gridH.toFloat(),
                sensorW, sensorH,
                bayerTex,
                dpcAvgTexId, dpcFlagTexId, fallbackFloatTexId,
                blackLevelPattern, bitDepth,
                dpcEnabled = true, m1 = m1, m2 = m2, theta = theta,
                isoModelA = isoModelA, isoModelB = isoModelB,
                corrStrength = corrStrength
            )
            logGlError("chain-dpc-correct", renderCount)

            bindTarget(dpcWorkBFboId, gridW, gridH)
            dpcShader.draw(
                DpcShaderProgram.PASS_COUPLET, transformMatrix,
                crop[0], crop[1], crop[2], crop[3],
                gridW.toFloat(), gridH.toFloat(),
                sensorW, sensorH,
                bayerTex,
                dpcAvgTexId, dpcFlagTexId, fallbackFloatTexId,
                blackLevelPattern, bitDepth,
                dpcEnabled = true, m1 = m1, m2 = m2, theta = theta,
                isoModelA = isoModelA, isoModelB = isoModelB,
                corrStrength = corrStrength
            )
            logGlError("chain-dpc-couplet", renderCount)
            outputTex = dpcWorkBTexId
        } else {
            // DPC disabled: pack raw → float sparse grid (black-subtracted,
            // clamped) using pass CORRECT in pure pass-through mode.
            bindTarget(dpcWorkAFboId, gridW, gridH)
            dpcShader.draw(
                DpcShaderProgram.PASS_CORRECT, transformMatrix,
                crop[0], crop[1], crop[2], crop[3],
                gridW.toFloat(), gridH.toFloat(),
                sensorW, sensorH,
                bayerTex,
                dpcAvgTexId, dpcFlagTexId, fallbackFloatTexId,
                blackLevelPattern, bitDepth,
                dpcEnabled = false, m1 = 0.4f, m2 = 10f, theta = 6f,
                isoModelA = isoModelA, isoModelB = isoModelB,
                corrStrength = 0f
            )
            logGlError("chain-dpc-passthrough", renderCount)
            outputTex = dpcWorkATexId
        }

        // Stage 3: green-guided GF over the sparse grid.
        if (rawDenoiseActive) {
            ensureDenoiseBuffers(gridW, gridH)
            bindTarget(denoiseFboId, denoiseFboWidth, denoiseFboHeight)
            rawDenoiseShader.draw(
                transformMatrix = transformMatrix,
                cropOriginX = crop[0], cropOriginY = crop[1],
                cropSizeX = crop[2], cropSizeY = crop[3],
                viewWidth = denoiseFboWidth.toFloat(), viewHeight = denoiseFboHeight.toFloat(),
                sensorWidth = sensorW, sensorHeight = sensorH,
                gridTex = outputTex,
                alpha = keepRawAlpha,
                eps = s3Eps
            )
            logGlError("chain-gf-denoise", renderCount)
            outputTex = denoisedTexId
        }

        return outputTex
    }

    private fun ensureSigmaBuffer(width: Int, height: Int) {
        if (sigmaBufferWidth == width && sigmaBufferHeight == height && sigmaFboId != 0) return
        deleteFboTex(sigmaFboId, sigmaTexId)
        sigmaFboId = 0; sigmaTexId = 0
        sigmaBufferWidth = width
        sigmaBufferHeight = height
        val (t, f) = allocRgba32fFbo(width, height)
        sigmaTexId = t; sigmaFboId = f
        CrashLogger.log(TAG, "Stage-2 sigma buffer: ${width}x${height} tex=$sigmaTexId")
    }

    private fun ensureOutNrBuffers(width: Int, height: Int) {
        if (outNrBufferWidth == width && outNrBufferHeight == height && outNr2FboId != 0) return
        deleteFboTex(statsHFboId, statsHTexId)
        deleteFboTex(statsVFboId, statsVTexId)
        deleteFboTex(outNr1FboId, outNr1TexId)
        deleteFboTex(outNr2FboId, outNr2TexId)
        statsHFboId = 0; statsHTexId = 0
        statsVFboId = 0; statsVTexId = 0
        outNr1FboId = 0; outNr1TexId = 0
        outNr2FboId = 0; outNr2TexId = 0
        outNrBufferWidth = width
        outNrBufferHeight = height

        while (true) {
            val (h, hf) = allocRgba32fFbo(width, height); statsHTexId = h; statsHFboId = hf
            if (statsHFboId == 0) break
            val (v, vf) = allocRgba32fFbo(width, height); statsVTexId = v; statsVFboId = vf
            if (statsVFboId == 0) break
            val (o1, o1f) = allocRgba32fFbo(width, height); outNr1TexId = o1; outNr1FboId = o1f
            if (outNr1FboId == 0) break
            val (o2, o2f) = allocRgba32fFbo(width, height); outNr2TexId = o2; outNr2FboId = o2f
            break
        }
        CrashLogger.log(TAG, "Stage-5 buffers: ${width}x${height} statsH=$statsHTexId statsV=$statsVTexId out1=$outNr1TexId out2=$outNr2TexId")
    }

    private fun destroyEgl() {
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)

            if (fboId != 0) {
                GLES20.glDeleteFramebuffers(1, intArrayOf(fboId), 0)
                fboId = 0
            }
            if (fboTextureId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(fboTextureId), 0)
                fboTextureId = 0
            }
            if (captureFboId != 0) {
                GLES20.glDeleteFramebuffers(1, intArrayOf(captureFboId), 0)
                captureFboId = 0
            }
            if (captureFboTextureId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(captureFboTextureId), 0)
                captureFboTextureId = 0
            }
            if (downscaleFboId != 0) {
                GLES20.glDeleteFramebuffers(1, intArrayOf(downscaleFboId), 0)
                downscaleFboId = 0
            }
            if (downscaleFboTextureId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(downscaleFboTextureId), 0)
                downscaleFboTextureId = 0
            }
            if (demosaicFboId != 0) {
                GLES20.glDeleteFramebuffers(1, intArrayOf(demosaicFboId), 0)
                demosaicFboId = 0
            }
            if (demosaicFboTextureId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(demosaicFboTextureId), 0)
                demosaicFboTextureId = 0
            }
            if (rawDemosaicFboId != 0) {
                GLES20.glDeleteFramebuffers(1, intArrayOf(rawDemosaicFboId), 0)
                rawDemosaicFboId = 0
            }
            if (rawDemosaicFboTextureId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(rawDemosaicFboTextureId), 0)
                rawDemosaicFboTextureId = 0
            }
            if (denoiseFboId != 0) {
                GLES20.glDeleteFramebuffers(1, intArrayOf(denoiseFboId), 0)
                denoiseFboId = 0
                GLES20.glDeleteTextures(1, intArrayOf(denoisedTexId), 0)
                denoisedTexId = 0
            }
            if (captureDenoiseFboId != 0) {
                GLES20.glDeleteFramebuffers(1, intArrayOf(captureDenoiseFboId), 0)
                GLES20.glDeleteTextures(1, intArrayOf(captureDenoisedTexId), 0)
                captureDenoiseFboId = 0
                captureDenoisedTexId = 0
            }
            deleteFboTex(s3PackFboId, s3PackTexId)
            s3PackFboId = 0
            s3PackTexId = 0

            deleteFboTex(dpcAvgFboId, dpcAvgTexId)
            deleteFboTex(dpcFlagFboId, dpcFlagTexId)
            deleteFboTex(dpcWorkAFboId, dpcWorkATexId)
            deleteFboTex(dpcWorkBFboId, dpcWorkBTexId)
            dpcAvgFboId = 0; dpcAvgTexId = 0
            dpcFlagFboId = 0; dpcFlagTexId = 0
            dpcWorkAFboId = 0; dpcWorkATexId = 0
            dpcWorkBFboId = 0; dpcWorkBTexId = 0
            deleteFboTex(sigmaFboId, sigmaTexId)
            sigmaFboId = 0; sigmaTexId = 0
            deleteFboTex(statsHFboId, statsHTexId)
            deleteFboTex(statsVFboId, statsVTexId)
            deleteFboTex(outNr1FboId, outNr1TexId)
            deleteFboTex(outNr2FboId, outNr2TexId)
            statsHFboId = 0; statsHTexId = 0
            statsVFboId = 0; statsVTexId = 0
            outNr1FboId = 0; outNr1TexId = 0
            outNr2FboId = 0; outNr2TexId = 0
            if (fallbackFloatTexId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(fallbackFloatTexId), 0)
                fallbackFloatTexId = 0
            }

            if (eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglSurface)
                eglSurface = EGL14.EGL_NO_SURFACE
            }
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(eglDisplay, eglContext)
                eglContext = EGL14.EGL_NO_CONTEXT
            }
            EGL14.eglTerminate(eglDisplay)
            eglDisplay = EGL14.EGL_NO_DISPLAY
        }

        glInitialized = false
    }

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        CrashLogger.log(TAG, "onSurfaceTextureAvailable: ${width}x${height} eglDisplay=$eglDisplay")
        requestRender()
    }

    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
        CrashLogger.log(TAG, "onSurfaceTextureSizeChanged: ${width}x${height}")
    }

    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
        CrashLogger.log(TAG, "onSurfaceTextureDestroyed")
        return true
    }

    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}

    companion object {
        private const val TAG = "PreviewRenderer"
        private const val EGL_RECORDABLE_ANDROID = 0x3142
    }
}
