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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

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

    private val yuvShader = YuvShaderProgram()
    private val bayerShader = BayerShaderProgram()
    private val nrShader = NrShaderProgram()
    private val blitShader = BlitShaderProgram()

    @Volatile var useBayerPath = false
        private set

    @Volatile private var bayerBuffer: ByteBuffer? = null
    @Volatile private var bayerWidth = 0
    @Volatile private var bayerHeight = 0
    private var bayerStridePixels = 0
    private var bayerFrameVersion = 0L

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
    @Volatile var wbGainR = 1f
    @Volatile var wbGainG = 1f
    @Volatile var wbGainB = 1f
    @Volatile var ccMatrix: FloatArray? = null
    var bayerLensShadingData: ShortArray? = null
    var bayerLensShadingWidth = 1
    private var bayerLensShadingHeight = 1

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
        bayerFrameVersion++
        hasNewFrame = true
        renderLock.withLock {
            frameCondition.signal()
        }
    }

    fun start() {
        running = true
        renderThread = Thread({ renderLoop() }, "PreviewRenderer").also { it.start() }
    }

    fun stop() {
        running = false
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

    private var renderFrameCount = 0
    private var bayerRenderCount = 0
    private var lastBayerCropLog: String? = null

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

        renderFrameCount++
        if (renderFrameCount == 1 || renderFrameCount % 30 == 0) {
            CrashLogger.log(TAG, "renderYuvFrame: #$renderFrameCount ${w}x${h} view=${viewW}x${viewH} fbo=$fboId eglSurface=$eglSurface")
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
        val buffer = bayerBuffer
        if (buffer == null || bayerWidth <= 0 || bayerHeight <= 0) {
            clearAndSwap(viewW, viewH)
            return
        }

        applyBayerCrop()

        bayerRenderCount++
        if (bayerRenderCount == 1 || bayerRenderCount % 300 == 0) {
            CrashLogger.log(
                TAG, "renderBayerFrame: #$bayerRenderCount ${bayerWidth}x${bayerHeight} " +
                    "stride=$bayerStridePixels fbo=$fboId demosaicFbo=$demosaicFboId " +
                    "bayerReady=${bayerShader.isReady()} nrReady=${nrShader.isReady()} " +
                    "v=$bayerFrameVersion"
            )
        }

        ensureDemosaicFbo(fboWidth, fboHeight)

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, demosaicFboId)
        GLES20.glViewport(0, 0, demosaicFboWidth, demosaicFboHeight)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        bayerShader.uploadBayer(buffer.duplicate(), bayerWidth, bayerHeight, bayerStridePixels)
        logGlError("after uploadBayer", bayerRenderCount)

        bayerShader.drawDemosaic(
            demosaicFboWidth, demosaicFboHeight,
            computePreviewTransform(bayerWidth, bayerHeight),
            bayerBlackLevelPattern,
            bayerColorMap,
            bayerBitDepth,
            agxWhiteLevel, agxBlackLevel,
            boxAA = 4,
            wbGains = floatArrayOf(wbGainR, wbGainG, wbGainB),
            colorMat = ccMatrix
        )
        logGlError("after drawDemosaic", bayerRenderCount)

        if (bayerRenderCount <= 3 || bayerRenderCount % 300 == 0) {
            val px = probePixel(demosaicFboId, demosaicFboWidth, demosaicFboHeight)
            val reg = probeRegion(
                demosaicFboId,
                demosaicFboWidth / 4, demosaicFboHeight / 4,
                demosaicFboWidth / 2, demosaicFboHeight / 2
            )
            CrashLogger.log(TAG, "demosaic probe: center=$px region=$reg")
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId)
        GLES20.glViewport(0, 0, fboWidth, fboHeight)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        nrShader.draw(
            demosaicFboTextureId,
            bayerNrStrength,
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

        if (bayerRenderCount <= 3 || bayerRenderCount % 300 == 0) {
            val px = probePixel(fboId, fboWidth, fboHeight)
            val reg = probeRegion(
                fboId,
                fboWidth / 4, fboHeight / 4,
                fboWidth / 2, fboHeight / 2
            )
            CrashLogger.log(TAG, "nr probe: center=$px region=$reg")
        }
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

    private fun probeRegion(fboId: Int, x: Int, y: Int, w: Int, h: Int): String {
        return try {
            val buf = ByteBuffer.allocateDirect(w * h * 4)
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId)
            GLES20.glReadPixels(x, y, w, h, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buf)
            var min = 255
            var max = 0
            var i = buf.position()
            val limit = buf.limit()
            while (i < limit) {
                val v0 = buf.get(i).toInt() and 0xFF
                val v1 = buf.get(i + 1).toInt() and 0xFF
                val v2 = buf.get(i + 2).toInt() and 0xFF
                if (v0 < min) min = v0
                if (v1 < min) min = v1
                if (v2 < min) min = v2
                if (v0 > max) max = v0
                if (v1 > max) max = v1
                if (v2 > max) max = v2
                i += 4
            }
            "min=$min max=$max"
        } catch (t: Throwable) {
            "probe failed: ${t.message}"
        }
    }

    private fun probePixel(fboId: Int, width: Int, height: Int): String {
        return try {
            val buf = ByteBuffer.allocateDirect(4)
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId)
            GLES20.glReadPixels(width / 2, height / 2, 1, 1, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buf)
            val r = buf.get(0).toInt() and 0xFF
            val g = buf.get(1).toInt() and 0xFF
            val b = buf.get(2).toInt() and 0xFF
            val a = buf.get(3).toInt() and 0xFF
            "$r,$g,$b,$a"
        } catch (t: Throwable) {
            "probe failed: ${t.message}"
        }
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
                    bayerShader.drawDemosaic(
                        rawDemosaicFboWidth, rawDemosaicFboHeight,
                        captureMatrix,
                        bayerBlackLevelPattern,
                        bayerColorMap,
                        bayerBitDepth,
                        rawCaptureReq.agxWhiteLevel, rawCaptureReq.agxBlackLevel,
                        boxAA = if (outW < rawCaptureReq.rawW || outH < rawCaptureReq.rawH) 4 else 0,
                        wbGains = floatArrayOf(wbGainR, wbGainG, wbGainB),
                        colorMat = ccMatrix
                    )
                    logGlError("raw after drawDemosaic", bayerRenderCount)

                    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, downscaleFboId)
                    GLES20.glViewport(0, 0, downscaleFboWidth, downscaleFboHeight)
                    GLES20.glClearColor(0f, 0f, 0f, 1f)
                    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

                    nrShader.draw(
                        rawDemosaicFboTextureId,
                        0f,
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

            if (useBayerPath) {
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

            blitShader.draw(fboTextureId)

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
        blitShader.destroy()
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
        blitShader.create()

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
