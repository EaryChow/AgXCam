package com.agx.camera.gpu

import android.graphics.SurfaceTexture
import android.opengl.*
import android.util.Log
import android.view.Surface
import android.view.TextureView
import com.agx.camera.camera.ZoomController
import java.nio.ByteBuffer
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class PreviewRenderer(private val textureView: TextureView) : TextureView.SurfaceTextureListener {

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var eglConfig: EGLConfig? = null

    private var fboId = 0
    private var fboTextureId = 0
    private var fboWidth = 640
    private var fboHeight = 480

    private val yuvShader = YuvShaderProgram()
    private val blitShader = BlitShaderProgram()

    private var renderThread: Thread? = null
    private val renderLock = ReentrantLock()
    private val frameCondition = renderLock.newCondition()
    @Volatile private var hasNewFrame = false
    @Volatile private var running = false

    private var sensorOrientation = 0
    private var isFrontCamera = false

    val zoomController = ZoomController()

    fun setPreviewSize(width: Int, height: Int) {
        fboWidth = width
        fboHeight = height
    }

    private var yPlane: ByteBuffer? = null
    private var uPlane: ByteBuffer? = null
    private var vPlane: ByteBuffer? = null
    private var yuvWidth = 0
    private var yuvHeight = 0

    var onFirstFrameRendered: (() -> Unit)? = null
    private var firstFrameReported = false

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

    fun setSensorOrientation(orientation: Int) {
        sensorOrientation = orientation
    }

    fun setFrontCamera(front: Boolean) {
        isFrontCamera = front
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

    private fun renderLoop() {
        initEgl()

        while (running) {
            renderLock.withLock {
                while (!hasNewFrame && running) {
                    frameCondition.await(100, java.util.concurrent.TimeUnit.MILLISECONDS)
                }
            }
            if (!running) break
            hasNewFrame = false

            if (eglDisplay == EGL14.EGL_NO_DISPLAY || eglSurface == EGL14.EGL_NO_SURFACE) {
                Thread.sleep(16)
                continue
            }

            val y = yPlane ?: continue
            val u = uPlane ?: continue
            val v = vPlane ?: continue
            val w = yuvWidth
            val h = yuvHeight
            if (w <= 0 || h <= 0) continue

            if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
                Log.w(TAG, "eglMakeCurrent failed")
                continue
            }

            val viewW = textureView.width
            val viewH = textureView.height
            if (viewW <= 0 || viewH <= 0) continue

            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId)
            GLES20.glViewport(0, 0, fboWidth, fboHeight)
            GLES20.glClearColor(0f, 0f, 0f, 1f)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

            yuvShader.uploadY(y.duplicate(), w, h)
            yuvShader.uploadU(u.duplicate(), w / 2, h / 2)
            yuvShader.uploadV(v.duplicate(), w / 2, h / 2)

            yuvShader.draw(
                fboWidth, fboHeight,
                zoomController.zoomFactor,
                zoomController.zoomCenterX,
                zoomController.zoomCenterY,
                sensorOrientation,
                isFrontCamera
            )

            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
            GLES20.glViewport(0, 0, viewW, viewH)
            GLES20.glClearColor(0f, 0f, 0f, 1f)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

            blitShader.draw(fboTextureId)

            EGL14.eglSwapBuffers(eglDisplay, eglSurface)

            if (!firstFrameReported) {
                firstFrameReported = true
                onFirstFrameRendered?.invoke()
            }
        }

        yuvShader.destroy()
        blitShader.destroy()
    }

    fun initEgl() {
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
            EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
            EGL14.EGL_NONE
        )
        eglContext = EGL14.eglCreateContext(
            eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0
        )

        createEglSurface()
        createFbo(fboWidth, fboHeight)

        yuvShader.create()
        blitShader.create()

        Log.d(TAG, "EGL initialized: display=$eglDisplay context=$eglContext")
    }

    fun createEglSurface() {
        val st = textureView.surfaceTexture ?: return
        val surfaceAttribs = intArrayOf(EGL14.EGL_NONE)
        eglSurface = EGL14.eglCreateWindowSurface(
            eglDisplay, eglConfig, st, surfaceAttribs, 0
        )
        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)
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
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        Log.d(TAG, "FBO created: ${width}x${height}")
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
    }

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        Log.d(TAG, "Surface available: ${width}x${height}")
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            createEglSurface()
        }
    }

    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
        Log.d(TAG, "Surface size changed: ${width}x${height}")
    }

    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
        Log.d(TAG, "Surface destroyed")
        return true
    }

    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}

    companion object {
        private const val TAG = "PreviewRenderer"
        private const val EGL_RECORDABLE_ANDROID = 0x3142
    }
}
