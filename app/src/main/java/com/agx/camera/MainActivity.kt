package com.agx.camera

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.TextureView
import android.view.View
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.agx.camera.camera.*
import com.agx.camera.gpu.PreviewRenderer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var camera2Manager: Camera2Manager
    private lateinit var lensManager: LensManager
    private lateinit var previewRenderer: PreviewRenderer

    private var currentFlashMode = FlashMode.OFF
    private var currentWbMode = WhiteBalanceMode.AUTO
    private var cameraReady = false

    private lateinit var textureView: TextureView
    private lateinit var devBanner: TextView
    private lateinit var flashButton: TextView
    private lateinit var wbButton: TextView
    private lateinit var zoomLabel: TextView
    private lateinit var zoomSlider: SeekBar
    private lateinit var lensSelector: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        textureView = findViewById(R.id.preview_texture)
        devBanner = findViewById(R.id.dev_banner)
        flashButton = findViewById(R.id.flash_button)
        wbButton = findViewById(R.id.wb_button)
        zoomLabel = findViewById(R.id.zoom_label)
        zoomSlider = findViewById(R.id.zoom_slider)
        lensSelector = findViewById(R.id.lens_selector)

        if (BuildConfig.AGX_ENABLE_YUV_FALLBACK) {
            devBanner.visibility = View.VISIBLE
        }

        lensManager = LensManager(this)
        camera2Manager = Camera2Manager(this)

        previewRenderer = PreviewRenderer(textureView).apply {
            onFirstFrameRendered = {
                Log.d(TAG, "First frame rendered")
            }
        }

        textureView.surfaceTextureListener = previewRenderer

        setupUI()
        checkPermissions()
    }

    private fun setupUI() {
        flashButton.setOnClickListener {
            currentFlashMode = currentFlashMode.cycle()
            updateFlashUI()
            if (cameraReady) {
                camera2Manager.setFlashMode(currentFlashMode)
            }
        }

        wbButton.setOnClickListener {
            currentWbMode = when (currentWbMode) {
                WhiteBalanceMode.AUTO -> WhiteBalanceMode.KELVIN
                WhiteBalanceMode.KELVIN -> WhiteBalanceMode.GRAY_CARD
                WhiteBalanceMode.GRAY_CARD -> WhiteBalanceMode.AUTO
            }
            updateWbUI()
        }

        zoomSlider.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                if (!fromUser) return
                val zoom = ZoomController.MIN_ZOOM +
                    (progress / 400f) * (ZoomController.MAX_ZOOM - ZoomController.MIN_ZOOM)
                previewRenderer.zoomController.setZoom(zoom)
                zoomLabel.text = String.format("%.1fx", zoom)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })

        previewRenderer.zoomController.listener = { zoom, _, _ ->
            val progress = ((zoom - ZoomController.MIN_ZOOM) /
                (ZoomController.MAX_ZOOM - ZoomController.MIN_ZOOM) * 400).toInt()
            if (zoomSlider.progress != progress) {
                zoomSlider.progress = progress
            }
            zoomLabel.text = String.format("%.1fx", zoom)
        }

        updateFlashUI()
        updateWbUI()
    }

    private fun updateFlashUI() {
        val icon = when (currentFlashMode) {
            FlashMode.OFF -> "\u26A1"
            FlashMode.AUTO -> "\u26A1A"
            FlashMode.ON -> "\u26A1!"
            FlashMode.TORCH -> "\uD83D\uDCA1"
        }
        flashButton.text = icon
    }

    private fun updateWbUI() {
        val label = when (currentWbMode) {
            WhiteBalanceMode.AUTO -> "WB"
            WhiteBalanceMode.KELVIN -> "WB\u00B0K"
            WhiteBalanceMode.GRAY_CARD -> "WB\u25A1"
        }
        wbButton.text = label
    }

    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            initCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initCamera()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    private fun initCamera() {
        val hasSupported = lensManager.enumerate()
        if (!hasSupported) {
            Toast.makeText(this, "No supported camera found (requires hardware level FULL+)", Toast.LENGTH_LONG).show()
            Log.e(TAG, "No camera with INFO_SUPPORTED_HARDWARE_LEVEL >= FULL found")
            return
        }

        val primary = lensManager.selectPrimary()
        if (primary == null) {
            Log.e(TAG, "No primary lens found")
            return
        }

        val previewSize = lensManager.getBestPreviewSize(primary)
        Log.d(TAG, "Selected preview size: ${previewSize.width}x${previewSize.height}")

        buildLensSelectorUI()

        previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
        previewRenderer.start()

        val sensorOrientation = lensManager.getSensorOrientation(primary)
        previewRenderer.setSensorOrientation(sensorOrientation)

        camera2Manager.onFrameAvailable = frameHandler@{ image ->
            if (!cameraReady) return@frameHandler
            val planes = image.planes
            val w = image.width
            val h = image.height

            val yBuffer = planes[0].buffer
            val yRowStride = planes[0].rowStride
            val yCopy = extractPlane(yBuffer, w, h, yRowStride, 1)

            val uvBuffer = planes[1].buffer
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride
            val uvWidth = w / 2
            val uvHeight = h / 2
            val uCopy = extractPlane(uvBuffer, uvWidth, uvHeight, uvRowStride, uvPixelStride)
            val vCopy = extractPlane(planes[2].buffer, uvWidth, uvHeight, planes[2].rowStride, planes[2].pixelStride)

            previewRenderer.setYuvFrame(yCopy, uCopy, vCopy, w, h)
        }

        camera2Manager.onSessionReady = { width, height ->
            Log.d(TAG, "Camera session ready: ${width}x${height}")
            cameraReady = true
        }

        camera2Manager.onError = { error ->
            Log.e(TAG, error)
        }

        camera2Manager.startBackgroundThread()
        camera2Manager.openCamera(primary, previewSize)
    }

    private fun extractPlane(
        src: ByteBuffer, width: Int, height: Int,
        rowStride: Int, pixelStride: Int
    ): ByteBuffer {
        val dst = ByteBuffer.allocateDirect(width * height).order(java.nio.ByteOrder.nativeOrder())
        src.position(0)

        if (pixelStride == 1) {
            // Tightly packed (Y plane or tightly packed UV) — row-by-row bulk copy
            for (row in 0 until height) {
                val srcOffset = row * rowStride
                val dstOffset = row * width
                src.position(srcOffset)
                src.limit(srcOffset + width)
                dst.position(dstOffset)
                dst.put(src)
            }
        } else if (pixelStride == 2) {
            // Interleaved UV (NV12/NV21) — bulk copy rows, then deinterleave
            for (row in 0 until height) {
                val srcRowStart = row * rowStride
                val dstRowStart = row * width
                src.position(srcRowStart)
                for (col in 0 until width) {
                    dst.put(dstRowStart + col, src.get(srcRowStart + col * pixelStride))
                }
            }
        } else {
            // Generic stride with non-standard pixel stride
            for (row in 0 until height) {
                val srcRowStart = row * rowStride
                val dstRowStart = row * width
                for (col in 0 until width) {
                    val srcIdx = srcRowStart + col * pixelStride
                    if (srcIdx < src.capacity()) {
                        dst.put(dstRowStart + col, src.get(srcIdx))
                    }
                }
            }
        }

        dst.position(0)
        src.position(0)
        src.limit(src.capacity())
        return dst
    }

    private fun buildLensSelectorUI() {
        // Week 1: single lens only. Multi-lens selector deferred to Week 9.
        lensSelector.removeAllViews()
        val primaryLens = lensManager.activeLens ?: return
        val btn = TextView(this).apply {
            text = primaryLens.label
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 12f
            setPadding(24, 12, 24, 12)
            setBackgroundColor(0xFF4488FF.toInt())
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            params.marginStart = 8
            layoutParams = params
        }
        lensSelector.addView(btn)
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            if (!cameraReady) {
                val lens = lensManager.activeLens ?: lensManager.selectPrimary()
                if (lens != null) {
                    val previewSize = lensManager.getBestPreviewSize(lens)
                    previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
                    previewRenderer.start()
                    camera2Manager.startBackgroundThread()
                    camera2Manager.openCamera(lens, previewSize)
                }
            }
        }
    }

    override fun onPause() {
        super.onPause()
        camera2Manager.close()
        camera2Manager.stopBackgroundThread()
        cameraReady = false
        previewRenderer.stop()
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CAMERA = 100
    }
}
