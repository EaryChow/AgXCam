package com.agx.camera

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.TextureView
import android.view.View
import android.widget.*
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.agx.camera.camera.*
import com.agx.camera.color.AgxParams
import com.agx.camera.color.AgxPrecomputer
import com.agx.camera.color.WhiteBalanceMath
import com.agx.camera.gpu.PreviewRenderer

class MainActivity : AppCompatActivity() {

    private lateinit var camera2Manager: Camera2Manager
    private lateinit var lensManager: LensManager
    private lateinit var previewRenderer: PreviewRenderer
    private lateinit var presetManager: PresetManager

    private var currentFlashMode = FlashMode.OFF
    private var currentWbMode = WhiteBalanceMode.AUTO
    private var kelvinState = KelvinState()
    private var agxParams = AgxParams()
    private var photoOutput = PhotoOutputSettings()
    private var cameraReady = false
    private var settingsPanelOpen = false

    private lateinit var textureView: TextureView
    private lateinit var devBanner: TextView
    private lateinit var flashButton: TextView
    private lateinit var wbButton: TextView
    private lateinit var settingsButton: TextView
    private lateinit var zoomLabel: TextView
    private lateinit var zoomSlider: SeekBar
    private lateinit var lensSelector: LinearLayout
    private lateinit var settingsPanel: ScrollView

    // Preset
    private lateinit var presetSpinner: Spinner
    private lateinit var presetSaveBtn: TextView
    private lateinit var presetDeleteBtn: TextView
    private lateinit var presetRenameBtn: TextView
    private lateinit var presetResetBtn: TextView

    // Curve
    private lateinit var contrastLabel: TextView
    private lateinit var contrastSlider: SeekBar
    private lateinit var toeLabel: TextView
    private lateinit var toeSlider: SeekBar
    private lateinit var shoulderLabel: TextView
    private lateinit var shoulderSlider: SeekBar

    // Inset
    private lateinit var usePreForPostCb: CheckBox
    private lateinit var insetRotRLabel: TextView; private lateinit var insetRotRSlider: SeekBar
    private lateinit var insetRotGLabel: TextView; private lateinit var insetRotGSlider: SeekBar
    private lateinit var insetRotBLabel: TextView; private lateinit var insetRotBSlider: SeekBar
    private lateinit var insetPurRLabel: TextView; private lateinit var insetPurRSlider: SeekBar
    private lateinit var insetPurGLabel: TextView; private lateinit var insetPurGSlider: SeekBar
    private lateinit var insetPurBLabel: TextView; private lateinit var insetPurBSlider: SeekBar

    // Outset
    private lateinit var outsetSection: LinearLayout
    private lateinit var outsetRotRLabel: TextView; private lateinit var outsetRotRSlider: SeekBar
    private lateinit var outsetRotGLabel: TextView; private lateinit var outsetRotGSlider: SeekBar
    private lateinit var outsetRotBLabel: TextView; private lateinit var outsetRotBSlider: SeekBar
    private lateinit var outsetPurRLabel: TextView; private lateinit var outsetPurRSlider: SeekBar
    private lateinit var outsetPurGLabel: TextView; private lateinit var outsetPurGSlider: SeekBar
    private lateinit var outsetPurBLabel: TextView; private lateinit var outsetPurBSlider: SeekBar
    private lateinit var copyInsetBtn: TextView

    // Tinting
    private lateinit var tintingScaleLabel: TextView; private lateinit var tintingScaleSlider: SeekBar
    private lateinit var tintingHueLabel: TextView; private lateinit var tintingHueSlider: SeekBar

    // NR
    private lateinit var nrLabel: TextView; private lateinit var nrSlider: SeekBar

    // WB
    private lateinit var kelvinLabel: TextView; private lateinit var kelvinSlider: SeekBar
    private lateinit var tintLabel: TextView; private lateinit var tintSlider: SeekBar

    // Output
    private lateinit var jpegLabel: TextView; private lateinit var jpegSlider: SeekBar
    private lateinit var resolutionSpinner: Spinner

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
        settingsButton = findViewById(R.id.settings_button)
        zoomLabel = findViewById(R.id.zoom_label)
        zoomSlider = findViewById(R.id.zoom_slider)
        lensSelector = findViewById(R.id.lens_selector)
        settingsPanel = findViewById(R.id.settings_panel)

        presetSpinner = findViewById(R.id.preset_spinner)
        presetSaveBtn = findViewById(R.id.preset_save_btn)
        presetDeleteBtn = findViewById(R.id.preset_delete_btn)
        presetRenameBtn = findViewById(R.id.preset_rename_btn)
        presetResetBtn = findViewById(R.id.preset_reset_btn)

        contrastLabel = findViewById(R.id.contrast_label); contrastSlider = findViewById(R.id.contrast_slider)
        toeLabel = findViewById(R.id.toe_label); toeSlider = findViewById(R.id.toe_slider)
        shoulderLabel = findViewById(R.id.shoulder_label); shoulderSlider = findViewById(R.id.shoulder_slider)

        usePreForPostCb = findViewById(R.id.use_pre_for_post_cb)
        insetRotRLabel = findViewById(R.id.inset_rot_r_label); insetRotRSlider = findViewById(R.id.inset_rot_r_slider)
        insetRotGLabel = findViewById(R.id.inset_rot_g_label); insetRotGSlider = findViewById(R.id.inset_rot_g_slider)
        insetRotBLabel = findViewById(R.id.inset_rot_b_label); insetRotBSlider = findViewById(R.id.inset_rot_b_slider)
        insetPurRLabel = findViewById(R.id.inset_pur_r_label); insetPurRSlider = findViewById(R.id.inset_pur_r_slider)
        insetPurGLabel = findViewById(R.id.inset_pur_g_label); insetPurGSlider = findViewById(R.id.inset_pur_g_slider)
        insetPurBLabel = findViewById(R.id.inset_pur_b_label); insetPurBSlider = findViewById(R.id.inset_pur_b_slider)

        outsetSection = findViewById(R.id.outset_section)
        outsetRotRLabel = findViewById(R.id.outset_rot_r_label); outsetRotRSlider = findViewById(R.id.outset_rot_r_slider)
        outsetRotGLabel = findViewById(R.id.outset_rot_g_label); outsetRotGSlider = findViewById(R.id.outset_rot_g_slider)
        outsetRotBLabel = findViewById(R.id.outset_rot_b_label); outsetRotBSlider = findViewById(R.id.outset_rot_b_slider)
        outsetPurRLabel = findViewById(R.id.outset_pur_r_label); outsetPurRSlider = findViewById(R.id.outset_pur_r_slider)
        outsetPurGLabel = findViewById(R.id.outset_pur_g_label); outsetPurGSlider = findViewById(R.id.outset_pur_g_slider)
        outsetPurBLabel = findViewById(R.id.outset_pur_b_label); outsetPurBSlider = findViewById(R.id.outset_pur_b_slider)
        copyInsetBtn = findViewById(R.id.copy_inset_to_outset_btn)

        tintingScaleLabel = findViewById(R.id.tinting_scale_label); tintingScaleSlider = findViewById(R.id.tinting_scale_slider)
        tintingHueLabel = findViewById(R.id.tinting_hue_label); tintingHueSlider = findViewById(R.id.tinting_hue_slider)

        nrLabel = findViewById(R.id.nr_label); nrSlider = findViewById(R.id.nr_slider)

        kelvinLabel = findViewById(R.id.kelvin_label); kelvinSlider = findViewById(R.id.kelvin_slider)
        tintLabel = findViewById(R.id.tint_label); tintSlider = findViewById(R.id.tint_slider)

        jpegLabel = findViewById(R.id.jpeg_label); jpegSlider = findViewById(R.id.jpeg_slider)
        resolutionSpinner = findViewById(R.id.resolution_spinner)

        if (BuildConfig.AGX_ENABLE_YUV_FALLBACK) {
            devBanner.visibility = View.VISIBLE
        }

        lensManager = LensManager(this)
        camera2Manager = Camera2Manager(this)
        presetManager = PresetManager(this)
        presetManager.ensureDefault()

        previewRenderer = PreviewRenderer(textureView).apply {
            onFirstFrameRendered = { Log.d(TAG, "First frame rendered") }
        }

        textureView.surfaceTextureListener = previewRenderer

        setupUI()
        loadPreset(PresetManager.PRESET_DEFAULT)
        checkPermissions()
    }

    private fun setupUI() {
        flashButton.setOnClickListener {
            currentFlashMode = currentFlashMode.cycle()
            updateFlashUI()
            if (cameraReady) camera2Manager.setFlashMode(currentFlashMode)
        }

        wbButton.setOnClickListener {
            currentWbMode = when (currentWbMode) {
                WhiteBalanceMode.AUTO -> WhiteBalanceMode.KELVIN
                WhiteBalanceMode.KELVIN -> WhiteBalanceMode.GRAY_CARD
                WhiteBalanceMode.GRAY_CARD -> WhiteBalanceMode.AUTO
            }
            updateWbUI()
            syncWbSliders()
            uploadAgxUniforms()
        }

        settingsButton.setOnClickListener {
            settingsPanelOpen = !settingsPanelOpen
            settingsPanel.visibility = if (settingsPanelOpen) View.VISIBLE else View.GONE
        }

        zoomSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            val zoom = ZoomController.MIN_ZOOM + (v / 400f) * (ZoomController.MAX_ZOOM - ZoomController.MIN_ZOOM)
            previewRenderer.zoomController.setZoom(zoom)
            zoomLabel.text = String.format("%.1fx", zoom)
        })

        previewRenderer.zoomController.listener = { zoom, _, _ ->
            val progress = ((zoom - ZoomController.MIN_ZOOM) / (ZoomController.MAX_ZOOM - ZoomController.MIN_ZOOM) * 400).toInt()
            if (zoomSlider.progress != progress) zoomSlider.progress = progress
            zoomLabel.text = String.format("%.1fx", zoom)
        }

        setupSettingsPanel()
        updateFlashUI()
        updateWbUI()
    }

    private fun setupSettingsPanel() {
        // --- Curve ---
        contrastSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(contrast = 1.4f + v * 0.1f)
            contrastLabel.text = String.format("Contrast  %.1f", agxParams.contrast)
            uploadAgxUniforms()
        })
        toeSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(toe = 0.7f + v * 0.1f)
            toeLabel.text = String.format("Toe  %.1f", agxParams.toe)
            uploadAgxUniforms()
        })
        shoulderSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(shoulder = 0.7f + v * 0.1f)
            shoulderLabel.text = String.format("Shoulder  %.1f", agxParams.shoulder)
            uploadAgxUniforms()
        })

        // --- Inset ---
        usePreForPostCb.setOnCheckedChangeListener { _, checked ->
            agxParams = agxParams.copy(usePreForPost = checked)
            outsetSection.visibility = if (checked) View.GONE else View.VISIBLE
            uploadAgxUniforms()
        }

        val rotRange = 0.5236f // ±0.2618
        fun rotToProgress(v: Float) = ((v + rotRange) / (rotRange * 2) * 524).toInt().coerceIn(0, 524)
        fun progressToRot(p: Float) = (p / 524f * rotRange * 2) - rotRange

        insetRotRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(rgbRotation = floatArrayOf(progressToRot(v.toFloat()), agxParams.rgbRotation[1], agxParams.rgbRotation[2]))
            insetRotRLabel.text = String.format("RGB Rot R  %.3f", agxParams.rgbRotation[0])
            uploadAgxUniforms()
        })
        insetRotGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(rgbRotation = floatArrayOf(agxParams.rgbRotation[0], progressToRot(v.toFloat()), agxParams.rgbRotation[2]))
            insetRotGLabel.text = String.format("RGB Rot G  %.3f", agxParams.rgbRotation[1])
            uploadAgxUniforms()
        })
        insetRotBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(rgbRotation = floatArrayOf(agxParams.rgbRotation[0], agxParams.rgbRotation[1], progressToRot(v.toFloat())))
            insetRotBLabel.text = String.format("RGB Rot B  %.3f", agxParams.rgbRotation[2])
            uploadAgxUniforms()
        })

        insetPurRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(purityAttenuation = floatArrayOf(v.toFloat(), agxParams.purityAttenuation[1], agxParams.purityAttenuation[2]))
            insetPurRLabel.text = String.format("Purity R  %.1f", agxParams.purityAttenuation[0])
            uploadAgxUniforms()
        })
        insetPurGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(purityAttenuation = floatArrayOf(agxParams.purityAttenuation[0], v.toFloat(), agxParams.purityAttenuation[2]))
            insetPurGLabel.text = String.format("Purity G  %.1f", agxParams.purityAttenuation[1])
            uploadAgxUniforms()
        })
        insetPurBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(purityAttenuation = floatArrayOf(agxParams.purityAttenuation[0], agxParams.purityAttenuation[1], v.toFloat()))
            insetPurBLabel.text = String.format("Purity B  %.1f", agxParams.purityAttenuation[2])
            uploadAgxUniforms()
        })

        // --- Outset ---
        outsetRotRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(reverseRgbRotation = floatArrayOf(progressToRot(v.toFloat()), agxParams.reverseRgbRotation[1], agxParams.reverseRgbRotation[2]))
            outsetRotRLabel.text = String.format("Rev Rot R  %.3f", agxParams.reverseRgbRotation[0])
            uploadAgxUniforms()
        })
        outsetRotGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(reverseRgbRotation = floatArrayOf(agxParams.reverseRgbRotation[0], progressToRot(v.toFloat()), agxParams.reverseRgbRotation[2]))
            outsetRotGLabel.text = String.format("Rev Rot G  %.3f", agxParams.reverseRgbRotation[1])
            uploadAgxUniforms()
        })
        outsetRotBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(reverseRgbRotation = floatArrayOf(agxParams.reverseRgbRotation[0], agxParams.reverseRgbRotation[1], progressToRot(v.toFloat())))
            outsetRotBLabel.text = String.format("Rev Rot B  %.3f", agxParams.reverseRgbRotation[2])
            uploadAgxUniforms()
        })

        outsetPurRSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(restorePurity = floatArrayOf(v.toFloat(), agxParams.restorePurity[1], agxParams.restorePurity[2]))
            outsetPurRLabel.text = String.format("Restore R  %.1f", agxParams.restorePurity[0])
            uploadAgxUniforms()
        })
        outsetPurGSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(restorePurity = floatArrayOf(agxParams.restorePurity[0], v.toFloat(), agxParams.restorePurity[2]))
            outsetPurGLabel.text = String.format("Restore G  %.1f", agxParams.restorePurity[1])
            uploadAgxUniforms()
        })
        outsetPurBSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(restorePurity = floatArrayOf(agxParams.restorePurity[0], agxParams.restorePurity[1], v.toFloat()))
            outsetPurBLabel.text = String.format("Restore B  %.1f", agxParams.restorePurity[2])
            uploadAgxUniforms()
        })

        copyInsetBtn.setOnClickListener {
            agxParams = agxParams.copy(
                reverseRgbRotation = agxParams.rgbRotation.copyOf(),
                restorePurity = agxParams.purityAttenuation.copyOf()
            )
            syncOutsetSliders()
            uploadAgxUniforms()
        }

        // --- Tinting ---
        tintingScaleSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(tintingScale = (v - 200) * 0.001f)
            tintingScaleLabel.text = String.format("Scale  %.3f", agxParams.tintingScale)
            uploadAgxUniforms()
        })
        tintingHueSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(tintingHue = (v - 314) * 0.01f)
            tintingHueLabel.text = String.format("Hue  %.2f", agxParams.tintingHue)
            uploadAgxUniforms()
        })

        // --- NR ---
        nrSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            agxParams = agxParams.copy(nrStrength = v / 100f)
            nrLabel.text = String.format("NR Strength  %.1f", agxParams.nrStrength)
            uploadAgxUniforms()
        })

        // --- WB ---
        kelvinSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            kelvinState = kelvinState.copy(kelvin = 2000f + v * 100f)
            kelvinLabel.text = String.format("Kelvin  %.0fK", kelvinState.kelvin)
            if (currentWbMode == WhiteBalanceMode.KELVIN) uploadAgxUniforms()
        })
        tintSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            kelvinState = kelvinState.copy(tint = (v - 100).toFloat())
            tintLabel.text = String.format("Tint  %.0f", kelvinState.tint)
            if (currentWbMode == WhiteBalanceMode.KELVIN) uploadAgxUniforms()
        })

        // --- Output ---
        jpegSlider.setOnSeekBarChangeListener(simpleSeekBar { v ->
            photoOutput = photoOutput.copy(jpegQuality = v)
            jpegLabel.text = String.format("JPEG Quality  %d", v)
        })

        val resOptions = arrayOf("Full", "1/2", "1/4")
        resolutionSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, resOptions).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        resolutionSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                // Resolution applied at capture time (Week 5)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // --- Presets ---
        val presetAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, mutableListOf<String>())
        presetAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        presetSpinner.adapter = presetAdapter
        refreshPresetSpinner()

        presetSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val name = parent?.getItemAtPosition(position) as? String ?: return
                loadPreset(name)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        presetSaveBtn.setOnClickListener {
            val name = presetSpinner.selectedItem as? String ?: return@setOnClickListener
            if (name == PresetManager.PRESET_DEFAULT) return@setOnClickListener
            presetManager.save(PresetManager.Preset(name = name, agxParams = agxParams))
            Toast.makeText(this, "Saved: $name", Toast.LENGTH_SHORT).show()
        }

        presetDeleteBtn.setOnClickListener {
            val name = presetSpinner.selectedItem as? String ?: return@setOnClickListener
            if (name == PresetManager.PRESET_DEFAULT) {
                Toast.makeText(this, "Cannot delete Default", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            presetManager.delete(name)
            refreshPresetSpinner()
            Toast.makeText(this, "Deleted: $name", Toast.LENGTH_SHORT).show()
        }

        presetRenameBtn.setOnClickListener {
            val oldName = presetSpinner.selectedItem as? String ?: return@setOnClickListener
            if (oldName == PresetManager.PRESET_DEFAULT) {
                Toast.makeText(this, "Cannot rename Default", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val input = EditText(this).apply { setText(oldName); selectAll() }
            AlertDialog.Builder(this)
                .setTitle("Rename Preset")
                .setView(input)
                .setPositiveButton("Rename") { _, _ ->
                    val newName = input.text.toString().trim()
                    if (newName.isNotEmpty() && newName != oldName) {
                        presetManager.rename(oldName, newName)
                        refreshPresetSpinner()
                        selectPreset(newName)
                    }
                }
                .setNegativeButton("Cancel", null)
                .show()
        }

        presetResetBtn.setOnClickListener {
            loadPreset(PresetManager.PRESET_DEFAULT)
            selectPreset(PresetManager.PRESET_DEFAULT)
            Toast.makeText(this, "Reset to Default", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadPreset(name: String) {
        val preset = presetManager.load(name) ?: return
        agxParams = preset.agxParams
        syncAllSliders()
        uploadAgxUniforms()
    }

    private fun syncAllSliders() {
        contrastSlider.progress = ((agxParams.contrast - 1.4f) / 0.1f).toInt().coerceIn(0, 26)
        contrastLabel.text = String.format("Contrast  %.1f", agxParams.contrast)
        toeSlider.progress = ((agxParams.toe - 0.7f) / 0.1f).toInt().coerceIn(0, 93)
        toeLabel.text = String.format("Toe  %.1f", agxParams.toe)
        shoulderSlider.progress = ((agxParams.shoulder - 0.7f) / 0.1f).toInt().coerceIn(0, 93)
        shoulderLabel.text = String.format("Shoulder  %.1f", agxParams.shoulder)

        usePreForPostCb.isChecked = agxParams.usePreForPost
        outsetSection.visibility = if (agxParams.usePreForPost) View.GONE else View.VISIBLE
        syncInsetSliders()
        syncOutsetSliders()

        tintingScaleSlider.progress = (agxParams.tintingScale / 0.001f + 200).toInt().coerceIn(0, 400)
        tintingScaleLabel.text = String.format("Scale  %.3f", agxParams.tintingScale)
        tintingHueSlider.progress = (agxParams.tintingHue / 0.01f + 314).toInt().coerceIn(0, 628)
        tintingHueLabel.text = String.format("Hue  %.2f", agxParams.tintingHue)

        nrSlider.progress = (agxParams.nrStrength * 100).toInt().coerceIn(0, 100)
        nrLabel.text = String.format("NR Strength  %.1f", agxParams.nrStrength)

        kelvinSlider.progress = ((kelvinState.kelvin - 2000f) / 100f).toInt().coerceIn(0, 80)
        kelvinLabel.text = String.format("Kelvin  %.0fK", kelvinState.kelvin)
        tintSlider.progress = (kelvinState.tint + 100).toInt().coerceIn(0, 200)
        tintLabel.text = String.format("Tint  %.0f", kelvinState.tint)

        jpegSlider.progress = photoOutput.jpegQuality
        jpegLabel.text = String.format("JPEG Quality  %d", photoOutput.jpegQuality)

        syncWbSliders()
    }

    private fun syncInsetSliders() {
        val rotRange = 0.5236f
        fun rotToProgress(v: Float) = ((v + rotRange) / (rotRange * 2) * 524).toInt().coerceIn(0, 524)
        insetRotRSlider.progress = rotToProgress(agxParams.rgbRotation[0])
        insetRotRLabel.text = String.format("RGB Rot R  %.3f", agxParams.rgbRotation[0])
        insetRotGSlider.progress = rotToProgress(agxParams.rgbRotation[1])
        insetRotGLabel.text = String.format("RGB Rot G  %.3f", agxParams.rgbRotation[1])
        insetRotBSlider.progress = rotToProgress(agxParams.rgbRotation[2])
        insetRotBLabel.text = String.format("RGB Rot B  %.3f", agxParams.rgbRotation[2])

        insetPurRSlider.progress = agxParams.purityAttenuation[0].toInt().coerceIn(0, 600)
        insetPurRLabel.text = String.format("Purity R  %.1f", agxParams.purityAttenuation[0])
        insetPurGSlider.progress = agxParams.purityAttenuation[1].toInt().coerceIn(0, 600)
        insetPurGLabel.text = String.format("Purity G  %.1f", agxParams.purityAttenuation[1])
        insetPurBSlider.progress = agxParams.purityAttenuation[2].toInt().coerceIn(0, 600)
        insetPurBLabel.text = String.format("Purity B  %.1f", agxParams.purityAttenuation[2])
    }

    private fun syncOutsetSliders() {
        val rotRange = 0.5236f
        fun rotToProgress(v: Float) = ((v + rotRange) / (rotRange * 2) * 524).toInt().coerceIn(0, 524)
        outsetRotRSlider.progress = rotToProgress(agxParams.reverseRgbRotation[0])
        outsetRotRLabel.text = String.format("Rev Rot R  %.3f", agxParams.reverseRgbRotation[0])
        outsetRotGSlider.progress = rotToProgress(agxParams.reverseRgbRotation[1])
        outsetRotGLabel.text = String.format("Rev Rot G  %.3f", agxParams.reverseRgbRotation[1])
        outsetRotBSlider.progress = rotToProgress(agxParams.reverseRgbRotation[2])
        outsetRotBLabel.text = String.format("Rev Rot B  %.3f", agxParams.reverseRgbRotation[2])

        outsetPurRSlider.progress = agxParams.restorePurity[0].toInt().coerceIn(0, 600)
        outsetPurRLabel.text = String.format("Restore R  %.1f", agxParams.restorePurity[0])
        outsetPurGSlider.progress = agxParams.restorePurity[1].toInt().coerceIn(0, 600)
        outsetPurGLabel.text = String.format("Restore G  %.1f", agxParams.restorePurity[1])
        outsetPurBSlider.progress = agxParams.restorePurity[2].toInt().coerceIn(0, 600)
        outsetPurBLabel.text = String.format("Restore B  %.1f", agxParams.restorePurity[2])
    }

    private fun syncWbSliders() {
        val kelvinEnabled = currentWbMode == WhiteBalanceMode.KELVIN
        kelvinSlider.isEnabled = kelvinEnabled
        tintSlider.isEnabled = kelvinEnabled
        kelvinLabel.alpha = if (kelvinEnabled) 1.0f else 0.4f
        tintLabel.alpha = if (kelvinEnabled) 1.0f else 0.4f
    }

    private fun refreshPresetSpinner() {
        val names = presetManager.getNames()
        @Suppress("UNCHECKED_CAST")
        val adapter = presetSpinner.adapter as? ArrayAdapter<String> ?: return
        adapter.clear()
        adapter.addAll(names)
        adapter.notifyDataSetChanged()
    }

    private fun selectPreset(name: String) {
        val names = presetManager.getNames()
        val idx = names.indexOf(name)
        if (idx >= 0) presetSpinner.setSelection(idx)
    }

    private fun simpleSeekBar(onProgress: (Int) -> Unit) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
            if (fromUser) onProgress(progress)
        }
        override fun onStartTrackingTouch(seekBar: SeekBar) {}
        override fun onStopTrackingTouch(seekBar: SeekBar) {}
    }

    private fun updateFlashUI() {
        flashButton.text = when (currentFlashMode) {
            FlashMode.OFF -> "\u26A1"
            FlashMode.AUTO -> "\u26A1A"
            FlashMode.ON -> "\u26A1!"
            FlashMode.TORCH -> "\uD83D\uDCA1"
        }
    }

    private fun updateWbUI() {
        wbButton.text = when (currentWbMode) {
            WhiteBalanceMode.AUTO -> "WB"
            WhiteBalanceMode.KELVIN -> "WB\u00B0K"
            WhiteBalanceMode.GRAY_CARD -> "WB\u25A1"
        }
    }

    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            initCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
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
            return
        }

        val primary = lensManager.selectPrimary() ?: run {
            Log.e(TAG, "No primary lens found")
            return
        }

        val previewSize = lensManager.getBestPreviewSize(primary)
        buildLensSelectorUI()

        previewRenderer.setPreviewSize(previewSize.width, previewSize.height)
        previewRenderer.start()
        previewRenderer.setSensorOrientation(lensManager.getSensorOrientation(primary))

        uploadAgxUniforms()

        camera2Manager.onFrameAvailable = frameHandler@{ image ->
            if (!cameraReady) return@frameHandler
            val planes = image.planes
            val w = image.width
            val h = image.height

            val yCopy = extractPlane(planes[0].buffer, w, h, planes[0].rowStride, 1)
            val uvWidth = w / 2
            val uvHeight = h / 2
            val uCopy = extractPlane(planes[1].buffer, uvWidth, uvHeight, planes[1].rowStride, planes[1].pixelStride)
            val vCopy = extractPlane(planes[2].buffer, uvWidth, uvHeight, planes[2].rowStride, planes[2].pixelStride)

            previewRenderer.setYuvFrame(yCopy, uCopy, vCopy, w, h)
        }

        camera2Manager.onSessionReady = { width, height ->
            Log.d(TAG, "Camera session ready: ${width}x${height}")
            cameraReady = true
        }

        camera2Manager.onError = { Log.e(TAG, it) }

        camera2Manager.startBackgroundThread()
        camera2Manager.openCamera(primary, previewSize)
    }

    private fun uploadAgxUniforms() {
        val sceneLinearTo709 = when (currentWbMode) {
            WhiteBalanceMode.AUTO -> WhiteBalanceMath.buildAutoSceneLinearTo709(null)
            WhiteBalanceMode.KELVIN -> WhiteBalanceMath.buildSceneLinearTo709(kelvinState.kelvin, kelvinState.tint)
            WhiteBalanceMode.GRAY_CARD -> WhiteBalanceMath.buildAutoSceneLinearTo709(null)
        }

        val insetParams = agxParams.toInsetParams()
        val agx = AgxPrecomputer.compute(insetParams, sceneLinearTo709, whiteLevel = 1023f, blackLevel = 64f)

        previewRenderer.agxSceneLinearTo709 = agx.sceneLinearTo709
        previewRenderer.agxInsetMat = agx.insetMat
        previewRenderer.agxOutsetMat = agx.outsetMat
        previewRenderer.agxToRec2020 = agx.toRec2020
        previewRenderer.agxWhiteLevel = agx.whiteLevel
        previewRenderer.agxBlackLevel = agx.blackLevel
        previewRenderer.agxLogMin = -10f
        previewRenderer.agxLogMax = 6.5f
        previewRenderer.agxLogMidgray = agx.logMidgray
        previewRenderer.agxDisplayMidgray = agx.displayMidgray
        previewRenderer.agxContrast = agxParams.contrast
        previewRenderer.agxToe = agxParams.toe
        previewRenderer.agxShoulder = agxParams.shoulder
    }

    private fun extractPlane(src: java.nio.ByteBuffer, width: Int, height: Int, rowStride: Int, pixelStride: Int): java.nio.ByteBuffer {
        val dst = java.nio.ByteBuffer.allocateDirect(width * height).order(java.nio.ByteOrder.nativeOrder())
        src.position(0)
        if (pixelStride == 1) {
            for (row in 0 until height) {
                val srcOffset = row * rowStride
                src.position(srcOffset)
                src.limit(srcOffset + width)
                dst.position(row * width)
                dst.put(src)
            }
        } else {
            for (row in 0 until height) {
                val srcRowStart = row * rowStride
                val dstRowStart = row * width
                src.position(srcRowStart)
                for (col in 0 until width) {
                    val srcIdx = srcRowStart + col * pixelStride
                    if (srcIdx < src.capacity()) dst.put(dstRowStart + col, src.get(srcIdx))
                }
            }
        }
        dst.position(0)
        src.position(0)
        src.limit(src.capacity())
        return dst
    }

    private fun buildLensSelectorUI() {
        lensSelector.removeAllViews()
        val primaryLens = lensManager.activeLens ?: return
        val btn = TextView(this).apply {
            text = primaryLens.label
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 12f
            setPadding(24, 12, 24, 12)
            setBackgroundColor(0xFF4488FF.toInt())
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { marginStart = 8 }
        }
        lensSelector.addView(btn)
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED && !cameraReady) {
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
