package com.agx.camera

import android.content.Context
import com.agx.camera.color.AgxParams
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

class PresetManager(private val context: Context) {

    data class Preset(
        val name: String,
        val createdAt: String = "",
        val agxParams: AgxParams = AgxParams()
    )

    private val file: File = File(context.filesDir, "presets.json")

    init {
        if (!file.exists()) {
            writeRoot(JSONObject().apply {
                put("version", 1)
                put("user_presets", JSONArray())
            })
        }
    }

    fun getNames(): List<String> {
        val names = mutableListOf(PRESET_DEFAULT)
        val root = readRoot()
        val arr = root.optJSONArray("user_presets") ?: return names
        for (i in 0 until arr.length()) {
            names.add(arr.getJSONObject(i).optString("name", "Unnamed"))
        }
        return names
    }

    fun load(name: String): Preset? {
        if (name == PRESET_DEFAULT) return defaultPreset()
        val root = readRoot()
        val arr = root.optJSONArray("user_presets") ?: return null
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            if (obj.optString("name") == name) {
                return parsePreset(obj)
            }
        }
        return null
    }

    fun save(preset: Preset) {
        val root = readRoot()
        val arr = root.optJSONArray("user_presets") ?: JSONArray()

        var found = false
        for (i in 0 until arr.length()) {
            if (arr.getJSONObject(i).optString("name") == preset.name) {
                arr.put(i, serializePreset(preset))
                found = true
                break
            }
        }
        if (!found) {
            arr.put(serializePreset(preset))
        }

        root.put("user_presets", arr)
        writeRoot(root)
    }

    fun rename(oldName: String, newName: String) {
        if (oldName == PRESET_DEFAULT || oldName == newName) return
        val root = readRoot()
        val arr = root.optJSONArray("user_presets") ?: return
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            if (obj.optString("name") == oldName) {
                obj.put("name", newName)
                arr.put(i, obj)
                break
            }
        }
        root.put("user_presets", arr)
        writeRoot(root)
    }

    fun delete(name: String) {
        if (name == PRESET_DEFAULT) return
        val root = readRoot()
        val arr = root.optJSONArray("user_presets") ?: return
        val newArr = JSONArray()
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            if (obj.optString("name") != name) {
                newArr.put(obj)
            }
        }
        root.put("user_presets", newArr)
        writeRoot(root)
    }

    fun ensureDefault() {
        // Default is built-in, never stored in user_presets
    }

    private fun defaultPreset() = Preset(
        name = PRESET_DEFAULT,
        createdAt = "",
        agxParams = AgxParams()
    )

    private fun parsePreset(obj: JSONObject): Preset {
        val p = obj.optJSONObject("params") ?: JSONObject()
        return Preset(
            name = obj.optString("name", "Unnamed"),
            createdAt = obj.optString("created_at", ""),
            agxParams = AgxParams(
                contrast = p.optDouble("general_contrast", 3.0).toFloat(),
                toe = p.optDouble("toe_contrast", 1.5).toFloat(),
                shoulder = p.optDouble("shoulder_contrast", 2.5).toFloat(),
                middleGray = p.optDouble("middle_gray", 18.0).toFloat(),
                rotation = jsonArrayToFloatArray(p.optJSONArray("rotation"), floatArrayOf(0.3f, -0.3f, -0.053f)),
                attenuation = jsonArrayToFloatArray(p.optJSONArray("attenuation"), floatArrayOf(50.0f, 50.0f, 30.0f)),
                reverseRotation = jsonArrayToFloatArray(p.optJSONArray("reverse_rotation"), floatArrayOf(0.3f, -0.3f, -0.053f)),
                purityBoost = jsonArrayToFloatArray(p.optJSONArray("purity_boost"), floatArrayOf(10.0f, 30.0f, 4.0f)),
                tintingScale = p.optDouble("tinting_scale", 0.0).toFloat(),
                tintingHue = p.optDouble("tinting_hue", 0.0).toFloat(),
                vibrance = p.optDouble("vibrance", 0.6).toFloat(),
                useRotationForReverse = p.optBoolean("use_rotation_for_reverse", true),
                useAttenuationForBoost = p.optBoolean("use_attenuation_for_boost", false)
            )
        )
    }

    private fun serializePreset(preset: Preset): JSONObject {
        val p = preset.agxParams
        val params = JSONObject().apply {
            put("general_contrast", p.contrast.toDouble())
            put("toe_contrast", p.toe.toDouble())
            put("shoulder_contrast", p.shoulder.toDouble())
            put("middle_gray", p.middleGray.toDouble())
            put("rotation", floatArrayToJson(p.rotation))
            put("attenuation", floatArrayToJson(p.attenuation))
            put("reverse_rotation", floatArrayToJson(p.reverseRotation))
            put("purity_boost", floatArrayToJson(p.purityBoost))
            put("tinting_scale", p.tintingScale.toDouble())
            put("tinting_hue", p.tintingHue.toDouble())
            put("vibrance", p.vibrance.toDouble())
            put("use_rotation_for_reverse", p.useRotationForReverse)
            put("use_attenuation_for_boost", p.useAttenuationForBoost)
        }
        return JSONObject().apply {
            put("name", preset.name)
            put("created_at", preset.createdAt.ifEmpty { nowIso() })
            put("params", params)
        }
    }

    private fun floatArrayToJson(arr: FloatArray): JSONArray {
        return JSONArray().apply { arr.forEach { put(it.toDouble()) } }
    }

    private fun jsonArrayToFloatArray(arr: JSONArray?, fallback: FloatArray): FloatArray {
        if (arr == null || arr.length() != 3) return fallback.copyOf()
        return floatArrayOf(arr.getDouble(0).toFloat(), arr.getDouble(1).toFloat(), arr.getDouble(2).toFloat())
    }

    private fun nowIso(): String {
        val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
        sdf.timeZone = TimeZone.getTimeZone("UTC")
        return sdf.format(Date())
    }

    private fun readRoot(): JSONObject {
        return try {
            JSONObject(file.readText())
        } catch (_: Exception) {
            JSONObject().apply {
                put("version", 1)
                put("user_presets", JSONArray())
            }
        }
    }

    private fun writeRoot(root: JSONObject) {
        file.writeText(root.toString(2))
    }

    companion object {
        const val PRESET_DEFAULT = "Default"
    }
}
