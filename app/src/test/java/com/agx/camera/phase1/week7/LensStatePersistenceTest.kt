package com.agx.camera.phase1.week7

import com.agx.camera.camera.LensState
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.Assert.*
import org.junit.Test

class LensStatePersistenceTest {

    private val gson = Gson()

    private fun lensStateToJson(state: LensState): String {
        return JsonObject().apply {
            addProperty("zoomFactor", state.zoomFactor)
            addProperty("zoomCenterX", state.zoomCenterX)
            addProperty("zoomCenterY", state.zoomCenterY)
            addProperty("wbModeOrdinal", state.wbModeOrdinal)
            addProperty("kelvin", state.kelvin)
            addProperty("kelvinTint", state.kelvinTint)
            addProperty("flashModeOrdinal", state.flashModeOrdinal)
        }.toString()
    }

    private fun jsonToLensState(jsonStr: String): LensState {
        val json = JsonParser.parseString(jsonStr).asJsonObject
        return LensState(
            zoomFactor = json.get("zoomFactor")?.asFloat ?: 1.0f,
            zoomCenterX = json.get("zoomCenterX")?.asFloat ?: 0.5f,
            zoomCenterY = json.get("zoomCenterY")?.asFloat ?: 0.5f,
            wbModeOrdinal = json.get("wbModeOrdinal")?.asInt ?: 0,
            kelvin = json.get("kelvin")?.asFloat ?: 5500f,
            kelvinTint = json.get("kelvinTint")?.asFloat ?: 0f,
            flashModeOrdinal = json.get("flashModeOrdinal")?.asInt ?: 0
        )
    }

    @Test
    fun roundTrip_defaultValues() {
        val original = LensState()
        val json = lensStateToJson(original)
        val restored = jsonToLensState(json)

        assertEquals(original.zoomFactor, restored.zoomFactor, 0.001f)
        assertEquals(original.zoomCenterX, restored.zoomCenterX, 0.001f)
        assertEquals(original.zoomCenterY, restored.zoomCenterY, 0.001f)
        assertEquals(original.wbModeOrdinal, restored.wbModeOrdinal)
        assertEquals(original.kelvin, restored.kelvin, 0.001f)
        assertEquals(original.kelvinTint, restored.kelvinTint, 0.001f)
        assertEquals(original.flashModeOrdinal, restored.flashModeOrdinal)
    }

    @Test
    fun roundTrip_customValues() {
        val original = LensState(
            zoomFactor = 2.5f,
            zoomCenterX = 0.3f,
            zoomCenterY = 0.7f,
            wbModeOrdinal = 2,
            kelvin = 3200f,
            kelvinTint = 15f,
            flashModeOrdinal = 1
        )
        val json = lensStateToJson(original)
        val restored = jsonToLensState(json)

        assertEquals(2.5f, restored.zoomFactor, 0.001f)
        assertEquals(0.3f, restored.zoomCenterX, 0.001f)
        assertEquals(0.7f, restored.zoomCenterY, 0.001f)
        assertEquals(2, restored.wbModeOrdinal)
        assertEquals(3200f, restored.kelvin, 0.001f)
        assertEquals(15f, restored.kelvinTint, 0.001f)
        assertEquals(1, restored.flashModeOrdinal)
    }

    @Test
    fun roundTrip_zeroKelvin() {
        val original = LensState(kelvin = 0f, kelvinTint = -100f)
        val json = lensStateToJson(original)
        val restored = jsonToLensState(json)

        assertEquals(0f, restored.kelvin, 0.001f)
        assertEquals(-100f, restored.kelvinTint, 0.001f)
    }

    @Test
    fun roundTrip_maxValues() {
        val original = LensState(
            zoomFactor = 10.0f,
            zoomCenterX = 1.0f,
            zoomCenterY = 1.0f,
            wbModeOrdinal = 10,
            kelvin = 15000f,
            kelvinTint = 200f,
            flashModeOrdinal = 3
        )
        val json = lensStateToJson(original)
        val restored = jsonToLensState(json)

        assertEquals(10.0f, restored.zoomFactor, 0.001f)
        assertEquals(1.0f, restored.zoomCenterX, 0.001f)
        assertEquals(1.0f, restored.zoomCenterY, 0.001f)
        assertEquals(10, restored.wbModeOrdinal)
        assertEquals(15000f, restored.kelvin, 0.001f)
        assertEquals(200f, restored.kelvinTint, 0.001f)
        assertEquals(3, restored.flashModeOrdinal)
    }

    @Test
    fun jsonParsing_missingFields_usesDefaults() {
        val json = "{}"
        val restored = jsonToLensState(json)

        assertEquals(1.0f, restored.zoomFactor, 0.001f)
        assertEquals(0.5f, restored.zoomCenterX, 0.001f)
        assertEquals(0.5f, restored.zoomCenterY, 0.001f)
        assertEquals(0, restored.wbModeOrdinal)
        assertEquals(5500f, restored.kelvin, 0.001f)
        assertEquals(0f, restored.kelvinTint, 0.001f)
        assertEquals(0, restored.flashModeOrdinal)
    }

    @Test
    fun jsonParsing_partialFields_usesDefaultsForMissing() {
        val json = """{"zoomFactor": 3.0, "kelvin": 2700}"""
        val restored = jsonToLensState(json)

        assertEquals(3.0f, restored.zoomFactor, 0.001f)
        assertEquals(2700f, restored.kelvin, 0.001f)
        assertEquals(0.5f, restored.zoomCenterX, 0.001f)
    }

    @Test
    fun jsonParsing_extraFields_ignored() {
        val original = LensState(zoomFactor = 1.5f)
        val json = lensStateToJson(original).replace(
            "}",
            """, "unknown_field": 999, "extra": "data"}"""
        )
        val restored = jsonToLensState(json)

        assertEquals(1.5f, restored.zoomFactor, 0.001f)
    }

    @Test
    fun jsonIsValidJSONObject() {
        val state = LensState(zoomFactor = 2.0f, kelvin = 4000f)
        val json = lensStateToJson(state)
        val parsed = JsonParser.parseString(json).asJsonObject

        assertTrue(parsed.has("zoomFactor"))
        assertTrue(parsed.has("kelvin"))
        assertTrue(parsed.has("zoomCenterX"))
        assertTrue(parsed.has("wbModeOrdinal"))
        assertTrue(parsed.has("flashModeOrdinal"))
    }

    @Test
    fun lensState_equalityStructural() {
        val a = LensState(zoomFactor = 2.0f, kelvin = 3200f)
        val b = LensState(zoomFactor = 2.0f, kelvin = 3200f)
        val c = LensState(zoomFactor = 2.0f, kelvin = 5500f)

        assertEquals(a, b)
        assertNotEquals(a, c)
    }

    @Test
    fun lensState_dataClassCopy_preservesUnchangedFields() {
        val original = LensState(
            zoomFactor = 2.0f,
            zoomCenterX = 0.3f,
            zoomCenterY = 0.7f,
            wbModeOrdinal = 2,
            kelvin = 3200f,
            kelvinTint = 15f,
            flashModeOrdinal = 1
        )

        val modified = original.copy(kelvin = 6500f)

        assertEquals(2.0f, modified.zoomFactor, 0.001f)
        assertEquals(0.3f, modified.zoomCenterX, 0.001f)
        assertEquals(0.7f, modified.zoomCenterY, 0.001f)
        assertEquals(2, modified.wbModeOrdinal)
        assertEquals(6500f, modified.kelvin, 0.001f)
        assertEquals(15f, modified.kelvinTint, 0.001f)
        assertEquals(1, modified.flashModeOrdinal)
    }
}
