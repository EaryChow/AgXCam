package com.agx.camera.phase1.week7

import org.junit.Assert.*
import org.junit.Test
import java.io.File

class ReleaseBuildArtifactTest {

    private fun readBuildGradle(): String {
        val candidates = listOf(
            File("app/build.gradle.kts"),
            File("D:/coding_projects/AgXCam/app/build.gradle.kts"),
            File(System.getProperty("user.dir") ?: "", "app/build.gradle.kts")
        )
        val file = candidates.firstOrNull { it.exists() }
            ?: error("build.gradle.kts not found in any of: ${candidates.map { it.absolutePath }}")
        return file.readText()
    }

    @Test
    fun debugBuildType_hasYuvFallbackTrue() {
        val content = readBuildGradle()

        val defaultConfigSection = content.substringBefore("buildTypes")
        assertTrue(
            "defaultConfig should declare AGX_ENABLE_YUV_FALLBACK = true",
            defaultConfigSection.contains("AGX_ENABLE_YUV_FALLBACK") &&
                    defaultConfigSection.contains("\"true\"")
        )
    }

    @Test
    fun releaseBuildType_hasYuvFallbackFalse() {
        val content = readBuildGradle()

        val releaseSection = content.substringAfter("release {")
            .substringBefore("}")

        assertTrue(
            "Release buildType should declare AGX_ENABLE_YUV_FALLBACK = false",
            releaseSection.contains("AGX_ENABLE_YUV_FALLBACK") &&
                    releaseSection.contains("\"false\"")
        )
    }

    @Test
    fun releaseBuildType_hasMinifyDisabled() {
        val content = readBuildGradle()
        val releaseSection = content.substringAfter("release {")
            .substringBefore("}")

        assertTrue(
            "Release buildType should have isMinifyEnabled = false",
            releaseSection.contains("isMinifyEnabled = false")
        )
    }

    @Test
    fun buildConfigFeature_enabled() {
        val content = readBuildGradle()
        assertTrue(
            "buildFeatures.buildConfig should be true",
            content.contains("buildConfig = true")
        )
    }

    @Test
    fun testOptions_returnDefaultValues() {
        val content = readBuildGradle()
        assertTrue(
            "unitTests.isReturnDefaultValues should be true",
            content.contains("unitTests.isReturnDefaultValues = true")
        )
    }

    @Test
    fun packageIsCorrect() {
        val content = readBuildGradle()
        assertTrue(
            "applicationId should be com.agx.camera",
            content.contains("applicationId = \"com.agx.camera\"")
        )
        assertTrue(
            "namespace should be com.agx.camera",
            content.contains("namespace = \"com.agx.camera\"")
        )
    }

    @Test
    fun minSdkVersion_is30() {
        val content = readBuildGradle()
        assertTrue(
            "minSdk should be 30",
            content.contains("minSdk = 30")
        )
    }

    @Test
    fun debugYuvFallbackTrue_releaseYuvFallbackFalse_correctness() {
        val content = readBuildGradle()

        val defaultConfigMatch = Regex(
            """buildConfigField\("boolean",\s*"AGX_ENABLE_YUV_FALLBACK",\s*"(true|false)"\)"""
        ).find(content.substringBefore("buildTypes"))

        val releaseMatch = Regex(
            """buildConfigField\("boolean",\s*"AGX_ENABLE_YUV_FALLBACK",\s*"(true|false)"\)"""
        ).find(content.substringAfter("release {").substringBefore("}"))

        assertTrue("defaultConfig should have YUV fallback field", defaultConfigMatch != null)
        assertTrue("release should have YUV fallback field", releaseMatch != null)

        assertEquals("true", defaultConfigMatch!!.groupValues[1])
        assertEquals("false", releaseMatch!!.groupValues[1])
    }

    @Test
    fun buildConfigField_count() {
        val content = readBuildGradle()
        val count = content.split("buildConfigField").size - 1
        assertTrue(
            "Should have at least 2 buildConfigField declarations (default + release)",
            count >= 2
        )
    }
}
