import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.agx.camera"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.agx.camera"
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        buildConfigField("boolean", "AGX_ENABLE_YUV_FALLBACK", "true")
        // Permanent synthetic-Bayer harness (DIFF oracle). Off for normal builds;
        // flip to "true" to run SYNTH/DIFF on the 192x160 fixed-seed sensor.
        buildConfigField("boolean", "AGX_SYNTHETIC_BAYER", "false")
    }

    signingConfigs {
        create("release") {
            val props = Properties()
            val f = rootProject.file("keystore.properties")
            if (f.exists()) props.load(f.inputStream())
            storeFile = rootProject.file("release.keystore")
            storePassword = props.getProperty("storePassword", "")
            keyAlias = props.getProperty("keyAlias", "")
            keyPassword = props.getProperty("keyPassword", "")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            buildConfigField("boolean", "AGX_ENABLE_YUV_FALLBACK", "false")
            buildConfigField("boolean", "AGX_SYNTHETIC_BAYER", "false")
            signingConfig = signingConfigs.getByName("release")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
        // Forward -Dcapdump.dir into the test JVM so the device-capture replay
        // tests can be pointed at a local dump via: gradlew ... -Dcapdump.dir=<dir>.
        // No project-internal default path; when absent the replay tests skip.
        unitTests.all { t ->
            (t as Test).systemProperty("capdump.dir", System.getProperty("capdump.dir") ?: "")
        }
    }
    buildFeatures {
        buildConfig = true
    }

    if (project.hasProperty("enableNativeJpeg")) {
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
    }
}

dependencies {
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)
    implementation(libs.androidx.lifecycle.runtime.ktx)

    testImplementation(libs.junit)
    testImplementation("org.mockito:mockito-core:5.11.0")
    testImplementation("com.google.code.gson:gson:2.10.1")
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}

/**
 * Slow-test selection.
 *
 * JUnit 4 has no @Tag, so "slow" is a marker interface
 * (com.agx.camera.SlowTests) applied with @Category. These are the suites that
 * run full production-grid mirrors over megabytes of synthetic or captured
 * data - correct but slow, and they dominate the headless run.
 *
 *   gradlew testDebugUnitTest                          -> everything
 *   gradlew testDebugUnitTest -Dtest.categories=fast    -> skip SlowTests
 *   gradlew testDebugUnitTest -Dtest.categories=slow    -> only SlowTests
 *
 * Unset runs everything, so a plain invocation never silently drops coverage.
 * Selecting "slow" fails the task when nothing matches, because an empty run
 * that reports success is indistinguishable from a green one.
 */
val SLOW_CATEGORY = "com.agx.camera.SlowTests"
val testCategories = providers.systemProperty("test.categories").orNull?.trim()?.lowercase()

if (testCategories != null && testCategories.isNotEmpty() && testCategories != "slow" && testCategories != "fast") {
    logger.warn("unknown -Dtest.categories='$testCategories' (expected slow, fast or unset); running all tests")
}

tasks.withType<Test>().configureEach {
    if (testCategories == "slow") {
        // includeCategories, not filter.includeTestsMatching: the latter matches
        // class/method name patterns and silently ignores @Category markers.
        useJUnit { includeCategories(SLOW_CATEGORY) }
        filter.isFailOnNoMatchingTests = true
    } else if (testCategories == "fast") {
        useJUnit { excludeCategories(SLOW_CATEGORY) }
    }
}
