package com.agx.camera

import org.junit.Assume.assumeTrue
import java.io.File

/**
 * Shared device-capture-dump gate for the headless replay tests.
 *
 * The dumps are machine-local and gitignored, so a test that requires one has
 * to SKIP rather than fail - otherwise a fresh clone cannot run the unit suite
 * at all. The cost of skipping is that a green run can silently have executed
 * nothing, which is why this lives in one place and always names the missing
 * files and the property that points at them.
 *
 * Gradle forwards `-Dcapdump.dir` into the test JVM (see testOptions in
 * app/build.gradle.kts); there is no project-internal default path on purpose,
 * so a stale default can never masquerade as a real capture.
 */
object DeviceDumpGate {

    /** Property name; the directory is a machine-local, gitignored capture. */
    const val PROPERTY = "capdump.dir"

    /** Resolved dump directory, or an unusable placeholder when unset. */
    fun dir(): File {
        System.getProperty(PROPERTY)?.let { p ->
            val f = File(p)
            if (f.isDirectory) return f
        }
        return File("")
    }

    /**
     * Skips the calling test unless every named file is present. Returns the
     * directory on success so callers can read straight from it.
     */
    fun require(vararg required: String): File {
        val d = dir()
        val missing = required.filter { !File(d, it).exists() }
        assumeTrue(
            "device capture dump missing - skipping (produce it or set -D$PROPERTY=<dir>): " +
                "$d ${missing.joinToString { "[$it]" }}",
            missing.isEmpty()
        )
        return d
    }
}