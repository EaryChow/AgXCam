package com.agx.camera

/**
 * JUnit 4 category marker for the slow tests.
 *
 * JUnit 4 has no `@Tag`, so "slow" is expressed as a marker interface on the
 * class and selected/excluded from Gradle with `-Dtest.categories=`. Two
 * reasons this exists rather than a naming convention:
 *
 *  - the headless unit run is the release gate's correctness half, so it has to
 *    stay fast enough to actually be run;
 *  - the device-dump replay tests skip rather than fail when the dump is
 *    absent, which means a bare `testDebugUnitTest` reports a green run that
 *    silently executed nothing. Marking them lets the report say which tests
 *    were excluded instead.
 *
 * Usage (quote the property in PowerShell):
 *   everything:           gradlew testDebugUnitTest
 *   without slow tests:   gradlew testDebugUnitTest "-Dtest.categories=fast"
 *   slow only:            gradlew testDebugUnitTest "-Dtest.categories=slow"
 */
interface SlowTests