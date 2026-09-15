package com.agx.camera

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.*

object CrashLogger {

    private var appContext: Context? = null
    private val lock = Any()
    private val dateFormat = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)
    private val buffer = StringBuilder()
    private const val MAX_BUFFER_CHARS = 128 * 1024

    // Large out-of-band blocks (e.g. the S3 pack texel dump) that must not go
    // through the ring buffer. Saved alongside the log by the *Downloads writers.
    private val sections = LinkedHashMap<String, String>()
    private val sectionOrder = ArrayList<String>()

    fun init(context: Context) {
        appContext = context.applicationContext
        log("CrashLogger", "=== Session started ===")
    }

    fun log(tag: String, msg: String) {
        val line = "${dateFormat.format(Date())} $tag: $msg\n"
        synchronized(lock) {
            try {
                if (buffer.length > MAX_BUFFER_CHARS) {
                    val kept = buffer.toString().takeLast(MAX_BUFFER_CHARS / 2)
                    buffer.clear()
                    buffer.append(kept)
                }
                buffer.append(line)
            } catch (_: Exception) {}
        }
    }

    fun logException(tag: String, t: Throwable) {
        val sw = StringWriter()
        t.printStackTrace(PrintWriter(sw))
        log(tag, "EXCEPTION: ${t.javaClass.simpleName}: ${t.message}")
        log(tag, sw.toString())
    }

    fun installUncaughtHandler(original: Thread.UncaughtExceptionHandler?) {
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                log("CRASH", "Uncaught exception on thread '${thread.name}'")
                logException("CRASH", throwable)
                flushToFile()
                saveToDownloads()
            } catch (_: Exception) {}
            original?.uncaughtException(thread, throwable)
        }
    }

    private fun flushToFile() {
        val ctx = appContext ?: return
        synchronized(lock) {
            try {
                val f = File(ctx.filesDir, "crash_log.txt")
                f.writeText(buffer.toString())
            } catch (_: Exception) {}
        }
    }

    fun saveToDownloads() {
        val ctx = appContext ?: return
        val text = readLogWithSections()
        if (text.isBlank()) return
        writeDownloadsFile(ctx, "agxcam_crash_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.txt", text)
        clearSectionsForSave()
    }

    private fun clearSectionsForSave() {
        synchronized(lock) {
            sections.clear()
            sectionOrder.clear()
        }
    }

    private fun writeDownloadsFile(ctx: Context, filename: String, text: String) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val resolver = ctx.contentResolver
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, filename)
                    put(MediaStore.Downloads.MIME_TYPE, "text/plain")
                }
                val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                uri?.let {
                    resolver.openOutputStream(it)?.use { os ->
                        os.write(text.toByteArray())
                    }
                }
                Log.d("CrashLogger", "Saved crash log to Downloads via MediaStore: $filename")
            } else {
                @Suppress("DEPRECATION")
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                if (dir.exists() || dir.mkdirs()) {
                    File(dir, filename).writeText(text)
                    Log.d("CrashLogger", "Saved crash log to Downloads: ${dir.absolutePath}/$filename")
                }
            }
        } catch (e: Exception) {
            Log.e("CrashLogger", "Failed to save to Downloads: ${e.message}", e)
        }
    }

    fun readLog(): String {
        synchronized(lock) {
            return buffer.toString()
        }
    }

    fun clearLog() {
        synchronized(lock) {
            buffer.clear()
        }
    }

    fun setSection(name: String, text: String) {
        synchronized(lock) {
            if (!sections.containsKey(name)) sectionOrder.add(name)
            sections[name] = text
        }
    }

    fun clearSection(name: String) {
        synchronized(lock) {
            sections.remove(name)
            sectionOrder.remove(name)
        }
    }

    private fun readLogWithSections(): String {
        synchronized(lock) {
            if (sections.isEmpty()) return buffer.toString()
            val sb = StringBuilder(buffer.length + 1024)
            sb.append(buffer)
            for (name in sectionOrder) {
                val text = sections[name] ?: continue
                sb.append('\n').append("===== SECTION: $name =====\n")
                sb.append(text)
            }
            return sb.toString()
        }
    }

    fun saveDebugLogToDownloads(context: Context) {
        val text = readLogWithSections()
        if (text.isBlank()) return
        writeDownloadsFile(context, "agxcam_debug_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.txt", text)
        clearSectionsForSave()
    }
}
