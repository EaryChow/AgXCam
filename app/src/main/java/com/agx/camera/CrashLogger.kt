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
    // Large enough to keep a full multi-lens debug session: the per-request
    // readback logging runs at kilobytes per second, and at the old 128 KB
    // cap a session that switches lenses lost every earlier lens within a
    // minute. The same buffer backs crash reports in every build type;
    // 4 M chars (~8 MB) is acceptable for the dual role.
    private const val MAX_BUFFER_CHARS = 4 * 1024 * 1024

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
        writeDownloadsBytes(ctx, filename, text.toByteArray(Charsets.UTF_8), "text/plain")
    }

    /**
     * Writes an opaque blob to Downloads.
     * The content type is supplied by the caller so the platform does not try
     * to interpret or transcode the payload. Returns false when the platform
     * refused the insert, so the caller can tell the user instead of silently
     * losing a capture.
     */
    fun writeDownloadsBytes(
        context: Context,
        filename: String,
        bytes: ByteArray,
        mimeType: String
    ): Boolean {
        val ctx = context.applicationContext
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val resolver = ctx.contentResolver
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, filename)
                    put(MediaStore.Downloads.MIME_TYPE, mimeType)
                }
                val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                if (uri == null) {
                    Log.e("CrashLogger", "MediaStore refused the insert for $filename")
                    return false
                }
                resolver.openOutputStream(uri)?.use { os ->
                    os.write(bytes)
                } ?: run {
                    resolver.delete(uri, null, null)
                    return false
                }
                Log.d("CrashLogger", "Saved ${bytes.size} bytes to Downloads via MediaStore: $filename")
                true
            } else {
                @Suppress("DEPRECATION")
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                if (!dir.exists() && !dir.mkdirs()) return false
                File(dir, filename).writeBytes(bytes)
                Log.d("CrashLogger", "Saved ${bytes.size} bytes to Downloads: ${dir.absolutePath}/$filename")
                true
            }
        } catch (e: Exception) {
            Log.e("CrashLogger", "Failed to save $filename to Downloads: ${e.message}", e)
            false
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
