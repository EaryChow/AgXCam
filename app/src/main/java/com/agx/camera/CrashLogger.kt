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

    // Where the last successful blob write landed, so a report can name the
    // attachment instead of only its size.
    @Volatile private var lastDownloadsLocation: String? = null

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
     * Trims the captured-frame attachments down to the newest [keep].
     *
     * Each grab is about 15 MB and the file is only written on an explicit
     * arm-then-export, so nothing accumulates on its own - but a debug session
     * that grabs a frame after every change fills the Downloads folder, and a
     * full disk eventually breaks the *next* capture, which is the one being
     * used as evidence. Rotation makes that failure mode unreachable.
     *
     * Scoped to names this file's writer produces and to rows this package owns,
     * so it cannot touch unrelated downloads or another app's files. The oldest
     * go first: recency is what makes a grab worth keeping.
     */
    fun rotateDownloadsBlobs(context: Context, prefix: String, keep: Int) {
        if (keep < 1) return
        val ctx = context.applicationContext
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return
        try {
            val resolver = ctx.contentResolver
            val projection = arrayOf(
                MediaStore.Downloads._ID,
                MediaStore.Downloads.DISPLAY_NAME
            )
            val rows = ArrayList<Pair<Long, String>>()
            resolver.query(
                MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                projection,
                "${MediaStore.Downloads.OWNER_PACKAGE_NAME} = ? AND " +
                    "${MediaStore.Downloads.DISPLAY_NAME} LIKE ?",
                arrayOf(ctx.packageName, "$prefix%"),
                null
            )?.use { c ->
                val idCol = c.getColumnIndexOrThrow(MediaStore.Downloads._ID)
                val nameCol = c.getColumnIndexOrThrow(MediaStore.Downloads.DISPLAY_NAME)
                while (c.moveToNext()) {
                    rows.add(c.getLong(idCol) to c.getString(nameCol))
                }
            }
            if (rows.size <= keep) return
            // Query order is unspecified, so recency comes from the id: a
            // MediaStore row id increases with insertion time.
            rows.sortByDescending { it.first }
            var deleted = 0
            for ((id, name) in rows.drop(keep)) {
                val n = resolver.delete(
                    MediaStore.Downloads.EXTERNAL_CONTENT_URI.buildUpon()
                        .appendQueryParameter("q", "id=$id")
                        .build(),
                    null, null
                )
                if (n > 0) {
                    deleted++
                    log("CrashLogger", "rotateDownloadsBlobs: removed old attachment $name")
                }
            }
            if (deleted > 0) {
                log(
                    "CrashLogger",
                    "rotateDownloadsBlobs: kept newest $keep of ${rows.size} '$prefix' attachments, removed $deleted"
                )
            }
        } catch (e: Exception) {
            // Rotation is housekeeping. A failure here must never be the reason
            // a capture does not get written.
            log("CrashLogger", "rotateDownloadsBlobs failed: ${e.message}")
        }
    }

    /**
     * Writes an opaque blob (for example a serialized .framegrab) to Downloads.
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
                // MediaStore owns the file, so the display name is the only
                // stable handle. Recorded so a report can name the attachment
                // rather than only its size.
                lastDownloadsLocation = "Downloads/$filename"
                Log.d("CrashLogger", "Saved ${bytes.size} bytes to Downloads via MediaStore: $filename")
                true
            } else {
                @Suppress("DEPRECATION")
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                if (!dir.exists() && !dir.mkdirs()) return false
                File(dir, filename).writeBytes(bytes)
                lastDownloadsLocation = "${dir.absolutePath}/$filename"
                Log.d("CrashLogger", "Saved ${bytes.size} bytes to Downloads: ${dir.absolutePath}/$filename")
                true
            }
        } catch (e: Exception) {
            Log.e("CrashLogger", "Failed to save $filename to Downloads: ${e.message}", e)
            false
        }
    }

    /**
     * Where the last successful writeDownloadsBytes landed, or null if nothing
     * has been written or the last attempt failed.
     */
    fun lastDownloadsLocation(): String? = lastDownloadsLocation

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
