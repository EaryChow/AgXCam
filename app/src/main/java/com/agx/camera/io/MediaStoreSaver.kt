package com.agx.camera.io

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MediaStoreSaver(private val context: Context) {

    var lastSavedUri: Uri? = null
        private set

    fun saveJpeg(jpegData: ByteArray, metadata: CaptureMetadata): Uri? {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(metadata.captureWallClockMs))
        val displayName = "AgX_${timestamp}.jpg"

        val relativePath = Environment.DIRECTORY_DCIM + "/AgXCam"

        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.DATE_ADDED, metadata.captureWallClockMs / 1000)
            put(MediaStore.Images.Media.DATE_MODIFIED, System.currentTimeMillis() / 1000)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, relativePath)
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }

        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            ?: return null

        try {
            resolver.openOutputStream(uri)?.use { stream ->
                stream.write(jpegData)
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                contentValues.clear()
                contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(uri, contentValues, null, null)
            }

            Log.d(TAG, "Saved JPEG: $displayName → $uri")
            lastSavedUri = uri
            return uri
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save JPEG: $displayName", e)
            resolver.delete(uri, null, null)
            return null
        }
    }
}

private const val TAG = "MediaStoreSaver"
