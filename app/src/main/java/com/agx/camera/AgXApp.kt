package com.agx.camera

import android.app.Application
import android.util.Log

class AgXApp : Application() {

    override fun onCreate() {
        super.onCreate()
        CrashLogger.init(this)
        CrashLogger.installUncaughtHandler(Thread.getDefaultUncaughtExceptionHandler())
        Log.d("AgXApp", "Application created, crash logger active")
    }
}
