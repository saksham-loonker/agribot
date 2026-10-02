package com.sakshyam.agribot.featurescan.scan

import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock

/**
 * Slows analysis down as the phone heats up (long field walks in the sun), instead of letting the
 * system throttle or kill the app: full speed normally, ~4 fps when MODERATE, ~1.5 fps when SEVERE,
 * ~0.7 fps beyond that.
 */
class FramePacer(context: Context) {
    private val power = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    @Volatile private var lastAccepted = 0L

    fun minIntervalMs(): Long {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return 0L
        return when (power.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE, PowerManager.THERMAL_STATUS_LIGHT -> 0L
            PowerManager.THERMAL_STATUS_MODERATE -> 250L
            PowerManager.THERMAL_STATUS_SEVERE -> 700L
            else -> 1500L
        }
    }

    /** True if a frame may be analysed now; records the acceptance time. */
    fun tryAcquire(): Boolean {
        val now = SystemClock.elapsedRealtime()
        if (now - lastAccepted < minIntervalMs()) return false
        lastAccepted = now
        return true
    }
}
