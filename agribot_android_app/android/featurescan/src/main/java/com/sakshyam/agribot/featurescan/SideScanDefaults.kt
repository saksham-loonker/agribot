package com.sakshyam.agribot.featurescan

object SideScanDefaults {
    const val TARGET_FPS = 5
    const val MIN_IMPORTED_DEFAULT_FPS = 3
    const val MAX_IMPORTED_DEFAULT_FPS = 5

    fun nativeDefaultTargetFps(importedFps: Double): Int =
        importedFps.toInt().coerceIn(MIN_IMPORTED_DEFAULT_FPS, MAX_IMPORTED_DEFAULT_FPS)

    fun performanceStatus(targetFps: Int): String = "$targetFps FPS"
}
