package com.sakshyam.agribot.camera

class FrameSampler(
    targetFps: Int,
) {
    private var minIntervalNanos = 1_000_000_000L / targetFps.coerceAtLeast(1)
    private var lastAcceptedNanos = Long.MIN_VALUE

    fun updateTargetFps(targetFps: Int) {
        minIntervalNanos = 1_000_000_000L / targetFps.coerceAtLeast(1)
    }

    fun shouldAccept(timestampNanos: Long): Boolean {
        if (timestampNanos < 0L) return false
        if (lastAcceptedNanos == Long.MIN_VALUE ||
            (timestampNanos >= lastAcceptedNanos && timestampNanos - lastAcceptedNanos >= minIntervalNanos)
        ) {
            lastAcceptedNanos = timestampNanos
            return true
        }
        return false
    }
}
