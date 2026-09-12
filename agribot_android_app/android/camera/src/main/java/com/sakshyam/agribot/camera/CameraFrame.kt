package com.sakshyam.agribot.camera

data class CameraFrame(
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val timestampNanos: Long,
    val data: ByteArray,
)

data class CameraDiagnostics(
    val cameraId: String?,
    val analysisWidth: Int?,
    val analysisHeight: Int?,
    val targetFps: Int,
    val lateFrames: Int,
)
