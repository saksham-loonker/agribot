package com.sakshyam.agribot.featurescan

data class FrontBurstCaptureReadiness(
    val canCapture: Boolean,
    val status: String,
    val error: String?,
)

object FrontBurstCapturePolicy {
    const val MIN_FRAME_COUNT = 5
    const val TARGET_FRAME_COUNT = 7
    const val MAX_DETECTOR_CANDIDATES_PER_FRAME = 32
    const val MAX_CLASSIFIED_CANDIDATES_PER_FRAME = 4

    fun evaluate(frameCount: Int, distinctTimestampCount: Int = frameCount): FrontBurstCaptureReadiness =
        if (frameCount >= MIN_FRAME_COUNT && distinctTimestampCount >= FrontBurstConsensus.MIN_DISTINCT_FRAMES) {
            FrontBurstCaptureReadiness(
                canCapture = true,
                status = "Capture ready ($frameCount/$TARGET_FRAME_COUNT frames)",
                error = null,
            )
        } else if (frameCount >= MIN_FRAME_COUNT) {
            FrontBurstCaptureReadiness(
                canCapture = false,
                status = "Waiting for distinct camera frames ($distinctTimestampCount)",
                error = "Wait for at least ${FrontBurstConsensus.MIN_DISTINCT_FRAMES} distinct camera frames",
            )
        } else {
            FrontBurstCaptureReadiness(
                canCapture = false,
                status = "Collecting burst frames ($frameCount/$TARGET_FRAME_COUNT)",
                error = "Wait for at least $MIN_FRAME_COUNT Front Overview frames before capture",
            )
        }
}
