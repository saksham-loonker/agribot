package com.sakshyam.agribot.domain.scan

import com.sakshyam.agribot.domain.model.BoundingBox
import kotlinx.coroutines.flow.StateFlow

/** Upright camera frame, packed ARGB pixels. */
class VisionFrame(val width: Int, val height: Int, val argb: IntArray, val timestampNanos: Long) {
    init {
        require(width > 0 && height > 0 && argb.size == width * height) { "bad frame" }
    }
}

/** One detected and classified leaf. */
class AnalyzedLeaf(
    val box: BoundingBox,
    val detectorScore: Float,
    /** Calibrated class probabilities in label order. */
    val probabilities: FloatArray,
    val labelIndex: Int,
)

class FrameAnalysis(
    val frameWidth: Int,
    val frameHeight: Int,
    val leaves: List<AnalyzedLeaf>,
    /** Leaves detected (may exceed [leaves] when only the best few are classified). */
    val detectedLeaves: Int,
    val detectMs: Double,
    val classifyMs: Double,
)

/** What the loaded model bundle can recognise. */
data class VisionModelInfo(
    val bundleId: String,
    /** Model label keys in output order, e.g. "Early_blight", ..., "Other". */
    val labels: List<String>,
    val healthyIndex: Int,
    val otherIndex: Int,
    val classifierMembers: Int,
)

sealed interface VisionStatus {
    data object Loading : VisionStatus
    data class Ready(val info: VisionModelInfo) : VisionStatus
    data class Failed(val reason: String) : VisionStatus
}

interface LeafVision {
    val status: StateFlow<VisionStatus>

    /** Starts (or retries) loading the model bundle; safe to call repeatedly. */
    fun warmUp()

    /** Detect leaves and classify up to [maxLeaves] of the most confident ones. Never call on Main. */
    suspend fun analyze(frame: VisionFrame, maxLeaves: Int): FrameAnalysis
}
