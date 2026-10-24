package com.sakshyam.agribot.domain.scan

import com.sakshyam.agribot.domain.logic.FrameQualityAnalyzer
import com.sakshyam.agribot.domain.logic.FrameQualityAssessment

data class CheckedLeafAnalysis(val quality: FrameQualityAssessment, val analysis: FrameAnalysis?)

/** Block detail-free views. Dim/soft but structured views retain the quality warning and may be analyzed. */
suspend fun LeafVision.analyzeUsable(frame: VisionFrame, maxLeaves: Int): CheckedLeafAnalysis {
    require(maxLeaves >= 0) { "maxLeaves must be non-negative" }
    val quality = FrameQualityAnalyzer.evaluate(frame.width, frame.height, frame.argb)
    val usable = quality.contrast.isFinite() && quality.sharpness.isFinite() &&
        (quality.contrast >= 0.015f || quality.sharpness >= 0.012f)
    return CheckedLeafAnalysis(quality, if (usable) analyze(frame, maxLeaves) else null)
}
