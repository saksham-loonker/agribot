package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.EvidenceRetentionSnapshot
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.ScanConstants

class EvidenceRetentionPolicy(
    private val maxBytesPerRun: Long = ScanConstants.MAX_STORAGE_BYTES,
    private val maxFramesPerRun: Int = ScanConstants.MAX_EVIDENCE_FRAMES,
) {
    fun shouldCapture(decision: RecordedDecision, snapshot: EvidenceRetentionSnapshot, evidenceEnabled: Boolean = true): EvidencePlan {
        if (!evidenceEnabled) return EvidencePlan(false, "disabled")
        if (snapshot.bytesUsed >= maxBytesPerRun || snapshot.framesUsed >= maxFramesPerRun) {
            return EvidencePlan(false, "storage_cap_reached")
        }
        val required = decision.manualOverride ||
            decision.status == DecisionStatus.UNCERTAIN ||
            decision.action == PlantHealthAction.INSPECT_OR_TREAT
        return if (required) {
            EvidencePlan(true, "capture_pending")
        } else {
            EvidencePlan(false, "not_required")
        }
    }

    fun shouldCaptureManualSnapshot(snapshot: EvidenceRetentionSnapshot, evidenceEnabled: Boolean = true): EvidencePlan {
        if (!evidenceEnabled) return EvidencePlan(false, "disabled")
        if (snapshot.bytesUsed >= maxBytesPerRun || snapshot.framesUsed >= maxFramesPerRun) {
            return EvidencePlan(false, "storage_cap_reached")
        }
        return EvidencePlan(true, "capture_pending")
    }
}

data class EvidencePlan(
    val capture: Boolean,
    val status: String,
)
