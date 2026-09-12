package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.AGRIBOT_LABEL_ORDER
import com.sakshyam.agribot.domain.model.ModelManifest
import com.sakshyam.agribot.domain.model.ScanConstants
import com.sakshyam.agribot.domain.model.ValidationResult

object ModelManifestValidator {
    val expectedLabels: List<String> = AGRIBOT_LABEL_ORDER.map { it.displayName }

    fun validate(manifest: ModelManifest, labels: List<String>): ValidationResult {
        val errors = buildList {
            if (manifest.minAndroidSdk > 26) add("Manifest requires min SDK ${manifest.minAndroidSdk}; v1 target is 26")
            if (!manifest.cpuDefault) add("CPU default must be true")
            if (manifest.requiresNetwork) add("Native v1 model bundle must not require network")
            if (labels != expectedLabels) add("Label order mismatch: expected $expectedLabels, got $labels")
            if (!manifest.classifier.source.endsWith("classifier_deploy_fastcrop.pt")) {
                add("Side Scan classifier source must come from classifier_deploy_fastcrop.pt for v1")
            }
            if (manifest.classifier.confidenceThreshold != ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD) {
                add("Classifier confidence threshold must be ${ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD}")
            }
            if (manifest.classifier.highConfidenceThreshold != ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD) {
                add("Classifier high confidence threshold must be ${ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD}")
            }
            if (!manifest.classifier.confidenceTemperature.isFinite() || manifest.classifier.confidenceTemperature <= 0f) {
                add("Classifier confidence temperature must be finite and positive")
            }
            if (manifest.classifier.inputWidth <= 0 || manifest.classifier.inputHeight <= 0) add("Classifier input dimensions must be positive")
            manifest.detector?.let { detector ->
                if (!detector.source.endsWith("detector_nano_256.pt")) {
                    add("Front Overview detector source must come from detector_nano_256.pt for v1")
                }
                if ((detector.outputCandidates ?: 0) < 2) {
                    add("Front Overview detector output candidate count must be >= 2")
                }
            }
        }
        return ValidationResult(errors.isEmpty(), errors)
    }
}
