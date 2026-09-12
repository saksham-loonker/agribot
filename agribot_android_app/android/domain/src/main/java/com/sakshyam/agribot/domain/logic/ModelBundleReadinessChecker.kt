package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.ModelBundleReadiness
import com.sakshyam.agribot.domain.model.ModelEntry
import com.sakshyam.agribot.domain.model.ModelManifest

object ModelBundleReadinessChecker {
    private val sha256Pattern = Regex("^[a-fA-F0-9]{64}$")

    fun check(
        manifest: ModelManifest?,
        labels: List<String>,
        assetExists: (String) -> Boolean,
    ): ModelBundleReadiness {
        if (manifest == null) {
            return ModelBundleReadiness(
                isReady = false,
                statusText = "Not ready: model_manifest.json missing",
                errors = listOf("model_manifest.json missing"),
            )
        }

        val errors = buildList {
            addAll(ModelManifestValidator.validate(manifest, labels).errors)
            checkEntry("classifier", manifest.classifier, required = true, assetExists = assetExists)
            checkLabelsAsset(manifest, assetExists)
            checkLabelsHash(manifest)
            val detector = manifest.detector
            if (detector == null) {
                add("Missing detector entry for Front Row Overview")
            } else {
                checkEntry("detector", detector, required = true, assetExists = assetExists)
            }
        }.distinct()

        return ModelBundleReadiness(
            isReady = errors.isEmpty(),
            statusText = if (errors.isEmpty()) {
                "Ready: ${manifest.bundleId}"
            } else {
                "Not ready: ${errors.first()}"
            },
            errors = errors,
        )
    }

    private fun MutableList<String>.checkEntry(
        name: String,
        entry: ModelEntry,
        required: Boolean,
        assetExists: (String) -> Boolean,
    ) {
        if (required && !assetExists(entry.file)) {
            add("Missing $name asset: ${entry.file}")
        }
        if (!entry.file.endsWith(".tflite")) {
            add("${name.replaceFirstChar { it.uppercase() }} asset must be a .tflite file: ${entry.file}")
        }
        val sha = entry.sha256
        if (sha.isNullOrBlank()) {
            add("Missing SHA-256 for $name asset: ${entry.file}")
        } else if (!sha256Pattern.matches(sha)) {
            add("Invalid SHA-256 for $name asset: ${entry.file}")
        }
    }

    private fun MutableList<String>.checkLabelsAsset(
        manifest: ModelManifest,
        assetExists: (String) -> Boolean,
    ) {
        val labelsFile = manifest.classifier.labelsFile
        if (labelsFile.isNullOrBlank()) {
            add("Missing classifier labels file in manifest")
            return
        }
        if (!assetExists(labelsFile)) {
            add("Missing classifier labels asset: $labelsFile")
        }
    }

    private fun MutableList<String>.checkLabelsHash(manifest: ModelManifest) {
        val sha = manifest.labelsSha256
        if (sha.isNullOrBlank()) {
            add("Missing SHA-256 for labels asset: ${manifest.classifier.labelsFile ?: "labels/labels.json"}")
        } else if (!sha256Pattern.matches(sha)) {
            add("Invalid SHA-256 for labels asset: ${manifest.classifier.labelsFile ?: "labels/labels.json"}")
        }
    }
}
