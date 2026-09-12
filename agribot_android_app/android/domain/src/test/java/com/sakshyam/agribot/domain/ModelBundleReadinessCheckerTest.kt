package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.ModelBundleReadinessChecker
import com.sakshyam.agribot.domain.logic.ModelManifestValidator
import com.sakshyam.agribot.domain.model.ModelEntry
import com.sakshyam.agribot.domain.model.ModelManifest
import com.sakshyam.agribot.domain.model.ScanConstants
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ModelBundleReadinessCheckerTest {
    @Test
    fun readyWhenManifestLabelsHashesAndRequiredAssetsArePresent() {
        val readiness = ModelBundleReadinessChecker.check(
            manifest = manifest(),
            labels = ModelManifestValidator.expectedLabels,
            assetExists = { it in requiredAssets },
        )

        assertTrue(readiness.isReady)
        assertEquals(emptyList(), readiness.errors)
        assertEquals("Ready: agribot-model-bundle-v001", readiness.statusText)
    }

    @Test
    fun missingClassifierAndDetectorAssetsAreReported() {
        val readiness = ModelBundleReadinessChecker.check(
            manifest = manifest(
                classifier = classifier(sha256 = null),
                detector = detector(sha256 = ""),
            ),
            labels = ModelManifestValidator.expectedLabels,
            assetExists = { false },
        )

        assertFalse(readiness.isReady)
        assertTrue("Missing classifier asset: models/classifier_fastcrop_int8.tflite" in readiness.errors)
        assertTrue("Missing detector asset: models/detector_nano_256_int8.tflite" in readiness.errors)
        assertTrue("Missing SHA-256 for classifier asset: models/classifier_fastcrop_int8.tflite" in readiness.errors)
        assertTrue("Missing SHA-256 for detector asset: models/detector_nano_256_int8.tflite" in readiness.errors)
    }

    @Test
    fun labelsAssetHashIsRequiredForReadiness() {
        val readiness = ModelBundleReadinessChecker.check(
            manifest = manifest(labelsSha256 = null),
            labels = ModelManifestValidator.expectedLabels,
            assetExists = { it in requiredAssets },
        )

        assertFalse(readiness.isReady)
        assertTrue("Missing SHA-256 for labels asset: labels/labels.json" in readiness.errors)
    }

    @Test
    fun labelsAssetHashMustBeValidSha256() {
        val readiness = ModelBundleReadinessChecker.check(
            manifest = manifest(labelsSha256 = "not-a-sha"),
            labels = ModelManifestValidator.expectedLabels,
            assetExists = { it in requiredAssets },
        )

        assertFalse(readiness.isReady)
        assertTrue("Invalid SHA-256 for labels asset: labels/labels.json" in readiness.errors)
    }

    @Test
    fun classifierSourceMustMatchFastCropDeploymentArtifact() {
        val readiness = ModelBundleReadinessChecker.check(
            manifest = manifest(classifier = classifier(source = "models/yolov8s.pt")),
            labels = ModelManifestValidator.expectedLabels,
            assetExists = { it in requiredAssets },
        )

        assertFalse(readiness.isReady)
        assertTrue("Side Scan classifier source must come from classifier_deploy_fastcrop.pt for v1" in readiness.errors)
    }

    @Test
    fun labelOrderMismatchKeepsBundleNotReady() {
        val readiness = ModelBundleReadinessChecker.check(
            manifest = manifest(),
            labels = listOf("Healthy", "Early_blight"),
            assetExists = { it in requiredAssets },
        )

        assertFalse(readiness.isReady)
        assertTrue(readiness.errors.any { it.startsWith("Label order mismatch") })
    }

    @Test
    fun detectorWithSingleOutputCandidateIsNotFrontOverviewReady() {
        val readiness = ModelBundleReadinessChecker.check(
            manifest = manifest(detector = detector(outputCandidates = 1)),
            labels = ModelManifestValidator.expectedLabels,
            assetExists = { it in requiredAssets },
        )

        assertFalse(readiness.isReady)
        assertTrue("Front Overview detector output candidate count must be >= 2" in readiness.errors)
    }

    @Test
    fun detectorSourceMustMatchFrontOverviewCandidateArtifact() {
        val readiness = ModelBundleReadinessChecker.check(
            manifest = manifest(detector = detector(source = "models/experimental_detector.pt")),
            labels = ModelManifestValidator.expectedLabels,
            assetExists = { it in requiredAssets },
        )

        assertFalse(readiness.isReady)
        assertTrue("Front Overview detector source must come from detector_nano_256.pt for v1" in readiness.errors)
    }

    private fun manifest(
        classifier: ModelEntry = classifier(),
        detector: ModelEntry? = detector(),
        labelsSha256: String? = VALID_SHA,
    ) = ModelManifest(
        bundleId = "agribot-model-bundle-v001",
        createdAt = "2026-06-08T00:00:00Z",
        classifier = classifier,
        detector = detector,
        labelsSha256 = labelsSha256,
        minAndroidSdk = 26,
        cpuDefault = true,
        requiresNetwork = false,
    )

    private fun classifier(
        source: String = "models/classifier_deploy_fastcrop.pt",
        sha256: String? = VALID_SHA,
    ) = ModelEntry(
        file = "models/classifier_fastcrop_int8.tflite",
        source = source,
        inputWidth = 224,
        inputHeight = 224,
        labelsFile = "labels/labels.json",
        sha256 = sha256,
        confidenceThreshold = ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD,
        highConfidenceThreshold = ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    )

    private fun detector(
        sha256: String? = VALID_SHA,
        outputCandidates: Int = 1344,
        source: String = "models/detector_nano_256.pt",
    ) = ModelEntry(
        file = "models/detector_nano_256_int8.tflite",
        source = source,
        inputWidth = 256,
        inputHeight = 256,
        labelsFile = null,
        sha256 = sha256,
        confidenceThreshold = null,
        highConfidenceThreshold = null,
        outputCandidates = outputCandidates,
    )

    private companion object {
        private const val VALID_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        private val requiredAssets = setOf(
            "models/classifier_fastcrop_int8.tflite",
            "models/detector_nano_256_int8.tflite",
            "labels/labels.json",
        )
    }
}
