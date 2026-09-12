package com.sakshyam.agribot.featurescan.tracking

import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TrackEvidenceRegressionTest {
    private fun track(label: String = "Uncertain", pending: Boolean = false) = TrackedPlant(
        plantId = "test", createdAt = 1L, lastSeen = 1L,
        currentBbox = BoundingBox(0f, 0f, 100f, 100f), healthStatus = label,
        classificationPending = pending, confidence = 0.95f,
        positionHistory = emptyList(), detectionCount = 2, features = FloatArray(7),
    )
    private fun candidate(label: String, raw: String = label) = FrontOverviewCandidate(
        bboxPx = BoundingBox(0f, 0f, 100f, 100f), detectorConfidence = 0.95f,
        label = label, confidence = 0.95f, rawLabel = raw,
    )
    @Test fun firstDiseaseObservationDoesNotBecomeDiagnosis() {
        val first = track().updateWithDetection(candidate("Late_blight"), 2L, 0f, 0f, 0f)
        assertEquals("Uncertain", first.healthStatus)
        val second = first.updateWithDetection(candidate("Late_blight"), 3L, 0f, 0f, 0f)
        assertEquals("Late_blight", second.healthStatus)
    }
    @Test fun sustainedUncertaintyClearsStaleDisease() {
        var current = track("Late_blight")
        repeat(3) { current = current.updateWithDetection(candidate("Uncertain"), 2L + it, 0f, 0f, 0f) }
        assertEquals("Uncertain", current.healthStatus)
    }
    @Test fun duplicateTimestampCannotConfirmEvidence() {
        val current = track().updateWithDetection(candidate("Late_blight"), 1L, 0f, 0f, 0f)
        assertEquals(2, current.detectionCount)
        assertEquals("Uncertain", current.healthStatus)
    }
    @Test fun detectorOnlyBoxesAreNotConfirmedPlants() {
        assertFalse(track(pending = true).isConfirmed)
    }
    @Test fun longTractorRunHasBoundedPositionHistory() {
        var current = track("Healthy")
        repeat(300) { current = current.updateWithDetection(candidate("Healthy"), 2L + it, 0f, 0f, 0f) }
        assertTrue(current.positionHistory.size <= 120)
    }
}
