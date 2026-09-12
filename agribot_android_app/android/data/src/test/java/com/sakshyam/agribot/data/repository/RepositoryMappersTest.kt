package com.sakshyam.agribot.data.repository

import com.sakshyam.agribot.data.db.DecisionEntity
import com.sakshyam.agribot.domain.model.TreatmentStatus
import kotlin.test.Test
import kotlin.test.assertEquals

class RepositoryMappersTest {
    @Test
    fun decisionMapperRoundTripsTreatmentStatusAndNote() {
        val entity = DecisionEntity(
            id = "run_1_1",
            runId = "run_1",
            sequence = 1,
            timestamp = 1_781_000_000_000,
            epochTime = 1_781_000_000.0,
            mode = "SIDE_SCAN",
            fieldId = "field",
            rowId = "row",
            rowSide = null,
            rowIndex = 1,
            plantColumn = 1,
            plantNumber = 1,
            plantKey = "row:1",
            xM = null,
            yM = null,
            label = "Late_blight",
            confidence = 0.9f,
            rawLabel = "Late_blight",
            status = "OK",
            action = "INSPECT_OR_TREAT",
            framesUsed = 1,
            reason = "test",
            lateFrames = 0,
            gateAvgLatencyMs = 0.0,
            gateMaxLatencyMs = 0.0,
            modelVersion = "test",
            manualOverride = false,
            notes = null,
            scanIndex = 1,
            scanPass = null,
            fieldComplete = null,
            plannedTotalPlants = null,
            plantDisplay = null,
            plantId = 1,
            trackId = "track_1",
            threads = null,
            fpsTarget = null,
            evidencePath = null,
            evidenceStatus = null,
            treatmentStatus = TreatmentStatus.TREATED.name,
            treatmentNote = "Applied approved treatment",
            measurementDistanceM = null,
            measurementSource = null,
            measurementQuality = null,
            relativePlantWidth = null,
            relativePlantHeight = null,
            sizeSource = null,
            sizeQuality = null,
            gpsAccuracyM = null,
            gpsFixAgeSeconds = null,
            top2Margin = null,
            predictionEntropy = null,
        )

        val decision = entity.toDomain(geometry = null)

        assertEquals(TreatmentStatus.TREATED, decision.treatmentStatus)
        assertEquals("Applied approved treatment", decision.treatmentNote)
    }

    @Test
    fun unknownPersistedTreatmentStatusSafelyFallsBackToNotTreated() {
        val entity = decisionEntity().copy(treatmentStatus = "legacy_value")

        assertEquals(TreatmentStatus.NOT_TREATED, entity.toDomain(null).treatmentStatus)
    }

    private fun decisionEntity() = DecisionEntity(
        id = "run_1_1",
        runId = "run_1",
        sequence = 1,
        timestamp = 1_781_000_000_000,
        epochTime = 1_781_000_000.0,
        mode = "SIDE_SCAN",
        fieldId = "field",
        rowId = "row",
        rowSide = null,
        rowIndex = 1,
        plantColumn = 1,
        plantNumber = 1,
        plantKey = "row:1",
        xM = null,
        yM = null,
        label = "Healthy",
        confidence = 0.9f,
        rawLabel = "Healthy",
        status = "OK",
        action = "NONE",
        framesUsed = 1,
        reason = "test",
        lateFrames = 0,
        gateAvgLatencyMs = 0.0,
        gateMaxLatencyMs = 0.0,
        modelVersion = "test",
        manualOverride = false,
        notes = null,
        scanIndex = 1,
        scanPass = null,
        fieldComplete = null,
        plannedTotalPlants = null,
        plantDisplay = null,
        plantId = 1,
        trackId = "track_1",
        threads = null,
        fpsTarget = null,
        evidencePath = null,
        evidenceStatus = null,
        treatmentStatus = TreatmentStatus.NOT_TREATED.name,
        treatmentNote = null,
        measurementDistanceM = null,
        measurementSource = null,
        measurementQuality = null,
        relativePlantWidth = null,
        relativePlantHeight = null,
        sizeSource = null,
        sizeQuality = null,
        gpsAccuracyM = null,
        gpsFixAgeSeconds = null,
        top2Margin = null,
        predictionEntropy = null,
    )
}
