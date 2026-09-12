package com.sakshyam.agribot.featurescan

import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunState
import com.sakshyam.agribot.domain.model.ScanConstants
import com.sakshyam.agribot.domain.model.SideScanPosition
import com.sakshyam.agribot.domain.model.ThermalStatus
import com.sakshyam.agribot.domain.model.TreatmentStatus
import com.sakshyam.agribot.featurescan.tracking.PlantPosition3D
import com.sakshyam.agribot.featurescan.tracking.TrackedPlant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ScanUiStateTest {
    @Test
    fun sideScanDefaultsUsePiCompatibleFiveFpsBaseline() {
        val state = ScanUiState()

        assertEquals(5, state.targetFps)
        assertEquals("5 FPS", state.performanceStatus)
        assertTrue(state.saveEvidenceFrames)
    }

    @Test
    fun importedPiFpsIsCappedForNativeDefaultProfile() {
        val importedFps = SideScanDefaults.nativeDefaultTargetFps(10.0)

        assertEquals(5, importedFps)
        assertEquals("5 FPS", SideScanDefaults.performanceStatus(importedFps))
    }

    @Test
    fun lowImportedFpsIsRaisedToMinimumNativeDefaultProfile() {
        val importedFps = SideScanDefaults.nativeDefaultTargetFps(1.0)

        assertEquals(3, importedFps)
        assertEquals("3 FPS", SideScanDefaults.performanceStatus(importedFps))
    }

    @Test
    fun activeRecordingRunIsProtectedFromDeletion() {
        val runId = RunId("run-active")
        val state = ScanUiState(activeRunId = runId, runState = RunState.RECORDING)

        assertTrue(state.protectsActiveRunFromDeletion(runId))
    }

    @Test
    fun pausedActiveRunIsProtectedFromDeletion() {
        val runId = RunId("run-paused")
        val state = ScanUiState(activeRunId = runId, runState = RunState.PAUSED)

        assertTrue(state.protectsActiveRunFromDeletion(runId))
    }

    @Test
    fun completedActiveRunCanBeDeleted() {
        val runId = RunId("run-completed")
        val state = ScanUiState(activeRunId = runId, runState = RunState.COMPLETED)

        assertFalse(state.protectsActiveRunFromDeletion(runId))
    }

    @Test
    fun unrelatedRunCanBeDeletedWhileAnotherRunIsActive() {
        val state = ScanUiState(activeRunId = RunId("run-active"), runState = RunState.RECORDING)

        assertFalse(state.protectsActiveRunFromDeletion(RunId("run-history")))
    }

    // ==================== Plant Filter Tests ====================

    private fun makePlant(id: String, health: String, occluded: Boolean = false): TrackedPlant {
        return TrackedPlant(
            plantId = id,
            createdAt = 0L,
            lastSeen = 0L,
            currentBbox = BoundingBox(0f, 0f, 10f, 10f),
            healthStatus = health,
            confidence = 0.9f,
            positionHistory = emptyList(),
            detectionCount = 1,
            features = floatArrayOf(0f, 0f, 10f, 10f, 0.9f, 1f, 100f),
            isOccluded = occluded,
            occlusionStartTime = 0L,
        )
    }

    @Test
    fun filteredPlantsAllReturnsAllPlants() {
        val plants = listOf(
            makePlant("plant_1", "Healthy"),
            makePlant("plant_2", "Sick"),
            makePlant("plant_3", "Uncertain"),
        )
        val state = ScanUiState(trackedPlants = plants, plantFilter = PlantFilter.ALL)
        assertEquals(3, state.filteredPlants.size)
    }

    @Test
    fun filteredPlantsHealthyReturnsOnlyHealthy() {
        val plants = listOf(
            makePlant("plant_1", "Healthy"),
            makePlant("plant_2", "Sick"),
            makePlant("plant_3", "Healthy"),
        )
        val state = ScanUiState(trackedPlants = plants, plantFilter = PlantFilter.HEALTHY)
        assertEquals(2, state.filteredPlants.size)
        assertTrue(state.filteredPlants.all { it.healthStatus == "Healthy" })
    }

    @Test
    fun filteredPlantsSickReturnsOnlySick() {
        val plants = listOf(
            makePlant("plant_1", "Healthy"),
            makePlant("plant_2", "Sick"),
            makePlant("plant_3", "Disease"),
            makePlant("plant_4", "Unhealthy"),
        )
        val state = ScanUiState(trackedPlants = plants, plantFilter = PlantFilter.SICK)
        assertEquals(3, state.filteredPlants.size)
        assertTrue(state.filteredPlants.all {
            it.healthStatus in listOf("Sick", "Disease", "Unhealthy")
        })
    }

    @Test
    fun filteredPlantsSickIncludesModelDiseaseLabels() {
        val plants = listOf(
            makePlant("plant_1", "Early_blight"),
            makePlant("plant_2", "Healthy"),
            makePlant("plant_3", "Unknown"),
        )
        val state = ScanUiState(trackedPlants = plants, plantFilter = PlantFilter.SICK)

        assertEquals(listOf("plant_1"), state.filteredPlants.map { it.plantId })
    }

    @Test
    fun filteredPlantsUncertainReturnsOnlyUncertain() {
        val plants = listOf(
            makePlant("plant_1", "Healthy"),
            makePlant("plant_2", "Uncertain"),
            makePlant("plant_3", "Unknown"),
        )
        val state = ScanUiState(trackedPlants = plants, plantFilter = PlantFilter.UNCERTAIN)
        assertEquals(2, state.filteredPlants.size)
        assertTrue(state.filteredPlants.all {
            it.healthStatus in listOf("Uncertain", "Unknown")
        })
    }

    @Test
    fun filteredPlantsOccludedReturnsOnlyOccluded() {
        val plants = listOf(
            makePlant("plant_1", "Healthy", occluded = false),
            makePlant("plant_2", "Sick", occluded = true),
            makePlant("plant_3", "Healthy", occluded = true),
        )
        val state = ScanUiState(trackedPlants = plants, plantFilter = PlantFilter.OCCLUDED)
        assertEquals(2, state.filteredPlants.size)
        assertTrue(state.filteredPlants.all { it.isOccluded })
    }

    @Test
    fun selectedPlantReturnsCorrectPlant() {
        val plants = listOf(
            makePlant("plant_1", "Healthy"),
            makePlant("plant_2", "Sick"),
        )
        val state = ScanUiState(
            trackedPlants = plants,
            selectedPlantId = "plant_2"
        )
        assertEquals("plant_2", state.selectedPlant?.plantId)
    }

    @Test
    fun selectedPlantReturnsNullWhenNotSelected() {
        val state = ScanUiState(
            trackedPlants = listOf(makePlant("plant_1", "Healthy")),
            selectedPlantId = null
        )
        assertEquals(null, state.selectedPlant)
    }

    @Test
    fun defaultPlantFilterIsAll() {
        val state = ScanUiState()
        assertEquals(PlantFilter.ALL, state.plantFilter)
    }

    @Test
    fun defaultIsDataModeIsFalse() {
        val state = ScanUiState()
        assertFalse(state.isDataMode)
    }

    // ==================== Field Progress Tests ====================

    private fun makePosition(plantNumber: Int, plannedTotal: Int): SideScanPosition {
        return SideScanPosition(
            fieldId = "field_1",
            rowId = "A",
            rowIndex = 1,
            plantColumn = plantNumber,
            plantNumber = plantNumber,
            plantKey = "field_1:A:$plantNumber",
            xM = 0.0,
            yM = 0.0,
            scanIndex = plantNumber,
            scanPass = 0,
            fieldComplete = plantNumber >= plannedTotal,
            plannedTotalPlants = plannedTotal,
            plantDisplay = plantNumber.toString(),
        )
    }

    @Test
    fun fieldProgressIsNullWhenNoPosition() {
        val state = ScanUiState()
        assertEquals(null, state.fieldProgress)
    }

    @Test
    fun fieldProgressIsQuarterWhen25Of100() {
        val state = ScanUiState(currentPosition = makePosition(25, 100))
        assertEquals(0.25f, state.fieldProgress)
    }

    @Test
    fun fieldProgressIsOneWhenComplete() {
        val state = ScanUiState(currentPosition = makePosition(100, 100))
        assertEquals(1.0f, state.fieldProgress)
    }

    @Test
    fun fieldProgressClampsToOneWhenOver() {
        val state = ScanUiState(currentPosition = makePosition(150, 100))
        assertEquals(1.0f, state.fieldProgress)
    }

    @Test
    fun fieldProgressIsNullWhenPlantNumberZero() {
        val state = ScanUiState(currentPosition = makePosition(0, 100))
        assertEquals(null, state.fieldProgress)
    }

    // ==================== Search Query Tests ====================

    private fun makePlantWithNote(id: String, health: String, note: String? = null): TrackedPlant {
        return TrackedPlant(
            plantId = id,
            createdAt = 0L,
            lastSeen = 0L,
            currentBbox = BoundingBox(0f, 0f, 10f, 10f),
            healthStatus = health,
            confidence = 0.9f,
            positionHistory = listOf(PlantPosition3D(5f, 5f, 1.5f, 0f, 0f, 0f, 0L)),
            detectionCount = 1,
            features = floatArrayOf(0f, 0f, 10f, 10f, 0.9f, 1f, 100f),
            isOccluded = false,
            occlusionStartTime = 0L,
            treatmentNote = note,
        )
    }

    @Test
    fun searchQueryFiltersByPlantId() {
        val plants = listOf(
            makePlantWithNote("plant_1", "Healthy"),
            makePlantWithNote("plant_2", "Sick"),
        )
        val state = ScanUiState(
            trackedPlants = plants,
            plantSearchQuery = "plant_2",
        )
        assertEquals(1, state.filteredPlants.size)
        assertEquals("plant_2", state.filteredPlants[0].plantId)
    }

    @Test
    fun searchQueryFiltersByHealthStatus() {
        val plants = listOf(
            makePlantWithNote("plant_1", "Healthy"),
            makePlantWithNote("plant_2", "Sick"),
        )
        val state = ScanUiState(
            trackedPlants = plants,
            plantSearchQuery = "sick",
        )
        assertEquals(1, state.filteredPlants.size)
        assertEquals("Sick", state.filteredPlants[0].healthStatus)
    }

    @Test
    fun searchQueryFiltersByTreatmentNote() {
        val plants = listOf(
            makePlantWithNote("plant_1", "Sick", note = "Applied copper fungicide"),
            makePlantWithNote("plant_2", "Sick", note = "Waiting for review"),
        )
        val state = ScanUiState(
            trackedPlants = plants,
            plantSearchQuery = "copper",
        )
        assertEquals(1, state.filteredPlants.size)
        assertEquals("plant_1", state.filteredPlants[0].plantId)
    }

    @Test
    fun searchQueryIsCaseInsensitive() {
        val plants = listOf(
            makePlantWithNote("plant_1", "Healthy"),
            makePlantWithNote("plant_2", "Sick"),
        )
        val state = ScanUiState(
            trackedPlants = plants,
            plantSearchQuery = "PLANT_1",
        )
        assertEquals(1, state.filteredPlants.size)
        assertEquals("plant_1", state.filteredPlants[0].plantId)
    }

    @Test
    fun emptySearchQueryReturnsAll() {
        val plants = listOf(
            makePlantWithNote("plant_1", "Healthy"),
            makePlantWithNote("plant_2", "Sick"),
        )
        val state = ScanUiState(
            trackedPlants = plants,
            plantSearchQuery = "",
        )
        assertEquals(2, state.filteredPlants.size)
    }

    @Test
    fun searchQueryNoMatchReturnsEmpty() {
        val plants = listOf(
            makePlantWithNote("plant_1", "Healthy"),
            makePlantWithNote("plant_2", "Sick"),
        )
        val state = ScanUiState(
            trackedPlants = plants,
            plantSearchQuery = "nonexistent",
        )
        assertEquals(0, state.filteredPlants.size)
    }

    // ==================== New Defaults Tests ====================

    @Test
    fun confidenceThresholdDefaultIsCorrect() {
        val state = ScanUiState()
        assertEquals(ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD, state.confidenceThreshold)
        assertEquals(ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD, state.highConfidenceThreshold)
    }

    @Test
    fun syncStatusDefaultIsOffline() {
        val state = ScanUiState()
        assertTrue(state.syncStatus.contains("Offline"))
    }

    @Test
    fun thermalStatusDefaultIsNone() {
        val state = ScanUiState()
        assertEquals(ThermalStatus.NONE, state.thermalStatus)
    }

    @Test
    fun batteryPercentDefaultIsNull() {
        val state = ScanUiState()
        assertEquals(null, state.batteryPercent)
    }

    @Test
    fun modelBundleIdDefaultIsCorrect() {
        val state = ScanUiState()
        assertEquals("agribot-model-bundle-v001", state.modelBundleId)
    }

    @Test
    fun selectedRunSummaryDefaultIsNull() {
        val state = ScanUiState()
        assertEquals(null, state.selectedRunSummary)
    }
}
