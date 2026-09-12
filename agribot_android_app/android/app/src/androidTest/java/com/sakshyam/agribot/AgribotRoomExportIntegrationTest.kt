package com.sakshyam.agribot

import android.content.Context
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sakshyam.agribot.data.db.AgribotDatabase
import com.sakshyam.agribot.data.repository.FileExportRepository
import com.sakshyam.agribot.data.repository.RoomFieldLayoutRepository
import com.sakshyam.agribot.data.repository.RoomRunRepository
import com.sakshyam.agribot.domain.logic.RunEventFactory
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.DiagnosticsSnapshot
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RowId
import com.sakshyam.agribot.domain.model.RunConfig
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunState
import java.io.File
import java.time.Instant
import java.util.zip.ZipFile
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AgribotRoomExportIntegrationTest {
    private lateinit var context: Context
    private lateinit var database: AgribotDatabase
    private lateinit var runRepository: RoomRunRepository
    private lateinit var fieldLayoutRepository: RoomFieldLayoutRepository
    private lateinit var exportRepository: FileExportRepository

    @Before
    fun setUp() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        database = Room.inMemoryDatabaseBuilder(context, AgribotDatabase::class.java)
            .allowMainThreadQueries()
            .build()
        runRepository = RoomRunRepository(database)
        fieldLayoutRepository = RoomFieldLayoutRepository(database)
        exportRepository = FileExportRepository(context, runRepository, fieldLayoutRepository)
    }

    @After
    fun tearDown() {
        database.close()
    }

    @Test
    fun runPersistenceSupportsReplacementRetakeAndPiCompatibleExports() = runBlocking {
        val runId = RunId("android_test_${System.currentTimeMillis()}")
        val layout = fieldLayout(runId.value)
        fieldLayoutRepository.saveLayout(layout)
        fieldLayoutRepository.setActiveLayout(layout.id)

        runRepository.createRun(
            RunConfig(
                runId = runId,
                mode = RecordingMode.SIDE_SCAN,
                fieldLayoutId = layout.id,
                targetFps = 5,
                modelBundleId = "agribot-model-bundle-v001",
            ),
        )
        runRepository.appendEvent(
            RunEventFactory.started(
                runId = runId,
                timestamp = Instant.parse("2026-06-08T09:00:00Z"),
                mode = RecordingMode.SIDE_SCAN.name,
                targetFps = 5,
            ),
        )
        runRepository.appendDecision(recordedDecision(runId, sequence = 1, label = "Skipped"))
        assertEquals(1, runRepository.decisionsForRun(runId).size)

        runRepository.deleteDecision(DecisionId("${runId.value}_1"))
        assertTrue(runRepository.decisionsForRun(runId).isEmpty())

        runRepository.appendDecision(recordedDecision(runId, sequence = 1, label = "Healthy"))
        runRepository.markRunCompleted(runId)

        val persistedRun = runRepository.runById(runId)
        assertNotNull(persistedRun)
        assertEquals(RunState.COMPLETED, persistedRun!!.state)
        val decisions = runRepository.decisionsForRun(runId)
        assertEquals(1, decisions.size)
        assertEquals("Healthy", decisions.single().label)
        assertEquals("Field 2|B|1", decisions.single().plantKey)

        val csv = exportRepository.exportCsv(runId)
        val jsonl = exportRepository.exportJsonl(runId)
        val logs = exportRepository.exportLogs(runId)
        val bundle = exportRepository.exportBundle(
            runId,
            DiagnosticsSnapshot(
                appVersion = "1.0",
                modelBundleId = "agribot-model-bundle-v001",
                cameraId = "rear-default",
                supportedSizes = "1920x1080",
                analysisResolution = "1080x1920",
                cpuThreadProfile = "5 FPS / 4 CPU threads",
                latestBenchmarkResult = "84.2 ms",
                readinessSummary = "ready",
                thermalBatteryWarnings = "none",
                storageUse = "1.0 MB",
                permissionStatus = "granted",
                runState = "completed",
                lastExportPath = null,
                measurementSource = "PHONE_STEP_SENSOR",
                measurementQuality = "ESTIMATED",
                gpsStatus = "GPS weak",
                motionEventsObserved = true,
                captureQualityStatus = "accept",
                detectionStatus = "1 plant box found",
                latestDecisionReason = "manual_override",
            ),
        )

        val csvText = File(csv.absolutePath).readText()
        val jsonlText = File(jsonl.absolutePath).readText()
        assertTrue(csvText.contains("sequence,timestamp,epoch_time,field_id,row_id"))
        assertTrue(csvText.contains("Field 2,B,2,1,1,Field 2|B|1"))
        assertTrue(jsonlText.contains("\"plant_key\":\"Field 2|B|1\""))
        assertTrue(jsonlText.contains("\"evidence_path\":\"evidence_frames/${runId.value}_1_manual.jpg\""))
        assertTrue(!jsonlText.contains(context.filesDir.absolutePath))
        assertTrue(File(logs.absolutePath).name == "app_log.jsonl")

        ZipFile(File(bundle.absolutePath)).use { zip ->
            assertNotNull(zip.getEntry("metadata.json"))
            assertNotNull(zip.getEntry("summary.json"))
            assertNotNull(zip.getEntry("latest_run.json"))
            assertNotNull(zip.getEntry("field_layout.json"))
            assertNotNull(zip.getEntry("decisions.jsonl"))
            assertNotNull(zip.getEntry("events.csv"))
            assertNotNull(zip.getEntry("run_events.jsonl"))
            assertNotNull(zip.getEntry("app_log.jsonl"))
            val diagnosticsEntry = zip.getEntry("diagnostics.json")
            assertNotNull(diagnosticsEntry)
            val diagnosticsText = zip.getInputStream(diagnosticsEntry).bufferedReader().use { it.readText() }
            assertTrue(diagnosticsText.contains("\"measurement_source\":\"PHONE_STEP_SENSOR\""))
            assertNotNull(zip.getEntry("evidence_frames/${runId.value}_1_manual.jpg"))
        }
    }

    private fun fieldLayout(suffix: String): FieldLayout =
        FieldLayout(
            id = FieldId("field_2_$suffix"),
            name = "Field 2",
            activeRowId = RowId("B"),
            rows = listOf(
                FieldRow(RowId("A"), rowIndex = 1, plantsPerRow = 20, plantSpacingM = 0.45, rowSpacingM = 1.4, yM = 0.0),
                FieldRow(RowId("B"), rowIndex = 2, plantsPerRow = 15, plantSpacingM = 0.35, rowSpacingM = 1.4, yM = 1.4),
            ),
            startPlant = 1,
            plantStep = 1,
            plantCooldownSec = 2.0,
            updatedAt = Instant.parse("2026-06-08T09:00:00Z"),
        )

    private fun recordedDecision(runId: RunId, sequence: Int, label: String): RecordedDecision =
        RecordedDecision(
            id = DecisionId("${runId.value}_$sequence"),
            runId = runId,
            sequence = sequence,
            timestamp = Instant.parse("2026-06-08T09:00:0${sequence}Z"),
            epochTime = 1_780_907_200.0 + sequence,
            mode = RecordingMode.SIDE_SCAN,
            fieldId = "Field 2",
            rowId = "B",
            rowSide = null,
            rowIndex = 2,
            plantColumn = sequence,
            plantNumber = sequence,
            plantKey = "Field 2|B|$sequence",
            xM = 0.35 * (sequence - 1),
            yM = 1.4,
            bboxPx = null,
            geometryConfidence = null,
            plantPositionConfidence = null,
            geometryReason = null,
            label = label,
            confidence = if (label == "Healthy") 0.91f else 0.0f,
            rawLabel = label,
            status = if (label == "Healthy") DecisionStatus.OK else DecisionStatus.SKIPPED,
            action = if (label == "Healthy") PlantHealthAction.NONE else PlantHealthAction.RESCAN,
            framesUsed = 0,
            reason = if (label == "Healthy") "manual_override" else "operator_skip",
            lateFrames = 0,
            gateAvgLatencyMs = 0.0,
            gateMaxLatencyMs = 0.0,
            modelVersion = "agribot-model-bundle-v001",
            scanIndex = sequence,
            scanPass = 1,
            fieldComplete = false,
            plannedTotalPlants = 35,
            plantDisplay = "Field 2 / Row B / Plant $sequence",
            plantId = sequence,
            threads = 4,
            fpsTarget = 5.0,
            manualOverride = true,
            notes = null,
            evidencePath = seededEvidencePath(runId, sequence),
            evidenceStatus = "saved",
        )

    private fun seededEvidencePath(runId: RunId, sequence: Int): String {
        val evidenceDir = File(
            File(File(context.filesDir, "exports"), "agribot_run_${runId.value}"),
            "evidence_frames",
        ).apply { mkdirs() }
        val file = File(evidenceDir, "${runId.value}_${sequence}_manual.jpg")
        file.writeBytes(byteArrayOf(0xff.toByte(), 0xd8.toByte(), 0xff.toByte(), 0xd9.toByte()))
        return file.absolutePath
    }
}
