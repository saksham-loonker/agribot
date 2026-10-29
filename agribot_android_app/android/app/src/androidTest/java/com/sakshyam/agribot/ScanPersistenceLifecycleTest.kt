package com.sakshyam.agribot

import android.content.Context
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelStore
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sakshyam.agribot.data.db.AgribotDatabase
import com.sakshyam.agribot.data.repository.RoomFieldLayoutRepository
import com.sakshyam.agribot.data.repository.RoomRunRepository
import com.sakshyam.agribot.domain.model.*
import com.sakshyam.agribot.domain.repository.*
import com.sakshyam.agribot.domain.scan.*
import com.sakshyam.agribot.featurescan.check.CheckError
import com.sakshyam.agribot.featurescan.check.CheckPlantViewModel
import com.sakshyam.agribot.featurescan.walk.WalkError
import com.sakshyam.agribot.featurescan.walk.WalkRowsViewModel
import com.sakshyam.agribot.featurescan.walk.WalkSetup
import java.io.IOException
import java.time.Instant
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/** Real Room and Android ViewModels with injected write failures/delayed inference.
 * These tests check orchestration, not recognition accuracy or phone performance.
 */
@RunWith(AndroidJUnit4::class)
class ScanPersistenceLifecycleTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private lateinit var context: Context
    private lateinit var database: AgribotDatabase
    private lateinit var room: RoomRunRepository
    private lateinit var runs: FaultRepository
    private lateinit var vision: FakeVision
    private lateinit var store: ViewModelStore
    private val labels = listOf("Early_blight", "Healthy", "Late_blight", "Leaf Miner", "Magnesium Deficiency", "Nitrogen Deficiency", "Pottassium Deficiency", "Spotted Wilt Virus", "Other")
    private val settings = object : ScanSettingsRepository {
        override fun observeSettings() = flowOf(ScanSettings(saveEvidenceFrames = false))
        override suspend fun saveSettings(settings: ScanSettings) = Unit
    }
    private val photos = object : EvidenceRepository {
        override suspend fun retentionSnapshot(runId: RunId) = EvidenceRetentionSnapshot(0, 0)
        override suspend fun saveEvidence(runId: RunId, decisionId: DecisionId, frame: AnalysisFrame, kind: String): EvidenceCaptureResult = error("Photos disabled in test")
    }

    @Before fun setUp() {
        context = instrumentation.targetContext
        database = Room.inMemoryDatabaseBuilder(context, AgribotDatabase::class.java).build()
        room = RoomRunRepository(database, "lifecycle-test")
        runs = FaultRepository(room)
        vision = FakeVision()
        store = ViewModelStore()
    }

    @After fun tearDown() {
        ui { store.clear() }
        database.close()
    }

    private fun ui(action: () -> Unit) = instrumentation.runOnMainSync(action)
    private suspend fun until(condition: suspend () -> Boolean) = withTimeout(8_000) {
        while (!condition()) delay(10)
    }

    private fun checkVm(): CheckPlantViewModel {
        lateinit var vm: CheckPlantViewModel
        ui {
            vm = CheckPlantViewModel(context, vision, runs, photos, settings)
            store.put("check", vm)
            vm.setScreenActive(true)
        }
        return vm
    }

    private fun walkVm(saved: SavedStateHandle = SavedStateHandle()): WalkRowsViewModel {
        lateinit var vm: WalkRowsViewModel
        ui {
            vm = WalkRowsViewModel(context, saved, vision, runs, RoomFieldLayoutRepository(database), photos, settings)
            store.put("walk", vm)
            vm.setScreenActive(true)
        }
        return vm
    }

    private suspend fun start(vm: WalkRowsViewModel) {
        ui { vm.start(WalkSetup("Test field", 2, 5, 0.5, 0.3)) }
        until { vm.state.value.started }
    }

    private fun frame() = VisionFrame(64, 64, IntArray(64 * 64) { index ->
        if ((index / 64 / 4 + index % 64 / 4) % 2 == 0) 0xFF99CC66.toInt() else 0xFF225522.toInt()
    }, System.nanoTime())

    private suspend fun observeHealthyPlant(vm: WalkRowsViewModel) {
        repeat(6) {
            until { vm.isReady() }
            val before = vision.calls.get()
            ui { vm.onFrame(frame()) }
            until { vision.calls.get() > before && vm.isReady() }
        }
        assertEquals(VerdictKind.HEALTHY, vm.state.value.currentVerdict!!.kind)
    }

    private suspend fun diagnose(vm: CheckPlantViewModel) {
        repeat(12) {
            if (vm.state.value.verdict != null) return
            until { vm.isReady() }
            val before = vm.state.value.framesUsed
            ui { vm.onFrame(frame()) }
            until { vm.state.value.framesUsed > before }
        }
        assertNotNull(vm.state.value.verdict)
    }

    @Test fun quickCheckTransactionRollsBackTheRunWhenDecisionInsertFails() = runBlocking {
        database.openHelper.writableDatabase.execSQL("CREATE TRIGGER fail_check BEFORE INSERT ON decisions BEGIN SELECT RAISE(ABORT, 'injected'); END")
        val id = RunId("failed-check")
        assertTrue(runCatching { room.saveCompletedRun(config(id), decision(id, "check")) }.isFailure)
        assertNull(room.runById(id))
        assertTrue(room.recentRuns().isEmpty())
    }

    @Test fun multiPlantTransactionRollsBackEarlierWritesInTheSameMovement() = runBlocking {
        val id = RunId("batch")
        room.createRun(config(id))
        database.openHelper.writableDatabase.execSQL("CREATE TRIGGER fail_batch BEFORE INSERT ON decisions WHEN NEW.id = 'bad' BEGIN SELECT RAISE(ABORT, 'injected'); END")
        assertTrue(runCatching { room.appendDecisions(listOf(decision(id, "good"), decision(id, "bad").copy(sequence = 2))) }.isFailure)
        assertTrue(room.decisionsForRun(id).isEmpty())
    }

    @Test fun failedCheckSaveKeepsTheVerdictAndRetriesWithoutAnOrphanRun() = runBlocking {
        val vm = checkVm()
        diagnose(vm)
        runs.failQuickSave = true
        runs.saveGate = CompletableDeferred()
        ui { vm.save() }
        until { vm.state.value.saving }
        val verdict = vm.state.value.verdict
        ui { vm.checkAgain() }
        assertEquals(verdict, vm.state.value.verdict)
        runs.saveGate!!.complete(Unit)
        until { !vm.state.value.saving && vm.state.value.error == CheckError.SAVE_FAILED }
        assertEquals(verdict, vm.state.value.verdict)
        assertTrue(room.recentRuns().isEmpty())
        runs.failQuickSave = false
        runs.saveGate = null
        ui { vm.save() }
        until { vm.state.value.savedRunId != null }
        val saved = room.recentRuns().single()
        assertEquals(RunState.COMPLETED, saved.state)
        assertEquals(1, room.decisionsForRun(saved.runId).size)
    }

    @Test fun failedMoveKeepsTheCurrentPlantAndAllowsASafeRetry() = runBlocking {
        val vm = walkVm()
        start(vm)
        observeHealthyPlant(vm)
        val beforeFailure = vm.state.value.currentVerdict
        runs.failBatch = true
        ui { vm.nextPlant() }
        until { !vm.state.value.saving && vm.state.value.error == WalkError.SAVE_FAILED }
        assertEquals(1, vm.state.value.row)
        assertEquals(1, vm.state.value.plant)
        assertTrue(vm.state.value.paused)
        assertEquals(beforeFailure, vm.state.value.currentVerdict)
        val id = RunId(vm.state.value.runId!!)
        assertTrue(room.decisionsForRun(id).isEmpty())
        runs.failBatch = false
        ui { vm.nextPlant() }
        until { !vm.state.value.saving && vm.state.value.plant == 2 }
        val recorded = room.decisionsForRun(id).single()
        assertEquals(1, recorded.plantNumber)
        assertEquals(1, recorded.sequence)
        assertEquals("Healthy", recorded.label)
        assertEquals(beforeFailure!!.framesUsed, recorded.framesUsed)
    }

    @Test fun finishRetryDoesNotDuplicateTheAlreadyCommittedPlant() = runBlocking {
        val vm = walkVm()
        start(vm)
        observeHealthyPlant(vm)
        runs.failFinishOnce = true
        ui { vm.finish() }
        until { !vm.state.value.saving && vm.state.value.error == WalkError.SAVE_FAILED }
        assertFalse(vm.state.value.finished)
        val id = RunId(vm.state.value.runId!!)
        assertEquals(1, room.decisionsForRun(id).size)
        ui { vm.finish() }
        until { vm.state.value.finished }
        assertEquals(1, room.decisionsForRun(id).size)
        assertEquals(RunState.COMPLETED, room.runById(id)!!.state)
    }

    @Test fun checkRejectsAnInferenceThatSpansABackgroundTransition() = runBlocking {
        val vm = checkVm()
        val gate = vision.blockNext()
        ui { vm.onFrame(frame()) }
        withTimeout(8_000) { vision.entered.await() }
        ui { vm.setScreenActive(false); vm.setScreenActive(true) }
        gate.complete(Unit)
        until { vm.isReady() }
        assertEquals(0, vm.state.value.framesUsed)
        ui { vm.onFrame(frame()) }
        until { vm.state.value.framesUsed == 1 }
    }

    @Test fun pauseAndResumeRejectThePreviousInFlightFrame() = runBlocking {
        val vm = walkVm()
        start(vm)
        val gate = vision.blockNext()
        ui { vm.onFrame(frame()) }
        withTimeout(8_000) { vision.entered.await() }
        ui { vm.togglePause() }
        until { vm.state.value.paused && !vm.state.value.saving }
        ui { vm.togglePause() }
        until { !vm.state.value.paused && !vm.state.value.saving }
        gate.complete(Unit)
        until { vm.isReady() }
        assertEquals(0, vm.state.value.currentLeaves)
        ui { vm.setScreenActive(false) }
        until { vm.state.value.paused }
        ui { vm.setScreenActive(true) }
        assertTrue(vm.state.value.paused)
        assertFalse(vm.isReady())
    }

    @Test fun restoredCompletedWalkIsNeverReopenedAsRecording() = runBlocking {
        val id = RunId("completed-walk")
        room.saveCompletedRun(config(id), decision(id, "done"))
        val saved = SavedStateHandle(mapOf("walk_run" to id.value, "walk_setup" to arrayListOf("Field", "2", "5", "0.5", "0.3")))
        val vm = walkVm(saved)
        until { vm.state.value.finished }
        assertEquals(RunState.COMPLETED, room.runById(id)!!.state)
        assertFalse(vm.isReady())
    }

    @Test fun walkThatFinishesStartingInTheBackgroundRemainsPaused() = runBlocking {
        val vm = walkVm()
        runs.startGate = CompletableDeferred()
        ui { vm.start(WalkSetup("Field", 2, 5, 0.5, 0.3)) }
        withTimeout(8_000) { runs.createEntered.await() }
        ui { vm.setScreenActive(false) }
        runs.startGate!!.complete(Unit)
        until { vm.state.value.started }
        assertTrue(vm.state.value.paused)
        until { room.runById(RunId(vm.state.value.runId!!))?.state == RunState.PAUSED }
    }

    @Test fun failedModelsDoNotLeaveWalkStartWaitingForever() = runBlocking {
        vision.status.value = VisionStatus.Failed("injected model failure")
        val vm = walkVm()
        ui { vm.start(WalkSetup("Field", 2, 5, 0.5, 0.3)) }
        until { vm.state.value.error == WalkError.START_FAILED }
        assertFalse(vm.state.value.started)
    }

    @Test fun quickCheckWaitsForPrimaryWindowInsteadOfFinalizingOtherEarly() = runBlocking {
        vision.primaryRecovery = true
        vision.status.value = VisionStatus.Ready(VisionModelInfo("test-primary", labels, 1, 8, 1, primaryPolicy = PrimaryImagePolicy()))
        val vm = checkVm()
        repeat(6) {
            until { vm.isReady() }
            val before = vm.state.value.framesUsed
            ui { vm.onFrame(frame()) }
            until { vm.state.value.framesUsed > before }
            assertNull("Other must not stop the seven-frame recovery window", vm.state.value.verdict)
            assertTrue(vm.state.value.scanning)
        }
        diagnose(vm)
        val verdict = requireNotNull(vm.state.value.verdict)
        assertEquals(VerdictKind.HEALTHY, verdict.kind)
        assertTrue(verdict.primaryView)
        assertTrue(verdict.framesUsed >= 7)
        assertEquals(1, verdict.leavesSeen)
        assertEquals(0, verdict.leavesAgreeing)
        ui { vm.save() }
        until { vm.state.value.savedRunId != null }
        val saved = room.decisionsForRun(RunId(vm.state.value.savedRunId!!)).single().toVerdict().first
        assertTrue(saved.primaryView)
        assertEquals(verdict.framesUsed, saved.framesUsed)
    }

    private fun config(id: RunId) = RunConfig(id, RecordingMode.CHECK_PLANT, FieldId("test-field"), 0, "test-bundle")
    private fun decision(id: RunId, key: String) = VerdictRecords.toDecision(
        DecisionId(key), id, 1, RecordingMode.CHECK_PLANT, PlantPlace("test-field", null, null),
        PlantEvidence(9, 1, 8).verdict(), labels, "test-bundle", Instant.now(),
    )

    private class FaultRepository(private val delegate: RunRepository) : RunRepository by delegate {
        @Volatile var failQuickSave = false
        @Volatile var failBatch = false
        @Volatile var failFinishOnce = false
        @Volatile var saveGate: CompletableDeferred<Unit>? = null
        @Volatile var startGate: CompletableDeferred<Unit>? = null
        val createEntered = CompletableDeferred<Unit>()
        override suspend fun createRun(config: RunConfig): RunId {
            createEntered.complete(Unit)
            startGate?.await()
            return delegate.createRun(config)
        }
        override suspend fun saveCompletedRun(config: RunConfig, decision: RecordedDecision) {
            saveGate?.await()
            if (failQuickSave) throw IOException("injected quick-check save failure")
            delegate.saveCompletedRun(config, decision)
        }
        override suspend fun appendDecisions(decisions: List<RecordedDecision>) {
            if (failBatch) throw IOException("injected movement save failure")
            delegate.appendDecisions(decisions)
        }
        override suspend fun markRunCompleted(runId: RunId) {
            if (failFinishOnce) { failFinishOnce = false; throw IOException("injected finish failure") }
            delegate.markRunCompleted(runId)
        }
    }

    private inner class FakeVision : LeafVision {
        override val status = MutableStateFlow<VisionStatus>(VisionStatus.Ready(VisionModelInfo("test-bundle", labels, 1, 8, 1)))
        val entered = CompletableDeferred<Unit>()
        val calls = AtomicInteger()
        @Volatile var primaryRecovery = false
        private var gate: CompletableDeferred<Unit>? = null
        fun blockNext() = CompletableDeferred<Unit>().also { gate = it }
        override fun warmUp() = Unit
        override suspend fun analyze(frame: VisionFrame, maxLeaves: Int): FrameAnalysis {
            calls.incrementAndGet()
            val wait = gate.also { gate = null }
            if (wait != null) { entered.complete(Unit); wait.await() }
            val label = if (primaryRecovery) 8 else 1
            val probabilities = FloatArray(9) { if (it == label) 0.99f else 0.01f / 8 }
            val auxiliary = AuxiliaryLeafScores(.99, .99, .99)
            val primary = if (primaryRecovery) PrimaryFrameEvidence(.99, .99, .99, .99, listOf(AuxiliaryLeafViews(auxiliary, auxiliary))) else null
            return FrameAnalysis(64, 64, listOf(AnalyzedLeaf(BoundingBox(2f, 2f, 62f, 62f), 0.99f, probabilities, label)), 1, 1.0, 1.0, primary)
        }
    }
}
