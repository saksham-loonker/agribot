package com.sakshyam.agribot.featurescan.walk

import android.content.Context
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RowId
import com.sakshyam.agribot.domain.model.RunConfig
import com.sakshyam.agribot.domain.model.RunEvent
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunState
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.repository.EvidenceRepository
import com.sakshyam.agribot.domain.repository.FieldLayoutRepository
import com.sakshyam.agribot.domain.repository.RunRepository
import com.sakshyam.agribot.domain.repository.ScanSettingsRepository
import com.sakshyam.agribot.domain.scan.GpsTrailFilter
import com.sakshyam.agribot.domain.scan.LeafObservation
import com.sakshyam.agribot.domain.scan.LeafTracker
import com.sakshyam.agribot.domain.scan.LeafVision
import com.sakshyam.agribot.domain.scan.PlantEvidence
import com.sakshyam.agribot.domain.scan.PlantPlace
import com.sakshyam.agribot.domain.scan.PlantVerdict
import com.sakshyam.agribot.domain.scan.RowWalkTracker
import com.sakshyam.agribot.domain.scan.VerdictKind
import com.sakshyam.agribot.domain.scan.VerdictRecords
import com.sakshyam.agribot.domain.scan.VisionFrame
import com.sakshyam.agribot.domain.scan.VisionModelInfo
import com.sakshyam.agribot.domain.scan.VisionStatus
import com.sakshyam.agribot.domain.scan.analyzeUsable
import com.sakshyam.agribot.domain.scan.toVerdict
import com.sakshyam.agribot.featurescan.scan.FrameTools
import com.sakshyam.agribot.featurescan.scan.LocationFeed
import com.sakshyam.agribot.featurescan.scan.StepCounter
import com.sakshyam.agribot.featurescan.ui.LeafBox
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import java.time.Instant
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

data class WalkSetup(val fieldName: String, val rows: Int, val plantsPerRow: Int, val plantSpacingM: Double, val strideM: Double)

data class WalkUiState(
    val vision: VisionStatus = VisionStatus.Loading,
    val started: Boolean = false,
    val paused: Boolean = false,
    val finished: Boolean = false,
    val runId: String? = null,
    val row: Int = 1,
    val plant: Int = 1,
    val rows: Int = 1,
    val plantsPerRow: Int = 1,
    val stepsAvailable: Boolean = true,
    val boxes: List<LeafBox> = emptyList(),
    val currentLeaves: Int = 0,
    val currentVerdict: PlantVerdict? = null,
    val needsClearView: Boolean = false,
    /** Result per plant of the current row (index = plant - 1), null = not recorded yet. */
    val rowResults: List<VerdictKind?> = emptyList(),
    val lastSaved: Pair<Int, PlantVerdict>? = null,
    val labels: List<String> = emptyList(),
    val error: WalkError? = null,
    val saving: Boolean = false,
)

enum class WalkError { START_FAILED, ANALYSIS_FAILED, SAVE_FAILED }

@HiltViewModel
class WalkRowsViewModel @Inject constructor(
    @ApplicationContext context: Context,
    private val saved: SavedStateHandle,
    private val vision: LeafVision,
    private val runs: RunRepository,
    private val layouts: FieldLayoutRepository,
    private val evidenceRepo: EvidenceRepository,
    private val settings: ScanSettingsRepository,
) : ViewModel() {
    private val _state = MutableStateFlow(WalkUiState())
    val state: StateFlow<WalkUiState> = _state.asStateFlow()

    private val stepCounter = StepCounter(context)
    private val location = LocationFeed(context)
    private val pacer = com.sakshyam.agribot.featurescan.scan.FramePacer(context)
    private val retention = com.sakshyam.agribot.domain.logic.EvidenceRetentionPolicy()
    private val lock = Mutex()
    private val busy = AtomicBoolean(false)
    private val starting = AtomicBoolean(false)
    /** Incremented on every plant change; frames analysed for an older plant are dropped. */
    private val plantEpoch = AtomicInteger(0)
    private val screenActive = AtomicBoolean(false)
    private val commandBusy = AtomicBoolean(false)

    @Volatile private var info: VisionModelInfo? = null
    private var walk: RowWalkTracker? = null
    private var runId: RunId? = null
    private var fieldId: String = ""
    private var leafTracker = LeafTracker()
    private val trackPlant = HashMap<Long, Pair<Int, Int>>()
    private var evidence: PlantEvidence? = null
    private var bestFrame: VisionFrame? = null
    private var bestLeaves = 0
    private var frameIndex = 0L
    private var sequence = 0
    private val results = HashMap<Pair<Int, Int>, VerdictKind>()
    private val savedEvidenceRevision = HashMap<Pair<Int, Int>, Long>()
    private var evidenceRevision = 0L
    private val trail = GpsTrailFilter()
    @Volatile private var lastStepAt = 0L
    private val sensorLock = Any()
    private var sensorJobs: List<Job> = emptyList()

    init {
        vision.warmUp()
        restoreIfNeeded()
        viewModelScope.launch {
            vision.status.collect { s ->
                if (s is VisionStatus.Ready) info = s.info
                _state.update { it.copy(vision = s, labels = (s as? VisionStatus.Ready)?.info?.labels ?: it.labels) }
            }
        }
    }

    val stepsSupported: Boolean get() = stepCounter.isAvailable
    val stepPermission: Boolean get() = stepCounter.hasPermission

    fun retryModels() = vision.warmUp()
    fun clearError() = _state.update { it.copy(error = null) }

    fun setScreenActive(active: Boolean) {
        if (screenActive.getAndSet(active) == active) return
        plantEpoch.incrementAndGet()
        if (!active) {
            stopSensors()
            if (_state.value.started && !_state.value.finished) {
                _state.update { it.copy(paused = true, boxes = emptyList()) }
                viewModelScope.launch(Dispatchers.IO) {
                    try {
                        lock.withLock { if (_state.value.paused && !_state.value.finished) runId?.let { runs.markRunPaused(it) } }
                    } catch (e: CancellationException) { throw e }
                    catch (e: Exception) { saveFailed() }
                }
            }
        }
    }

    fun start(setup: WalkSetup) {
        if (_state.value.started || !starting.compareAndSet(false, true)) return
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val i = readyInfo()
                val proposedWalk = RowWalkTracker(setup.rows, setup.plantsPerRow, setup.plantSpacingM, setup.strideM)
                val fid = FieldId(UUID.randomUUID().toString())
                val rowsList = (1..setup.rows).map { r -> FieldRow(RowId(r.toString()), r, setup.plantsPerRow, setup.plantSpacingM, 1.0, r.toDouble()) }
                layouts.saveLayout(FieldLayout(fid, setup.fieldName, rowsList.first().id, rowsList, 1, 1, 0.0, Instant.now()))
                layouts.setActiveLayout(fid)
                val rid = RunId(UUID.randomUUID().toString())
                runs.createRun(RunConfig(rid, RecordingMode.WALK_ROWS, fid, 0, i.bundleId))
                lock.withLock {
                    runId = rid; fieldId = fid.value
                    walk = proposedWalk
                    evidence = PlantEvidence(i.labels.size, i.healthyIndex, i.otherIndex, i.referenceDiseasePriority, i.primaryPolicy)
                    saved[K_SETUP] = arrayListOf(setup.fieldName, setup.rows.toString(), setup.plantsPerRow.toString(), setup.plantSpacingM.toString(), setup.strideM.toString())
                    saved[K_RUN] = rid.value; saved[K_FIELD] = fid.value
                    publish { it.copy(started = true, paused = !screenActive.get(), runId = rid.value, rows = setup.rows, plantsPerRow = setup.plantsPerRow, stepsAvailable = stepsUsable) }
                    if (_state.value.paused) runs.markRunPaused(rid) else startSensors()
                }
            }.onFailure { e ->
                if (e is CancellationException) throw e
                starting.set(false); _state.update { it.copy(error = WalkError.START_FAILED) }
            }
        }
    }

    private fun startSensors() = synchronized(sensorLock) {
        stopSensors()
        if (!screenActive.get() || _state.value.paused || _state.value.finished) return@synchronized
        _state.update { it.copy(stepsAvailable = stepsUsable) }
        sensorJobs = listOf(
            viewModelScope.launch(Dispatchers.Default) {
                stepCounter.steps().catch { e ->
                    if (e is CancellationException) throw e
                    _state.update { it.copy(stepsAvailable = false) }
                }.collect {
                    try {
                        lock.withLock {
                            if (!screenActive.get() || _state.value.paused || _state.value.finished) return@withLock
                            lastStepAt = System.currentTimeMillis()
                            val proposed = walk?.copy() ?: return@withLock
                            proposed.onStep()
                            commitMovementLocked(proposed, resetTracks = false)
                        }
                    } catch (e: CancellationException) { throw e }
                    catch (e: Exception) { saveFailed() }
                }
            },
            viewModelScope.launch(Dispatchers.Default) {
                location.fixes().catch { e ->
                    if (e is CancellationException) throw e
                    // Manual plant navigation remains usable when GPS is revoked/unavailable.
                }.collect { fix ->
                    val walking = !stepsUsable || System.currentTimeMillis() - lastStepAt < 5_000
                    lock.withLock {
                        if (screenActive.get() && !_state.value.paused && !_state.value.finished) trail.offer(fix, walking)
                    }
                }
            },
        )
    }

    fun isReady(): Boolean = screenActive.get() && _state.value.vision is VisionStatus.Ready && _state.value.started && !_state.value.paused && !_state.value.finished && !_state.value.saving && !busy.get() && pacer.tryAcquire()

    fun onFrame(frame: VisionFrame) {
        if (!screenActive.get() || _state.value.vision !is VisionStatus.Ready || _state.value.paused || _state.value.finished || _state.value.saving) return
        if (!busy.compareAndSet(false, true)) return
        val epoch = plantEpoch.get()
        viewModelScope.launch(Dispatchers.Default) {
            try {
                if (!screenActive.get() || epoch != plantEpoch.get() || _state.value.paused) return@launch
                val i = info ?: return@launch
                val checked = vision.analyzeUsable(frame, i.primaryPolicy?.maximumLeaves ?: MAX_LEAVES)
                val q = FrameTools.quality(checked.quality)
                val a = checked.analysis
                lock.withLock {
                    val w = walk ?: return@withLock
                    val ev = evidence ?: return@withLock
                    if (!screenActive.get() || _state.value.paused || _state.value.finished || epoch != plantEpoch.get()) return@withLock
                    if (a == null) {
                        publish { it.copy(needsClearView = true, boxes = emptyList(), error = null) }
                        return@withLock
                    }
                    frameIndex++
                    val here = w.rowIndex to w.plantNumber
                    val obs = a.leaves.map { LeafObservation(it.box, it.detectorScore, it.probabilities, q.weight, it.referenceProbabilities) }
                    val ids = leafTracker.update(frameIndex, obs)
                    var used = 0
                    obs.forEachIndexed { k, o ->
                        // A leaf belongs to the plant where it first appeared, so leaves of the previous plant
                        // still in view are not counted for the next one.
                        if (trackPlant.getOrPut(ids[k]) { here } == here) { ev.add(ids[k], if (i.primaryPolicy != null) frame.timestampNanos else frameIndex, o); used++ }
                    }
                    if (used > 0 && used == obs.size) a.primaryEvidence?.let { ev.addPrimaryFrame(frame.timestampNanos, it) }
                    else ev.clearPrimaryFrames()
                    if (used > 0) evidenceRevision++
                    if (used >= bestLeaves && used > 0) { bestFrame = frame; bestLeaves = used }
                    val v = ev.verdict()
                    publish { it.copy(error = null, needsClearView = false, boxes = a.leaves.map { l -> LeafBox.of(l, a.frameWidth, a.frameHeight, i) }, currentLeaves = ev.leavesSeen, currentVerdict = v.takeIf { x -> x.kind != VerdictKind.NEED_MORE_VIEWS }) }
                }
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                if (epoch == plantEpoch.get()) _state.update { it.copy(error = WalkError.ANALYSIS_FAILED) }
            } finally {
                busy.set(false)
            }
        }
    }

    // A manual move means "I am at this plant now": leaves in view belong to the new plant.
    fun nextPlant() = move { it.nextPlant() }
    fun previousPlant() = move { it.previousPlant() }
    fun nextRow() = move { it.nextRow() }

    fun togglePause() {
        command {
            val id = runId ?: return@command
            plantEpoch.incrementAndGet()
            if (_state.value.paused) {
                if (!screenActive.get()) return@command
                runs.markRunRecording(id)
                _state.update { it.copy(paused = false, boxes = emptyList()) }
                startSensors()
            } else {
                _state.update { it.copy(paused = true, boxes = emptyList()) }
                stopSensors()
                runs.markRunPaused(id)
            }
        }
    }

    fun finish() {
        command {
            val w = walk ?: return@command
            _state.update { it.copy(paused = true, boxes = emptyList()) }
            plantEpoch.incrementAndGet()
            stopSensors()
            recordLocked(listOf(w.rowIndex to w.plantNumber))
            val id = runId ?: return@command
            if (trail.points.isNotEmpty()) {
                val payload = buildJsonObject {
                    put("length_m", trail.lengthM())
                    put("points", buildJsonArray {
                        trail.points.forEach { p -> add(buildJsonObject { put("lat", p.latitude); put("lon", p.longitude); put("acc", p.accuracyM); put("t", p.timeMs) }) }
                    })
                }
                runs.appendEvent(RunEvent(id.value + "_gps_trail", id, Instant.now(), "gps_trail", payload.toString()))
            }
            runs.markRunCompleted(id)
            saved.remove<String>(K_RUN)
            publish { it.copy(finished = true, boxes = emptyList()) }
        }
    }

    private fun command(block: suspend () -> Unit) {
        if (!commandBusy.compareAndSet(false, true)) return
        _state.update { it.copy(saving = true) }
        viewModelScope.launch(Dispatchers.IO) {
            try {
                lock.withLock {
                    if (_state.value.finished) return@withLock
                    block()
                    _state.update { it.copy(error = null) }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { saveFailed() }
            finally {
                commandBusy.set(false)
                _state.update { it.copy(saving = false) }
            }
        }
    }

    private fun move(change: (RowWalkTracker) -> Boolean) = command {
        val proposed = walk?.copy() ?: return@command
        if (change(proposed)) commitMovementLocked(proposed, resetTracks = true)
    }

    /** Commit observations before adopting or publishing a proposed position. Caller holds [lock]. */
    private suspend fun commitMovementLocked(proposed: RowWalkTracker, resetTracks: Boolean) {
        val before = walk ?: return
        val changed = before.rowIndex != proposed.rowIndex || before.plantNumber != proposed.plantNumber
        if (changed) {
            val positions = mutableListOf(before.rowIndex to before.plantNumber)
            if (before.rowIndex == proposed.rowIndex) {
                for (missed in (before.plantNumber + 1) until proposed.plantNumber) positions += before.rowIndex to missed
            }
            recordLocked(positions)
        }
        walk = proposed
        if (changed) resetEvidenceLocked()
        if (resetTracks) { leafTracker.reset(); trackPlant.clear() }
        publish { it }
    }

    private fun resetEvidenceLocked() {
        val i = info ?: return
        evidence = PlantEvidence(i.labels.size, i.healthyIndex, i.otherIndex, i.referenceDiseasePriority, i.primaryPolicy)
        bestFrame = null; bestLeaves = 0
        evidenceRevision++
        plantEpoch.incrementAndGet()
        publish { it.copy(currentLeaves = 0, currentVerdict = null) }
    }

    private suspend fun recordLocked(positions: List<Pair<Int, Int>>) {
        val ev = evidence ?: return
        val i = info ?: return
        val id = runId ?: return
        val pending = mutableListOf<Pair<RecordedDecision, PlantVerdict>>()
        for ((index, key) in positions.withIndex()) {
            val (row, plant) = key
            val v = if (index == 0) ev.verdict() else PlantEvidence(i.labels.size, i.healthyIndex, i.otherIndex, i.referenceDiseasePriority, i.primaryPolicy).verdict()
            // Passing back over a plant must not replace a real verdict with "not seen" / "too few views".
            if (v.kind == VerdictKind.NEED_MORE_VIEWS && results.containsKey(key)) continue
            if (index == 0 && savedEvidenceRevision[key] == evidenceRevision) continue
            val decisionId = DecisionId(UUID.randomUUID().toString())
            val fix = trail.points.lastOrNull()
            var decision = VerdictRecords.toDecision(decisionId, id, sequence + pending.size + 1, RecordingMode.WALK_ROWS, PlantPlace(fieldId, row, plant), v, i.labels,
                i.bundleId, Instant.now(), null, fix?.latitude, fix?.longitude, fix?.accuracyM)
            // Photos only for plants that need attention or a second look, within the per-run storage cap.
            val frame = if (index == 0) bestFrame else null
            if (frame != null && v.leavesSeen > 0) {
                val plan = retention.shouldCapture(decision, evidenceRepo.retentionSnapshot(id), settings.observeSettings().first().saveEvidenceFrames)
                if (plan.capture) {
                    val captured = evidenceRepo.saveEvidence(id, decisionId, FrameTools.toAnalysisFrame(frame), "walk")
                    decision = decision.copy(evidencePath = captured.path, evidenceStatus = captured.status)
                } else {
                    decision = decision.copy(evidenceStatus = plan.status)
                }
            }
            pending += decision to v
        }
        if (pending.isEmpty()) return
        try {
            runs.appendDecisions(pending.map { it.first })
        } catch (e: Exception) {
            if (e !is CancellationException) pending.forEach { (d, _) -> runCatching { evidenceRepo.deleteEvidence(id, d.id) } }
            throw e
        }
        sequence += pending.size
        saved[K_SEQ] = sequence
        pending.forEach { (d, verdict) -> results[d.rowIndex!! to d.plantNumber!!] = verdict.kind }
        pending.firstOrNull { (d, _) -> d.rowIndex == positions.first().first && d.plantNumber == positions.first().second }?.let { (d, verdict) ->
            savedEvidenceRevision[positions.first()] = evidenceRevision
            _state.update { it.copy(lastSaved = d.plantNumber!! to verdict) }
        }
    }

    private fun saveFailed() {
        plantEpoch.incrementAndGet()
        _state.update { it.copy(paused = true, boxes = emptyList(), error = WalkError.SAVE_FAILED) }
        stopSensors()
        viewModelScope.launch(Dispatchers.IO) {
            try {
                lock.withLock {
                    if (_state.value.paused && !_state.value.finished) runId?.let { runs.markRunPaused(it) }
                }
            } catch (e: CancellationException) { throw e }
            catch (_: Exception) { /* The visible save error remains; position/evidence are retained. */ }
        }
    }

    private fun stopSensors() = synchronized(sensorLock) {
        sensorJobs.forEach { it.cancel() }
        sensorJobs = emptyList()
    }

    private suspend fun readyInfo(): VisionModelInfo {
        val status = vision.status.first { it !is VisionStatus.Loading }
        check(status is VisionStatus.Ready) { "Models are unavailable" }
        return status.info.also { info = it }
    }

    private fun publish(f: (WalkUiState) -> WalkUiState) {
        val w = walk
        if (w != null) { saved[K_ROW] = w.rowIndex; saved[K_DIST] = w.distanceM; saved[K_STEPS] = w.steps }
        _state.update { s ->
            val base = f(s)
            if (w == null) base else base.copy(
                row = w.rowIndex, plant = w.plantNumber,
                rowResults = (1..w.plantsPerRow).map { p -> results[w.rowIndex to p] },
            )
        }
    }

    override fun onCleared() {
        stopSensors()
    }

    private val stepsUsable: Boolean get() = stepCounter.isAvailable && stepCounter.hasPermission

    /** After process death, continue the same run at the saved row/plant instead of losing it. */
    private fun restoreIfNeeded() {
        val rid = saved.get<String>(K_RUN) ?: return
        val setup = saved.get<ArrayList<String>>(K_SETUP) ?: return
        starting.set(true)
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val restoredRun = checkNotNull(runs.runById(RunId(rid))) { "Saved walk is missing" }
                if (restoredRun.state == RunState.COMPLETED) {
                    saved.remove<String>(K_RUN)
                    _state.update { it.copy(started = true, finished = true, runId = rid) }
                    return@launch
                }
                val i = readyInfo()
                val decisions = runs.decisionsForRun(RunId(rid))
                lock.withLock {
                    runId = RunId(rid); fieldId = saved.get<String>(K_FIELD).orEmpty()
                    walk = RowWalkTracker(setup[1].toInt(), setup[2].toInt(), setup[3].toDouble(), setup[4].toDouble()).also {
                        it.restore(saved.get<Int>(K_ROW) ?: 1, saved.get<Double>(K_DIST) ?: 0.0, saved.get<Int>(K_STEPS) ?: 0)
                    }
                    sequence = maxOf(saved.get<Int>(K_SEQ) ?: 0, decisions.maxOfOrNull { it.sequence } ?: 0)
                    decisions.groupBy { (it.rowIndex ?: 0) to (it.plantNumber ?: 0) }.forEach { (k, d) ->
                        results[k] = d.maxBy { it.sequence }.toVerdict().first.kind
                    }
                    evidence = PlantEvidence(i.labels.size, i.healthyIndex, i.otherIndex, i.referenceDiseasePriority, i.primaryPolicy)
                    runs.markRunPaused(RunId(rid))
                    publish { it.copy(started = true, paused = true, runId = rid, rows = setup[1].toInt(), plantsPerRow = setup[2].toInt(), stepsAvailable = stepsUsable) }
                }
            }.onFailure {
                if (it is CancellationException) throw it
                starting.set(false); _state.update { it.copy(error = WalkError.START_FAILED) }
            }
        }
    }

    companion object {
        private const val MAX_LEAVES = 4
        private const val K_RUN = "walk_run"
        private const val K_FIELD = "walk_field"
        private const val K_SETUP = "walk_setup"
        private const val K_ROW = "walk_row"
        private const val K_DIST = "walk_dist"
        private const val K_STEPS = "walk_steps"
        private const val K_SEQ = "walk_seq"
    }
}
