package com.sakshyam.agribot.featurescan.walk

import android.content.Context
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
import com.sakshyam.agribot.featurescan.scan.FrameTools
import com.sakshyam.agribot.featurescan.scan.LocationFeed
import com.sakshyam.agribot.featurescan.scan.StepCounter
import com.sakshyam.agribot.featurescan.ui.LeafBox
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import java.time.Instant
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
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
    /** Result per plant of the current row (index = plant - 1), null = not recorded yet. */
    val rowResults: List<VerdictKind?> = emptyList(),
    val lastSaved: Pair<Int, PlantVerdict>? = null,
    val labels: List<String> = emptyList(),
    val error: String? = null,
)

@HiltViewModel
class WalkRowsViewModel @Inject constructor(
    @ApplicationContext context: Context,
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
    private val trail = GpsTrailFilter()
    @Volatile private var lastStepAt = 0L
    private var sensorJobs: List<Job> = emptyList()

    init {
        vision.warmUp()
        viewModelScope.launch {
            vision.status.collect { s ->
                if (s is VisionStatus.Ready) info = s.info
                _state.update { it.copy(vision = s, labels = (s as? VisionStatus.Ready)?.info?.labels ?: it.labels) }
            }
        }
    }

    val stepsSupported: Boolean get() = stepCounter.isAvailable
    val stepPermission: Boolean get() = stepCounter.hasPermission

    fun start(setup: WalkSetup) {
        if (_state.value.started) return
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val i = info ?: vision.status.first { it is VisionStatus.Ready }.let { (it as VisionStatus.Ready).info.also { x -> info = x } }
                val fid = FieldId(UUID.randomUUID().toString())
                val rowsList = (1..setup.rows).map { r -> FieldRow(RowId(r.toString()), r, setup.plantsPerRow, setup.plantSpacingM, 1.0, r.toDouble()) }
                layouts.saveLayout(FieldLayout(fid, setup.fieldName, rowsList.first().id, rowsList, 1, 1, 0.0, Instant.now()))
                layouts.setActiveLayout(fid)
                val rid = RunId(UUID.randomUUID().toString())
                runs.createRun(RunConfig(rid, RecordingMode.WALK_ROWS, fid, 0, i.bundleId))
                lock.withLock {
                    runId = rid; fieldId = fid.value
                    walk = RowWalkTracker(setup.rows, setup.plantsPerRow, setup.plantSpacingM, setup.strideM)
                    evidence = PlantEvidence(i.labels.size, i.healthyIndex, i.otherIndex)
                    publish { it.copy(started = true, runId = rid.value, rows = setup.rows, plantsPerRow = setup.plantsPerRow, stepsAvailable = stepCounter.isAvailable && stepCounter.hasPermission) }
                }
                startSensors()
            }.onFailure { e -> _state.update { it.copy(error = e.message) } }
        }
    }

    private fun startSensors() {
        sensorJobs.forEach { it.cancel() }
        sensorJobs = listOf(
            viewModelScope.launch(Dispatchers.Default) {
                stepCounter.steps().collect {
                    lock.withLock {
                        if (_state.value.paused || _state.value.finished) return@withLock
                        lastStepAt = System.currentTimeMillis()
                        val w = walk ?: return@withLock
                        val before = w.plantNumber
                        w.onStep()
                        if (w.plantNumber != before) finalizeAndMoveLocked(w.rowIndex, before)
                    }
                }
            },
            viewModelScope.launch(Dispatchers.Default) {
                location.fixes().collect { fix ->
                    val walking = !stepCounter.isAvailable || System.currentTimeMillis() - lastStepAt < 5_000
                    lock.withLock { trail.offer(fix, walking) }
                }
            },
        )
    }

    fun isReady(): Boolean = _state.value.started && !_state.value.paused && !_state.value.finished && !busy.get() && pacer.tryAcquire()

    fun onFrame(frame: VisionFrame) {
        if (!busy.compareAndSet(false, true)) return
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val i = info ?: return@launch
                val q = FrameTools.quality(frame)
                val a = vision.analyze(frame, MAX_LEAVES)
                lock.withLock {
                    val w = walk ?: return@withLock
                    val ev = evidence ?: return@withLock
                    if (_state.value.paused || _state.value.finished) return@withLock
                    frameIndex++
                    val here = w.rowIndex to w.plantNumber
                    val obs = a.leaves.map { LeafObservation(it.box, it.detectorScore, it.probabilities, q.weight) }
                    val ids = leafTracker.update(frameIndex, obs)
                    var used = 0
                    obs.forEachIndexed { k, o ->
                        // A leaf belongs to the plant where it first appeared, so leaves of the previous plant
                        // still in view are not counted for the next one.
                        if (trackPlant.getOrPut(ids[k]) { here } == here) { ev.add(ids[k], frameIndex, o); used++ }
                    }
                    if (used >= bestLeaves && used > 0) { bestFrame = frame; bestLeaves = used }
                    val v = ev.verdict()
                    publish { it.copy(boxes = a.leaves.map { l -> LeafBox.of(l, a.frameWidth, a.frameHeight, i) }, currentLeaves = ev.leavesSeen, currentVerdict = v.takeIf { x -> x.kind != VerdictKind.NEED_MORE_VIEWS }) }
                }
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message) }
            } finally {
                busy.set(false)
            }
        }
    }

    fun nextPlant() = move { w -> val p = w.plantNumber; if (w.nextPlant()) finalizeAndMoveLocked(w.rowIndex, p) }
    fun previousPlant() = move { w -> val p = w.plantNumber; if (w.previousPlant()) finalizeAndMoveLocked(w.rowIndex, p) }

    fun nextRow() = move { w ->
        val r = w.rowIndex; val p = w.plantNumber
        if (w.nextRow()) {
            finalizeAndMoveLocked(r, p)
            leafTracker.reset(); trackPlant.clear()
        }
    }

    fun togglePause() {
        viewModelScope.launch {
            val id = runId ?: return@launch
            val paused = !_state.value.paused
            _state.update { it.copy(paused = paused, boxes = emptyList()) }
            if (paused) runs.markRunPaused(id) else runs.markRunRecording(id)
        }
    }

    fun finish() {
        viewModelScope.launch(Dispatchers.IO) {
            lock.withLock {
                val w = walk ?: return@withLock
                if (_state.value.finished) return@withLock
                finalizeLocked(w.rowIndex, w.plantNumber)
                sensorJobs.forEach { it.cancel() }
                val id = runId ?: return@withLock
                if (trail.points.isNotEmpty()) {
                    val payload = buildJsonObject {
                        put("length_m", trail.lengthM())
                        put("points", buildJsonArray {
                            trail.points.forEach { p -> add(buildJsonObject { put("lat", p.latitude); put("lon", p.longitude); put("acc", p.accuracyM); put("t", p.timeMs) }) }
                        })
                    }
                    runs.appendEvent(RunEvent(UUID.randomUUID().toString(), id, Instant.now(), "gps_trail", payload.toString()))
                }
                runs.markRunCompleted(id)
                publish { it.copy(finished = true, boxes = emptyList()) }
            }
        }
    }

    private fun move(block: suspend (RowWalkTracker) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            lock.withLock {
                val w = walk ?: return@withLock
                if (_state.value.finished) return@withLock
                block(w)
                publish { it }
            }
        }
    }

    /** Records [row]/[plant] and starts fresh evidence for the tracker's current plant. Caller holds [lock]. */
    private suspend fun finalizeAndMoveLocked(row: Int, plant: Int) {
        finalizeLocked(row, plant)
        val i = info ?: return
        evidence = PlantEvidence(i.labels.size, i.healthyIndex, i.otherIndex)
        bestFrame = null; bestLeaves = 0
        publish { it.copy(currentLeaves = 0, currentVerdict = null) }
    }

    private suspend fun finalizeLocked(row: Int, plant: Int) {
        val ev = evidence ?: return
        val i = info ?: return
        val id = runId ?: return
        val key = row to plant
        // Never overwrite a real verdict with "not seen" when the farmer only passed back over a plant.
        if (ev.observations == 0 && results.containsKey(key)) return
        val v = ev.verdict()
        val decisionId = DecisionId(UUID.randomUUID().toString())
        val fix = trail.points.lastOrNull()
        sequence++
        var decision = VerdictRecords.toDecision(decisionId, id, sequence, RecordingMode.WALK_ROWS, PlantPlace(fieldId, row, plant), v, i.labels,
            i.bundleId, Instant.now(), null, fix?.latitude, fix?.longitude, fix?.accuracyM)
        // Photos only for plants that need attention or a second look, within the per-run storage cap.
        val frame = bestFrame
        if (frame != null && v.leavesSeen > 0) {
            val plan = retention.shouldCapture(decision, evidenceRepo.retentionSnapshot(id), settings.observeSettings().first().saveEvidenceFrames)
            if (plan.capture) {
                val saved = runCatching { evidenceRepo.saveEvidence(id, decisionId, FrameTools.toAnalysisFrame(frame), "walk") }.getOrNull()
                decision = decision.copy(evidencePath = saved?.path, evidenceStatus = saved?.status)
            } else {
                decision = decision.copy(evidenceStatus = plan.status)
            }
        }
        runs.appendDecision(decision)
        results[key] = v.kind
        _state.update { it.copy(lastSaved = plant to v) }
    }

    private fun publish(f: (WalkUiState) -> WalkUiState) {
        val w = walk
        _state.update { s ->
            val base = f(s)
            if (w == null) base else base.copy(
                row = w.rowIndex, plant = w.plantNumber,
                rowResults = (1..w.plantsPerRow).map { p -> results[w.rowIndex to p] },
            )
        }
    }

    override fun onCleared() {
        sensorJobs.forEach { it.cancel() }
    }

    companion object {
        private const val MAX_LEAVES = 4
    }
}
