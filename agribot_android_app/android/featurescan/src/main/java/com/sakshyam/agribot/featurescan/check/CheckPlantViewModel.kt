package com.sakshyam.agribot.featurescan.check

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RunConfig
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.repository.EvidenceRepository
import com.sakshyam.agribot.domain.repository.RunRepository
import com.sakshyam.agribot.domain.repository.ScanSettingsRepository
import com.sakshyam.agribot.domain.scan.LeafObservation
import com.sakshyam.agribot.domain.scan.LeafTracker
import com.sakshyam.agribot.domain.scan.LeafVision
import com.sakshyam.agribot.domain.scan.PlantEvidence
import com.sakshyam.agribot.domain.scan.PlantPlace
import com.sakshyam.agribot.domain.scan.PlantVerdict
import com.sakshyam.agribot.domain.scan.VerdictKind
import com.sakshyam.agribot.domain.scan.VerdictRecords
import com.sakshyam.agribot.domain.scan.VisionFrame
import com.sakshyam.agribot.domain.scan.VisionModelInfo
import com.sakshyam.agribot.domain.scan.VisionStatus
import com.sakshyam.agribot.featurescan.scan.FrameTools
import com.sakshyam.agribot.featurescan.ui.LeafBox
import dagger.hilt.android.lifecycle.HiltViewModel
import java.time.Instant
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

enum class CheckHint { FIND_LEAVES, MOVE_CLOSER, HOLD_STEADY, TOO_DARK, NO_LEAVES }

data class CheckUiState(
    val vision: VisionStatus = VisionStatus.Loading,
    val scanning: Boolean = true,
    val hint: CheckHint = CheckHint.FIND_LEAVES,
    val boxes: List<LeafBox> = emptyList(),
    val leavesSeen: Int = 0,
    val framesUsed: Int = 0,
    /** 0..1 progress toward a decision, for the progress ring. */
    val progress: Float = 0f,
    val verdict: PlantVerdict? = null,
    val labels: List<String> = emptyList(),
    val saving: Boolean = false,
    val savedRunId: String? = null,
    val error: String? = null,
)

@HiltViewModel
class CheckPlantViewModel @Inject constructor(
    @dagger.hilt.android.qualifiers.ApplicationContext context: android.content.Context,
    private val vision: LeafVision,
    private val runs: RunRepository,
    private val evidenceRepo: EvidenceRepository,
    private val settings: ScanSettingsRepository,
) : ViewModel() {
    private val _state = MutableStateFlow(CheckUiState())
    val state: StateFlow<CheckUiState> = _state.asStateFlow()

    private val busy = AtomicBoolean(false)
    private val pacer = com.sakshyam.agribot.featurescan.scan.FramePacer(context)
    private val generation = AtomicInteger(0)
    private val lock = Mutex()
    @Volatile private var info: VisionModelInfo? = null
    private var tracker = LeafTracker()
    private var evidence: PlantEvidence? = null
    private var frameIndex = 0L
    private var framesWithoutLeaves = 0
    private var stableFrames = 0
    private var lastKey: Pair<VerdictKind, Int?>? = null
    @Volatile private var bestFrame: VisionFrame? = null
    private var bestFrameLeaves = 0

    init {
        vision.warmUp()
        viewModelScope.launch {
            vision.status.collect { s ->
                if (s is VisionStatus.Ready) info = s.info
                _state.update { it.copy(vision = s, labels = (s as? VisionStatus.Ready)?.info?.labels ?: it.labels) }
            }
        }
    }

    /** Camera thread: only accept a frame when nothing is in flight and we are still scanning. */
    fun isReady(): Boolean = _state.value.scanning && _state.value.vision is VisionStatus.Ready && !busy.get() && pacer.tryAcquire()

    fun onFrame(frame: VisionFrame) {
        if (!busy.compareAndSet(false, true)) return
        val gen = generation.get()
        viewModelScope.launch(Dispatchers.Default) {
            try {
                process(frame, gen)
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message) }
            } finally {
                busy.set(false)
            }
        }
    }

    private suspend fun process(frame: VisionFrame, gen: Int) {
        val i = info ?: return
        val quality = FrameTools.quality(frame)
        val analysis = vision.analyze(frame, MAX_LEAVES)
        lock.withLock {
            if (gen != generation.get() || !_state.value.scanning) return
            val ev = evidence ?: PlantEvidence(i.labels.size, i.healthyIndex, i.otherIndex).also { evidence = it }
            val boxes = analysis.leaves.map { LeafBox.of(it, analysis.frameWidth, analysis.frameHeight, i) }
            if (analysis.leaves.isEmpty()) {
                framesWithoutLeaves++
                val hint = when {
                    quality.tooDark -> CheckHint.TOO_DARK
                    framesWithoutLeaves > NO_LEAF_FRAMES_BEFORE_HELP -> CheckHint.NO_LEAVES
                    else -> CheckHint.FIND_LEAVES
                }
                _state.update { it.copy(hint = hint, boxes = emptyList()) }
                return
            }
            framesWithoutLeaves = 0
            frameIndex++
            val obs = analysis.leaves.map { LeafObservation(it.box, it.detectorScore, it.probabilities, quality.weight) }
            val ids = tracker.update(frameIndex, obs)
            obs.forEachIndexed { k, o -> ev.add(ids[k], frameIndex, o) }
            if (analysis.leaves.size >= bestFrameLeaves) { bestFrame = frame; bestFrameLeaves = analysis.leaves.size }

            val v = ev.verdict()
            val key = v.kind to v.labelIndex
            stableFrames = if (key == lastKey) stableFrames + 1 else 0
            lastKey = key
            val smallLeaves = analysis.leaves.all { (it.box.width * it.box.height) / (analysis.frameWidth.toFloat() * analysis.frameHeight) < 0.01f }
            val decided = v.kind != VerdictKind.NEED_MORE_VIEWS &&
                ((v.kind != VerdictKind.UNSURE && stableFrames >= STABLE_FRAMES) || ev.framesUsed >= MAX_FRAMES)
            val progress = (ev.framesUsed.toFloat() / MIN_FRAMES_FOR_PROGRESS).coerceAtMost(0.95f)
            _state.update {
                it.copy(
                    boxes = boxes, leavesSeen = ev.leavesSeen, framesUsed = ev.framesUsed,
                    hint = if (quality.tooDark) CheckHint.TOO_DARK else if (smallLeaves) CheckHint.MOVE_CLOSER else CheckHint.HOLD_STEADY,
                    progress = if (decided) 1f else progress,
                    verdict = if (decided) v else null, scanning = !decided,
                )
            }
        }
    }

    fun checkAgain() {
        viewModelScope.launch {
            lock.withLock {
                generation.incrementAndGet()
                tracker = LeafTracker(); evidence = null; frameIndex = 0; framesWithoutLeaves = 0
                stableFrames = 0; lastKey = null; bestFrame = null; bestFrameLeaves = 0
                _state.update { CheckUiState(vision = it.vision, labels = it.labels) }
            }
        }
    }

    fun save() {
        val s = _state.value
        val v = s.verdict ?: return
        val i = info ?: return
        if (s.saving || s.savedRunId != null) return
        _state.update { it.copy(saving = true) }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val runId = RunId(UUID.randomUUID().toString())
                runs.createRun(RunConfig(runId, RecordingMode.CHECK_PLANT, FieldId(QUICK_CHECK_FIELD), 0, i.bundleId))
                val decisionId = DecisionId(UUID.randomUUID().toString())
                val keepPhoto = settings.observeSettings().first().saveEvidenceFrames
                val photo = if (keepPhoto) bestFrame?.let { f -> evidenceRepo.saveEvidence(runId, decisionId, FrameTools.toAnalysisFrame(f), "check").path } else null
                runs.appendDecision(
                    VerdictRecords.toDecision(decisionId, runId, 1, RecordingMode.CHECK_PLANT, PlantPlace(QUICK_CHECK_FIELD, null, null),
                        v, i.labels, i.bundleId, Instant.now(), evidencePath = photo),
                )
                runs.markRunCompleted(runId)
                runId.value
            }.onSuccess { id -> _state.update { it.copy(saving = false, savedRunId = id) } }
                .onFailure { e -> _state.update { it.copy(saving = false, error = e.message) } }
        }
    }

    fun clearError() = _state.update { it.copy(error = null) }

    companion object {
        const val QUICK_CHECK_FIELD = "quick-check"
        private const val MAX_LEAVES = 6
        private const val STABLE_FRAMES = 2
        private const val MAX_FRAMES = 14
        private const val MIN_FRAMES_FOR_PROGRESS = 6f
        private const val NO_LEAF_FRAMES_BEFORE_HELP = 15
    }
}
