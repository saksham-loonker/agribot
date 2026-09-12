package com.sakshyam.agribot.featurescan

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sakshyam.agribot.domain.logic.EvidenceRetentionPolicy
import com.sakshyam.agribot.domain.logic.ExportSerializer
import com.sakshyam.agribot.domain.logic.FarmerConfigImporter
import com.sakshyam.agribot.domain.logic.FieldLayoutEditor
import com.sakshyam.agribot.domain.logic.FrameQualityAnalyzer
import com.sakshyam.agribot.domain.logic.FrameQualityStatus
import com.sakshyam.agribot.domain.logic.FrontBurstProcessor
import com.sakshyam.agribot.domain.logic.FrontCorridorSetupPlanner
import com.sakshyam.agribot.domain.logic.DiagnosticsPresenter
import com.sakshyam.agribot.domain.logic.NonPlantSceneGate
import com.sakshyam.agribot.domain.logic.PlantDecisionGate
import com.sakshyam.agribot.domain.logic.RunDetailPresenter
import com.sakshyam.agribot.domain.logic.RunEventFactory
import com.sakshyam.agribot.domain.logic.RunFieldMapPresenter
import com.sakshyam.agribot.domain.logic.RunHistoryPresenter
import com.sakshyam.agribot.domain.logic.RunRecoveryPlanner
import com.sakshyam.agribot.domain.logic.ScanSettingsValidator
import com.sakshyam.agribot.domain.logic.WalkingCalibrationPolicy
import com.sakshyam.agribot.domain.logic.RunSummaryReducer
import com.sakshyam.agribot.domain.logic.SideScanMapper
import com.sakshyam.agribot.domain.model.AnalysisFrame
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionFilter
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.EvidenceCaptureResult
import com.sakshyam.agribot.domain.model.ExportedFile
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.FramePrediction
import com.sakshyam.agribot.domain.model.FrontCaptureCalibration
import com.sakshyam.agribot.domain.model.FrontBurstReview
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import com.sakshyam.agribot.domain.model.FrontOverviewDecisionDraft
import com.sakshyam.agribot.domain.model.PlantDecision
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RowId
import com.sakshyam.agribot.domain.model.RunConfig
import com.sakshyam.agribot.domain.model.RunEvent
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunState
import com.sakshyam.agribot.domain.model.ScanSettings
import com.sakshyam.agribot.domain.model.StepCalibrationStatus
import com.sakshyam.agribot.domain.model.ThermalStatus
import com.sakshyam.agribot.domain.model.ScanConstants
import com.sakshyam.agribot.domain.model.TreatmentStatus
import com.sakshyam.agribot.domain.repository.EvidenceRepository
import com.sakshyam.agribot.domain.repository.ExportRepository
import com.sakshyam.agribot.domain.repository.FieldLayoutRepository
import com.sakshyam.agribot.domain.repository.InferenceRepository
import com.sakshyam.agribot.domain.repository.RunRepository
import com.sakshyam.agribot.domain.repository.ScanSettingsRepository
import com.sakshyam.agribot.featurescan.tracking.PlantTracker
import com.sakshyam.agribot.featurescan.tracking.PlantTrackingMath
import com.sakshyam.agribot.featurescan.tracking.TrackedPlant
import com.sakshyam.agribot.ml.inference.ModelUnavailableException
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.ArrayDeque
import javax.inject.Inject
import kotlin.math.abs
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.Job
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch

@HiltViewModel
class SideScanViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val fieldLayoutRepository: FieldLayoutRepository,
    private val runRepository: RunRepository,
    private val exportRepository: ExportRepository,
    private val evidenceRepository: EvidenceRepository,
    private val inferenceRepository: InferenceRepository,
    private val scanSettingsRepository: ScanSettingsRepository,
) : ViewModel() {
    private val _state = MutableStateFlow(ScanUiState())
    val state: StateFlow<ScanUiState> = _state.asStateFlow()

    private var gate = PlantDecisionGate()
    private var pendingDecisionPlantId: String? = null
    private var mapper: SideScanMapper? = null
    private var nextScanIndex = 1
    private var latestFrame: AnalysisFrame? = null
    private var nextPlantReadyAtNanos = 0L
    private val recentFrames = ArrayDeque<AnalysisFrame>(FrontBurstCapturePolicy.TARGET_FRAME_COUNT)
    private var pendingFrontReviewWork: PendingFrontReviewWork? = null
    @Volatile private var inferenceInFlight = false
    @Volatile private var inferenceGeneration = 0L
    private var inferenceJob: Job? = null
    private val evidencePolicy = EvidenceRetentionPolicy()
    private val performanceGovernor = SideScanPerformanceGovernor()
    private var pauseAfterCurrentDecisionGroup = false
    private var scanSettings = ScanSettings()
    private var lastThermalWarningReason: String? = null
    private val plantTracker = PlantTracker(context)
    private var currentSessionId: String = "session_${System.currentTimeMillis()}"
    @Volatile private var modelBundleMismatch = false
    @Volatile private var vegetationCoverage: Float = 0f
    @Volatile private var sceneDropCount: Boolean = false
    @Volatile private var modelReadinessPassed = false

    init {
        // Sensors and location start only after the operator explicitly begins a scan.
        viewModelScope.launch {
            ensureDefaultLayout()
            recoverInterruptedRun()
            refreshRunHistory()
            refreshDiagnostics()
            fieldLayoutRepository.observeActiveLayout().collect { layout ->
                mapper = layout?.let(::SideScanMapper)
                _state.update { current ->
                    current.copy(
                        activeLayout = layout,
                        currentPosition = mapper?.position(nextScanIndex),
                        frontCorridorSetup = layout?.let {
                            FrontCorridorSetupPlanner.coerceForLayout(it, current.frontCorridorSetup)
                        },
                    )
                }
                refreshDiagnostics()
            }
        }
        
        // Update sensor data periodically
        viewModelScope.launch {
            while (true) {
                updateSensorData()
                delay(200) // Update 5 times per second
            }
        }
        
        viewModelScope.launch {
            inferenceRepository.observeModelManifest().collect { manifest ->
                val validatedBundleId = manifest?.bundleId ?: ScanConstants.DEFAULT_MODEL_BUNDLE_ID
                // Validate that the manifest bundle ID matches our expected constant
                modelBundleMismatch = manifest != null && validatedBundleId != ScanConstants.DEFAULT_MODEL_BUNDLE_ID
                _state.update { current ->
                    current.copy(
                        modelBundleId = validatedBundleId,
                        modelStatus = if (modelBundleMismatch) {
                            "Warning: Model bundle mismatch. Expected ${ScanConstants.DEFAULT_MODEL_BUNDLE_ID}, got $validatedBundleId"
                        } else if (modelReadinessPassed) {
                            "Model ready: $validatedBundleId"
                        } else {
                            current.modelStatus
                        },
                        modelReady = modelReadinessPassed && !modelBundleMismatch,
                        modelReadinessErrors = if (modelBundleMismatch) {
                            (current.modelReadinessErrors + "Model bundle ID does not match the validated app bundle").distinct()
                        } else {
                            emptyList()
                        },
                        error = if (modelBundleMismatch) {
                            "Model bundle validation failed"
                        } else {
                            current.error?.takeUnless { it == "Model bundle validation failed" }
                        },
                    )
                }
                refreshDiagnostics()
            }
        }
        viewModelScope.launch {
            inferenceRepository.observeModelReadiness().collect { readiness ->
                modelReadinessPassed = readiness.isReady
                val readinessErrors = (readiness.errors + if (modelBundleMismatch) {
                    listOf("Model bundle ID does not match the validated app bundle")
                } else {
                    emptyList()
                }).distinct()
                _state.update {
                    it.copy(
                        modelStatus = if (modelBundleMismatch) {
                            "Model blocked: bundle validation failed"
                        } else {
                            readiness.statusText
                        },
                        modelReady = readiness.isReady && !modelBundleMismatch,
                        modelLoadingProgress = readiness.progress.coerceIn(0f, 1f),
                        modelReadinessErrors = readinessErrors,
                        error = if (readiness.isReady && !modelBundleMismatch && it.error?.contains("model", ignoreCase = true) == true) {
                            null
                        } else {
                            it.error
                        },
                    )
                }
                refreshDiagnostics()
            }
        }
        viewModelScope.launch {
            scanSettingsRepository.observeSettings().collect { settings ->
                val safe = ScanSettingsValidator.sanitize(settings)
                scanSettings = safe
                if (_state.value.runState !in setOf(RunState.RECORDING, RunState.PAUSED) &&
                    _state.value.pendingFrontReview == null
                ) {
                    plantTracker.setStepLengthM(safe.stepLengthM)
                }
                inferenceRepository.setConfidenceThreshold(safe.confidenceThreshold)
                inferenceRepository.setCpuThreads(safe.cpuThreads)
                _state.update { current ->
                    current.copy(
                        selectedMode = if (current.isRecording) current.selectedMode else safe.defaultMode,
                        targetFps = if (current.isRecording) current.targetFps else safe.targetFps,
                        cpuThreads = safe.cpuThreads,
                    saveEvidenceFrames = safe.saveEvidenceFrames,
                    operatorDisplayName = safe.operatorDisplayName.orEmpty(),
                    darkMode = safe.darkMode,
                    sunMode = safe.sunMode,
                    gpsReferenceEnabled = safe.gpsReferenceEnabled,
                    showOnboarding = safe.showOnboarding,
                    cropType = safe.cropType,
                        confidenceThreshold = safe.confidenceThreshold,
                        highConfidenceThreshold = safe.highConfidenceThreshold,
                        modelBundleId = safe.modelBundleId,
                        performanceStatus = if (current.isRecording) {
                            current.performanceStatus
                        } else {
                            SideScanDefaults.performanceStatus(safe.targetFps)
                        },
                    )
                }
                refreshDiagnostics()
            }
        }
    }

    private fun updateSensorData() {
        val motion = plantTracker.motionMeasurement
        _state.update { current ->
            current.copy(
                sensorData = SensorData(
                    tilt = plantTracker.currentTilt,
                    roll = plantTracker.currentRoll,
                    pitch = plantTracker.currentPitch,
                    speedMps = motion.speed.value,
                    distanceM = motion.distance.value,
                    stepCount = motion.stepCount,
                    qualityPercent = plantTracker.sensorQualityPercent,
                    motionStatus = plantTracker.motionStatus,
                    gpsStatus = plantTracker.gpsStatus,
                    gpsAccuracyM = plantTracker.gpsAccuracyM,
                    gpsFixAgeSeconds = plantTracker.gpsFixAgeSeconds,
                    gpsDistanceM = plantTracker.gpsDistanceM,
                    gpsProvider = plantTracker.gpsProvider,
                    gpsPathPoints = plantTracker.gpsPathPoints,
                    gpsPathStatus = plantTracker.gpsPathStatus,
                    fieldWidthM = plantTracker.fieldWidthM,
                    fieldLengthM = plantTracker.fieldLengthM,
                    stepSensorAvailable = plantTracker.stepSensorAvailable,
                    motionEventsObserved = plantTracker.motionEventsObserved,
                    measurementSource = plantTracker.measurementSource,
                    distanceQuality = motion.distance.quality,
                    distanceSource = motion.distance.source,
                    distanceMessage = motion.distance.message,
                    stepLengthM = scanSettings.stepLengthM,
                    stepCalibrationStatus = scanSettings.stepCalibrationStatus,
                ),
                walkingCalibration = current.walkingCalibration.copy(
                    currentSteps = if (current.walkingCalibration.active) {
                        (plantTracker.currentStepCount - current.walkingCalibration.startStepCount).coerceAtLeast(0)
                    } else {
                        current.walkingCalibration.currentSteps
                    },
                )
            )
        }
    }

    fun setCameraPermission(granted: Boolean) {
        val previous = _state.value
        if (previous.cameraPermissionGranted && !granted && previous.activeRunId != null) {
            viewModelScope.launch {
                runRepository.appendEvent(
                    RunEventFactory.permissionLost(
                        runId = previous.activeRunId,
                        timestamp = Instant.now(),
                        permission = "camera",
                        previousState = previous.runState.name.lowercase(),
                    ),
                )
                if (previous.runState == RunState.RECORDING) {
                    runRepository.markRunPaused(previous.activeRunId)
                }
                refreshRunHistory()
                refreshDiagnostics()
            }
        }
        _state.update {
            it.copy(
                cameraPermissionGranted = granted,
                runState = if (!granted && it.runState == RunState.RECORDING) RunState.PAUSED else it.runState,
            )
        }
        refreshDiagnostics()
    }

    fun setLocationPermission(granted: Boolean, requested: Boolean = false) {
        _state.update {
            it.copy(
                locationPermissionGranted = granted,
                locationPermissionRequested = it.locationPermissionRequested || requested,
            )
        }
        if (granted && _state.value.gpsReferenceEnabled && _state.value.isRecording) {
            plantTracker.startTracking(useGpsReference = true)
        }
        refreshDiagnostics()
    }

    fun setActivityRecognitionPermission(granted: Boolean, requested: Boolean = false) {
        _state.update {
            it.copy(
                activityRecognitionPermissionGranted = granted,
                activityRecognitionPermissionRequested = it.activityRecognitionPermissionRequested || requested,
            )
        }
        if (granted && _state.value.isRecording) {
            // Re-register only the newly available step sensor. The tracker
            // keeps its accumulated path; granting the optional permission
            // must not reset the farmer's measurement.
            plantTracker.startTracking(useGpsReference = _state.value.gpsReferenceEnabled)
        }
        refreshDiagnostics()
    }

    fun selectMode(mode: RecordingMode) {
        invalidateInferenceWork()
        latestFrame = null
        recentFrames.clear()
        pendingFrontReviewWork = null
        _state.update { it.copy(selectedMode = mode, pendingFrontReview = null) }
        saveScanSettings(scanSettings.copy(defaultMode = mode))
    }

    fun adjustTargetFps(delta: Int) {
        val updated = ScanSettingsValidator.sanitize(scanSettings.copy(targetFps = scanSettings.targetFps + delta))
        saveScanSettings(updated)
    }

    fun toggleEvidenceFrames() {
        saveScanSettings(scanSettings.copy(saveEvidenceFrames = !scanSettings.saveEvidenceFrames))
    }

    fun updateOperatorDisplayName(value: String) {
        saveScanSettings(scanSettings.copy(operatorDisplayName = value))
    }

    fun setDarkMode(enabled: Boolean) {
        saveScanSettings(scanSettings.copy(darkMode = enabled))
    }

    fun setSunMode(enabled: Boolean) {
        saveScanSettings(scanSettings.copy(sunMode = enabled))
    }

    fun setGpsReferenceEnabled(enabled: Boolean) {
        saveScanSettings(scanSettings.copy(gpsReferenceEnabled = enabled))
        plantTracker.setLocationReferenceEnabled(enabled)
    }

    /**
     * Starts an explicit stride calibration without opening the camera or
     * creating a run. The operator walks a known distance and finishes it
     * from the settings card; existing plant results remain untouched.
     */
    fun startWalkingCalibration(knownDistanceM: Float) {
        val current = _state.value
        when {
            current.runState in setOf(RunState.RECORDING, RunState.PAUSED) || current.pendingFrontReview != null -> {
                _state.update {
                    it.copy(
                        walkingCalibration = it.walkingCalibration.copy(
                            active = false,
                            message = context.getString(R.string.walking_calibration_blocked),
                        ),
                    )
                }
            }
            !WalkingCalibrationPolicy.isValidKnownDistance(knownDistanceM) -> {
                _state.update {
                    it.copy(
                        walkingCalibration = it.walkingCalibration.copy(
                            active = false,
                            message = context.getString(R.string.walking_calibration_invalid_distance),
                        ),
                    )
                }
            }
            else -> {
                plantTracker.beginScan()
                plantTracker.startTracking(useGpsReference = false, tractorMounted = false)
                val startStepCount = plantTracker.currentStepCount
                _state.update {
                    it.copy(
                        error = null,
                        walkingCalibration = WalkingCalibrationUiState(
                            active = true,
                            knownDistanceM = knownDistanceM,
                            startStepCount = startStepCount,
                            currentSteps = 0,
                            message = context.getString(R.string.walking_calibration_default_message),
                        ),
                    )
                }
                updateSensorData()
            }
        }
    }

    fun finishWalkingCalibration() {
        val calibration = _state.value.walkingCalibration
        val knownDistanceM = calibration.knownDistanceM
        if (!calibration.active || knownDistanceM == null) return

        val steps = (plantTracker.currentStepCount - calibration.startStepCount).coerceAtLeast(0)
        val result = WalkingCalibrationPolicy.calibrate(knownDistanceM, steps)
        val calibratedStepLengthM: Float? = if (result.accepted) result.stepLengthM else null
        val message = when {
            result.accepted -> context.getString(R.string.walking_calibration_saved_message)
            steps < WalkingCalibrationPolicy.MIN_STEPS -> context.getString(R.string.walking_calibration_too_few_steps)
            else -> context.getString(R.string.walking_calibration_unrealistic)
        }
        plantTracker.stopTracking()
        if (calibratedStepLengthM != null) {
            val updated = ScanSettingsValidator.sanitize(
                scanSettings.copy(
                    stepLengthM = calibratedStepLengthM,
                    stepCalibrationStatus = StepCalibrationStatus.CALIBRATED,
                ),
            )
            plantTracker.setStepLengthM(updated.stepLengthM)
            saveScanSettings(updated)
        }
        _state.update {
            it.copy(
                walkingCalibration = WalkingCalibrationUiState(message = message),
                sensorData = it.sensorData.copy(
                    stepLengthM = calibratedStepLengthM ?: scanSettings.stepLengthM,
                    stepCalibrationStatus = if (calibratedStepLengthM != null) StepCalibrationStatus.CALIBRATED else scanSettings.stepCalibrationStatus,
                ),
            )
        }
        updateSensorData()
    }

    fun cancelWalkingCalibration() {
        if (!_state.value.walkingCalibration.active) return
        plantTracker.stopTracking()
        _state.update {
            it.copy(
                walkingCalibration = WalkingCalibrationUiState(message = context.getString(R.string.walking_calibration_cancelled)),
            )
        }
        updateSensorData()
    }

    fun completeOnboarding() {
        saveScanSettings(scanSettings.copy(showOnboarding = false))
    }

    fun setShowOnboarding(enabled: Boolean) {
        saveScanSettings(scanSettings.copy(showOnboarding = enabled))
    }

    fun setCropType(cropType: String?) {
        saveScanSettings(scanSettings.copy(cropType = cropType))
    }

    fun setConfidenceThreshold(threshold: Float) {
        val clamped = threshold.coerceIn(0.1f, 0.95f)
        saveScanSettings(scanSettings.copy(confidenceThreshold = clamped))
    }

    fun setModelBundleId(bundleId: String) {
        saveScanSettings(scanSettings.copy(modelBundleId = bundleId))
    }

    fun selectActiveRow(rowIdValue: String) {
        viewModelScope.launch {
            val layout = _state.value.activeLayout ?: return@launch
            val updated = FieldLayoutEditor.selectActiveRow(layout, RowId(rowIdValue))
            fieldLayoutRepository.saveLayout(updated)
            fieldLayoutRepository.setActiveLayout(updated.id)
        }
    }

    fun selectFrontLeftRow(rowIdValue: String) {
        val layout = _state.value.activeLayout ?: return
        val current = _state.value.frontCorridorSetup ?: FrontCorridorSetupPlanner.defaultFor(layout) ?: return
        _state.update {
            it.copy(frontCorridorSetup = FrontCorridorSetupPlanner.selectLeft(layout, current, rowIdValue))
        }
    }

    fun selectFrontRightRow(rowIdValue: String) {
        val layout = _state.value.activeLayout ?: return
        val current = _state.value.frontCorridorSetup ?: FrontCorridorSetupPlanner.defaultFor(layout) ?: return
        _state.update {
            it.copy(frontCorridorSetup = FrontCorridorSetupPlanner.selectRight(layout, current, rowIdValue))
        }
    }

    fun adjustFrontNearestLeft(delta: Int) {
        val layout = _state.value.activeLayout ?: return
        val current = _state.value.frontCorridorSetup ?: FrontCorridorSetupPlanner.defaultFor(layout) ?: return
        _state.update {
            it.copy(frontCorridorSetup = FrontCorridorSetupPlanner.adjustNearestLeft(layout, current, delta))
        }
    }

    fun adjustFrontNearestRight(delta: Int) {
        val layout = _state.value.activeLayout ?: return
        val current = _state.value.frontCorridorSetup ?: FrontCorridorSetupPlanner.defaultFor(layout) ?: return
        _state.update {
            it.copy(frontCorridorSetup = FrontCorridorSetupPlanner.adjustNearestRight(layout, current, delta))
        }
    }

    fun adjustActiveRowPlants(delta: Int) {
        viewModelScope.launch {
            val layout = _state.value.activeLayout ?: return@launch
            val row = layout.rows.firstOrNull { it.id == layout.activeRowId } ?: return@launch
            val updated = FieldLayoutEditor.updatePlantsPerRow(layout, row.id, row.plantsPerRow + delta)
            fieldLayoutRepository.saveLayout(updated)
            fieldLayoutRepository.setActiveLayout(updated.id)
        }
    }

    fun adjustPlantCooldown(deltaSec: Double) {
        viewModelScope.launch {
            val layout = _state.value.activeLayout ?: return@launch
            val updated = FieldLayoutEditor.updatePlantCooldown(layout, layout.plantCooldownSec + deltaSec)
            fieldLayoutRepository.saveLayout(updated)
            fieldLayoutRepository.setActiveLayout(updated.id)
        }
    }

    fun start() {
        viewModelScope.launch {
            invalidateInferenceWork()
            val current = _state.value
            if (!current.cameraPermissionGranted) {
                _state.update {
                    it.copy(
                        runState = RunState.PERMISSION_REQUIRED,
                        error = "Camera access is needed to see plants before a scan can start",
                    )
                }
                refreshDiagnostics()
                return@launch
            }
            if (!current.modelReady) {
                _state.update {
                    it.copy(
                        runState = RunState.READY,
                        error = "Plant checking is not ready yet. Open diagnostics and try again.",
                    )
                }
                refreshDiagnostics()
                return@launch
            }
            val layout = _state.value.activeLayout ?: return@launch
            if (_state.value.selectedMode == RecordingMode.FRONT_ROW_OVERVIEW) {
                startFrontOverviewPreview()
                return@launch
            }
            plantTracker.clearAll()
            plantTracker.beginScan()
            plantTracker.startTracking(useGpsReference = _state.value.gpsReferenceEnabled)
            val runId = RunId("android_${RUN_ID_FORMAT.format(Instant.now())}")
            nextScanIndex = 1
            nextPlantReadyAtNanos = 0L
            gate = PlantDecisionGate(highConfidence = _state.value.highConfidenceThreshold)
            pendingDecisionPlantId = null
            mapper = SideScanMapper(layout)
            performanceGovernor.reset(_state.value.targetFps)
            pauseAfterCurrentDecisionGroup = false
            lastThermalWarningReason = null
            runRepository.createRun(
                RunConfig(
                    runId = runId,
                    mode = _state.value.selectedMode,
                    fieldLayoutId = layout.id,
                    targetFps = _state.value.targetFps,
                    modelBundleId = _state.value.modelBundleId,
                ),
            )
            runRepository.appendEvent(
                RunEventFactory.started(
                    runId = runId,
                    timestamp = Instant.now(),
                    mode = _state.value.selectedMode.name,
                    targetFps = _state.value.targetFps,
                ),
            )
            runRepository.appendEvent(
                RunEventFactory.modelLoaded(
                    runId = runId,
                    timestamp = Instant.now(),
                    bundleId = _state.value.modelBundleId,
                    cpuDefault = true,
                ),
            )
            _state.update {
                it.copy(
                    runState = RunState.RECORDING,
                    activeRunId = runId,
                    activeRunFieldMap = RunFieldMapPresenter.map(layout, emptyList()),
                    latestDecision = null,
                    decisionsRecorded = 0,
                    trackedPlants = emptyList(),
                    sensorData = SensorData(),
                    currentPosition = mapper?.position(nextScanIndex),
                    manualSnapshotStatus = null,
                    detectionStatus = "Point at one plant",
                    error = null,
                )
            }
            refreshRunHistory()
            refreshDiagnostics()
        }
    }

    private fun startFrontOverviewPreview() {
        invalidateInferenceWork()
        plantTracker.clearAll()
        plantTracker.beginScan()
        pendingDecisionPlantId = null
        plantTracker.startTracking(useGpsReference = _state.value.gpsReferenceEnabled)
        recentFrames.clear()
        nextPlantReadyAtNanos = 0L
        _state.update {
            it.copy(
                selectedMode = RecordingMode.FRONT_ROW_OVERVIEW,
                runState = RunState.RECORDING,
                activeRunId = null,
                activeRunFieldMap = null,
                latestDecision = null,
                decisionsRecorded = 0,
                trackedPlants = emptyList(),
                sensorData = SensorData(),
                pendingFrontReview = null,
                frontOverviewStatus = "Align rows inside guide rails",
                detectionStatus = "Align the row inside the guide",
                error = null,
            )
        }
        refreshDiagnostics()
    }

    fun pause() {
        invalidateInferenceWork()
        viewModelScope.launch {
            plantTracker.stopTracking()
            pendingDecisionPlantId = null
            val current = _state.value
            val runId = current.activeRunId
            if (runId == null) {
                if (current.selectedMode == RecordingMode.FRONT_ROW_OVERVIEW && current.runState == RunState.RECORDING) {
                    _state.update {
                        it.copy(
                            runState = RunState.PAUSED,
                            frontOverviewStatus = "Front Overview paused",
                            error = null,
                        )
                    }
                    refreshDiagnostics()
                }
                return@launch
            }
            runRepository.appendEvent(
                RunEventFactory.gpsPathSnapshot(
                    runId = runId,
                    timestamp = Instant.now(),
                    status = plantTracker.gpsPathStatus,
                    provider = plantTracker.gpsProvider,
                    points = plantTracker.gpsPathPoints,
                    snapshotLabel = "pause",
                ),
            )
            runRepository.markRunPaused(runId)
            runRepository.appendEvent(
                RunEventFactory.paused(
                    runId = runId,
                    timestamp = Instant.now(),
                    decisionsRecorded = _state.value.decisionsRecorded,
                ),
            )
            _state.update { it.copy(runState = RunState.PAUSED) }
            refreshRunHistory()
            refreshDiagnostics()
        }
    }

    fun resume() {
        invalidateInferenceWork()
        viewModelScope.launch {
            val current = _state.value
            plantTracker.startTracking(useGpsReference = current.gpsReferenceEnabled)
            val runId = current.activeRunId
            if (runId == null) {
                if (current.selectedMode == RecordingMode.FRONT_ROW_OVERVIEW && current.runState == RunState.PAUSED) {
                    recentFrames.clear()
                    nextPlantReadyAtNanos = 0L
                    _state.update {
                        it.copy(
                            runState = RunState.RECORDING,
                            frontOverviewStatus = "Align rows inside guide rails (0/${FrontBurstCapturePolicy.TARGET_FRAME_COUNT} frames)",
                            error = null,
                        )
                    }
                    refreshDiagnostics()
                }
                return@launch
            }
            runRepository.markRunRecording(runId)
            runRepository.appendEvent(
                RunEventFactory.resumed(
                    runId = runId,
                    timestamp = Instant.now(),
                    nextScanIndex = nextScanIndex,
                ),
            )
            _state.update { it.copy(runState = RunState.RECORDING) }
            refreshRunHistory()
            refreshDiagnostics()
        }
    }

    fun clearTrackedPlants() {
        plantTracker.clearAll()
        _state.update { it.copy(trackedPlants = emptyList()) }
    }

    fun resetWalkMeasurement() {
        val current = _state.value
        if (current.runState !in setOf(RunState.RECORDING, RunState.PAUSED)) {
            _state.update { it.copy(error = "Start or resume a scan before resetting distance") }
            return
        }
        plantTracker.resetWalkMeasurement()
        updateSensorData()
        current.activeRunId?.let { runId ->
            viewModelScope.launch {
                runRepository.appendEvent(
                    RunEventFactory.measurementReset(
                        runId = runId,
                        timestamp = Instant.now(),
                    ),
                )
                refreshRunHistory()
                refreshDiagnostics()
            }
        }
    }

    fun setPlantFilter(filter: PlantFilter) {
        _state.update { it.copy(plantFilter = filter) }
    }

    fun setPlantSearchQuery(query: String) {
        _state.update { it.copy(plantSearchQuery = query) }
    }

    fun selectPlant(plantId: String?) {
        _state.update { it.copy(selectedPlantId = plantId) }
    }

    fun setPlantTreatment(plantId: String, status: TreatmentStatus, note: String?) {
        plantTracker.setTreatment(plantId, status, note)
        val current = _state.value
        _state.update {
            it.copy(trackedPlants = plantTracker.trackedPlantsFlow.value)
        }
        val runId = current.activeRunId ?: current.selectedRunId
        if (runId != null) {
            viewModelScope.launch {
                runRepository.updateDecisionTreatment(runId, plantId, status, note)
                refreshSelectedRunDetail()
            }
        }
    }

    fun enterDataMode() {
        if (_state.value.isRecording) pause()
        _state.update { it.copy(isDataMode = true, selectedPlantId = null) }
        _state.value.activeRunId?.let(::selectRunDetail)
    }

    fun exitDataMode() {
        _state.update { it.copy(isDataMode = false, selectedPlantId = null) }
    }

    fun onAppBackgrounded() {
        val current = _state.value
        if (current.walkingCalibration.active) {
            plantTracker.stopTracking()
            _state.update {
                it.copy(
                    walkingCalibration = WalkingCalibrationUiState(
                        message = context.getString(R.string.walking_calibration_background),
                    ),
                )
            }
            updateSensorData()
            return
        }
        if (current.runState != RunState.RECORDING) return
        invalidateInferenceWork()
        plantTracker.stopTracking()
        if (current.activeRunId == null) {
            latestFrame = null
            recentFrames.clear()
            pendingFrontReviewWork = null
            nextPlantReadyAtNanos = 0L
            _state.update {
                it.copy(
                    runState = RunState.READY,
                    pendingFrontReview = null,
                    frontOverviewStatus = "Paused when app backgrounded",
                    error = "Front Overview capture paused when the app backgrounded",
                )
            }
            refreshDiagnostics()
            return
        }
        pause()
    }

    fun stop() {
        invalidateInferenceWork()
        viewModelScope.launch {
            plantTracker.stopTracking()
            val runId = _state.value.activeRunId
            if (runId != null) {
                runRepository.appendEvent(
                    RunEventFactory.gpsPathSnapshot(
                        runId = runId,
                        timestamp = Instant.now(),
                        status = plantTracker.gpsPathStatus,
                        provider = plantTracker.gpsProvider,
                        points = plantTracker.gpsPathPoints,
                        snapshotLabel = "final",
                    ),
                )
                runRepository.appendEvent(
                    RunEventFactory.stopped(
                        runId = runId,
                        timestamp = Instant.now(),
                        decisionsRecorded = _state.value.decisionsRecorded,
                    ),
                )
                runRepository.markRunCompleted(runId)
            }
            _state.update {
                it.copy(
                    runState = if (runId == null) RunState.READY else RunState.COMPLETED,
                    pendingFrontReview = if (runId == null) null else it.pendingFrontReview,
                    frontOverviewStatus = if (runId == null) "Ready" else it.frontOverviewStatus,
                    cooldownRemainingMs = 0,
                    cooldownStatus = null,
                )
            }
            latestFrame = null
            recentFrames.clear()
            refreshRunHistory()
            refreshDiagnostics()
        }
    }

    fun acceptPrediction(prediction: FramePrediction) {
        if (!_state.value.isRecording) return
        val decision = gate.add(prediction) ?: return
        recordDecision(decision, rawLabel = prediction.rawLabel, manual = false, notes = null, original = null, auditManual = false)
        gate.reset()
    }

    private fun consumeCooldownOrShowStatus(): Boolean {
        val readyAt = nextPlantReadyAtNanos
        if (readyAt <= 0L) {
            _state.update { it.copy(cooldownRemainingMs = 0, cooldownStatus = null) }
            return true
        }
        val now = System.nanoTime()
        if (readyAt <= now) {
            nextPlantReadyAtNanos = 0L
            _state.update { it.copy(cooldownRemainingMs = 0, cooldownStatus = null) }
            return true
        }
        val remainingMs = ((readyAt - now) / 1_000_000L).coerceAtLeast(1)
        _state.update {
            it.copy(
                cooldownRemainingMs = remainingMs,
                cooldownStatus = "Next plant ready in ${remainingMs} ms",
            )
        }
        return false
    }

    private fun armCooldownAfterDecision() {
        val layout = _state.value.activeLayout
        val cooldownMs = ((layout?.plantCooldownSec ?: 2.0) * 1000.0).roundToLongCompat().coerceAtLeast(0L)
        if (cooldownMs == 0L) {
            nextPlantReadyAtNanos = 0L
            _state.update { it.copy(cooldownRemainingMs = 0, cooldownStatus = null) }
            return
        }
        nextPlantReadyAtNanos = System.nanoTime() + cooldownMs * 1_000_000L
        _state.update {
            it.copy(
                cooldownRemainingMs = cooldownMs,
                cooldownStatus = "Next plant ready in ${cooldownMs} ms",
            )
        }
        viewModelScope.launch {
            while (_state.value.isRecording && _state.value.selectedMode == RecordingMode.SIDE_SCAN) {
                val remaining = remainingCooldownMs()
                if (remaining <= 0L) break
                delay(remaining.coerceAtMost(500L))
            }
            if (_state.value.isRecording && _state.value.selectedMode == RecordingMode.SIDE_SCAN) {
                nextPlantReadyAtNanos = 0L
                _state.update { it.copy(cooldownRemainingMs = 0, cooldownStatus = null) }
            }
        }
    }

    private fun remainingCooldownMs(): Long {
        val readyAt = nextPlantReadyAtNanos
        if (readyAt <= 0L) return 0L
        return ((readyAt - System.nanoTime()) / 1_000_000L).coerceAtLeast(0L)
    }

    private fun Double.roundToLongCompat(): Long = java.lang.Math.round(this)

    private fun resetCooldownAfterManualAction() {
        nextPlantReadyAtNanos = 0L
        _state.update { it.copy(cooldownRemainingMs = 0, cooldownStatus = null) }
    }

    /** Cancel the coroutine and invalidate results that may still be running
     * inside a non-cancellable TFLite call.  The generation check is required
     * because cancelling a coroutine cannot interrupt native interpreter work.
     */
    private fun invalidateInferenceWork() {
        inferenceGeneration += 1L
        inferenceJob?.cancel()
        inferenceJob = null
        inferenceInFlight = false
    }

    private fun isInferenceCurrent(token: Long, mode: RecordingMode): Boolean =
        token == inferenceGeneration &&
            _state.value.isRecording &&
            _state.value.selectedMode == mode

    fun markManual(label: String, status: DecisionStatus = DecisionStatus.MANUAL, notes: String? = null) {
        val decision = PlantDecision(
            label = label,
            confidence = 1.0f,
            status = status,
            framesUsed = 0,
            reason = "manual_override",
        )
        recordDecision(decision, rawLabel = label, manual = true, notes = notes, original = _state.value.latestDecision, auditManual = true)
        pauseAfterCurrentDecisionGroup = false
        resetCooldownAfterManualAction()
    }

    fun onAnalysisFrame(frame: AnalysisFrame) {
        // Don't process frames when not recording — prevents auto-capture before run starts
        if (!_state.value.isRecording) return
        plantTracker.observeCameraMotion(frame)
        val quality = FrameQualityAnalyzer.evaluate(frame.width, frame.height, frame.rgbPixels)
        _state.update {
            it.copy(
                analysisWidth = frame.width,
                analysisHeight = frame.height,
                captureQualityStatus = quality.status.name.lowercase(),
                captureQualityMessage = quality.message,
                captureQualityAction = quality.action,
                detectionStatus = if (quality.status == FrameQualityStatus.RETRY) {
                    "Improve the photo before checking a plant"
                } else {
                    it.detectionStatus
                },
            )
        }
        if (quality.status == FrameQualityStatus.RETRY) {
            // Keep one rejected frame for an explicitly requested evidence
            // snapshot, but never feed it to detection, tracking, or counting.
            latestFrame = frame
            return
        }
        latestFrame = frame
        if (recentFrames.size >= FrontBurstCapturePolicy.TARGET_FRAME_COUNT) {
            recentFrames.removeFirst()
        }
        recentFrames.addLast(frame)
        _state.update {
            it.copy(
                frontOverviewStatus = if (it.selectedMode == RecordingMode.FRONT_ROW_OVERVIEW && it.isRecording) {
                    "Align rows inside guide rails (${recentFrames.size}/${FrontBurstCapturePolicy.TARGET_FRAME_COUNT} frames)"
                } else {
                    it.frontOverviewStatus
                },
            )
        }
        refreshDiagnostics()
        if (inferenceInFlight) return

        // Use plant tracking for both modes — automatic detection with manual override
        if (_state.value.selectedMode == RecordingMode.SIDE_SCAN) {
            if (!consumeCooldownOrShowStatus()) return
            // Side scan: use object detection + plant tracking for bounding boxes
            inferenceInFlight = true
            val token = inferenceGeneration
            inferenceJob = viewModelScope.launch {
                try {
                    val inferenceStartedAt = System.nanoTime()
                    val vegetation = com.sakshyam.agribot.domain.logic.VegetationCoverageAnalyzer.evaluate(
                        frame.width, frame.height, frame.rgbPixels,
                    )
                    val rawCandidates = inferenceRepository.detectFrontFrame(
                        frame = frame,
                        maxClassifiedCandidates = MAX_SIDE_SCAN_CLASSIFIED_CANDIDATES,
                        focusX = frame.width / 2f,
                        focusY = frame.height / 2f,
                    )
                    if (!isInferenceCurrent(token, RecordingMode.SIDE_SCAN)) return@launch
                    val sceneDecision = NonPlantSceneGate.apply(
                        candidates = rawCandidates,
                        vegetation = vegetation,
                    )
                    val candidates = sceneDecision.candidates
                    sceneDropCount = sceneDecision.dropped
                    vegetationCoverage = vegetation.coverage
                    if (sceneDecision.dropped) {
                        _state.update { current ->
                            current.copy(
                                trackedPlants = emptyList(),
                                detectionStatus = context.getString(R.string.scene_no_plants_detected),
                            )
                        }
                        return@launch
                    }
                    val latencyMs = (System.nanoTime() - inferenceStartedAt) / 1_000_000.0
                    val thermalStatus = AndroidThermalStatusReader.read(context)
                    val performance = performanceGovernor.recordSelectedFrameLatency(
                        latencyMs = latencyMs,
                        thermalStatus = thermalStatus,
                    )
                    pauseAfterCurrentDecisionGroup = pauseAfterCurrentDecisionGroup || performance.pauseAfterCurrentGroup
                    appendThermalWarningIfNeeded(thermalStatus, performance)
                    _state.update {
                        it.copy(
                            modelStatus = "Model ready: ${state.value.modelBundleId}",
                            latestLatencyMs = latencyMs,
                            targetFps = performance.targetFps,
                            performanceStatus = performance.performanceStatus,
                            error = null,
                        )
                    }
                    // Process detections with plant tracker for bounding boxes
                    val trackedPlants = plantTracker.processFrameDetections(
                        candidates = candidates,
                        frameWidth = frame.width,
                        frameHeight = frame.height,
                        timestamp = System.currentTimeMillis()
                    )
                    if (!isInferenceCurrent(token, RecordingMode.SIDE_SCAN)) return@launch
                    _state.update { current ->
                        current.copy(
                            trackedPlants = plantTracker.trackedPlantsFlow.value,
                            detectionStatus = if (candidates.isEmpty()) {
                                "No plant found in this view · point at one plant"
                            } else {
                                "${candidates.size} plant box${if (candidates.size == 1) "" else "es"} found"
                            },
                        )
                    }
                    // Feed the same tracked plant through the multi-frame gate
                    // used by Side Scan.  The old path recorded every detector
                    // result as OK, including the detector's placeholder
                    // "Uncertain" label, and never used gate stability.
                    if (trackedPlants.isNotEmpty()) {
                        val cx = frame.width / 2f
                        val cy = frame.height / 2f
                        // Once a decision group has started, keep sampling the
                        // same stable track. Selecting a fresh frame-centre
                        // candidate on every inference let detector jitter
                        // switch plants between the two gate frames and
                        // produced "Uncertain" with a high but unrelated score.
                        val pendingPlant = pendingDecisionPlantId
                            ?.let { targetId -> trackedPlants.firstOrNull { it.plantId == targetId } }
                        if (pendingDecisionPlantId != null && pendingPlant == null) {
                            // Never switch a temporal decision window to a
                            // different plant during a one-frame occlusion.
                            // Abandon the incomplete window and wait for a new
                            // stable target instead of creating a duplicate or
                            // mixing labels from two plants.
                            gate.reset()
                            pendingDecisionPlantId = null
                        }
                        val closestPlant = pendingPlant ?: if (pendingDecisionPlantId == null) {
                            trackedPlants.minByOrNull {
                                abs(it.currentBbox.centerX - cx) + abs(it.currentBbox.centerY - cy)
                            }
                        } else {
                            null
                        }
                        val closestCandidate = closestPlant?.let { plant ->
                            candidates
                                .maxByOrNull { candidate ->
                                    PlantTrackingMath.calculateMatchScore(
                                        existingBbox = plant.currentBbox,
                                        newBbox = candidate.bboxPx,
                                        existingFeatures = plant.features,
                                        newFeatures = PlantTrackingMath.extractFeatures(candidate.bboxPx, candidate.confidence),
                                    )
                                }
                                ?.takeIf { candidate ->
                                    PlantTrackingMath.calculateIoU(plant.currentBbox, candidate.bboxPx) > 0.05f ||
                                        PlantTrackingMath.calculateSpatialDistanceScore(plant.currentBbox, candidate.bboxPx) > 0.25f
                                }
                        }
                        if (closestPlant != null && closestCandidate != null) {
                            if (closestCandidate.rawLabel.equals("detector_only", ignoreCase = true)) {
                                // A detector-only box is useful for the overlay
                                // and physical count, but it has no classifier
                                // evidence. Never let repeated missing crop
                                // classifications turn into an automatic
                                // uncertain decision; ask the farmer to centre
                                // the plant and wait for a real crop result.
                                _state.update {
                                    it.copy(detectionStatus = context.getString(R.string.center_plant_for_check))
                                }
                            } else {
                                if (pendingDecisionPlantId != closestPlant.plantId) {
                                    gate.reset()
                                    pendingDecisionPlantId = closestPlant.plantId
                                }
                                val prediction = closestCandidate.toFramePrediction(
                                    modelVersion = _state.value.modelBundleId,
                                    latencyMs = latencyMs,
                                )
                                val decision = gate.add(prediction)
                                if (decision != null) {
                                    recordDecision(
                                        decision = decision,
                                        rawLabel = prediction.rawLabel,
                                        manual = false,
                                        notes = null,
                                        original = null,
                                        auditManual = false,
                                    )
                                    gate.reset()
                                    pendingDecisionPlantId = null
                                    armCooldownAfterDecision()
                                }
                            }
                        }
                    }
                } catch (error: ModelUnavailableException) {
                    _state.update { it.copy(modelStatus = error.message ?: "Model unavailable", error = error.message) }
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: RuntimeException) {
                    _state.update { it.copy(error = error.message ?: "Detection failed") }
                } finally {
                    if (token == inferenceGeneration) {
                        inferenceInFlight = false
                        inferenceJob = null
                        refreshDiagnostics()
                    }
                }
            }
        } else {
            // Front overview mode - use plant tracking
            processFrameWithPlantTracking(frame)
        }
    }

    private fun processFrameWithPlantTracking(frame: AnalysisFrame) {
        if (inferenceInFlight) return
        inferenceInFlight = true
        val token = inferenceGeneration
        inferenceJob = viewModelScope.launch {
            try {
                // Run object detection to get bounding boxes
                val inferenceStartedAt = System.nanoTime()
                val vegetation = com.sakshyam.agribot.domain.logic.VegetationCoverageAnalyzer.evaluate(
                    frame.width, frame.height, frame.rgbPixels,
                )
                val rawCandidates = inferenceRepository.detectFrontFrame(frame)
                if (!isInferenceCurrent(token, RecordingMode.FRONT_ROW_OVERVIEW)) return@launch
                val sceneDecision = NonPlantSceneGate.apply(rawCandidates, vegetation)
                vegetationCoverage = vegetation.coverage
                sceneDropCount = sceneDecision.dropped
                val candidates = sceneDecision.candidates
                val latencyMs = (System.nanoTime() - inferenceStartedAt) / 1_000_000.0
                val thermalStatus = AndroidThermalStatusReader.read(context)
                val performance = performanceGovernor.recordSelectedFrameLatency(
                    latencyMs = latencyMs,
                    thermalStatus = thermalStatus,
                )
                
                if (sceneDecision.dropped) {
                    _state.update { current ->
                        current.copy(
                            trackedPlants = emptyList(),
                            latestLatencyMs = latencyMs,
                            targetFps = performance.targetFps,
                            performanceStatus = performance.performanceStatus,
                            detectionStatus = context.getString(R.string.scene_no_plants_detected),
                        )
                    }
                    refreshDiagnostics()
                    return@launch
                }

                // Process detections with plant tracker
                plantTracker.processFrameDetections(
                    candidates = candidates,
                    frameWidth = frame.width,
                    frameHeight = frame.height,
                    timestamp = System.currentTimeMillis()
                )
                if (!isInferenceCurrent(token, RecordingMode.FRONT_ROW_OVERVIEW)) return@launch
                
                // Update UI state with tracked plants
                _state.update { current ->
                    current.copy(
                        trackedPlants = plantTracker.trackedPlantsFlow.value,
                        latestLatencyMs = latencyMs,
                        targetFps = performance.targetFps,
                        performanceStatus = performance.performanceStatus,
                        detectionStatus = if (candidates.isEmpty()) {
                            "No plant found in this view · align the row inside the guide"
                        } else {
                            "${candidates.size} plant box${if (candidates.size == 1) "" else "es"} found"
                        },
                    )
                }
                
                refreshDiagnostics()
            } catch (error: ModelUnavailableException) {
                _state.update { it.copy(modelStatus = error.message ?: "Model unavailable", error = error.message) }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: RuntimeException) {
                _state.update { it.copy(error = error.message ?: "Detection failed") }
            } finally {
                if (token == inferenceGeneration) {
                    inferenceInFlight = false
                    inferenceJob = null
                }
            }
        }
    }

    fun skipCurrentPlant() {
        val decision = PlantDecision(
            label = "Skipped",
            confidence = 0f,
            status = DecisionStatus.SKIPPED,
            framesUsed = 0,
            reason = "operator_skip",
        )
        recordDecision(decision, rawLabel = "Skipped", manual = true, notes = null, original = null, auditManual = false)
        pauseAfterCurrentDecisionGroup = false
        resetCooldownAfterManualAction()
    }

    fun captureManualSnapshot() {
        viewModelScope.launch {
            val runId = _state.value.activeRunId ?: return@launch
            val timestamp = Instant.now()
            val frame = latestFrame
            val plan = evidencePolicy.shouldCaptureManualSnapshot(
                snapshot = evidenceRepository.retentionSnapshot(runId),
                evidenceEnabled = _state.value.saveEvidenceFrames,
            )
            val capture = when {
                !plan.capture -> EvidenceCaptureResult(path = null, status = plan.status)
                frame == null -> EvidenceCaptureResult(path = null, status = "no_frame")
                else -> runCatching {
                    evidenceRepository.saveEvidence(
                        runId = runId,
                        decisionId = DecisionId("${runId.value}_manual_snapshot_${timestamp.toEpochMilli()}"),
                        frame = frame,
                        kind = "manual_snapshot",
                    )
                }.getOrElse { EvidenceCaptureResult(path = null, status = "save_failed") }
            }
            runRepository.appendEvent(
                RunEventFactory.manualSnapshot(
                    runId = runId,
                    timestamp = timestamp,
                    evidencePath = capture.path,
                    evidenceStatus = capture.status,
                    operatorDisplayName = scanSettings.operatorDisplayName,
                ),
            )
            val snapshot = evidenceRepository.retentionSnapshot(runId)
            _state.update {
                it.copy(
                    storageUsedBytes = snapshot.bytesUsed,
                    manualSnapshotStatus = capture.status.toManualSnapshotLabel(),
                )
            }
            refreshDiagnostics()
        }
    }

    fun retakeCurrentPlant() {
        gate.reset()
        pendingDecisionPlantId = null
        val current = _state.value
        if (current.decisionsRecorded > 0) {
            val replacementIndex = (nextScanIndex - 1).coerceAtLeast(1)
            val runId = current.activeRunId
            nextScanIndex = replacementIndex
            nextPlantReadyAtNanos = 0L
            if (runId != null) {
                val decisionId = DecisionId("${runId.value}_$replacementIndex")
                viewModelScope.launch {
                    evidenceRepository.deleteEvidence(runId, decisionId)
                    runRepository.deleteDecision(decisionId)
                    runRepository.appendEvent(
                        RunEventFactory.retakeRequested(
                            runId = runId,
                            timestamp = Instant.now(),
                            decisionId = decisionId,
                            scanIndex = replacementIndex,
                        ),
                    )
                    refreshActiveRunFieldMap(runId)
                    refreshRunHistory()
                    refreshDiagnostics()
                }
            }
        }
        _state.update {
            it.copy(
                latestDecision = null,
                decisionsRecorded = if (it.decisionsRecorded > 0) it.decisionsRecorded - 1 else 0,
                error = null,
                currentPosition = mapper?.position(nextScanIndex),
                cooldownRemainingMs = 0,
                cooldownStatus = null,
            )
        }
    }

    fun captureFrontBurst() {
        viewModelScope.launch {
            val current = _state.value
            if (current.selectedMode != RecordingMode.FRONT_ROW_OVERVIEW || current.runState != RunState.RECORDING) {
                _state.update {
                    it.copy(
                        frontOverviewStatus = "Resume Front Overview before capture",
                        error = "Resume Front Overview before capturing a burst",
                    )
                }
                refreshDiagnostics()
                return@launch
            }
            val layout = current.activeLayout ?: return@launch
            captureFrontBurst(layout)
        }
    }

    fun confirmFrontReview() {
        viewModelScope.launch {
            val pending = pendingFrontReviewWork ?: return@launch
            val runId = RunId("android_${RUN_ID_FORMAT.format(Instant.now())}")
            lastThermalWarningReason = null
            runRepository.createRun(
                RunConfig(
                    runId = runId,
                    mode = RecordingMode.FRONT_ROW_OVERVIEW,
                    fieldLayoutId = pending.layout.id,
                    targetFps = _state.value.targetFps,
                    modelBundleId = _state.value.modelBundleId,
                ),
            )
            runRepository.appendEvent(
                RunEventFactory.started(
                    runId = runId,
                    timestamp = Instant.now(),
                    mode = RecordingMode.FRONT_ROW_OVERVIEW.name,
                    targetFps = _state.value.targetFps,
                ),
            )
            runRepository.appendEvent(
                RunEventFactory.modelLoaded(
                    runId = runId,
                    timestamp = Instant.now(),
                    bundleId = _state.value.modelBundleId,
                    cpuDefault = true,
                ),
            )
            runRepository.appendEvent(
                frontBurstReviewEvent(
                    runId = runId,
                    calibration = pending.calibration,
                    capturedFrameCount = pending.framesUsed,
                    reason = pending.review.reason,
                    reviewRequired = pending.review.reviewRequired,
                    detectorStatus = pending.detectorStatus,
                    candidateCount = pending.candidateCount,
                ),
            )
            appendFrontOverviewDecisions(
                runId = runId,
                layout = pending.layout,
                review = pending.review,
                framesUsed = pending.framesUsed,
                evidenceFrame = pending.evidenceFrame,
            )
            runRepository.markRunCompleted(runId)
            pendingFrontReviewWork = null
            _state.update {
                it.copy(
                    selectedMode = RecordingMode.FRONT_ROW_OVERVIEW,
                    activeRunId = runId,
                    runState = RunState.COMPLETED,
                    decisionsRecorded = pending.review.decisions.size,
                    pendingFrontReview = null,
                    frontOverviewStatus = "Committed review: ${pending.review.reason}",
                    error = null,
                )
            }
            refreshActiveRunFieldMap(runId)
            refreshRunHistory()
            refreshDiagnostics()
        }
    }

    fun markFrontReviewUncertain() {
        val pending = pendingFrontReviewWork ?: return
        val correctedReview = FrontReviewPresenter.markAllUncertain(pending.review)
        pendingFrontReviewWork = pending.copy(review = correctedReview)
        _state.update {
            it.copy(
                pendingFrontReview = pendingFrontReviewUi(pending.copy(review = correctedReview)),
                frontOverviewStatus = "Review required: ${correctedReview.reason}",
            )
        }
    }

    fun correctFrontReviewDecision(index: Int, label: String, status: DecisionStatus) {
        val pending = pendingFrontReviewWork ?: return
        val correctedReview = FrontReviewPresenter.correctDecision(
            review = pending.review,
            index = index,
            label = label,
            status = status,
        )
        val correctedPending = pending.copy(review = correctedReview)
        pendingFrontReviewWork = correctedPending
        _state.update {
            it.copy(
                pendingFrontReview = pendingFrontReviewUi(correctedPending),
                frontOverviewStatus = "Review required: ${correctedReview.reason}",
            )
        }
    }

    fun discardFrontReview() {
        pendingFrontReviewWork = null
        _state.update {
            it.copy(
                runState = RunState.READY,
                pendingFrontReview = null,
                frontOverviewStatus = "Review discarded",
                error = null,
            )
        }
        refreshDiagnostics()
    }

    fun selectRunDetail(runId: RunId) {
        viewModelScope.launch {
            _state.update {
                it.copy(
                    selectedRunId = runId,
                    runDetailLoading = true,
                    selectedRunRows = emptyList(),
                    selectedRunFieldMap = null,
                    selectedRunSummary = null,
                    lastExportedFile = null,
                    selectedDecisionFilter = DecisionFilter.ALL,
                    pendingDeleteRunId = null,
                )
            }
            try {
                refreshSelectedRunDetail()
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: RuntimeException) {
                _state.update {
                    if (it.selectedRunId != runId) it
                    else it.copy(runDetailLoading = false, error = "Could not read this scan. Please try again.")
                }
            }
        }
    }

    fun setDecisionFilter(filter: DecisionFilter) {
        viewModelScope.launch {
            _state.update { it.copy(selectedDecisionFilter = filter) }
            refreshSelectedRunDetail()
        }
    }

    fun requestDeleteRun(runId: RunId) {
        _state.update { it.copy(pendingDeleteRunId = runId) }
    }

    fun cancelDeleteRun() {
        _state.update { it.copy(pendingDeleteRunId = null) }
    }

    fun confirmDeleteRun(runId: RunId) {
        viewModelScope.launch {
            val current = _state.value
            if (current.protectsActiveRunFromDeletion(runId)) {
                _state.update {
                    it.copy(
                        pendingDeleteRunId = null,
                        error = "Stop the active run before deleting it",
                    )
                }
                refreshDiagnostics()
                return@launch
            }
            try {
                // Delete the ZIP and run directory first. If artifact cleanup
                // fails, keep the Room row so the operator can retry safely.
                exportRepository.deleteRunArtifacts(runId)
                runRepository.deleteRun(runId)
                val clearActiveRun = _state.value.activeRunId == runId
                _state.update {
                    it.copy(
                        activeRunId = if (clearActiveRun) null else it.activeRunId,
                        selectedRunId = if (it.selectedRunId == runId) null else it.selectedRunId,
                        selectedRunRows = if (it.selectedRunId == runId) emptyList() else it.selectedRunRows,
                        selectedRunFieldMap = if (it.selectedRunId == runId) null else it.selectedRunFieldMap,
                        activeRunFieldMap = if (clearActiveRun) null else it.activeRunFieldMap,
                        pendingDeleteRunId = null,
                        decisionsRecorded = if (clearActiveRun) 0 else it.decisionsRecorded,
                        error = null,
                    )
                }
                refreshRunHistory()
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: RuntimeException) {
                _state.update {
                    it.copy(
                        pendingDeleteRunId = null,
                        error = "Delete failed: ${error.message ?: runId.value}",
                    )
                }
            } finally {
                refreshDiagnostics()
            }
        }
    }

    fun exportCsv(runId: RunId? = _state.value.activeRunId) {
        exportRun(runId, exportKind = "csv") { targetRunId -> exportRepository.exportCsv(targetRunId) }
    }

    fun exportJsonl(runId: RunId? = _state.value.activeRunId) {
        exportRun(runId, exportKind = "jsonl") { targetRunId -> exportRepository.exportJsonl(targetRunId) }
    }

    fun exportLogs(runId: RunId? = _state.value.activeRunId) {
        exportRun(runId, exportKind = "logs") { targetRunId -> exportRepository.exportLogs(targetRunId) }
    }

    fun exportBundle(runId: RunId? = _state.value.activeRunId) {
        exportRun(runId, exportKind = "bundle") { targetRunId ->
            exportRepository.exportBundle(targetRunId, _state.value.diagnostics)
        }
    }

    fun exportPdf(runId: RunId? = _state.value.activeRunId) {
        exportRun(runId, exportKind = "pdf") { targetRunId -> exportRepository.exportPdf(targetRunId) }
    }

    private fun exportRun(runId: RunId?, exportKind: String, exporter: suspend (RunId) -> ExportedFile) {
        viewModelScope.launch {
            val targetRunId = runId
            if (targetRunId == null) {
                _state.update { it.copy(error = "No run selected for export") }
                refreshDiagnostics()
                return@launch
            }
            try {
                val exported = exporter(targetRunId)
                runRepository.appendEvent(
                    RunEventFactory.exportCreated(
                        runId = targetRunId,
                        timestamp = Instant.now(),
                        exportKind = exportKind,
                        absolutePath = exported.absolutePath,
                    ),
                )
                _state.update {
                    it.copy(
                        lastExportPath = exported.absolutePath,
                        lastExportedFile = exported,
                        error = null,
                    )
                }
                refreshRunHistory()
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: RuntimeException) {
                _state.update { it.copy(error = "Export failed: ${error.message ?: exportKind}") }
            } finally {
                refreshDiagnostics()
            }
        }
    }

    private fun recordDecision(
        decision: PlantDecision,
        rawLabel: String,
        manual: Boolean,
        notes: String?,
        original: PlantDecision?,
        auditManual: Boolean,
    ) {
        val runId = _state.value.activeRunId ?: return
        val position = mapper?.position(nextScanIndex) ?: return
        val action = if (manual && decision.status == DecisionStatus.SKIPPED) {
            PlantHealthAction.RESCAN
        } else {
            ExportSerializer.actionForDecision(decision.label, decision.status)
        }
        val lateFrames = if (manual) 0 else performanceGovernor.lateFramesInCurrentDecisionGroup()
        val trackedPlant = pendingDecisionPlantId?.let { plantId ->
            plantTracker.trackedPlantsFlow.value.firstOrNull { it.plantId == plantId }
        }
        val motionMeasurement = plantTracker.motionMeasurement
        val recorded = RecordedDecision(
            id = DecisionId("${runId.value}_${position.scanIndex}"),
            runId = runId,
            sequence = position.scanIndex,
            timestamp = Instant.now(),
            epochTime = System.currentTimeMillis() / 1000.0,
            mode = _state.value.selectedMode,
            fieldId = position.fieldId,
            rowId = position.rowId,
            rowSide = null,
            rowIndex = position.rowIndex,
            plantColumn = position.plantColumn,
            plantNumber = position.plantNumber,
            plantKey = position.plantKey,
            xM = position.xM,
            yM = position.yM,
            bboxPx = null,
            geometryConfidence = null,
            plantPositionConfidence = null,
            geometryReason = null,
            label = decision.label,
            confidence = decision.confidence,
            rawLabel = rawLabel,
            status = decision.status,
            action = action,
            framesUsed = decision.framesUsed,
            reason = decision.reason,
            lateFrames = lateFrames,
            gateAvgLatencyMs = gate.snapshot.map { it.latencyMs }.average().takeIf { !it.isNaN() } ?: 0.0,
            gateMaxLatencyMs = gate.snapshot.maxOfOrNull { it.latencyMs } ?: 0.0,
            modelVersion = _state.value.modelBundleId,
            scanIndex = position.scanIndex,
            scanPass = position.scanPass,
            fieldComplete = position.fieldComplete,
            plannedTotalPlants = position.plannedTotalPlants,
            plantDisplay = position.plantDisplay,
            plantId = position.plantNumber,
            trackId = trackedPlant?.plantId,
            threads = _state.value.cpuThreads,
            fpsTarget = _state.value.targetFps.toDouble(),
             manualOverride = manual,
             notes = notes,
             treatmentStatus = trackedPlant?.treatmentStatus ?: TreatmentStatus.NOT_TREATED,
             treatmentNote = trackedPlant?.treatmentNote,
             measurementDistanceM = motionMeasurement.distance.value?.toDouble(),
            measurementSource = motionMeasurement.source.name,
            measurementQuality = motionMeasurement.distance.quality.name,
            relativePlantWidth = trackedPlant?.relativeWidth?.toDouble(),
            relativePlantHeight = trackedPlant?.relativeHeight?.toDouble(),
            sizeSource = trackedPlant?.sizeSource?.name,
            sizeQuality = trackedPlant?.sizeQuality?.name,
            gpsAccuracyM = plantTracker.gpsAccuracyM?.toDouble(),
            gpsFixAgeSeconds = plantTracker.gpsFixAgeSeconds?.toDouble(),
            top2Margin = trackedPlant?.top2Margin?.toDouble(),
            predictionEntropy = trackedPlant?.predictionEntropy?.toDouble(),
        )
        viewModelScope.launch {
            val withEvidence = attachEvidence(recorded, latestFrame, manual)
            runRepository.appendDecision(withEvidence)
            if (auditManual) {
                runRepository.appendEvent(
                    RunEventFactory.manualOverride(
                        runId = runId,
                        timestamp = Instant.now(),
                        decisionId = withEvidence.id,
                        original = original,
                        corrected = decision,
                        reason = notes,
                        operatorDisplayName = scanSettings.operatorDisplayName,
                    ),
                )
            }
            val snapshot = evidenceRepository.retentionSnapshot(runId)
            _state.update { it.copy(storageUsedBytes = snapshot.bytesUsed) }
            refreshActiveRunFieldMap(runId)
            refreshRunHistory()
            refreshDiagnostics()
        }
        nextScanIndex += 1
        performanceGovernor.resetDecisionGroup()
        armCooldownAfterDecision()
        _state.update {
            it.copy(
                latestDecision = decision,
                decisionsRecorded = it.decisionsRecorded + 1,
                currentPosition = mapper?.position(nextScanIndex),
                cooldownRemainingMs = it.cooldownRemainingMs,
                cooldownStatus = it.cooldownStatus,
            )
        }
        if (pauseAfterCurrentDecisionGroup) {
            pauseAfterCurrentDecisionGroup = false
            pause()
        }
    }

    private suspend fun refreshRunHistory() {
        val runs = runRepository.recentRuns(limit = 8)
        val items = runs.map { run ->
            val decisions = runRepository.decisionsForRun(run.runId)
            val summary = RunSummaryReducer.reduce(run.runId, decisions)
            val layout = fieldLayoutRepository.layoutById(run.fieldLayoutId)
            RunHistoryPresenter.toItem(run, summary, layout?.name ?: run.fieldLayoutId.value)
        }
        val current = _state.value
        val mapRunId = when {
            current.activeRunId != null -> current.activeRunId
            current.selectedMode == RecordingMode.FRONT_ROW_OVERVIEW -> null
            else -> runs.firstOrNull()?.runId
        }
        val mapRun = mapRunId?.let { runRepository.runById(it) }
        val mapLayout = mapRun?.let { fieldLayoutRepository.layoutById(it.fieldLayoutId) }
        val mapDecisions = mapRunId?.let { runRepository.decisionsForRun(it) }.orEmpty()
        val map = mapRunId?.let { RunFieldMapPresenter.map(mapLayout, mapDecisions) }
        _state.update {
            val canAdoptLatestCompletedRun = it.activeRunId == null &&
                it.selectedMode != RecordingMode.FRONT_ROW_OVERVIEW &&
                it.runState == RunState.READY
            it.copy(
                runHistory = items,
                activeRunFieldMap = map,
                activeRunId = if (canAdoptLatestCompletedRun) mapRunId else it.activeRunId,
            )
        }
        refreshSelectedRunDetail()
    }

    private suspend fun recoverInterruptedRun() {
        val latestRun = runRepository.recentRuns(limit = 1).firstOrNull()
        val decisions = latestRun?.let { runRepository.decisionsForRun(it.runId) }.orEmpty()
        val recovery = RunRecoveryPlanner.plan(latestRun, decisions)
        if (!recovery.shouldRecover || latestRun == null) return

        val layout = fieldLayoutRepository.layoutById(latestRun.fieldLayoutId)
        if (layout != null) {
            fieldLayoutRepository.setActiveLayout(layout.id)
            mapper = SideScanMapper(layout)
        }
        if (recovery.shouldMarkPaused) {
            runRepository.markRunPaused(latestRun.runId)
            runRepository.appendEvent(
                RunEvent(
                    id = "${latestRun.runId.value}_recovered_interrupted_run",
                    runId = latestRun.runId,
                    timestamp = Instant.now(),
                    type = "recovered_interrupted_run",
                    payloadJson = """{"previous_state":"recording","recovered_state":"paused","next_scan_index":${recovery.nextScanIndex}}""",
                ),
            )
        }
        nextScanIndex = recovery.nextScanIndex
        _state.update {
            it.copy(
                selectedMode = latestRun.mode,
                activeRunId = latestRun.runId,
                runState = recovery.recoveredState ?: RunState.PAUSED,
                activeLayout = layout ?: it.activeLayout,
                activeRunFieldMap = RunFieldMapPresenter.map(layout ?: it.activeLayout, decisions),
                decisionsRecorded = recovery.decisionCount,
                currentPosition = mapper?.position(nextScanIndex),
                error = recovery.message,
            )
        }
    }

    private suspend fun refreshSelectedRunDetail() {
        val selectedRunId = _state.value.selectedRunId ?: return
        val run = runRepository.runById(selectedRunId)
        val layout = run?.let { fieldLayoutRepository.layoutById(it.fieldLayoutId) }
        val decisions = runRepository.decisionsForRun(selectedRunId)
        val rows = RunDetailPresenter.rows(decisions, _state.value.selectedDecisionFilter)
        val fieldMap = RunFieldMapPresenter.map(layout, decisions)
        val summary = RunSummaryReducer.reduce(selectedRunId, decisions)
        _state.update {
            if (it.selectedRunId != selectedRunId) it
            else it.copy(selectedRunRows = rows, selectedRunFieldMap = fieldMap, selectedRunSummary = summary, runDetailLoading = false)
        }
    }

    private suspend fun refreshActiveRunFieldMap(runId: RunId? = null) {
        val targetRunId = runId ?: _state.value.activeRunId ?: return
        val run = runRepository.runById(targetRunId) ?: return
        val layout = fieldLayoutRepository.layoutById(run.fieldLayoutId)
        val decisions = runRepository.decisionsForRun(targetRunId)
        val fieldMap = RunFieldMapPresenter.map(layout, decisions)
        _state.update { current ->
            if (current.activeRunId == targetRunId) {
                current.copy(activeRunFieldMap = fieldMap)
            } else {
                current
            }
        }
    }

    private suspend fun captureFrontBurst(layout: FieldLayout) {
        val calibration = frontCalibration()
        val burstFrames = recentFrames.toList()
        val readiness = FrontBurstCapturePolicy.evaluate(
            frameCount = burstFrames.size,
            distinctTimestampCount = burstFrames.map { it.timestampNanos }.distinct().size,
        )
        if (!readiness.canCapture) {
            _state.update {
                it.copy(
                    selectedMode = RecordingMode.FRONT_ROW_OVERVIEW,
                    frontOverviewStatus = readiness.status,
                    error = readiness.error,
                )
            }
            refreshDiagnostics()
            return
        }
        val detectorResult = detectFrontCandidates(burstFrames)
        val review = if (calibration == null) {
            FrontBurstProcessor.processWithoutCalibration(
                capturedFrameCount = burstFrames.size,
                candidates = detectorResult.candidates,
            )
        } else {
            FrontBurstProcessor.process(
                capturedFrameCount = burstFrames.size,
                calibration = calibration,
                candidates = detectorResult.candidates,
                classifierConfidenceThreshold = _state.value.confidenceThreshold,
            )
        }
        val pendingReview = PendingFrontReviewWork(
            layout = layout,
            calibration = calibration,
            review = review,
            framesUsed = burstFrames.size,
            evidenceFrame = burstFrames.lastOrNull(),
            detectorStatus = detectorResult.status,
            candidateCount = detectorResult.candidates.size,
        )
        pendingFrontReviewWork = pendingReview
        val status = if (detectorResult.candidates.isEmpty()) {
            "Review required: ${review.reason}"
        } else {
            "Review required: ${review.reason} (${detectorResult.candidates.size} detector candidates)"
        }
        _state.update {
            it.copy(
                selectedMode = RecordingMode.FRONT_ROW_OVERVIEW,
                activeRunId = null,
                runState = RunState.READY,
                decisionsRecorded = 0,
                pendingFrontReview = pendingFrontReviewUi(pendingReview),
                frontOverviewStatus = if (calibration == null) "$status · camera calibration required" else status,
                error = if (calibration == null) {
                    "Camera geometry is not calibrated; results remain Unknown until you review or calibrate this view"
                } else null,
            )
        }
        refreshDiagnostics()
    }

    private fun pendingFrontReviewUi(pending: PendingFrontReviewWork): FrontReviewUiState =
        FrontReviewPresenter.toUiState(
            review = pending.review,
            detectorStatus = pending.detectorStatus,
            candidateCount = pending.candidateCount,
        )

    private fun frontCalibration(): FrontCaptureCalibration? {
        // Row layout is not a camera calibration. The previous implementation
        // filled camera height, rails, tilt, and frame dimensions with defaults
        // and then emitted plant coordinates as if they were measured. Until a
        // real camera calibration is persisted, callers receive explicit
        // Unknown geometry from FrontBurstProcessor.processWithoutCalibration.
        return null
    }

    private suspend fun appendThermalWarningIfNeeded(
        thermalStatus: ThermalStatus,
        performance: SideScanPerformanceDecision,
    ) {
        val runId = _state.value.activeRunId ?: return
        if (performance.reason == "within_policy") return
        if (!performance.reason.startsWith("thermal_")) return
        if (lastThermalWarningReason == performance.reason) return
        lastThermalWarningReason = performance.reason
        runRepository.appendEvent(
            RunEventFactory.thermalWarning(
                runId = runId,
                timestamp = Instant.now(),
                thermalStatus = thermalStatus.name.lowercase(),
                action = if (performance.pauseAfterCurrentGroup) "pause_after_group" else "reduced_fps",
                targetFps = performance.targetFps,
            ),
        )
    }

    private fun frontBurstReviewEvent(
        runId: RunId,
        calibration: FrontCaptureCalibration?,
        capturedFrameCount: Int,
        reason: String,
        reviewRequired: Boolean,
        detectorStatus: String,
        candidateCount: Int,
    ): RunEvent =
        RunEvent(
            id = "${runId.value}_front_burst_review",
            runId = runId,
            timestamp = Instant.now(),
            type = "front_burst_review",
            payloadJson = buildString {
                append("{\"captured_frame_count\":$capturedFrameCount,\"review_required\":$reviewRequired")
                append(",\"reason\":\"${reason.escapeJson()}\"")
                if (calibration == null) {
                    append(",\"calibration_status\":\"missing\"")
                } else {
                    append(",\"calibration_status\":\"validated\"")
                    append(",\"left_row_id\":\"${calibration.leftRowId.escapeJson()}\"")
                    append(",\"right_row_id\":\"${calibration.rightRowId.escapeJson()}\"")
                    append(",\"nearest_left_plant_number\":${calibration.nearestLeftPlantNumber}")
                    append(",\"nearest_right_plant_number\":${calibration.nearestRightPlantNumber}")
                    append(",\"camera_distance_m\":${calibration.cameraDistanceToNearestRowM}")
                    append(",\"camera_height_m\":${calibration.cameraHeightM}")
                    append(",\"tilt_degrees\":${calibration.cameraTiltDegrees}")
                    append(",\"guide_rail_left_px\":${calibration.guideRailLeftPx}")
                    append(",\"guide_rail_right_px\":${calibration.guideRailRightPx}")
                }
                append(",\"detector_status\":\"${detectorStatus.escapeJson()}\"")
                append(",\"detector_candidate_count\":$candidateCount}")
            },
        )

    private suspend fun detectFrontCandidates(frames: List<AnalysisFrame>): FrontDetectorResult {
        if (frames.isEmpty()) {
            return FrontDetectorResult(status = "no_camera_frame", candidates = emptyList())
        }
        return try {
            val boundedFrames = frames
                .distinctBy { it.timestampNanos }
                .take(FrontBurstCapturePolicy.TARGET_FRAME_COUNT)
            val perFrame = boundedFrames.mapIndexed { index, frame ->
                FrontBurstConsensus.FrameCandidates(
                    frameIndex = index,
                    candidates = inferenceRepository.detectFrontFrame(
                        frame = frame,
                        maxClassifiedCandidates = FrontBurstCapturePolicy.MAX_CLASSIFIED_CANDIDATES_PER_FRAME,
                    ),
                )
            }
            val candidates = FrontBurstConsensus.requireDistinctFrameConsensus(perFrame)
            val rawCount = perFrame.sumOf { it.candidates.size }
            FrontDetectorResult(
                status = when {
                    rawCount == 0 -> "no_detector_candidates"
                    candidates.any { it.rawLabel == "insufficient_frame_consensus" } -> "insufficient_frame_consensus"
                    else -> "detector_candidates"
                },
                candidates = candidates,
            )
        } catch (error: ModelUnavailableException) {
            FrontDetectorResult(status = error.message ?: "model_unavailable", candidates = emptyList())
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: RuntimeException) {
            FrontDetectorResult(status = error.message ?: "detector_failed", candidates = emptyList())
        }
    }

    private suspend fun appendFrontOverviewDecisions(
        runId: RunId,
        layout: FieldLayout,
        review: FrontBurstReview,
        framesUsed: Int,
        evidenceFrame: AnalysisFrame?,
    ) {
        review.decisions.forEachIndexed { index, draft ->
            val recorded = frontRecordedDecision(
                runId = runId,
                layout = layout,
                sequence = index + 1,
                draft = draft,
                framesUsed = framesUsed,
            )
            runRepository.appendDecision(attachEvidence(recorded, evidenceFrame, manual = recorded.manualOverride))
            if (recorded.manualOverride) {
                runRepository.appendEvent(frontReviewCorrectionEvent(runId, recorded))
            }
        }
    }

    private fun frontRecordedDecision(
        runId: RunId,
        layout: FieldLayout,
        sequence: Int,
        draft: FrontOverviewDecisionDraft,
        framesUsed: Int,
    ): RecordedDecision {
        val row = draft.rowId?.let { rowId -> layout.rows.firstOrNull { it.id.value == rowId } }
        val trackedPlant = plantTracker.trackedPlantsFlow.value
            .filter { tracked ->
                val centerDistance = abs(tracked.currentBbox.centerX - draft.bboxPx.centerX) +
                    abs(tracked.currentBbox.centerY - draft.bboxPx.centerY)
                centerDistance <= maxOf(draft.bboxPx.width, draft.bboxPx.height) * 2f
            }
            .minByOrNull { tracked ->
                abs(tracked.currentBbox.centerX - draft.bboxPx.centerX) +
                    abs(tracked.currentBbox.centerY - draft.bboxPx.centerY)
            }
        val motionMeasurement = plantTracker.motionMeasurement
        val plantKey = if (draft.rowId != null && draft.plantNumber != null) {
            "${draft.rowId}:${draft.plantNumber}"
        } else {
            null
        }
        return RecordedDecision(
            id = DecisionId("${runId.value}_front_$sequence"),
            runId = runId,
            sequence = sequence,
            timestamp = Instant.now(),
            epochTime = System.currentTimeMillis() / 1000.0,
            mode = RecordingMode.FRONT_ROW_OVERVIEW,
            fieldId = layout.id.value,
            rowId = draft.rowId,
            rowSide = draft.rowSide,
            rowIndex = row?.rowIndex,
            plantColumn = draft.plantColumnEstimate,
            plantNumber = draft.plantNumber,
            plantKey = plantKey,
            xM = null,
            yM = row?.yM,
            bboxPx = draft.bboxPx,
            geometryConfidence = draft.rowAssignmentConfidence,
            plantPositionConfidence = draft.plantPositionConfidence,
            geometryReason = draft.geometryReason,
            label = draft.label,
            confidence = draft.confidence,
            rawLabel = draft.rawLabel,
            status = draft.status,
            action = draft.action,
            framesUsed = framesUsed,
            reason = draft.reason,
            lateFrames = 0,
            gateAvgLatencyMs = 0.0,
            gateMaxLatencyMs = 0.0,
            modelVersion = _state.value.modelBundleId,
            scanIndex = sequence,
            scanPass = null,
            fieldComplete = null,
            plannedTotalPlants = null,
            plantDisplay = plantKey,
            plantId = draft.plantNumber,
            trackId = trackedPlant?.plantId,
            threads = _state.value.cpuThreads,
            fpsTarget = _state.value.targetFps.toDouble(),
             manualOverride = draft.status == DecisionStatus.MANUAL || draft.reason.startsWith("manual_front_review"),
             notes = draft.reason.takeIf { it.startsWith("manual_front_review") },
             treatmentStatus = trackedPlant?.treatmentStatus ?: TreatmentStatus.NOT_TREATED,
             treatmentNote = trackedPlant?.treatmentNote,
             measurementDistanceM = motionMeasurement.distance.value?.toDouble(),
            measurementSource = motionMeasurement.source.name,
            measurementQuality = motionMeasurement.distance.quality.name,
            relativePlantWidth = trackedPlant?.relativeWidth?.toDouble(),
            relativePlantHeight = trackedPlant?.relativeHeight?.toDouble(),
            sizeSource = trackedPlant?.sizeSource?.name,
            sizeQuality = trackedPlant?.sizeQuality?.name,
            gpsAccuracyM = plantTracker.gpsAccuracyM?.toDouble(),
            gpsFixAgeSeconds = plantTracker.gpsFixAgeSeconds?.toDouble(),
            top2Margin = trackedPlant?.top2Margin?.toDouble(),
            predictionEntropy = trackedPlant?.predictionEntropy?.toDouble(),
        )
    }

    private fun frontReviewCorrectionEvent(runId: RunId, decision: RecordedDecision): RunEvent =
        RunEventFactory.manualOverride(
            runId = runId,
            timestamp = Instant.now(),
            decisionId = decision.id,
            original = null,
            corrected = PlantDecision(
                label = decision.label,
                confidence = decision.confidence,
                status = decision.status,
                framesUsed = decision.framesUsed,
                reason = decision.reason,
            ),
            reason = decision.notes,
            operatorDisplayName = scanSettings.operatorDisplayName,
        )

    private fun refreshDiagnostics() {
        _state.update { current ->
            val thermalStatus = AndroidThermalStatusReader.read(context)
            val battery = AndroidBatteryStatusReader.read(context)
            val thermalBatteryWarnings = DeviceHealthStatusPresenter.describe(
                thermalStatus = thermalStatus,
                battery = battery,
                includeNormalBattery = true,
            )
            current.copy(
                thermalBatteryWarnings = thermalBatteryWarnings,
                batteryPercent = battery.percent,
                thermalStatus = thermalStatus,
                diagnostics = DiagnosticsPresenter.snapshot(
                    appVersion = "1.0",
                    modelBundleId = current.modelBundleId,
                    cameraPermissionGranted = current.cameraPermissionGranted,
                    analysisWidth = current.analysisWidth,
                    analysisHeight = current.analysisHeight,
                    targetFps = current.targetFps,
                    cpuThreads = current.cpuThreads,
                    modelStatus = current.modelStatus,
                    latestLatencyMs = current.latestLatencyMs,
                    runState = current.runState,
                    storageUsedBytes = current.storageUsedBytes,
                    lastExportPath = current.lastExportPath,
                    thermalBatteryWarnings = thermalBatteryWarnings,
                    measurementSource = current.sensorData.measurementSource,
                    measurementQuality = current.sensorData.distanceQuality.name,
                    gpsStatus = current.sensorData.gpsStatus,
                    gpsProvider = current.sensorData.gpsProvider,
                    gpsPathStatus = current.sensorData.gpsPathStatus,
                    gpsPathPointCount = current.sensorData.gpsPathPoints.size,
                    motionEventsObserved = current.sensorData.motionEventsObserved,
                    captureQualityStatus = current.captureQualityStatus,
                    detectionStatus = current.detectionStatus,
                    latestDecisionReason = current.latestDecision?.reason,
                    stepLengthM = current.sensorData.stepLengthM,
                    stepCalibrationStatus = current.sensorData.stepCalibrationStatus.name,
                ),
            )
        }
    }

    private suspend fun attachEvidence(
        decision: RecordedDecision,
        frame: AnalysisFrame?,
        manual: Boolean,
    ): RecordedDecision {
        val snapshot = evidenceRepository.retentionSnapshot(decision.runId)
        val plan = evidencePolicy.shouldCapture(
            decision = decision,
            snapshot = snapshot,
            evidenceEnabled = _state.value.saveEvidenceFrames,
        )
        if (!plan.capture) return decision.copy(evidenceStatus = plan.status)
        if (frame == null) return decision.copy(evidenceStatus = "no_frame")
        val kind = when {
            manual -> "manual"
            decision.status == DecisionStatus.UNCERTAIN -> "uncertain"
            else -> "sick"
        }
        val capture = runCatching {
            evidenceRepository.saveEvidence(decision.runId, decision.id, frame, kind)
        }.getOrElse { EvidenceCaptureResult(path = null, status = "save_failed") }
        return decision.copy(evidencePath = capture.path, evidenceStatus = capture.status)
    }

    private suspend fun ensureDefaultLayout() {
        if (fieldLayoutRepository.observeActiveLayout().first() != null) return
        val imported = runCatching {
            context.assets.open("farmer_config.json").bufferedReader(Charsets.UTF_8).use { reader ->
                FarmerConfigImporter.importFromJson(reader.readText())
            }
        }.getOrNull()

        if (imported != null) {
            imported.layouts.forEach { layout ->
                fieldLayoutRepository.saveLayout(layout)
            }
            fieldLayoutRepository.setActiveLayout(imported.activeLayout.id)
            val importedFps = SideScanDefaults.nativeDefaultTargetFps(imported.recordingProfile.fps)
            val importedThreads = imported.recordingProfile.threads.coerceIn(1, 8)
            saveScanSettings(scanSettings.copy(targetFps = importedFps, cpuThreads = importedThreads))
            _state.update {
                it.copy(
                    targetFps = importedFps,
                    cpuThreads = importedThreads,
                    performanceStatus = SideScanDefaults.performanceStatus(importedFps),
                )
            }
            return
        }

        val layout = FieldLayout(
            id = FieldId("field_2"),
            name = "Field 2",
            activeRowId = RowId("B"),
            rows = listOf(
                FieldRow(RowId("A"), 1, 20, 0.45, 1.4, 0.0),
                FieldRow(RowId("B"), 2, 15, 0.35, 1.4, 1.4),
                FieldRow(RowId("C"), 3, 22, 0.5, 1.4, 2.8),
            ),
            startPlant = 1,
            plantStep = 1,
            plantCooldownSec = 2.0,
            updatedAt = Instant.now(),
        )
        fieldLayoutRepository.saveLayout(layout)
        fieldLayoutRepository.setActiveLayout(layout.id)
    }

    private fun saveScanSettings(settings: ScanSettings) {
        val safe = ScanSettingsValidator.sanitize(settings)
        scanSettings = safe
        inferenceRepository.setConfidenceThreshold(safe.confidenceThreshold)
        inferenceRepository.setCpuThreads(safe.cpuThreads)
        _state.update { current ->
            current.copy(
                selectedMode = if (current.isRecording) current.selectedMode else safe.defaultMode,
                targetFps = if (current.isRecording) current.targetFps else safe.targetFps,
                cpuThreads = safe.cpuThreads,
                saveEvidenceFrames = safe.saveEvidenceFrames,
                gpsReferenceEnabled = safe.gpsReferenceEnabled,
                confidenceThreshold = safe.confidenceThreshold,
                highConfidenceThreshold = safe.highConfidenceThreshold,
                modelBundleId = safe.modelBundleId,
                performanceStatus = if (current.isRecording) {
                    current.performanceStatus
                } else {
                    SideScanDefaults.performanceStatus(safe.targetFps)
                },
            )
        }
        viewModelScope.launch {
            scanSettingsRepository.saveSettings(safe)
            refreshDiagnostics()
        }
    }

    override fun onCleared() {
        invalidateInferenceWork()
        super.onCleared()
        plantTracker.stopTracking()
    }

    private companion object {
        // Side Scan is an intentional one-plant interaction.  Classifying
        // several boxes in a frame (especially a laptop/phone screen showing
        // multiple plant photos) lets unrelated crops contaminate one track.
        const val MAX_SIDE_SCAN_CLASSIFIED_CANDIDATES = 1
        val RUN_ID_FORMAT: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")
            .withZone(ZoneId.systemDefault())
    }
}

private data class FrontDetectorResult(
    val status: String,
    val candidates: List<FrontOverviewCandidate>,
)

private data class PendingFrontReviewWork(
    val layout: FieldLayout,
    val calibration: FrontCaptureCalibration?,
    val review: FrontBurstReview,
    val framesUsed: Int,
    val evidenceFrame: AnalysisFrame?,
    val detectorStatus: String,
    val candidateCount: Int,
)

private fun FrontOverviewCandidate.toFramePrediction(
    modelVersion: String,
    latencyMs: Double = 0.0,
): FramePrediction {
    val safeLabel = label.ifBlank { "Uncertain" }
    val isUncertain = safeLabel.equals("Uncertain", ignoreCase = true) ||
        safeLabel.equals("Unknown", ignoreCase = true)
    val isHealthy = safeLabel.equals("Healthy", ignoreCase = true)
    return FramePrediction(
        label = if (isUncertain) "Uncertain" else if (isHealthy) "Healthy" else safeLabel,
        confidence = confidence.coerceIn(0f, 1f),
        rawLabel = rawLabel.ifBlank { safeLabel },
        isDisease = !isHealthy && !isUncertain,
        isUncertain = isUncertain,
        latencyMs = latencyMs.coerceAtLeast(0.0),
        modelVersion = modelVersion,
        top2Margin = top2Margin,
        entropy = entropy,
    )
}

private fun String.escapeJson(): String =
    buildString {
        this@escapeJson.forEach { char ->
            when (char) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> append(char)
            }
        }
    }

private fun String.toManualSnapshotLabel(): String = when (this) {
    "saved" -> "Manual snapshot saved"
    "disabled" -> "Evidence disabled"
    "storage_cap_reached" -> "Evidence storage cap reached"
    "no_frame" -> "No camera frame available"
    else -> this
}
