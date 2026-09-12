package com.sakshyam.agribot.domain.repository

import com.sakshyam.agribot.domain.model.ExportedFile
import com.sakshyam.agribot.domain.model.AnalysisFrame
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DiagnosticsSnapshot
import com.sakshyam.agribot.domain.model.EvidenceCaptureResult
import com.sakshyam.agribot.domain.model.EvidenceRetentionSnapshot
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FramePrediction
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import com.sakshyam.agribot.domain.model.ModelBundleReadiness
import com.sakshyam.agribot.domain.model.ModelManifest
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.Run
import com.sakshyam.agribot.domain.model.RunConfig
import com.sakshyam.agribot.domain.model.RunEvent
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunSummary
import com.sakshyam.agribot.domain.model.ScanSettings
import com.sakshyam.agribot.domain.model.TreatmentStatus
import com.sakshyam.agribot.domain.model.ValidationResult
import kotlinx.coroutines.flow.Flow

interface RunRepository {
    suspend fun createRun(config: RunConfig): RunId
    suspend fun markRunRecording(runId: RunId)
    suspend fun markRunPaused(runId: RunId)
    suspend fun markRunCompleted(runId: RunId)
    suspend fun appendDecision(decision: RecordedDecision)
    suspend fun replaceDecision(decision: RecordedDecision)
    /** Persist operator treatment state for the latest decision belonging to a tracked plant. */
    suspend fun updateDecisionTreatment(runId: RunId, trackId: String, status: TreatmentStatus, note: String?)
    suspend fun deleteDecision(decisionId: DecisionId)
    suspend fun appendEvent(event: RunEvent)
    fun observeRun(runId: RunId): Flow<Run?>
    fun observeLatestRun(): Flow<Run?>
    fun observeRunSummary(runId: RunId): Flow<RunSummary>
    suspend fun runById(runId: RunId): Run?
    suspend fun recentRuns(limit: Int = 20): List<Run>
    suspend fun decisionsForRun(runId: RunId): List<RecordedDecision>
    suspend fun eventsForRun(runId: RunId): List<RunEvent>
    suspend fun deleteRun(runId: RunId)
}

interface FieldLayoutRepository {
    fun observeLayouts(): Flow<List<FieldLayout>>
    fun observeActiveLayout(): Flow<FieldLayout?>
    suspend fun layoutById(fieldId: FieldId): FieldLayout?
    suspend fun saveLayout(layout: FieldLayout)
    suspend fun setActiveLayout(fieldId: FieldId)
}

interface InferenceRepository {
    suspend fun classifySideFrame(frame: AnalysisFrame): FramePrediction
    suspend fun detectFrontFrame(
        frame: AnalysisFrame,
        maxClassifiedCandidates: Int = Int.MAX_VALUE,
        focusX: Float? = null,
        focusY: Float? = null,
    ): List<FrontOverviewCandidate>
    fun setConfidenceThreshold(threshold: Float) {}
    /** Applied between inference calls so a running interpreter is never closed. */
    fun setCpuThreads(threads: Int) {}
    fun observeModelManifest(): Flow<ModelManifest?>
    fun observeModelReadiness(): Flow<ModelBundleReadiness>
}

interface EvidenceRepository {
    suspend fun saveEvidence(runId: RunId, decisionId: DecisionId, frame: AnalysisFrame, kind: String): EvidenceCaptureResult
    suspend fun retentionSnapshot(runId: RunId): EvidenceRetentionSnapshot
    /** Remove all evidence files associated with a decision being retaken/deleted. */
    suspend fun deleteEvidence(runId: RunId, decisionId: DecisionId) {}
}

interface ExportRepository {
    suspend fun exportCsv(runId: RunId): ExportedFile
    suspend fun exportJsonl(runId: RunId): ExportedFile
    suspend fun exportLogs(runId: RunId): ExportedFile
    suspend fun exportBundle(runId: RunId, diagnostics: DiagnosticsSnapshot? = null): ExportedFile
    suspend fun exportPdf(runId: RunId): ExportedFile

    /** Remove all local export artifacts for a run before its database row is deleted. */
    suspend fun deleteRunArtifacts(runId: RunId) {}
}

interface ModelManifestRepository {
    suspend fun loadManifest(): ModelManifest?
    suspend fun validateManifest(manifest: ModelManifest): ValidationResult
}

interface ScanSettingsRepository {
    fun observeSettings(): Flow<ScanSettings>
    suspend fun saveSettings(settings: ScanSettings)
}
