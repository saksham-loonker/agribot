package com.sakshyam.agribot.data.repository

import androidx.room.withTransaction
import com.sakshyam.agribot.data.db.AgribotDatabase
import com.sakshyam.agribot.data.db.ScanRunEntity
import com.sakshyam.agribot.domain.logic.RunSummaryReducer
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.Run
import com.sakshyam.agribot.domain.model.RunConfig
import com.sakshyam.agribot.domain.model.RunEvent
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunState
import com.sakshyam.agribot.domain.model.RunSummary
import com.sakshyam.agribot.domain.model.TreatmentStatus
import com.sakshyam.agribot.domain.repository.RunRepository
import java.time.Instant
import javax.inject.Inject
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

class RoomRunRepository @Inject constructor(
    private val database: AgribotDatabase,
) : RunRepository {
    override suspend fun createRun(config: RunConfig): RunId {
        val now = Instant.now().toEpochMilli()
        database.runDao().insertRun(
            ScanRunEntity(
                runId = config.runId.value,
                mode = config.mode.name,
                fieldLayoutId = config.fieldLayoutId.value,
                startedAt = now,
                completedAt = null,
                state = RunState.RECORDING.name,
                note = null,
                modelBundleId = config.modelBundleId,
                cameraId = null,
                analysisWidth = null,
                analysisHeight = null,
                targetFps = config.targetFps,
                appVersion = "1.0",
            ),
        )
        return config.runId
    }

    override suspend fun markRunRecording(runId: RunId) {
        database.runDao().updateRunState(runId.value, RunState.RECORDING.name, null)
    }

    override suspend fun markRunPaused(runId: RunId) {
        database.runDao().updateRunState(runId.value, RunState.PAUSED.name, null)
    }

    override suspend fun markRunCompleted(runId: RunId) {
        database.runDao().updateRunState(runId.value, RunState.COMPLETED.name, Instant.now().toEpochMilli())
    }

    override suspend fun appendDecision(decision: RecordedDecision) {
        replaceDecision(decision)
    }

    override suspend fun replaceDecision(decision: RecordedDecision) {
        database.withTransaction {
            database.decisionDao().upsertDecision(decision.toEntity())
            val geometry = decision.toGeometryEntity()
            if (geometry == null) {
                database.decisionDao().deleteGeometry(decision.id.value)
            } else {
                database.decisionDao().upsertGeometry(geometry)
            }
        }
    }

    override suspend fun updateDecisionTreatment(
        runId: RunId,
        trackId: String,
        status: TreatmentStatus,
        note: String?,
    ) {
        database.withTransaction {
            database.decisionDao().latestDecisionIdForTrack(runId.value, trackId)?.let { decisionId ->
                database.decisionDao().updateTreatment(decisionId, status.name, note)
            }
        }
    }

    override suspend fun deleteDecision(decisionId: DecisionId) {
        database.withTransaction {
            database.decisionDao().deleteGeometry(decisionId.value)
            database.decisionDao().deleteDecision(decisionId.value)
        }
    }

    override suspend fun appendEvent(event: RunEvent) {
        database.runEventDao().upsertEvent(event.toEntity())
    }

    override fun observeRun(runId: RunId): Flow<Run?> =
        database.runDao().observeRun(runId.value).map { it?.toDomain() }

    override fun observeLatestRun(): Flow<Run?> =
        database.runDao().observeLatestRun().map { it?.toDomain() }

    override fun observeRunSummary(runId: RunId): Flow<RunSummary> =
        database.decisionDao().observeDecisions(runId.value).map { rows ->
            val decisions = rows.map { row -> row.toDomain(database.decisionDao().getGeometry(row.id)) }
            RunSummaryReducer.reduce(runId, decisions)
        }

    override suspend fun runById(runId: RunId): Run? =
        database.runDao().getRun(runId.value)?.toDomain()

    override suspend fun recentRuns(limit: Int): List<Run> =
        database.runDao().getRecentRuns(limit).map { it.toDomain() }

    override suspend fun decisionsForRun(runId: RunId): List<RecordedDecision> =
        database.decisionDao().getDecisions(runId.value).map { row ->
            row.toDomain(database.decisionDao().getGeometry(row.id))
        }

    override suspend fun eventsForRun(runId: RunId): List<RunEvent> =
        database.runEventDao().getEvents(runId.value).map { it.toDomain() }

    override suspend fun deleteRun(runId: RunId) {
        database.withTransaction {
            database.runEventDao().deleteEvents(runId.value)
            database.runDao().deleteRun(runId.value)
        }
    }
}
