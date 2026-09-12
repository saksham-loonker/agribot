package com.sakshyam.agribot.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow

@Dao
interface FieldLayoutDao {
    @Query("SELECT * FROM field_layouts ORDER BY updatedAt DESC")
    fun observeLayouts(): Flow<List<FieldLayoutEntity>>

    @Query("SELECT * FROM field_layouts WHERE isActive = 1 LIMIT 1")
    fun observeActiveLayout(): Flow<FieldLayoutEntity?>

    @Query("SELECT * FROM field_layouts WHERE id = :id LIMIT 1")
    suspend fun getLayout(id: String): FieldLayoutEntity?

    @Query("SELECT * FROM field_rows WHERE fieldLayoutId = :fieldLayoutId ORDER BY rowIndex ASC")
    suspend fun getRows(fieldLayoutId: String): List<FieldRowEntity>

    @Query("SELECT * FROM field_rows WHERE fieldLayoutId = :fieldLayoutId ORDER BY rowIndex ASC")
    fun observeRows(fieldLayoutId: String): Flow<List<FieldRowEntity>>

    @Upsert
    suspend fun upsertLayout(layout: FieldLayoutEntity)

    @Upsert
    suspend fun upsertRows(rows: List<FieldRowEntity>)

    @Query("DELETE FROM field_rows WHERE fieldLayoutId = :fieldLayoutId")
    suspend fun deleteRows(fieldLayoutId: String)

    @Query("UPDATE field_layouts SET isActive = CASE WHEN id = :fieldLayoutId THEN 1 ELSE 0 END")
    suspend fun setActive(fieldLayoutId: String)
}

@Dao
interface RunDao {
    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertRun(run: ScanRunEntity)

    @Upsert
    suspend fun upsertRun(run: ScanRunEntity)

    @Query("UPDATE scan_runs SET state = :state, completedAt = :completedAt WHERE runId = :runId")
    suspend fun updateRunState(runId: String, state: String, completedAt: Long?)

    @Query("SELECT * FROM scan_runs WHERE runId = :runId LIMIT 1")
    fun observeRun(runId: String): Flow<ScanRunEntity?>

    @Query("SELECT * FROM scan_runs ORDER BY startedAt DESC LIMIT 1")
    fun observeLatestRun(): Flow<ScanRunEntity?>

    @Query("SELECT * FROM scan_runs ORDER BY startedAt DESC LIMIT :limit")
    suspend fun getRecentRuns(limit: Int): List<ScanRunEntity>

    @Query("SELECT * FROM scan_runs WHERE runId = :runId LIMIT 1")
    suspend fun getRun(runId: String): ScanRunEntity?

    @Query("DELETE FROM scan_runs WHERE runId = :runId")
    suspend fun deleteRun(runId: String)
}

@Dao
interface DecisionDao {
    @Upsert
    suspend fun upsertDecision(decision: DecisionEntity)

    @Upsert
    suspend fun upsertGeometry(geometry: FrontGeometryEntity)

    @Query("DELETE FROM front_geometry WHERE decisionId = :decisionId")
    suspend fun deleteGeometry(decisionId: String)

    @Query("SELECT * FROM decisions WHERE runId = :runId ORDER BY sequence ASC")
    fun observeDecisions(runId: String): Flow<List<DecisionEntity>>

    @Query("SELECT * FROM decisions WHERE runId = :runId ORDER BY sequence ASC")
    suspend fun getDecisions(runId: String): List<DecisionEntity>

    @Query("DELETE FROM decisions WHERE id = :decisionId")
    suspend fun deleteDecision(decisionId: String)

    @Query("SELECT * FROM front_geometry WHERE decisionId = :decisionId LIMIT 1")
    suspend fun getGeometry(decisionId: String): FrontGeometryEntity?

    @Query("SELECT id FROM decisions WHERE runId = :runId AND trackId = :trackId ORDER BY sequence DESC LIMIT 1")
    suspend fun latestDecisionIdForTrack(runId: String, trackId: String): String?

    @Query("UPDATE decisions SET treatmentStatus = :treatmentStatus, treatmentNote = :treatmentNote WHERE id = :decisionId")
    suspend fun updateTreatment(decisionId: String, treatmentStatus: String, treatmentNote: String?)
}

@Dao
interface RunEventDao {
    @Upsert
    suspend fun upsertEvent(event: RunEventEntity)

    @Query("SELECT * FROM run_events WHERE runId = :runId ORDER BY timestamp ASC")
    suspend fun getEvents(runId: String): List<RunEventEntity>

    @Query("DELETE FROM run_events WHERE runId = :runId")
    suspend fun deleteEvents(runId: String)
}

@Dao
interface ModelBundleDao {
    @Upsert
    suspend fun upsertBundle(bundle: ModelBundleEntity)

    @Query("SELECT * FROM model_bundles WHERE active = 1 LIMIT 1")
    fun observeActiveBundle(): Flow<ModelBundleEntity?>

    @Query("SELECT * FROM model_bundles WHERE active = 1 LIMIT 1")
    suspend fun getActiveBundle(): ModelBundleEntity?
}
