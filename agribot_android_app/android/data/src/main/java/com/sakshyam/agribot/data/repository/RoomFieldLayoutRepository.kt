package com.sakshyam.agribot.data.repository

import androidx.room.withTransaction
import com.sakshyam.agribot.data.db.AgribotDatabase
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.repository.FieldLayoutRepository
import java.time.Instant
import javax.inject.Inject
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

class RoomFieldLayoutRepository @Inject constructor(
    private val database: AgribotDatabase,
) : FieldLayoutRepository {
    override fun observeLayouts(): Flow<List<FieldLayout>> =
        database.fieldLayoutDao().observeLayouts().map { layouts ->
            layouts.map { entity -> entity.toDomain(database.fieldLayoutDao().getRows(entity.id)) }
        }

    override fun observeActiveLayout(): Flow<FieldLayout?> =
        database.fieldLayoutDao().observeActiveLayout().map { entity ->
            entity?.toDomain(database.fieldLayoutDao().getRows(entity.id))
        }

    override suspend fun layoutById(fieldId: FieldId): FieldLayout? =
        database.fieldLayoutDao().getLayout(fieldId.value)?.let { entity ->
            entity.toDomain(database.fieldLayoutDao().getRows(entity.id))
        }

    override suspend fun saveLayout(layout: FieldLayout) {
        database.withTransaction {
            val activeLayout = database.fieldLayoutDao().observeActiveLayout().first()
            val isActive = activeLayout == null || activeLayout.id == layout.id.value
            database.fieldLayoutDao().upsertLayout(layout.toEntity(isActive = isActive, now = Instant.now()))
            database.fieldLayoutDao().deleteRows(layout.id.value)
            database.fieldLayoutDao().upsertRows(layout.toRowEntities())
        }
    }

    override suspend fun setActiveLayout(fieldId: FieldId) {
        database.fieldLayoutDao().setActive(fieldId.value)
    }
}
