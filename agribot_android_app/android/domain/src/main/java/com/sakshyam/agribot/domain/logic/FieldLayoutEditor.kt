package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.RowId
import java.time.Instant

object FieldLayoutEditor {
    fun selectActiveRow(layout: FieldLayout, rowId: RowId, updatedAt: Instant = Instant.now()): FieldLayout {
        require(layout.rows.any { row -> row.id == rowId }) { "Unknown row ${rowId.value} for field ${layout.id.value}" }
        return layout.copy(activeRowId = rowId, updatedAt = updatedAt)
    }

    fun updatePlantsPerRow(
        layout: FieldLayout,
        rowId: RowId,
        plantsPerRow: Int,
        updatedAt: Instant = Instant.now(),
    ): FieldLayout {
        require(layout.rows.any { row -> row.id == rowId }) { "Unknown row ${rowId.value} for field ${layout.id.value}" }
        val safePlantsPerRow = plantsPerRow.coerceAtLeast(MIN_PLANTS_PER_ROW)
        return layout.copy(
            rows = layout.rows.map { row ->
                if (row.id == rowId) row.copy(plantsPerRow = safePlantsPerRow) else row
            },
            updatedAt = updatedAt,
        )
    }

    fun updatePlantCooldown(layout: FieldLayout, plantCooldownSec: Double, updatedAt: Instant = Instant.now()): FieldLayout =
        layout.copy(
            plantCooldownSec = plantCooldownSec.coerceIn(MIN_COOLDOWN_SEC, MAX_COOLDOWN_SEC),
            updatedAt = updatedAt,
        )

    const val MIN_PLANTS_PER_ROW = 1
    const val MIN_COOLDOWN_SEC = 0.5
    const val MAX_COOLDOWN_SEC = 30.0
}
