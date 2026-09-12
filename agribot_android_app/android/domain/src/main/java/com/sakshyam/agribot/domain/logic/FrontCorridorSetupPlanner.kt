package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.FrontCorridorSetup

object FrontCorridorSetupPlanner {
    fun defaultFor(layout: FieldLayout): FrontCorridorSetup? {
        val corridor = FrontCorridorPlanner.selectAdjacentRows(layout) ?: return null
        return FrontCorridorSetup(
            leftRowId = corridor.left.id.value,
            rightRowId = corridor.right.id.value,
            nearestLeftPlantNumber = clampPlant(layout.startPlant, corridor.left),
            nearestRightPlantNumber = clampPlant(layout.startPlant, corridor.right),
        )
    }

    fun coerceForLayout(layout: FieldLayout, current: FrontCorridorSetup?): FrontCorridorSetup? {
        if (current == null) return defaultFor(layout)
        val rows = sortedRows(layout)
        if (rows.size < 2) return null
        val left = rows.firstOrNull { it.id.value == current.leftRowId } ?: return defaultFor(layout)
        val right = rows.firstOrNull {
            it.id.value == current.rightRowId && it.id != left.id
        } ?: alternativeRight(rows, left) ?: return defaultFor(layout)
        return FrontCorridorSetup(
            leftRowId = left.id.value,
            rightRowId = right.id.value,
            nearestLeftPlantNumber = clampPlant(current.nearestLeftPlantNumber, left),
            nearestRightPlantNumber = clampPlant(current.nearestRightPlantNumber, right),
        )
    }

    fun selectLeft(
        layout: FieldLayout,
        current: FrontCorridorSetup,
        rowId: String,
    ): FrontCorridorSetup {
        val rows = sortedRows(layout)
        val left = rows.firstOrNull { it.id.value == rowId } ?: return coerceForLayout(layout, current) ?: current
        val right = rows.firstOrNull {
            it.id.value == current.rightRowId && it.id != left.id
        } ?: alternativeRight(rows, left) ?: return current
        return FrontCorridorSetup(
            leftRowId = left.id.value,
            rightRowId = right.id.value,
            nearestLeftPlantNumber = clampPlant(current.nearestLeftPlantNumber, left),
            nearestRightPlantNumber = clampPlant(current.nearestRightPlantNumber, right),
        )
    }

    fun selectRight(
        layout: FieldLayout,
        current: FrontCorridorSetup,
        rowId: String,
    ): FrontCorridorSetup {
        val rows = sortedRows(layout)
        val right = rows.firstOrNull { it.id.value == rowId } ?: return coerceForLayout(layout, current) ?: current
        val left = rows.firstOrNull {
            it.id.value == current.leftRowId && it.id != right.id
        } ?: alternativeLeft(rows, right) ?: return current
        return FrontCorridorSetup(
            leftRowId = left.id.value,
            rightRowId = right.id.value,
            nearestLeftPlantNumber = clampPlant(current.nearestLeftPlantNumber, left),
            nearestRightPlantNumber = clampPlant(current.nearestRightPlantNumber, right),
        )
    }

    fun adjustNearestLeft(
        layout: FieldLayout,
        current: FrontCorridorSetup,
        delta: Int,
    ): FrontCorridorSetup {
        val left = layout.rows.firstOrNull { it.id.value == current.leftRowId } ?: return current
        return current.copy(
            nearestLeftPlantNumber = clampPlant(current.nearestLeftPlantNumber + delta, left),
        )
    }

    fun adjustNearestRight(
        layout: FieldLayout,
        current: FrontCorridorSetup,
        delta: Int,
    ): FrontCorridorSetup {
        val right = layout.rows.firstOrNull { it.id.value == current.rightRowId } ?: return current
        return current.copy(
            nearestRightPlantNumber = clampPlant(current.nearestRightPlantNumber + delta, right),
        )
    }

    private fun sortedRows(layout: FieldLayout): List<FieldRow> =
        layout.rows.sortedBy { it.rowIndex }

    private fun alternativeRight(rows: List<FieldRow>, left: FieldRow): FieldRow? =
        rows.firstOrNull { it.rowIndex > left.rowIndex } ?: rows.firstOrNull { it.id != left.id }

    private fun alternativeLeft(rows: List<FieldRow>, right: FieldRow): FieldRow? =
        rows.lastOrNull { it.rowIndex < right.rowIndex } ?: rows.firstOrNull { it.id != right.id }

    private fun clampPlant(value: Int, row: FieldRow): Int =
        value.coerceIn(1, row.plantsPerRow.coerceAtLeast(1))
}
