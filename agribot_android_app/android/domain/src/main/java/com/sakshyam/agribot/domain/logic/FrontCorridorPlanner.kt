package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FrontCorridorRows

object FrontCorridorPlanner {
    fun selectAdjacentRows(layout: FieldLayout): FrontCorridorRows? {
        val rows = layout.rows.sortedBy { it.rowIndex }
        if (rows.size < 2) return null
        val activeIndex = rows.indexOfFirst { it.id == layout.activeRowId }.takeIf { it >= 0 } ?: 0
        val leftIndex = if (activeIndex < rows.lastIndex) activeIndex else activeIndex - 1
        val rightIndex = leftIndex + 1
        return FrontCorridorRows(
            left = rows.getOrNull(leftIndex) ?: return null,
            right = rows.getOrNull(rightIndex) ?: return null,
        )
    }
}
