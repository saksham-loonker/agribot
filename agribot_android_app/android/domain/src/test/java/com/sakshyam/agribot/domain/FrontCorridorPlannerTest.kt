package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.FrontCorridorPlanner
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.RowId
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals

class FrontCorridorPlannerTest {
    @Test
    fun activeMiddleRowPairsWithNextRow() {
        val corridor = FrontCorridorPlanner.selectAdjacentRows(layout(active = "B"))

        assertEquals("B", corridor?.left?.id?.value)
        assertEquals("C", corridor?.right?.id?.value)
    }

    @Test
    fun activeLastRowPairsPreviousToActiveWithoutWrapping() {
        val corridor = FrontCorridorPlanner.selectAdjacentRows(layout(active = "C"))

        assertEquals("B", corridor?.left?.id?.value)
        assertEquals("C", corridor?.right?.id?.value)
    }

    @Test
    fun singleRowLayoutCannotCreateFrontCorridor() {
        val corridor = FrontCorridorPlanner.selectAdjacentRows(
            layout(active = "A").copy(rows = listOf(row("A", 1))),
        )

        assertEquals(null, corridor)
    }

    private fun layout(active: String) = FieldLayout(
        id = FieldId("field_2"),
        name = "Field 2",
        activeRowId = RowId(active),
        rows = listOf(row("A", 1), row("B", 2), row("C", 3)),
        startPlant = 1,
        plantStep = 1,
        plantCooldownSec = 2.0,
        updatedAt = Instant.parse("2026-06-08T00:00:00Z"),
    )

    private fun row(id: String, index: Int) = FieldRow(
        id = RowId(id),
        rowIndex = index,
        plantsPerRow = 20,
        plantSpacingM = 0.45,
        rowSpacingM = 1.4,
        yM = (index - 1) * 1.4,
    )
}
