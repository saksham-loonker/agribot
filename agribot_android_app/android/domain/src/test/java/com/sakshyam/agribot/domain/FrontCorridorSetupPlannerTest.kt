package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.FrontCorridorSetupPlanner
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.RowId
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class FrontCorridorSetupPlannerTest {
    @Test
    fun createsDefaultSetupFromActiveAdjacentRows() {
        val setup = FrontCorridorSetupPlanner.defaultFor(layout(active = "B"))

        assertEquals("B", setup?.leftRowId)
        assertEquals("C", setup?.rightRowId)
        assertEquals(1, setup?.nearestLeftPlantNumber)
        assertEquals(1, setup?.nearestRightPlantNumber)
    }

    @Test
    fun rowSelectionKeepsLeftAndRightDistinct() {
        val setup = FrontCorridorSetupPlanner.defaultFor(layout(active = "A"))!!

        val movedLeft = FrontCorridorSetupPlanner.selectLeft(layout(active = "A"), setup, "B")
        val movedRight = FrontCorridorSetupPlanner.selectRight(layout(active = "A"), movedLeft, "B")

        assertEquals("B", movedLeft.leftRowId)
        assertEquals("C", movedLeft.rightRowId)
        assertEquals("A", movedRight.leftRowId)
        assertEquals("B", movedRight.rightRowId)
    }

    @Test
    fun nearestPlantNumbersAreClampedToExistingRows() {
        val setup = FrontCorridorSetupPlanner.defaultFor(layout(active = "A"))!!

        val adjusted = FrontCorridorSetupPlanner.adjustNearestLeft(
            layout = layout(active = "A"),
            current = setup,
            delta = -100,
        )

        assertEquals(1, adjusted.nearestLeftPlantNumber)
    }

    @Test
    fun singleRowLayoutCannotCreateSetup() {
        val setup = FrontCorridorSetupPlanner.defaultFor(
            layout(active = "A").copy(rows = listOf(row("A", 1))),
        )

        assertNull(setup)
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
