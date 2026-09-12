package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.FieldLayoutEditor
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.RowId
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class FieldLayoutEditorTest {
    @Test
    fun selectsAnExistingActiveRowWithoutChangingRows() {
        val updated = FieldLayoutEditor.selectActiveRow(layout(), RowId("C"), now)

        assertEquals(RowId("C"), updated.activeRowId)
        assertEquals(listOf("A", "C"), updated.rows.map { it.id.value })
        assertEquals(now, updated.updatedAt)
    }

    @Test
    fun rejectsUnknownActiveRow() {
        assertFailsWith<IllegalArgumentException> {
            FieldLayoutEditor.selectActiveRow(layout(), RowId("Z"), now)
        }
    }

    @Test
    fun updatesPlantsPerRowForActiveRowOnly() {
        val updated = FieldLayoutEditor.updatePlantsPerRow(layout(), RowId("A"), 25, now)

        assertEquals(25, updated.rows.first { it.id == RowId("A") }.plantsPerRow)
        assertEquals(18, updated.rows.first { it.id == RowId("C") }.plantsPerRow)
        assertEquals(now, updated.updatedAt)
    }

    @Test
    fun clampsPlantsAndCooldownToFieldSafeBounds() {
        val fewPlants = FieldLayoutEditor.updatePlantsPerRow(layout(), RowId("A"), -4, now)
        val lowCooldown = FieldLayoutEditor.updatePlantCooldown(layout(), 0.1, now)
        val highCooldown = FieldLayoutEditor.updatePlantCooldown(layout(), 60.0, now)

        assertEquals(1, fewPlants.rows.first { it.id == RowId("A") }.plantsPerRow)
        assertEquals(0.5, lowCooldown.plantCooldownSec)
        assertEquals(30.0, highCooldown.plantCooldownSec)
    }

    private val now = Instant.parse("2026-06-08T10:00:00Z")

    private fun layout() = FieldLayout(
        id = FieldId("field_2"),
        name = "Field 2",
        activeRowId = RowId("A"),
        rows = listOf(
            FieldRow(RowId("A"), 1, 12, 0.45, 1.4, 0.0),
            FieldRow(RowId("C"), 3, 18, 0.5, 1.4, 2.8),
        ),
        startPlant = 1,
        plantStep = 1,
        plantCooldownSec = 2.0,
        updatedAt = Instant.parse("2026-06-08T09:00:00Z"),
    )
}
