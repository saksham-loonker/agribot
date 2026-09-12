package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.SideScanMapper
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.RowId
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SideScanMapperTest {
    @Test
    fun startsAtActiveRowAndHonorsRowOverridesFromFarmerConfig() {
        val mapper = SideScanMapper(field2Layout())

        val first = mapper.position(scanIndex = 1)
        val fifteenth = mapper.position(scanIndex = 15)
        val sixteenth = mapper.position(scanIndex = 16)

        assertEquals("Field 2", first.fieldId)
        assertEquals("B", first.rowId)
        assertEquals(2, first.rowIndex)
        assertEquals(1, first.plantColumn)
        assertEquals(1, first.plantNumber)
        assertEquals(0.0, first.xM)
        assertEquals(1.4, first.yM)
        assertFalse(first.fieldComplete)

        assertEquals("B", fifteenth.rowId)
        assertEquals(15, fifteenth.plantColumn)
        assertEquals(4.9, fifteenth.xM)

        assertEquals("C", sixteenth.rowId)
        assertEquals(1, sixteenth.plantColumn)
        assertEquals(2.8, sixteenth.yM)
    }

    @Test
    fun wrapsIntoNextScanPassAfterPlannedTotal() {
        val mapper = SideScanMapper(field2Layout())

        val last = mapper.position(scanIndex = 57)
        val wrapped = mapper.position(scanIndex = 58)

        assertEquals("A", last.rowId)
        assertTrue(last.fieldComplete)
        assertEquals(1, last.scanPass)

        assertEquals("B", wrapped.rowId)
        assertEquals(1, wrapped.plantColumn)
        assertEquals(2, wrapped.scanPass)
        assertTrue(wrapped.fieldComplete)
    }

    private fun field2Layout() = FieldLayout(
        id = FieldId("field_2"),
        name = "Field 2",
        activeRowId = RowId("B"),
        rows = listOf(
            FieldRow(RowId("A"), rowIndex = 1, plantsPerRow = 20, plantSpacingM = 0.45, rowSpacingM = 1.4, yM = 0.0),
            FieldRow(RowId("B"), rowIndex = 2, plantsPerRow = 15, plantSpacingM = 0.35, rowSpacingM = 1.4, yM = 1.4),
            FieldRow(RowId("C"), rowIndex = 3, plantsPerRow = 22, plantSpacingM = 0.5, rowSpacingM = 1.4, yM = 2.8),
        ),
        startPlant = 1,
        plantStep = 1,
        plantCooldownSec = 2.0,
        updatedAt = Instant.parse("2026-05-30T13:37:22Z"),
    )
}
