package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.SideScanPosition

class SideScanMapper(fieldLayout: FieldLayout) {
    private val startPlant = fieldLayout.startPlant.coerceAtLeast(1)
    private val plantStep = fieldLayout.plantStep.coerceAtLeast(1)
    private val fieldName = fieldLayout.name.ifBlank { fieldLayout.id.value }
    private val rowSpecs: List<FieldRow>
    private val plannedTotal: Int

    init {
        require(fieldLayout.rows.isNotEmpty()) { "Field layout must have at least one row" }
        val activeIndex = fieldLayout.rows.indexOfFirst { it.id == fieldLayout.activeRowId }.let { index ->
            if (index >= 0) index else 0
        }
        rowSpecs = fieldLayout.rows.drop(activeIndex) + fieldLayout.rows.take(activeIndex)
        plannedTotal = rowSpecs.sumOf { it.plantsPerRow.coerceAtLeast(1) }.coerceAtLeast(1)
    }

    fun position(scanIndex: Int): SideScanPosition {
        val zeroBased = (scanIndex - 1).coerceAtLeast(0)
        val passIndex = zeroBased / plannedTotal
        var plantCol = zeroBased % plannedTotal
        var rowSpec = rowSpecs.first()

        for (candidate in rowSpecs) {
            val plants = candidate.plantsPerRow.coerceAtLeast(1)
            if (plantCol < plants) {
                rowSpec = candidate
                break
            }
            plantCol -= plants
        }

        val rowId = rowSpec.id.value
        val plantNumber = startPlant + plantCol * plantStep
        return SideScanPosition(
            fieldId = fieldName,
            rowId = rowId,
            rowIndex = rowSpec.rowIndex,
            plantColumn = plantCol + 1,
            plantNumber = plantNumber,
            plantKey = "$fieldName|$rowId|$plantNumber",
            xM = round4(plantCol * rowSpec.plantSpacingM),
            yM = rowSpec.yM,
            scanIndex = scanIndex.coerceAtLeast(1),
            scanPass = passIndex + 1,
            fieldComplete = zeroBased + 1 >= plannedTotal,
            plannedTotalPlants = plannedTotal,
            plantDisplay = plantDisplay(fieldName, rowId, plantNumber),
        )
    }

    private fun round4(value: Double): Double =
        kotlin.math.round(value * 10_000.0) / 10_000.0

    companion object {
        fun plantDisplay(fieldId: String, rowId: String, plantNumber: Int): String =
            "$fieldId / Row $rowId / Plant $plantNumber"
    }
}
