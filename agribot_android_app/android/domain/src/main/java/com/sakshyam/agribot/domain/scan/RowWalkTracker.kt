package com.sakshyam.agribot.domain.scan

import kotlin.math.floor

/**
 * Which plant of the current row the farmer is standing at, from walked distance.
 * Distance comes from the step sensor x calibrated stride (GPS is far too coarse: 3-10 m).
 * Manual Next/Previous re-anchor the distance to the start of the chosen plant, so the farmer can
 * always correct drift and later steps continue from there.
 */
class RowWalkTracker(
    val rowCount: Int,
    val plantsPerRow: Int,
    val plantSpacingM: Double,
    strideM: Double,
) {
    init {
        require(rowCount >= 1 && plantsPerRow >= 1)
        require(plantSpacingM > 0.0)
    }

    var strideM: Double = strideM
        private set
    var rowIndex: Int = 1
        private set
    var distanceM: Double = 0.0
        private set
    var steps: Int = 0
        private set

    /** 1-based plant number in the current row. */
    val plantNumber: Int
        get() = (floor(distanceM / plantSpacingM).toInt() + 1).coerceIn(1, plantsPerRow)

    val atLastPlant: Boolean get() = plantNumber == plantsPerRow
    val atLastRow: Boolean get() = rowIndex == rowCount

    fun onStep() {
        steps++
        distanceM += strideM
    }

    fun setStride(m: Double) {
        require(m in 0.2..2.0) { "stride out of range" }
        strideM = m
    }

    fun nextPlant(): Boolean {
        if (plantNumber >= plantsPerRow) return false
        jumpTo(plantNumber + 1); return true
    }

    fun previousPlant(): Boolean {
        if (plantNumber <= 1) return false
        jumpTo(plantNumber - 1); return true
    }

    /** Re-anchors at the middle of [plant] so a step either way doesn't immediately change plant. */
    fun jumpTo(plant: Int) {
        distanceM = (plant.coerceIn(1, plantsPerRow) - 0.5) * plantSpacingM
    }

    /** Restores a saved position (process death). */
    fun restore(row: Int, distance: Double, stepCount: Int) {
        rowIndex = row.coerceIn(1, rowCount)
        distanceM = distance.coerceAtLeast(0.0)
        steps = stepCount
    }

    /** Starts the next row; returns false if this was the last row. */
    fun nextRow(): Boolean {
        if (rowIndex >= rowCount) return false
        rowIndex++
        distanceM = 0.0
        return true
    }
}
