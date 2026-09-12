package com.sakshyam.agribot.featurescan

import com.sakshyam.agribot.domain.model.RunFieldMap
import com.sakshyam.agribot.domain.model.RunFieldMapCell
import com.sakshyam.agribot.domain.model.RunFieldMapRow
import kotlin.test.Test
import kotlin.test.assertEquals

class FarmerResultsSummaryTest {
    private fun summary(vararg cells: RunFieldMapCell) = FarmerResultsSummary.from(
        emptyList(), RunFieldMap("Field", listOf(RunFieldMapRow("1", "Row 1", cells.toList()))),
    )
    @Test fun unscannedPositionsAreNotResults() {
        assertEquals(0, summary(RunFieldMapCell(1, "", "empty", null)).total)
    }
    @Test fun uncertainHealthyIsReviewNotHealthy() {
        val result = summary(RunFieldMapCell(1, "Plant 1: Healthy", "uncertain", 1))
        assertEquals(0, result.healthy)
        assertEquals(1, result.review)
    }
    @Test fun categoriesAreExclusive() {
        val result = summary(
            RunFieldMapCell(1, "Plant 1: Healthy", "ok", 1),
            RunFieldMapCell(2, "Plant 2: Late_blight", "ok", 2),
            RunFieldMapCell(3, "Plant 3: Unknown", "ok", 3),
            RunFieldMapCell(4, "Plant 4: Healthy", "skipped", 4),
        )
        assertEquals(4, result.total)
        assertEquals(1, result.healthy)
        assertEquals(1, result.check)
        assertEquals(2, result.review)
    }
    @Test fun blankLabelsRequireReview() {
        assertEquals(1, summary(RunFieldMapCell(1, " ", "ok", 1)).review)
    }
}
