package com.sakshyam.agribot.featurescan

import com.sakshyam.agribot.domain.model.RunFieldMap
import com.sakshyam.agribot.featurescan.tracking.TrackedPlant

enum class FarmerResultCategory { HEALTHY, CHECK, REVIEW }

data class FarmerFinding(
    val id: String,
    val location: String,
    val label: String,
    val category: FarmerResultCategory,
    val saved: Boolean,
)

/** Counts use one exclusive category per observation; planned positions are excluded. */
data class FarmerResultsSummary(val findings: List<FarmerFinding>) {
    val total get() = findings.size
    val healthy get() = findings.count { it.category == FarmerResultCategory.HEALTHY }
    val check get() = findings.count { it.category == FarmerResultCategory.CHECK }
    val review get() = findings.count { it.category == FarmerResultCategory.REVIEW }

    companion object {
        fun from(plants: List<TrackedPlant>, savedMap: RunFieldMap?): FarmerResultsSummary {
            val findings = if (savedMap != null) {
                savedMap.rows.flatMap { row ->
                    row.cells.filterNot { it.status.equals("empty", true) }.map { cell ->
                        val label = cell.label.substringAfter(": ", cell.label).trim()
                        FarmerFinding(
                            id = "${row.rowId}:${cell.plantNumber}",
                            location = "${row.rowLabel} · #${cell.plantNumber}",
                            label = label,
                            category = category(label, cell.status.equals("uncertain", true) || cell.status.equals("skipped", true)),
                            saved = true,
                        )
                    }
                }
            } else {
                plants.filter { it.isConfirmed }.map { plant ->
                    FarmerFinding(
                        id = plant.plantId, location = plant.plantId.replace('_', ' '), label = plant.healthStatus,
                        category = category(plant.healthStatus, plant.isOccluded || plant.classificationPending), saved = false,
                    )
                }
            }
            return FarmerResultsSummary(findings)
        }

        private fun category(label: String, requiresReview: Boolean): FarmerResultCategory = when {
            requiresReview || label.isBlank() || label.equals("Uncertain", true) || label.equals("Unknown", true) -> FarmerResultCategory.REVIEW
            label.equals("Healthy", true) -> FarmerResultCategory.HEALTHY
            else -> FarmerResultCategory.CHECK
        }
    }
}
