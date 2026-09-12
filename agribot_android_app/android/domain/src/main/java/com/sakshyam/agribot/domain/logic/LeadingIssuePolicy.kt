package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.LeadingObservedIssue
import com.sakshyam.agribot.domain.model.ObservedIssue

/**
 * Conservative aggregation for the map headline. It reports an observed issue,
 * never causal proof, and refuses to promote a one-off or tied result.
 */
object LeadingIssuePolicy {
    private val NON_ISSUE_LABELS = setOf("healthy", "uncertain", "unknown", "skipped", "manual")

    fun evaluate(
        observations: List<ObservedIssue>,
        plannedCount: Int? = null,
        minimumInspected: Int = 3,
        minimumVotes: Int = 2,
        minimumMargin: Int = 1,
    ): LeadingObservedIssue {
        val inspected = observations.filter { observation ->
                observation.confirmed &&
                !observation.reviewRequired &&
                observation.geometryValid &&
                normalize(observation.label) !in NON_ACTIONABLE_LABELS
        }
        val counts = inspected
            .filter { normalize(it.label) !in NON_ISSUE_LABELS }
            .groupingBy { normalize(it.label) }
            .eachCount()
            .toList()
            .sortedWith(compareByDescending<Pair<String, Int>> { it.second }.thenBy { it.first })
        val top = counts.firstOrNull()
        val second = counts.getOrNull(1)?.second ?: 0
        val margin = (top?.second ?: 0) - second
        val label = top?.first
        val canPromote = inspected.size >= minimumInspected &&
            top != null &&
            top.second >= minimumVotes &&
            margin >= minimumMargin
        return LeadingObservedIssue(
            label = if (canPromote) label else null,
            affectedCount = if (canPromote) top!!.second else 0,
            inspectedCount = inspected.size,
            plannedCount = plannedCount?.takeIf { it > 0 },
            marginOverSecond = margin.coerceAtLeast(0),
            provisional = canPromote && (inspected.size < 10 || margin < 2),
            reason = when {
                inspected.isEmpty() -> "no_confirmed_issue_observations"
                inspected.size < minimumInspected -> "more_confirmed_plants_needed"
                top == null -> "no_issue_label"
                top.second < minimumVotes -> "more_votes_needed"
                margin < minimumMargin -> "no_clear_leader"
                else -> "leading_observed_issue"
            },
        )
    }

    private fun normalize(label: String): String = label
        .trim()
        .replace('_', ' ')
        .lowercase()
        .ifBlank { "unknown" }

    private val NON_ACTIONABLE_LABELS = setOf("uncertain", "unknown", "skipped", "manual")
}
