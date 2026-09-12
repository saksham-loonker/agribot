package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.LeadingIssuePolicy
import com.sakshyam.agribot.domain.model.ObservedIssue
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class LeadingIssuePolicyTest {
    @Test
    fun oneUncertainObservationCannotLeadTheField() {
        val result = LeadingIssuePolicy.evaluate(
            observations = listOf(
                ObservedIssue("p1", "Late_blight", confirmed = true),
                ObservedIssue("p2", "Uncertain", confirmed = false, reviewRequired = true),
            ),
            plannedCount = 10,
        )

        assertNull(result.label)
        assertEquals(1, result.inspectedCount)
        assertEquals("more_confirmed_plants_needed", result.reason)
    }

    @Test
    fun leaderIncludesDenominatorAndCoverage() {
        val result = LeadingIssuePolicy.evaluate(
            observations = listOf(
                ObservedIssue("p1", "Early_blight", confirmed = true),
                ObservedIssue("p2", "Early_blight", confirmed = true),
                ObservedIssue("p3", "Healthy", confirmed = true),
            ),
            plannedCount = 20,
        )

        assertEquals("early blight", result.label)
        assertEquals(2, result.affectedCount)
        assertEquals(3, result.inspectedCount)
        assertEquals("3 / 20 checked", result.coverageLabel)
        assertEquals(true, result.provisional)
    }

    @Test
    fun tiedIssuesDoNotProduceAHeadline() {
        val result = LeadingIssuePolicy.evaluate(
            observations = listOf(
                ObservedIssue("p1", "Early_blight", confirmed = true),
                ObservedIssue("p2", "Early_blight", confirmed = true),
                ObservedIssue("p3", "Late_blight", confirmed = true),
                ObservedIssue("p4", "Late_blight", confirmed = true),
            ),
        )

        assertNull(result.label)
        assertEquals("no_clear_leader", result.reason)
    }
}
