package com.sakshyam.agribot.data.repository

import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.RunId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class LocalExportPathNamesTest {
    @Test
    fun preservesNormalAndroidRunAndDecisionNames() {
        val runId = RunId("android_20260608_145500")
        val decisionId = DecisionId("android_20260608_145500_12")

        assertEquals("agribot_run_android_20260608_145500", LocalExportPathNames.runDirectoryName(runId))
        assertEquals("android_20260608_145500_12_manual.jpg", LocalExportPathNames.evidenceFileName(decisionId, "manual"))
    }

    @Test
    fun sanitizesImportedRunIdsBeforeUsingThemAsDirectories() {
        val name = LocalExportPathNames.runDirectoryName(RunId("../pi\\run:field A"))

        assertTrue(name.startsWith("agribot_run_pi_run_field_A_"))
        assertSafePathSegment(name)
    }

    @Test
    fun sanitizesDecisionAndEvidenceKindBeforeUsingThemAsFileNames() {
        val name = LocalExportPathNames.evidenceFileName(
            decisionId = DecisionId("..\\decision/12"),
            kind = "manual snapshot",
        )

        assertTrue(name.startsWith("decision_12_"))
        assertTrue(name.endsWith(".jpg"))
        assertSafePathSegment(name)
    }

    private fun assertSafePathSegment(name: String) {
        assertFalse(name.contains(".."))
        assertFalse(name.contains('/'))
        assertFalse(name.contains('\\'))
        assertFalse(name.contains(':'))
    }
}
