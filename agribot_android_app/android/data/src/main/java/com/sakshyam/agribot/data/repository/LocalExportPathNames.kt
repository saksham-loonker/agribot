package com.sakshyam.agribot.data.repository

import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.RunId
import java.security.MessageDigest

internal object LocalExportPathNames {
    fun runDirectoryName(runId: RunId): String = "agribot_run_${safeSegment(runId.value)}"

    fun evidenceFileName(decisionId: DecisionId, kind: String): String =
        "${safeSegment(decisionId.value)}_${safeSegment(kind)}.jpg"

    fun evidenceFilePrefix(decisionId: DecisionId): String = "${safeSegment(decisionId.value)}_"

    private fun safeSegment(raw: String): String {
        val normalized = raw
            .trim()
            .replace(Regex("[/\\\\]+"), "_")
            .replace(Regex("[^A-Za-z0-9_.-]"), "_")
            .replace(Regex("\\.{2,}"), "_")
            .replace(Regex("_+"), "_")
            .trim('_', '.', '-')
        val base = (normalized.ifBlank { "id" }).take(MAX_SEGMENT_BASE_LENGTH)
        val unchanged = base == raw && raw.length <= MAX_SEGMENT_BASE_LENGTH
        return if (unchanged) base else "${base}_${sha256(raw).take(HASH_SUFFIX_LENGTH)}"
    }

    private fun sha256(value: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(value.toByteArray(Charsets.UTF_8))
        return digest.joinToString("") { "%02x".format(it) }
    }

    private const val MAX_SEGMENT_BASE_LENGTH = 80
    private const val HASH_SUFFIX_LENGTH = 12
}
