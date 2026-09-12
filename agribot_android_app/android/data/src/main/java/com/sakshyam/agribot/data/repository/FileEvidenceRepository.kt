package com.sakshyam.agribot.data.repository

import android.content.Context
import com.sakshyam.agribot.domain.model.AnalysisFrame
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.EvidenceCaptureResult
import com.sakshyam.agribot.domain.model.EvidenceRetentionSnapshot
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.repository.EvidenceRepository
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import javax.inject.Named
import javax.inject.Inject
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class FileEvidenceRepository @Inject constructor(
    @ApplicationContext private val context: Context,
    @Named("io") private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
) : EvidenceRepository {
    override suspend fun saveEvidence(
        runId: RunId,
        decisionId: DecisionId,
        frame: AnalysisFrame,
        kind: String,
    ): EvidenceCaptureResult = withContext(ioDispatcher) {
        if (frame.jpegBytes.isEmpty()) {
            return@withContext EvidenceCaptureResult(path = null, status = "no_frame")
        }
        val dir = evidenceDir(runId)
        val file = File(dir, LocalExportPathNames.evidenceFileName(decisionId, kind))
        val temporary = File(dir, ".${file.name}.${System.nanoTime()}.tmp")
        try {
            temporary.outputStream().use { output ->
                output.write(frame.jpegBytes)
                output.flush()
                output.fd.sync()
            }
            try {
                Files.move(
                    temporary.toPath(),
                    file.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE,
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(temporary.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
            } catch (_: UnsupportedOperationException) {
                Files.move(temporary.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
        } finally {
            temporary.delete()
        }
        EvidenceCaptureResult(path = file.absolutePath, status = "saved")
    }

    override suspend fun retentionSnapshot(runId: RunId): EvidenceRetentionSnapshot = withContext(ioDispatcher) {
        val frames = evidenceDir(runId).listFiles { file -> file.isFile && file.extension.lowercase() in setOf("jpg", "jpeg") }
            ?.toList()
            .orEmpty()
        EvidenceRetentionSnapshot(
            bytesUsed = frames.sumOf { it.length() },
            framesUsed = frames.size,
        )
    }

    override suspend fun deleteEvidence(runId: RunId, decisionId: DecisionId): Unit = withContext(ioDispatcher) {
        val dir = evidenceDir(runId)
        val prefix = LocalExportPathNames.evidenceFilePrefix(decisionId)
        dir.listFiles { file -> file.isFile && file.name.startsWith(prefix) }
            ?.forEach { file ->
                if (!file.delete() && file.exists()) {
                    throw java.io.IOException("Unable to delete evidence file: ${file.absolutePath}")
                }
            }
        Unit
    }

    private fun evidenceDir(runId: RunId): File =
        File(File(File(context.filesDir, "exports"), LocalExportPathNames.runDirectoryName(runId)), "evidence_frames")
            .apply { mkdirs() }
}
