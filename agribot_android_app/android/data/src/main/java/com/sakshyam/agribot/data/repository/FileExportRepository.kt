package com.sakshyam.agribot.data.repository

import android.content.Context
import android.graphics.pdf.PdfDocument
import com.sakshyam.agribot.domain.logic.ExportSerializer
import com.sakshyam.agribot.domain.logic.RunSummaryReducer
import com.sakshyam.agribot.domain.model.ExportedFile
import com.sakshyam.agribot.domain.model.DiagnosticsSnapshot
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.repository.ExportRepository
import com.sakshyam.agribot.domain.repository.FieldLayoutRepository
import com.sakshyam.agribot.domain.repository.RunRepository
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import javax.inject.Named
import javax.inject.Inject
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

class FileExportRepository @Inject constructor(
    @ApplicationContext private val context: Context,
    private val runRepository: RunRepository,
    private val fieldLayoutRepository: FieldLayoutRepository,
    @Named("io") private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
) : ExportRepository {
    override suspend fun exportCsv(runId: RunId): ExportedFile = withContext(ioDispatcher) {
        val decisions = runRepository.decisionsForRun(runId)
        val file = exportFile(runId, "events.csv", ExportSerializer.toEventsCsv(decisions))
        ExportedFile(file.name, "text/csv", file.absolutePath)
    }

    override suspend fun exportJsonl(runId: RunId): ExportedFile = withContext(ioDispatcher) {
        val decisions = runRepository.decisionsForRun(runId)
        val file = exportFile(runId, "decisions.jsonl", ExportSerializer.toJsonl(decisions))
        ExportedFile(file.name, "application/x-jsonlines", file.absolutePath)
    }

    override suspend fun exportLogs(runId: RunId): ExportedFile = withContext(ioDispatcher) {
        val events = runRepository.eventsForRun(runId)
        val file = exportFile(runId, "app_log.jsonl", ExportSerializer.toRunEventsJsonl(events))
        ExportedFile(file.name, "application/x-jsonlines", file.absolutePath)
    }

    override suspend fun exportPdf(runId: RunId): ExportedFile = withContext(ioDispatcher) {
        val run = runRepository.runById(runId) ?: error("Run not found for export: ${runId.value}")
        val decisions = runRepository.decisionsForRun(runId)
        val summary = RunSummaryReducer.reduce(runId, decisions)
        val layout = fieldLayoutRepository.layoutById(run.fieldLayoutId)
        val pdfDocument = PdfDocument()
        var pageNumber = 1
        var currentPage = pdfDocument.startPage(PdfDocument.PageInfo.Builder(595, 842, pageNumber).create())
        var canvas = currentPage.canvas
        val paint = android.text.TextPaint().apply {
            textSize = 12f
            isAntiAlias = true
        }
        val titlePaint = android.text.TextPaint().apply {
            textSize = 20f
            isAntiAlias = true
            isFakeBoldText = true
        }
        val headerPaint = android.text.TextPaint().apply {
            textSize = 14f
            isAntiAlias = true
            isFakeBoldText = true
        }
        var y = 60f
        fun newPage() {
            pdfDocument.finishPage(currentPage)
            pageNumber++
            currentPage = pdfDocument.startPage(PdfDocument.PageInfo.Builder(595, 842, pageNumber).create())
            canvas = currentPage.canvas
            y = 60f
        }
        canvas.drawText("Agribot Scan Report", 50f, y, titlePaint)
        y += 30f
        canvas.drawText("Run ID: ${run.runId.value}", 50f, y, paint)
        y += 20f
        canvas.drawText("Field: ${layout?.name ?: run.fieldLayoutId.value}", 50f, y, paint)
        y += 20f
        canvas.drawText("Mode: ${run.mode.name}", 50f, y, paint)
        y += 20f
        canvas.drawText("State: ${run.state.name}", 50f, y, paint)
        y += 30f
        canvas.drawText("Summary", 50f, y, headerPaint)
        y += 20f
        canvas.drawText("Total decisions: ${summary.decisions}", 50f, y, paint)
        y += 15f
        canvas.drawText("Healthy (OK): ${summary.ok}", 50f, y, paint)
        y += 15f
        canvas.drawText("Sick: ${summary.sick}", 50f, y, paint)
        y += 15f
        canvas.drawText("Uncertain: ${summary.uncertain}", 50f, y, paint)
        y += 15f
        canvas.drawText("Uncertain rate: ${"%.1f".format(Locale.US, summary.uncertainRate * 100.0)}%", 50f, y, paint)
        y += 15f
        canvas.drawText("Avg confidence: ${"%.1f".format(Locale.US, summary.avgConfidence * 100.0)}%", 50f, y, paint)
        y += 30f
        canvas.drawText("Plant Decisions", 50f, y, headerPaint)
        y += 20f
        decisions.forEach { decision ->
            if (y > 780f) {
                newPage()
            }
            val status = decision.status.name
            val label = decision.label
            val conf = "${(decision.confidence * 100).toInt()}%"
            val plantNum = decision.plantNumber?.toString() ?: "-"
            canvas.drawText("Plant #$plantNum: $label ($conf) - $status", 50f, y, paint)
            y += 15f
        }
        pdfDocument.finishPage(currentPage)
        val runDir = File(exportsRoot(), LocalExportPathNames.runDirectoryName(runId)).apply { mkdirs() }
        val pdfFile = File(runDir, "scan_report.pdf")
        val temporaryPdf = temporarySibling(pdfFile)
        try {
            temporaryPdf.outputStream().use { pdfDocument.writeTo(it) }
            promote(temporaryPdf, pdfFile)
        } finally {
            pdfDocument.close()
            temporaryPdf.delete()
        }
        ExportedFile(pdfFile.name, "application/pdf", pdfFile.absolutePath)
    }

    override suspend fun exportBundle(runId: RunId, diagnostics: DiagnosticsSnapshot?): ExportedFile = withContext(ioDispatcher) {
        val run = runRepository.runById(runId) ?: error("Run not found for export: ${runId.value}")
        val decisions = runRepository.decisionsForRun(runId)
        val events = runRepository.eventsForRun(runId)
        val summary = RunSummaryReducer.reduce(runId, decisions)
        val layout = fieldLayoutRepository.layoutById(run.fieldLayoutId)
        val runDir = File(exportsRoot(), LocalExportPathNames.runDirectoryName(runId)).apply { mkdirs() }
        atomicWrite(File(runDir, "decisions.jsonl"), ExportSerializer.toJsonl(decisions))
        atomicWrite(File(runDir, "events.csv"), ExportSerializer.toEventsCsv(decisions))
        val eventLog = ExportSerializer.toRunEventsJsonl(events)
        atomicWrite(File(runDir, "run_events.jsonl"), eventLog)
        atomicWrite(File(runDir, "app_log.jsonl"), eventLog)
        atomicWrite(File(runDir, "metadata.json"), ExportSerializer.toMetadataJson(run, layout))
        atomicWrite(File(runDir, "summary.json"), ExportSerializer.toSummaryJson(summary, run))
        atomicWrite(File(runDir, "latest_run.json"), ExportSerializer.toLatestRunJson(run, summary))
        atomicWrite(File(runDir, "field_layout.json"),
            layout?.let { ExportSerializer.toFieldLayoutJson(it) } ?: "{}\n",
        )
        atomicWrite(File(runDir, "diagnostics.json"), ExportSerializer.toDiagnosticsJson(diagnostics))
        File(runDir, "evidence_frames").mkdirs()
        val zipFile = File(exportsRoot(), "${runDir.name}.zip")
        val temporaryZip = temporarySibling(zipFile)
        try {
            zipDirectory(runDir, temporaryZip)
            promote(temporaryZip, zipFile)
        } finally {
            temporaryZip.delete()
        }
        ExportedFile(zipFile.name, "application/zip", zipFile.absolutePath)
    }

    override suspend fun deleteRunArtifacts(runId: RunId) = withContext(ioDispatcher) {
        val root = exportsRoot()
        val runDir = File(root, LocalExportPathNames.runDirectoryName(runId))
        val zipFile = File(root, "${runDir.name}.zip")
        // Both paths are derived from the same sanitized run id and are kept
        // below files/exports. Never recursively delete an arbitrary caller path.
        if (runDir.exists() && !runDir.toPath().normalize().startsWith(root.toPath().normalize())) {
            throw IOException("Refusing to delete export path outside exports root")
        }
        if (zipFile.exists() && !zipFile.toPath().normalize().startsWith(root.toPath().normalize())) {
            throw IOException("Refusing to delete ZIP path outside exports root")
        }
        if (runDir.exists() && !runDir.deleteRecursively()) {
            throw IOException("Unable to delete run export directory: ${runDir.absolutePath}")
        }
        if (zipFile.exists() && !zipFile.delete()) {
            throw IOException("Unable to delete run export ZIP: ${zipFile.absolutePath}")
        }
    }

    private fun exportFile(runId: RunId, name: String, content: String): File {
        val runDir = File(exportsRoot(), LocalExportPathNames.runDirectoryName(runId)).apply { mkdirs() }
        return File(runDir, name).also { atomicWrite(it, content) }
    }

    private fun exportsRoot(): File =
        File(context.filesDir, "exports").apply { mkdirs() }

    private fun zipDirectory(sourceDir: File, zipFile: File) {
        ZipOutputStream(zipFile.outputStream().buffered()).use { zip ->
            sourceDir.walkTopDown()
                .filter { file -> file.isFile }
                .forEach { file ->
                    val relativePath = sourceDir.toPath().relativize(file.toPath()).toString().replace('\\', '/')
                    zip.putNextEntry(ZipEntry(relativePath))
                    file.inputStream().use { input -> input.copyTo(zip) }
                    zip.closeEntry()
                }
        }
    }

    private fun atomicWrite(target: File, content: String) {
        target.parentFile?.mkdirs()
        val temporary = temporarySibling(target)
        try {
            temporary.outputStream().use { output ->
                output.write(content.toByteArray(Charsets.UTF_8))
                output.flush()
                output.fd.sync()
            }
            promote(temporary, target)
        } finally {
            temporary.delete()
        }
    }

    private fun temporarySibling(target: File): File =
        File(target.parentFile, ".${target.name}.${System.nanoTime()}.tmp")

    private fun promote(temporary: File, target: File) {
        try {
            Files.move(
                temporary.toPath(),
                target.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.ATOMIC_MOVE,
            )
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(
                temporary.toPath(),
                target.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
            )
        } catch (_: UnsupportedOperationException) {
            Files.move(
                temporary.toPath(),
                target.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
            )
        }
    }

}
