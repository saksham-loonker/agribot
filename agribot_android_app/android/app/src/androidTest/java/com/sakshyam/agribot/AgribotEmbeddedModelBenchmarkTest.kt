package com.sakshyam.agribot

import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sakshyam.agribot.domain.model.AnalysisFrame
import com.sakshyam.agribot.ml.inference.TfliteInferenceRepository
import java.io.File
import java.time.Instant
import java.util.Locale
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AgribotEmbeddedModelBenchmarkTest {
    @Test
    fun writeEmbeddedModelBenchmarkReport() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val repository = TfliteInferenceRepository(context)
        val readiness = repository.observeModelReadiness().first()
        assertTrue(readiness.errors.joinToString(), readiness.isReady)

        val classifierFrame = syntheticLeafFrame(width = 224, height = 224)
        val detectorFrame = syntheticLeafFrame(width = 256, height = 256)

        repository.classifySideFrame(classifierFrame)
        repository.detectFrontFrame(detectorFrame)

        val classifierLatencies = measureIterations(CLASSIFIER_ITERATIONS) {
            repository.classifySideFrame(classifierFrame)
        }
        val detectorLatencies = measureIterations(DETECTOR_ITERATIONS) {
            repository.detectFrontFrame(detectorFrame)
        }

        val report = benchmarkJson(
            readinessStatus = readiness.statusText,
            classifierLatencies = classifierLatencies,
            detectorLatencies = detectorLatencies,
        )
        val output = File(File(context.filesDir, "benchmarks"), "embedded_model_benchmark.json")
        output.parentFile?.mkdirs()
        output.writeText(report, Charsets.UTF_8)
        assertTrue(output.isFile)
    }

    private suspend fun measureIterations(iterations: Int, block: suspend () -> Unit): List<Double> =
        (1..iterations).map {
            val start = System.nanoTime()
            block()
            (System.nanoTime() - start) / 1_000_000.0
        }

    private fun benchmarkJson(
        readinessStatus: String,
        classifierLatencies: List<Double>,
        detectorLatencies: List<Double>,
    ): String =
        """
        {
          "created_at": "${Instant.now()}",
          "package": "$PACKAGE_NAME",
          "model_bundle_id": "agribot-model-bundle-v001",
          "readiness_status": "${readinessStatus.escapeJson()}",
          "device": {
            "manufacturer": "${Build.MANUFACTURER.escapeJson()}",
            "model": "${Build.MODEL.escapeJson()}",
            "brand": "${Build.BRAND.escapeJson()}",
            "device": "${Build.DEVICE.escapeJson()}",
            "sdk_int": ${Build.VERSION.SDK_INT},
            "supported_abis": "${Build.SUPPORTED_ABIS.joinToString(",").escapeJson()}",
            "is_emulator": ${isProbablyEmulator()}
          },
          "classifier": ${latencyJson(CLASSIFIER_ITERATIONS, classifierLatencies)},
          "detector": ${latencyJson(DETECTOR_ITERATIONS, detectorLatencies)},
          "limits": {
            "emulator_results_are_smoke_only": true,
            "target_phone_classifier_avg_ms_required_max": 160,
            "target_phone_classifier_p95_ms_required_max": 200,
            "target_phone_detector_p95_ms_required_max": 300
          }
        }
        """.trimIndent()

    private fun latencyJson(iterations: Int, values: List<Double>): String =
        """
        {
          "iterations": $iterations,
          "avg_ms": ${values.average().jsonDouble()},
          "p95_ms": ${percentile(values, 0.95).jsonDouble()},
          "max_ms": ${values.maxOrNull()?.jsonDouble() ?: "0.000"}
        }
        """.trimIndent()

    private fun percentile(values: List<Double>, percentile: Double): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val index = ((sorted.size - 1) * percentile).toInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }

    private fun Double.jsonDouble(): String = String.format(Locale.US, "%.3f", this)

    private fun isProbablyEmulator(): Boolean {
        val fingerprint = Build.FINGERPRINT.lowercase(Locale.US)
        val model = Build.MODEL.lowercase(Locale.US)
        val manufacturer = Build.MANUFACTURER.lowercase(Locale.US)
        return fingerprint.contains("generic") ||
            fingerprint.contains("emulator") ||
            model.contains("sdk") ||
            model.contains("emulator") ||
            manufacturer.contains("genymotion")
    }

    private fun syntheticLeafFrame(width: Int, height: Int): AnalysisFrame {
        val pixels = IntArray(width * height) { index ->
            val x = index % width
            val y = index / width
            if (x in width / 4 until width * 3 / 4 && y in height / 4 until height * 3 / 4) {
                0xFF2FA84F.toInt()
            } else {
                0xFF3C3024.toInt()
            }
        }
        return AnalysisFrame(
            width = width,
            height = height,
            rotationDegrees = 0,
            timestampNanos = 1L,
            rgbPixels = pixels,
            jpegBytes = ByteArray(0),
        )
    }

    private fun String.escapeJson(): String =
        buildString {
            this@escapeJson.forEach { char ->
                when (char) {
                    '\\' -> append("\\\\")
                    '"' -> append("\\\"")
                    '\n' -> append("\\n")
                    '\r' -> append("\\r")
                    '\t' -> append("\\t")
                    else -> append(char)
                }
            }
        }

    private companion object {
        const val CLASSIFIER_ITERATIONS = 5
        const val DETECTOR_ITERATIONS = 3
        const val PACKAGE_NAME = "com.sakshyam.agribot"
    }
}
