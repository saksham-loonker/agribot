package com.sakshyam.agribot

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sakshyam.agribot.domain.model.AGRIBOT_LABEL_ORDER
import com.sakshyam.agribot.domain.model.AnalysisFrame
import com.sakshyam.agribot.ml.inference.TfliteInferenceRepository
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AgribotEmbeddedModelIntegrationTest {
    @Test
    fun embeddedModelBundleLoadsAndRunsClassifierAndDetectorOnAndroid() {
        runBlocking {
            val repository = TfliteInferenceRepository(InstrumentationRegistry.getInstrumentation().targetContext)
            val readiness = repository.observeModelReadiness().first { it.progress >= 1f }
            assertTrue(readiness.errors.joinToString(), readiness.isReady)

            val classifierFrame = syntheticLeafFrame(width = 224, height = 224)
            val prediction = repository.classifySideFrame(classifierFrame)
            val expectedLabels = AGRIBOT_LABEL_ORDER.map { it.displayName }.toSet()

            assertTrue(prediction.rawLabel, prediction.rawLabel in expectedLabels)
            assertTrue("confidence=${prediction.confidence}", prediction.confidence in 0.0f..1.0f)
            assertFalse(prediction.modelVersion.isBlank())

            val detectorFrame = syntheticLeafFrame(width = 256, height = 256)
            val candidates = repository.detectFrontFrame(detectorFrame)
            assertNotNull(candidates)
            candidates.forEach { candidate ->
                assertTrue("detectorConfidence=${candidate.detectorConfidence}", candidate.detectorConfidence in 0.0f..1.0f)
                assertTrue(candidate.bboxPx.left <= candidate.bboxPx.right)
                assertTrue(candidate.bboxPx.top <= candidate.bboxPx.bottom)
            }
            // Exercise interpreter replacement between calls on the real runtime.
            for (threads in listOf(1, 2, 4)) {
                repository.setCpuThreads(threads)
                val updated = repository.classifySideFrame(classifierFrame)
                assertTrue(updated.rawLabel, updated.rawLabel in expectedLabels)
                assertTrue(updated.confidence in 0f..1f)
                assertNotNull(repository.detectFrontFrame(detectorFrame))
            }
        }
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
}
