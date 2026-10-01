package com.sakshyam.agribot.ml.vision

import java.io.File
import kotlin.math.abs
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Golden parity against the Python reference (golden/canon_reference.py) used to evaluate the shipped
 * .tflite files: Kotlin preprocessing must produce the same tensors, and the Kotlin decision maths must
 * turn the reference logits into the same probabilities and label.
 */
class GoldenParityTest {
    private val dir = File(javaClass.classLoader!!.getResource("golden/golden_expected.json")!!.toURI()).parentFile
    private val expected = Json.parseToJsonElement(File(dir, "golden_expected.json").readText()).jsonObject
    private val entries = expected["entries"]!!.jsonArray.map { it.jsonObject }
    private val manifest = VisionManifest.parse(File("../app/src/main/assets/model_manifest.json").readText())

    private fun frame(name: String): RgbFrame = TestPng.read(File(dir, name))

    private fun checkTensor(values: FloatArray, checks: JsonObject, tol: Double, what: String) {
        assertEquals("$what mean", checks["mean"]!!.jsonPrimitive.double, values.average(), tol)
        for ((idx, v) in checks["samples"]!!.jsonObject) {
            assertEquals("$what sample $idx", v.jsonPrimitive.double, values[idx.toInt()].toDouble(), tol)
        }
    }

    @Test fun `golden set is present and matches the bundled manifest`() {
        assertTrue(entries.size >= 10)
        assertEquals(manifest.labels.size, 9)
        assertEquals("Other", manifest.labels[manifest.otherIndex])
    }

    @Test fun `classifier preprocessing matches the reference for every golden image`() {
        for (e in entries) {
            val f = frame(e["file"]!!.jsonPrimitive.content)
            assertEquals(e["width"]!!.jsonPrimitive.int, f.width)
            val checks = e["classifier_input_checks"]!!.jsonArray
            manifest.members.forEachIndexed { k, m ->
                val t = CanonicalImageOps.classifierInput(f, m.inputSize)!!
                checkTensor(t, checks[k].jsonObject, 1e-3, "${e["file"]} member $k")
            }
        }
    }

    @Test fun `detector letterbox and input match the reference`() {
        for (e in entries) {
            val f = frame(e["file"]!!.jsonPrimitive.content)
            val (t, lb) = CanonicalImageOps.detectorInput(f, manifest.detector.inputSize)
            val c = e["detector_input_checks"]!!.jsonObject
            val ref = c["letterbox"]!!.jsonArray
            assertEquals(ref[0].jsonPrimitive.double, lb.scale, 1e-9)
            assertEquals(ref[1].jsonPrimitive.int, lb.padX)
            assertEquals(ref[2].jsonPrimitive.int, lb.padY)
            assertEquals(c["mean"]!!.jsonPrimitive.double, t.average(), 1e-4)
        }
    }

    @Test fun `decision maths reproduces reference probabilities and labels`() {
        val d = EnsembleDecision(manifest.labels.size, manifest.otherIndex, manifest.otherLogitBias, manifest.temperature, manifest.energyRejectBelow)
        for (e in entries) {
            for (key in listOf("whole_image", "top_crop")) {
                val o = (e[key] as? JsonObject) ?: continue
                val logits = o["member_logits"]!!.jsonArray.map { z -> FloatArray(z.jsonArray.size) { z.jsonArray[it].jsonPrimitive.float } }
                val r = d.combine(logits)
                val p = o["probs"]!!.jsonArray.map { it.jsonPrimitive.double }
                for (k in p.indices) assertEquals("${e["file"]} $key p[$k]", p[k], r.probabilities[k].toDouble(), 1e-5)
                assertEquals("${e["file"]} $key label", o["label"]!!.jsonPrimitive.content, manifest.labels[r.labelIndex])
            }
        }
    }

    @Test fun `crop rectangle and decoder agree with the reference on golden detections`() {
        // Re-decode is covered on-device; here check that every reference top box yields a valid crop.
        for (e in entries) {
            val top = (e["top_crop"] as? JsonObject) ?: continue
            val b = top["box"]!!.jsonArray.map { it.jsonPrimitive.float }
            val f = frame(e["file"]!!.jsonPrimitive.content)
            val r = CanonicalImageOps.cropRect(f.width, f.height, PixelBox(b[0], b[1], b[2], b[3]), 0.10)
            assertTrue(r[2] > r[0] && r[3] > r[1])
            assertTrue(abs(r[0] - (b[0] - 0.1f * (b[2] - b[0]))) <= 1.0f || r[0] == 0)
        }
    }
}
