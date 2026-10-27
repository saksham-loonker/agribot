package com.sakshyam.agribot.ml.vision

import org.junit.Assert.*
import org.junit.Test
import java.io.File
import kotlinx.serialization.json.*

class ViewConfirmationTest {
    private val guard = ViewConfirmation(EnsembleDecision(4, 3, 0.0, 1.0, null), 1, 3)
    private fun confident(label: Int) = FloatArray(4).apply { this[label] = 10f }

    @Test fun `paired model output requires the new manifest contract`() {
        val root = Json.parseToJsonElement(File("../app/src/main/assets/model_manifest.json").readText()).jsonObject.toMutableMap()
        root.remove("primary_recovery")
        val classifier = root["classifier"]!!.jsonObject.toMutableMap()
        classifier.remove("reference_disease_priority")
        val member = classifier["members"]!!.jsonArray[0].jsonObject.toMutableMap()
        member["output"] = JsonArray(listOf(JsonPrimitive(1), JsonPrimitive(18)))
        classifier["members"] = JsonArray(listOf(JsonObject(member)))
        classifier["view_confirmation"] = JsonObject(mapOf(
            "policy" to JsonPrimitive("changed_positive_full_roi"),
            "reference_logit_offset" to JsonPrimitive(9),
            "padding_value" to JsonPrimitive(114),
        ))
        root["schema"] = JsonPrimitive(3)
        root["classifier"] = JsonObject(classifier)
        assertTrue(VisionManifest.parse(JsonObject(root).toString()).fullRoiConfirmation)
        root["schema"] = JsonPrimitive(2)
        assertThrows(IllegalArgumentException::class.java) { VisionManifest.parse(JsonObject(root).toString()) }
        root["schema"] = JsonPrimitive(3)
        member["output"] = JsonArray(listOf(JsonPrimitive(1), JsonPrimitive(9)))
        classifier["members"] = JsonArray(listOf(JsonObject(member)))
        root["classifier"] = JsonObject(classifier)
        assertThrows(IllegalArgumentException::class.java) { VisionManifest.parse(JsonObject(root).toString()) }
    }

    @Test fun `a new Healthy call requires the full leaf to agree`() {
        val healthy = confident(1)
        assertTrue(guard.needsConfirmation(confident(3), healthy))
        assertFalse(guard.confirms(healthy, confident(3)))
        assertTrue(guard.confirms(healthy, healthy))
    }

    @Test fun `existing Healthy and disease decisions stay on the original path`() {
        for (label in listOf(0, 1, 2)) {
            assertFalse(guard.needsConfirmation(confident(label), confident(label)))
        }
        assertFalse(guard.needsConfirmation(confident(1), FloatArray(4)))
    }

    @Test fun `a different disease cannot confirm a proposed diagnosis`() {
        assertTrue(guard.needsConfirmation(confident(3), confident(0)))
        assertFalse(guard.confirms(confident(0), confident(2)))
    }

    @Test fun `an unrecognized correction also needs agreement when configured`() {
        val allDecisions = ViewConfirmation(EnsembleDecision(4, 3, 0.0, 1.0, null), 1, 3, true)
        assertFalse(guard.needsConfirmation(confident(1), confident(3)))
        assertTrue(allDecisions.needsConfirmation(confident(1), confident(3)))
        assertFalse(allDecisions.confirms(confident(3), confident(1)))
        assertTrue(allDecisions.confirms(confident(3), confident(3)))
        assertFalse(allDecisions.needsConfirmation(confident(3), confident(3)))
    }

    @Test fun `the second view preserves tips cropped from a wide leaf`() {
        val frame = RgbFrame(8, 4, IntArray(32) { i ->
            (0xFF shl 24) or (if (i % 8 < 2 || i % 8 >= 6) 0xFF0000 else 0x0000FF)
        })
        val centered = CanonicalImageOps.classifierInput(frame, 8)!!
        val full = FloatArray(8 * 8 * 3)
        assertTrue(CanonicalImageOps.classifierLetterboxInputInto(frame, 8, full))
        fun red(tensor: FloatArray, pixel: Int) = (tensor[pixel * 3] * CanonicalImageOps.IMAGENET_STD[0] + CanonicalImageOps.IMAGENET_MEAN[0]) * 255f
        assertEquals(0f, red(centered, 16), 0.0001f)
        assertEquals(255f, red(full, 16), 0.0001f)
        assertEquals(114f, red(full, 0), 0.0001f)
    }

    @Test fun `full ROI downsampling uses canonical area averages`() {
        val frame = RgbFrame(6, 3, IntArray(18) { i -> (0xFF shl 24) or ((i % 6 * 30) shl 16) })
        val output = FloatArray(4 * 4 * 3)
        assertTrue(CanonicalImageOps.classifierLetterboxInputInto(frame, 4, output))
        val actual = FloatArray(4) { x -> (output[(4 + x) * 3] * CanonicalImageOps.IMAGENET_STD[0] + CanonicalImageOps.IMAGENET_MEAN[0]) * 255f }
        assertArrayEquals(floatArrayOf(10f, 50f, 100f, 140f), actual, 0.0001f)
    }

    @Test fun `schema four requires both reference outputs and disease priority`() {
        val root = Json.parseToJsonElement(File("../app/src/main/assets/model_manifest.json").readText()).jsonObject.toMutableMap()
        root.remove("primary_recovery")
        val classifier = root["classifier"]!!.jsonObject.toMutableMap()
        val member = classifier["members"]!!.jsonArray[0].jsonObject.toMutableMap()
        member["output"] = JsonArray(listOf(JsonPrimitive(1), JsonPrimitive(18)))
        classifier["members"] = JsonArray(listOf(JsonObject(member)))
        classifier["view_confirmation"] = JsonObject(mapOf(
            "policy" to JsonPrimitive("changed_positive_full_roi"),
            "reference_logit_offset" to JsonPrimitive(9),
            "padding_value" to JsonPrimitive(114),
        ))
        root["schema"] = JsonPrimitive(4)
        classifier.remove("reference_disease_priority")
        root["classifier"] = JsonObject(classifier)
        assertThrows(IllegalArgumentException::class.java) { VisionManifest.parse(JsonObject(root).toString()) }
        classifier["reference_disease_priority"] = JsonPrimitive(true)
        root["classifier"] = JsonObject(classifier)
        assertTrue(VisionManifest.parse(JsonObject(root).toString()).referenceDiseasePriority)
        root["schema"] = JsonPrimitive(3)
        assertThrows(IllegalArgumentException::class.java) { VisionManifest.parse(JsonObject(root).toString()) }
    }
}
