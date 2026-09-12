package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.model.ScanConstants
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.float
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class BundledThresholdContractTest {
    @Test
    fun shippedAssetUsesRuntimeThresholds() {
        val root = generateSequence(Path.of("").toAbsolutePath()) { it.parent }
            .first { Files.exists(it.resolve("app/src/main/assets/model_manifest.json")) }
        val manifest = Json.parseToJsonElement(
            Files.readString(root.resolve("app/src/main/assets/model_manifest.json")),
        ).jsonObject
        val classifier = manifest.getValue("models").jsonObject.getValue("classifier").jsonObject
        assertEquals(ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD, classifier.getValue("confidence_threshold").jsonPrimitive.float)
        assertEquals(ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD, classifier.getValue("high_confidence_threshold").jsonPrimitive.float)
    }
}
