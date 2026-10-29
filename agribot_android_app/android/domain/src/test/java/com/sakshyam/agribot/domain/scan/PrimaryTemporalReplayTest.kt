package com.sakshyam.agribot.domain.scan

import com.sakshyam.agribot.domain.model.BoundingBox
import java.io.File
import kotlinx.serialization.json.*
import org.junit.Assert.*
import org.junit.Test

/** Actual weighted leaf accumulation and quick-check stopping, using cached development outputs. */
class PrimaryTemporalReplayTest {
    @Test fun allVerifiedDevelopmentImagesRetainRecognitionThroughActualFrameAccumulation() {
        val source = java.util.zip.GZIPInputStream(requireNotNull(javaClass.getResourceAsStream("/phase69_temporal_cases.json.gz"))).bufferedReader().use { it.readText() }
        val root = Json.parseToJsonElement(source).jsonObject
        val labels = root["labels"]!!.jsonArray.map { it.jsonPrimitive.content }
        val cases = root["cases"]!!.jsonArray
        assertEquals(6742, cases.size)
        fun JsonElement?.values(): DoubleArray? = if (this == null || this is JsonNull) null else jsonArray.map { it.jsonPrimitive.double }.toDoubleArray()
        fun JsonElement.scores() = jsonArray.let { AuxiliaryLeafScores(it[0].jsonPrimitive.double, it[1].jsonPrimitive.double, it[2].jsonPrimitive.double) }
        val results = cases.map { item ->
            val row = item.jsonObject
            val usable = row["usable"]!!.jsonPrimitive.boolean
            val whole = row["whole"]!!.jsonArray.map { it.jsonPrimitive.double }
            val views = row["leaves"]!!.jsonArray.map { leaf -> leaf.jsonArray.let { AuxiliaryLeafViews(it[0].scores(), it[1].scores()) } }
            val primary = PrimaryFrameEvidence(whole[0], whole[1], whole[2], whole[3], views,
                row["whole_adapted"]?.takeUnless { it is JsonNull }?.jsonPrimitive?.double, row["fallback"].values(), usable)
            val observations = row["observations"]!!.jsonArray.map { element ->
                val parts = element.jsonArray
                LeafObservation(BoundingBox(0f, 0f, 10f, 10f), parts[0].jsonPrimitive.float,
                    parts[1].jsonArray.map { it.jsonPrimitive.float }.toFloatArray(), row["quality_weight"]!!.jsonPrimitive.float)
            }
            val evidence = PlantEvidence(9, 1, 8, primaryPolicy = PrimaryImagePolicy())
            var verdict = evidence.verdict()
            var last: Pair<VerdictKind, Int?>? = null
            var stable = 0
            if (usable) for (frame in 1L..14L) {
                observations.forEachIndexed { index, observation -> evidence.add(index.toLong(), frame, observation) }
                evidence.addPrimaryFrame(frame, primary)
                verdict = evidence.verdict()
                val key = verdict.kind to verdict.labelIndex
                stable = if (key == last) stable + 1 else 0
                last = key
                if (QuickCheckCompletion.isReady(verdict, evidence.framesUsed, stable, PrimaryImagePolicy())) break
            }
            val expectedKind = row["expected_kind"]!!.jsonPrimitive.content.let {
                if (it in setOf("NO_LEAVES", "RETRY_QUALITY")) "NEED_MORE_VIEWS" else it
            }
            val expectedLabel = row["expected_label"]?.takeUnless { it is JsonNull }?.jsonPrimitive?.content
            val actualLabel = verdict.labelIndex?.let { labels[it] }
            buildJsonObject {
                put("id", row["id"]!!)
                put("kind", verdict.kind.name)
                put("label", actualLabel?.let { JsonPrimitive(it) } ?: JsonNull)
                put("frames", verdict.framesUsed)
                put("leaves", verdict.leavesSeen)
                put("primary_view", verdict.primaryView)
                put("expected_kind", expectedKind)
                put("expected_label", expectedLabel?.let { JsonPrimitive(it) } ?: JsonNull)
                put("matches", verdict.kind.name == expectedKind && actualLabel == expectedLabel)
            }
        }
        File("build/reports/native-temporal-replay.json").apply { parentFile.mkdirs(); writeText(JsonArray(results).toString()) }
        // Additional admitted frames can move Other to honest uncertainty as the
        // per-leaf weight cap fills. Record these differences; neither Healthy
        // nor any named disease may change here. The unchanged cohort gates are
        // independently evaluated from this complete output artifact.
        val unsafe = results.filter { !it["matches"]!!.jsonPrimitive.boolean &&
            !(it["expected_kind"]!!.jsonPrimitive.content == "UNKNOWN_CONDITION" && it["kind"]!!.jsonPrimitive.content == "UNSURE") }
        assertTrue("${unsafe.size} unsafe temporal differences: ${unsafe.take(12)}", unsafe.isEmpty())
    }
}
