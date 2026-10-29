package com.sakshyam.agribot.domain.scan

import com.sakshyam.agribot.domain.model.BoundingBox
import java.io.File
import kotlinx.serialization.json.*

/** Headless production-domain replay for a frozen acceptance protocol. No fitted parameters. */
object RecognitionAcceptanceReplay {
    @JvmStatic fun main(args: Array<String>) {
        require(args.size == 2)
        val root = Json.parseToJsonElement(File(args[0]).readText()).jsonObject
        val output = root["cases"]!!.jsonArray.map { item ->
            val row = item.jsonObject
            buildJsonObject {
                put("id", row["id"]!!)
                put("baseline", replay(row["baseline"]!!.jsonObject, false))
                put("candidate", replay(row["candidate"]!!.jsonObject, true))
            }
        }
        File(args[1]).apply { parentFile.mkdirs(); writeText(buildJsonObject {
            put("protocol_sha256", root["protocol_sha256"]!!)
            put("cases", JsonArray(output))
        }.toString()) }
    }

    private fun replay(row: JsonObject, candidate: Boolean): JsonObject {
        val policy = if (candidate) PrimaryImagePolicy() else null
        val evidence = PlantEvidence(9, 1, 8, primaryPolicy = policy)
        val usable = row["usable"]!!.jsonPrimitive.boolean
        val quality = row["quality_weight"]!!.jsonPrimitive.float
        fun JsonElement?.values(): DoubleArray? = if (this == null || this is JsonNull) null else jsonArray.map { it.jsonPrimitive.double }.toDoubleArray()
        fun JsonElement.scores() = jsonArray.let { AuxiliaryLeafScores(it[0].jsonPrimitive.double, it[1].jsonPrimitive.double, it[2].jsonPrimitive.double) }
        val observations = row["observations"]!!.jsonArray.map { element ->
            val parts = element.jsonArray
            LeafObservation(BoundingBox(0f, 0f, 10f, 10f), parts[0].jsonPrimitive.float,
                parts[1].jsonArray.map { it.jsonPrimitive.float }.toFloatArray(), quality)
        }
        val primary = if (candidate) {
            val whole = row["whole"]!!.jsonArray.map { it.jsonPrimitive.double }
            val views = row["leaves"]!!.jsonArray.map { leaf -> leaf.jsonArray.let { AuxiliaryLeafViews(it[0].scores(), it[1].scores()) } }
            PrimaryFrameEvidence(whole[0], whole[1], whole[2], whole[3], views,
                row["whole_adapted"]?.takeUnless { it is JsonNull }?.jsonPrimitive?.double, row["fallback"].values(), usable)
        } else null
        var verdict = evidence.verdict()
        var last: Pair<VerdictKind, Int?>? = null
        var stable = 0
        if (usable) for (frame in 1L..14L) {
            observations.forEachIndexed { index, observation -> evidence.add(index.toLong(), frame, observation) }
            primary?.let { evidence.addPrimaryFrame(frame, it) }
            verdict = evidence.verdict()
            val key = verdict.kind to verdict.labelIndex
            stable = if (key == last) stable + 1 else 0
            last = key
            if (QuickCheckCompletion.isReady(verdict, evidence.framesUsed, stable, policy)) break
        }
        return buildJsonObject {
            put("kind", verdict.kind.name)
            put("label_index", verdict.labelIndex?.let { JsonPrimitive(it) } ?: JsonNull)
            put("confidence", verdict.confidence)
            put("frames", verdict.framesUsed)
            put("leaves", verdict.leavesSeen)
            put("leaves_agreeing", verdict.leavesAgreeing)
            put("partial", verdict.partial)
            put("primary_view", verdict.primaryView)
        }
    }
}
