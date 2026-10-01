package com.sakshyam.agribot.ml.vision

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.double
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Parsed `model_manifest.json` (schema 2). Every runtime constant (sizes, thresholds, calibration)
 * comes from here; nothing about the model contract is hard-coded in the app.
 */
data class VisionManifest(
    val bundleId: String,
    val members: List<Member>,
    val labels: List<String>,
    val otherIndex: Int,
    val otherLogitBias: Double,
    val temperature: Double,
    val energyRejectBelow: Double?,
    val detector: Detector,
) {
    data class Member(val file: String, val sha256: String, val inputSize: Int)
    data class Detector(
        val file: String,
        val sha256: String,
        val inputSize: Int,
        val candidates: Int,
        val scoreThreshold: Double,
        val nmsIou: Double,
        val maxDetections: Int,
    )

    val healthyIndex: Int get() = labels.indexOf("Healthy")

    companion object {
        fun parse(text: String): VisionManifest {
            val root = Json.parseToJsonElement(text).jsonObject
            require(root.int("schema") == 2) { "unsupported model manifest schema" }
            val c = root.obj("classifier")
            val labels = c["labels"]!!.jsonArray.map { it.jsonPrimitive.content }
            val members = c["members"]!!.jsonArray.map { m ->
                val o = m.jsonObject
                val shape = o["input"]!!.jsonArray.map { it.jsonPrimitive.int }
                require(shape.size == 4 && shape[0] == 1 && shape[3] == 3 && shape[1] == shape[2]) { "classifier input must be [1,S,S,3]" }
                Member(o.str("file"), o.str("sha256"), shape[1])
            }
            require(members.isNotEmpty()) { "no classifier members" }
            val d = root.obj("detector")
            val dIn = d["input"]!!.jsonArray.map { it.jsonPrimitive.int }
            val dOut = d["output"]!!.jsonArray.map { it.jsonPrimitive.int }
            require(dIn.size == 4 && dIn[1] == dIn[2] && dIn[3] == 3) { "detector input must be [1,S,S,3]" }
            require(dOut.size == 3 && dOut[1] == 5) { "detector output must be [1,5,N]" }
            require(d["output_rows"]!!.jsonArray.map { it.jsonPrimitive.content } == listOf("x1", "y1", "x2", "y2", "score")) { "unexpected detector rows" }
            val otherIndex = c.int("other_index")
            require(otherIndex in labels.indices && "Healthy" in labels) { "labels must contain Healthy and the Other index" }
            val energyGate = (c["energy_gate"] as? JsonPrimitive)?.content == "true"
            return VisionManifest(
                bundleId = root.str("bundle_id"),
                members = members,
                labels = labels,
                otherIndex = otherIndex,
                otherLogitBias = c["other_logit_bias"]!!.jsonPrimitive.double,
                temperature = c["temperature"]!!.jsonPrimitive.double,
                energyRejectBelow = if (energyGate) (c["energy_reject_below"] as? JsonPrimitive)?.doubleOrNull else null,
                detector = Detector(
                    file = d.str("file"), sha256 = d.str("sha256"), inputSize = dIn[1], candidates = dOut[2],
                    scoreThreshold = d["score_threshold"]!!.jsonPrimitive.double,
                    nmsIou = d["nms_iou"]!!.jsonPrimitive.double,
                    maxDetections = d.int("max_detections"),
                ),
            )
        }

        private fun JsonObject.obj(k: String) = requireNotNull(this[k]) { "missing $k" }.jsonObject
        private fun JsonObject.str(k: String) = requireNotNull(this[k]) { "missing $k" }.jsonPrimitive.content
        private fun JsonObject.int(k: String) = requireNotNull(this[k]) { "missing $k" }.jsonPrimitive.int
    }
}
