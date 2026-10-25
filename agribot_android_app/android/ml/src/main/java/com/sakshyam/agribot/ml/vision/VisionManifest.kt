package com.sakshyam.agribot.ml.vision

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.double
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Parsed `model_manifest.json` (schemas 2–4). Every runtime constant (sizes, thresholds, calibration)
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
    val fullRoiConfirmation: Boolean = false,
    val confirmUnknownChanges: Boolean = false,
    val referenceDiseasePriority: Boolean = false,
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
            val schema = root.int("schema")
            require(schema in 2..4) { "unsupported model manifest schema" }
            val c = root.obj("classifier")
            require(c.str("layout") == "NHWC") { "classifier must use NHWC" }
            for ((key, expected) in listOf("mean" to CanonicalImageOps.IMAGENET_MEAN, "std" to CanonicalImageOps.IMAGENET_STD)) {
                val values = c[key]!!.jsonArray.map { it.jsonPrimitive.double }
                require(values.size == 3 && values.indices.all { values[it].isFinite() && kotlin.math.abs(values[it] - expected[it]) < 1e-6 }) { "unsupported classifier normalization $key" }
            }
            val labels = c["labels"]!!.jsonArray.map { it.jsonPrimitive.content }
            require(labels.size >= 2 && labels.distinct().size == labels.size && labels.all { it.isNotBlank() }) { "labels must be distinct and nonempty" }
            val members = c["members"]!!.jsonArray.map { m ->
                val o = m.jsonObject
                val shape = o["input"]!!.jsonArray.map { it.jsonPrimitive.int }
                require(shape.size == 4 && shape[0] == 1 && shape[3] == 3 && shape[1] > 0 && shape[1] == shape[2]) { "classifier input must be [1,S,S,3]" }
                require(validModelFile(o.str("file")) && validSha(o.str("sha256"))) { "invalid classifier file/hash" }
                Member(o.str("file"), o.str("sha256"), shape[1])
            }
            require(members.isNotEmpty()) { "no classifier members" }
            val confirmation = c["view_confirmation"]?.jsonObject
            require((schema >= 3) == (confirmation != null)) { "paired models require full-ROI confirmation" }
            val referencePriority = c["reference_disease_priority"]?.jsonPrimitive?.booleanOrNull == true
            require((schema == 4) == referencePriority) { "schema 4 requires reference disease priority" }
            if (confirmation != null) {
                require(members.size == 1 && confirmation.str("policy") in setOf("changed_positive_full_roi", "changed_decision_full_roi") &&
                    confirmation.int("reference_logit_offset") == labels.size &&
                    confirmation["padding_value"]!!.jsonPrimitive.double == 114.0) { "unsupported view confirmation contract" }
                val output = c["members"]!!.jsonArray[0].jsonObject["output"]!!.jsonArray.map { it.jsonPrimitive.int }
                require(output == listOf(1, labels.size * 2)) { "classifier needs candidate and baseline logits" }
            }
            val d = root.obj("detector")
            require(d.str("layout") == "NHWC") { "detector must use NHWC" }
            val dIn = d["input"]!!.jsonArray.map { it.jsonPrimitive.int }
            val dOut = d["output"]!!.jsonArray.map { it.jsonPrimitive.int }
            require(dIn.size == 4 && dIn[0] == 1 && dIn[1] > 0 && dIn[1] == dIn[2] && dIn[3] == 3) { "detector input must be [1,S,S,3]" }
            require(dOut.size == 3 && dOut[0] == 1 && dOut[1] == 5 && dOut[2] > 0) { "detector output must be [1,5,N]" }
            require(validModelFile(d.str("file")) && validSha(d.str("sha256"))) { "invalid detector file/hash" }
            require(d["output_rows"]!!.jsonArray.map { it.jsonPrimitive.content } == listOf("x1", "y1", "x2", "y2", "score")) { "unexpected detector rows" }
            val otherIndex = c.int("other_index")
            require(otherIndex in labels.indices && labels[otherIndex] == "Other" && "Healthy" in labels) { "labels must contain Healthy and the Other index" }
            val energyGate = (c["energy_gate"] as? JsonPrimitive)?.content == "true"
            val temperature = c["temperature"]!!.jsonPrimitive.double
            val bias = c["other_logit_bias"]!!.jsonPrimitive.double
            val energy = (c["energy_reject_below"] as? JsonPrimitive)?.doubleOrNull
            require(temperature.isFinite() && temperature > 0 && bias.isFinite()) { "invalid classifier calibration" }
            require(!energyGate || (energy != null && energy.isFinite())) { "enabled energy gate needs a finite threshold" }
            require(d["score_threshold"]!!.jsonPrimitive.double in 0.0..1.0 && d["nms_iou"]!!.jsonPrimitive.double in 0.0..1.0 && d.int("max_detections") > 0) { "invalid detector thresholds" }
            return VisionManifest(
                bundleId = root.str("bundle_id"),
                members = members,
                labels = labels,
                otherIndex = otherIndex,
                otherLogitBias = bias,
                temperature = temperature,
                energyRejectBelow = if (energyGate) energy else null,
                detector = Detector(
                    file = d.str("file"), sha256 = d.str("sha256"), inputSize = dIn[1], candidates = dOut[2],
                    scoreThreshold = d["score_threshold"]!!.jsonPrimitive.double,
                    nmsIou = d["nms_iou"]!!.jsonPrimitive.double,
                    maxDetections = d.int("max_detections"),
                ),
                fullRoiConfirmation = confirmation != null,
                confirmUnknownChanges = confirmation?.str("policy") == "changed_decision_full_roi",
                referenceDiseasePriority = referencePriority,
            )
        }

        private fun validModelFile(file: String) = Regex("[A-Za-z0-9_-]+\\.tflite").matches(file)
        private fun validSha(sha: String) = Regex("[a-fA-F0-9]{64}").matches(sha)
        private fun JsonObject.obj(k: String) = requireNotNull(this[k]) { "missing $k" }.jsonObject
        private fun JsonObject.str(k: String) = requireNotNull(this[k]) { "missing $k" }.jsonPrimitive.content
        private fun JsonObject.int(k: String) = requireNotNull(this[k]) { "missing $k" }.jsonPrimitive.int
    }
}
