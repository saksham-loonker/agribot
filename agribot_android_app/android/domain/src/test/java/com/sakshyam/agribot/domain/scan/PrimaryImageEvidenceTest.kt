package com.sakshyam.agribot.domain.scan

import com.sakshyam.agribot.domain.model.BoundingBox
import kotlinx.serialization.json.*
import org.junit.Assert.*
import org.junit.Test

class PrimaryImageEvidenceTest {
    private val policy = PrimaryImagePolicy()
    private fun probabilities(healthy: Double = .1, other: Double = .8) = DoubleArray(9).also {
        it[1] = healthy; it[8] = other; it[0] = 1 - healthy - other
    }
    private fun frame(positive: Double = .98) = PrimaryFrameEvidence(positive, .9, .99, .99, emptyList(), originalWholeProbabilities = probabilities())
    private fun evidence() = PlantEvidence(9, 1, 8, primaryPolicy = policy)
    private fun observation(index: Int) = LeafObservation(BoundingBox(0f, 0f, 10f, 10f), 1f, FloatArray(9).also { it[index] = 1f }, 1f)

    @Test fun primaryRecoveryNeedsSevenDistinctAdmittedFramesAndKeepsZeroLeaves() {
        val evidence = evidence()
        for (index in 1L..6L) {
            evidence.addPrimaryFrame(index, frame())
            assertNotEquals(VerdictKind.HEALTHY, evidence.verdict().kind)
        }
        repeat(100) { evidence.addPrimaryFrame(6L, frame()) }
        assertNotEquals(VerdictKind.HEALTHY, evidence.verdict().kind)
        evidence.addPrimaryFrame(7L, frame())
        val verdict = evidence.verdict()
        assertEquals(VerdictKind.HEALTHY, verdict.kind)
        assertTrue(verdict.primaryView)
        assertEquals(7, verdict.framesUsed)
        assertEquals(0, verdict.leavesSeen)
        assertEquals(0, verdict.leavesAgreeing)
    }

    @Test fun contraryViewBreaksTheWholeRecoveryWindow() {
        val evidence = evidence()
        for (index in 1L..6L) evidence.addPrimaryFrame(index, frame())
        evidence.addPrimaryFrame(7L, frame(.2))
        evidence.addPrimaryFrame(8L, frame())
        assertNotEquals(VerdictKind.HEALTHY, evidence.verdict().kind)
        for (index in 9L..14L) evidence.addPrimaryFrame(index, frame())
        assertEquals(VerdictKind.HEALTHY, evidence.verdict().kind)
        assertEquals(14, evidence.verdict().framesUsed)
    }

    @Test fun imageRecoveryNeverDisplacesAnOriginalDisease() {
        val evidence = evidence()
        for (index in 1L..7L) {
            evidence.add(1L, index, observation(0))
            evidence.addPrimaryFrame(index, frame())
        }
        assertEquals(VerdictKind.DISEASE, evidence.verdict().kind)
        assertEquals(0, evidence.verdict().labelIndex)
        assertFalse(evidence.verdict().primaryView)
    }

    @Test fun duplicateLeafCallbacksDoNotInventObservations() {
        val evidence = evidence()
        repeat(20) { evidence.add(1L, 1L, observation(1)) }
        assertEquals(1, evidence.observations)
        assertEquals(VerdictKind.NEED_MORE_VIEWS, evidence.verdict().kind)
    }

    @Test fun malformedAuxiliaryProbabilitiesFailClosed() {
        assertThrows(IllegalArgumentException::class.java) { AuxiliaryLeafScores(.9, Double.NaN, .8) }
        assertThrows(IllegalArgumentException::class.java) { frame(Double.POSITIVE_INFINITY) }
        assertThrows(IllegalArgumentException::class.java) { PrimaryImagePolicy(minimumFrames = 0) }
    }

    @Test fun allFrozenMobilePolicyCasesMatchWithoutPublisherOrFilenameInputs() {
        val text = java.util.zip.GZIPInputStream(requireNotNull(javaClass.getResourceAsStream("/phase69_policy_cases.json.gz"))).bufferedReader().use { it.readText() }
        val root = Json.parseToJsonElement(text).jsonObject
        assertEquals(0, root["observation_alias_kind_differences"]!!.jsonPrimitive.int)
        val cases = root["cases"]!!.jsonArray
        assertEquals(6742, cases.size)
        fun JsonElement?.values(): DoubleArray? = if (this == null || this is JsonNull) null else jsonArray.map { it.jsonPrimitive.double }.toDoubleArray()
        fun JsonElement.scores(): AuxiliaryLeafScores = jsonArray.let { AuxiliaryLeafScores(it[0].jsonPrimitive.double, it[1].jsonPrimitive.double, it[2].jsonPrimitive.double) }
        for (element in cases) {
            val row = element.jsonObject
            val rawKind = row["baseline_kind"]!!.jsonPrimitive.content
            val kind = when (rawKind) { "NO_LEAVES", "RETRY_QUALITY" -> VerdictKind.NEED_MORE_VIEWS; else -> VerdictKind.valueOf(rawKind) }
            val label = row["baseline_label"]?.takeUnless { it is JsonNull }?.jsonPrimitive?.int
            val baseline = PlantVerdict(kind, label, 0f, null, 0f, 0, 0, 0, false)
            val whole = row["whole"]!!.jsonArray.map { it.jsonPrimitive.double }
            val leaves = row["leaves"]!!.jsonArray.map { leaf -> leaf.jsonArray.let { AuxiliaryLeafViews(it[0].scores(), it[1].scores()) } }
            val packet = PrimaryFrameEvidence(whole[0], whole[1], whole[2], whole[3], leaves,
                row["whole_adapted"]?.takeUnless { it is JsonNull }?.jsonPrimitive?.double, row["fallback"].values(), row["usable"]!!.jsonPrimitive.boolean)
            val actual = PrimaryImageRouting.action(baseline, row["posterior"].values(), packet, policy, 1, 8)
            assertEquals(row["id"]!!.jsonPrimitive.content, row["expected_action"]!!.jsonPrimitive.content, actual.name)
        }
    }
}
