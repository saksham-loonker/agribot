package com.sakshyam.agribot.domain.scan

/** What the app tells the farmer about one plant. */
enum class VerdictKind {
    /** Not enough leaf evidence yet. */
    NEED_MORE_VIEWS,
    HEALTHY,
    DISEASE,
    /** Leaves look unwell but not like any condition the model knows. */
    UNKNOWN_CONDITION,
    /** Evidence is split; ask the farmer to look again / move closer. */
    UNSURE,
}

data class PlantVerdict(
    val kind: VerdictKind,
    /** Class index of the reported condition (healthy/disease/other), or null when not decided. */
    val labelIndex: Int?,
    /** Mean calibrated probability of the reported class over the leaves that support it. */
    val confidence: Float,
    val runnerUpIndex: Int?,
    val runnerUpConfidence: Float,
    /** Distinct leaves (tracks) that contributed. */
    val leavesSeen: Int,
    /** Distinct leaves whose own best class equals [labelIndex]. */
    val leavesAgreeing: Int,
    val framesUsed: Int,
    /** True when a disease was found on some leaves while most leaves look healthy. */
    val partial: Boolean,
)

/** Thresholds for turning leaf evidence into a plant verdict. */
data class VerdictPolicy(
    val minFrames: Int = 3,
    val minObservations: Int = 5,
    /** Per-leaf weight cap, so one leaf seen in many frames can't dominate. */
    val maxLeafWeight: Double = 3.0,
    /** Reporting "Healthy" hides disease, so it needs a stronger case than a disease. */
    val healthyMinProb: Float = 0.60f,
    val healthyMinMargin: Float = 0.20f,
    val diseaseMinProb: Float = 0.50f,
    val diseaseMinMargin: Float = 0.15f,
    val otherMinProb: Float = 0.50f,
    /** A disease seen confidently on this many distinct leaves is reported even if most leaves are healthy. */
    val partialDiseaseLeaves: Int = 2,
    val partialDiseaseLeafProb: Float = 0.70f,
)

/**
 * Accumulates leaf observations for ONE plant across frames and leaves (composite multi-leaf,
 * multi-frame vote). Each leaf track contributes its weighted mean probability vector with a capped
 * weight; the plant posterior is the weight-averaged mean over leaves.
 */
class PlantEvidence(
    private val numClasses: Int,
    private val healthyIndex: Int,
    private val otherIndex: Int,
) {
    private class LeafAcc(n: Int) {
        val sumP = DoubleArray(n)
        var sumW = 0.0
        var hits = 0
    }

    private val leaves = LinkedHashMap<Long, LeafAcc>()
    private val frames = HashSet<Long>()
    var observations = 0
        private set

    fun add(trackId: Long, frameIndex: Long, obs: LeafObservation) {
        require(obs.probabilities.size == numClasses)
        val w = (obs.detectorScore * obs.frameQuality).toDouble().coerceIn(0.05, 1.0)
        val acc = leaves.getOrPut(trackId) { LeafAcc(numClasses) }
        for (k in 0 until numClasses) acc.sumP[k] += w * obs.probabilities[k]
        acc.sumW += w
        acc.hits++
        frames += frameIndex
        observations++
    }

    fun clear() {
        leaves.clear(); frames.clear(); observations = 0
    }

    val framesUsed: Int get() = frames.size
    val leavesSeen: Int get() = leaves.size

    fun verdict(policy: VerdictPolicy = VerdictPolicy()): PlantVerdict {
        if (leaves.isEmpty()) return empty()
        val means = leaves.values.map { a -> DoubleArray(numClasses) { a.sumP[it] / a.sumW } }
        val weights = leaves.values.map { minOf(it.sumW, policy.maxLeafWeight) }
        val total = weights.sum()
        val plant = DoubleArray(numClasses)
        for ((m, w) in means.zip(weights)) for (k in 0 until numClasses) plant[k] += m[k] * w / total
        val order = plant.indices.sortedByDescending { plant[it] }
        val top = order[0]
        val second = order[1]
        val p1 = plant[top].toFloat()
        val p2 = plant[second].toFloat()
        fun agreeing(idx: Int) = means.count { m -> m.indices.maxBy { m[it] } == idx }
        val ready = frames.size >= policy.minFrames && observations >= policy.minObservations
        if (!ready) {
            return PlantVerdict(VerdictKind.NEED_MORE_VIEWS, null, p1, top, p1, leaves.size, 0, frames.size, false)
        }
        // Disease present on several distinct leaves while the plant average says something else.
        val accs = leaves.values.toList()
        val partialCandidates = (0 until numClasses).filter { it != healthyIndex && it != otherIndex }.mapNotNull { d ->
            val strong = means.indices.filter { i -> means[i][d] >= policy.partialDiseaseLeafProb && accs[i].hits >= 2 }
            if (strong.size >= policy.partialDiseaseLeaves) Triple(d, strong.size, strong.map { means[it][d] }.average()) else null
        }
        // Override only healthy/unknown/undecided plants; a confident disease verdict stands on its own.
        val bestPartial = partialCandidates.maxWithOrNull(compareBy<Triple<Int, Int, Double>> { it.second }.thenBy { it.third })
        if (bestPartial != null && top != bestPartial.first && (top == healthyIndex || top == otherIndex || p1 - p2 < policy.diseaseMinMargin)) {
            return PlantVerdict(
                VerdictKind.DISEASE, bestPartial.first, bestPartial.third.toFloat(), top, p1,
                leaves.size, bestPartial.second, frames.size, partial = true,
            )
        }
        val margin = p1 - p2
        val kind = when (top) {
            otherIndex -> if (p1 >= policy.otherMinProb) VerdictKind.UNKNOWN_CONDITION else VerdictKind.UNSURE
            healthyIndex -> if (p1 >= policy.healthyMinProb && margin >= policy.healthyMinMargin) VerdictKind.HEALTHY else VerdictKind.UNSURE
            else -> if (p1 >= policy.diseaseMinProb && margin >= policy.diseaseMinMargin) VerdictKind.DISEASE else VerdictKind.UNSURE
        }
        return PlantVerdict(kind, top, p1, second, p2, leaves.size, agreeing(top), frames.size, partial = false)
    }

    private fun empty() = PlantVerdict(VerdictKind.NEED_MORE_VIEWS, null, 0f, null, 0f, 0, 0, frames.size, false)
}
