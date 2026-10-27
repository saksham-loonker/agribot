package com.sakshyam.agribot.domain.scan

/** Frozen phase-69 thresholds, supplied by the model manifest. */
data class PrimaryImagePolicy(
    val healthyMin: Double = .85,
    val negativeMax: Double = .001,
    val genusMin: Double = .5,
    val naturalMin: Double = .5,
    val compactMin: Double = .1,
    val closedOtherMax: Double = .1,
    val closedDiseaseMassMin: Double = .6,
    val strongLocalMax: Double = .2,
    val partialLocalMax: Double = .3,
    val partialLocalLeaves: Int = 2,
    val isolatedMin: Double = .95,
    val localCenterMin: Double = .5,
    val wholeHealthyMin: Double = .5,
    val wholeLocalLeaves: Int = 2,
    val minimumFrames: Int = 7,
    val maximumLeaves: Int = 6,
) {
    init {
        require(listOf(healthyMin, negativeMax, genusMin, naturalMin, compactMin, closedOtherMax,
            closedDiseaseMassMin, strongLocalMax, partialLocalMax, isolatedMin, localCenterMin,
            wholeHealthyMin).all { it.isFinite() && it in 0.0..1.0 })
        require(minimumFrames >= 7 && maximumLeaves > 0 && partialLocalLeaves >= 2 && wholeLocalLeaves >= 2)
    }
}

data class AuxiliaryLeafScores(val positive: Double, val negative: Double, val adapted: Double) {
    init { require(listOf(positive, negative, adapted).all { it.isFinite() && it in 0.0..1.0 }) }
}

data class AuxiliaryLeafViews(val center: AuxiliaryLeafScores, val full: AuxiliaryLeafScores)

/** Scores from one real, admitted primary image and its actual detected leaves. */
data class PrimaryFrameEvidence(
    val positive: Double,
    val negative: Double,
    val genus: Double,
    val natural: Double,
    val leaves: List<AuxiliaryLeafViews>,
    val wholeAdapted: Double? = null,
    val originalWholeProbabilities: DoubleArray? = null,
    val usable: Boolean = true,
) {
    init {
        require(listOf(positive, negative, genus, natural).all { it.isFinite() && it in 0.0..1.0 })
        require(wholeAdapted == null || (wholeAdapted.isFinite() && wholeAdapted in 0.0..1.0))
        require(originalWholeProbabilities == null || originalWholeProbabilities.all { it.isFinite() && it in 0.0..1.0 })
    }
}

enum class PrimaryAction { NONE, RECOVER_HEALTHY, ABSTAIN_HEALTHY }

/** Pure scalar policy. Publisher, filename and ground-truth labels are never inputs. */
object PrimaryImageRouting {
    fun action(
        baseline: PlantVerdict,
        posterior: DoubleArray?,
        evidence: PrimaryFrameEvidence,
        policy: PrimaryImagePolicy,
        healthyIndex: Int,
        otherIndex: Int,
    ): PrimaryAction {
        if (baseline.kind == VerdictKind.DISEASE || !evidence.usable) return PrimaryAction.NONE
        val positive = evidence.leaves.flatMap { listOf(it.center.positive, it.full.positive) }.minOrNull() ?: 1.0
        val negative = evidence.leaves.minOfOrNull { maxOf(it.center.negative, it.full.negative) } ?: 1.0
        val leafVeto = negative <= policy.negativeMax
        val scoped = evidence.natural >= policy.naturalMin && evidence.genus >= policy.genusMin
        if (baseline.kind == VerdictKind.HEALTHY && !(scoped || leafVeto)) return PrimaryAction.NONE
        val rejected = leafVeto || (scoped && evidence.negative <= policy.negativeMax)
        if (baseline.kind == VerdictKind.HEALTHY) return if (rejected) PrimaryAction.ABSTAIN_HEALTHY else PrimaryAction.NONE
        if (rejected || evidence.positive < policy.healthyMin || evidence.genus < policy.genusMin || positive < policy.compactMin) return PrimaryAction.NONE
        if (baseline.kind == VerdictKind.UNSURE && baseline.labelIndex != null && baseline.labelIndex != healthyIndex && baseline.labelIndex != otherIndex) return PrimaryAction.NONE
        val probabilities = posterior ?: evidence.originalWholeProbabilities ?: return PrimaryAction.NONE
        require(healthyIndex in probabilities.indices && otherIndex in probabilities.indices)
        val diseaseMass = probabilities.indices.filter { it != healthyIndex && it != otherIndex }.sumOf { probabilities[it] }
        if (probabilities[otherIndex] <= policy.closedOtherMax && diseaseMass >= policy.closedDiseaseMassMin) return PrimaryAction.NONE
        val enoughLocalViews = if (evidence.natural >= policy.naturalMin) evidence.leaves.size >= 2 else evidence.leaves.isNotEmpty()
        val eligible = enoughLocalViews
        val localNegative = evidence.leaves.map { maxOf(it.center.adapted, it.full.adapted) }
        val conflict = eligible && (localNegative.any { it <= policy.strongLocalMax } || localNegative.count { it <= policy.partialLocalMax } >= policy.partialLocalLeaves)
        val isolated = eligible && evidence.natural < policy.naturalMin && evidence.leaves.all { minOf(it.center.adapted, it.full.adapted) >= policy.isolatedMin }
        if (!(scoped || isolated) || conflict) return PrimaryAction.NONE
        if (evidence.leaves.count { it.center.adapted < policy.localCenterMin } >= policy.wholeLocalLeaves) {
            if (evidence.wholeAdapted == null || evidence.wholeAdapted < policy.wholeHealthyMin) return PrimaryAction.NONE
        }
        return PrimaryAction.RECOVER_HEALTHY
    }
}
