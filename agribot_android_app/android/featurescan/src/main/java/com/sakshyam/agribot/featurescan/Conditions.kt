package com.sakshyam.agribot.featurescan

import androidx.annotation.StringRes

/** Localised text for one model label. Keys are the model's label strings (see model_manifest.json). */
data class ConditionText(
    @StringRes val name: Int,
    @StringRes val body: Int,
    val advice: List<Int>,
    val kind: ConditionKind,
)

enum class ConditionKind { HEALTHY, DISEASE, DEFICIENCY, PEST, VIRUS, OTHER }

object Conditions {
    private val byKey: Map<String, ConditionText> = mapOf(
        "Early_blight" to ConditionText(
            R.string.cond_early_blight, R.string.cond_early_blight_body,
            listOf(R.string.cond_early_blight_do1, R.string.cond_early_blight_do2, R.string.cond_early_blight_do3, R.string.cond_early_blight_do4),
            ConditionKind.DISEASE,
        ),
        "Late_blight" to ConditionText(
            R.string.cond_late_blight, R.string.cond_late_blight_body,
            listOf(R.string.cond_late_blight_do1, R.string.cond_late_blight_do2, R.string.cond_late_blight_do3, R.string.cond_late_blight_do4),
            ConditionKind.DISEASE,
        ),
        "Leaf Miner" to ConditionText(
            R.string.cond_leaf_miner, R.string.cond_leaf_miner_body,
            listOf(R.string.cond_leaf_miner_do1, R.string.cond_leaf_miner_do2, R.string.cond_leaf_miner_do3),
            ConditionKind.PEST,
        ),
        "Magnesium Deficiency" to ConditionText(
            R.string.cond_magnesium, R.string.cond_magnesium_body,
            listOf(R.string.cond_magnesium_do1, R.string.cond_magnesium_do2, R.string.cond_magnesium_do3),
            ConditionKind.DEFICIENCY,
        ),
        "Nitrogen Deficiency" to ConditionText(
            R.string.cond_nitrogen, R.string.cond_nitrogen_body,
            listOf(R.string.cond_nitrogen_do1, R.string.cond_nitrogen_do2, R.string.cond_nitrogen_do3),
            ConditionKind.DEFICIENCY,
        ),
        // The model label is misspelled ("Pottassium") in the source dataset; only the display text is fixed.
        "Pottassium Deficiency" to ConditionText(
            R.string.cond_potassium, R.string.cond_potassium_body,
            listOf(R.string.cond_potassium_do1, R.string.cond_potassium_do2, R.string.cond_potassium_do3),
            ConditionKind.DEFICIENCY,
        ),
        "Spotted Wilt Virus" to ConditionText(
            R.string.cond_spotted_wilt, R.string.cond_spotted_wilt_body,
            listOf(R.string.cond_spotted_wilt_do1, R.string.cond_spotted_wilt_do2, R.string.cond_spotted_wilt_do3, R.string.cond_spotted_wilt_do4),
            ConditionKind.VIRUS,
        ),
        "Healthy" to ConditionText(
            R.string.cond_healthy, R.string.cond_healthy_body,
            listOf(R.string.cond_healthy_do1, R.string.cond_healthy_do2),
            ConditionKind.HEALTHY,
        ),
        "Other" to ConditionText(
            R.string.cond_other, R.string.cond_other_body,
            listOf(R.string.cond_other_do1, R.string.cond_other_do2, R.string.cond_other_do3),
            ConditionKind.OTHER,
        ),
    )

    val knownKeys: Set<String> get() = byKey.keys

    fun forKey(key: String?): ConditionText? = key?.let { byKey[it] }

    @StringRes
    fun nameRes(key: String?): Int = forKey(key)?.name ?: R.string.cond_unknown_label
}
