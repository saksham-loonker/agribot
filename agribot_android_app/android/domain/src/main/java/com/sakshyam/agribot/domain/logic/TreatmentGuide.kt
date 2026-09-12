package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.TreatmentRecommendation

/**
 * Maps plant disease labels to recommended treatments.
 * Farmers can look up what action to take when a disease is detected.
 */
object TreatmentGuide {

    private val recommendations: Map<String, TreatmentRecommendation> = mapOf(
        "Early_blight" to TreatmentRecommendation(
            diseaseLabel = "Early_blight",
            treatment = "Remove infected leaves and apply fungicide",
            chemical = "Chlorothalonil or Mancozeb",
            organic = "Copper-based fungicide or neem oil",
            notes = "Apply every 7-10 days. Rotate crops annually.",
        ),
        "Late_blight" to TreatmentRecommendation(
            diseaseLabel = "Late_blight",
            treatment = "Immediately remove and destroy infected plants",
            chemical = "Metalaxyl or Mefenoxam",
            organic = "Copper hydroxide",
            notes = "Highly contagious. Do not compost infected material.",
        ),
        "Leaf Miner" to TreatmentRecommendation(
            diseaseLabel = "Leaf Miner",
            treatment = "Remove infested leaves and use row covers",
            chemical = "Spinosad or Abamectin",
            organic = "Neem oil or insecticidal soap",
            notes = "Encourage natural predators like parasitic wasps.",
        ),
        "Magnesium Deficiency" to TreatmentRecommendation(
            diseaseLabel = "Magnesium Deficiency",
            treatment = "Apply magnesium supplement",
            chemical = "Epsom salt (magnesium sulfate) solution",
            organic = "Compost tea or kelp meal",
            notes = "Dissolve 1-2 tbsp Epsom salt per gallon of water.",
        ),
        "Nitrogen Deficiency" to TreatmentRecommendation(
            diseaseLabel = "Nitrogen Deficiency",
            treatment = "Apply nitrogen-rich fertilizer",
            chemical = "Urea or Ammonium nitrate",
            organic = "Compost, aged manure, or fish emulsion",
            notes = "Water before applying to prevent root burn.",
        ),
        "Pottassium Deficiency" to TreatmentRecommendation(
            diseaseLabel = "Pottassium Deficiency",
            treatment = "Apply potassium supplement",
            chemical = "Potassium chloride or sulfate of potash",
            organic = "Wood ash or kelp meal",
            notes = "Best applied in early morning or evening.",
        ),
        "Spotted Wilt Virus" to TreatmentRecommendation(
            diseaseLabel = "Spotted Wilt Virus",
            treatment = "Remove infected plants and control thrips",
            chemical = "Insecticidal soap for thrips control",
            organic = "Neem oil and reflective mulch",
            notes = "No cure for the virus. Focus on vector control.",
        ),
    )

    /**
     * Returns the treatment recommendation for a disease label, or null if not found.
     */
    fun recommendationFor(label: String): TreatmentRecommendation? {
        val normalized = label.trim().lowercase().replace("_", " ")
        return recommendations.entries.find { entry ->
            entry.key.lowercase().replace("_", " ") == normalized
        }?.value
    }

    /**
     * Returns a list of all known disease labels that have treatment recommendations.
     */
    fun knownDiseaseLabels(): List<String> = recommendations.keys.toList()
}
