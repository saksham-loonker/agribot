package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.TreatmentRecommendation
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class TreatmentGuideTest {

    @Test
    fun recommendationForKnownDiseaseReturnsTreatment() {
        val recommendation = TreatmentGuide.recommendationFor("Early_blight")
        assertNotNull(recommendation)
        assertEquals("Remove infected leaves and apply fungicide", recommendation.treatment)
        assertEquals("Chlorothalonil or Mancozeb", recommendation.chemical)
        assertEquals("Copper-based fungicide or neem oil", recommendation.organic)
        assertNotNull(recommendation.notes)
    }

    @Test
    fun recommendationForLateBlightReturnsTreatment() {
        val recommendation = TreatmentGuide.recommendationFor("Late_blight")
        assertNotNull(recommendation)
        assertEquals("Immediately remove and destroy infected plants", recommendation.treatment)
        assertEquals("Metalaxyl or Mefenoxam", recommendation.chemical)
    }

    @Test
    fun recommendationForLeafMinerReturnsTreatment() {
        val recommendation = TreatmentGuide.recommendationFor("Leaf Miner")
        assertNotNull(recommendation)
        assertEquals("Remove infested leaves and use row covers", recommendation.treatment)
    }

    @Test
    fun recommendationForNutrientDeficienciesReturnsTreatment() {
        val magnesium = TreatmentGuide.recommendationFor("Magnesium Deficiency")
        assertNotNull(magnesium)
        assertEquals("Apply magnesium supplement", magnesium.treatment)

        val nitrogen = TreatmentGuide.recommendationFor("Nitrogen Deficiency")
        assertNotNull(nitrogen)
        assertEquals("Apply nitrogen-rich fertilizer", nitrogen.treatment)

        val potassium = TreatmentGuide.recommendationFor("Pottassium Deficiency")
        assertNotNull(potassium)
        assertEquals("Apply potassium supplement", potassium.treatment)
    }

    @Test
    fun recommendationForSpottedWiltVirusReturnsTreatment() {
        val recommendation = TreatmentGuide.recommendationFor("Spotted Wilt Virus")
        assertNotNull(recommendation)
        assertEquals("Remove infected plants and control thrips", recommendation.treatment)
    }

    @Test
    fun recommendationForUnknownLabelReturnsNull() {
        assertNull(TreatmentGuide.recommendationFor("Unknown Disease"))
        assertNull(TreatmentGuide.recommendationFor(""))
    }

    @Test
    fun recommendationForHealthyLabelReturnsNull() {
        assertNull(TreatmentGuide.recommendationFor("Healthy"))
    }

    @Test
    fun recommendationForLabelIsCaseInsensitive() {
        val upper = TreatmentGuide.recommendationFor("EARLY_BLIGHT")
        assertNotNull(upper)
        assertEquals("Remove infected leaves and apply fungicide", upper.treatment)

        val mixed = TreatmentGuide.recommendationFor("early_blight")
        assertNotNull(mixed)
        assertEquals("Remove infected leaves and apply fungicide", mixed.treatment)
    }

    @Test
    fun recommendationForLabelWithSpacesIsNormalized() {
        val withSpaces = TreatmentGuide.recommendationFor("Early blight")
        assertNotNull(withSpaces)
        assertEquals("Remove infected leaves and apply fungicide", withSpaces.treatment)
    }

    @Test
    fun knownDiseaseLabelsReturnsAllRecommendations() {
        val labels = TreatmentGuide.knownDiseaseLabels()
        assertEquals(7, labels.size)
        assertTrue(labels.contains("Early_blight"))
        assertTrue(labels.contains("Late_blight"))
        assertTrue(labels.contains("Leaf Miner"))
        assertTrue(labels.contains("Magnesium Deficiency"))
        assertTrue(labels.contains("Nitrogen Deficiency"))
        assertTrue(labels.contains("Pottassium Deficiency"))
        assertTrue(labels.contains("Spotted Wilt Virus"))
    }

    @Test
    fun everyKnownLabelHasRecommendation() {
        TreatmentGuide.knownDiseaseLabels().forEach { label ->
            val recommendation = TreatmentGuide.recommendationFor(label)
            assertNotNull(recommendation, "No recommendation for $label")
            assertTrue(recommendation.treatment.isNotBlank(), "Empty treatment for $label")
        }
    }
}
