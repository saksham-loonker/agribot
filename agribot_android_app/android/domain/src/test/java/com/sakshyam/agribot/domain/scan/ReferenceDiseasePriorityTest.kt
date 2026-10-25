package com.sakshyam.agribot.domain.scan

import com.sakshyam.agribot.domain.model.BoundingBox
import org.junit.Assert.*
import org.junit.Test
import kotlin.random.Random

class ReferenceDiseasePriorityTest {
    private fun p(d: Float, h: Float, other: Float) = FloatArray(9).apply { this[0] = d; this[1] = h; this[8] = other }
    private fun observation(probabilities: FloatArray, reference: FloatArray? = null, quality: Float = 1f) =
        LeafObservation(BoundingBox(0f, 0f, 10f, 10f), 0.9f, probabilities, quality, reference)

    @Test fun `healthy recovery cannot outvote an accumulated reference disease`() {
        val old = PlantEvidence(9, 1, 8)
        val candidate = PlantEvidence(9, 1, 8)
        val protected = PlantEvidence(9, 1, 8, referenceDiseasePriority = true)
        for (frame in 1L..7L) for (track in 0..2) {
            val reference = if (track == 0) p(.99f, .005f, .005f) else p(.4f, .25f, .35f)
            val proposed = if (track == 0) reference else p(.005f, .99f, .005f)
            old.add(track.toLong(), frame, observation(reference))
            candidate.add(track.toLong(), frame, observation(proposed))
            protected.add(track.toLong(), frame, observation(proposed, reference))
        }
        assertEquals(VerdictKind.HEALTHY, candidate.verdict().kind)
        assertEquals(VerdictKind.DISEASE, old.verdict().kind)
        assertEquals(old.verdict(), protected.verdict())
        assertEquals(3, protected.leavesSeen)
        assertEquals(7, protected.framesUsed)
        protected.clear()
        assertEquals(VerdictKind.NEED_MORE_VIEWS, protected.verdict().kind)
        assertEquals(0, protected.observations)
    }

    @Test fun `healthy recovery remains available when reference cannot identify a condition`() {
        val evidence = PlantEvidence(9, 1, 8, referenceDiseasePriority = true)
        for (frame in 1L..7L) evidence.add(1, frame, observation(p(.005f, .99f, .005f), p(.05f, .05f, .9f)))
        assertEquals(VerdictKind.HEALTHY, evidence.verdict().kind)
        assertEquals(1, evidence.verdict().leavesSeen)
    }

    @Test fun `reference disease priority requires the complete model contract`() {
        val evidence = PlantEvidence(9, 1, 8, referenceDiseasePriority = true)
        assertThrows(IllegalArgumentException::class.java) { evidence.add(1, 1, observation(p(.1f, .8f, .1f))) }
        assertEquals(0, evidence.observations)
        assertEquals(0, evidence.framesUsed)
    }

    @Test fun `reference diseases survive changing quality and candidate predictions`() {
        val rng = Random(347)
        val old = PlantEvidence(9, 1, 8)
        val protected = PlantEvidence(9, 1, 8, referenceDiseasePriority = true)
        var checked = 0
        for (frame in 1L..180L) {
            for (track in 0L..3L) {
                val reference = FloatArray(9) { rng.nextFloat() * .01f }
                reference[listOf(0, 2, 3, 4, 5, 6, 7)[((frame / 40) % 7).toInt()]] += .9f
                val total = reference.sum()
                for (i in reference.indices) reference[i] /= total
                val quality = .3f + .7f * rng.nextFloat()
                old.add(track, frame, observation(reference, quality = quality))
                protected.add(track, frame, observation(p(.005f, .99f, .005f), reference, quality))
            }
            val reference = old.verdict()
            if (reference.kind == VerdictKind.DISEASE) {
                assertEquals(reference, protected.verdict())
                checked++
            }
        }
        assertTrue(checked > 30)
    }
}
