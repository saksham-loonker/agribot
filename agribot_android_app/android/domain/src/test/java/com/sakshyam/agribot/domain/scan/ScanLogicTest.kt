package com.sakshyam.agribot.domain.scan

import com.sakshyam.agribot.domain.model.BoundingBox
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import org.junit.Test

private const val N = 9
private const val EB = 0
private const val HEALTHY = 1
private const val LB = 2
private const val OTHER = 8

private fun probs(vararg pairs: Pair<Int, Float>): FloatArray {
    val p = FloatArray(N)
    val rest = 1f - pairs.sumOf { it.second.toDouble() }.toFloat()
    for (k in 0 until N) p[k] = rest / (N - pairs.size)
    for ((k, v) in pairs) p[k] = v
    return p
}

private fun box(x: Float, y: Float = 10f, s: Float = 40f) = BoundingBox(x, y, x + s, y + s)

class PlantEvidenceTest {
    private fun evidence() = PlantEvidence(N, HEALTHY, OTHER)

    private fun PlantEvidence.feed(frames: Int, leaves: Map<Long, FloatArray>) {
        for (f in 0 until frames) for ((id, p) in leaves) add(id, f.toLong(), LeafObservation(box(id * 50f), 0.9f, p))
    }

    @Test fun `needs more views before deciding`() {
        val e = evidence()
        e.feed(2, mapOf(1L to probs(HEALTHY to 0.95f)))
        assertEquals(VerdictKind.NEED_MORE_VIEWS, e.verdict().kind)
    }

    @Test fun `consistent healthy leaves give healthy`() {
        val e = evidence()
        e.feed(4, mapOf(1L to probs(HEALTHY to 0.9f), 2L to probs(HEALTHY to 0.85f)))
        val v = e.verdict()
        assertEquals(VerdictKind.HEALTHY, v.kind)
        assertEquals(HEALTHY, v.labelIndex)
        assertEquals(2, v.leavesSeen)
        assertEquals(2, v.leavesAgreeing)
        assertFalse(v.partial)
    }

    @Test fun `two strongly blighted leaves are reported even when most leaves look healthy`() {
        val e = evidence()
        e.feed(4, mapOf(
            1L to probs(HEALTHY to 0.95f), 2L to probs(HEALTHY to 0.95f), 3L to probs(HEALTHY to 0.95f),
            4L to probs(HEALTHY to 0.95f), 5L to probs(LB to 0.85f), 6L to probs(LB to 0.80f),
        ))
        val v = e.verdict()
        assertEquals(VerdictKind.DISEASE, v.kind)
        assertEquals(LB, v.labelIndex)
        assertTrue(v.partial)
        assertEquals(2, v.leavesAgreeing)
    }

    @Test fun `a single diseased leaf does not override a healthy plant`() {
        val e = evidence()
        e.feed(4, mapOf(1L to probs(HEALTHY to 0.95f), 2L to probs(HEALTHY to 0.95f), 3L to probs(LB to 0.9f)))
        val v = e.verdict()
        assertEquals(VerdictKind.HEALTHY, v.kind)
    }

    @Test fun `one leaf seen in many frames cannot outvote other leaves`() {
        val e = evidence()
        repeat(30) { f -> e.add(1L, f.toLong(), LeafObservation(box(0f), 0.9f, probs(EB to 0.9f))) }
        repeat(4) { f ->
            for (id in 2L..4L) e.add(id, f.toLong(), LeafObservation(box(id * 50f), 0.9f, probs(HEALTHY to 0.9f)))
        }
        val v = e.verdict()
        assertEquals(HEALTHY, v.labelIndex)
    }

    @Test fun `split evidence is unsure`() {
        val e = evidence()
        e.feed(5, mapOf(1L to probs(EB to 0.45f, LB to 0.40f)))
        assertEquals(VerdictKind.UNSURE, e.verdict().kind)
    }

    @Test fun `unknown condition when Other dominates`() {
        val e = evidence()
        e.feed(4, mapOf(1L to probs(OTHER to 0.8f), 2L to probs(OTHER to 0.7f)))
        assertEquals(VerdictKind.UNKNOWN_CONDITION, e.verdict().kind)
    }

    @Test fun `empty evidence asks for more views`() {
        val v = evidence().verdict()
        assertEquals(VerdictKind.NEED_MORE_VIEWS, v.kind)
        assertNull(v.labelIndex)
    }
}

class LeafTrackerTest {
    @Test fun `overlapping boxes keep identity and new boxes get new ids`() {
        val t = LeafTracker()
        val a = t.update(0, listOf(LeafObservation(box(0f), 1f, probs()), LeafObservation(box(200f), 1f, probs())))
        val b = t.update(1, listOf(LeafObservation(box(5f), 1f, probs()), LeafObservation(box(400f), 1f, probs())))
        assertEquals(a[0], b[0])
        assertTrue(b[1] != a[0] && b[1] != a[1])
    }

    @Test fun `tracks expire after missed frames`() {
        val t = LeafTracker(maxMissedFrames = 2)
        val a = t.update(0, listOf(LeafObservation(box(0f), 1f, probs())))
        val b = t.update(10, listOf(LeafObservation(box(0f), 1f, probs())))
        assertTrue(a[0] != b[0])
    }
}

class RowWalkTrackerTest {
    @Test fun `steps advance plants by spacing and clamp at row end`() {
        val r = RowWalkTracker(rowCount = 2, plantsPerRow = 3, plantSpacingM = 0.6, strideM = 0.3)
        assertEquals(1, r.plantNumber)
        r.onStep(); assertEquals(1, r.plantNumber)
        r.onStep(); assertEquals(2, r.plantNumber)
        repeat(20) { r.onStep() }
        assertEquals(3, r.plantNumber)
        assertTrue(r.atLastPlant)
    }

    @Test fun `manual next and previous re-anchor distance`() {
        val r = RowWalkTracker(1, 5, 0.5, 0.7)
        assertTrue(r.nextPlant()); assertEquals(2, r.plantNumber)
        assertTrue(r.previousPlant()); assertEquals(1, r.plantNumber)
        assertFalse(r.previousPlant())
        r.jumpTo(5); assertFalse(r.nextPlant())
    }

    @Test fun `next row resets distance and stops after last row`() {
        val r = RowWalkTracker(2, 4, 0.5, 0.5)
        repeat(3) { r.onStep() }
        assertTrue(r.nextRow()); assertEquals(2, r.rowIndex); assertEquals(1, r.plantNumber)
        assertFalse(r.nextRow())
    }
}

class GpsTrailFilterTest {
    private val base = GpsFix(18.5204, 73.8567, 5.0, 0)

    @Test fun `stationary jitter never grows the trail`() {
        val f = GpsTrailFilter()
        assertTrue(f.offer(base, walkingRecently = false))
        for (i in 1..100) {
            val jitter = GpsFix(base.latitude + (i % 3 - 1) * 2e-5, base.longitude + (i % 2) * 2e-5, 5.0, i * 1000L)
            f.offer(jitter, walkingRecently = false)
        }
        assertEquals(1, f.points.size)
        assertEquals(0.0, f.lengthM())
    }

    @Test fun `walking adds points spaced beyond accuracy and rejects bad fixes`() {
        val f = GpsTrailFilter()
        f.offer(base, true)
        assertFalse(f.offer(base.copy(accuracyM = 40.0, timeMs = 1000), true))          // inaccurate
        assertFalse(f.offer(base.copy(latitude = base.latitude + 1e-5, timeMs = 2000), true)) // ~1 m: too close
        assertTrue(f.offer(base.copy(latitude = base.latitude + 9e-5, timeMs = 10_000), true)) // ~10 m in 10 s
        assertFalse(f.offer(base.copy(latitude = base.latitude + 9e-3, timeMs = 11_000), true)) // 1 km in 1 s
        assertEquals(2, f.points.size)
        assertTrue(f.lengthM() in 9.0..11.0)
    }
}
