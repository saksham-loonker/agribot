package com.sakshyam.agribot.domain.scan

import com.sakshyam.agribot.domain.model.BoundingBox
import kotlin.math.max
import kotlin.math.min

/** One classified leaf in one camera frame. */
class LeafObservation(
    val box: BoundingBox,
    val detectorScore: Float,
    /** Calibrated class probabilities from the classifier ensemble (sums to 1). */
    val probabilities: FloatArray,
    /** 0..1 frame quality (sharpness/exposure); scales how much this frame counts. */
    val frameQuality: Float = 1f,
)

/**
 * Frame-to-frame leaf identity by greedy IoU association. Lets the plant aggregator weigh evidence per
 * physical leaf, so one leaf seen in many frames cannot outvote several different leaves.
 */
class LeafTracker(
    private val minIou: Float = 0.30f,
    private val maxMissedFrames: Int = 6,
) {
    private class Track(val id: Long, var box: BoundingBox, var lastFrame: Long)

    private val tracks = ArrayList<Track>()
    private var nextId = 1L

    /** Returns the track id assigned to each observation (same order as [observations]). */
    fun update(frameIndex: Long, observations: List<LeafObservation>): LongArray {
        tracks.removeAll { frameIndex - it.lastFrame > maxMissedFrames }
        val pairs = ArrayList<Triple<Float, Int, Int>>()
        for (o in observations.indices) for (t in tracks.indices) {
            val iou = iou(observations[o].box, tracks[t].box)
            if (iou >= minIou) pairs += Triple(iou, o, t)
        }
        pairs.sortByDescending { it.first }
        val ids = LongArray(observations.size) { -1L }
        val usedTracks = HashSet<Int>()
        for ((_, o, t) in pairs) {
            if (ids[o] != -1L || t in usedTracks) continue
            ids[o] = tracks[t].id
            tracks[t].box = observations[o].box
            tracks[t].lastFrame = frameIndex
            usedTracks += t
        }
        for (o in observations.indices) if (ids[o] == -1L) {
            val tr = Track(nextId++, observations[o].box, frameIndex)
            tracks += tr
            ids[o] = tr.id
        }
        return ids
    }

    fun reset() {
        tracks.clear()
    }

    companion object {
        fun iou(a: BoundingBox, b: BoundingBox): Float {
            val ix = max(0f, min(a.right, b.right) - max(a.left, b.left))
            val iy = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
            val inter = ix * iy
            val union = max(0f, a.width) * max(0f, a.height) + max(0f, b.width) * max(0f, b.height) - inter
            return if (union > 0f) inter / union else 0f
        }
    }
}
