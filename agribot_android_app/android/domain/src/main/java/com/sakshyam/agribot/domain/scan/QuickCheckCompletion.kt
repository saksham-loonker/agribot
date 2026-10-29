package com.sakshyam.agribot.domain.scan

/** Shared by the live scan and offline native replay; no image labels enter this decision. */
object QuickCheckCompletion {
    fun isReady(verdict: PlantVerdict, framesUsed: Int, stableFrames: Int, primaryPolicy: PrimaryImagePolicy?): Boolean {
        val recoveryWindowReady = verdict.kind == VerdictKind.DISEASE || framesUsed >= (primaryPolicy?.minimumFrames ?: 0)
        return recoveryWindowReady && verdict.kind != VerdictKind.NEED_MORE_VIEWS &&
            ((verdict.kind != VerdictKind.UNSURE && stableFrames >= 2) || framesUsed >= 14)
    }
}
