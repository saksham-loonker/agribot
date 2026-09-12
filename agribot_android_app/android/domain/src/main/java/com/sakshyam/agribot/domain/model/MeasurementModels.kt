package com.sakshyam.agribot.domain.model

/**
 * The quality of a value is intentionally separate from the numeric value.
 * A missing measurement must never be represented by a silently displayed zero.
 */
enum class MeasurementQuality {
    NOT_STARTED,
    UNAVAILABLE,
    ESTIMATED,
    REFERENCED,
    RELATIVE,
    CALIBRATED,
    SUSPECT,
}

enum class MeasurementSource {
    NONE,
    PHONE_STEP_SENSOR,
    PHONE_ACCELERATION,
    PHONE_SENSORS,
    GPS_REFERENCE,
    GPS_AND_PHONE_SENSORS,
    CAMERA_RELATIVE,
    CAMERA_CALIBRATED,
}

data class MeasurementValue(
    val value: Float?,
    val quality: MeasurementQuality,
    val source: MeasurementSource,
    val message: String,
) {
    init {
        require(value == null || value.isFinite()) { "Measurement values must be finite" }
        require(value == null || value >= 0f) { "Measurement values cannot be negative" }
        if (quality == MeasurementQuality.NOT_STARTED || quality == MeasurementQuality.UNAVAILABLE) {
            require(value == null) { "Unavailable measurements must not carry a numeric value" }
        }
    }

    val isAvailable: Boolean
        get() = value != null && quality != MeasurementQuality.SUSPECT
}

data class MotionMeasurementResult(
    val distance: MeasurementValue,
    val speed: MeasurementValue,
    val stepCount: Int?,
    val source: MeasurementSource,
    val statusMessage: String,
    val gpsReferenceUsable: Boolean,
    val motionEventsObserved: Boolean,
    val gpsFixAgeSeconds: Float?,
) {
    init {
        require(stepCount == null || stepCount >= 0) { "Step count cannot be negative" }
        require(gpsFixAgeSeconds == null || gpsFixAgeSeconds.isFinite()) {
            "GPS fix age must be finite"
        }
    }
}

data class ObservedIssue(
    val stableId: String,
    val label: String,
    val confirmed: Boolean,
    val reviewRequired: Boolean = false,
    val geometryValid: Boolean = true,
)

data class LeadingObservedIssue(
    val label: String?,
    val affectedCount: Int,
    val inspectedCount: Int,
    val plannedCount: Int?,
    val marginOverSecond: Int,
    val provisional: Boolean,
    val reason: String,
) {
    val coverageLabel: String
        get() = if (plannedCount != null && plannedCount > 0) {
            "$inspectedCount / $plannedCount checked"
        } else {
            "$inspectedCount checked"
        }
}
