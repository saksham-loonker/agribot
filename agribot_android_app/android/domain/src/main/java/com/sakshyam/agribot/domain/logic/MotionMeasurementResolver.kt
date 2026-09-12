package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.MeasurementQuality
import com.sakshyam.agribot.domain.model.MeasurementSource
import com.sakshyam.agribot.domain.model.MeasurementValue
import com.sakshyam.agribot.domain.model.MotionMeasurementResult

data class MotionMeasurementInput(
    val trackingActive: Boolean,
    val motionEventsObserved: Boolean,
    val stepEventsObserved: Boolean = false,
    val sensorDistanceM: Float?,
    val sensorSpeedMps: Float?,
    val stepCount: Int?,
    val gpsReferenceEnabled: Boolean,
    val gpsFixAvailable: Boolean,
    val gpsDistanceM: Float?,
    val gpsSpeedMps: Float?,
    val gpsAccuracyM: Float?,
    val gpsFixAgeSeconds: Float?,
    val tractorMounted: Boolean = false,
)

/**
 * Chooses a source based on observed, valid events rather than sensor catalogue
 * presence. This policy is pure so it can be tested without an Android device.
 */
object MotionMeasurementResolver {
    private const val MAX_GPS_ACCURACY_M = 25f
    private const val MAX_GPS_AGE_SECONDS = 5f
    private const val MAX_SPEED_MPS = 3f

    fun resolve(input: MotionMeasurementInput): MotionMeasurementResult {
        if (!input.trackingActive) return notStarted()

        val sensorDistance = nonNegativeFinite(input.sensorDistanceM)
        val sensorSpeed = nonNegativeFinite(input.sensorSpeedMps)?.coerceAtMost(MAX_SPEED_MPS)
        val sensorUsable = input.motionEventsObserved && sensorDistance != null && sensorSpeed != null
        val gpsAge = nonNegativeFinite(input.gpsFixAgeSeconds)
        val gpsUsable = input.gpsReferenceEnabled &&
            input.gpsFixAvailable &&
            nonNegativeFinite(input.gpsDistanceM) != null &&
            nonNegativeFinite(input.gpsAccuracyM)?.let { it <= MAX_GPS_ACCURACY_M } == true &&
            gpsAge?.let { it <= MAX_GPS_AGE_SECONDS } == true

        if (input.tractorMounted) {
            val message = if (gpsUsable) {
                "Tractor GPS travel estimate · plant positioning needs camera tracking"
            } else {
                "Tractor travel unavailable · waiting for camera tracking or fresh GPS"
            }
            val speed = nonNegativeFinite(input.gpsSpeedMps)?.takeIf { it <= 20f }
            return MotionMeasurementResult(
                distance = if (gpsUsable) available(input.gpsDistanceM!!, MeasurementQuality.ESTIMATED, MeasurementSource.GPS_REFERENCE, message) else unavailable(message),
                speed = if (gpsUsable && speed != null) available(speed, MeasurementQuality.ESTIMATED, MeasurementSource.GPS_REFERENCE, message) else unavailable(message),
                stepCount = null,
                source = if (gpsUsable) MeasurementSource.GPS_REFERENCE else MeasurementSource.NONE,
                statusMessage = message,
                gpsReferenceUsable = gpsUsable,
                motionEventsObserved = input.motionEventsObserved,
                gpsFixAgeSeconds = gpsAge,
            )
        }

        return when {
            gpsUsable -> {
                val gpsDistance = nonNegativeFinite(input.gpsDistanceM)!!
                val resolvedDistance = when {
                    sensorUsable && gpsDistance > 0f && sensorDistance!! > 0f ->
                        blendDistance(sensorDistance, gpsDistance)
                    sensorUsable -> sensorDistance!!
                    else -> gpsDistance
                }
                val gpsSpeed = nonNegativeFinite(input.gpsSpeedMps)?.coerceAtMost(MAX_SPEED_MPS)
                val resolvedSpeed = when {
                    sensorUsable && gpsSpeed != null -> blend(sensorSpeed!!, gpsSpeed)
                    gpsSpeed != null -> gpsSpeed
                    sensorUsable -> sensorSpeed!!
                    else -> null
                }
                val source = if (sensorUsable) {
                    MeasurementSource.GPS_AND_PHONE_SENSORS
                } else {
                    MeasurementSource.GPS_REFERENCE
                }
                MotionMeasurementResult(
                    distance = available(
                        value = resolvedDistance,
                        quality = MeasurementQuality.REFERENCED,
                        source = source,
                        message = "GPS reference is fresh",
                    ),
                    speed = resolvedSpeed?.let {
                        available(
                            value = it,
                            quality = MeasurementQuality.REFERENCED,
                            source = source,
                            message = "Speed uses the active field reference",
                        )
                    } ?: unavailable("Speed unavailable until a motion or GPS speed event arrives"),
                    stepCount = input.stepCount.takeIf { sensorUsable },
                    source = source,
                    statusMessage = if (sensorUsable) {
                        "Walking distance active · GPS reference"
                    } else {
                        "GPS reference active · motion events pending"
                    },
                    gpsReferenceUsable = true,
                    motionEventsObserved = input.motionEventsObserved,
                    gpsFixAgeSeconds = gpsAge,
                )
            }

            sensorUsable -> {
                val source = if (input.stepEventsObserved) {
                    MeasurementSource.PHONE_STEP_SENSOR
                } else {
                    MeasurementSource.PHONE_ACCELERATION
                }
                MotionMeasurementResult(
                distance = available(
                    value = sensorDistance!!,
                    quality = MeasurementQuality.ESTIMATED,
                    source = source,
                    message = "Approximate distance from phone motion",
                ),
                speed = available(
                    value = sensorSpeed!!,
                    quality = MeasurementQuality.ESTIMATED,
                    source = source,
                    message = "Approximate speed from phone motion",
                ),
                stepCount = input.stepCount,
                source = source,
                statusMessage = if (input.gpsReferenceEnabled && input.gpsFixAvailable) {
                    "Motion estimate active · GPS reference weak or stale"
                } else {
                    "Motion estimate active · GPS reference unavailable"
                },
                gpsReferenceUsable = false,
                motionEventsObserved = true,
                gpsFixAgeSeconds = gpsAge,
                )
            }

            else -> {
                val message = if (input.gpsReferenceEnabled && input.gpsFixAvailable) {
                    "Motion signal weak · GPS reference not trusted"
                } else {
                    "Motion signal weak · distance unavailable"
                }
                MotionMeasurementResult(
                    distance = unavailable(message),
                    speed = unavailable(message),
                    stepCount = null,
                    source = MeasurementSource.NONE,
                    statusMessage = message,
                    gpsReferenceUsable = false,
                    motionEventsObserved = input.motionEventsObserved,
                    gpsFixAgeSeconds = gpsAge,
                )
            }
        }
    }

    fun notStarted(): MotionMeasurementResult = MotionMeasurementResult(
        distance = MeasurementValue(null, MeasurementQuality.NOT_STARTED, MeasurementSource.NONE, "Start a scan to measure distance"),
        speed = MeasurementValue(null, MeasurementQuality.NOT_STARTED, MeasurementSource.NONE, "Start a scan to measure speed"),
        stepCount = null,
        source = MeasurementSource.NONE,
        statusMessage = "Motion sensors idle",
        gpsReferenceUsable = false,
        motionEventsObserved = false,
        gpsFixAgeSeconds = null,
    )

    /**
     * Freeze the last reading when a scan stops.
     *
     * Stopping a sensor listener is a lifecycle event, not evidence that the
     * farmer travelled zero metres. The next explicit scan reset still clears
     * the session, while this function preserves every numeric value and only
     * changes the status text shown after completion.
     */
    fun retainAfterStop(result: MotionMeasurementResult): MotionMeasurementResult =
        result.copy(
            statusMessage = when {
                result.distance.value != null || result.speed.value != null ->
                    "Last scan measurement kept"
                result.distance.quality == MeasurementQuality.NOT_STARTED ->
                    "No scan measurement yet"
                else ->
                    "No valid movement signal captured"
            },
        )

    private fun available(
        value: Float,
        quality: MeasurementQuality,
        source: MeasurementSource,
        message: String,
    ): MeasurementValue = MeasurementValue(value, quality, source, message)

    private fun unavailable(message: String): MeasurementValue =
        MeasurementValue(null, MeasurementQuality.UNAVAILABLE, MeasurementSource.NONE, message)

    private fun nonNegativeFinite(value: Float?): Float? =
        value?.takeIf { it.isFinite() && it >= 0f }

    private fun blend(left: Float, right: Float): Float =
        ((left * 0.55f) + (right * 0.45f)).coerceIn(0f, MAX_SPEED_MPS)

    private fun blendDistance(left: Float, right: Float): Float =
        ((left * 0.55f) + (right * 0.45f)).coerceAtLeast(0f)
}
