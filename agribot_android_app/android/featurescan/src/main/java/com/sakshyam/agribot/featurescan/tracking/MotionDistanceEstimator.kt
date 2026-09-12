package com.sakshyam.agribot.featurescan.tracking

import com.sakshyam.agribot.domain.logic.WalkingCalibrationPolicy
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.math.sqrt

data class MotionEstimate(
    val distanceM: Float,
    val speedMps: Float,
    val stepCount: Int,
    val stepEventsObserved: Boolean,
    val motionEventsObserved: Boolean,
)

/**
 * Sensor-only distance estimator with a hardware step-detector path and a
 * conservative acceleration-derived step fallback. Android devices can expose
 * a step sensor but suppress its events until ACTIVITY_RECOGNITION is granted;
 * using the mere presence of that sensor as a permanent switch made distance
 * and speed stay at zero on those phones.
 *
 * The acceleration fallback deliberately detects periodic step-like pulses
 * instead of integrating acceleration magnitude into velocity. Magnitude has
 * no walking direction and naive integration turns vibration, tilt, and
 * sensor bias into invented distance.
 */
class MotionDistanceEstimator(
    initialStepLengthM: Float = DEFAULT_STEP_LENGTH_M,
    private val stepFreshnessNanos: Long = STEP_FRESHNESS_NANOS,
) {
    private var stepLengthM: Float = initialStepLengthM
        .takeIf { WalkingCalibrationPolicy.isValidStepLength(it) }
        ?: DEFAULT_STEP_LENGTH_M
    private var lastMotionTimestampNanos = 0L
    private var lastHardwareStepTimestampNanos = 0L
    private var lastDetectedStepTimestampNanos = 0L
    private var distanceM = 0f
    private var speedMps = 0f
    private var stepCount = 0
    private var accelerationBaseline = 0f
    private var accelerationBaselineInitialized = false
    private var stepPulseActive = false
    private var stepPulsePeak = 0f
    private var stepPulsePeakTimestampNanos = 0L

    fun reset() {
        lastMotionTimestampNanos = 0L
        lastHardwareStepTimestampNanos = 0L
        lastDetectedStepTimestampNanos = 0L
        distanceM = 0f
        speedMps = 0f
        stepCount = 0
        accelerationBaseline = 0f
        accelerationBaselineInitialized = false
        stepPulseActive = false
        stepPulsePeak = 0f
        stepPulsePeakTimestampNanos = 0L
    }

    /**
     * Applies a validated stride profile. Invalid values are rejected so a
     * malformed setting can never turn a short walk into a huge field length.
     */
    fun setStepLengthM(value: Float): Boolean {
        if (!WalkingCalibrationPolicy.isValidStepLength(value)) return false
        stepLengthM = value
        return true
    }

    val configuredStepLengthM: Float
        get() = stepLengthM

    fun onStep(timestampNanos: Long, eventValue: Float = 1f): MotionEstimate {
        if (timestampNanos <= 0L) return snapshot()
        val steps = eventValue.roundToInt().coerceAtLeast(1)
        stepCount += steps
        distanceM += steps * stepLengthM
        updateSpeed(timestampNanos, steps, STEP_SPEED_WEIGHT)
        lastHardwareStepTimestampNanos = timestampNanos
        lastDetectedStepTimestampNanos = timestampNanos
        lastMotionTimestampNanos = timestampNanos
        stepPulseActive = false
        stepPulsePeak = 0f
        stepPulsePeakTimestampNanos = 0L
        return snapshot()
    }

    fun onLinearAcceleration(timestampNanos: Long, x: Float, y: Float, z: Float = 0f): MotionEstimate {
        val previous = lastMotionTimestampNanos
        lastMotionTimestampNanos = timestampNanos
        if (timestampNanos <= 0L || previous <= 0L || timestampNanos <= previous) return snapshot()

        // Fresh step events already account for the same walking motion.  The
        // fallback resumes automatically once events go stale or never arrive.
        if (lastHardwareStepTimestampNanos > 0L &&
            timestampNanos - lastHardwareStepTimestampNanos <= stepFreshnessNanos
        ) {
            return snapshot()
        }

        val accelerationMagnitude = sqrt(x * x + y * y + z * z)
        if (!accelerationMagnitude.isFinite()) return snapshot()

        if (!accelerationBaselineInitialized) {
            accelerationBaseline = accelerationMagnitude
            accelerationBaselineInitialized = true
            return snapshot()
        }

        // A slow baseline removes device-specific bias while retaining the
        // short periodic impulses caused by walking. All three axes are used
        // because an upright phone often carries the signal on Z.
        accelerationBaseline += (accelerationMagnitude - accelerationBaseline) * BASELINE_WEIGHT
        val pulse = abs(accelerationMagnitude - accelerationBaseline)
        if (pulse >= STEP_PULSE_THRESHOLD) {
            if (!stepPulseActive) {
                stepPulseActive = true
                stepPulsePeak = pulse
                stepPulsePeakTimestampNanos = timestampNanos
            } else if (pulse > stepPulsePeak) {
                stepPulsePeak = pulse
                stepPulsePeakTimestampNanos = timestampNanos
            }
        } else if (stepPulseActive) {
            registerEstimatedStep(stepPulsePeakTimestampNanos)
            stepPulseActive = false
            stepPulsePeak = 0f
            stepPulsePeakTimestampNanos = 0L
        }
        return snapshot()
    }

    fun snapshot(): MotionEstimate = MotionEstimate(
        distanceM = distanceM.coerceAtLeast(0f),
        speedMps = speedMps.coerceIn(0f, MAX_SPEED_MPS),
        stepCount = stepCount,
        stepEventsObserved = lastHardwareStepTimestampNanos > 0L &&
            lastMotionTimestampNanos - lastHardwareStepTimestampNanos in 0..stepFreshnessNanos,
        motionEventsObserved = lastMotionTimestampNanos > 0L,
    )

    private fun registerEstimatedStep(timestampNanos: Long) {
        if (timestampNanos <= 0L) return
        val previous = lastDetectedStepTimestampNanos
        if (previous > 0L && timestampNanos - previous < MIN_STEP_INTERVAL_NANOS) return
        stepCount += 1
        distanceM += stepLengthM
        updateSpeed(timestampNanos, 1, ACCELERATION_SPEED_WEIGHT)
        lastDetectedStepTimestampNanos = timestampNanos
    }

    private fun updateSpeed(timestampNanos: Long, steps: Int, weight: Float) {
        val previous = lastDetectedStepTimestampNanos
        if (previous <= 0L || timestampNanos <= previous) return
        val elapsedSeconds = ((timestampNanos - previous) / 1_000_000_000f).coerceAtLeast(0.25f)
        val stepSpeed = (steps * stepLengthM / elapsedSeconds).coerceIn(0f, MAX_SPEED_MPS)
        speedMps = smooth(speedMps, stepSpeed, weight)
    }

    private fun smooth(current: Float, next: Float, weight: Float): Float =
        (current * (1f - weight) + next * weight).coerceIn(0f, MAX_SPEED_MPS)

    private companion object {
        const val DEFAULT_STEP_LENGTH_M = 0.72f
        const val STEP_FRESHNESS_NANOS = 1_500_000_000L
        const val MIN_STEP_INTERVAL_NANOS = 280_000_000L
        const val STEP_PULSE_THRESHOLD = 0.35f
        const val BASELINE_WEIGHT = 0.02f
        const val MAX_SPEED_MPS = 3f
        const val STEP_SPEED_WEIGHT = 0.35f
        const val ACCELERATION_SPEED_WEIGHT = 0.2f
    }
}
