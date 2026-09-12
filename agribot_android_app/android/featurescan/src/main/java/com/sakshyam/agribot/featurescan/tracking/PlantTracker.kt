package com.sakshyam.agribot.featurescan.tracking

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Looper
import android.os.SystemClock
import androidx.core.content.ContextCompat
import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.AnalysisFrame
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import com.sakshyam.agribot.domain.logic.MotionMeasurementInput
import com.sakshyam.agribot.domain.logic.MotionMeasurementResolver
import com.sakshyam.agribot.domain.model.GpsPathPoint
import com.sakshyam.agribot.domain.model.GpsPathPolicy
import com.sakshyam.agribot.domain.model.GpsPathStatus
import com.sakshyam.agribot.domain.model.MeasurementQuality
import com.sakshyam.agribot.domain.model.MeasurementSource
import com.sakshyam.agribot.domain.model.MotionMeasurementResult
import com.sakshyam.agribot.domain.model.TreatmentStatus
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Plant tracking system that identifies and tracks individual plants across frames
 * using multiple algorithms including IoU, cosine similarity, and spatial consistency.
 *
 * Features:
 * - IoU (Intersection over Union) tracking for spatial consistency
 * - Cosine similarity for feature-based re-identification after occlusion
 * - Unique plant ID generation system
 * - Stale plant removal (5+ seconds without detection)
 * - Sensor data integration for 3D positioning
 * - Occlusion handling with re-identification
 */
class PlantTracker(private val context: Context) : SensorEventListener {

    // Sensor fusion for tilt compensation and device-only distance estimation.
    private val sensorManager: SensorManager
    private val accelerometer: Sensor?
    private val linearAcceleration: Sensor?
    private val gyroscope: Sensor?
    private val rotationVector: Sensor?
    private val stepDetector: Sensor?
    private val locationManager: LocationManager
    private val motionEstimator = MotionDistanceEstimator()
    private val visualEstimator = VisualMotionEstimator()
    @Volatile private var attitudeSample: Pair<Long, FloatArray>? = null
    private var previousFrameAttitude: FloatArray? = null
    @Volatile var visualMotion: VisualMotion? = null
        private set

    /** Called for every analyzed frame, independent of classifier cooldown. */
    fun observeCameraMotion(frame: AnalysisFrame) {
        if (!trackingSessionActive || !tractorMounted) return
        val sample = attitudeSample
        val attitude = sample?.takeIf {
            kotlin.math.abs(frame.timestampNanos - it.first) <= 250_000_000L
        }?.second
        val previous = previousFrameAttitude
        val angle = if (attitude != null && previous != null) {
            // trace(R_previous^T R_current) gives angle independent of Euler wrap.
            val trace = attitude.indices.sumOf { (attitude[it] * previous[it]).toDouble() }
            kotlin.math.acos(((trace - 1.0) / 2.0).coerceIn(-1.0, 1.0)).toFloat()
        } else null
        previousFrameAttitude = attitude
        visualMotion = visualEstimator.observe(frame, angle)
        refreshResolvedMetrics()
    }
    private val registeredSensorTypes = mutableSetOf<Int>()
    private val registeredLocationProviders = mutableSetOf<String>()
    private val locationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            recordGpsLocation(location)
        }
    }
    private var sensorsRegistered = false
    private var trackingSessionActive = false
    private var tractorMounted = true
    private var locationReferenceEnabled = false
    private var lastGpsLocation: Location? = null
    private var gpsAnchorLocation: Location? = null
    private var gpsLastUpdateElapsedRealtimeNanos = 0L
    private var sensorDistanceM = 0f
    private var sensorSpeedMps = 0f
    private var gpsSpeedMps: Float? = null
    private var linearAccelerationEventsObserved = false
    private var lastLinearAccelerationEventNanos = 0L
    private var linearAccelerationSignalObserved = false
    private var lastLinearAccelerationSignalNanos = 0L
    private var sensorMotionEventsObserved = false
    private var sensorStepEventsObserved = false
    private var gpsFixAvailable = false
    @Volatile var gpsFixAgeSeconds: Float? = null
    private var gpsPointCount = 0
    private var gpsPathSequence = 0
    private var minLatitude = Double.POSITIVE_INFINITY
    private var maxLatitude = Double.NEGATIVE_INFINITY
    private var minLongitude = Double.POSITIVE_INFINITY
    private var maxLongitude = Double.NEGATIVE_INFINITY

    // Tracking state
    private val trackedPlants = mutableMapOf<String, TrackedPlant>()
    private val _trackedPlantsFlow = MutableStateFlow<List<TrackedPlant>>(emptyList())
    val trackedPlantsFlow: StateFlow<List<TrackedPlant>> = _trackedPlantsFlow.asStateFlow()

    // Sensor data
    private val gravity = FloatArray(3)
    private val geomagnetic = FloatArray(3)
    private val rotationMatrix = FloatArray(9)
    private val orientation = FloatArray(3)
    private var gravityInitialized = false

    @Volatile var currentTilt: Float = 0f
    @Volatile var currentRoll: Float = 0f
    @Volatile var currentPitch: Float = 0f
    @Volatile var currentSpeedMps: Float? = null
    @Volatile var currentDistanceM: Float? = null
    @Volatile var currentStepCount: Int = 0
    @Volatile var sensorQualityPercent: Int = 0
    @Volatile var motionStatus: String = "Motion sensors idle"
    @Volatile var gpsStatus: String = "GPS reference off"
    @Volatile var gpsAccuracyM: Float? = null
    @Volatile var gpsDistanceM: Float? = null
    @Volatile var gpsProvider: String? = null
    @Volatile var gpsPathPoints: List<GpsPathPoint> = emptyList()
    @Volatile var gpsPathStatus: GpsPathStatus = GpsPathStatus.UNAVAILABLE
    @Volatile var fieldWidthM: Float? = null
    @Volatile var fieldLengthM: Float? = null
    @Volatile var measurementSource: String = "No measurement yet"
    @Volatile var motionMeasurement: MotionMeasurementResult = MotionMeasurementResolver.notStarted()
    val stepSensorAvailable: Boolean
        get() = stepDetector != null
    val motionEventsObserved: Boolean
        get() = sensorMotionEventsObserved
    val configuredStepLengthM: Float
        get() = motionEstimator.configuredStepLengthM

    // Plant ID generator
    private var nextPlantId = 1

    init {
        sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        linearAcceleration = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        stepDetector = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR)
        locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        sensorQualityPercent = availableSensorQuality()
    }

    fun startTracking(useGpsReference: Boolean = true, tractorMounted: Boolean = true) {
        this.tractorMounted = tractorMounted
        trackingSessionActive = true
        locationReferenceEnabled = useGpsReference
        if (!sensorsRegistered) {
            // Keep raw acceleration registered as a bounded fallback even on
            // phones that advertise a linear-acceleration sensor. Some vendor
            // stacks expose the sensor but fail to deliver its events during a
            // camera session; waiting on catalogue presence leaves distance at
            // zero. The event watchdog below prevents double integration while
            // the primary stream is healthy.
            registerSensor(linearAcceleration, SensorManager.SENSOR_DELAY_GAME)
            registerSensor(accelerometer, SensorManager.SENSOR_DELAY_GAME)
            if (rotationVector != null) {
                registerSensor(rotationVector, SensorManager.SENSOR_DELAY_GAME)
            } else if (linearAcceleration != null && Sensor.TYPE_ACCELEROMETER !in registeredSensorTypes) {
                registerSensor(accelerometer, SensorManager.SENSOR_DELAY_GAME)
            }
            registerSensor(gyroscope, SensorManager.SENSOR_DELAY_GAME)
            if (!tractorMounted && hasActivityRecognitionPermission()) {
                registerSensor(stepDetector, SensorManager.SENSOR_DELAY_NORMAL)
            }
            sensorsRegistered = registeredSensorTypes.isNotEmpty()
            motionStatus = when {
                registeredSensorTypes.isEmpty() -> "No motion sensor available"
                linearAcceleration == null ||
                    Sensor.TYPE_LINEAR_ACCELERATION !in registeredSensorTypes ||
                    !linearAccelerationSignalObserved ->
                    "Motion active · acceleration fallback"
                else -> "Motion sensors active"
            }
            sensorQualityPercent = availableSensorQuality()
        }
        if (!tractorMounted && hasActivityRecognitionPermission() &&
            stepDetector != null &&
            Sensor.TYPE_STEP_DETECTOR !in registeredSensorTypes
        ) {
            if (registerSensor(stepDetector, SensorManager.SENSOR_DELAY_NORMAL)) {
                motionStatus = "Motion sensors active; step sensor enabled"
            }
        }
        if (useGpsReference) startLocationTracking() else stopLocationTracking()
        refreshResolvedMetrics()
    }

    fun setStepLengthM(value: Float): Boolean = motionEstimator.setStepLengthM(value)

    fun stopTracking() {
        trackingSessionActive = false
        if (sensorsRegistered) {
            sensorManager.unregisterListener(this)
            sensorsRegistered = false
        }
        registeredSensorTypes.clear()
        // Keep the last captured values for the results screen and export.
        // beginScan()/resetWalkMeasurement() are the explicit reset points;
        // stopping a sensor listener must not turn a completed walk into a
        // zero-valued snapshot during the next periodic UI refresh.
        motionMeasurement = MotionMeasurementResolver.retainAfterStop(motionMeasurement)
        motionStatus = motionMeasurement.statusMessage
        sensorQualityPercent = availableSensorQuality()
        measurementSource = motionMeasurement.source.displayName()
        stopLocationTracking(refreshMetrics = false)
    }

    fun setLocationReferenceEnabled(enabled: Boolean) {
        locationReferenceEnabled = enabled
        if (enabled && trackingSessionActive) startLocationTracking() else stopLocationTracking()
    }

    /** Start a fresh phone-sensor measurement session for the next field run. */
    fun beginScan() {
        visualEstimator.reset()
        previousFrameAttitude = null
        visualMotion = null
        motionEstimator.reset()
        sensorDistanceM = 0f
        sensorSpeedMps = 0f
        currentSpeedMps = null
        currentDistanceM = null
        currentStepCount = 0
        sensorMotionEventsObserved = false
        sensorStepEventsObserved = false
        linearAccelerationEventsObserved = false
        lastLinearAccelerationEventNanos = 0L
        linearAccelerationSignalObserved = false
        lastLinearAccelerationSignalNanos = 0L
        gravityInitialized = false
        motionMeasurement = MotionMeasurementResolver.notStarted()
        motionStatus = if (sensorsRegistered) "Motion sensors active" else "Motion sensors idle"
        resetGpsReference()
    }

    /**
     * Reset only the operator's walk measurement while preserving tracked plants.
     * A current GPS fix becomes the new anchor; it is never treated as distance
     * travelled until a later accepted point forms a segment.
     */
    fun resetWalkMeasurement() {
        motionEstimator.reset()
        sensorDistanceM = 0f
        sensorSpeedMps = 0f
        currentSpeedMps = null
        currentDistanceM = null
        currentStepCount = 0
        sensorMotionEventsObserved = false
        sensorStepEventsObserved = false
        linearAccelerationEventsObserved = false
        lastLinearAccelerationEventNanos = 0L
        linearAccelerationSignalObserved = false
        lastLinearAccelerationSignalNanos = 0L
        gravityInitialized = false

        val anchor = lastGpsLocation?.let(::Location)
        if (anchor == null) {
            resetGpsReference()
        } else {
            gpsAnchorLocation = Location(anchor)
            gpsDistanceM = 0f
            gpsSpeedMps = null
            gpsFixAvailable = true
            gpsPointCount = 1
            minLatitude = anchor.latitude
            maxLatitude = anchor.latitude
            minLongitude = anchor.longitude
            maxLongitude = anchor.longitude
            fieldWidthM = null
            fieldLengthM = null
            gpsAccuracyM = anchor.accuracy.takeIf { it.isFinite() }
            gpsLastUpdateElapsedRealtimeNanos = anchor.elapsedRealtimeNanos.takeIf { it > 0L }
                ?: SystemClock.elapsedRealtimeNanos()
            gpsFixAgeSeconds = null
            gpsPathSequence = 1
            gpsPathPoints = listOf(
                GpsPathPoint(
                    sequence = gpsPathSequence,
                    eastM = 0.0,
                    northM = 0.0,
                    accuracyM = gpsAccuracyM,
                    provider = anchor.provider?.takeIf { it.isNotBlank() },
                    elapsedRealtimeNanos = gpsLastUpdateElapsedRealtimeNanos,
                ),
            )
            gpsProvider = anchor.provider?.takeIf { it.isNotBlank() }
            gpsPathStatus = GpsPathStatus.ANCHOR_ONLY
            gpsStatus = "GPS reference reset; waiting for next point"
        }
        motionMeasurement = MotionMeasurementResolver.notStarted()
        refreshResolvedMetrics()
    }

    private fun startLocationTracking() {
        if (!locationReferenceEnabled) return
        if (!hasLocationPermission()) {
            gpsStatus = "Location permission needed"
            refreshResolvedMetrics()
            return
        }
        val enabledProviders = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)
            .filter { provider -> runCatching { locationManager.isProviderEnabled(provider) }.getOrDefault(false) }
        if (enabledProviders.isEmpty()) {
            gpsStatus = "Location providers disabled"
            refreshResolvedMetrics()
            return
        }
        try {
            registeredLocationProviders.clear()
            for (provider in enabledProviders) {
                runCatching {
                    locationManager.requestLocationUpdates(
                        provider,
                        GPS_UPDATE_INTERVAL_MS,
                        GPS_MIN_DISTANCE_M,
                        locationListener,
                        Looper.getMainLooper(),
                    )
                    registeredLocationProviders += provider
                }
                locationManager.getLastKnownLocation(provider)?.let(::recordGpsLocation)
            }
            if (registeredLocationProviders.isEmpty()) {
                gpsStatus = "Location unavailable"
            } else if (gpsStatus != "GPS reference active" && gpsStatus != "GPS weak") {
                gpsStatus = "Waiting for location fix"
            }
        } catch (_: SecurityException) {
            gpsStatus = "Location permission needed"
            refreshResolvedMetrics()
        } catch (_: IllegalArgumentException) {
            gpsStatus = "GPS unavailable"
            refreshResolvedMetrics()
        }
    }

    private fun stopLocationTracking(refreshMetrics: Boolean = true) {
        runCatching { locationManager.removeUpdates(locationListener) }
        registeredLocationProviders.clear()
        gpsStatus = "GPS reference off"
        gpsPathStatus = when {
            gpsPathPoints.size >= 2 -> GpsPathStatus.REFERENCE_PATH
            gpsPathPoints.size == 1 -> GpsPathStatus.ANCHOR_ONLY
            trackingSessionActive -> GpsPathStatus.MOTION_ONLY
            else -> GpsPathStatus.UNAVAILABLE
        }
        if (refreshMetrics) {
            measurementSource = "No measurement yet"
            refreshResolvedMetrics()
        }
    }

    private fun resetGpsReference() {
        lastGpsLocation = null
        gpsAnchorLocation = null
        gpsLastUpdateElapsedRealtimeNanos = 0L
        gpsDistanceM = null
        gpsAccuracyM = null
        gpsSpeedMps = null
        gpsFixAvailable = false
        gpsFixAgeSeconds = null
        gpsPointCount = 0
        gpsPathSequence = 0
        gpsPathPoints = emptyList()
        gpsProvider = null
        minLatitude = Double.POSITIVE_INFINITY
        maxLatitude = Double.NEGATIVE_INFINITY
        minLongitude = Double.POSITIVE_INFINITY
        maxLongitude = Double.NEGATIVE_INFINITY
        fieldWidthM = null
        fieldLengthM = null
        gpsStatus = when {
            !locationReferenceEnabled -> "GPS reference off"
            !hasLocationPermission() -> "Location permission needed"
            !hasEnabledLocationProvider() -> "Location providers disabled"
            else -> "Waiting for GPS"
        }
        gpsPathStatus = when {
            !locationReferenceEnabled && trackingSessionActive -> GpsPathStatus.MOTION_ONLY
            !locationReferenceEnabled -> GpsPathStatus.UNAVAILABLE
            else -> GpsPathStatus.WAITING_FOR_FIX
        }
        measurementSource = "No measurement yet"
    }

    private fun recordGpsLocation(location: Location) {
        if (!location.latitude.isFinite() || !location.longitude.isFinite()) {
            gpsStatus = "GPS coordinate rejected"
            refreshResolvedMetrics()
            return
        }
        if (!location.hasAccuracy() || location.accuracy > GPS_MAX_ACCEPTED_ACCURACY_M) {
            gpsAccuracyM = location.accuracy.takeIf { location.hasAccuracy() && it.isFinite() }
            gpsProvider = location.provider?.takeIf { it.isNotBlank() }
            gpsStatus = "GPS weak"
            refreshResolvedMetrics()
            return
        }
        val previous = lastGpsLocation
        if (previous != null) {
            val segmentM = previous.distanceTo(location)
            val elapsedSeconds = ((location.elapsedRealtimeNanos - previous.elapsedRealtimeNanos) / 1_000_000_000f)
                .coerceAtLeast(0.25f)
            if (segmentM > MAX_GPS_SEGMENT_M) {
                gpsStatus = "GPS jump ignored"
                refreshResolvedMetrics()
                return
            }
            if (segmentM >= GPS_MIN_DISTANCE_M) {
                gpsDistanceM = (gpsDistanceM ?: 0f) + segmentM
            }
            val reportedSpeed = if (location.hasSpeed()) location.speed else segmentM / elapsedSeconds
            gpsSpeedMps = reportedSpeed.takeIf { it.isFinite() && it >= 0f }?.coerceIn(0f, 4f)
        } else {
            // A first accepted fix is a valid zero-length reference, not an
            // unavailable value. Subsequent fixes extend this path.
            gpsDistanceM = 0f
            gpsSpeedMps = location.speed.takeIf {
                location.hasSpeed() && it.isFinite() && it >= 0f
            }?.coerceIn(0f, 4f)
        }
        lastGpsLocation = Location(location)
        gpsProvider = location.provider?.takeIf { it.isNotBlank() }
        gpsLastUpdateElapsedRealtimeNanos = if (location.elapsedRealtimeNanos > 0L) {
            location.elapsedRealtimeNanos
        } else {
            SystemClock.elapsedRealtimeNanos()
        }
        gpsAccuracyM = location.accuracy.takeIf { it.isFinite() }
        gpsFixAvailable = true
        gpsPointCount += 1
        minLatitude = minOf(minLatitude, location.latitude)
        maxLatitude = maxOf(maxLatitude, location.latitude)
        minLongitude = minOf(minLongitude, location.longitude)
        maxLongitude = maxOf(maxLongitude, location.longitude)
        updateFieldEnvelope()
        if (gpsAnchorLocation == null) {
            gpsAnchorLocation = Location(location)
            gpsPathSequence = 1
            gpsPathPoints = listOf(
                GpsPathPoint(
                    sequence = gpsPathSequence,
                    eastM = 0.0,
                    northM = 0.0,
                    accuracyM = gpsAccuracyM,
                    provider = gpsProvider,
                    elapsedRealtimeNanos = gpsLastUpdateElapsedRealtimeNanos,
                ),
            )
        } else if (previous != null && previous.distanceTo(location) >= GPS_MIN_DISTANCE_M) {
            val anchor = gpsAnchorLocation
            val next = anchor?.let {
                GpsPathPolicy.project(
                    sequence = gpsPathSequence + 1,
                    originLatitude = it.latitude,
                    originLongitude = it.longitude,
                    latitude = location.latitude,
                    longitude = location.longitude,
                    accuracyM = gpsAccuracyM,
                    provider = gpsProvider,
                    elapsedRealtimeNanos = gpsLastUpdateElapsedRealtimeNanos,
                )
            }
            if (next != null) {
                gpsPathPoints = GpsPathPolicy.appendBounded(gpsPathPoints, next)
                if (gpsPathPoints.lastOrNull()?.sequence == next.sequence) {
                    gpsPathSequence = next.sequence
                }
            }
        }
        gpsPathStatus = when {
            gpsPathPoints.size >= 2 -> GpsPathStatus.REFERENCE_PATH
            gpsPathPoints.size == 1 -> GpsPathStatus.ANCHOR_ONLY
            else -> GpsPathStatus.WAITING_FOR_FIX
        }
        gpsStatus = if (location.accuracy <= GPS_REFERENCE_ACCURACY_M) {
            "GPS reference active"
        } else {
            "GPS weak"
        }
        refreshResolvedMetrics()
    }

    private fun updateFieldEnvelope() {
        if (!minLatitude.isFinite() || !minLongitude.isFinite()) return
        val northSouth = FloatArray(3)
        val eastWest = FloatArray(3)
        Location.distanceBetween(minLatitude, minLongitude, maxLatitude, minLongitude, northSouth)
        Location.distanceBetween(minLatitude, minLongitude, minLatitude, maxLongitude, eastWest)
        if (gpsPointCount >= 2) {
            fieldLengthM = northSouth[0].takeIf { it.isFinite() && it >= MIN_FIELD_ENVELOPE_M }
            fieldWidthM = eastWest[0].takeIf { it.isFinite() && it >= MIN_FIELD_ENVELOPE_M }
        }
    }

    private fun hasLocationPermission(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

    private fun hasActivityRecognitionPermission(): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.Q ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACTIVITY_RECOGNITION) == PackageManager.PERMISSION_GRANTED

    private fun hasEnabledLocationProvider(): Boolean =
        listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)
            .any { provider -> runCatching { locationManager.isProviderEnabled(provider) }.getOrDefault(false) }

    private fun gpsIsReliable(): Boolean {
        if (!locationReferenceEnabled) {
            if (trackingSessionActive && gpsPathPoints.isEmpty()) gpsPathStatus = GpsPathStatus.MOTION_ONLY
            return false
        }
        if (!gpsFixAvailable) {
            if (gpsPathPoints.isEmpty()) {
                gpsPathStatus = if (trackingSessionActive) {
                    GpsPathStatus.WAITING_FOR_FIX
                } else {
                    GpsPathStatus.UNAVAILABLE
                }
            }
            return false
        }
        val accuracy = gpsAccuracyM ?: return false
        if (accuracy <= 0f || accuracy > GPS_REFERENCE_ACCURACY_M) return false
        val ageNanos = SystemClock.elapsedRealtimeNanos() - gpsLastUpdateElapsedRealtimeNanos
        val fresh = ageNanos in 0..GPS_STALE_AFTER_NANOS
        gpsFixAgeSeconds = if (ageNanos >= 0L) ageNanos / 1_000_000_000f else null
        gpsPathStatus = if (fresh) {
            when {
                gpsPathPoints.size >= 2 -> GpsPathStatus.REFERENCE_PATH
                gpsPathPoints.size == 1 -> GpsPathStatus.ANCHOR_ONLY
                else -> GpsPathStatus.WAITING_FOR_FIX
            }
        } else if (gpsPathPoints.isNotEmpty()) {
            GpsPathStatus.STALE
        } else {
            GpsPathStatus.WAITING_FOR_FIX
        }
        return fresh
    }

    private fun refreshResolvedMetrics() {
        gpsIsReliable()
        val result = MotionMeasurementResolver.resolve(
            MotionMeasurementInput(
                trackingActive = trackingSessionActive,
                motionEventsObserved = sensorMotionEventsObserved,
                stepEventsObserved = sensorStepEventsObserved,
                sensorDistanceM = sensorDistanceM,
                sensorSpeedMps = sensorSpeedMps,
                stepCount = currentStepCount,
                gpsReferenceEnabled = locationReferenceEnabled,
                gpsFixAvailable = gpsFixAvailable,
                gpsDistanceM = gpsDistanceM,
                gpsSpeedMps = gpsSpeedMps,
                gpsAccuracyM = gpsAccuracyM,
                gpsFixAgeSeconds = gpsFixAgeSeconds,
                tractorMounted = tractorMounted,
            ),
        )
        motionMeasurement = result
        currentDistanceM = result.distance.value
        currentSpeedMps = result.speed.value
        measurementSource = result.source.displayName()
        motionStatus = if (tractorMounted && visualMotion != null) {
            "${visualMotion!!.reason} · ${result.statusMessage}"
        } else result.statusMessage
    }

    /**
     * Clear all tracked plants. Used when starting a new scan or clearing data.
     */
    fun clearAll() {
        trackedPlants.clear()
        _trackedPlantsFlow.value = emptyList()
        nextPlantId = 1
        beginScan()
    }

    /**
     * Set treatment status and notes for a tracked plant.
     */
    fun setTreatment(plantId: String, status: TreatmentStatus, note: String?) {
        val plant = trackedPlants[plantId] ?: return
        trackedPlants[plantId] = plant.copy(
            treatmentStatus = status,
            treatmentNote = note,
        )
        _trackedPlantsFlow.value = trackedPlants.values.toList()
    }

    /**
     * Process new frame detections and update plant tracking.
     *
     * Algorithm:
     * 1. For each new detection, try to match with existing tracked plants using:
     *    - IoU (Intersection over Union) for spatial overlap
     *    - Cosine similarity for feature-based matching (re-identification after occlusion)
     *    - Spatial distance score for proximity
     * 2. Unmatched detections become new tracked plants
     * 3. Existing plants not seen in this frame are marked as occluded (not immediately removed)
     * 4. Plants not seen for 5+ seconds are removed
     */
    fun processFrameDetections(
        candidates: List<FrontOverviewCandidate>,
        frameWidth: Int,
        frameHeight: Int,
        timestamp: Long
    ): List<TrackedPlant> {
        val currentPlants = mutableListOf<TrackedPlant>()
        val matchedIndices = mutableSetOf<Int>()
        val matchedPlantIds = mutableSetOf<String>()

        // Try to match new detections with existing plants
        for ((candidateIndex, candidate) in candidates.withIndex()) {
            var bestMatch: TrackedPlant? = null
            var bestMatchScore = 0f

            // Find best matching existing plant
            for (existingPlant in trackedPlants.values) {
                if (existingPlant.plantId in matchedPlantIds) continue
                val similarityScore = PlantTrackingMath.calculateMatchScore(
                    existingBbox = existingPlant.currentBbox,
                    newBbox = candidate.bboxPx,
                    existingFeatures = existingPlant.features,
                    newFeatures = PlantTrackingMath.extractFeatures(candidate.bboxPx, candidate.confidence),
                )

                if (similarityScore > bestMatchScore && similarityScore > 0.35f) {
                    bestMatch = existingPlant
                    bestMatchScore = similarityScore
                }
            }

            if (bestMatch != null) {
                // Update existing plant
                val measurement = estimateMeasurement(candidate.bboxPx, frameWidth, frameHeight)
                val updatedPlant = bestMatch.updateWithDetection(
                    candidate = candidate,
                    timestamp = timestamp,
                    tilt = currentTilt,
                    roll = currentRoll,
                    pitch = currentPitch,
                    distanceM = measurement.distanceM,
                    widthM = measurement.widthM,
                    heightM = measurement.heightM,
                    relativeWidth = measurement.relativeWidth,
                    relativeHeight = measurement.relativeHeight,
                    sizeQuality = measurement.sizeQuality,
                    sizeSource = measurement.sizeSource,
                    measurementSource = measurement.measurementSource,
                    positionZ = measurement.distanceFromCameraM,
                )
                trackedPlants[bestMatch.plantId] = updatedPlant
                currentPlants.add(updatedPlant)
                matchedIndices.add(candidateIndex)
                matchedPlantIds.add(bestMatch.plantId)
            }
        }

        // Add new plants for unmatched detections
        for ((candidateIndex, candidate) in candidates.withIndex()) {
            if (candidateIndex !in matchedIndices) {
                val newPlant = createNewPlant(candidate, timestamp, frameWidth, frameHeight)
                trackedPlants[newPlant.plantId] = newPlant
                currentPlants.add(newPlant)
            }
        }

        // Mark existing plants that were not matched as occluded (but don't remove yet)
        val currentPlantIds = currentPlants.map { it.plantId }.toSet()
        for ((plantId, plant) in trackedPlants) {
            if (plantId !in currentPlantIds) {
                val updatedPlant = plant.markOccluded(timestamp)
                trackedPlants[plantId] = updatedPlant
            }
        }

        // Remove plants not seen for too long
        removeStalePlants(timestamp)

        // Update flow
        _trackedPlantsFlow.value = trackedPlants.values.toList()

        return currentPlants
    }

    private fun createNewPlant(
        candidate: FrontOverviewCandidate,
        timestamp: Long,
        frameWidth: Int,
        frameHeight: Int
    ): TrackedPlant {
        val plantId = generatePlantId()
        val measurement = estimateMeasurement(candidate.bboxPx, frameWidth, frameHeight)
        val position3d = map2DTo3D(candidate.bboxPx, measurement)

        return TrackedPlant(
            plantId = plantId,
            createdAt = timestamp,
            lastSeen = timestamp,
            currentBbox = candidate.bboxPx,
            healthStatus = "Uncertain",
            pendingHealthLabel = candidate.label.takeUnless { it.isUncertainHealthLabel() },
            pendingHealthLabelStreak = if (candidate.label.isUncertainHealthLabel()) 0 else 1,
            classificationPending = candidate.rawLabel.equals("detector_only", ignoreCase = true),
            confidence = candidate.confidence,
            positionHistory = listOf(position3d),
            detectionCount = 1,
            features = PlantTrackingMath.extractFeatures(
                candidate.bboxPx,
                if (candidate.rawLabel.equals("detector_only", ignoreCase = true)) {
                    candidate.detectorConfidence
                } else {
                    candidate.confidence
                },
            ),
            isOccluded = false,
            occlusionStartTime = 0L,
            distanceM = measurement.distanceM,
            widthM = measurement.widthM,
            heightM = measurement.heightM,
            relativeWidth = measurement.relativeWidth,
            relativeHeight = measurement.relativeHeight,
            sizeQuality = measurement.sizeQuality,
            sizeSource = measurement.sizeSource,
            measurementSource = measurement.measurementSource,
        )
    }

    private fun generatePlantId(): String {
        return "plant_${nextPlantId++}"
    }

    private fun removeStalePlants(currentTimestamp: Long) {
        val staleThreshold = 5000L // 5 seconds
        val stalePlantIds = trackedPlants.values
            .filter { currentTimestamp - it.lastSeen > staleThreshold }
            .map { it.plantId }

        stalePlantIds.forEach { plantId ->
            trackedPlants.remove(plantId)
        }
    }

    /**
     * Map 2D screen coordinates to 3D world coordinates using sensor data.
     * Uses tilt, roll, and pitch to compensate for camera angle.
     */
    private fun map2DTo3D(
        bbox: BoundingBox,
        measurement: SensorPlantMeasurement,
    ): PlantPosition3D {
        val z3d = measurement.distanceFromCameraM

        return PlantPosition3D(
            positionX = bbox.centerX,
            positionY = bbox.centerY,
            positionZ = z3d,
            tilt = currentTilt,
            roll = currentRoll,
            pitch = currentPitch,
            timestamp = System.currentTimeMillis(),
            distanceM = measurement.distanceM,
        )
    }

    /**
     * Derive all run measurements locally. There is no network or backend
     * dependency in this path: distance comes from GPS when accurate and
     * otherwise the step/motion sensors, count comes from detector tracks, and
     * dimensions come from the camera frame scaled by the local motion/GPS reference.
     */
    private fun estimateMeasurement(
        bbox: BoundingBox,
        frameWidth: Int,
        frameHeight: Int,
    ): SensorPlantMeasurement {
        val safeWidth = frameWidth.coerceAtLeast(1)
        val safeHeight = frameHeight.coerceAtLeast(1)
        // A phone's walk distance and GPS envelope cannot recover an individual
        // plant's physical width or height. Until a reference-object or camera
        // calibration is completed, expose only a relative image measurement.
        val relativeWidth = (bbox.width / safeWidth).coerceIn(0f, 1f)
        val relativeHeight = (bbox.height / safeHeight).coerceIn(0f, 1f)
        return SensorPlantMeasurement(
            distanceM = motionMeasurement.distance.value,
            widthM = null,
            heightM = null,
            relativeWidth = relativeWidth,
            relativeHeight = relativeHeight,
            distanceFromCameraM = null,
            sizeQuality = MeasurementQuality.RELATIVE,
            sizeSource = MeasurementSource.CAMERA_RELATIVE,
            measurementSource = motionMeasurement.source.displayName(),
        )
    }

    // SensorEventListener implementation
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR -> {
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                attitudeSample = event.timestamp to rotationMatrix.copyOf()
                SensorManager.getOrientation(rotationMatrix, orientation)
                currentRoll = orientation[2]
                currentPitch = orientation[1]
                currentTilt = sqrt(orientation[1] * orientation[1] + orientation[2] * orientation[2])
                sensorQualityPercent = availableSensorQuality()
            }
            Sensor.TYPE_STEP_DETECTOR -> {
                if (tractorMounted) return
                applyMotionEstimate(motionEstimator.onStep(event.timestamp, event.values.firstOrNull() ?: 1f))
            }
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                if (tractorMounted) return
                linearAccelerationEventsObserved = true
                lastLinearAccelerationEventNanos = event.timestamp
                val linearX = event.values.getOrElse(0) { 0f }
                val linearY = event.values.getOrElse(1) { 0f }
                val linearZ = event.values.getOrElse(2) { 0f }
                val linearMagnitude = sqrt(linearX * linearX + linearY * linearY + linearZ * linearZ)
                if (linearMagnitude.isFinite() && linearMagnitude >= LINEAR_ACCELERATION_SIGNAL_FLOOR) {
                    linearAccelerationSignalObserved = true
                    lastLinearAccelerationSignalNanos = event.timestamp
                }
                applyMotionEstimate(
                    motionEstimator.onLinearAcceleration(
                        timestampNanos = event.timestamp,
                        x = linearX,
                        y = linearY,
                        z = linearZ,
                    ),
                )
            }
            Sensor.TYPE_ACCELEROMETER -> {
                val rawX = event.values.getOrElse(0) { 0f }
                val rawY = event.values.getOrElse(1) { 0f }
                val rawZ = event.values.getOrElse(2) { 0f }
                // Low-pass filter for gravity
                val alpha = 0.8f
                if (!gravityInitialized) {
                    // Seed the gravity estimate from the first sample. Starting
                    // the low-pass filter at zero creates a fake acceleration
                    // spike on every new scan and can add a phantom step.
                    gravity[0] = rawX
                    gravity[1] = rawY
                    gravity[2] = rawZ
                    gravityInitialized = true
                } else {
                    gravity[0] = alpha * gravity[0] + (1 - alpha) * rawX
                    gravity[1] = alpha * gravity[1] + (1 - alpha) * rawY
                    gravity[2] = alpha * gravity[2] + (1 - alpha) * rawZ
                }

                if (rotationVector == null) {
                    SensorManager.getRotationMatrix(rotationMatrix, null, gravity, geomagnetic)
                    SensorManager.getOrientation(rotationMatrix, orientation)
                    currentRoll = orientation[2]
                    currentPitch = orientation[1]
                    currentTilt = sqrt(orientation[1] * orientation[1] + orientation[2] * orientation[2])
                }
                val linearAccelerationStale = !linearAccelerationEventsObserved ||
                    event.timestamp - lastLinearAccelerationEventNanos > LINEAR_ACCELERATION_STALE_NANOS
                val linearAccelerationSignalStale = !linearAccelerationSignalObserved ||
                    event.timestamp - lastLinearAccelerationSignalNanos > LINEAR_ACCELERATION_STALE_NANOS
                if (!tractorMounted && (linearAcceleration == null || linearAccelerationStale || linearAccelerationSignalStale)) {
                    applyMotionEstimate(
                        motionEstimator.onLinearAcceleration(
                            timestampNanos = event.timestamp,
                            x = rawX - gravity[0],
                            y = rawY - gravity[1],
                            z = rawZ - gravity[2],
                        ),
                    )
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                // Gyroscope data can be used for more precise rotation tracking
                // Currently used for sensor fusion with accelerometer
                // The orientation is primarily computed from accelerometer + geomagnetic
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        sensorQualityPercent = availableSensorQuality()
    }

    private fun applyMotionEstimate(estimate: MotionEstimate) {
        sensorDistanceM = estimate.distanceM
        sensorSpeedMps = estimate.speedMps
        currentStepCount = estimate.stepCount
        sensorMotionEventsObserved = estimate.motionEventsObserved
        sensorStepEventsObserved = estimate.stepEventsObserved
        sensorQualityPercent = availableSensorQuality()
        refreshResolvedMetrics()
    }

    private fun registerSensor(sensor: Sensor?, rate: Int): Boolean {
        if (sensor == null) return false
        val registered = runCatching { sensorManager.registerListener(this, sensor, rate) }
            .getOrDefault(false)
        if (registered) registeredSensorTypes += sensor.type
        return registered
    }

    private fun availableSensorQuality(): Int {
        if (!trackingSessionActive) return 0
        if (!sensorMotionEventsObserved && !sensorStepEventsObserved) return 0
        val base = (registeredSensorTypes.size * 25).coerceAtMost(100)
        val stabilityPenalty = (abs(currentTilt) / 1.4f * 20f).roundToInt().coerceIn(0, 20)
        return (base - stabilityPenalty).coerceIn(0, 100)
    }

    private data class SensorPlantMeasurement(
        val distanceM: Float?,
        val widthM: Float?,
        val heightM: Float?,
        val relativeWidth: Float,
        val relativeHeight: Float,
        val distanceFromCameraM: Float?,
        val sizeQuality: MeasurementQuality,
        val sizeSource: MeasurementSource,
        val measurementSource: String,
    )

    private companion object {
        const val GPS_UPDATE_INTERVAL_MS = 1_000L
        const val GPS_MIN_DISTANCE_M = 0.5f
        const val GPS_MAX_ACCEPTED_ACCURACY_M = 50f
        const val GPS_REFERENCE_ACCURACY_M = 25f
        const val GPS_STALE_AFTER_NANOS = 5_000_000_000L
        const val LINEAR_ACCELERATION_STALE_NANOS = 2_000_000_000L
        const val LINEAR_ACCELERATION_SIGNAL_FLOOR = 0.08f
        const val MAX_GPS_SEGMENT_M = 25f
        const val MIN_FIELD_ENVELOPE_M = 0.5f
    }
}

private fun MeasurementSource.displayName(): String = when (this) {
    MeasurementSource.NONE -> "No measurement yet"
    MeasurementSource.PHONE_STEP_SENSOR -> "Phone step sensor"
    MeasurementSource.PHONE_ACCELERATION -> "Phone acceleration"
    MeasurementSource.PHONE_SENSORS -> "Phone motion sensors"
    MeasurementSource.GPS_REFERENCE -> "GPS field reference"
    MeasurementSource.GPS_AND_PHONE_SENSORS -> "GPS reference + phone sensors"
    MeasurementSource.CAMERA_RELATIVE -> "Camera relative size"
    MeasurementSource.CAMERA_CALIBRATED -> "Calibrated camera size"
}

/**
 * Data class representing a tracked plant with full history.
 *
 * @property plantId Unique identifier for the plant
 * @property createdAt Timestamp when the plant was first detected
 * @property lastSeen Timestamp when the plant was last seen
 * @property currentBbox Current bounding box of the plant
 * @property healthStatus Current health status (label from classifier)
 * @property classificationPending A detector box waiting for a classifier crop; not a disease result
 * @property confidence Confidence score of the detection
 * @property positionHistory List of 3D positions over time
 * @property detectionCount Number of times this plant has been detected
 * @property features Feature vector for cosine similarity matching
 * @property isOccluded Whether the plant is currently occluded (not detected in last frame)
 * @property occlusionStartTime Timestamp when occlusion started
 */
data class TrackedPlant(
    val plantId: String,
    val createdAt: Long,
    val lastSeen: Long,
    val currentBbox: BoundingBox,
    val healthStatus: String,
    val classificationPending: Boolean = false,
    val confidence: Float,
    val positionHistory: List<PlantPosition3D>,
    val detectionCount: Int,
    val features: FloatArray,
    val isOccluded: Boolean = false,
    val occlusionStartTime: Long = 0L,
    val treatmentStatus: TreatmentStatus = TreatmentStatus.NOT_TREATED,
    val treatmentNote: String? = null,
    /** A competing label must be observed twice before replacing a stable label. */
    val pendingHealthLabel: String? = null,
    val pendingHealthLabelStreak: Int = 0,
    val uncertainObservationStreak: Int = 0,
        val distanceM: Float? = null,
        val widthM: Float? = null,
        val heightM: Float? = null,
        val relativeWidth: Float? = null,
        val relativeHeight: Float? = null,
        val sizeQuality: MeasurementQuality = MeasurementQuality.UNAVAILABLE,
        val sizeSource: MeasurementSource = MeasurementSource.NONE,
        val measurementSource: String = "Measurement unavailable",
        val top2Margin: Float? = null,
        val predictionEntropy: Float? = null,
) {
    /** A physical count is published only after the same track is seen twice. */
    val isConfirmed: Boolean
        get() = detectionCount >= 2 && !classificationPending

    fun updateWithDetection(
        candidate: FrontOverviewCandidate,
        timestamp: Long,
        tilt: Float,
        roll: Float,
        pitch: Float,
        distanceM: Float? = this.distanceM,
        widthM: Float? = this.widthM,
        heightM: Float? = this.heightM,
        relativeWidth: Float? = this.relativeWidth,
        relativeHeight: Float? = this.relativeHeight,
        sizeQuality: MeasurementQuality = this.sizeQuality,
        sizeSource: MeasurementSource = this.sizeSource,
        measurementSource: String = this.measurementSource,
        positionZ: Float? = positionHistory.lastOrNull()?.positionZ,
    ): TrackedPlant {
        // Re-delivery of a camera frame is not independent evidence.
        if (timestamp <= lastSeen) return this
        val updatedPosition = PlantPosition3D(
            positionX = candidate.bboxPx.centerX,
            positionY = candidate.bboxPx.centerY,
            positionZ = positionZ,
            tilt = tilt,
            roll = roll,
            pitch = pitch,
            timestamp = timestamp,
            distanceM = distanceM,
        )

        val updatedHistory = (positionHistory.takeLast(119) + updatedPosition)
        val detectorOnly = candidate.rawLabel.equals("detector_only", ignoreCase = true)
        val stableClassification = resolveClassification(candidate, detectorOnly)

        return copy(
            lastSeen = timestamp,
            currentBbox = candidate.bboxPx,
            healthStatus = stableClassification.label,
            classificationPending = if (detectorOnly) classificationPending else false,
            // Confidence belongs to the currently displayed label. A transient
            // uncertain/contradictory frame must not overwrite a stable result.
            confidence = stableClassification.confidence,
            positionHistory = updatedHistory,
            detectionCount = detectionCount + 1,
            features = extractUpdatedFeatures(candidate, detectorOnly),
            isOccluded = false,
            occlusionStartTime = 0L,
            distanceM = distanceM,
            widthM = widthM,
            heightM = heightM,
            relativeWidth = relativeWidth,
            relativeHeight = relativeHeight,
            sizeQuality = sizeQuality,
            sizeSource = sizeSource,
            measurementSource = measurementSource,
            top2Margin = if (detectorOnly) top2Margin else candidate.top2Margin,
            predictionEntropy = if (detectorOnly) predictionEntropy else candidate.entropy,
            pendingHealthLabel = stableClassification.pendingLabel,
            pendingHealthLabelStreak = stableClassification.pendingStreak,
            uncertainObservationStreak = if (!detectorOnly && candidate.label.isUncertainHealthLabel()) {
                uncertainObservationStreak + 1
            } else if (detectorOnly) uncertainObservationStreak else 0,
        )
    }

    private fun resolveClassification(
        candidate: FrontOverviewCandidate,
        detectorOnly: Boolean,
    ): StableClassification {
        if (detectorOnly) {
            return StableClassification(healthStatus, confidence, pendingHealthLabel, pendingHealthLabelStreak)
        }

        val incoming = candidate.label.trim().ifBlank { UNCERTAIN_LABEL }
        val current = healthStatus.trim().ifBlank { UNCERTAIN_LABEL }
        if (incoming.isUncertainHealthLabel()) {
            return if (current.isUncertainHealthLabel() || uncertainObservationStreak >= 2) {
                StableClassification(incoming, candidate.confidence, null, 0)
            } else {
                StableClassification(current, confidence, null, 0)
            }
        }
        if (current.equals(incoming, ignoreCase = true)) {
            return StableClassification(incoming, candidate.confidence, null, 0)
        }

        val nextStreak = if (pendingHealthLabel.equals(incoming, ignoreCase = true)) {
            pendingHealthLabelStreak + 1
        } else {
            1
        }
        return if (nextStreak >= REQUIRED_LABEL_OBSERVATIONS) {
            StableClassification(incoming, candidate.confidence, null, 0)
        } else {
            StableClassification(current, confidence, incoming, nextStreak)
        }
    }

    private data class StableClassification(
        val label: String,
        val confidence: Float,
        val pendingLabel: String?,
        val pendingStreak: Int,
    )

    private companion object {
        const val REQUIRED_LABEL_OBSERVATIONS = 2
        const val UNCERTAIN_LABEL = "Uncertain"
    }

    /**
     * Mark this plant as occluded (not detected in the current frame).
     * The plant is kept in the tracking system for re-identification.
     */
    fun markOccluded(timestamp: Long): TrackedPlant {
        return copy(
            isOccluded = true,
            occlusionStartTime = if (occlusionStartTime == 0L) timestamp else occlusionStartTime,
        )
    }

    private fun extractUpdatedFeatures(candidate: FrontOverviewCandidate, detectorOnly: Boolean): FloatArray {
        val featureConfidence = if (detectorOnly) candidate.detectorConfidence else candidate.confidence
        return PlantTrackingMath.extractFeatures(candidate.bboxPx, featureConfidence)
    }
}

private fun String.isUncertainHealthLabel(): Boolean =
    isBlank() || equals("Uncertain", ignoreCase = true) || equals("Unknown", ignoreCase = true)

/**
 * 3D position data for a plant detection.
 */
data class PlantPosition3D(
    val positionX: Float,
    val positionY: Float,
    val positionZ: Float?,
    val tilt: Float,
    val roll: Float,
    val pitch: Float,
    val timestamp: Long,
    val distanceM: Float? = null,
)
