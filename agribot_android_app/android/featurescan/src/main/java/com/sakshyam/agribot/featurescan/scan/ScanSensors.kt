package com.sakshyam.agribot.featurescan.scan

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Looper
import androidx.core.content.ContextCompat
import com.sakshyam.agribot.domain.logic.FrameQualityAnalyzer
import com.sakshyam.agribot.domain.logic.FrameQualityStatus
import com.sakshyam.agribot.domain.model.AnalysisFrame
import com.sakshyam.agribot.domain.scan.GpsFix
import com.sakshyam.agribot.domain.scan.VisionFrame
import java.io.ByteArrayOutputStream
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

/** Per-frame quality: [weight] scales leaf evidence; [tooDark] drives the on-screen hint. */
data class FrameQuality(val weight: Float, val tooDark: Boolean)

object FrameTools {
    fun quality(frame: VisionFrame): FrameQuality {
        val q = FrameQualityAnalyzer.evaluate(frame.width, frame.height, frame.argb)
        val weight = if (q.status == FrameQualityStatus.ACCEPT) (0.4f + 12f * q.sharpness).coerceIn(0.4f, 1f) else 0.3f
        return FrameQuality(weight, q.brightness < 0.12f)
    }

    /** JPEG evidence for the evidence repository. Call off the main thread. */
    fun toAnalysisFrame(frame: VisionFrame, quality: Int = 85): AnalysisFrame {
        val bmp = Bitmap.createBitmap(frame.argb, frame.width, frame.height, Bitmap.Config.ARGB_8888)
        val jpeg = ByteArrayOutputStream().use { out -> bmp.compress(Bitmap.CompressFormat.JPEG, quality, out); out.toByteArray() }
        bmp.recycle()
        return AnalysisFrame(frame.width, frame.height, 0, frame.timestampNanos, frame.argb, jpeg)
    }
}

/** Hardware step detector. Emits Unit per step; [isAvailable] is false on phones without one. */
class StepCounter(private val context: Context) {
    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val sensor: Sensor? = sm.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR)

    val isAvailable: Boolean get() = sensor != null

    val hasPermission: Boolean
        get() = Build.VERSION.SDK_INT < Build.VERSION_CODES.Q ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACTIVITY_RECOGNITION) == PackageManager.PERMISSION_GRANTED

    fun steps(): Flow<Unit> = callbackFlow {
        val s = sensor
        if (s == null || !hasPermission) { close(); return@callbackFlow }
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) { trySend(Unit) }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }
        sm.registerListener(listener, s, SensorManager.SENSOR_DELAY_GAME)
        awaitClose { sm.unregisterListener(listener) }
    }
}

/** GPS fixes for the field trail (1 s, 1 m). Requires location permission; emits nothing otherwise. */
class LocationFeed(private val context: Context) {
    private val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    val hasPermission: Boolean
        get() = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED

    @SuppressLint("MissingPermission")
    fun fixes(): Flow<GpsFix> = callbackFlow {
        if (!hasPermission || !lm.isProviderEnabled(LocationManager.GPS_PROVIDER)) { close(); return@callbackFlow }
        val listener = LocationListener { loc: Location ->
            if (loc.hasAccuracy()) trySend(GpsFix(loc.latitude, loc.longitude, loc.accuracy.toDouble(), loc.time))
        }
        lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 1f, listener, Looper.getMainLooper())
        awaitClose { lm.removeUpdates(listener) }
    }
}
