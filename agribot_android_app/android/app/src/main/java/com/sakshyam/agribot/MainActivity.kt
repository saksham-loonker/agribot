package com.sakshyam.agribot

import android.Manifest
import android.os.Build
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import com.sakshyam.agribot.designsystem.AgribotTheme
import com.sakshyam.agribot.featurescan.AgribotScanApp
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    private var cameraPermissionGranted by mutableStateOf(false)
    private var locationPermissionGranted by mutableStateOf(false)
    private var locationPermissionRequested by mutableStateOf(false)
    private var activityRecognitionPermissionGranted by mutableStateOf(false)
    private var activityRecognitionPermissionRequested by mutableStateOf(false)

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        cameraPermissionGranted = granted
    }

    private val locationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { result ->
        locationPermissionGranted = result[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
            result[Manifest.permission.ACCESS_COARSE_LOCATION] == true
        locationPermissionRequested = true
    }

    private val activityRecognitionPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        activityRecognitionPermissionGranted = granted || Build.VERSION.SDK_INT < Build.VERSION_CODES.Q
        activityRecognitionPermissionRequested = true
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraPermissionGranted = hasCameraPermission()
        locationPermissionGranted = hasLocationPermission()
        activityRecognitionPermissionGranted = hasActivityRecognitionPermission()
        setContent {
            AgribotTheme {
                AgribotScanApp(
                    cameraPermissionGranted = cameraPermissionGranted,
                    locationPermissionGranted = locationPermissionGranted,
                    locationPermissionRequested = locationPermissionRequested,
                    activityRecognitionPermissionGranted = activityRecognitionPermissionGranted,
                    activityRecognitionPermissionRequested = activityRecognitionPermissionRequested,
                    onRequestCameraPermission = {
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    },
                    onRequestLocationPermission = {
                        locationPermissionLauncher.launch(
                            arrayOf(
                                Manifest.permission.ACCESS_FINE_LOCATION,
                                Manifest.permission.ACCESS_COARSE_LOCATION,
                            ),
                        )
                    },
                    onRequestActivityRecognitionPermission = {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                            activityRecognitionPermissionLauncher.launch(Manifest.permission.ACTIVITY_RECOGNITION)
                        } else {
                            activityRecognitionPermissionGranted = true
                            activityRecognitionPermissionRequested = true
                        }
                    },
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        cameraPermissionGranted = hasCameraPermission()
        locationPermissionGranted = hasLocationPermission()
        activityRecognitionPermissionGranted = hasActivityRecognitionPermission()
    }

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun hasLocationPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

    private fun hasActivityRecognitionPermission(): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.Q ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACTIVITY_RECOGNITION) == PackageManager.PERMISSION_GRANTED
}
