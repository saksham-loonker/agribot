package com.sakshyam.agribot.camera

import android.Manifest
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.sakshyam.agribot.domain.model.AnalysisFrame
import java.util.concurrent.Executors

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    targetFps: Int = DEFAULT_TARGET_FPS,
    onFrame: (AnalysisFrame) -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val executor = remember { Executors.newSingleThreadExecutor() }
    val latestOnFrame = rememberUpdatedState(onFrame)
    val analyzer = remember {
        SideScanAnalyzer(targetFps) { frame -> latestOnFrame.value(frame) }
    }

    // AndroidView's factory is intentionally one-shot. Keep the analyzer
    // instance bound to CameraX and update its bounded sampler when the user
    // changes the scan rate; recreating the remembered analyzer alone would not
    // change the live camera pipeline.
    LaunchedEffect(analyzer, targetFps) {
        analyzer.updateTargetFps(targetFps)
    }

    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener(
                    {
                        val provider = cameraProviderFuture.get()
                        val preview = Preview.Builder().build().also { it.setSurfaceProvider(surfaceProvider) }
                        val analysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build()
                            .also { it.setAnalyzer(executor, analyzer) }
                        provider.unbindAll()
                        provider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            analysis,
                        )
                    },
                    ContextCompat.getMainExecutor(ctx),
                )
            }
        },
    )

    DisposableEffect(Unit) {
        onDispose {
            // Do not call future.get() from the main thread while leaving the
            // screen. If CameraX is still initializing, waiting here can make
            // the permission dialog or navigation appear frozen.
            val providerFuture = ProcessCameraProvider.getInstance(context)
            providerFuture.addListener(
                { runCatching { providerFuture.get().unbindAll() } },
                ContextCompat.getMainExecutor(context),
            )
            executor.shutdownNow()
        }
    }
}

private const val DEFAULT_TARGET_FPS = 5

val requiredCameraPermission: String = Manifest.permission.CAMERA
