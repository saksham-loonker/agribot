package com.sakshyam.agribot.camera

import android.Manifest
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
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
import com.sakshyam.agribot.domain.scan.VisionFrame
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference

val requiredCameraPermission: String = Manifest.permission.CAMERA

/**
 * Back-camera preview plus an analysis stream whose frames cover exactly the visible preview area
 * (shared ViewPort), delivered upright as ARGB. A new frame is only produced when [isReady] returns
 * true, so a slow consumer never builds a backlog.
 */
@Composable
fun LeafCamera(
    modifier: Modifier = Modifier,
    torchOn: Boolean = false,
    isReady: () -> Boolean,
    onFrame: (VisionFrame) -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val executor = remember { Executors.newSingleThreadExecutor() }
    val latestOnFrame = rememberUpdatedState(onFrame)
    val latestReady = rememberUpdatedState(isReady)
    val cameraRef = remember { AtomicReference<Camera?>(null) }

    LaunchedEffect(torchOn) { cameraRef.get()?.cameraControl?.enableTorch(torchOn) }

    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                val providerFuture = ProcessCameraProvider.getInstance(ctx)
                // ViewPort needs the view to be laid out.
                post {
                    providerFuture.addListener({
                        val provider = providerFuture.get()
                        val selector = ResolutionSelector.Builder()
                            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                            .setResolutionStrategy(ResolutionStrategy(Size(1280, 960), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                            .build()
                        val preview = Preview.Builder().setResolutionSelector(selector).build().also { it.setSurfaceProvider(surfaceProvider) }
                        val analysis = ImageAnalysis.Builder()
                            .setResolutionSelector(selector)
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                            .build()
                        analysis.setAnalyzer(executor) { proxy ->
                            proxy.use {
                                if (latestReady.value()) latestOnFrame.value(it.toVisionFrame())
                            }
                        }
                        val group = UseCaseGroup.Builder().addUseCase(preview).addUseCase(analysis)
                        viewPort?.let { group.setViewPort(it) }
                        provider.unbindAll()
                        val camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, group.build())
                        cameraRef.set(camera)
                        camera.cameraControl.enableTorch(torchOn)
                    }, ContextCompat.getMainExecutor(ctx))
                }
            }
        },
    )

    DisposableEffect(Unit) {
        onDispose {
            val providerFuture = ProcessCameraProvider.getInstance(context)
            providerFuture.addListener({ runCatching { providerFuture.get().unbindAll() } }, ContextCompat.getMainExecutor(context))
            cameraRef.set(null)
            executor.shutdown()
        }
    }
}

private fun ImageProxy.toVisionFrame(): VisionFrame {
    val plane = planes[0]
    val buf = plane.buffer.duplicate().apply { rewind() }
    val bytes = ByteArray(buf.remaining()).also { buf.get(it) }
    val crop = cropRect
    val up = RgbaFrameConverter.toUpright(bytes, plane.rowStride, crop.left, crop.top, crop.width(), crop.height(), imageInfo.rotationDegrees)
    return VisionFrame(up.width, up.height, up.argb, imageInfo.timestamp)
}
