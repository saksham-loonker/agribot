package com.sakshyam.agribot.camera

import android.Manifest
import android.util.Log
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCase
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
import java.util.concurrent.atomic.AtomicBoolean
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
    val useCases = remember { AtomicReference<List<UseCase>>(emptyList()) }
    val disposed = remember { AtomicBoolean(false) }
    val latestTorch = rememberUpdatedState(torchOn)
    // Reused RGBA staging buffer: the analyzer runs on one thread, so one copy per frame is enough.
    val staging = remember { AtomicReference(ByteArray(0)) }

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
                        if (disposed.get()) return@addListener   // screen closed before the camera was ready
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
                                if (latestReady.value()) latestOnFrame.value(it.toVisionFrame(staging))
                            }
                        }
                        val group = UseCaseGroup.Builder().addUseCase(preview).addUseCase(analysis)
                        val vp = viewPort
                        if (vp != null) group.setViewPort(vp) else Log.w("LeafCamera", "ViewPort unavailable; overlay may not match the preview exactly")
                        provider.unbind(*useCases.get().toTypedArray())
                        val camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, group.build())
                        useCases.set(listOf(preview, analysis))
                        cameraRef.set(camera)
                        camera.cameraControl.enableTorch(latestTorch.value)
                    }, ContextCompat.getMainExecutor(ctx))
                }
            }
        },
    )

    DisposableEffect(Unit) {
        onDispose {
            disposed.set(true)
            cameraRef.set(null)
            val providerFuture = ProcessCameraProvider.getInstance(context)
            providerFuture.addListener({
                // Unbind only this screen's use cases, then stop the analyzer thread.
                runCatching { providerFuture.get().unbind(*useCases.getAndSet(emptyList()).toTypedArray()) }
                executor.shutdown()
            }, ContextCompat.getMainExecutor(context))
        }
    }
}

private fun ImageProxy.toVisionFrame(staging: AtomicReference<ByteArray>): VisionFrame {
    val plane = planes[0]
    val buf = plane.buffer.duplicate().apply { rewind() }
    val n = buf.remaining()
    val bytes = staging.get().let { if (it.size >= n) it else ByteArray(n).also { b -> staging.set(b) } }
    buf.get(bytes, 0, n)
    val crop = cropRect
    val up = RgbaFrameConverter.toUpright(bytes, plane.rowStride, crop.left, crop.top, crop.width(), crop.height(), imageInfo.rotationDegrees)
    return VisionFrame(up.width, up.height, up.argb, imageInfo.timestamp)
}
