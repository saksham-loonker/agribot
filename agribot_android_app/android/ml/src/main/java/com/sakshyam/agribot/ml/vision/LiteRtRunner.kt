package com.sakshyam.agribot.ml.vision

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate

/** Private workspace; creation, invocation and cleanup must share one physical thread for GPU. */
internal class LiteRtRunner(model: MappedByteBuffer, threads: Int, useXnnpack: Boolean, backend: InferenceBackend) : AutoCloseable {
    private val ownerThread = Thread.currentThread()
    private val requiresThreadAffinity = backend == InferenceBackend.GPU
    private val delegate: Delegate? = when (backend) {
        InferenceBackend.CPU -> null
        InferenceBackend.GPU -> GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(false))
        InferenceBackend.NNAPI -> NnApiDelegate(NnApiDelegate.Options().setUseNnapiCpu(false))
    }
    private val interpreter = try {
        Interpreter(model, Interpreter.Options().setNumThreads(threads)
            .setUseXNNPACK(useXnnpack && backend == InferenceBackend.CPU).apply {
                delegate?.let { addDelegate(it) }
            })
    } catch (error: Throwable) {
        try { delegate?.close() } catch (cleanup: Throwable) {
            if (cleanup !== error) error.addSuppressed(cleanup)
        }
        throw error
    }
    val inputShape: IntArray = interpreter.getInputTensor(0).shape()
    val outputShape: IntArray = interpreter.getOutputTensor(0).shape()
    val inputValues = FloatArray(inputShape.fold(1) { a, b -> a * b })
    private val input = ByteBuffer.allocateDirect(inputValues.size * 4).order(ByteOrder.nativeOrder())
    private val output = ByteBuffer.allocateDirect(outputShape.fold(4) { a, b -> a * b }).order(ByteOrder.nativeOrder())
    private val inputFloats = input.asFloatBuffer()
    private val outputFloats = output.asFloatBuffer()
    val outputSize: Int = outputShape.fold(1) { a, b -> a * b }

    fun run(values: FloatArray): FloatArray {
        checkThread()
        require(values.size == inputValues.size) { "input ${values.size} floats != tensor ${inputValues.size}" }
        inputFloats.rewind()
        inputFloats.put(values)
        input.rewind()
        output.rewind()
        interpreter.run(input, output)
        outputFloats.rewind()
        // Output ownership is independent: callers retain logits/probabilities after the next run.
        return FloatArray(outputSize).also { outputFloats.get(it) }
    }

    override fun close() {
        checkThread()
        try { interpreter.close() } finally { delegate?.close() }
    }

    private fun checkThread() {
        check(!requiresThreadAffinity || Thread.currentThread() === ownerThread) {
            "GPU runtime must be created, invoked and closed on its dedicated inference thread"
        }
    }
}
