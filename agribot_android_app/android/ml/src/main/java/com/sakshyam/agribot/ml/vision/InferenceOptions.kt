package com.sakshyam.agribot.ml.vision

/** Runtime tuning does not change the model, its pixels or the number of evaluated leaves. */
enum class InferenceBackend { CPU, GPU, NNAPI }

data class InferenceOptions(
    val detectorThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4),
    val classifierThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4),
    val detectorBackend: InferenceBackend = InferenceBackend.CPU,
    val classifierBackend: InferenceBackend = InferenceBackend.CPU,
) {
    init {
        require(detectorThreads in 1..8 && classifierThreads in 1..8) { "inference threads must be 1..8" }
    }
}
