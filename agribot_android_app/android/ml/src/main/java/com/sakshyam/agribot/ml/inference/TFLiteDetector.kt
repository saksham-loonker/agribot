package com.sakshyam.agribot.ml.inference

import android.content.Context
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import com.sakshyam.agribot.ml.preprocessing.RgbImage
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import com.sakshyam.agribot.domain.model.ScanConstants
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

class TFLiteDetector(
    context: Context,
    assetModelPath: String,
    private val inputWidth: Int = 256,
    private val inputHeight: Int = 256,
    private val minScore: Float = ScanConstants.MIN_DETECTOR_CONFIDENCE,
    private val threadCount: Int = 4,
    private val outputLayoutHint: String? = null,
    private val expectedCandidates: Int? = null,
    private val classCount: Int? = null,
    private val hasObjectness: Boolean = false,
    private val coordinateSpaceHint: String? = null,
    private val labels: List<String> = emptyList(),
) : AutoCloseable {
    private val interpreter: Interpreter = Interpreter(loadMappedAsset(context, assetModelPath), Interpreter.Options().apply {
        setNumThreads(threadCount.coerceAtLeast(1))
        setUseXNNPACK(true)
    })
    private val outputShape: IntArray = interpreter.getOutputTensor(0).shape()
    private val outputLayout: DetectorTensorLayout = DetectorTensorLayout.from(
        shape = outputShape,
        outputLayoutHint = outputLayoutHint,
        expectedCandidates = expectedCandidates,
    )
    private val inputDataType = interpreter.getInputTensor(0).dataType()
    private val outputDataType = interpreter.getOutputTensor(0).dataType()
    @Volatile
    var lastDiagnostics: DetectorDiagnostics = DetectorDiagnostics.empty(outputShape, inputDataType, outputDataType)
        private set
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(4 * inputWidth * inputHeight * 3)
        .order(ByteOrder.nativeOrder())
    private val nmsOutput: Array<Array<FloatArray>>? =
        if (outputLayout.format == DetectorTensorLayout.Format.NMS_XYXY_SCORE_CLASS) {
            Array(1) { Array(outputLayout.candidateCount) { FloatArray(outputLayout.fieldCount) } }
        } else {
            null
        }
    private val rawOutput: Array<Array<FloatArray>>? =
        if (outputLayout.format == DetectorTensorLayout.Format.RAW_YOLO_FEATURES_FIRST ||
            outputLayout.format == DetectorTensorLayout.Format.RAW_YOLO_CANDIDATES_FIRST
        ) {
            Array(1) {
                Array(
                    if (outputLayout.format == DetectorTensorLayout.Format.RAW_YOLO_FEATURES_FIRST) {
                        outputLayout.fieldCount
                    } else {
                        outputLayout.candidateCount
                    },
                ) {
                    FloatArray(
                        if (outputLayout.format == DetectorTensorLayout.Format.RAW_YOLO_FEATURES_FIRST) {
                            outputLayout.candidateCount
                        } else {
                            outputLayout.fieldCount
                        },
                    )
                }
            }
        } else {
            null
        }

    init {
        require(outputLayout.isSupported) {
            "Unsupported detector output tensor shape ${outputShape.contentToString()}"
        }
        require(inputDataType == DataType.FLOAT32 && outputDataType == DataType.FLOAT32) {
            "Detector requires FLOAT32 input/output; got input=$inputDataType output=$outputDataType"
        }
        require(inputWidth > 0 && inputHeight > 0) { "Detector input dimensions must be positive" }
        require(inputWidth.toLong() * inputHeight.toLong() <= MAX_INPUT_PIXELS) {
            "Detector input dimensions exceed the bounded pixel budget"
        }
        if (expectedCandidates != null) {
            require(outputLayout.candidateCount == expectedCandidates) {
                "Detector output candidate count ${outputLayout.candidateCount} does not match manifest $expectedCandidates"
            }
        }
        if (classCount != null && outputLayout.format != DetectorTensorLayout.Format.NMS_XYXY_SCORE_CLASS) {
            val expectedFields = 4 + classCount.coerceAtLeast(0) + if (hasObjectness) 1 else 0
            require(outputLayout.fieldCount == expectedFields) {
                "Detector output field count ${outputLayout.fieldCount} does not match class/objectness contract $expectedFields"
            }
        }
    }

    fun detect(image: RgbImage): List<FrontOverviewCandidate> {
        return when (outputLayout.format) {
            DetectorTensorLayout.Format.NMS_XYXY_SCORE_CLASS -> {
                val output = nmsOutput ?: return emptyList()
                val detectorInput = DetectorInputPreprocessor.letterbox(image, inputWidth, inputHeight)
                synchronized(this) {
                    interpreter.run(toInputBuffer(detectorInput.image), output)
                    val result = DetectorOutputParser.parseNmsRowsDetailed(
                        rows = output[0],
                        frameWidth = image.width,
                        frameHeight = image.height,
                        inputWidth = inputWidth,
                        inputHeight = inputHeight,
                        minScore = minScore,
                        maxCandidates = ScanConstants.MAX_DETECTOR_CANDIDATES,
                        coordinateSpace = coordinateSpace,
                        transform = detectorInput.transform,
                        labels = labels,
                    )
                    updateDiagnostics(result.diagnostics)
                    result.candidates
                }
            }
            DetectorTensorLayout.Format.RAW_YOLO_FEATURES_FIRST,
            DetectorTensorLayout.Format.RAW_YOLO_CANDIDATES_FIRST,
            -> {
                val output = rawOutput ?: return emptyList()
                val detectorInput = DetectorInputPreprocessor.letterbox(image, inputWidth, inputHeight)
                synchronized(this) {
                    interpreter.run(toInputBuffer(detectorInput.image), output)
                    val rawRows = if (outputLayout.format == DetectorTensorLayout.Format.RAW_YOLO_FEATURES_FIRST) {
                        output[0]
                    } else {
                        transposeCandidateRows(output[0])
                    }
                    logTopRawCandidate(rawRows)
                    val result = DetectorOutputParser.parseRawYoloOutputDetailed(
                        rows = rawRows,
                        frameWidth = image.width,
                        frameHeight = image.height,
                        inputWidth = inputWidth,
                        inputHeight = inputHeight,
                        minScore = minScore,
                        maxCandidates = ScanConstants.MAX_DETECTOR_CANDIDATES,
                        coordinateSpace = coordinateSpace,
                        transform = detectorInput.transform,
                        classCount = classCount,
                        hasObjectness = hasObjectness,
                        labels = labels,
                    )
                    updateDiagnostics(result.diagnostics)
                    result.candidates
                }
            }
            DetectorTensorLayout.Format.UNSUPPORTED -> emptyList()
        }
    }

    override fun close() {
        interpreter.close()
    }

    private fun updateDiagnostics(diagnostics: DetectorOutputParser.ParseDiagnostics) {
        lastDiagnostics = DetectorDiagnostics(
            inputShape = interpreter.getInputTensor(0).shape().copyOf(),
            outputShape = outputShape.copyOf(),
            inputDataType = inputDataType,
            outputDataType = outputDataType,
            rawCandidateCount = diagnostics.rawCandidateCount,
            scoreValidCount = diagnostics.scoreValidCount,
            acceptedBeforeNms = diagnostics.acceptedBeforeNms,
            acceptedAfterNms = diagnostics.acceptedAfterNms,
            maxRawConfidence = diagnostics.maxRawConfidence,
            rejectionCounts = diagnostics.rejectionCounts,
        )
        Log.i(TAG, lastDiagnostics.toLogString())
    }

    private fun logTopRawCandidate(rows: Array<FloatArray>) {
        val scoreRow = rows.getOrNull(4) ?: return
        val index = scoreRow.indices.maxByOrNull { scoreRow[it] } ?: return
        val coordinates = (0..3).mapNotNull { field -> rows.getOrNull(field)?.getOrNull(index) }
        if (coordinates.size == 4) {
            Log.d(
                TAG,
                "rawTop index=$index xywh=${coordinates.joinToString(prefix = "[", postfix = "]")} score=${scoreRow[index]}",
            )
        }
    }

    private fun toInputBuffer(image: RgbImage): ByteBuffer {
        inputBuffer.clear()
        for (pixel in image.pixels) {
            inputBuffer.putFloat(((pixel shr 16) and 0xff) / 255f)
            inputBuffer.putFloat(((pixel shr 8) and 0xff) / 255f)
            inputBuffer.putFloat((pixel and 0xff) / 255f)
        }
        inputBuffer.rewind()
        return inputBuffer
    }

    private val coordinateSpace: DetectorCoordinateSpace
        get() = when (coordinateSpaceHint?.trim()?.lowercase()) {
            "normalized", "normalised", "0_1" -> DetectorCoordinateSpace.NORMALIZED
            "model_input_pixels", "pixels" -> DetectorCoordinateSpace.MODEL_INPUT_PIXELS
            else -> DetectorCoordinateSpace.MODEL_INPUT_PIXELS
        }

    private fun transposeCandidateRows(rows: Array<FloatArray>): Array<FloatArray> {
        if (rows.isEmpty()) return emptyArray()
        val fieldCount = rows.first().size
        return Array(fieldCount) { field ->
            FloatArray(rows.size) { candidate -> rows[candidate][field] }
        }
    }

    private fun loadMappedAsset(context: Context, path: String): MappedByteBuffer {
        context.assets.openFd(path).use { descriptor ->
            FileInputStream(descriptor.fileDescriptor).use { input ->
                val channel = input.channel
                return channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
            }
        }
    }

    private companion object {
        const val MAX_INPUT_PIXELS = 1_000_000L
    }
}

data class DetectorDiagnostics(
    val inputShape: IntArray,
    val outputShape: IntArray,
    val inputDataType: DataType,
    val outputDataType: DataType,
    val rawCandidateCount: Int,
    val scoreValidCount: Int,
    val acceptedBeforeNms: Int,
    val acceptedAfterNms: Int,
    val maxRawConfidence: Float?,
    val rejectionCounts: Map<String, Int>,
) {
    fun toLogString(): String =
        "detector input=${inputShape.contentToString()}($inputDataType) output=${outputShape.contentToString()}($outputDataType) " +
            "raw=$rawCandidateCount scoreValid=$scoreValidCount acceptedBeforeNms=$acceptedBeforeNms " +
            "accepted=$acceptedAfterNms maxRaw=${maxRawConfidence ?: "none"} rejections=$rejectionCounts"

    companion object {
        fun empty(shape: IntArray, inputType: DataType, outputType: DataType) = DetectorDiagnostics(
            inputShape = intArrayOf(), outputShape = shape.copyOf(), inputDataType = inputType,
            outputDataType = outputType, rawCandidateCount = 0, scoreValidCount = 0,
            acceptedBeforeNms = 0, acceptedAfterNms = 0, maxRawConfidence = null,
            rejectionCounts = mapOf("not_run" to 1),
        )
    }
}

private const val TAG = "AgribotDetector"
