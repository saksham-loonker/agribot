package com.sakshyam.agribot.ml

import com.sakshyam.agribot.ml.inference.ModelAssetDigestVerifier
import com.sakshyam.agribot.ml.inference.ModelUnavailableException
import java.io.ByteArrayInputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ModelAssetDigestVerifierTest {
    @Test
    fun computesSha256ForAssetBytes() {
        val digest = ModelAssetDigestVerifier.sha256 {
            ByteArrayInputStream("agribot".toByteArray())
        }

        assertEquals("06ee10f4cf96467029dc30cb62b1798d734e6ed70d57a10400805229ed7c9b6a", digest)
    }

    @Test
    fun acceptsMatchingDigestIgnoringCase() {
        ModelAssetDigestVerifier.verify(
            path = "models/classifier.tflite",
            expectedSha256 = "06EE10F4CF96467029DC30CB62B1798D734E6ED70D57A10400805229ED7C9B6A",
        ) {
            ByteArrayInputStream("agribot".toByteArray())
        }
    }

    @Test
    fun rejectsMissingExpectedDigest() {
        val error = assertFailsWith<ModelUnavailableException> {
            ModelAssetDigestVerifier.verify(
                path = "models/classifier.tflite",
                expectedSha256 = null,
            ) {
                ByteArrayInputStream("agribot".toByteArray())
            }
        }

        assertEquals("Missing SHA-256 for model asset: models/classifier.tflite", error.message)
    }

    @Test
    fun rejectsDigestMismatch() {
        val error = assertFailsWith<ModelUnavailableException> {
            ModelAssetDigestVerifier.verify(
                path = "models/classifier.tflite",
                expectedSha256 = "0000000000000000000000000000000000000000000000000000000000000000",
            ) {
                ByteArrayInputStream("agribot".toByteArray())
            }
        }

        assertEquals(
            "SHA-256 mismatch for model asset models/classifier.tflite: expected 0000000000000000000000000000000000000000000000000000000000000000, got 06ee10f4cf96467029dc30cb62b1798d734e6ed70d57a10400805229ed7c9b6a",
            error.message,
        )
    }
}
