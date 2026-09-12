package com.sakshyam.agribot.ml.inference

import java.io.InputStream
import java.security.MessageDigest

object ModelAssetDigestVerifier {
    fun verify(path: String, expectedSha256: String?, openStream: () -> InputStream) {
        val expected = expectedSha256?.trim()?.lowercase()
        if (expected.isNullOrBlank()) {
            throw ModelUnavailableException("Missing SHA-256 for model asset: $path")
        }

        val actual = sha256(openStream)
        if (actual != expected) {
            throw ModelUnavailableException("SHA-256 mismatch for model asset $path: expected $expected, got $actual")
        }
    }

    fun sha256(openStream: () -> InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        openStream().use { input ->
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString(separator = "") { byte -> "%02x".format(byte) }
    }
}
