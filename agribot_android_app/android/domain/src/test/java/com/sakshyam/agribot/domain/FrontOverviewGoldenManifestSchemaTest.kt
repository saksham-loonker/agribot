package com.sakshyam.agribot.domain

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.name
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class FrontOverviewGoldenManifestSchemaTest {
    @Test
    fun frontOverviewManifestSchemaPreservesPlanRequiredFields() {
        val schema = goldenRoot().resolve("expected/front_overview_manifest.schema.json").readJsonObject()
        val required = schema.getArray("required").map { it.jsonPrimitive.content }.toSet()

        assertEquals(
            setOf(
                "fixture_id",
                "source_file",
                "mode",
                "reviewer",
                "reviewed_at",
                "camera_pose",
                "expected",
                "acceptance",
            ),
            required,
        )

        val properties = schema.getObject("properties")
        assertEquals("FRONT_ROW_OVERVIEW", properties.getObject("mode").getString("const"))
        assertEquals("^front_overview/.+", properties.getObject("source_file").getString("pattern"))
        assertTrue(properties.getObject("expected").getObject("items").getArray("required").containsText("bbox_px"))
        assertTrue(properties.getObject("expected").getObject("items").getArray("required").containsText("row_side"))
        assertTrue(properties.getObject("expected").getObject("items").getArray("required").containsText("plant_number"))
        assertTrue(properties.getObject("acceptance").getArray("required").containsText("allow_unknown"))
    }

    @Test
    fun frontOverviewManifestExampleMatchesSchemaContract() {
        val manifest = goldenRoot().resolve("expected/front_overview_manifest.example.jsonl")
        val entries = Files.readAllLines(manifest)
            .filter { it.isNotBlank() }
            .map { Json.parseToJsonElement(it).jsonObject }

        assertEquals(1, entries.size)
        val entry = entries.single()
        assertEquals("FRONT_ROW_OVERVIEW", entry.getString("mode"))
        assertTrue(entry.getString("fixture_id").isNotBlank())
        assertTrue(entry.getString("source_file").startsWith("front_overview/"))
        assertTrue(entry.getString("reviewer").isNotBlank())
        assertTrue(DATE_PATTERN.matches(entry.getString("reviewed_at")))

        val pose = entry.getObject("camera_pose")
        assertTrue(pose.getNumber("distance_m") > 0.0)
        assertTrue(pose.getNumber("height_m") > 0.0)
        assertTrue(pose.getNumber("tilt_degrees") in -90.0..90.0)

        val expected = entry.getArray("expected").single().jsonObject
        assertEquals(listOf(120, 220, 260, 420), expected.getArray("bbox_px").map { it.jsonPrimitive.content.toInt() })
        assertEquals("left", expected.getString("row_side"))
        assertEquals("A", expected.getString("row_id"))
        assertEquals(12, expected.getNumber("plant_number").toInt())
        assertEquals("plant_candidate", expected.getString("label_group"))
        assertEquals("clear", expected.getString("geometry_status"))

        val acceptance = entry.getObject("acceptance")
        assertTrue(acceptance.getBoolean("row_side_required"))
        assertTrue(acceptance.getBoolean("plant_number_required"))
        assertEquals(false, acceptance.getBoolean("allow_unknown"))
    }

    private fun goldenRoot(): Path {
        val start = Path.of("").toAbsolutePath()
        val candidates = generateSequence(start) { it.parent }
            .flatMap { root ->
                sequenceOf(root.resolve("android_test_assets"), root.parent?.resolve("android_test_assets"))
                    .filterNotNull()
            }
        return candidates.firstOrNull { it.exists() && it.name == "android_test_assets" }
            ?: error("android_test_assets not found from $start")
    }

    private fun Path.readJsonObject(): JsonObject =
        Json.parseToJsonElement(Files.readString(this)).jsonObject

    private fun JsonObject.getObject(name: String): JsonObject =
        assertNotNull(this[name], "Missing object $name").jsonObject

    private fun JsonObject.getArray(name: String): JsonArray =
        assertNotNull(this[name], "Missing array $name").jsonArray

    private fun JsonObject.getString(name: String): String =
        assertNotNull(this[name], "Missing string $name").jsonPrimitive.content

    private fun JsonObject.getNumber(name: String): Double =
        assertNotNull(this[name], "Missing number $name").jsonPrimitive.content.toDouble()

    private fun JsonObject.getBoolean(name: String): Boolean =
        assertNotNull(this[name], "Missing boolean $name").jsonPrimitive.boolean

    private fun Iterable<JsonElement>.containsText(value: String): Boolean =
        any { it.jsonPrimitive.content == value }

    private companion object {
        val DATE_PATTERN = Regex("\\d{4}-\\d{2}-\\d{2}")
    }
}
