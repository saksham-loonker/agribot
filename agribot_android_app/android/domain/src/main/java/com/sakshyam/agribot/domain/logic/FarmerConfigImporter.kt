package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.ImportedFarmerConfig
import com.sakshyam.agribot.domain.model.RecordingProfile
import com.sakshyam.agribot.domain.model.RowId
import com.sakshyam.agribot.domain.model.ScanConstants
import java.time.Instant
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

object FarmerConfigImporter {
    private val json = Json { ignoreUnknownKeys = true }

    fun importFromJson(raw: String): ImportedFarmerConfig {
        val root = json.parseToJsonElement(raw).jsonObject
        val layoutJson = root["field_layout"]?.jsonObject
        val fieldsJson = layoutJson?.get("fields") as? JsonArray
        val updatedAt = parseInstant(layoutJson?.string("updated_at"))
        val layouts = if (fieldsJson != null && fieldsJson.isNotEmpty()) {
            fieldsJson.mapIndexed { index, element -> parseField(index + 1, element.jsonObject, root, updatedAt) }
        } else {
            listOf(parseFallbackField(root, updatedAt))
        }

        val activeFieldId = layoutJson?.string("active_field_id")
        val active = layouts.firstOrNull { it.id.value == activeFieldId }
            ?: layouts.firstOrNull { it.name == root.string("field_id") }
            ?: layouts.first()

        return ImportedFarmerConfig(
            activeLayout = active,
            layouts = layouts,
            recordingProfile = RecordingProfile(
                fps = root.double("fps", 5.0),
                threads = root.int("threads", 2),
                classifierSourceNote = root.string("classifier"),
                confidenceThreshold = root.float("conf", ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD),
                highConfidenceThreshold = root.float("high_conf", ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD),
            ),
        )
    }

    private fun parseField(index: Int, field: JsonObject, root: JsonObject, updatedAt: Instant): FieldLayout {
        val id = field.string("id") ?: "field_$index"
        val name = field.string("name") ?: field.string("field_id") ?: root.string("field_id") ?: "Field $index"
        val activeRowId = field.string("active_row_id") ?: root.string("row_id") ?: "A"
        val rowCount = field.int("row_count", 1)
        val defaultPlants = field.int("plants_per_row", 100)
        val defaultRowSpacing = field.double("row_spacing_m", 1.0)
        val defaultPlantSpacing = field.double("plant_spacing_m", 0.5)
        val rows = parseRows(
            rowsElement = field["rows"],
            rowCount = rowCount,
            defaultPlants = defaultPlants,
            defaultRowSpacing = defaultRowSpacing,
            defaultPlantSpacing = defaultPlantSpacing,
        )
        return FieldLayout(
            id = FieldId(id),
            name = name,
            activeRowId = RowId(activeRowId),
            rows = rows,
            startPlant = field.int("start_plant", root.int("start_plant", 1)),
            plantStep = field.int("plant_step", root.int("plant_step", 1)),
            plantCooldownSec = field.double("plant_cooldown_sec", root.double("plant_cooldown_sec", 2.0)),
            updatedAt = updatedAt,
        )
    }

    private fun parseFallbackField(root: JsonObject, updatedAt: Instant): FieldLayout {
        val name = root.string("field_id") ?: "Field 1"
        val rows = parseRows(
            rowsElement = null,
            rowCount = root.int("row_count", 1),
            defaultPlants = root.int("plants_per_row", 100),
            defaultRowSpacing = root.double("row_spacing_m", 1.0),
            defaultPlantSpacing = root.double("plant_spacing_m", 0.5),
        )
        return FieldLayout(
            id = FieldId("field_1"),
            name = name,
            activeRowId = RowId(root.string("row_id") ?: "A"),
            rows = rows,
            startPlant = root.int("start_plant", 1),
            plantStep = root.int("plant_step", 1),
            plantCooldownSec = root.double("plant_cooldown_sec", 2.0),
            updatedAt = updatedAt,
        )
    }

    private fun parseRows(
        rowsElement: JsonElement?,
        rowCount: Int,
        defaultPlants: Int,
        defaultRowSpacing: Double,
        defaultPlantSpacing: Double,
    ): List<FieldRow> {
        val rowObjects = rowsElement?.jsonArray?.mapNotNull { it as? JsonObject }.orEmpty()
        if (rowObjects.isNotEmpty()) {
            return rowObjects.mapIndexed { index, row ->
                FieldRow(
                    id = RowId(row.string("row_id") ?: row.string("id") ?: rowIdForIndex(index)),
                    rowIndex = row.int("row_index", index + 1),
                    plantsPerRow = row.int("plants_per_row", defaultPlants).coerceAtLeast(1),
                    plantSpacingM = row.double("plant_spacing_m", defaultPlantSpacing).coerceAtLeast(0.0),
                    rowSpacingM = row.double("row_spacing_m", defaultRowSpacing).coerceAtLeast(0.0),
                    yM = row.double("y_m", index * defaultRowSpacing),
                )
            }
        }

        var yM = 0.0
        return (0 until rowCount.coerceAtLeast(1)).map { index ->
            FieldRow(
                id = RowId(rowIdForIndex(index)),
                rowIndex = index + 1,
                plantsPerRow = defaultPlants.coerceAtLeast(1),
                plantSpacingM = defaultPlantSpacing.coerceAtLeast(0.0),
                rowSpacingM = defaultRowSpacing.coerceAtLeast(0.0),
                yM = yM,
            ).also {
                yM += defaultRowSpacing.coerceAtLeast(0.0)
            }
        }
    }

    private fun rowIdForIndex(index: Int): String {
        var number = index.coerceAtLeast(0)
        var value = ""
        while (true) {
            val rem = number % 26
            number /= 26
            value = ('A'.code + rem).toChar() + value
            if (number == 0) return value
            number -= 1
        }
    }

    private fun parseInstant(value: String?): Instant =
        runCatching { if (value == null) Instant.EPOCH else Instant.parse(value) }
            .getOrElse { Instant.EPOCH }

    private fun JsonObject.string(name: String): String? =
        this[name]?.jsonPrimitive?.content?.takeIf { it.isNotBlank() }

    private fun JsonObject.int(name: String, fallback: Int): Int =
        this[name]?.jsonPrimitive?.intOrNull ?: fallback

    private fun JsonObject.double(name: String, fallback: Double): Double =
        this[name]?.jsonPrimitive?.doubleOrNull ?: fallback

    private fun JsonObject.float(name: String, fallback: Float): Float =
        this[name]?.jsonPrimitive?.floatOrNull ?: fallback
}
