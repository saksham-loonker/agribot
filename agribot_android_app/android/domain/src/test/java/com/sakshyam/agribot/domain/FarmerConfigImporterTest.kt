package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.FarmerConfigImporter
import kotlin.test.Test
import kotlin.test.assertEquals

class FarmerConfigImporterTest {
    @Test
    fun importsActiveFieldRowsAndRecordingProfileFromPiConfig() {
        val imported = FarmerConfigImporter.importFromJson(CONFIG)

        assertEquals("field_2", imported.activeLayout.id.value)
        assertEquals("Field 2", imported.activeLayout.name)
        assertEquals("B", imported.activeLayout.activeRowId.value)
        assertEquals(2.0, imported.activeLayout.plantCooldownSec)
        assertEquals(10.0, imported.recordingProfile.fps)
        assertEquals(4, imported.recordingProfile.threads)
        assertEquals("runtime_exports/classifier_openvino_model", imported.recordingProfile.classifierSourceNote)

        val rowB = imported.activeLayout.rows.single { it.id.value == "B" }
        assertEquals(15, rowB.plantsPerRow)
        assertEquals(0.35, rowB.plantSpacingM)
        assertEquals(1.4, rowB.yM)
    }

    private companion object {
        val CONFIG = """
            {
              "field_id": "Field 2",
              "row_id": "B",
              "start_plant": 1,
              "plant_step": 1,
              "plant_cooldown_sec": 2.0,
              "field_layout": {
                "active_field_id": "field_2",
                "fields": [
                  {
                    "id": "field_2",
                    "name": "Field 2",
                    "field_id": "Field 2",
                    "active_row_id": "B",
                    "row_count": 3,
                    "plants_per_row": 20,
                    "row_spacing_m": 1.4,
                    "plant_spacing_m": 0.45,
                    "start_plant": 1,
                    "plant_step": 1,
                    "plant_cooldown_sec": 2.0,
                    "rows": [
                      {"id": "A", "row_id": "A", "row_index": 1, "plants_per_row": 20, "plant_spacing_m": 0.45, "row_spacing_m": 1.4, "y_m": 0.0},
                      {"id": "B", "row_id": "B", "row_index": 2, "plants_per_row": 15, "plant_spacing_m": 0.35, "row_spacing_m": 1.4, "y_m": 1.4},
                      {"id": "C", "row_id": "C", "row_index": 3, "plants_per_row": 22, "plant_spacing_m": 0.5, "row_spacing_m": 1.4, "y_m": 2.8}
                    ]
                  }
                ],
                "updated_at": "2026-05-30T19:07:22+05:30"
              },
              "fps": 10.0,
              "threads": 4,
              "classifier": "runtime_exports/classifier_openvino_model"
            }
        """.trimIndent()
    }
}
