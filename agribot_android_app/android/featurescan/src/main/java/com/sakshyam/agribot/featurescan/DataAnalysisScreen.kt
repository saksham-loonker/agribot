package com.sakshyam.agribot.featurescan

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sakshyam.agribot.domain.logic.TreatmentGuide
import com.sakshyam.agribot.domain.logic.LeadingIssuePolicy
import com.sakshyam.agribot.domain.model.ObservedIssue
import com.sakshyam.agribot.domain.model.GpsPathStatus
import com.sakshyam.agribot.domain.model.RunFieldMap
import com.sakshyam.agribot.domain.model.RunFieldMapCell
import com.sakshyam.agribot.domain.model.TreatmentStatus
import com.sakshyam.agribot.designsystem.healthy
import com.sakshyam.agribot.designsystem.sick
import com.sakshyam.agribot.designsystem.uncertain
import com.sakshyam.agribot.featurescan.tracking.TrackedPlant
import kotlin.math.roundToInt

@Composable
private fun AutomaticCauseMap(
    plants: List<TrackedPlant>,
    fieldName: String,
    layout: com.sakshyam.agribot.domain.model.FieldLayout?,
    fieldMap: com.sakshyam.agribot.domain.model.RunFieldMap?,
    plannedCount: Int?,
) {
    val issue = leadingIssue(plants, plannedCount)
    val savedMapCount = fieldMap.observedCells().size
    Card(modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(24.dp)) {
        Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(stringResource(R.string.automatic_field_map), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Text(fieldName, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                StatusLegend()
            }
            if (plants.isEmpty() && savedMapCount > 0) {
                Text(
                    stringResource(R.string.saved_map_checked_count, savedMapCount),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                val leadingLabel = issue.label
                if (leadingLabel != null) {
                    SurfaceCauseBanner(leadingLabel, issue.affectedCount, issue.coverageLabel, issue.provisional)
                } else {
                    Text(stringResource(R.string.no_leading_issue, issue.coverageLabel), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            FieldMapGrid(fieldMap = fieldMap, layout = layout)
            Text(
                stringResource(R.string.map_legend),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

private fun RunFieldMap?.observedCells(): List<RunFieldMapCell> =
    this?.rows.orEmpty().flatMap { row -> row.cells }.filter { cell -> cell.status != "empty" }

private fun RunFieldMapCell.mapLabel(): String = label.substringAfter(": ", label).trim()

private fun String.isUncertainMapLabel(): Boolean =
    equals("Uncertain", ignoreCase = true) || equals("Unknown", ignoreCase = true)

@Composable
private fun SurfaceCauseBanner(label: String, count: Int, coverage: String, provisional: Boolean) {
    androidx.compose.material3.Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.sick.copy(alpha = 0.10f),
    ) {
        Row(modifier = Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(modifier = Modifier.size(10.dp).background(MaterialTheme.colorScheme.sick, RoundedCornerShape(50)))
            Spacer(Modifier.width(8.dp))
            Column {
                Text(stringResource(R.string.leading_observed_issue), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.sick, fontWeight = FontWeight.Bold)
                Text("${displayLabel(label)} · ${stringResource(R.string.plants_count, count, coverage)}", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                if (provisional) Text(stringResource(R.string.provisional_recheck), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun StatusLegend() {
    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
        LegendDot(MaterialTheme.colorScheme.healthy)
        LegendDot(MaterialTheme.colorScheme.uncertain)
        LegendDot(MaterialTheme.colorScheme.sick)
        LegendDot(MaterialTheme.colorScheme.outlineVariant)
    }
}

@Composable
private fun LegendDot(color: Color) {
    Box(modifier = Modifier.size(9.dp).background(color, RoundedCornerShape(50)))
}

@Composable
private fun PlantResultRow(plant: TrackedPlant, selected: Boolean, onSelect: () -> Unit) {
    val detailsDescription = stringResource(R.string.open_plant_details, plant.plantId)
    val healthLabel = if (plant.classificationPending) {
        stringResource(R.string.classification_pending)
    } else {
        displayLabel(plant.healthStatus)
    }
    Card(
        modifier = Modifier.fillMaxWidth().semantics { contentDescription = detailsDescription },
        onClick = onSelect,
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surface,
        ),
    ) {
        Row(modifier = Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(modifier = Modifier.size(12.dp).background(statusColor(plant), RoundedCornerShape(50)))
            Spacer(Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(plant.plantId, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                Text(
                    healthLabel,
                    style = MaterialTheme.typography.bodySmall,
                    color = statusColor(plant),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                if (!plant.isConfirmed) {
                    Text(
                        stringResource(R.string.preview_not_counted),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            Column(horizontalAlignment = Alignment.End) {
                Text("${(plant.confidence * 100).roundToInt()}%", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    Text("${formatPlantSize(plant)} · ${formatMeters(plant.distanceM)}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun EmptyResults(hasAnyPlants: Boolean, onClear: () -> Unit) {
    val title = stringResource(if (hasAnyPlants) R.string.no_matching_plants else R.string.no_plants_recorded)
    val body = stringResource(if (hasAnyPlants) R.string.try_all_or_clear else R.string.start_walk_results)
    Card(modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp)) {
        Column(modifier = Modifier.padding(22.dp), verticalArrangement = Arrangement.spacedBy(10.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text(
                body,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (hasAnyPlants) OutlinedButton(onClick = onClear) { Text(stringResource(R.string.show_all_plants)) }
        }
    }
}

@Composable
internal fun PlantDetailCard(plant: TrackedPlant, viewModel: SideScanViewModel) {
    val recommendation = TreatmentGuide.recommendationFor(plant.healthStatus)
    var note by remember(plant.plantId, plant.treatmentNote) { mutableStateOf(plant.treatmentNote.orEmpty()) }
    val detailsTitle = stringResource(R.string.plant_details, plant.plantId)
    val visibility = stringResource(if (plant.isOccluded || plant.classificationPending) R.string.needs_review_label else R.string.tracked_label)
    val healthLabel = if (plant.classificationPending) {
        stringResource(R.string.classification_pending)
    } else {
        displayLabel(plant.healthStatus)
    }
    Card(modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(22.dp)) {
        Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(11.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column {
                    Text(detailsTitle, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Text("$healthLabel · ${plant.measurementSource}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Text("${(plant.confidence * 100).roundToInt()}%", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = statusColor(plant))
            }
            HorizontalDivider()
            if (plant.classificationPending) {
                Text(
                    stringResource(R.string.classification_pending_detail),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            DetailLine(stringResource(R.string.plant_image_size), formatPlantSize(plant))
            DetailLine(stringResource(R.string.path_distance_label), formatMeters(plant.distanceM))
            DetailLine(stringResource(R.string.detections_label), stringResource(R.string.frames_count, plant.detectionCount))
            DetailLine(stringResource(R.string.visibility_label), visibility)
            recommendation?.let {
                SurfaceCauseBanner(it.diseaseLabel, 1, stringResource(R.string.checked_count, 1), true)
                Text(it.treatment, style = MaterialTheme.typography.bodySmall)
            }
            Text(stringResource(R.string.treatment_status_label), style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
            TreatmentStatusSelector(
                currentStatus = plant.treatmentStatus,
                onStatusChange = { viewModel.setPlantTreatment(plant.plantId, it, note.ifBlank { null }) },
            )
            if (plant.treatmentStatus != TreatmentStatus.NOT_APPLICABLE) {
                OutlinedTextField(
                    value = note,
                    onValueChange = {
                        note = it
                        viewModel.setPlantTreatment(plant.plantId, plant.treatmentStatus, it.ifBlank { null })
                    },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text(stringResource(R.string.treatment_note_label)) },
                    singleLine = true,
                )
            }
        }
    }
}

@Composable
private fun DetailLine(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun TreatmentStatusSelector(currentStatus: TreatmentStatus, onStatusChange: (TreatmentStatus) -> Unit) {
    val statuses = listOf(
        TreatmentStatus.NOT_TREATED to stringResource(R.string.not_treated),
        TreatmentStatus.TREATED to stringResource(R.string.treated),
        TreatmentStatus.RESCAN_NEEDED to stringResource(R.string.rescan),
        TreatmentStatus.NOT_APPLICABLE to stringResource(R.string.not_applicable),
    )
    Row(modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
        statuses.forEach { (status, label) ->
            FilterChip(selected = currentStatus == status, onClick = { onStatusChange(status) }, label = { Text(label, fontSize = 12.sp) })
        }
    }
}

@Composable
private fun filterLabel(filter: PlantFilter): String = when (filter) {
    PlantFilter.ALL -> stringResource(R.string.all_filter)
    PlantFilter.HEALTHY -> stringResource(R.string.healthy_label)
    PlantFilter.SICK -> stringResource(R.string.needs_care_label)
    PlantFilter.UNCERTAIN -> stringResource(R.string.uncertain_label)
    PlantFilter.OCCLUDED -> stringResource(R.string.occluded_label)
}

private fun isSick(plant: TrackedPlant): Boolean = !plant.isOccluded && !isUncertain(plant) && !plant.healthStatus.equals("healthy", true)

private fun isUncertain(plant: TrackedPlant): Boolean =
    plant.classificationPending ||
        plant.isOccluded ||
        plant.healthStatus.equals("uncertain", true) ||
        plant.healthStatus.equals("unknown", true)

private fun leadingIssue(plants: List<TrackedPlant>, plannedCount: Int?) = LeadingIssuePolicy.evaluate(
    observations = plants.map { plant ->
        ObservedIssue(
            stableId = plant.plantId,
            label = plant.healthStatus,
            confirmed = plant.detectionCount >= 2 && !plant.isOccluded,
            reviewRequired = isUncertain(plant),
            geometryValid = plant.currentBbox.width > 0f && plant.currentBbox.height > 0f,
        )
    },
    plannedCount = plannedCount,
)

private fun statusColor(plant: TrackedPlant): Color = when {
    isUncertain(plant) -> Color(0xFFD69E2E)
    isSick(plant) -> Color(0xFFC95B47)
    else -> Color(0xFF3F9B6B)
}

private fun displayLabel(label: String): String = label
    .replace('_', ' ')
    .replace("Pottassium Deficiency", "Potassium Deficiency", ignoreCase = true)

@Composable
private fun formatMeters(value: Float?): String = value?.let { stringResource(R.string.meters_value, it) } ?: stringResource(R.string.measurement_unavailable)

@Composable
private fun formatPlantSize(plant: TrackedPlant): String = when {
    plant.widthM != null && plant.heightM != null -> stringResource(R.string.physical_size, plant.widthM, plant.heightM)
    plant.relativeWidth != null && plant.relativeHeight != null -> stringResource(
        R.string.relative_size,
        plant.relativeWidth * 100f,
        plant.relativeHeight * 100f,
    )
    else -> stringResource(R.string.size_unavailable)
}

@Composable
private fun gpsPathLabel(status: GpsPathStatus): String = when (status) {
    GpsPathStatus.UNAVAILABLE -> stringResource(R.string.measurement_unavailable)
    GpsPathStatus.WAITING_FOR_FIX -> stringResource(R.string.gps_waiting_for_fix)
    GpsPathStatus.ANCHOR_ONLY -> stringResource(R.string.gps_anchor_only)
    GpsPathStatus.REFERENCE_PATH -> stringResource(R.string.gps_reference_only)
    GpsPathStatus.STALE -> stringResource(R.string.gps_stale)
    GpsPathStatus.MOTION_ONLY -> stringResource(R.string.gps_motion_only)
}
