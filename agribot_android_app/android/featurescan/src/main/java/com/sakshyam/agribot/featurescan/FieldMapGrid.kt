package com.sakshyam.agribot.featurescan

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sakshyam.agribot.designsystem.healthy
import com.sakshyam.agribot.designsystem.sick
import com.sakshyam.agribot.designsystem.uncertain
import com.sakshyam.agribot.domain.logic.RunFieldMapPresenter
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.RunFieldMap
import com.sakshyam.agribot.domain.model.RunFieldMapCell

/**
 * Renders every configured row and plant position. Empty cells are intentional:
 * they mean "not observed" and must not be silently dropped or called healthy.
 */
@Composable
fun FieldMapGrid(
    fieldMap: RunFieldMap?,
    layout: FieldLayout?,
    modifier: Modifier = Modifier,
) {
    val renderedMap = fieldMap ?: layout?.let { RunFieldMapPresenter.map(it, emptyList()) }
    if (renderedMap == null || renderedMap.rows.isEmpty()) {
        Text(
            text = "No field layout is available for this map yet.",
            modifier = modifier,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        return
    }

    val cellSize = if (LocalConfiguration.current.screenWidthDp < 380) 26.dp else 29.dp
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        renderedMap.rows.forEach { row ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = row.rowLabel,
                    modifier = Modifier.width(52.dp),
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                )
                Row(
                    modifier = Modifier
                        .weight(1f)
                        .horizontalScroll(rememberScrollState())
                        .semantics {
                            contentDescription = "${row.rowLabel}, ${row.cells.size} plant positions"
                        },
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    row.cells.forEach { cell ->
                        FieldMapCell(cell = cell, size = cellSize)
                    }
                }
            }
        }
    }
}

@Composable
private fun FieldMapCell(cell: RunFieldMapCell, size: androidx.compose.ui.unit.Dp) {
    val color = fieldMapStatusColor(cell.status)
    Box(
        modifier = Modifier
            .size(size)
            .background(color, RoundedCornerShape(7.dp))
            .semantics {
                contentDescription = "${cell.label}, ${cell.status}"
            },
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = cell.plantNumber.toString(),
            color = MaterialTheme.colorScheme.onSurface,
            fontSize = 9.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(1.dp),
        )
    }
}

@Composable
private fun fieldMapStatusColor(status: String): Color = when (status.lowercase()) {
    "ok" -> MaterialTheme.colorScheme.healthy.copy(alpha = 0.78f)
    "sick" -> MaterialTheme.colorScheme.sick.copy(alpha = 0.84f)
    "uncertain", "manual", "skipped" -> MaterialTheme.colorScheme.uncertain.copy(alpha = 0.84f)
    else -> MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.52f)
}
