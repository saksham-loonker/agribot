package com.sakshyam.agribot.featurescan.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Help
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.sakshyam.agribot.designsystem.healthy
import com.sakshyam.agribot.designsystem.sick
import com.sakshyam.agribot.designsystem.uncertain
import com.sakshyam.agribot.designsystem.unknownCondition
import com.sakshyam.agribot.domain.scan.AnalyzedLeaf
import com.sakshyam.agribot.domain.scan.PlantVerdict
import com.sakshyam.agribot.domain.scan.VerdictKind
import com.sakshyam.agribot.domain.scan.VisionModelInfo
import com.sakshyam.agribot.featurescan.Conditions
import com.sakshyam.agribot.featurescan.R

enum class Tone { HEALTHY, ATTENTION, UNSURE, UNKNOWN, NEUTRAL }

@Composable
fun Tone.color(): Color = when (this) {
    Tone.HEALTHY -> MaterialTheme.colorScheme.healthy
    Tone.ATTENTION -> MaterialTheme.colorScheme.sick
    Tone.UNSURE -> MaterialTheme.colorScheme.uncertain
    Tone.UNKNOWN -> MaterialTheme.colorScheme.unknownCondition
    Tone.NEUTRAL -> MaterialTheme.colorScheme.outline
}

fun VerdictKind.tone(): Tone = when (this) {
    VerdictKind.HEALTHY -> Tone.HEALTHY
    VerdictKind.DISEASE -> Tone.ATTENTION
    VerdictKind.UNKNOWN_CONDITION -> Tone.UNKNOWN
    VerdictKind.UNSURE, VerdictKind.NEED_MORE_VIEWS -> Tone.UNSURE
}

/** A leaf box normalised to the analysed frame (which equals the visible preview area). */
data class LeafBox(val left: Float, val top: Float, val right: Float, val bottom: Float, val tone: Tone) {
    companion object {
        fun of(leaf: AnalyzedLeaf, frameW: Int, frameH: Int, info: VisionModelInfo): LeafBox {
            val tone = when (leaf.labelIndex) {
                info.healthyIndex -> Tone.HEALTHY
                info.otherIndex -> Tone.UNKNOWN
                else -> Tone.ATTENTION
            }
            return LeafBox(leaf.box.left / frameW, leaf.box.top / frameH, leaf.box.right / frameW, leaf.box.bottom / frameH, tone)
        }
    }
}

@Composable
fun LeafOverlay(boxes: List<LeafBox>, modifier: Modifier = Modifier) {
    val colors = Tone.entries.associateWith { it.color() }
    Canvas(modifier) {
        for (b in boxes) {
            val c = colors.getValue(b.tone)
            drawRoundRect(
                color = c,
                topLeft = Offset(b.left * size.width, b.top * size.height),
                size = Size((b.right - b.left) * size.width, (b.bottom - b.top) * size.height),
                cornerRadius = androidx.compose.ui.geometry.CornerRadius(12f, 12f),
                style = Stroke(width = 5f),
            )
        }
    }
}

@Composable
fun ModeCard(title: String, body: String, icon: ImageVector, onClick: () -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true) {
    Card(
        modifier = modifier.fillMaxWidth().heightIn(min = 120.dp).clickable(enabled = enabled, onClick = onClick),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
        shape = RoundedCornerShape(20.dp),
    ) {
        Row(Modifier.padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(56.dp).background(MaterialTheme.colorScheme.primary, CircleShape), contentAlignment = Alignment.Center) {
                Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.onPrimary, modifier = Modifier.size(30.dp))
            }
            Spacer(Modifier.width(16.dp))
            Column {
                Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer)
                Spacer(Modifier.height(4.dp))
                Text(body, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onPrimaryContainer)
            }
        }
    }
}

@Composable
fun StatTile(value: String, label: String, tone: Tone, modifier: Modifier = Modifier) {
    Card(modifier, colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.padding(12.dp).fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(value, style = MaterialTheme.typography.headlineMedium, color = tone.color(), fontWeight = FontWeight.Bold)
            Text(label, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
fun confidenceText(p: Float): String = stringResource(
    when {
        p >= 0.80f -> R.string.confidence_high
        p >= 0.60f -> R.string.confidence_medium
        else -> R.string.confidence_low
    },
)

/** Localised headline for a verdict. */
@Composable
fun verdictTitle(v: PlantVerdict, labels: List<String>): String {
    val key = v.labelIndex?.let { labels.getOrNull(it) }
    return when (v.kind) {
        VerdictKind.HEALTHY -> stringResource(R.string.verdict_healthy)
        VerdictKind.DISEASE -> if (v.partial) stringResource(R.string.verdict_disease_partial, stringResource(Conditions.nameRes(key))) else stringResource(Conditions.nameRes(key))
        VerdictKind.UNKNOWN_CONDITION -> stringResource(R.string.verdict_unknown)
        VerdictKind.UNSURE, VerdictKind.NEED_MORE_VIEWS -> stringResource(R.string.verdict_unsure)
    }
}

@Composable
fun VerdictCard(v: PlantVerdict, labels: List<String>, modifier: Modifier = Modifier) {
    val tone = v.kind.tone()
    val key = v.labelIndex?.let { labels.getOrNull(it) }
    val runnerKey = v.runnerUpIndex?.let { labels.getOrNull(it) }
    val icon = when (tone) {
        Tone.HEALTHY -> Icons.Filled.CheckCircle
        Tone.ATTENTION -> Icons.Filled.Warning
        Tone.UNKNOWN -> Icons.Filled.Info
        else -> Icons.Filled.Help
    }
    val title = verdictTitle(v, labels)
    Card(modifier.fillMaxWidth().semantics { contentDescription = title }, shape = RoundedCornerShape(20.dp)) {
        Column {
            Row(Modifier.fillMaxWidth().background(tone.color()).padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(icon, contentDescription = null, tint = Color.White, modifier = Modifier.size(36.dp))
                Spacer(Modifier.width(12.dp))
                Text(title, style = MaterialTheme.typography.headlineSmall, color = Color.White, fontWeight = FontWeight.Bold)
            }
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                if (v.kind != VerdictKind.UNSURE && v.kind != VerdictKind.NEED_MORE_VIEWS) {
                    Text("${confidenceText(v.confidence)} · ${stringResource(R.string.percent, (v.confidence * 100).toInt())}", style = MaterialTheme.typography.titleMedium)
                    LinearProgressIndicator(progress = { v.confidence.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth().height(8.dp), color = tone.color())
                }
                if (v.leavesSeen > 0) Text(stringResource(R.string.leaves_agree, v.leavesAgreeing, v.leavesSeen), style = MaterialTheme.typography.bodyMedium)
                when {
                    v.kind == VerdictKind.UNSURE || v.kind == VerdictKind.NEED_MORE_VIEWS ->
                        Text(stringResource(R.string.verdict_unsure_body, stringResource(Conditions.nameRes(key)), stringResource(Conditions.nameRes(runnerKey))))
                    v.partial -> Text(stringResource(R.string.verdict_partial_body, v.leavesAgreeing))
                }
                val cond = Conditions.forKey(if (v.kind == VerdictKind.UNKNOWN_CONDITION) "Other" else key)
                if (cond != null && v.kind != VerdictKind.UNSURE && v.kind != VerdictKind.NEED_MORE_VIEWS) {
                    Text(stringResource(R.string.what_it_is), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Text(stringResource(cond.body), style = MaterialTheme.typography.bodyLarge)
                    Text(stringResource(R.string.what_to_do), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    cond.advice.forEach { Text("• " + stringResource(it), style = MaterialTheme.typography.bodyLarge) }
                }
                Text(stringResource(R.string.disclaimer_short), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}
