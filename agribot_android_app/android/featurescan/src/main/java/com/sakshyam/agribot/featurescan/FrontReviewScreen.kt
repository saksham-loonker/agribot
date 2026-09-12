package com.sakshyam.agribot.featurescan

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.sakshyam.agribot.designsystem.healthy
import com.sakshyam.agribot.designsystem.sick
import com.sakshyam.agribot.designsystem.uncertain
import com.sakshyam.agribot.domain.model.DecisionStatus

@Composable
fun FrontReviewScreen(
    state: ScanUiState,
    viewModel: SideScanViewModel,
) {
    val review = state.pendingFrontReview ?: return

    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = PaddingValues(top = 18.dp, bottom = 28.dp),
    ) {
        item {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(stringResource(R.string.review_before_saving), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                    Text(stringResource(R.string.front_burst_review), style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
                    Text(
                        stringResource(R.string.check_automatic_calls),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                OutlinedButton(onClick = viewModel::discardFrontReview) { Text(stringResource(R.string.discard)) }
            }
        }

        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(22.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
            ) {
                Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(stringResource(R.string.plant_calls, review.decisionCount), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text(
                        stringResource(R.string.frames_candidates, review.capturedFrameCount, review.candidateCount),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.76f),
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.14f))
                    Text("${review.reason} · ${review.detectorStatus}", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onPrimaryContainer)
                }
            }
        }

        item {
            Text(stringResource(R.string.review_each_call), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        }

        if (review.decisions.isEmpty()) {
            item {
                Card(modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp)) {
                    Text(
                        stringResource(R.string.no_plants_burst),
                        modifier = Modifier.padding(18.dp),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        } else {
            items(review.decisions, key = { it.index }) { decision ->
                FrontReviewDecisionCard(decision, viewModel)
            }
        }

        item {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedButton(
                    onClick = viewModel::markFrontReviewUncertain,
                    modifier = Modifier.fillMaxWidth().height(50.dp),
                ) {
                    Text(stringResource(R.string.mark_all_review))
                }
                Button(
                    onClick = viewModel::confirmFrontReview,
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                ) {
                    Text(stringResource(R.string.confirm_save_map), fontWeight = FontWeight.Bold)
                }
                Text(
                    stringResource(R.string.local_run_note),
                    modifier = Modifier.fillMaxWidth(),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun FrontReviewDecisionCard(
    decision: FrontReviewDecisionUiState,
    viewModel: SideScanViewModel,
) {
    val statusColor = when {
        decision.statusLabel.equals("ok", true) -> MaterialTheme.colorScheme.healthy
        decision.statusLabel.equals("uncertain", true) -> MaterialTheme.colorScheme.uncertain
        else -> MaterialTheme.colorScheme.sick
    }
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics { contentDescription = "Review ${decision.positionLabel}" },
        shape = RoundedCornerShape(18.dp),
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(decision.positionLabel, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    Text(
                        formatReviewLabel(decision.label),
                        style = MaterialTheme.typography.bodyMedium,
                        color = statusColor,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Column(horizontalAlignment = androidx.compose.ui.Alignment.End) {
                    Text(decision.confidenceLabel, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    Text(decision.statusLabel, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ReviewActionButton(stringResource(R.string.healthy_action), MaterialTheme.colorScheme.healthy, Modifier.weight(1f)) {
                    viewModel.correctFrontReviewDecision(decision.index, "Healthy", DecisionStatus.OK)
                }
                ReviewActionButton(stringResource(R.string.needs_care_action), MaterialTheme.colorScheme.sick, Modifier.weight(1f)) {
                    viewModel.correctFrontReviewDecision(decision.index, "Sick", DecisionStatus.MANUAL)
                }
                ReviewActionButton(stringResource(R.string.review_action), MaterialTheme.colorScheme.uncertain, Modifier.weight(1f)) {
                    viewModel.correctFrontReviewDecision(decision.index, "Uncertain", DecisionStatus.UNCERTAIN)
                }
            }
        }
    }
}

@Composable
private fun ReviewActionButton(label: String, color: Color, modifier: Modifier, onClick: () -> Unit) {
    OutlinedButton(onClick = onClick, modifier = modifier) {
        Text(label, color = color, style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
    }
}

private fun formatReviewLabel(label: String): String = label.replace('_', ' ')
