package com.sakshyam.agribot.featurescan.results

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sakshyam.agribot.domain.logic.RunFieldMapPresenter
import com.sakshyam.agribot.domain.logic.RunSummaryReducer
import com.sakshyam.agribot.domain.model.ExportedFile
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.ReportText
import com.sakshyam.agribot.domain.model.Run
import com.sakshyam.agribot.domain.model.RunFieldMap
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunSummary
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.repository.ExportRepository
import com.sakshyam.agribot.domain.repository.FieldLayoutRepository
import com.sakshyam.agribot.domain.repository.RunRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

data class ResultsUiState(
    val loading: Boolean = true,
    val missing: Boolean = false,
    val loadFailed: Boolean = false,
    val run: Run? = null,
    val layout: FieldLayout? = null,
    /** Latest decision per plant, field order. */
    val plants: List<RecordedDecision> = emptyList(),
    val summary: RunSummary? = null,
    val map: RunFieldMap? = null,
    val walkedM: Double? = null,
    val share: ExportedFile? = null,
    val deleted: Boolean = false,
    val error: String? = null,
) {
    val notSeen: Int get() = plants.count { it.status == DecisionStatus.SKIPPED }
    val attention: List<RecordedDecision> get() = plants.filter { it.status == DecisionStatus.OK && it.action == PlantHealthAction.INSPECT_OR_TREAT }
}

@HiltViewModel
class ResultsViewModel @Inject constructor(
    savedState: SavedStateHandle,
    private val runs: RunRepository,
    private val layouts: FieldLayoutRepository,
    private val exports: ExportRepository,
) : ViewModel() {
    private val runId = RunId(checkNotNull(savedState.get<String>("runId")))
    private val _state = MutableStateFlow(ResultsUiState())
    val state: StateFlow<ResultsUiState> = _state.asStateFlow()

    init { load() }

    private fun load() {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val run = runs.runById(runId) ?: run { _state.update { it.copy(loading = false, missing = true) }; return@launch }
                val decisions = runs.decisionsForRun(runId)
                val layout = layouts.layoutById(run.fieldLayoutId)
                val plants = decisions.groupBy { it.plantKey ?: "seq-${it.sequence}" }.values.map { d -> d.maxBy { it.sequence } }
                    .sortedWith(compareBy({ it.rowIndex ?: 0 }, { it.plantNumber ?: 0 }, { it.sequence }))
                val walked = runs.eventsForRun(runId).lastOrNull { it.type == "gps_trail" }?.let {
                    runCatching { Json.parseToJsonElement(it.payloadJson).jsonObject["length_m"]!!.jsonPrimitive.double }.getOrNull()
                }
                _state.update {
                    it.copy(
                        loading = false, run = run, layout = layout, plants = plants,
                        summary = RunSummaryReducer.reduce(runId, decisions),
                        map = if (layout != null && layout.rows.isNotEmpty()) RunFieldMapPresenter.map(layout, decisions) else null,
                        walkedM = walked,
                    )
                }
            }.onFailure { _state.update { it.copy(loading = false, loadFailed = true) } }
        }
    }

    fun exportPdf(text: ReportText) = export { exports.exportPdf(runId, text) }
    fun exportCsv() = export { exports.exportCsv(runId) }
    fun exportZip() = export { exports.exportBundle(runId, null) }

    private fun export(block: suspend () -> ExportedFile) {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { block() }
                .onSuccess { f -> _state.update { it.copy(share = f) } }
                .onFailure { e -> _state.update { it.copy(error = e.message) } }
        }
    }

    fun shareHandled() = _state.update { it.copy(share = null) }
    fun clearError() = _state.update { it.copy(error = null) }

    fun delete() {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { exports.deleteRunArtifacts(runId); runs.deleteRun(runId) }
                .onSuccess { _state.update { it.copy(deleted = true) } }
                .onFailure { e -> _state.update { it.copy(error = e.message) } }
        }
    }
}
