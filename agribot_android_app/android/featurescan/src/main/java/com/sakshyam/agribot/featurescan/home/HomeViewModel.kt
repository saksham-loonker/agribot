package com.sakshyam.agribot.featurescan.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sakshyam.agribot.domain.logic.RunSummaryReducer
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.Run
import com.sakshyam.agribot.domain.model.RunSummary
import com.sakshyam.agribot.domain.model.ScanSettings
import com.sakshyam.agribot.domain.repository.FieldLayoutRepository
import com.sakshyam.agribot.domain.repository.RunRepository
import com.sakshyam.agribot.domain.repository.ScanSettingsRepository
import com.sakshyam.agribot.domain.scan.LeafVision
import com.sakshyam.agribot.domain.scan.VisionStatus
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class RunListItem(val run: Run, val fieldName: String?, val summary: RunSummary, val first: RecordedDecision?)

@HiltViewModel
class HomeViewModel @Inject constructor(
    private val runs: RunRepository,
    private val layouts: FieldLayoutRepository,
    private val vision: LeafVision,
    settingsRepo: ScanSettingsRepository,
) : ViewModel() {
    private val _items = MutableStateFlow<List<RunListItem>?>(null)
    val items: StateFlow<List<RunListItem>?> = _items.asStateFlow()
    val visionStatus: StateFlow<VisionStatus> = vision.status
    val settings: StateFlow<ScanSettings?> = settingsRepo.observeSettings().stateIn(viewModelScope, SharingStarted.Eagerly, null)

    init { vision.warmUp() }

    fun refresh(limit: Int = 50) {
        viewModelScope.launch(Dispatchers.IO) {
            val list = runs.recentRuns(limit).map { r ->
                val d = runs.decisionsForRun(r.runId)
                RunListItem(r, layouts.layoutById(r.fieldLayoutId)?.name, RunSummaryReducer.reduce(r.runId, d), d.minByOrNull { it.sequence })
            }.filter { it.summary.decisions > 0 }
            _items.value = list
        }
    }

    fun retryModels() = vision.warmUp()
}
