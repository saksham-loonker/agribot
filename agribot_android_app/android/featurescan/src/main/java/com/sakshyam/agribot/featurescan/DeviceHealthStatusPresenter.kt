package com.sakshyam.agribot.featurescan

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import com.sakshyam.agribot.domain.model.ThermalStatus

data class BatterySnapshot(
    val percent: Int?,
    val low: Boolean,
)

object DeviceHealthStatusPresenter {
    fun describe(
        thermalStatus: ThermalStatus,
        battery: BatterySnapshot,
        includeNormalBattery: Boolean = false,
    ): String {
        val warnings = buildList {
            if (thermalStatus == ThermalStatus.MODERATE || thermalStatus == ThermalStatus.SEVERE || thermalStatus == ThermalStatus.CRITICAL) {
                add("thermal ${thermalStatus.name.lowercase()}")
            }
            if (battery.low) {
                add("battery low ${battery.percent?.let { "$it%" } ?: "unknown"}")
            } else if (includeNormalBattery && battery.percent != null) {
                add("battery ${battery.percent}%")
            }
        }
        return warnings.ifEmpty { listOf("none") }.joinToString("; ")
    }
}

object AndroidBatteryStatusReader {
    fun read(context: Context): BatterySnapshot {
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            ?: return BatterySnapshot(percent = null, low = false)
        val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        val percent = if (level >= 0 && scale > 0) {
            ((level * 100.0) / scale).toInt().coerceIn(0, 100)
        } else {
            null
        }
        val low = intent.getBooleanExtra(BatteryManager.EXTRA_BATTERY_LOW, false) ||
            (percent != null && percent <= LOW_BATTERY_PERCENT)
        return BatterySnapshot(percent = percent, low = low)
    }

    private const val LOW_BATTERY_PERCENT = 15
}
