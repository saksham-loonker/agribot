package com.sakshyam.agribot.featurescan

import com.sakshyam.agribot.domain.model.ThermalStatus
import kotlin.test.Test
import kotlin.test.assertEquals

class DeviceHealthStatusPresenterTest {
    @Test
    fun reportsNoneWhenThermalAndBatteryAreNormal() {
        val status = DeviceHealthStatusPresenter.describe(
            thermalStatus = ThermalStatus.NONE,
            battery = BatterySnapshot(percent = 82, low = false),
        )

        assertEquals("none", status)
    }

    @Test
    fun reportsThermalAndLowBatteryWarningsTogether() {
        val status = DeviceHealthStatusPresenter.describe(
            thermalStatus = ThermalStatus.SEVERE,
            battery = BatterySnapshot(percent = 12, low = true),
        )

        assertEquals("thermal severe; battery low 12%", status)
    }

    @Test
    fun reportsBatteryPercentWhenKnownWithoutWarning() {
        val status = DeviceHealthStatusPresenter.describe(
            thermalStatus = ThermalStatus.NONE,
            battery = BatterySnapshot(percent = 47, low = false),
            includeNormalBattery = true,
        )

        assertEquals("battery 47%", status)
    }
}
