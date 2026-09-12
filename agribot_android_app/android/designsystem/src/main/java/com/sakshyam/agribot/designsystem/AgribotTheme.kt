package com.sakshyam.agribot.designsystem

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.sp

private val LightColors = lightColorScheme(
    primary = Color(0xFF176342),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD6F0DE),
    onPrimaryContainer = Color(0xFF062C1A),
    secondary = Color(0xFF41618B),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFF0E7D2),
    onSecondaryContainer = Color(0xFF302A1D),
    tertiary = Color(0xFF9A4F38),
    error = Color(0xFFB3261E),
    background = Color(0xFFF7F8F3),
    surface = Color(0xFFFDFEFA),
    surfaceVariant = Color(0xFFE8EEE7),
    onSurface = Color(0xFF1A1C19),
    onSurfaceVariant = Color(0xFF46534A),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF7AD0A5),
    onPrimary = Color(0xFF003824),
    primaryContainer = Color(0xFF0B4C32),
    onPrimaryContainer = Color(0xFFD6F0DE),
    secondary = Color(0xFFADC6FF),
    secondaryContainer = Color(0xFF373328),
    onSecondaryContainer = Color(0xFFF4E8CF),
    tertiary = Color(0xFFFFB694),
    background = Color(0xFF101511),
    surface = Color(0xFF111411),
    surfaceVariant = Color(0xFF28332B),
    onSurface = Color(0xFFE2E4DE),
    onSurfaceVariant = Color(0xFFB9C8BB),
)

private val SunColors = lightColorScheme(
    primary = Color(0xFF003C24),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFFFF4C2),
    onPrimaryContainer = Color.Black,
    secondary = Color(0xFF173F72),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFFFF4C2),
    onSecondaryContainer = Color.Black,
    tertiary = Color(0xFF7A2F1D),
    error = Color(0xFF8F0000),
    background = Color.White,
    surface = Color.White,
    surfaceVariant = Color(0xFFF0F0E5),
    onSurface = Color.Black,
    onSurfaceVariant = Color(0xFF263229),
)

private val AgribotTypography = Typography().let { base ->
    base.copy(
        bodyLarge = base.bodyLarge.copy(fontSize = 17.sp, lineHeight = 25.sp),
        bodyMedium = base.bodyMedium.copy(fontSize = 15.sp, lineHeight = 22.sp),
        labelLarge = base.labelLarge.copy(fontSize = 14.sp, lineHeight = 20.sp),
        displaySmall = base.displaySmall.copy(fontFamily = FontFamily.SansSerif, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold, fontSize = 34.sp),
        headlineMedium = base.headlineMedium.copy(fontFamily = FontFamily.SansSerif, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold),
        headlineSmall = base.headlineSmall.copy(fontFamily = FontFamily.SansSerif, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold),
        titleMedium = base.titleMedium.copy(fontFamily = FontFamily.SansSerif, fontWeight = androidx.compose.ui.text.font.FontWeight.SemiBold),
    )
}

@Composable
fun AgribotTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    sunMode: Boolean = false,
    content: @Composable () -> Unit,
) {
    val colors = when {
        sunMode -> SunColors
        darkTheme -> DarkColors
        else -> LightColors
    }
    MaterialTheme(
        colorScheme = colors,
        typography = AgribotTypography,
        content = content,
    )
}

val ColorScheme.healthy: Color
    get() = if (surface.luminance() < 0.2f) Color(0xFF8FE0AE) else Color(0xFF176B40)

val ColorScheme.sick: Color
    get() = if (surface.luminance() < 0.2f) Color(0xFFFF9B8D) else Color(0xFFB3261E)

val ColorScheme.uncertain: Color
    get() = if (surface.luminance() < 0.2f) Color(0xFFFFD866) else Color(0xFF795900)

val ColorScheme.sunModeAccent: Color
    get() = Color(0xFFFFE066)

val ColorScheme.sunModeText: Color
    get() = Color(0xFF000000)
