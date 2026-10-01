package com.sakshyam.agribot

import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import com.sakshyam.agribot.featurescan.AgribotApp
import dagger.hilt.android.AndroidEntryPoint

/** AppCompatActivity so the in-app language choice applies on every supported Android version. */
@AndroidEntryPoint
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { AgribotApp() }
    }
}
