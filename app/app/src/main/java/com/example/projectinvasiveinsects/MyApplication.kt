package com.example.projectinvasiveinsects

import android.app.Application

class MyApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        val sharedPref = getSharedPreferences("AppPrefs", MODE_PRIVATE)
        sharedPref.edit().clear().apply()
    }
}