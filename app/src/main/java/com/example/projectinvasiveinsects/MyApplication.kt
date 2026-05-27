// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects

import android.app.Application

class MyApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        val sharedPref = getSharedPreferences("AppPrefs", MODE_PRIVATE)
        sharedPref.edit().clear().apply()
    }
}