// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.projectinvasiveinsects.repository.DetectionRepository

class HistoryViewModelFactory(private val repository: DetectionRepository) :
    ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(HistoryViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return HistoryViewModel(repository) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}