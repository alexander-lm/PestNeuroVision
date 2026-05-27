// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.projectinvasiveinsects.repository.InsectRepository

class InsectViewModelFactory(private val repository: InsectRepository) :
    ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(InsectViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return InsectViewModel(repository) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}