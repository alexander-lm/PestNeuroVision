package com.example.projectinvasiveinsects.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.projectinvasiveinsects.repository.DetectionRepository

@Suppress("UNCHECKED_CAST")
class DetectionViewModelFactory(
    private val repository: DetectionRepository
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(DetectionViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return DetectionViewModel(repository) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}