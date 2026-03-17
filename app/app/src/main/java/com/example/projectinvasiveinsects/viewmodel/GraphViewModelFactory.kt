package com.example.projectinvasiveinsects.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.projectinvasiveinsects.repository.DetectionRepository

class GraphViewModelFactory(private val repository: DetectionRepository) :
    ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(GraphViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return GraphViewModel(repository) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}