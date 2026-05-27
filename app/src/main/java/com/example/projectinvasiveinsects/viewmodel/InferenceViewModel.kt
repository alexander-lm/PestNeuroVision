// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

// viewmodel/InferenceViewModel.kt
package com.example.projectinvasiveinsects.viewmodel

import android.graphics.Bitmap
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.projectinvasiveinsects.data.InferenceResult
import com.example.projectinvasiveinsects.repository.InferenceRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class InferenceViewModel(private val inferenceRepository: InferenceRepository) : ViewModel() {

    private val _inferenceResult = MutableLiveData<InferenceResult>()
    val inferenceResult: LiveData<InferenceResult> = _inferenceResult

    fun initModel(): Boolean = inferenceRepository.initInterpreter()

    fun runDetection(bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.IO) {
            _inferenceResult.postValue(InferenceResult.Loading)
            val result = inferenceRepository.runInference(bitmap)
            _inferenceResult.postValue(result)
        }
    }

    override fun onCleared() {
        super.onCleared()
        inferenceRepository.closeInterpreter()
    }
}

// viewmodel/InferenceViewModelFactory.kt
class InferenceViewModelFactory(
    private val repository: InferenceRepository
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(InferenceViewModel::class.java))
            return InferenceViewModel(repository) as T
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}