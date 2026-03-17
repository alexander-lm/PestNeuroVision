package com.example.projectinvasiveinsects.viewmodel

import android.util.Log
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.projectinvasiveinsects.data.DetectionWithInsects
import com.example.projectinvasiveinsects.data.entity.Detection
import com.example.projectinvasiveinsects.repository.DetectionRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class HistoryViewModel(private val repository: DetectionRepository) : ViewModel() {

    val detectionList = MutableLiveData<List<DetectionWithInsects>>()
    val deleteStatus = MutableLiveData<Boolean>()

    private var currentUserId: Int = 0

    fun loadDetections(userId: Int) {
        currentUserId = userId
        viewModelScope.launch {
            val detections = repository.getDetectionsByUser(userId)

            val result = detections.mapIndexed { index, detection ->
                val insectNames = repository.getInsectNamesByDetectionId(detection.id)
                val imagePath = repository.getImagePathByDetectionId(detection.id) ?: ""
                DetectionWithInsects(
                    detection = detection,
                    insectNames = insectNames,
                    imagePath = imagePath,
                    displayNumber = detections.size - index
                )
            }

            detectionList.postValue(result)
        }
    }

    fun deleteDetection(detectionId: Int) {
        viewModelScope.launch {
            val imagePath = repository.getImagePathByDetectionId(detectionId)

            if (!imagePath.isNullOrEmpty()) {
                val imageFile = File(imagePath)
                if (imageFile.exists()) {
                    imageFile.delete()
                    Log.d("DETECTION", "Imagen eliminada: $imagePath")
                }
            }

            repository.deleteDetailsByDetectionId(detectionId)
            repository.deleteDetection(detectionId)

            deleteStatus.postValue(true)
            loadDetections(currentUserId)
        }
    }

    fun getInsectIdByName(name: String, onResult: (Int) -> Unit) {
        viewModelScope.launch {
            val cleanName = name.replace(Regex("\\s*\\(\\d+\\)"), "").trim()
            Log.d("INSECT_CLICK", "Nombre limpio: '$cleanName'")
            val insectId = repository.getInsectIdByName(cleanName)
            withContext(Dispatchers.Main) {
                onResult(insectId)
            }
        }
    }
}