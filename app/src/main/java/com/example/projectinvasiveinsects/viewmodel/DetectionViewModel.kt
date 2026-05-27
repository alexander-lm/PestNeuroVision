// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.viewmodel

import android.util.Log
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.projectinvasiveinsects.data.entity.Detection
import com.example.projectinvasiveinsects.data.entity.DetectionDetail
import com.example.projectinvasiveinsects.repository.DetectionRepository
import com.example.projectinvasiveinsects.resource.Resource
import kotlinx.coroutines.launch

class DetectionViewModel(private val repository: DetectionRepository) : ViewModel() {

    private var _saveDetectionStatus = MutableLiveData<Resource<Long>>()
    val saveDetectionStatus: MutableLiveData<Resource<Long>>
        get() = _saveDetectionStatus

    fun saveDetection(
        userId: Int,
        date: String,
        time: String,
        imagePath: String,
        detections: List<DetectionDetail>
    ) {
        viewModelScope.launch {
            Log.d("FK_CHECK", "userId que se intenta guardar: $userId")
            _saveDetectionStatus.postValue(Resource.Loading(null))
            try {
                val detection = Detection(
                    userId = userId,
                    date = date,
                    time = time,
                    status = "1"
                )
                val detectionId = repository.saveDetection(detection)
                if (detectionId == -1L) throw Exception("Error al guardar la detección")

                detections.forEach { det ->
                    Log.d( "FK_CHECK", "Intentando guardar insect_id: ${det.insectId}")
                    val detectionDetail = DetectionDetail(
                        detectionId = detectionId.toInt(),
                        insectId = det.insectId,
                        accuracyPercentage = det.accuracyPercentage,
                        image = imagePath,
                        status = "1"
                    )
                    val detailId = repository.saveDetectionDetail(detectionDetail)


                    Log.d(
                        "DETECTION_DETAIL", """
        ┌─────────────────────────────
        │ Detail ID    : $detailId
        │ Detection ID : ${detectionDetail.detectionId}
        │ Insect ID    : ${detectionDetail.insectId}
        │ Accuracy     : ${"%.2f".format(detectionDetail.accuracyPercentage)}%
        │ Image        : ${detectionDetail.image}
        └─────────────────────────────
    """.trimIndent()
                    )

                        Log.d(
                            "DETECTION_DETAIL",
                            "Total detalles guardados: ${detections.size} para detectionId: $detectionId"
                        )


                    if (detailId == -1L) throw Exception("Error al guardar detalle insecto: ${det.insectId}")
                }


                val allDetails = repository.getAllDetails()
                Log.d("DETECTION_DETAIL", "=== TODOS LOS REGISTROS EN tbl_detection_details ===")
                allDetails.forEach { d ->
                    Log.d(
                        "DETECTION_DETAIL",
                        "id=${d.id} | detectionId=${d.detectionId} | insectId=${d.insectId} | accuracy=${
                            "%.2f".format(d.accuracyPercentage)
                        }%"
                    )
                }
                Log.d("DETECTION_DETAIL", "=== TOTAL: ${allDetails.size} registros ===")



                _saveDetectionStatus.postValue(Resource.Success(detectionId, true))
            } catch (e: Exception) {
                _saveDetectionStatus.postValue(
                    Resource.Error(
                        null,
                        e.message ?: "Error desconocido"
                    )
                )
            }
        }
    }

}