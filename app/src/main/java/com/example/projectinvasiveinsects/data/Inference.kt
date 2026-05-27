// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.data

import android.graphics.Bitmap
import android.graphics.RectF

data class DetectionResult(
    val rect: RectF,
    val score: Float,
    val classIndex: Int,
    val label: String
)

sealed class InferenceResult {
    object Loading : InferenceResult()
    data class Success(val bitmap: Bitmap, val detections: List<DetectionResult>) : InferenceResult()
    data class Empty(val message: String) : InferenceResult()
    data class Error(val message: String) : InferenceResult()
}