package com.example.projectinvasiveinsects.data

import com.example.projectinvasiveinsects.data.entity.Detection

data class DetectionWithInsects(
    val detection: Detection,
    val insectNames: List<String> = emptyList(),
    val imagePath: String = "",
    var expand: Boolean = false,
    var displayNumber: Int = 0
)