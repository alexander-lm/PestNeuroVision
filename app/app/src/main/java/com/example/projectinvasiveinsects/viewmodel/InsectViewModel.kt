package com.example.projectinvasiveinsects.viewmodel

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.projectinvasiveinsects.data.entity.ControlMeasure
import com.example.projectinvasiveinsects.data.entity.Insect
import com.example.projectinvasiveinsects.repository.InsectRepository
import kotlinx.coroutines.launch

class InsectViewModel(private val repository: InsectRepository) : ViewModel() {

    val insectList = MutableLiveData<List<Insect>>()
    val selectedInsect = MutableLiveData<Insect>()
    val controlMeasures = MutableLiveData<List<ControlMeasure>>()

    fun loadAllInsects() {
        viewModelScope.launch {
            insectList.postValue(repository.getAllInsects())
        }
    }

    fun loadInsectDetail(insectId: Int) {
        viewModelScope.launch {
            selectedInsect.postValue(repository.getInsectById(insectId))
            controlMeasures.postValue(repository.getControlMeasuresByInsectId(insectId))
        }
    }
}