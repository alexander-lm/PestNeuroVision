package com.example.projectinvasiveinsects.viewmodel

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.projectinvasiveinsects.data.User
import com.example.projectinvasiveinsects.repository.UserRepository
import kotlinx.coroutines.launch

class AccountViewModel(private val repository: UserRepository) : ViewModel() {

    val user = MutableLiveData<User?>()
    val updateStatus = MutableLiveData<Boolean>()
    val deactivateStatus = MutableLiveData<Boolean>()

    fun loadUser(userId: Int) {
        viewModelScope.launch {
            try {
                val result = repository.getUserById(userId)
                user.postValue(result)
            } catch (e: Exception) {
                user.postValue(null)
            }
        }
    }

    fun updateUser(
        userId: Int,
        names: String,
        paternalSurname: String,
        maternalSurname: String,
        userLogin: String,
        password: String
    ) {
        viewModelScope.launch {
            try {
                repository.updateUser(userId, names, paternalSurname, maternalSurname, userLogin, password)
                updateStatus.postValue(true)
            } catch (e: Exception) {
                updateStatus.postValue(false)
            }
        }
    }

    fun deactivateUser(userId: Int) {
        viewModelScope.launch {
            try {
                repository.deactivateUser(userId)
                deactivateStatus.postValue(true)
            } catch (e: Exception) {
                deactivateStatus.postValue(false)
            }
        }
    }
}