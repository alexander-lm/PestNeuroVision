// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.viewmodel

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.projectinvasiveinsects.data.User
import com.example.projectinvasiveinsects.repository.UserRepository
import kotlinx.coroutines.launch

class RegisterViewModel(private val repository: UserRepository) : ViewModel() {

    val registerStatus = MutableLiveData<Boolean>()

    fun registerUser(user: User) {
        viewModelScope.launch {
            try {
                repository.insertUser(user)
                registerStatus.postValue(true)
            } catch (e: Exception) {
                registerStatus.postValue(false)
            }
        }
    }
}