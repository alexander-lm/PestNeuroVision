package com.example.projectinvasiveinsects.viewmodel

import android.widget.Toast
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.projectinvasiveinsects.data.User
import com.example.projectinvasiveinsects.repository.UserRepository
import com.example.projectinvasiveinsects.resource.Resource
import kotlinx.coroutines.async
import kotlinx.coroutines.launch



class UserViewModel(private val repository: UserRepository) : ViewModel() {

    class LoginException: Exception("Incorrect username or password")


    private var _getUserLoginDataStatus = MutableLiveData<Resource<User>>()
    val getUserLoginDataStatus: MutableLiveData<Resource<User>>
        get() = _getUserLoginDataStatus


    fun getUserLoginDataStatus(user: String, password: String) {
        viewModelScope.launch {
            _getUserLoginDataStatus.postValue(Resource.Loading(null))
            val data = repository.verifyLoginUser(user, password)
            try {
                if (data == null){
                    throw LoginException()
                }
                _getUserLoginDataStatus.postValue(Resource.Success(data, true))


            } catch (exception: Exception) {
                _getUserLoginDataStatus.postValue(Resource.Error(null, exception.message!!))
            }

        }
    }


}