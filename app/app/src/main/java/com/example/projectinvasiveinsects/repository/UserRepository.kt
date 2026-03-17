package com.example.projectinvasiveinsects.repository

import androidx.lifecycle.LiveData
import com.example.projectinvasiveinsects.data.User
import com.example.projectinvasiveinsects.data.UserDao

class UserRepository(private val userDao: UserDao) {

    val allUsers: LiveData<List<User>> = userDao.getAll()

    suspend fun verifyLoginUser(user: String, password: String): User{
        return userDao.readLoginData(user, password)
    }

    suspend fun insertUser(user: User): Long {
        return userDao.insertUser(user)
    }
    suspend fun getUserById(userId: Int): User? {
        return userDao.getUserById(userId)
    }

    suspend fun updateUser(
        userId: Int,
        names: String,
        paternalSurname: String,
        maternalSurname: String,
        user: String,
        password: String
    ) {
        userDao.updateUser(userId, names, paternalSurname, maternalSurname, user, password)
    }

    suspend fun deactivateUser(userId: Int) {
        userDao.deactivateUser(userId)
    }

}