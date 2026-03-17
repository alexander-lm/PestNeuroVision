package com.example.projectinvasiveinsects.data

import androidx.lifecycle.LiveData
import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface UserDao {

    @Query("SELECT * FROM tbl_users")
    fun getAll(): LiveData<List<User>>

    @Query("SELECT * FROM tbl_users WHERE user LIKE :user AND password LIKE :password AND status = '1'")
    suspend fun readLoginData(user: String, password: String): User

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertUser(user: User): Long


    @Query("SELECT * FROM tbl_users WHERE id = :userId LIMIT 1")
    suspend fun getUserById(userId: Int): User?

    @Query("""
    UPDATE tbl_users 
    SET names = :names, 
        paternal_surname = :paternalSurname, 
        maternal_surname = :maternalSurname, 
        user = :user, 
        password = :password 
    WHERE id = :userId
""")
    suspend fun updateUser(
        userId: Int,
        names: String,
        paternalSurname: String,
        maternalSurname: String,
        user: String,
        password: String
    )

    @Query("UPDATE tbl_users SET status = '0' WHERE id = :userId")
    suspend fun deactivateUser(userId: Int)
}