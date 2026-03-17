package com.example.projectinvasiveinsects.data

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "tbl_users")
data class User(
    @PrimaryKey(autoGenerate = true)
    val id: Int = 0,
    val names: String,
    @ColumnInfo(name = "paternal_surname")
    val paternalSurname: String,
    @ColumnInfo(name = "maternal_surname")
    val maternalSurname: String,
    val user: String,
    val password: String,
    val status: String
)