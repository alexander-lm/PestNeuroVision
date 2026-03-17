package com.example.projectinvasiveinsects.data.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "tbl_insects")
data class Insect(
    @PrimaryKey(autoGenerate = true)
    val id: Int = 0,
    @ColumnInfo(name = "scientific_name")
    val scientificName: String,
    @ColumnInfo(name = "common_name")
    val commonName: String,
    @ColumnInfo(name = "life_stage")
    val lifeStage: String,
    val description: String,
    val status: String
)