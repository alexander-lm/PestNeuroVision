package com.example.projectinvasiveinsects.data.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "tbl_control_measures",
    foreignKeys = [
        ForeignKey(
            entity = Insect::class,
            parentColumns = ["id"],
            childColumns = ["insect_id"],
            onDelete = ForeignKey.NO_ACTION,
            onUpdate = ForeignKey.NO_ACTION
        )
    ],
    indices = [Index("insect_id")]
)
data class ControlMeasure(
    @PrimaryKey(autoGenerate = true)
    val id: Int = 0,
    @ColumnInfo(name = "insect_id")
    val insectId: Int,
    val details: String,
    val status: String
)