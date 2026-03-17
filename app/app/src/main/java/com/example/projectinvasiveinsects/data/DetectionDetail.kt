package com.example.projectinvasiveinsects.data.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "tbl_detection_details",
    foreignKeys = [
        ForeignKey(
            entity = Detection::class,
            parentColumns = ["id"],
            childColumns = ["detection_id"],
            onDelete = ForeignKey.NO_ACTION,
            onUpdate = ForeignKey.NO_ACTION
        ),
        ForeignKey(
            entity = Insect::class,
            parentColumns = ["id"],
            childColumns = ["insect_id"],
            onDelete = ForeignKey.NO_ACTION,
            onUpdate = ForeignKey.NO_ACTION
        )
    ],
    indices = [
        Index("detection_id"),
        Index("insect_id")
    ]
)
data class DetectionDetail(
    @PrimaryKey(autoGenerate = true)
    val id: Int = 0,
    @ColumnInfo(name = "detection_id")
    val detectionId: Int,
    @ColumnInfo(name = "insect_id")
    val insectId: Int,
    @ColumnInfo(name = "accuracy_percentage")
    val accuracyPercentage: Double,
    val image: String,
    val status: String
)