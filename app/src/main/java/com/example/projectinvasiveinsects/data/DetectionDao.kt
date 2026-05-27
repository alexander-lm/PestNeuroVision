// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.projectinvasiveinsects.data.entity.Detection
import com.example.projectinvasiveinsects.data.entity.DetectionDetail

@Dao
interface DetectionDao {
    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertDetection(detection: Detection): Long

    @Query("SELECT * FROM tbl_detections ORDER BY id DESC")
    suspend fun getAllDetections(): List<Detection>


    @Query("SELECT * FROM tbl_detections WHERE user_id = :userId ORDER BY id DESC")
    suspend fun getDetectionsByUser(userId: Int): List<Detection>

    @Query("DELETE FROM tbl_detections WHERE id = :detectionId")
    suspend fun deleteDetection(detectionId: Int)
}