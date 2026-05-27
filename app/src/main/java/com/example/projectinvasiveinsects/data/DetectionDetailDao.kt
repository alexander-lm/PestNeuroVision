// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.projectinvasiveinsects.data.entity.DetectionDetail

@Dao
interface DetectionDetailDao {
    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertDetectionDetail(detectionDetail: DetectionDetail): Long

    @Query("SELECT * FROM tbl_detection_details")
    suspend fun getAllDetails(): List<DetectionDetail>

    @Query("SELECT * FROM tbl_detection_details WHERE detection_id = :detectionId")
    suspend fun getDetailsByDetectionId(detectionId: Int): List<DetectionDetail>


    @Query("""
    SELECT i.scientific_name || ' - ' || i.life_stage || ' (' || COUNT(*) || ')'
    FROM tbl_detection_details dd
    INNER JOIN tbl_insects i ON dd.insect_id = i.id
    WHERE dd.detection_id = :detectionId
    GROUP BY i.scientific_name, i.life_stage
""")
    suspend fun getInsectNamesByDetectionId(detectionId: Int): List<String>


    @Query("SELECT image FROM tbl_detection_details WHERE detection_id = :detectionId LIMIT 1")
    suspend fun getImagePathByDetectionId(detectionId: Int): String?

    @Query("DELETE FROM tbl_detection_details WHERE detection_id = :detectionId")
    suspend fun deleteDetailsByDetectionId(detectionId: Int)


    @Query("""
    SELECT id FROM tbl_insects 
    WHERE scientific_name || ' - ' || life_stage = :name 
    LIMIT 1
""")
    suspend fun getInsectIdByName(name: String): Int


    @Query("""
    SELECT i.scientific_name || ' - ' || i.life_stage AS name, COUNT(*) AS count
    FROM tbl_detection_details dd
    INNER JOIN tbl_insects i ON dd.insect_id = i.id
    INNER JOIN tbl_detections d ON dd.detection_id = d.id
    WHERE d.user_id = :userId
    GROUP BY i.scientific_name, i.life_stage
    ORDER BY count DESC
""")
    suspend fun getInsectDetectionCounts(userId: Int): List<InsectCount>

    data class InsectCount(
        val name: String,
        val count: Int
    )


    @Query("""
    SELECT d.date AS day, COUNT(*) AS count
    FROM tbl_detection_details dd
    INNER JOIN tbl_insects i ON dd.insect_id = i.id
    INNER JOIN tbl_detections d ON dd.detection_id = d.id
    WHERE d.user_id = :userId
      AND (i.scientific_name || ' - ' || i.life_stage) = :insectName
      AND d.date >= :startDate
    GROUP BY d.date
    ORDER BY d.date ASC
""")
    suspend fun getDailyCountsForInsect(
        userId: Int,
        insectName: String,
        startDate: String
    ): List<DailyCount>

    data class DailyCount(
        val day: String,
        val count: Int
    )

    @Query("""
    SELECT d.date AS day, COUNT(*) AS count,
           i.scientific_name || ' - ' || i.life_stage AS insectName
    FROM tbl_detection_details dd
    INNER JOIN tbl_insects i ON dd.insect_id = i.id
    INNER JOIN tbl_detections d ON dd.detection_id = d.id
    WHERE d.user_id = :userId
      AND (i.scientific_name || ' - ' || i.life_stage) IN (:insectNames)
      AND d.date >= :startDate
    GROUP BY d.date, i.scientific_name, i.life_stage
    ORDER BY i.scientific_name, d.date ASC
""")
    suspend fun getDailyCountsForInsects(
        userId: Int,
        insectNames: List<String>,
        startDate: String
    ): List<DailyCountWithInsect>

    data class DailyCountWithInsect(
        val day: String,
        val count: Int,
        val insectName: String
    )


    @Query("""
    SELECT i.scientific_name || ' - ' || i.life_stage AS name, COUNT(*) AS count
    FROM tbl_detection_details dd
    INNER JOIN tbl_insects i ON dd.insect_id = i.id
    INNER JOIN tbl_detections d ON dd.detection_id = d.id
    WHERE d.user_id = :userId
      AND d.date = :date
    GROUP BY i.scientific_name, i.life_stage
    ORDER BY count DESC
""")
    suspend fun getInsectCountsByDay(userId: Int, date: String): List<InsectCount>

    @Query("""
    SELECT i.scientific_name || ' - ' || i.life_stage AS name, COUNT(*) AS count
    FROM tbl_detection_details dd
    INNER JOIN tbl_insects i ON dd.insect_id = i.id
    INNER JOIN tbl_detections d ON dd.detection_id = d.id
    WHERE d.user_id = :userId
      AND d.date >= :startDate
      AND d.date <= :endDate
    GROUP BY i.scientific_name, i.life_stage
    ORDER BY count DESC
""")
    suspend fun getInsectCountsByRange(userId: Int, startDate: String, endDate: String): List<InsectCount>


    @Query("""
        SELECT i.scientific_name || ' (' || i.life_stage || ')' AS name, COUNT(*) AS count
        FROM tbl_detection_details dd
        INNER JOIN tbl_insects i ON dd.insect_id = i.id
        INNER JOIN tbl_detections d ON dd.detection_id = d.id
        WHERE d.user_id = :userId
        GROUP BY i.scientific_name, i.life_stage
        ORDER BY count DESC
    """)
    suspend fun getInsectDetectionCountsForGraph(userId: Int): List<InsectCount>

    @Query("""
        SELECT i.scientific_name || ' (' || i.life_stage || ')' AS name, COUNT(*) AS count
        FROM tbl_detection_details dd
        INNER JOIN tbl_insects i ON dd.insect_id = i.id
        INNER JOIN tbl_detections d ON dd.detection_id = d.id
        WHERE d.user_id = :userId
          AND d.date = :date
        GROUP BY i.scientific_name, i.life_stage
        ORDER BY count DESC
    """)
    suspend fun getInsectCountsByDayForGraph(userId: Int, date: String): List<InsectCount>

    @Query("""
        SELECT i.scientific_name || ' (' || i.life_stage || ')' AS name, COUNT(*) AS count
        FROM tbl_detection_details dd
        INNER JOIN tbl_insects i ON dd.insect_id = i.id
        INNER JOIN tbl_detections d ON dd.detection_id = d.id
        WHERE d.user_id = :userId
          AND d.date >= :startDate
          AND d.date <= :endDate
        GROUP BY i.scientific_name, i.life_stage
        ORDER BY count DESC
    """)
    suspend fun getInsectCountsByRangeForGraph(userId: Int, startDate: String, endDate: String): List<InsectCount>

    @Query("""
        SELECT d.date AS day, COUNT(*) AS count,
               i.scientific_name || ' (' || i.life_stage || ')' AS insectName
        FROM tbl_detection_details dd
        INNER JOIN tbl_insects i ON dd.insect_id = i.id
        INNER JOIN tbl_detections d ON dd.detection_id = d.id
        WHERE d.user_id = :userId
          AND (i.scientific_name || ' (' || i.life_stage || ')') IN (:insectNames)
          AND d.date >= :startDate
        GROUP BY d.date, i.scientific_name, i.life_stage
        ORDER BY i.scientific_name, d.date ASC
    """)
    suspend fun getDailyCountsForInsectsForGraph(
        userId: Int,
        insectNames: List<String>,
        startDate: String
    ): List<DailyCountWithInsect>

}