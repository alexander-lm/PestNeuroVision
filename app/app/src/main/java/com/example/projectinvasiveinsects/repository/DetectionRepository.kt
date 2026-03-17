package com.example.projectinvasiveinsects.repository

import com.example.projectinvasiveinsects.data.DetectionDao
import com.example.projectinvasiveinsects.data.DetectionDetailDao
import com.example.projectinvasiveinsects.data.entity.Detection
import com.example.projectinvasiveinsects.data.entity.DetectionDetail

class DetectionRepository(
    private val detectionDao: DetectionDao,
    private val detectionDetailDao: DetectionDetailDao
) {
    suspend fun getAllDetections(): List<Detection> {
        return detectionDao.getAllDetections()
    }
    suspend fun getDetectionsByUser(userId: Int): List<Detection> {
        return detectionDao.getDetectionsByUser(userId)
    }

    suspend fun getAllDetails(): List<DetectionDetail> {
        return detectionDetailDao.getAllDetails()
    }

    suspend fun saveDetection(detection: Detection): Long {
        return detectionDao.insertDetection(detection)
    }

    suspend fun saveDetectionDetail(detectionDetail: DetectionDetail): Long {
        return detectionDetailDao.insertDetectionDetail(detectionDetail)
    }

    suspend fun getInsectNamesByDetectionId(detectionId: Int): List<String> {
        return detectionDetailDao.getInsectNamesByDetectionId(detectionId)
    }

    suspend fun deleteDetection(detectionId: Int) {
        detectionDao.deleteDetection(detectionId)
    }

    suspend fun getImagePathByDetectionId(detectionId: Int): String? {
        return detectionDetailDao.getImagePathByDetectionId(detectionId)
    }

    suspend fun deleteDetailsByDetectionId(detectionId: Int) {
        detectionDetailDao.deleteDetailsByDetectionId(detectionId)
    }

    suspend fun getInsectIdByName(name: String): Int {
        return detectionDetailDao.getInsectIdByName(name)
    }

    suspend fun getInsectDetectionCounts(userId: Int): List<DetectionDetailDao.InsectCount> {
        return detectionDetailDao.getInsectDetectionCounts(userId)
    }

    suspend fun getDailyCountsForInsects(
        userId: Int,
        insectNames: List<String>,
        startDate: String
    ): List<DetectionDetailDao.DailyCountWithInsect> {
        return detectionDetailDao.getDailyCountsForInsects(userId, insectNames, startDate)
    }

    suspend fun getInsectCountsByDay(userId: Int, date: String): List<DetectionDetailDao.InsectCount> {
        return detectionDetailDao.getInsectCountsByDay(userId, date)
    }

    suspend fun getInsectCountsByRange(userId: Int, startDate: String, endDate: String): List<DetectionDetailDao.InsectCount> {
        return detectionDetailDao.getInsectCountsByRange(userId, startDate, endDate)
    }

    suspend fun getInsectDetectionCountsForGraph(userId: Int): List<DetectionDetailDao.InsectCount> {
        return detectionDetailDao.getInsectDetectionCountsForGraph(userId)
    }

    suspend fun getInsectCountsByDayForGraph(userId: Int, date: String): List<DetectionDetailDao.InsectCount> {
        return detectionDetailDao.getInsectCountsByDayForGraph(userId, date)
    }

    suspend fun getInsectCountsByRangeForGraph(userId: Int, startDate: String, endDate: String): List<DetectionDetailDao.InsectCount> {
        return detectionDetailDao.getInsectCountsByRangeForGraph(userId, startDate, endDate)
    }

    suspend fun getDailyCountsForInsectsForGraph(
        userId: Int,
        insectNames: List<String>,
        startDate: String
    ): List<DetectionDetailDao.DailyCountWithInsect> {
        return detectionDetailDao.getDailyCountsForInsectsForGraph(userId, insectNames, startDate)
    }
}