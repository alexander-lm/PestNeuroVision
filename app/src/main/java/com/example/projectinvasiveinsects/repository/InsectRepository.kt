// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.repository

import com.example.projectinvasiveinsects.data.InsectDao
import com.example.projectinvasiveinsects.data.entity.ControlMeasure
import com.example.projectinvasiveinsects.data.entity.Insect

class InsectRepository(private val insectDao: InsectDao) {

    suspend fun getAllInsects(): List<Insect> {
        return insectDao.getAllInsects()
    }

    suspend fun getInsectById(insectId: Int): Insect {
        return insectDao.getInsectById(insectId)
    }

    suspend fun getControlMeasuresByInsectId(insectId: Int): List<ControlMeasure> {
        return insectDao.getControlMeasuresByInsectId(insectId)
    }
}