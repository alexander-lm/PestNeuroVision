// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.data

import androidx.room.Dao
import androidx.room.Query
import com.example.projectinvasiveinsects.data.entity.ControlMeasure
import com.example.projectinvasiveinsects.data.entity.Insect

@Dao
interface InsectDao{

    @Query("SELECT * FROM tbl_insects WHERE status = '1'")
    suspend fun getAllInsects(): List<Insect>

    @Query("SELECT * FROM tbl_insects WHERE id = :insectId")
    suspend fun getInsectById(insectId: Int): Insect

    @Query("SELECT * FROM tbl_control_measures WHERE insect_id = :insectId")
    suspend fun getControlMeasuresByInsectId(insectId: Int): List<ControlMeasure>
}