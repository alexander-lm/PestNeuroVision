// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.data


import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.example.projectinvasiveinsects.data.entity.ControlMeasure
import com.example.projectinvasiveinsects.data.entity.Detection
import com.example.projectinvasiveinsects.data.entity.DetectionDetail
import com.example.projectinvasiveinsects.data.entity.Insect

@Database(
    entities = [
        User::class,
        Insect::class,
        Detection::class,
        DetectionDetail::class,
        ControlMeasure::class],
    version = 1,
    exportSchema = true
)
abstract class InvasiveInsectsDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
    abstract fun insectDao(): InsectDao
    abstract fun detectionDao(): DetectionDao
    abstract fun detectionDetailDao(): DetectionDetailDao
    abstract fun controlMeasureDao(): ControlMeasureDao

    companion object {
        @Volatile
        private var INSTANCE: InvasiveInsectsDatabase? = null

        fun getDatabase(context: Context): InvasiveInsectsDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    InvasiveInsectsDatabase::class.java,
                    "invasive_insects_database"
                ).createFromAsset("database/invasive_insects_database.db")
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }


}