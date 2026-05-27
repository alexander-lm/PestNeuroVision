// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.viewmodel

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.projectinvasiveinsects.data.DetectionDetailDao
import com.example.projectinvasiveinsects.repository.DetectionRepository
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale

class GraphViewModel(private val repository: DetectionRepository) : ViewModel() {

    private val _pieCounts = MutableLiveData<List<DetectionDetailDao.InsectCount>>()
    val pieCounts: LiveData<List<DetectionDetailDao.InsectCount>> = _pieCounts

    val insectCounts = MutableLiveData<List<DetectionDetailDao.InsectCount>>()
    private val _dailyCounts = MutableLiveData<List<Pair<String, Int>>>()
    val dailyCounts: LiveData<List<Pair<String, Int>>> = _dailyCounts

    private val _multiDailyCounts = MutableLiveData<Map<String, List<Pair<String, Int>>>>()
    val multiDailyCounts: LiveData<Map<String, List<Pair<String, Int>>>> = _multiDailyCounts

    fun loadInsectCounts(userId: Int) {
        viewModelScope.launch {
            insectCounts.postValue(repository.getInsectDetectionCountsForGraph(userId))
        }
    }

    fun loadDailyCountsForInsects(userId: Int, selectedInsects: List<String>) {
        viewModelScope.launch {
            val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())

            val last7Days = (6 downTo 0).map { daysAgo ->
                Calendar.getInstance().apply {
                    add(Calendar.DAY_OF_YEAR, -daysAgo)
                }.let { dateFormat.format(it.time) }
            }

            val startDate = last7Days.first()
            val rawCounts = repository.getDailyCountsForInsectsForGraph(userId, selectedInsects, startDate)
            val groupedByInsect = rawCounts.groupBy { it.insectName }

            val result = selectedInsects.associate { insectName ->
                val countMap = groupedByInsect[insectName]?.associate { it.day to it.count } ?: emptyMap()
                insectName to last7Days.map { day -> Pair(day, countMap[day] ?: 0) }
            }

            _multiDailyCounts.postValue(result)
        }
    }

    enum class BarFilter { DAY, WEEK, MONTH }

    private val _barCounts = MutableLiveData<List<DetectionDetailDao.InsectCount>>()
    val barCounts: LiveData<List<DetectionDetailDao.InsectCount>> = _barCounts

    fun loadBarCounts(userId: Int, filter: BarFilter) {
        viewModelScope.launch {
            val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
            val today = dateFormat.format(Calendar.getInstance().time)

            val counts = when (filter) {
                BarFilter.DAY -> repository.getInsectCountsByDayForGraph(userId, today)
                BarFilter.WEEK -> {
                    val start = Calendar.getInstance().apply {
                        add(Calendar.DAY_OF_YEAR, -6)
                    }.let { dateFormat.format(it.time) }
                    repository.getInsectCountsByRangeForGraph(userId, start, today)
                }
                BarFilter.MONTH -> {
                    val start = Calendar.getInstance().apply {
                        add(Calendar.DAY_OF_YEAR, -29)
                    }.let { dateFormat.format(it.time) }
                    repository.getInsectCountsByRangeForGraph(userId, start, today)
                }
            }
            _barCounts.postValue(counts)
        }
    }

    fun loadPieCounts(userId: Int, filter: BarFilter) {
        viewModelScope.launch {
            val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
            val today = dateFormat.format(Calendar.getInstance().time)

            val counts = when (filter) {
                BarFilter.DAY -> repository.getInsectCountsByDayForGraph(userId, today)
                BarFilter.WEEK -> {
                    val start = Calendar.getInstance().apply {
                        add(Calendar.DAY_OF_YEAR, -6)
                    }.let { dateFormat.format(it.time) }
                    repository.getInsectCountsByRangeForGraph(userId, start, today)
                }
                BarFilter.MONTH -> {
                    val start = Calendar.getInstance().apply {
                        add(Calendar.DAY_OF_YEAR, -29)
                    }.let { dateFormat.format(it.time) }
                    repository.getInsectCountsByRangeForGraph(userId, start, today)
                }
            }
            _pieCounts.postValue(counts)
        }
    }
}