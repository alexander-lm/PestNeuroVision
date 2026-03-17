package com.example.projectinvasiveinsects.ui

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.text.Spannable
import android.text.SpannableString
import android.text.SpannableStringBuilder
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.projectinvasiveinsects.R
import com.example.projectinvasiveinsects.data.InvasiveInsectsDatabase
import com.example.projectinvasiveinsects.databinding.FragmentGraphBinding
import com.example.projectinvasiveinsects.repository.DetectionRepository
import com.example.projectinvasiveinsects.viewmodel.GraphViewModel
import com.example.projectinvasiveinsects.viewmodel.GraphViewModelFactory
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter
import com.github.mikephil.charting.highlight.Highlight
import com.github.mikephil.charting.listener.OnChartValueSelectedListener
import java.text.SimpleDateFormat
import java.util.Locale

class GraphFragment : Fragment() {

    private lateinit var montserrat: Typeface
    private lateinit var montserratBold: Typeface
    private lateinit var montserratItalic: Typeface
    private var _binding: FragmentGraphBinding? = null
    private val binding get() = _binding!!
    private lateinit var viewModel: GraphViewModel

    private var allInsectNames = listOf<String>()
    private var selectedInsects = mutableListOf<String>()
    private var checkedItems = BooleanArray(0)
    private var currentUserId = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val detectionDao = InvasiveInsectsDatabase.getDatabase(requireContext()).detectionDao()
        val detectionDetailDao = InvasiveInsectsDatabase.getDatabase(requireContext()).detectionDetailDao()
        val repository = DetectionRepository(detectionDao, detectionDetailDao)
        viewModel = ViewModelProvider(this, GraphViewModelFactory(repository))
            .get(GraphViewModel::class.java)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentGraphBinding.inflate(inflater, container, false)
        montserrat = Typeface.createFromAsset(requireContext().assets, "fonts/montserrat_regular.ttf")
        montserratBold = Typeface.createFromAsset(requireContext().assets, "fonts/montserrat_bold.ttf")
        try {
            montserratItalic = Typeface.createFromAsset(requireContext().assets, "fonts/montserrat_italic.ttf")
        } catch (e: Exception) {
            montserratItalic = montserrat
        }

        binding.tvLineInfo.typeface = montserrat
        binding.tvBarInfo.typeface = montserrat
        binding.tvInsectInfo.typeface = montserrat

        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val prefs = requireContext().getSharedPreferences("AppPrefs", Context.MODE_PRIVATE)
        currentUserId = prefs.getInt("user_id", 0)

        viewModel.insectCounts.observe(viewLifecycleOwner) { counts ->
            if (counts.isNotEmpty()) {
                allInsectNames = counts.map { it.name }
                checkedItems = BooleanArray(allInsectNames.size) { it == 0 }
                selectedInsects = mutableListOf(allInsectNames[0])
                updateButtonText()
                viewModel.loadDailyCountsForInsects(currentUserId, selectedInsects)
            }
        }

        viewModel.multiDailyCounts.observe(viewLifecycleOwner) { data ->
            if (data.isNotEmpty()) {
                setupLineChart(data)
            }
        }

        binding.btnSelectInsects.setOnClickListener {
            showInsectSelectionDialog()
        }

        binding.tilSelectInsects.setEndIconOnClickListener {
            showInsectSelectionDialog()
        }

        if (viewModel.insectCounts.value == null) {
            viewModel.loadInsectCounts(currentUserId)
        }

        binding.rgBarFilter.setOnCheckedChangeListener { _, checkedId ->
            val filter = when (checkedId) {
                R.id.rbDay -> GraphViewModel.BarFilter.DAY
                R.id.rbWeek -> GraphViewModel.BarFilter.WEEK
                R.id.rbMonth -> GraphViewModel.BarFilter.MONTH
                else -> GraphViewModel.BarFilter.DAY
            }
            viewModel.loadBarCounts(currentUserId, filter)
        }

        viewModel.barCounts.observe(viewLifecycleOwner) { counts ->
            if (counts.isNotEmpty()) {
                setupBarChart(counts.map { it.name }, counts.map { it.count.toFloat() })
            } else {
                binding.barChart.setNoDataTextTypeface(montserrat)
                binding.barChart.setNoDataTextColor(Color.parseColor("#FFAA00"))
                binding.barChart.clear()
                binding.barChart.setNoDataText("No detections during this period.")
                binding.barChart.invalidate()
            }
        }

        viewModel.loadBarCounts(currentUserId, GraphViewModel.BarFilter.DAY)

        binding.rgPieFilter.setOnCheckedChangeListener { _, checkedId ->
            val filter = when (checkedId) {
                R.id.rbPieDay -> GraphViewModel.BarFilter.DAY
                R.id.rbPieWeek -> GraphViewModel.BarFilter.WEEK
                R.id.rbPieMonth -> GraphViewModel.BarFilter.MONTH
                else -> GraphViewModel.BarFilter.DAY
            }
            viewModel.loadPieCounts(currentUserId, filter)
        }

        viewModel.pieCounts.observe(viewLifecycleOwner) { counts ->
            if (counts.isNotEmpty()) {
                setupPieChart(counts.map { it.name }, counts.map { it.count.toFloat() })
            } else {
                binding.pieChart.setNoDataTextTypeface(montserrat)
                binding.pieChart.setNoDataTextColor(Color.parseColor("#FFAA00"))
                binding.pieChart.clear()
                binding.pieChart.setNoDataText("No detections during this period.")
                binding.pieChart.invalidate()
            }
        }

        viewModel.loadPieCounts(currentUserId, GraphViewModel.BarFilter.DAY)
        binding.lineChart.setNoDataTextTypeface(montserrat)
        binding.lineChart.setNoDataTextColor(Color.parseColor("#FFAA00"))
    }

    private fun italicSpan() = object : android.text.style.MetricAffectingSpan() {
        override fun updateMeasureState(p: android.text.TextPaint) { p.typeface = montserratItalic }
        override fun updateDrawState(p: android.text.TextPaint) { p.typeface = montserratItalic }
    }

    private fun regularSpan() = object : android.text.style.MetricAffectingSpan() {
        override fun updateMeasureState(p: android.text.TextPaint) { p.typeface = montserrat }
        override fun updateDrawState(p: android.text.TextPaint) { p.typeface = montserrat }
    }

    private fun boldSpan() = object : android.text.style.MetricAffectingSpan() {
        override fun updateMeasureState(p: android.text.TextPaint) { p.typeface = montserratBold }
        override fun updateDrawState(p: android.text.TextPaint) { p.typeface = montserratBold }
    }

    private fun buildInsectLabel(name: String): SpannableString {
        val idx = name.lastIndexOf(" (")
        return if (idx >= 0) {
            val spannable = SpannableString(name)
            spannable.setSpan(italicSpan(), 0, idx, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
            spannable.setSpan(regularSpan(), idx, spannable.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
            spannable
        } else {
            SpannableString(name)
        }
    }

    private fun buildInfoRow(label: String, value: CharSequence): SpannableStringBuilder {
        val builder = SpannableStringBuilder()
        val start = builder.length
        builder.append("$label: ")
        builder.setSpan(boldSpan(), start, builder.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
        builder.append(value)
        return builder
    }

    private fun setupBarChart(labels: List<String>, values: List<Float>) {
        val entries = values.mapIndexed { i, v -> BarEntry(i.toFloat(), v) }

        val colors = listOf(
            Color.parseColor("#247B2A"), Color.parseColor("#8D2B01"),
            Color.parseColor("#E62A1F"), Color.parseColor("#FFDA33"),
            Color.parseColor("#1A50E5"), Color.parseColor("#0099C8"),
            Color.parseColor("#0AA876"), Color.parseColor("#DC9628"),
            Color.parseColor("#A41711"), Color.parseColor("#000000")
        )

        val dataSet = BarDataSet(entries, "Detecciones").apply {
            this.colors = colors
            valueTextSize = 12f
            valueTextColor = Color.BLACK
        }

        val barData = BarData(dataSet)
        barData.setValueTypeface(montserrat)

        binding.barChart.apply {
            data = barData
            xAxis.apply {
                valueFormatter = object : com.github.mikephil.charting.formatter.ValueFormatter() {
                    override fun getFormattedValue(value: Float): String {
                        val index = value.toInt()
                        if (index >= labels.size) return ""
                        return labels[index]
                    }
                }
                granularity = 1f
                labelRotationAngle = -80f
                textSize = 9f
                typeface = montserrat
                isGranularityEnabled = true
                setLabelCount(labels.size)
            }
            axisLeft.apply {
                granularity = 1f
                typeface = montserrat
            }
            axisRight.isEnabled = false
            description.isEnabled = false
            legend.apply {
                isEnabled = false
                typeface = montserrat
                textSize = 10f
                orientation = Legend.LegendOrientation.HORIZONTAL
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                verticalAlignment = Legend.LegendVerticalAlignment.BOTTOM
                isWordWrapEnabled = true
                yOffset = 8f
            }
            setFitBars(true)
            setExtraBottomOffset(8f)
            animateY(800)
            post {
                setExtraTopOffset(0f)
                notifyDataSetChanged()
                invalidate()
            }

            setOnChartValueSelectedListener(object : OnChartValueSelectedListener {
                override fun onValueSelected(e: Entry?, h: Highlight?) {
                    if (e == null || h == null) return
                    val index = h.x.toInt()
                    if (index >= labels.size) return
                    val cantidad = values[index].toInt()

                    val builder = SpannableStringBuilder()
                    builder.append(buildInfoRow("Insect", buildInsectLabel(labels[index])))
                    builder.append("\n")
                    builder.append(buildInfoRow("Number of Detections", "$cantidad"))

                    binding.tvBarInfo.apply {
                        visibility = View.VISIBLE
                        text = builder
                    }
                }
                override fun onNothingSelected() {
                    binding.tvBarInfo.visibility = View.GONE
                }
            })
            invalidate()
        }
    }

    private fun setupPieChart(labels: List<String>, values: List<Float>) {
        val entries = values.mapIndexed { i, v -> PieEntry(v, labels[i].take(12)) }

        val colors = listOf(
            Color.parseColor("#247B2A"), Color.parseColor("#8D2B01"),
            Color.parseColor("#E62A1F"), Color.parseColor("#FFDA33"),
            Color.parseColor("#1A50E5"), Color.parseColor("#0099C8"),
            Color.parseColor("#0AA876"), Color.parseColor("#DC9628"),
            Color.parseColor("#A41711"), Color.parseColor("#000000")
        )

        val dataSet = PieDataSet(entries, "").apply {
            this.colors = colors
            valueTextSize = 12f
            valueTextColor = Color.WHITE
            sliceSpace = 0f
            valueFormatter = object : com.github.mikephil.charting.formatter.ValueFormatter() {
                override fun getFormattedValue(value: Float) = "${"%.1f".format(value)}%"
            }
        }

        val pieData = PieData(dataSet)
        pieData.setValueTypeface(montserrat)

        binding.pieChart.apply {
            data = pieData
            setEntryLabelTypeface(montserrat)
            description.isEnabled = false
            isDrawHoleEnabled = true
            holeRadius = 40f
            setUsePercentValues(true)
            setHoleColor(Color.parseColor("#F5F5F5"))
            legend.apply {
                isEnabled = false
                typeface = montserrat
                textSize = 10f
                orientation = Legend.LegendOrientation.HORIZONTAL
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                verticalAlignment = Legend.LegendVerticalAlignment.BOTTOM
                isWordWrapEnabled = true
                yOffset = 8f
            }
            setExtraBottomOffset(8f)
            animateY(800)

            val total = values.sum()
            setOnChartValueSelectedListener(object : OnChartValueSelectedListener {
                override fun onValueSelected(e: Entry?, h: Highlight?) {
                    if (e == null || h == null) return
                    val index = h.x.toInt()
                    if (index >= labels.size) return
                    val porcentaje = if (total > 0) (values[index] / total * 100) else 0f

                    val builder = SpannableStringBuilder()
                    builder.append(buildInfoRow("Insect", buildInsectLabel(labels[index])))
                    builder.append("\n")
                    builder.append(buildInfoRow("Percentage", "${"%.1f".format(porcentaje)}%"))

                    binding.tvInsectInfo.apply {
                        visibility = View.VISIBLE
                        text = builder
                    }
                }
                override fun onNothingSelected() {
                    binding.tvInsectInfo.visibility = View.GONE
                }
            })
            invalidate()
        }
    }

    private val lineColors = listOf(
        Color.parseColor("#247B2A"), Color.parseColor("#8D2B01"),
        Color.parseColor("#E62A1F"), Color.parseColor("#FFDA33"),
        Color.parseColor("#1A50E5"), Color.parseColor("#0099C8"),
        Color.parseColor("#0AA876"), Color.parseColor("#DC9628"),
        Color.parseColor("#A41711"), Color.parseColor("#000000")
    )

    private fun setupLineChart(data: Map<String, List<Pair<String, Int>>>) {
        val inputFormat = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val outputFormat = SimpleDateFormat("dd/MM", Locale.getDefault())

        val labels = data.values.first().map { (day, _) ->
            try { outputFormat.format(inputFormat.parse(day)!!) } catch (e: Exception) { day }
        }

        val dataSets = data.entries.mapIndexed { colorIndex, (insectName, dailyPairs) ->
            val entries = dailyPairs.mapIndexed { i, (_, count) -> Entry(i.toFloat(), count.toFloat()) }
            val color = lineColors[colorIndex % lineColors.size]

            LineDataSet(entries, insectName).apply {
                this.color = color
                setCircleColor(color)
                lineWidth = 2.5f
                circleRadius = 4f
                setDrawFilled(data.size == 1)
                fillColor = color
                fillAlpha = 60
                valueTextSize = 9f
                valueTextColor = Color.BLACK
                mode = LineDataSet.Mode.CUBIC_BEZIER
                highLightColor = color
            }
        }

        val lineData = LineData(dataSets)
        lineData.setValueTypeface(montserrat)

        binding.lineChart.apply {
            this.data = lineData
            xAxis.apply {
                valueFormatter = IndexAxisValueFormatter(labels)
                granularity = 1f
                position = XAxis.XAxisPosition.BOTTOM
                setLabelCount(labels.size, true)
                textSize = 10f
                typeface = montserrat
                setDrawGridLines(false)
            }
            axisLeft.apply {
                granularity = 1f
                axisMinimum = 0f
                typeface = montserrat
            }
            axisRight.isEnabled = false
            description.isEnabled = false
            legend.apply {
                isEnabled = data.size > 1
                typeface = montserrat
                textSize = 10f
                orientation = Legend.LegendOrientation.HORIZONTAL
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                verticalAlignment = Legend.LegendVerticalAlignment.BOTTOM
                isWordWrapEnabled = true
                yOffset = 2f
            }
            setExtraBottomOffset(20f)
            setTouchEnabled(true)
            isDragEnabled = false
            setScaleEnabled(false)
            animateX(600)

            setOnChartValueSelectedListener(object : OnChartValueSelectedListener {
                override fun onValueSelected(e: Entry?, h: Highlight?) {
                    if (e == null || h == null) return
                    val index = e.x.toInt()
                    val dataSetIndex = h.dataSetIndex
                    val insectName = dataSets[dataSetIndex].label ?: return
                    val dayLabel = if (index < labels.size) labels[index] else ""
                    val cantidad = e.y.toInt()
                    val esHoy = index == labels.lastIndex

                    val builder = SpannableStringBuilder()
                    builder.append(buildInfoRow("Insect", buildInsectLabel(insectName)))
                    builder.append("\n")
                    builder.append(buildInfoRow("Date", if (esHoy) "Today" else dayLabel))
                    builder.append("\n")
                    builder.append(buildInfoRow("Number of Detections", "$cantidad"))

                    binding.tvLineInfo.apply {
                        visibility = View.VISIBLE
                        text = builder
                    }
                }
                override fun onNothingSelected() {
                    binding.tvLineInfo.visibility = View.GONE
                }
            })
            invalidate()
        }
    }

    private fun showInsectSelectionDialog() {
        if (allInsectNames.isEmpty()) return

        val options = listOf("All") + allInsectNames

        val spannableOptions = options.map { name ->
            val idx = name.lastIndexOf(" (")
            if (idx >= 0) {
                val spannable = SpannableString(name)
                spannable.setSpan(
                    object : android.text.style.MetricAffectingSpan() {
                        override fun updateMeasureState(p: android.text.TextPaint) {
                            p.typeface = montserratItalic
                            p.textSize = 13f * resources.displayMetrics.scaledDensity
                        }
                        override fun updateDrawState(p: android.text.TextPaint) {
                            p.typeface = montserratItalic
                            p.textSize = 13f * resources.displayMetrics.scaledDensity
                        }
                    },
                    0, idx,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
                spannable.setSpan(
                    object : android.text.style.MetricAffectingSpan() {
                        override fun updateMeasureState(p: android.text.TextPaint) {
                            p.typeface = montserrat
                            p.textSize = 13f * resources.displayMetrics.scaledDensity
                        }
                        override fun updateDrawState(p: android.text.TextPaint) {
                            p.typeface = montserrat
                            p.textSize = 13f * resources.displayMetrics.scaledDensity
                        }
                    },
                    idx, spannable.length,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
                spannable
            } else {
                val spannable = SpannableString(name)
                spannable.setSpan(
                    object : android.text.style.MetricAffectingSpan() {
                        override fun updateMeasureState(p: android.text.TextPaint) {
                            p.typeface = montserrat
                            p.textSize = 13f * resources.displayMetrics.scaledDensity
                        }
                        override fun updateDrawState(p: android.text.TextPaint) {
                            p.typeface = montserrat
                            p.textSize = 13f * resources.displayMetrics.scaledDensity
                        }
                    },
                    0, spannable.length,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
                spannable
            }
        }.toTypedArray()

        val dialogChecked = BooleanArray(options.size).apply {
            if (selectedInsects.size == allInsectNames.size) this[0] = true
            allInsectNames.forEachIndexed { i, name ->
                this[i + 1] = selectedInsects.contains(name)
            }
        }

        val dialog = AlertDialog.Builder(requireContext())
            .setTitle("Select insects")
            .setMultiChoiceItems(spannableOptions, dialogChecked) { dialog, which, isChecked ->
                if (which == 0) {
                    for (i in dialogChecked.indices) dialogChecked[i] = isChecked
                    (dialog as AlertDialog).listView.apply {
                        for (i in 0 until count) setItemChecked(i, isChecked)
                    }
                } else {
                    dialogChecked[which] = isChecked
                    if (!isChecked) {
                        dialogChecked[0] = false
                        (dialog as AlertDialog).listView.setItemChecked(0, false)
                    }
                }
            }
            .setPositiveButton("Apply") { _, _ ->
                selectedInsects = allInsectNames.filterIndexed { i, _ ->
                    dialogChecked[i + 1]
                }.toMutableList()

                if (selectedInsects.isEmpty()) {
                    selectedInsects = mutableListOf(allInsectNames[0])
                    dialogChecked[1] = true
                }

                updateButtonText()
                viewModel.loadDailyCountsForInsects(currentUserId, selectedInsects)
            }
            .setNegativeButton("Cancel", null)
            .create()

        dialog.setOnShowListener {
            dialog.window?.setBackgroundDrawableResource(R.drawable.dialog_bg_rounded)

            val titleView = dialog.window?.findViewById<android.widget.TextView>(
                androidx.appcompat.R.id.alertTitle
            )
            titleView?.typeface = montserratBold

            dialog.getButton(AlertDialog.BUTTON_POSITIVE).apply {
                typeface = montserratBold
                textSize = 15f
            }
            dialog.getButton(AlertDialog.BUTTON_NEGATIVE).apply {
                typeface = montserratBold
                textSize = 15f
            }
        }

        dialog.show()
    }

    private fun updateButtonText() {
        binding.btnSelectInsects.setText(
            when {
                selectedInsects.size == allInsectNames.size -> "All insects"
                selectedInsects.size == 1 -> selectedInsects[0].take(30)
                else -> "${selectedInsects.size} selected insects"
            },
            false
        )
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}