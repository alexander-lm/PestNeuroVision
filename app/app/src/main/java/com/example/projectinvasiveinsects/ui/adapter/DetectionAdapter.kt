package com.example.projectinvasiveinsects.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.AnimationUtils
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.projectinvasiveinsects.R
import com.example.projectinvasiveinsects.data.DetectionWithInsects
import com.example.projectinvasiveinsects.databinding.SingleItemBinding

class DetectionAdapter(
    private var detections: List<DetectionWithInsects>,
    private val onViewPhoto: (String) -> Unit,
    private val onDelete: (Int) -> Unit,
    private val onInsectClick: (String) -> Unit
) : RecyclerView.Adapter<DetectionAdapter.ViewHolder>() {

    inner class ViewHolder(val binding: SingleItemBinding) :
        RecyclerView.ViewHolder(binding.root)


    private var searchQuery: String = ""

    fun setSearchQuery(query: String) {
        searchQuery = query
        notifyDataSetChanged()
    }


    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = SingleItemBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        with(holder) {
            with(detections[position]) {

                binding.tvDetectionId.text = "Detection ${detections[position].displayNumber}"
                binding.tvDetectionIdDb.text = "Id: ${detection.id}"
                binding.tvDate.text = "Date: ${detection.date}"
                binding.tvTime.text = "Time: ${detection.time}"

                binding.layoutInsects.removeAllViews()
                insectNames.forEach { name ->
                    val tv = TextView(binding.root.context).apply {

                        val regex = Regex("^(.+?) - (.+?) \\((\\d+)\\)$")
                        val match = regex.find(name)

                        val spannable = if (match != null) {
                            val insectName = match.groupValues[1]
                            val lifeStage = match.groupValues[2]
                            val count = match.groupValues[3]

                            val fullText = "• $insectName (${lifeStage}) - (${count})"
                            val spannableStr = android.text.SpannableString(fullText)

                            val nameStart = 2
                            val nameEnd = nameStart + insectName.length
                            spannableStr.setSpan(
                                android.text.style.StyleSpan(android.graphics.Typeface.ITALIC),
                                nameStart, nameEnd,
                                android.text.Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                            )

                            spannableStr.setSpan(
                                android.text.style.UnderlineSpan(),
                                2, spannableStr.length,
                                android.text.Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                            )

                            if (searchQuery.isNotEmpty()) {
                                val lowerText = fullText.lowercase()
                                val lowerQuery = searchQuery.lowercase()
                                var index = lowerText.indexOf(lowerQuery)
                                while (index >= 0) {
                                    spannableStr.setSpan(
                                        android.text.style.BackgroundColorSpan(
                                            android.graphics.Color.parseColor("#F7C000")
                                        ),
                                        index, index + lowerQuery.length,
                                        android.text.Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                                    )
                                    index = lowerText.indexOf(lowerQuery, index + 1)
                                }
                            }
                            spannableStr

                        } else {
                            android.text.SpannableString("• $name").also {
                                it.setSpan(android.text.style.UnderlineSpan(), 2, it.length,
                                    android.text.Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
                            }
                        }

                        text = spannable
                        textSize = 14f
                        setPadding(0, 8, 0, 8)
                        setTextColor(android.graphics.Color.parseColor("#1B5E20"))
                        typeface = androidx.core.content.res.ResourcesCompat.getFont(
                            binding.root.context,
                            R.font.montserrat_regular
                        )
                        setOnClickListener { onInsectClick(name) }
                    }
                    binding.layoutInsects.addView(tv)
                }

                binding.expandedView.visibility = if (expand) View.VISIBLE else View.GONE
                binding.ivArrow.rotation = if (expand) 180f else 0f

                binding.cardLayout.setOnClickListener {
                    this.expand = !this.expand

                    if (this.expand) {
                        val anim = AnimationUtils.loadAnimation(
                            binding.root.context, R.anim.expand
                        )
                        binding.expandedView.startAnimation(anim)
                        binding.expandedView.visibility = View.VISIBLE
                    } else {
                        val anim = AnimationUtils.loadAnimation(
                            binding.root.context, R.anim.collapse
                        )
                        binding.expandedView.startAnimation(anim)
                        binding.expandedView.visibility = View.GONE
                    }

                    binding.ivArrow.animate()
                        .rotation(if (this.expand) 180f else 0f)
                        .setDuration(300)
                        .start()
                }

                binding.btnViewPhoto.setOnClickListener {
                    onViewPhoto(imagePath)
                }

                binding.btnDelete.setOnClickListener {
                    onDelete(detection.id)
                }
            }
        }
    }

    override fun getItemCount() = detections.size

    fun updateList(newList: List<DetectionWithInsects>) {
        detections = newList
        notifyDataSetChanged()
    }
}