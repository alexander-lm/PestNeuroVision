package com.example.projectinvasiveinsects.ui

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.os.bundleOf
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.fragment.findNavController
import androidx.recyclerview.widget.RecyclerView
import com.example.projectinvasiveinsects.R
import com.example.projectinvasiveinsects.data.InvasiveInsectsDatabase
import com.example.projectinvasiveinsects.data.entity.Insect
import com.example.projectinvasiveinsects.databinding.FragmentInsectListBinding
import com.example.projectinvasiveinsects.databinding.ItemInsectBinding
import com.example.projectinvasiveinsects.repository.InsectRepository
import com.example.projectinvasiveinsects.viewmodel.InsectViewModel
import com.example.projectinvasiveinsects.viewmodel.InsectViewModelFactory

class InsectListFragment : Fragment() {

    private var _binding: FragmentInsectListBinding? = null
    private val binding get() = _binding!!
    private lateinit var viewModel: InsectViewModel
    private lateinit var insectAdapter: InsectAdapter

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentInsectListBinding.inflate(inflater, container, false)

        val insectDao = InvasiveInsectsDatabase.getDatabase(requireContext()).insectDao()
        val repository = InsectRepository(insectDao)
        viewModel = ViewModelProvider(this, InsectViewModelFactory(repository))
            .get(InsectViewModel::class.java)

        viewModel.insectList.observe(viewLifecycleOwner) { insects ->
            insectAdapter = InsectAdapter(insects) { insect ->
                findNavController().navigate(
                    R.id.action_insectListFragment_to_informationFragment,
                    bundleOf("insect_id" to insect.id)
                )
            }
            binding.rvInsects.adapter = insectAdapter

            binding.etSearch.addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                    val query = s.toString().trim().lowercase()
                    val filtered = insects.filter {
                        it.scientificName.lowercase().contains(query) ||
                                it.commonName.lowercase().contains(query) ||
                                it.lifeStage.lowercase().contains(query)
                    }
                    insectAdapter.updateList(filtered, query)
                }
                override fun afterTextChanged(s: Editable?) {}
            })
        }

        viewModel.loadAllInsects()

        return binding.root
    }


    override fun onResume() {
        super.onResume()
        binding.etSearch.setText("")
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}

class InsectAdapter(
    insects: List<Insect>,
    private val onClick: (Insect) -> Unit
) : RecyclerView.Adapter<InsectAdapter.ViewHolder>() {

    private var searchQuery: String = ""
    private var filteredInsects: List<Insect> = insects

    inner class ViewHolder(val binding: ItemInsectBinding) :
        RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemInsectBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val insect = filteredInsects[position]

        holder.binding.tvCommonName.text = highlightText(insect.commonName, searchQuery)
        holder.binding.tvScientificName.text = highlightText(insect.scientificName, searchQuery)
        holder.binding.tvLifeStage.text = highlightText(insect.lifeStage, searchQuery)

        val imageName = "${insect.scientificName.trim().lowercase().replace(" ", "_")}_${insect.lifeStage.trim().lowercase().replace(" ", "_")}"
        val resId = holder.itemView.context.resources.getIdentifier(
            imageName, "drawable", holder.itemView.context.packageName
        )

        if (resId != 0) {
            holder.binding.ivInsect.setImageResource(resId)
        } else {
            holder.binding.ivInsect.setImageResource(R.drawable.ic_launcher_foreground)
            Log.w("INSECT_IMG", "Imagen no encontrada: $imageName")
        }

        holder.binding.root.setOnClickListener { onClick(insect) }
    }

    private fun highlightText(text: String, query: String): android.text.SpannableString {
        val spannable = android.text.SpannableString(text)
        if (query.isEmpty()) return spannable
        val lowerText = text.lowercase()
        var index = lowerText.indexOf(query.lowercase())
        while (index >= 0) {
            spannable.setSpan(
                android.text.style.BackgroundColorSpan(
                    android.graphics.Color.parseColor("#F7C000")
                ),
                index,
                index + query.length,
                android.text.Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
            )
            index = lowerText.indexOf(query.lowercase(), index + 1)
        }
        return spannable
    }

    override fun getItemCount() = filteredInsects.size

    fun updateList(newList: List<Insect>, query: String = "") {
        filteredInsects = newList
        searchQuery = query
        notifyDataSetChanged()
    }

}