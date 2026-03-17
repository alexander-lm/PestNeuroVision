package com.example.projectinvasiveinsects.ui

import android.graphics.Typeface
import android.os.Bundle
import android.text.Spannable
import android.text.SpannableString
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.projectinvasiveinsects.R
import com.example.projectinvasiveinsects.data.InvasiveInsectsDatabase
import com.example.projectinvasiveinsects.databinding.FragmentInformationBinding
import com.example.projectinvasiveinsects.repository.InsectRepository
import com.example.projectinvasiveinsects.viewmodel.InsectViewModel
import com.example.projectinvasiveinsects.viewmodel.InsectViewModelFactory

class InformationFragment : Fragment() {

    private var _binding: FragmentInformationBinding? = null
    private val binding get() = _binding!!
    private lateinit var viewModel: InsectViewModel
    private lateinit var montserratBold: Typeface

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentInformationBinding.inflate(inflater, container, false)

        montserratBold = androidx.core.content.res.ResourcesCompat.getFont(
            requireContext(), R.font.montserrat_bold
        ) ?: Typeface.defaultFromStyle(Typeface.BOLD)

        val insectId = arguments?.getInt("insect_id") ?: return binding.root

        val insectDao = InvasiveInsectsDatabase.getDatabase(requireContext()).insectDao()
        val repository = InsectRepository(insectDao)
        viewModel = ViewModelProvider(this, InsectViewModelFactory(repository))
            .get(InsectViewModel::class.java)

        viewModel.selectedInsect.observe(viewLifecycleOwner) { insect ->
            val scientificName = insect.scientificName
            val lifeStage = " (${insect.lifeStage})"
            val fullText = "$scientificName$lifeStage"
            val spannable = SpannableString(fullText)

            spannable.setSpan(
                object : android.text.style.MetricAffectingSpan() {
                    override fun updateMeasureState(p: android.text.TextPaint) { p.typeface = montserratBold }
                    override fun updateDrawState(p: android.text.TextPaint) { p.typeface = montserratBold }
                },
                scientificName.length, fullText.length,
                Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
            )

            binding.tvScientificName.text = spannable
            binding.tvCommonName.text = insect.commonName
            binding.tvDescription.text = insect.description

            val imageName = "${insect.scientificName.trim().lowercase().replace(" ", "_")}_${insect.lifeStage.trim().lowercase().replace(" ", "_")}"
            val resId = requireContext().resources.getIdentifier(
                imageName, "drawable", requireContext().packageName
            )
            if (resId != 0) {
                binding.ivResultImage.setImageResource(resId)
            } else {
                binding.ivResultImage.setImageResource(R.drawable.ic_launcher_foreground)
            }
        }

        viewModel.controlMeasures.observe(viewLifecycleOwner) { measures ->
            binding.layoutMedidas.removeAllViews()
            measures.forEach { measure ->
                val tv = TextView(requireContext()).apply {
                    text = measure.details
                    textSize = 15f
                    setTextColor(resources.getColor(R.color.black, null))
                    setPadding(0, 8, 0, 8)
                    justificationMode = android.graphics.text.LineBreaker.JUSTIFICATION_MODE_INTER_WORD
                    typeface = androidx.core.content.res.ResourcesCompat.getFont(
                        requireContext(),
                        R.font.montserrat_regular
                    )
                }
                binding.layoutMedidas.addView(tv)
            }
        }

        viewModel.loadInsectDetail(insectId)

        return binding.root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}