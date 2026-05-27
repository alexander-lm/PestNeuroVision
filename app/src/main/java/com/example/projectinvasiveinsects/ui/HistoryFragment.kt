// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.ui

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.core.os.bundleOf
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.projectinvasiveinsects.data.InvasiveInsectsDatabase
import com.example.projectinvasiveinsects.databinding.FragmentHistoryBinding
import com.example.projectinvasiveinsects.repository.DetectionRepository
import com.example.projectinvasiveinsects.viewmodel.HistoryViewModel
import com.example.projectinvasiveinsects.viewmodel.HistoryViewModelFactory
import androidx.navigation.fragment.findNavController
import com.example.projectinvasiveinsects.R

class HistoryFragment : Fragment() {

    private var _binding: FragmentHistoryBinding? = null
    private val binding get() = _binding!!
    private lateinit var historyViewModel: HistoryViewModel
    private lateinit var adapter: DetectionAdapter

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentHistoryBinding.inflate(inflater, container, false)

        adapter = DetectionAdapter(
            detections = emptyList(),
            onViewPhoto = { imagePath ->
                PhotoViewerDialog(imagePath).show(parentFragmentManager, "PhotoViewerDialog")
            },
            onDelete = { detectionId ->

                CustomDialog.show(
                    context = requireContext(),
                    title = "Remove Detection",
                    message = "Are you sure you want to remove this detection?",
                    positiveText = "Yes",
                    negativeText = "Cancel",
                    iconTint = CustomDialog.IconTint.CHERRY,
                    onPositive = { historyViewModel.deleteDetection(detectionId) }
                )
            },
            onInsectClick = { insectName ->
                historyViewModel.getInsectIdByName(insectName) { insectId ->
                    findNavController().navigate(
                        R.id.action_historyFragment_to_informationFragment,
                        bundleOf("insect_id" to insectId)
                    )
                }
            }
        )

        binding.rvList.adapter = adapter

        val detectionDao = InvasiveInsectsDatabase.getDatabase(requireContext()).detectionDao()
        val detectionDetailDao =
            InvasiveInsectsDatabase.getDatabase(requireContext()).detectionDetailDao()
        val repository = DetectionRepository(detectionDao, detectionDetailDao)
        historyViewModel = ViewModelProvider(this, HistoryViewModelFactory(repository))
            .get(HistoryViewModel::class.java)

        historyViewModel.detectionList.observe(viewLifecycleOwner) { list ->
            adapter.updateList(list)

            binding.etSearch.addTextChangedListener(object : android.text.TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                    val query = s.toString().trim().lowercase()
                    if (query.isEmpty()) {
                        list.forEach { it.expand = false }
                        adapter.setSearchQuery("")
                        adapter.updateList(list)
                    } else {
                        val filtered = list.filter { item ->
                            item.detection.date.lowercase().contains(query) ||
                                    item.detection.time.lowercase().contains(query) ||
                                    item.insectNames.any { it.lowercase().contains(query) }
                        }
                        filtered.forEach { it.expand = true }
                        adapter.setSearchQuery(query)
                        adapter.updateList(filtered)
                    }
                }
                override fun afterTextChanged(s: android.text.Editable?) {}
            })
        }

        val prefs = requireContext().getSharedPreferences("AppPrefs", Context.MODE_PRIVATE)
        val userId = prefs.getInt("user_id", 0)

        historyViewModel.loadDetections(userId)

        historyViewModel.deleteStatus.observe(viewLifecycleOwner) { success ->
            if (success) {
                showDeletedDialog()
            }
        }
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

    private fun showDeletedDialog() {
        val dialogView = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_detection_result, null)

        val dialog = AlertDialog.Builder(requireContext())
            .setView(dialogView)
            .setCancelable(true)
            .create()

        dialog.window?.setBackgroundDrawableResource(android.R.color.transparent)

        dialogView.findViewById<TextView>(R.id.tvDetectionCount).text = "Detection removed"
        dialogView.findViewById<TextView>(R.id.tvSubtitle).text = "The detection has been successfully removed"

        dialogView.findViewById<Button>(R.id.btnDone).setOnClickListener {
            dialog.dismiss()
        }

        dialog.show()
    }
}