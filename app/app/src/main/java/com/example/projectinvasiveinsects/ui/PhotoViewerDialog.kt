package com.example.projectinvasiveinsects.ui

import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.AnimationUtils
import androidx.fragment.app.DialogFragment
import com.example.projectinvasiveinsects.R
import com.example.projectinvasiveinsects.databinding.DialogPhotoViewerBinding

class PhotoViewerDialog(private val imagePath: String) : DialogFragment() {

    private var _binding: DialogPhotoViewerBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = DialogPhotoViewerBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val bitmap = BitmapFactory.decodeFile(imagePath)
        if (bitmap != null) {
            binding.ivPhoto.setImageBitmap(bitmap)
        } else {
            binding.ivPhoto.setImageResource(android.R.drawable.ic_menu_report_image)
        }
        binding.btnCloseIcon.setOnClickListener { dismiss() }
    }

    override fun onStart() {
        super.onStart()
        dialog?.window?.apply {
            val margin = (resources.displayMetrics.widthPixels * 0.02).toInt()
            setLayout(
                resources.displayMetrics.widthPixels - (margin * 2),
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            setBackgroundDrawableResource(android.R.color.transparent)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}