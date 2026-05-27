// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.fragment.findNavController
import com.example.projectinvasiveinsects.R
import com.example.projectinvasiveinsects.data.DetectionResult
import com.example.projectinvasiveinsects.data.InferenceResult
import com.example.projectinvasiveinsects.data.InvasiveInsectsDatabase
import com.example.projectinvasiveinsects.data.entity.DetectionDetail
import com.example.projectinvasiveinsects.databinding.FragmentDetectorBinding
import com.example.projectinvasiveinsects.repository.DetectionRepository
import com.example.projectinvasiveinsects.repository.InferenceRepository
import com.example.projectinvasiveinsects.resource.Resource
import com.example.projectinvasiveinsects.viewmodel.DetectionViewModel
import com.example.projectinvasiveinsects.viewmodel.DetectionViewModelFactory
import com.example.projectinvasiveinsects.viewmodel.InferenceViewModel
import com.example.projectinvasiveinsects.viewmodel.InferenceViewModelFactory
import com.facebook.shimmer.Shimmer
import com.facebook.shimmer.ShimmerFrameLayout
import com.google.android.material.button.MaterialButton
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class DetectorFragment : Fragment() {

    private lateinit var shimmerLayout: ShimmerFrameLayout
    private var isImageProcessed = false
    private lateinit var detectionViewModel: DetectionViewModel
    private lateinit var inferenceViewModel: InferenceViewModel
    private var finalDetections: List<DetectionResult> = emptyList()
    private var _binding: FragmentDetectorBinding? = null
    private val binding get() = _binding!!

    private lateinit var ivPreview: ImageView
    private lateinit var btnCamera: ImageButton
    //private lateinit var btnGallery: MaterialButton
    private var latestTmpUri: Uri? = null
    private var currentBitmap: Bitmap? = null

    private val pickMedia = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) actualizarInterfazConImagen(uri)
    }

    private val takePicture = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) latestTmpUri?.let { actualizarInterfazConImagen(it) }
    }

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) abrirCamara()
        else Toast.makeText(requireContext(), "Se requiere permiso de cámara", Toast.LENGTH_SHORT).show()
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentDetectorBinding.inflate(inflater, container, false)
        val root = binding.root

        shimmerLayout = binding.shimmerLayout
        shimmerLayout.stopShimmer()
        shimmerLayout.setShimmer(null)

        binding.root.findViewById<View>(R.id.placeholderView).visibility = View.VISIBLE
        binding.ivPreview.visibility = View.GONE

        ivPreview = binding.ivPreview
        btnCamera = binding.btnCamera
        //btnGallery = binding.btnGallery

        binding.btnGallery.setOnClickListener {
            pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }

        btnCamera.setOnClickListener {
            val camPerm = android.Manifest.permission.CAMERA
            if (requireContext().checkSelfPermission(camPerm) ==
                android.content.pm.PackageManager.PERMISSION_GRANTED
            ) abrirCamara()
            else requestCameraPermission.launch(camPerm)
        }

        binding.btnDetect.isEnabled = false
        binding.btnDetect.setOnClickListener {
            currentBitmap?.let { inferenceViewModel.runDetection(it) }
                ?: Toast.makeText(requireContext(), "Select an image first", Toast.LENGTH_SHORT).show()
        }

        binding.btnSaveDetection.isEnabled = false
        binding.btnSaveDetection.setOnClickListener {
            guardarDeteccion()
        }

        val inferenceRepository = InferenceRepository(requireContext())
        inferenceViewModel = ViewModelProvider(this, InferenceViewModelFactory(inferenceRepository))
            .get(InferenceViewModel::class.java)

        if (!inferenceViewModel.initModel()) {
            Toast.makeText(requireContext(), "Error loading model", Toast.LENGTH_LONG).show()
        }

        inferenceViewModel.inferenceResult.observe(viewLifecycleOwner) { result: InferenceResult ->
            when (result) {
                is InferenceResult.Loading -> startShimmer()

                is InferenceResult.Success -> {
                    finalDetections = result.detections
                    currentBitmap = result.bitmap
                    binding.ivPreview.setImageBitmap(null)
                    binding.ivPreview.setImageBitmap(result.bitmap)
                    binding.ivPreview.invalidate()
                    binding.btnSaveDetection.isEnabled = true
                    binding.btnDetect.isEnabled = false
                    isImageProcessed = true
                    stopShimmer()
                    showDetectionDialog("${result.detections.size} detection(s) found")
                }

                is InferenceResult.Empty -> {
                    stopShimmer()
                    Toast.makeText(requireContext(), result.message, Toast.LENGTH_SHORT).show()
                    binding.btnSaveDetection.isEnabled = false
                    binding.btnDetect.isEnabled = false
                    isImageProcessed = true
                }

                is InferenceResult.Error -> {
                    stopShimmer()
                    Toast.makeText(requireContext(), "Error: ${result.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        val detectionDao = InvasiveInsectsDatabase.getDatabase(requireContext()).detectionDao()
        val detectionDetailDao = InvasiveInsectsDatabase.getDatabase(requireContext()).detectionDetailDao()
        val detectionRepository = DetectionRepository(detectionDao, detectionDetailDao)
        detectionViewModel = ViewModelProvider(this, DetectionViewModelFactory(detectionRepository))
            .get(DetectionViewModel::class.java)

        detectionViewModel.saveDetectionStatus.observe(viewLifecycleOwner) { resource ->
            when (resource) {
                is Resource.Loading -> {}
                is Resource.Success -> {
                    binding.btnSaveDetection.isEnabled = false
                    showDetectionDialog("Detection saved successfully")
                }
                is Resource.Error -> {
                    Toast.makeText(requireContext(), "Error: ${resource.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }

        binding.btnGoHistory.setOnClickListener {
            findNavController().navigate(R.id.historyFragment)
        }

        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        shimmerLayout.stopShimmer()
        _binding = null
    }

    private fun startShimmer() {
        shimmerLayout.setShimmer(
            Shimmer.AlphaHighlightBuilder()
                .setDuration(1200)
                .setBaseAlpha(0.7f)
                .setHighlightAlpha(1.0f)
                .setDirection(Shimmer.Direction.LEFT_TO_RIGHT)
                .setAutoStart(true)
                .build()
        )
        shimmerLayout.startShimmer()
    }

    private fun stopShimmer() {
        shimmerLayout.stopShimmer()
        shimmerLayout.setShimmer(null)
    }

    private fun actualizarInterfazConImagen(uri: Uri) {
        ivPreview.setPadding(0, 0, 0, 0)
        ivPreview.scaleType = ImageView.ScaleType.FIT_CENTER
        currentBitmap = uriToBitmap(uri)
        ivPreview.setImageBitmap(currentBitmap)
        ivPreview.visibility = View.VISIBLE
        binding.root.findViewById<View>(R.id.placeholderView).visibility = View.GONE
        ivPreview.invalidate()
        binding.btnSaveDetection.isEnabled = false
        binding.btnDetect.isEnabled = true
        isImageProcessed = false
    }

    private fun getTmpFileUri(): Uri {
        val tmpFile = File.createTempFile("plant_image", ".png", requireContext().cacheDir).apply {
            createNewFile()
            deleteOnExit()
        }
        return FileProvider.getUriForFile(
            requireContext(),
            "${requireContext().packageName}.provider",
            tmpFile
        )
    }

    private fun abrirCamara() {
        val uri = getTmpFileUri()
        latestTmpUri = uri
        takePicture.launch(uri)
    }

    private fun uriToBitmap(uri: Uri): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            ImageDecoder.decodeBitmap(
                ImageDecoder.createSource(requireContext().contentResolver, uri)
            )
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(requireContext().contentResolver, uri)
        }.copy(Bitmap.Config.ARGB_8888, true)
    }

    private fun guardarDeteccion() {
        if (finalDetections.isEmpty()) return

        val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val sdfTime = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        val currentDate = sdf.format(Date())
        val currentTime = sdfTime.format(Date())

        val timestamp = System.currentTimeMillis()
        val imageName = "deteccion_${timestamp}.jpg"
        val folder = File("${requireContext().getExternalFilesDir(null)}/detecciones/")
        if (!folder.exists()) folder.mkdirs()
        val imagePath = "${folder.absolutePath}/$imageName"

        currentBitmap?.compress(Bitmap.CompressFormat.JPEG, 95, File(imagePath).outputStream())

        val prefs = requireContext().getSharedPreferences("AppPrefs", Context.MODE_PRIVATE)
        val userId = prefs.getInt("user_id", 0)

        val detectionDataList = finalDetections.map { det ->
            DetectionDetail(
                detectionId = 0,
                insectId = det.classIndex + 1,
                accuracyPercentage = (det.score * 100).toDouble(),
                image = imagePath,
                status = "1"
            )
        }

        detectionViewModel.saveDetection(
            userId = userId,
            date = currentDate,
            time = currentTime,
            imagePath = imagePath,
            detections = detectionDataList
        )
    }

    private fun showDetectionDialog(message: String) {
        val dialogView = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_detection_result, null)

        val dialog = AlertDialog.Builder(requireContext())
            .setView(dialogView)
            .setCancelable(true)
            .create()

        dialog.window?.setBackgroundDrawableResource(android.R.color.transparent)

        dialogView.findViewById<TextView>(R.id.tvDetectionCount).text = message

        dialogView.findViewById<Button>(R.id.btnDone).setOnClickListener {
            dialog.dismiss()
        }

        dialog.show()
    }
}