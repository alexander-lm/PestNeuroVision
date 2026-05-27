// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.ui

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageDecoder
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import com.example.projectinvasiveinsects.databinding.FragmentDetectorBinding
import com.google.android.material.button.MaterialButton
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class DetectorFragment_bak : Fragment() {

    private var finalDetections: List<DetectionResult> = emptyList()
    private var _binding: FragmentDetectorBinding? = null
    private val binding get() = _binding!!

    private lateinit var ivPreview: ImageView
    private lateinit var btnCamera: ImageButton
    //private lateinit var btnGallery: ImageButton
    private var latestTmpUri: Uri? = null

    private var currentBitmap: Bitmap? = null

    private val INPUT_SIZE = 640
    private val SCORE_THRESHOLD = 0.10f
    private val IOU_THRESHOLD = 0.45f

    private val CLASS_NAMES = listOf(
        "Bemisia tabaci (adult)",
        "Ceratitis capitata (adult)",
        "Dione juno (adult)",
        "Dione juno (larva)",
        "Ligyrus gibbosus (adult)",
        "Liriomyza huidobrensis (adult)",
        "Myzus persicae (nymph)",
        "Myzus persicae sp (adult)",
        "Spodoptera frugiperda (adult)",
        "Spodoptera frugiperda (larva)"
    )

    private var tfliteInterpreter: Interpreter? = null


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
        else Toast.makeText(requireContext(), "Se requiere permiso de cámara", Toast.LENGTH_SHORT)
            .show()
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentDetectorBinding.inflate(inflater, container, false)

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

        binding.btnDetect.setOnClickListener { runInferenceAndDraw() }

        setupInterpreter()

        binding.btnSaveDetection.isEnabled = false




        return binding.root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        tfliteInterpreter?.close()
        tfliteInterpreter = null
        _binding = null
    }


    private fun setupInterpreter() {
        try {
            val model = loadModelFile("pestneurovision_model.tflite")
            val options = Interpreter.Options().apply {
                numThreads = 4
            }
            tfliteInterpreter = Interpreter(model, options)
            Log.d("YOLO", "Intérprete TFLite inicializado correctamente")

            val inputShape = tfliteInterpreter!!.getInputTensor(0).shape()
            val outputShape = tfliteInterpreter!!.getOutputTensor(0).shape()
            Log.d("YOLO", "Input shape: ${inputShape.toList()}")
            Log.d("YOLO", "Output shape: ${outputShape.toList()}")

        } catch (e: Exception) {
            Log.e("YOLO", "Error inicializando intérprete: ${e.message}")
            Toast.makeText(
                requireContext(),
                "Error cargando modelo: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val assetFileDescriptor = requireContext().assets.openFd(fileName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFileDescriptor.startOffset,
            assetFileDescriptor.declaredLength
        )
    }

    private fun runInferenceAndDraw() {
        val interpreter = tfliteInterpreter ?: run {
            Toast.makeText(requireContext(), "Modelo no cargado", Toast.LENGTH_SHORT).show()
            return
        }

        try {

            val originalBitmap = currentBitmap ?: run {
                Toast.makeText(
                    requireContext(),
                    "Selecciona una imagen primero",
                    Toast.LENGTH_SHORT
                ).show()
                return
            }

            val origW = originalBitmap.width.toFloat()
            val origH = originalBitmap.height.toFloat()

            val scaledBitmap =
                Bitmap.createScaledBitmap(originalBitmap, INPUT_SIZE, INPUT_SIZE, true)

            val inputBuffer = bitmapToByteBuffer(scaledBitmap)

            val outputTensor = interpreter.getOutputTensor(0)
            val outputShape = outputTensor.shape()
            Log.d("YOLO", "Output shape en inferencia: ${outputShape.toList()}")

             val outputData = Array(outputShape[0]) {
                Array(outputShape[1]) { FloatArray(outputShape[2]) }
            }

            interpreter.run(inputBuffer, outputData)

            val detections = parseDetections(outputData, origW, origH)

            val finalDetections = nonMaxSuppression(detections, IOU_THRESHOLD)

            Log.d(
                "YOLO",
                "Detecciones finales: ${finalDetections.map { "${it.label} rect=${it.rect} score=${it.score}" }}"
            )

            val resultBitmap = scaledBitmap.copy(Bitmap.Config.ARGB_8888, true)

            if (finalDetections.isEmpty()) {
                Toast.makeText(requireContext(), "No se detectaron insectos", Toast.LENGTH_SHORT)
                    .show()
                binding.btnSaveDetection.isEnabled = false
            } else {
                drawBoundingBoxes(resultBitmap, finalDetections)
                currentBitmap = resultBitmap
                binding.ivPreview.setImageBitmap(null)
                binding.ivPreview.setImageBitmap(resultBitmap)
                binding.ivPreview.invalidate()
                binding.btnSaveDetection.isEnabled = true
                Toast.makeText(
                    requireContext(),
                    "${finalDetections.size} detección(es) encontrada(s)",
                    Toast.LENGTH_SHORT
                ).show()
            }

        } catch (e: Exception) {
            Log.e("YOLO", "Error en inferencia: ${e.message}")
            e.printStackTrace()
            Toast.makeText(requireContext(), "Error: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {

        val byteBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            byteBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            byteBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
            byteBuffer.putFloat((pixel and 0xFF) / 255.0f)
        }

        return byteBuffer
    }

    data class DetectionResult(
        val rect: RectF,
        val score: Float,
        val classIndex: Int,
        val label: String
    )

    private fun parseDetections(
        output: Array<Array<FloatArray>>,
        origW: Float,
        origH: Float
    ): List<DetectionResult> {
        val results = mutableListOf<DetectionResult>()

        val numClasses = CLASS_NAMES.size
        val data = output[0]

        val isTransposed = data.size == (4 + numClasses)
        val numAnchors = if (isTransposed) data[0].size else data.size

        for (i in 0 until numAnchors) {
            val cx: Float
            val cy: Float
            val w: Float
            val h: Float
            val classScores: FloatArray

            if (isTransposed) {
                cx = data[0][i]
                cy = data[1][i]
                w = data[2][i]
                h = data[3][i]
                classScores = FloatArray(numClasses) { c -> data[4 + c][i] }
            } else {
                cx = data[i][0]
                cy = data[i][1]
                w = data[i][2]
                h = data[i][3]
                classScores = FloatArray(numClasses) { c -> data[i][4 + c] }
            }

            val maxScore = classScores.max()
            if (maxScore < SCORE_THRESHOLD) continue

            val classIndex = classScores.indexOfFirst { it == maxScore }

            val left   = (cx - w / 2f) * INPUT_SIZE
            val top    = (cy - h / 2f) * INPUT_SIZE
            val right  = (cx + w / 2f) * INPUT_SIZE
            val bottom = (cy + h / 2f) * INPUT_SIZE

            results.add(
                DetectionResult(
                    rect = RectF(left, top, right, bottom),
                    score = maxScore,
                    classIndex = classIndex,
                    label = CLASS_NAMES.getOrElse(classIndex) { "unknown" }
                )
            )
        }

        return results
    }

    private fun nonMaxSuppression(
        detections: List<DetectionResult>,
        iouThreshold: Float
    ): List<DetectionResult> {
        val sorted = detections.sortedByDescending { it.score }.toMutableList()
        val result = mutableListOf<DetectionResult>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)
            sorted.removeAll { iou(best.rect, it.rect) > iouThreshold }
        }

        return result
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = maxOf(a.left, b.left)
        val interTop = maxOf(a.top, b.top)
        val interRight = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)

        val interArea = maxOf(0f, interRight - interLeft) * maxOf(0f, interBottom - interTop)
        val unionArea = (a.width() * a.height()) + (b.width() * b.height()) - interArea

        return if (unionArea <= 0f) 0f else interArea / unionArea
    }

    private fun drawBoundingBoxes(bitmap: Bitmap, detections: List<DetectionResult>) {
        val canvas = Canvas(bitmap)

        val boxPaint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 6f
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 24f
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        val bgPaint = Paint().apply {
            color = Color.RED
            style = Paint.Style.FILL
        }

        val textHeight = 30f
        val padding = 8f
        val bitmapW = bitmap.width.toFloat()
        val bitmapH = bitmap.height.toFloat()

        detections.forEach { det ->
            canvas.drawRect(det.rect, boxPaint)

            val text = "${det.label} ${"%.0f".format(det.score * 100)}%"
            val textWidth = textPaint.measureText(text)
            val labelW = textWidth + padding * 2

            val hayEspacioArriba = det.rect.top - textHeight >= 0
            val bgTop    = if (hayEspacioArriba) det.rect.top - textHeight else det.rect.bottom
            val bgBottom = if (hayEspacioArriba) det.rect.top             else det.rect.bottom + textHeight
            val textY    = if (hayEspacioArriba) det.rect.top - padding   else det.rect.bottom + textHeight - padding

            val bgLeft  = minOf(det.rect.left, bitmapW - labelW)
                .coerceAtLeast(0f)
            val bgRight = bgLeft + labelW

            canvas.drawRect(bgLeft, bgTop, bgRight, bgBottom, bgPaint)

            canvas.drawText(text, bgLeft + padding, textY, textPaint)
        }
    }

    private fun actualizarInterfazConImagen(uri: Uri) {

        ivPreview.setPadding(0, 0, 0, 0)
        ivPreview.scaleType = ImageView.ScaleType.FIT_CENTER
        currentBitmap = uriToBitmap(uri)
        ivPreview.setImageBitmap(currentBitmap)
        ivPreview.invalidate()
        binding.btnSaveDetection.isEnabled = false
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
}


