// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.ui

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.projectinvasiveinsects.data.DetectionResult
import com.example.projectinvasiveinsects.databinding.ActivityRealTimeDetectionBinding
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors

class RealTimeDetectionActivity : AppCompatActivity() {

    private var outputData: Array<Array<FloatArray>>? = null

    private var lastInferenceTime = 0L
    private val INFERENCE_INTERVAL_MS = 200L

    private lateinit var binding: ActivityRealTimeDetectionBinding
    private var tfliteInterpreter: Interpreter? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private val cameraExecutor = Executors.newSingleThreadExecutor()

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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            supportActionBar?.hide()
            binding = ActivityRealTimeDetectionBinding.inflate(layoutInflater)
            setContentView(binding.root)

            setupInterpreter()

            if (checkSelfPermission(android.Manifest.permission.CAMERA) ==
                android.content.pm.PackageManager.PERMISSION_GRANTED) {
                binding.cameraPreview.post { startCamera() }
            } else {
                requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 100)
            }

            binding.btnCerrar.setOnClickListener { finish() }

        } catch (e: Exception) {
            Log.e("YOLO_RT", "CRASH en onCreate: ${e.message}")
            e.printStackTrace()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.isNotEmpty() &&
            grantResults[0] == android.content.pm.PackageManager.PERMISSION_GRANTED) {
            binding.cameraPreview.post { startCamera() }
        } else {
            finish()
        }
    }

    private fun setupInterpreter() {
        try {
            val model = loadModelFile("pestneurovision_model.tflite")
            tfliteInterpreter = Interpreter(model, Interpreter.Options().apply { numThreads = 4 })
        } catch (e: Exception) {
            Log.e("YOLO_RT", "Error cargando modelo: ${e.message}")
        }
    }

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val afd = assets.openFd(fileName)
        val fis = FileInputStream(afd.fileDescriptor)
        return fis.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
    }

    private fun startCamera() {
        Log.d("YOLO_RT", "Iniciando cámara...")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                Log.d("YOLO_RT", "CameraProvider obtenido")
                cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder()
                    .setTargetResolution(Size(640, 640))
                    .build()

                preview.setSurfaceProvider(binding.cameraPreview.surfaceProvider)

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetResolution(Size(480, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                    .also { analysis ->
                        analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                            procesarFrame(imageProxy)
                        }
                    }

                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
                Log.d("YOLO_RT", "Cámara vinculada correctamente")

            } catch (e: Exception) {
                Log.e("YOLO_RT", "Error en cameraProviderFuture: ${e.message}")
                e.printStackTrace()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun procesarFrame(imageProxy: ImageProxy) {

        val currentTime = System.currentTimeMillis()
        if (currentTime - lastInferenceTime < INFERENCE_INTERVAL_MS) {
            imageProxy.close()

            if (currentTime - lastInferenceTime > 500L) {
                runOnUiThread { binding.overlayView.setDetections(emptyList(), INPUT_SIZE) }
            }
            return

        }
        lastInferenceTime = currentTime


        try {
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            val bitmap = yuv420ToBitmap(imageProxy)
            imageProxy.close()
            if (bitmap == null) return

            val correctedBitmap = if (rotationDegrees != 0) {
                val matrix = android.graphics.Matrix()
                matrix.postRotate(rotationDegrees.toFloat())
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                    .also { bitmap.recycle() }
            } else {
                bitmap
            }

            val scaledBitmap = Bitmap.createScaledBitmap(correctedBitmap, INPUT_SIZE, INPUT_SIZE, true)
            correctedBitmap.recycle()

            val inputBuffer = bitmapToByteBuffer(scaledBitmap)
            scaledBitmap.recycle()

            val interpreter = tfliteInterpreter ?: return

            if (outputData == null) {
                val outputShape = interpreter.getOutputTensor(0).shape()
                outputData = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
            }

            val localOutputData = outputData ?: return

            interpreter.run(inputBuffer, localOutputData)
            val detections = parseDetections(localOutputData)


            val finalDetections = nonMaxSuppression(detections, IOU_THRESHOLD)

            Log.d("YOLO_RT", "Frame procesado - detecciones: ${finalDetections.size}")
            if (finalDetections.isNotEmpty()) {
                Log.d("YOLO_RT", "Detecciones: ${finalDetections.map { "${it.label} score=${it.score}" }}")
            }

            runOnUiThread {
                binding.overlayView.setDetections(finalDetections, INPUT_SIZE)
            }

        } catch (e: Exception) {
            imageProxy.close()
            Log.e("YOLO_RT", "Error procesando frame: ${e.message}")
        }
    }


    private fun yuv420ToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val bitmap = imageProxy.toBitmap()
            bitmap.copy(Bitmap.Config.ARGB_8888, true).also { bitmap.recycle() }
        } catch (e: Exception) {
            Log.e("YOLO_RT", "Error en toBitmap: ${e.message}")
            try {
                val yBuffer = imageProxy.planes[0].buffer
                val uBuffer = imageProxy.planes[1].buffer
                val vBuffer = imageProxy.planes[2].buffer
                val ySize = yBuffer.remaining()
                val uSize = uBuffer.remaining()
                val vSize = vBuffer.remaining()
                val nv21 = ByteArray(ySize + uSize + vSize)
                yBuffer.get(nv21, 0, ySize)
                vBuffer.get(nv21, ySize, vSize)
                uBuffer.get(nv21, ySize + vSize, uSize)
                val yuvImage = android.graphics.YuvImage(
                    nv21, android.graphics.ImageFormat.NV21,
                    imageProxy.width, imageProxy.height, null
                )
                val out = java.io.ByteArrayOutputStream()
                yuvImage.compressToJpeg(
                    android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height), 100, out
                )
                val bytes = out.toByteArray()
                android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    ?.copy(Bitmap.Config.ARGB_8888, true)
            } catch (e2: Exception) {
                Log.e("YOLO_RT", "Error en fallback YUV: ${e2.message}")
                null
            }
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

    private fun parseDetections(output: Array<Array<FloatArray>>): List<DetectionResult> {
        val results = mutableListOf<DetectionResult>()
        val numClasses = CLASS_NAMES.size
        val data = output[0]
        val isTransposed = data.size == (4 + numClasses)
        val numAnchors = if (isTransposed) data[0].size else data.size

        for (i in 0 until numAnchors) {
            val cx: Float; val cy: Float; val w: Float; val h: Float
            val classScores: FloatArray

            if (isTransposed) {
                cx = data[0][i]; cy = data[1][i]; w = data[2][i]; h = data[3][i]
                classScores = FloatArray(numClasses) { c -> data[4 + c][i] }
            } else {
                cx = data[i][0]; cy = data[i][1]; w = data[i][2]; h = data[i][3]
                classScores = FloatArray(numClasses) { c -> data[i][4 + c] }
            }

            val maxScore = classScores.max()
            if (maxScore < SCORE_THRESHOLD) continue

            val classIndex = classScores.indexOfFirst { it == maxScore }

            results.add(DetectionResult(
                rect = android.graphics.RectF(
                    (cx - w / 2f) * INPUT_SIZE,
                    (cy - h / 2f) * INPUT_SIZE,
                    (cx + w / 2f) * INPUT_SIZE,
                    (cy + h / 2f) * INPUT_SIZE
                ),
                score = maxScore,
                classIndex = classIndex,
                label = CLASS_NAMES.getOrElse(classIndex) { "unknown" }
            ))
        }
        return results
    }

    private fun nonMaxSuppression(detections: List<DetectionResult>, iouThreshold: Float): List<DetectionResult> {
        val sorted = detections.sortedByDescending { it.score }.toMutableList()
        val result = mutableListOf<DetectionResult>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)
            sorted.removeAll { iou(best.rect, it.rect) > iouThreshold }
        }
        return result
    }

    private fun iou(a: android.graphics.RectF, b: android.graphics.RectF): Float {
        val interArea = maxOf(0f, minOf(a.right, b.right) - maxOf(a.left, b.left)) *
                maxOf(0f, minOf(a.bottom, b.bottom) - maxOf(a.top, b.top))
        val unionArea = a.width() * a.height() + b.width() * b.height() - interArea
        return if (unionArea <= 0f) 0f else interArea / unionArea
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()
        tfliteInterpreter?.close()
    }
}