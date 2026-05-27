// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.repository

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.example.projectinvasiveinsects.data.DetectionResult
import com.example.projectinvasiveinsects.data.InferenceResult
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class InferenceRepository(private val context: Context) {

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
        "Spodoptera frugiperda (adult)",
        "Spodoptera frugiperda (larva)"
    )

    private var interpreter: Interpreter? = null

    fun initInterpreter(): Boolean {
        return try {
            val model = loadModelFile("pestneurovision_model.tflite")
            val options = Interpreter.Options().apply { numThreads = 4 }
            interpreter = Interpreter(model, options)
            val inputShape = interpreter!!.getInputTensor(0).shape()
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            Log.d("YOLO", "Input: ${inputShape.toList()}, Output: ${outputShape.toList()}")
            true
        } catch (e: Exception) {
            Log.e("YOLO", "Error init interpreter: ${e.message}")
            false
        }
    }

    fun runInference(originalBitmap: Bitmap): InferenceResult {
        val interp = interpreter ?: return InferenceResult.Error("Model not loaded")
        return try {
            val scaledBitmap = Bitmap.createScaledBitmap(originalBitmap, INPUT_SIZE, INPUT_SIZE, true)
            val inputBuffer = bitmapToByteBuffer(scaledBitmap)
            val outputShape = interp.getOutputTensor(0).shape()
            val outputData = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
            interp.run(inputBuffer, outputData)

            val detections = parseDetections(outputData, originalBitmap.width.toFloat(), originalBitmap.height.toFloat())
            val finalDetections = nonMaxSuppression(detections, IOU_THRESHOLD)

            if (finalDetections.isEmpty()) {
                InferenceResult.Empty("No insects were detected")
            } else {
                val resultBitmap = scaledBitmap.copy(Bitmap.Config.ARGB_8888, true)
                drawBoundingBoxes(resultBitmap, finalDetections)
                InferenceResult.Success(resultBitmap, finalDetections)
            }
        } catch (e: Exception) {
            Log.e("YOLO", "Inference error: ${e.message}")
            InferenceResult.Error(e.message ?: "Unknown error")
        }
    }

    fun closeInterpreter() {
        interpreter?.close()
        interpreter = null
    }

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val afd = context.assets.openFd(fileName)
        return FileInputStream(afd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        pixels.forEach { pixel ->
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
            buffer.putFloat((pixel and 0xFF) / 255.0f)
        }
        return buffer
    }

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
                rect = RectF(
                    (cx - w / 2f) * INPUT_SIZE, (cy - h / 2f) * INPUT_SIZE,
                    (cx + w / 2f) * INPUT_SIZE, (cy + h / 2f) * INPUT_SIZE
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

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = maxOf(a.left, b.left); val interTop = maxOf(a.top, b.top)
        val interRight = minOf(a.right, b.right); val interBottom = minOf(a.bottom, b.bottom)
        val interArea = maxOf(0f, interRight - interLeft) * maxOf(0f, interBottom - interTop)
        val unionArea = (a.width() * a.height()) + (b.width() * b.height()) - interArea
        return if (unionArea <= 0f) 0f else interArea / unionArea
    }

    private fun drawBoundingBoxes(bitmap: Bitmap, detections: List<DetectionResult>) {
        val canvas = Canvas(bitmap)
        val boxPaint = Paint().apply { color = Color.RED; style = Paint.Style.STROKE; strokeWidth = 6f }
        val textPaint = Paint().apply { color = Color.WHITE; textSize = 24f; isAntiAlias = true }
        val bgPaint = Paint().apply { color = Color.RED }
        val textHeight = 30f; val padding = 8f; val bitmapW = bitmap.width.toFloat()
        detections.forEach { det ->
            canvas.drawRect(det.rect, boxPaint)
            val text = "${det.label} ${"%.0f".format(det.score * 100)}%"
            val textWidth = textPaint.measureText(text)
            val labelW = textWidth + padding * 2
            val hayEspacioArriba = det.rect.top - textHeight >= 0
            val bgTop = if (hayEspacioArriba) det.rect.top - textHeight else det.rect.bottom
            val bgBottom = if (hayEspacioArriba) det.rect.top else det.rect.bottom + textHeight
            val textY = if (hayEspacioArriba) det.rect.top - padding else det.rect.bottom + textHeight - padding
            val bgLeft = minOf(det.rect.left, bitmapW - labelW).coerceAtLeast(0f)
            canvas.drawRect(bgLeft, bgTop, bgLeft + labelW, bgBottom, bgPaint)
            canvas.drawText(text, bgLeft + padding, textY, textPaint)
        }
    }
}