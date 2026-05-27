// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.example.projectinvasiveinsects.data.DetectionResult

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<DetectionResult> = emptyList()
    private var inputSize: Int = 640

    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 28f
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val bgPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
    }

    fun setDetections(results: List<DetectionResult>, modelInputSize: Int) {
        detections = results
        inputSize = modelInputSize
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val textHeight = 32f
        val padding = 8f

        val overlayW = width.toFloat()
        val overlayH = height.toFloat()

        val scale = minOf(overlayW / inputSize, overlayH / inputSize)
        val offsetX = (overlayW - inputSize * scale) / 2f
        val offsetY = (overlayH - inputSize * scale) / 2f

        detections.forEach { det ->
            val rect = RectF(
                det.rect.left * scale + offsetX,
                det.rect.top * scale + offsetY,
                det.rect.right * scale + offsetX,
                det.rect.bottom * scale + offsetY
            )

            rect.left   = rect.left.coerceIn(0f, overlayW)
            rect.top    = rect.top.coerceIn(0f, overlayH)
            rect.right  = rect.right.coerceIn(0f, overlayW)
            rect.bottom = rect.bottom.coerceIn(0f, overlayH)

            canvas.drawRect(rect, boxPaint)

            val text = "${det.label} ${"%.0f".format(det.score * 100)}%"
            val textWidth = textPaint.measureText(text)
            val labelW = textWidth + padding * 2

            val hayEspacioArriba = rect.top - textHeight >= 0
            val bgTop    = if (hayEspacioArriba) rect.top - textHeight else rect.bottom
            val bgBottom = if (hayEspacioArriba) rect.top             else rect.bottom + textHeight
            val textY    = if (hayEspacioArriba) rect.top - padding   else rect.bottom + textHeight - padding

            val bgLeft  = minOf(rect.left, overlayW - labelW).coerceAtLeast(0f)
            val bgRight = bgLeft + labelW

            canvas.drawRect(bgLeft, bgTop, bgRight, bgBottom, bgPaint)
            canvas.drawText(text, bgLeft + padding, textY, textPaint)
        }
    }
}