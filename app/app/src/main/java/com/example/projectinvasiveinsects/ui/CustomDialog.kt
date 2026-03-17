package com.example.projectinvasiveinsects.ui

import android.content.Context
import android.view.LayoutInflater
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import com.example.projectinvasiveinsects.R

object CustomDialog {

    object IconTint {
        const val DEFAULT = "default"
        const val RED = "#F44336"
        const val YELLOW = "#FFC107"
        const val GREEN = "#14460a"
        const val BLUE = "#1E88E5"
        const val CHERRY = "#531A01"
        const val ORANGE = "#FFAA00"
    }

    fun show(
        context: Context,
        title: String,
        message: String,
        positiveText: String = "Yes",
        negativeText: String = "Cancel",
        iconRes: Int = android.R.drawable.ic_dialog_alert,
        iconTint: String = IconTint.DEFAULT,
        onPositive: () -> Unit,
        onNegative: (() -> Unit)? = null
    ) {
        val dialogView = LayoutInflater.from(context)
            .inflate(R.layout.dialog_custom, null)

        val dialog = AlertDialog.Builder(context)
            .setView(dialogView)
            .setCancelable(true)
            .create()

        dialog.window?.setBackgroundDrawableResource(android.R.color.transparent)

        dialogView.findViewById<TextView>(R.id.tvTitle).text = title
        dialogView.findViewById<TextView>(R.id.tvMessage).text = message

        dialogView.findViewById<ImageView>(R.id.ivIcon).apply {
            setImageResource(iconRes)
            if (iconTint == IconTint.DEFAULT) {
                clearColorFilter()
            } else {
                setColorFilter(android.graphics.Color.parseColor(iconTint))
            }
        }

        dialogView.findViewById<android.widget.ImageButton>(R.id.btnClose).setOnClickListener {
            dialog.dismiss()
        }

        dialogView.findViewById<com.google.android.material.button.MaterialButton>(R.id.btnNegative).apply {
            text = negativeText
            setOnClickListener {
                onNegative?.invoke()
                dialog.dismiss()
            }
        }

        dialogView.findViewById<com.google.android.material.button.MaterialButton>(R.id.btnPositive).apply {
            text = positiveText
            setOnClickListener {
                onPositive()
                dialog.dismiss()
            }
        }

        dialog.show()
    }
}