// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.ui

import android.graphics.Typeface
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.TextPaint
import android.text.method.LinkMovementMethod
import android.text.style.MetricAffectingSpan
import android.text.style.URLSpan
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.res.ResourcesCompat
import androidx.fragment.app.Fragment
import com.example.projectinvasiveinsects.R

class AboutFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_about, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        requireActivity().findViewById<TextView>(R.id.tvToolbarTitle)?.text = "ABOUT"

        val boldItalic = ResourcesCompat.getFont(requireContext(), R.font.montserrat_bold_italic)

        val credits = listOf(
            Triple(R.id.tvCredit1, "Bemisia tabaci",         " (adult)"),
            Triple(R.id.tvCredit2, "Ceratitis capitata",     " (adult)"),
            Triple(R.id.tvCredit3, "Dione juno",             " (adult)"),
            Triple(R.id.tvCredit4, "Dione juno",             " (larva)"),
            Triple(R.id.tvCredit5, "Ligyrus gibbosus",       " (adult)"),
            //Triple(R.id.tvCredit6, "Liriomyza huidobrensis", " (adult)"),
            Triple(R.id.tvCredit7, "Myzus persicae",         " (nymph)"),
            Triple(R.id.tvCredit8, "Spodoptera frugiperda",  " (adult)"),
            Triple(R.id.tvCredit9, "Spodoptera frugiperda",  " (larva)"),
        )

        credits.forEach { (viewId, scientificName, rest) ->
            val fullText = "$scientificName$rest"
            val spannable = SpannableString(fullText)
            if (boldItalic != null) {
                spannable.setSpan(
                    CustomTypefaceSpan(boldItalic),
                    0,
                    scientificName.length,
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }
            view.findViewById<TextView>(viewId)?.text = spannable
        }

        // Cursiva para el nombre científico en tvCredit10
        val tv10 = view.findViewById<TextView>(R.id.tvCredit10)
        if (tv10 != null && boldItalic != null) {
            val fullText10 = tv10.text.toString()
            val scientificName10 = "Spodoptera frugiperda"
            val start10 = fullText10.indexOf(scientificName10)
            if (start10 != -1) {
                val spannable10 = SpannableString(fullText10)
                spannable10.setSpan(
                    CustomTypefaceSpan(boldItalic),
                    start10,
                    start10 + scientificName10.length,
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                )
                tv10.text = spannable10
            }
        }

        // Links de iNaturalist
        val subCredits = listOf(
            Pair(R.id.tvCreditSub1,  "60075120"),
            Pair(R.id.tvCreditSub2,  "69185885"),
            Pair(R.id.tvCreditSub3,  "240437816"),
            Pair(R.id.tvCreditSub4,  "194123388"),
            Pair(R.id.tvCreditSub5,  "225035301"),
            Pair(R.id.tvCreditSub7,  "38478019"),
            Pair(R.id.tvCreditSub8,  "134892035"),
            Pair(R.id.tvCreditSub9,  "17408158"),
            Pair(R.id.tvCreditSub10, "69886018"),
        )

        subCredits.forEach { (viewId, obsNumber) ->
            val tv = view.findViewById<TextView>(viewId) ?: return@forEach
            val text = tv.text.toString()
            val start = text.indexOf(obsNumber)
            if (start == -1) return@forEach

            val url = "https://www.inaturalist.org/observations/$obsNumber"
            val spannable = SpannableString(text)
            spannable.setSpan(URLSpan(url), start, start + obsNumber.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            tv.text = spannable
            tv.movementMethod = LinkMovementMethod.getInstance()
        }

        // Link DOI para Mugala et al.
//        val tvSub6 = view.findViewById<TextView>(R.id.tvCreditSub6)
//        if (tvSub6 != null) {
//            val text = tvSub6.text.toString()
//            val keyword = "Journal of African Entomology"
//            val start = text.indexOf(keyword)
//            if (start != -1) {
//                val url = "https://doi.org//10.17159/2254-8854/2022/a11455"
//                val spannable = SpannableString(text)
//                spannable.setSpan(URLSpan(url), start, start + keyword.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
//                tvSub6.text = spannable
//                tvSub6.movementMethod = LinkMovementMethod.getInstance()
//            }
//        }
    }
    private fun applyObservationLinks(parent: ViewGroup, links: Map<String, String>) {
        for (i in 0 until parent.childCount) {
            val child = parent.getChildAt(i)
            if (child is TextView) {
                val text = child.text.toString()
                val obsNumber = links.keys.firstOrNull { text.contains(it) } ?: continue
                val url = links[obsNumber] ?: continue

                val spannable = SpannableString(text)
                val start = text.indexOf(obsNumber)
                val end = start + obsNumber.length

                spannable.setSpan(
                    URLSpan(url),
                    start,
                    end,
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                )

                child.text = spannable
                child.movementMethod = LinkMovementMethod.getInstance()
            }
        }
    }
}

class CustomTypefaceSpan(private val typeface: Typeface) : MetricAffectingSpan() {
    override fun updateDrawState(paint: TextPaint) {
        paint.typeface = typeface
    }
    override fun updateMeasureState(paint: TextPaint) {
        paint.typeface = typeface
    }
}