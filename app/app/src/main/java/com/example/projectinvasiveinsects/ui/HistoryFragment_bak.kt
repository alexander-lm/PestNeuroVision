package com.example.projectinvasiveinsects.ui

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.projectinvasiveinsects.R
import com.example.projectinvasiveinsects.data.Language
import com.example.projectinvasiveinsects.databinding.FragmentDetectorBinding
import com.example.projectinvasiveinsects.databinding.FragmentHistoryBinding
import com.example.projectinvasiveinsects.ui.adapter.RvAdapter


class HistoryFragment_bak : Fragment() {

    private var _binding: FragmentHistoryBinding? = null
    private val binding get() = _binding!!

    private var languageList = ArrayList<Language>()
    private lateinit var rvAdapter: RvAdapter


    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {

        _binding = FragmentHistoryBinding.inflate(inflater, container, false)

        binding.rvList.layoutManager = LinearLayoutManager(requireContext())

        rvAdapter = RvAdapter(languageList)
        binding.rvList.adapter = rvAdapter

        val language1 = Language(
            "Java",
            "Java is an Object Oriented Programming language." +
                    " Java is used in all kind of applications like Mobile Applications (Android is Java based), " +
                    "desktop applications, web applications, client server applications, enterprise applications and many more. ",
            false
        )
        val language2 = Language(
            "Kotlin",
            "Kotlin is a statically typed, general-purpose programming language" +
                    " developed by JetBrains, that has built world-class IDEs like IntelliJ IDEA, PhpStorm, Appcode, etc.",
            false
        )
        val language3 = Language(
            "Python",
            "Python is a high-level, general-purpose and a very popular programming language." +
                    " Python programming language (latest Python 3) is being used in web development, Machine Learning applications, " +
                    "along with all cutting edge technology in Software Industry.",
            false
        )
        val language4 = Language(
            "CPP",
            "C++ is a general purpose programming language and widely used now a days for " +
                    "competitive programming. It has imperative, object-oriented and generic programming features. ",
            false
        )

        languageList.add(language1)
        languageList.add(language2)
        languageList.add(language3)
        languageList.add(language4)

        rvAdapter.notifyDataSetChanged()

        return binding.root
    }

    override fun onDestroy() {
        super.onDestroy()
        _binding = null
    }

}