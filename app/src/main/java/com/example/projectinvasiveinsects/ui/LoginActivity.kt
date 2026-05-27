// This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

package com.example.projectinvasiveinsects.ui

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProvider
import com.example.projectinvasiveinsects.data.InvasiveInsectsDatabase
import com.example.projectinvasiveinsects.databinding.ActivityLoginBinding
import com.example.projectinvasiveinsects.repository.UserRepository
import com.example.projectinvasiveinsects.resource.Resource
import com.example.projectinvasiveinsects.viewmodel.UserViewModel
import com.example.projectinvasiveinsects.viewmodel.UserViewModelFactory

class LoginActivity : AppCompatActivity() {

    private lateinit var viewModel: UserViewModel
    private lateinit var binding: ActivityLoginBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val sharedPref = getSharedPreferences("AppPrefs", MODE_PRIVATE)
        val isLoggedIn = sharedPref.getBoolean("isLoggedIn", false)

        if (isLoggedIn) {
            val user = sharedPref.getString("user", "")
            val userId = sharedPref.getInt("user_id", 0)
            val intent = Intent(this, HomeActivity::class.java)
            intent.putExtra("user", user)
            intent.putExtra("user_id", userId)
            startActivity(intent)
            finish()
            return
        }

        binding = ActivityLoginBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setupViewModel()
        setupObservers()
        onLoginClick()

        binding.tvSignUp.setOnClickListener {
            startActivity(Intent(this, RegisterActivity::class.java))
        }
    }

    private fun login() {
        val dd = viewModel.getUserLoginDataStatus(
            binding.etUser.text.toString(),
            binding.etPassword.text.toString()
        )
    }

    private fun onLoginClick() {
        binding.bLogin.setOnClickListener {
            login()
        }
    }

    private fun setupViewModel() {
        val database = InvasiveInsectsDatabase.getDatabase(this)
        val repository = UserRepository(database.userDao())

        val viewModelFactory = UserViewModelFactory(repository)
        viewModel = ViewModelProvider(this, viewModelFactory)[UserViewModel::class.java]
    }

    private fun setupObservers() {
        viewModel.getUserLoginDataStatus.observe(this, Observer { resource ->
            when (resource) {
                is Resource.Success -> {

                    val sharedPref = getSharedPreferences("AppPrefs", MODE_PRIVATE)
                    with(sharedPref.edit()) {
                        putBoolean("isLoggedIn", true)
                        putInt("user_id", resource.data!!.id.toInt())
                        putString("user", resource.data.user.toString())
                        putString("names", resource.data.names)
                        putString("paternal_surname", resource.data.paternalSurname)
                        putString("maternal_surname", resource.data.maternalSurname)
                        apply()
                    }


                    val intent = Intent(applicationContext, HomeActivity::class.java)
                    intent.putExtra("user_id", resource.data!!.id)
                    intent.putExtra("user", resource.data.user.toString())
                    startActivity(intent)



                    finish()
                }
                is Resource.Error ->
                    Toast.makeText(this, "Error: ${resource.message}", Toast.LENGTH_SHORT).show()

                is Resource.Loading ->
                    Log.d("msg:Resource.Loading ", "Load")
            }
        })
    }

}