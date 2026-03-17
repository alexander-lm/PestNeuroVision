package com.example.projectinvasiveinsects.ui

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import com.example.projectinvasiveinsects.R
import com.example.projectinvasiveinsects.data.InvasiveInsectsDatabase
import com.example.projectinvasiveinsects.data.User
import com.example.projectinvasiveinsects.databinding.ActivityRegisterBinding
import com.example.projectinvasiveinsects.repository.UserRepository
import com.example.projectinvasiveinsects.viewmodel.RegisterViewModel
import com.example.projectinvasiveinsects.viewmodel.RegisterViewModelFactory

class RegisterActivity : AppCompatActivity() {

    private lateinit var binding: ActivityRegisterBinding
    private lateinit var viewModel: RegisterViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityRegisterBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val database = InvasiveInsectsDatabase.getDatabase(this)
        val repository = UserRepository(database.userDao())
        viewModel = ViewModelProvider(this, RegisterViewModelFactory(repository))
            .get(RegisterViewModel::class.java)

        binding.btnCancel.setOnClickListener {
            finish()
        }

        binding.btnRegister.setOnClickListener {
            val names = binding.etNames.text.toString().trim()
            val paternalSurname = binding.etPaternalSurname.text.toString().trim()
            val maternalSurname = binding.etMaternalSurname.text.toString().trim()
            val user = binding.etUser.text.toString().trim()
            val password = binding.etPassword.text.toString().trim()

            if (names.isEmpty() || paternalSurname.isEmpty() || maternalSurname.isEmpty()
                || user.isEmpty() || password.isEmpty()) {
                Toast.makeText(this, "All fields are required", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            viewModel.registerUser(
                User(
                    names = names,
                    paternalSurname = paternalSurname,
                    maternalSurname = maternalSurname,
                    user = user,
                    password = password,
                    status = "1"
                )
            )
        }

        viewModel.registerStatus.observe(this) { success ->
            if (success) {
                showSuccessDialog(
                    binding.etNames.text.toString().trim(),
                    binding.etPaternalSurname.text.toString().trim(),
                    binding.etMaternalSurname.text.toString().trim(),
                    binding.etUser.text.toString().trim()
                )
            } else {
                Toast.makeText(this, "Error creating user", Toast.LENGTH_SHORT).show()
            }
        }
    }


    private fun showSuccessDialog(names: String, paternalSurname: String, maternalSurname: String, user: String) {
        val dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_register_success, null)

        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .setCancelable(false)
            .create()

        dialog.window?.setBackgroundDrawableResource(android.R.color.transparent)

        dialogView.findViewById<TextView>(R.id.tvWelcomeMessage).text =
            "User successfully created!\n$names $paternalSurname $maternalSurname\nUser: $user"

        dialogView.findViewById<Button>(R.id.btnDone).setOnClickListener {
            dialog.dismiss()
            val intent = Intent(this@RegisterActivity, LoginActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            startActivity(intent)
        }

        dialog.show()
    }

}