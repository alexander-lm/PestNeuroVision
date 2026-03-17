package com.example.projectinvasiveinsects.ui

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.projectinvasiveinsects.data.InvasiveInsectsDatabase
import com.example.projectinvasiveinsects.databinding.FragmentAccountBinding
import com.example.projectinvasiveinsects.repository.UserRepository
import com.example.projectinvasiveinsects.viewmodel.AccountViewModel
import com.example.projectinvasiveinsects.viewmodel.AccountViewModelFactory

class AccountFragment : Fragment() {

    private var _binding: FragmentAccountBinding? = null
    private val binding get() = _binding!!
    private lateinit var viewModel: AccountViewModel
    private var currentUserId = 0

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentAccountBinding.inflate(inflater, container, false)
        val root = binding.root

        val prefs = requireContext().getSharedPreferences("AppPrefs", Context.MODE_PRIVATE)
        currentUserId = prefs.getInt("user_id", 0)

        val database = InvasiveInsectsDatabase.getDatabase(requireContext())
        val repository = UserRepository(database.userDao())
        viewModel = ViewModelProvider(this, AccountViewModelFactory(repository))
            .get(AccountViewModel::class.java)

        // Cargar datos actuales del usuario
        viewModel.loadUser(currentUserId)

        viewModel.user.observe(viewLifecycleOwner) { user ->
            user?.let {
                binding.etNames.setText(it.names)
                binding.etPaternalSurname.setText(it.paternalSurname)
                binding.etMaternalSurname.setText(it.maternalSurname)
                binding.etUser.setText(it.user)
                binding.etPassword.setText(it.password)
            }
        }

        // Guardar cambios
        binding.btnSave.setOnClickListener {
            val names = binding.etNames.text.toString().trim()
            val paternalSurname = binding.etPaternalSurname.text.toString().trim()
            val maternalSurname = binding.etMaternalSurname.text.toString().trim()
            val user = binding.etUser.text.toString().trim()
            val password = binding.etPassword.text.toString().trim()

            if (names.isEmpty() || paternalSurname.isEmpty() || maternalSurname.isEmpty()
                || user.isEmpty() || password.isEmpty()) {
                Toast.makeText(requireContext(), "All fields are required", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            viewModel.updateUser(currentUserId, names, paternalSurname, maternalSurname, user, password)
        }

        binding.btnDeactivate.setOnClickListener {
            CustomDialog.show(
                context = requireContext(),
                title = "Cancellation",
                message = "Are you sure? You will no longer be able to access the system",
                positiveText = "Confirm",
                iconTint = CustomDialog.IconTint.CHERRY,
                onPositive = { viewModel.deactivateUser(currentUserId) }
            )
        }

        // Observers de resultado
        viewModel.updateStatus.observe(viewLifecycleOwner) { success ->
            if (success) {
                // Actualizar SharedPreferences con los nuevos datos
                val prefs = requireContext().getSharedPreferences("AppPrefs", Context.MODE_PRIVATE)
                with(prefs.edit()) {
                    putString("names", binding.etNames.text.toString().trim())
                    putString("paternal_surname", binding.etPaternalSurname.text.toString().trim())
                    putString("maternal_surname", binding.etMaternalSurname.text.toString().trim())
                    putString("user", binding.etUser.text.toString().trim())
                    apply()
                }

                CustomDialog.show(
                    context = requireContext(),
                    title = "Changes saved",
                    message = "You need to log out to apply the changes. Do you want to log out?",
                    positiveText = "Yes",
                    iconTint = CustomDialog.IconTint.ORANGE,
                    onPositive = {
                        val prefs = requireContext().getSharedPreferences("AppPrefs", Context.MODE_PRIVATE)
                        prefs.edit().clear().apply()
                        val intent = Intent(requireContext(), LoginActivity::class.java)
                        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                        startActivity(intent)
                    }
                )

            } else {
                Toast.makeText(requireContext(), "Error updating data", Toast.LENGTH_SHORT).show()
            }
        }

        viewModel.deactivateStatus.observe(viewLifecycleOwner) { success ->
            if (success) {
                Toast.makeText(requireContext(), "You have successfully unsubscribed", Toast.LENGTH_SHORT).show()
                // Limpiar sesión y volver al login
                val prefs = requireContext().getSharedPreferences("AppPrefs", Context.MODE_PRIVATE)
                prefs.edit().clear().apply()
                val intent = Intent(requireContext(), LoginActivity::class.java)
                intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                startActivity(intent)
            } else {
                Toast.makeText(requireContext(), "Error unsubscribing", Toast.LENGTH_SHORT).show()
            }
        }

        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}