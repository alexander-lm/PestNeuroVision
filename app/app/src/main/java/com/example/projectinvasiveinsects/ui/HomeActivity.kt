package com.example.projectinvasiveinsects.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.drawerlayout.widget.DrawerLayout
import androidx.navigation.NavController
import androidx.navigation.NavOptions
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.core.view.GravityCompat
import com.example.projectinvasiveinsects.R
import com.example.projectinvasiveinsects.databinding.ActivityHomeBinding

class HomeActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHomeBinding
    private lateinit var navController: NavController
    private lateinit var appBarConfiguration: AppBarConfiguration

    private val fadeNavOptions by lazy {
        NavOptions.Builder()
            .setEnterAnim(R.anim.fade_in)
            .setExitAnim(R.anim.fade_out)
            .setPopEnterAnim(R.anim.fade_in)
            .setPopExitAnim(R.anim.fade_out)
            .build()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHomeBinding.inflate(layoutInflater)
        setContentView(binding.root)
        showWelcomeDialog()
        setupToolBar()
    }

    fun setupToolBar() {
        setSupportActionBar(binding.myToolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)

        val toggle = ActionBarDrawerToggle(
            this,
            binding.drawerLayout,
            binding.myToolbar,
            R.string.open_drawer,
            R.string.close_drawer
        )
        toggle.isDrawerIndicatorEnabled = false
        binding.drawerLayout.addDrawerListener(toggle)
        toggle.syncState()

        binding.myToolbar.setNavigationIcon(R.drawable.ic_menu)
        binding.myToolbar.setNavigationOnClickListener {
            binding.drawerLayout.openDrawer(GravityCompat.START)
        }

        val sharedPref = getSharedPreferences("AppPrefs", MODE_PRIVATE)
        val names = sharedPref.getString("names", "")
        val paternalSurname = sharedPref.getString("paternal_surname", "")
        val maternalSurname = sharedPref.getString("maternal_surname", "")

        val headerView = binding.navView.getHeaderView(0)
        headerView.findViewById<TextView>(R.id.tvUserName).text =
            "$names $paternalSurname $maternalSurname"
        headerView.findViewById<TextView>(R.id.tvUserEmail).text =
            "User: ${sharedPref.getString("user", "")}"

        val navHostFragment =
            supportFragmentManager.findFragmentById(R.id.nav_host_fragment) as NavHostFragment
        navController = navHostFragment.navController

        appBarConfiguration = AppBarConfiguration(
            setOf(
                R.id.detectorFragment,
                R.id.historyFragment,
                R.id.informationFragment,
                R.id.graphFragment,
                R.id.accountFragment
            ),
            binding.drawerLayout
        )

        setupActionBarWithNavController(navController, appBarConfiguration)

        navController.addOnDestinationChangedListener { _, destination, _ ->
            binding.myToolbar.setNavigationIcon(R.drawable.ic_menu)
            val title = when (destination.id) {
                R.id.detectorFragment    -> "INSECT DETECTOR"
                R.id.historyFragment     -> "DETECTION HISTORY"
                R.id.insectListFragment  -> "INSECT INFORMATION"
                R.id.graphFragment       -> "STATISTICAL GRAPHS"
                R.id.accountFragment     -> "MY ACCOUNT"
                R.id.informationFragment -> "INSECT DETAIL"
                else -> "MENU"
            }
            binding.myToolbar.findViewById<TextView>(R.id.tvToolbarTitle).text = title
        }

        binding.navView.setNavigationItemSelectedListener { menuItem ->
            binding.drawerLayout.closeDrawers()
            when (menuItem.itemId) {
                R.id.detectorFragment -> {
                    navController.navigate(R.id.detectorFragment, null, fadeNavOptions)
                    true
                }
                R.id.historyFragment -> {
                    navController.navigate(R.id.historyFragment, null, fadeNavOptions)
                    true
                }
                R.id.insectListFragment -> {
                    navController.navigate(R.id.insectListFragment, null, fadeNavOptions)
                    true
                }
                R.id.graphFragment -> {
                    navController.navigate(R.id.graphFragment, null, fadeNavOptions)
                    true
                }
                R.id.accountFragment -> {
                    navController.navigate(R.id.accountFragment, null, fadeNavOptions)
                    true
                }
                R.id.logOut -> {
                    showLogoutConfirmationDialog()
                    true
                }
                else -> false
            }
        }
    }

    private fun showLogoutConfirmationDialog() {
        CustomDialog.show(
            context = this,
            title = "Log Out",
            message = "Are you sure you want to leave the system?",
            iconTint = CustomDialog.IconTint.YELLOW,
            onPositive = { performLogout() }
        )
    }

    private fun performLogout() {
        val sharedPref = getSharedPreferences("AppPrefs", MODE_PRIVATE)
        sharedPref.edit().clear().apply()

        val intent = android.content.Intent(this, LoginActivity::class.java)
        intent.flags =
            android.content.Intent.FLAG_ACTIVITY_NEW_TASK or
                    android.content.Intent.FLAG_ACTIVITY_CLEAR_TASK

        startActivity(intent)
        finish()

        Toast.makeText(this, "Session successfully closed", Toast.LENGTH_SHORT).show()
    }

    private fun showWelcomeDialog() {
        val prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE)
        val names = prefs.getString("names", "")
        val paternalSurname = prefs.getString("paternal_surname", "")
        val maternalSurname = prefs.getString("maternal_surname", "")

        val dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_welcome, null)

        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .setCancelable(true)
            .create()

        dialog.window?.setBackgroundDrawableResource(android.R.color.transparent)

        dialogView.findViewById<TextView>(R.id.tvWelcomeMessage).text =
            "Welcome!\n$names $paternalSurname $maternalSurname"

        dialogView.findViewById<Button>(R.id.btnDone).setOnClickListener {
            dialog.dismiss()
        }

        dialog.show()
    }

    override fun onDestroy() {
        super.onDestroy()
    }
}