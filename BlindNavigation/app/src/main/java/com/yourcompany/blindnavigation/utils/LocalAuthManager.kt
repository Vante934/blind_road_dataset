// 文件路径：app/src/main/java/com/yourcompany/blindnavigation/utils/LocalAuthManager.kt
package com.yourcompany.blindnavigation.utils

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.google.gson.Gson
import com.yourcompany.blindnavigation.models.User

object LocalAuthManager {
    private const val PREFS_NAME = "auth_prefs"
    private const val KEY_IS_LOGGED_IN = "is_logged_in"
    private const val KEY_CURRENT_USER = "current_user"
    private const val KEY_USER_TOKEN = "user_token"

    private lateinit var prefs: SharedPreferences
    private val gson = Gson()
    private var currentUser: User? = null

    fun init(context: Context) {
        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        loadSavedUser()
    }

    private fun loadSavedUser() {
        val isLoggedIn = prefs.getBoolean(KEY_IS_LOGGED_IN, false)
        if (isLoggedIn) {
            val userJson = prefs.getString(KEY_CURRENT_USER, null)
            currentUser = userJson?.let { gson.fromJson(it, User::class.java) }
        }
    }

    fun login(username: String, password: String): Boolean {
        // 先检查本地存储（用于离线登录）
        val users = getAllUsers()
        val user = users.find { it.username == username && it.password == password }

        return if (user != null) {
            currentUser = user
            saveUser(user)
            true
        } else {
            false
        }
    }

    // 新增：网络登录成功后的处理
    fun loginSuccess(user: User, token: String) {
        val updatedUser = user.copy(token = token)
        currentUser = updatedUser
        saveUser(updatedUser)
        saveToken(token)
    }

    fun register(username: String, password: String, email: String? = null): Boolean {
        // 检查用户名是否已存在
        val users = getAllUsers().toMutableList()
        if (users.any { it.username == username }) {
            return false
        }

        val newUser = User(
            id = users.size + 1,
            username = username,
            password = password,
            email = email
        )
        users.add(newUser)
        saveAllUsers(users)

        // 自动登录
        currentUser = newUser
        saveUser(newUser)
        return true
    }

    // 新增：网络注册成功后的处理
    fun registerSuccess(user: User, token: String) {
        val updatedUser = user.copy(token = token)

        // 保存到用户列表
        val users = getAllUsers().toMutableList()
        users.add(updatedUser)
        saveAllUsers(users)

        // 设置为当前用户
        currentUser = updatedUser
        saveUser(updatedUser)
        saveToken(token)
    }

    fun logout(): Boolean {
        return try {
            prefs.edit().clear().apply()
            currentUser = null
            true
        } catch (e: Exception) {
            Log.e("LocalAuthManager", "Logout failed", e)
            false
        }
    }

    fun isLoggedIn(): Boolean {
        return prefs.getBoolean(KEY_IS_LOGGED_IN, false)
    }

    fun getCurrentUser(): User? {
        return currentUser
    }

    fun updateUser(updatedUser: User): Boolean {
        return try {
            currentUser = updatedUser
            saveUser(updatedUser)

            val users = getAllUsers().toMutableList()
            val index = users.indexOfFirst { it.username == updatedUser.username }
            if (index >= 0) {
                users[index] = updatedUser
                saveAllUsers(users)
            }
            true
        } catch (e: Exception) {
            Log.e("LocalAuthManager", "更新用户信息失败", e)
            false
        }
    }

    fun getToken(): String? {
        return prefs.getString(KEY_USER_TOKEN, null)
    }



    private fun saveUser(user: User) {
        val userJson = gson.toJson(user)
        prefs.edit().apply {
            putString(KEY_CURRENT_USER, userJson)
            putBoolean(KEY_IS_LOGGED_IN, true)
            apply()
        }
    }

    private fun saveToken(token: String) {
        prefs.edit().putString(KEY_USER_TOKEN, token).apply()
    }

    private fun getAllUsers(): List<User> {
        val usersJson = prefs.getString("all_users", "[]")
        return try {
            gson.fromJson(usersJson, Array<User>::class.java).toList()
        } catch (e: Exception) {
            emptyList()
        }
    }

    private fun saveAllUsers(users: List<User>) {
        val usersJson = gson.toJson(users)
        prefs.edit().putString("all_users", usersJson).apply()
    }

    fun printAllUsers() {
        val users = getAllUsers()
        Log.d("LocalAuthManager", "所有用户: $users")
    }
}