// 文件路径：app/src/main/java/com/yourcompany/blindnavigation/models/User.kt
package com.yourcompany.blindnavigation.models

data class User(
    val id: Int = 0,
    val username: String,
    val password: String,
    val email: String? = null,
    val phone: String? = null,
    val emergencyContact: String? = null,
    val nickname: String? = null,
    val token: String? = null,
    val preferences: UserPreferences? = null,
    val registerTime: Long = System.currentTimeMillis(),
    val lastLoginTime: Long = System.currentTimeMillis(),
    val avatar: String? = null
)

data class UserPreferences(
    val voiceSpeed: Float = 1.0f,
    val vibrationIntensity: Int = 5,
    val preferredLanguage: String = "zh-CN",
    val themeColor: String = "yellow_black"
)