package com.yourcompany.blindnavigation.utils

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log

enum class VibrationType {
    NAVIGATION,         // 智能导航
    OBSTACLE,           // 障碍检测
    LOCATION,           // 当前位置
    TRANSPORT,          // 公共交通
    ASSISTANCE,         // 紧急求助
    SETTINGS            // 系统设置
}

object VibrationManager {
    fun vibrate(context: Context, type: VibrationType) {
        try {
            val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            val pattern = when (type) {
                VibrationType.NAVIGATION -> longArrayOf(0, 100) // 短促单次
                VibrationType.OBSTACLE -> longArrayOf(0, 400) // 长单次
                VibrationType.LOCATION -> longArrayOf(0, 80, 100, 80) // 两次短震动
                VibrationType.TRANSPORT -> longArrayOf(0, 80, 80, 80, 80, 80) // 三次短震动
                VibrationType.ASSISTANCE -> longArrayOf(0, 200, 100, 200, 100, 400) // 持续交错
                VibrationType.SETTINGS -> longArrayOf(0, 80, 100, 250, 100, 80) // 特殊模式
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(pattern, -1)
            }
        } catch (e: Exception) {
            Log.e("VibrationManager", "Vibration failed: ${e.message}")
        }
    }
}