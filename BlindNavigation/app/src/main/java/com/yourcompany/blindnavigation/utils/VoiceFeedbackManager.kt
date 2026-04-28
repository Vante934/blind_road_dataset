package com.yourcompany.blindnavigation.utils

import android.content.Context
import android.util.Log

object VoiceFeedbackManager {
    private var lastSpeakText = ""
    private var lastSpeakTime = 0L
    private val SPEAK_DEBOUNCE_MS = 2000L  // 2秒内相同内容不重复

    fun provideFeedback(context: Context, text: String? = null, vibrationType: VibrationType? = null) {
        Log.e("VOICE_MANAGER", "========== provideFeedback 被调用 ==========")
        Log.e("VOICE_MANAGER", "text: $text")

        if (text.isNullOrBlank()) {
            Log.e("VOICE_MANAGER", "⚠️ 文本为空，跳过播报")
            return
        }

        val currentTime = System.currentTimeMillis()

        // 防重复播报
        if (text == lastSpeakText && currentTime - lastSpeakTime < SPEAK_DEBOUNCE_MS) {
            Log.e("VOICE_MANAGER", "⏭️ 跳过重复播报: $text (${currentTime - lastSpeakTime}ms)")
            return
        }

        lastSpeakText = text
        lastSpeakTime = currentTime

        // 振动反馈
        vibrationType?.let {
            Log.e("VOICE_MANAGER", "📳 振动: $it")
            VibrationManager.vibrate(context, it)
        }

        // 语音播报（使用队列，不会互相打断）
        Log.e("VOICE_MANAGER", "🔊 开始语音播报: $text")
        try {
            BaiduTTSManager.speak(context, text)
            Log.e("VOICE_MANAGER", "✅ 已添加到播放队列")
        } catch (e: Exception) {
            Log.e("VOICE_MANAGER", "❌ 添加失败: ${e.message}", e)
        }
    }
}