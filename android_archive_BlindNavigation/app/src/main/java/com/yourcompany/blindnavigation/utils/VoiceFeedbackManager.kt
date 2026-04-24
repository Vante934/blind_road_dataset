package com.yourcompany.blindnavigation.utils

import android.content.Context

object VoiceFeedbackManager {
    fun provideFeedback(context: Context, text: String? = null, vibrationType: VibrationType? = null) {
        vibrationType?.let { VibrationManager.vibrate(context, it) }
        text?.let { BaiduTTSManager.speak(context, it) }
    }
}