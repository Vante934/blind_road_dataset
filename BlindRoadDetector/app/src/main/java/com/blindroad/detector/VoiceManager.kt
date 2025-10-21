package com.blindroad.detector

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.*

class VoiceManager(private val context: Context) {
    
    companion object {
        private const val TAG = "VoiceManager"
    }
    
    private var textToSpeech: TextToSpeech? = null
    private var isInitialized = false
    private var lastSpeakTime = 0L
    private val speakCooldown = 3000L // 3秒冷却时间
    
    init {
        // 延迟初始化TTS，避免在构造函数中初始化
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            initializeTTS()
        }, 1000) // 延迟1秒初始化
    }
    
    private fun initializeTTS() {
        try {
            // 检查TTS引擎是否可用
            val checkIntent = android.content.Intent(android.speech.tts.TextToSpeech.Engine.ACTION_CHECK_TTS_DATA)
            val resolveInfo = context.packageManager.resolveActivity(checkIntent, 0)
            
            if (resolveInfo == null) {
                Log.e(TAG, "TTS引擎不可用，设备不支持TTS")
                isInitialized = false
                return
            }
            
            Log.d(TAG, "TTS引擎可用，开始初始化")
            
            textToSpeech = TextToSpeech(context) { status ->
                try {
                    Log.d(TAG, "TTS回调状态: $status")
                    
                    when (status) {
                        TextToSpeech.SUCCESS -> {
                            Log.d(TAG, "TTS引擎创建成功，开始设置语言")
                            
                            // 尝试设置中文
                            var result = textToSpeech?.setLanguage(Locale.CHINESE)
                            Log.d(TAG, "设置中文语言结果: $result")
                            
                            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                                Log.w(TAG, "中文语音包不可用，尝试简体中文")
                                result = textToSpeech?.setLanguage(Locale.SIMPLIFIED_CHINESE)
                                Log.d(TAG, "设置简体中文结果: $result")
                                
                                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                                    Log.w(TAG, "简体中文不可用，使用默认语言")
                                    result = textToSpeech?.setLanguage(Locale.US)
                                    Log.d(TAG, "设置英文结果: $result")
                                }
                            }
                            
                            // 设置语音参数
                            textToSpeech?.setSpeechRate(1.0f)
                            textToSpeech?.setPitch(1.0f)
                            
                            textToSpeech?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                                override fun onStart(utteranceId: String?) {
                                    Log.d(TAG, "开始播放语音: $utteranceId")
                                }
                                
                                override fun onDone(utteranceId: String?) {
                                    Log.d(TAG, "语音播放完成: $utteranceId")
                                }
                                
                                override fun onError(utteranceId: String?) {
                                    Log.e(TAG, "语音播放错误: $utteranceId")
                                }
                            })
                            
                            isInitialized = true
                            Log.d(TAG, "TTS初始化成功")
                        }
                        TextToSpeech.ERROR -> {
                            Log.e(TAG, "TTS初始化失败，ERROR状态")
                            isInitialized = false
                        }
                        else -> {
                            Log.e(TAG, "TTS初始化失败，未知状态码: $status")
                            isInitialized = false
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "TTS初始化过程中发生异常", e)
                    isInitialized = false
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "创建TTS引擎失败", e)
            isInitialized = false
        }
    }
    
    fun speak(text: String) {
        if (!isInitialized || textToSpeech == null) {
            Log.w(TAG, "TTS未初始化，尝试重新初始化")
            initializeTTS()
            // 使用Toast作为备用方案
            showToastMessage(text)
            return
        }
        
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastSpeakTime < speakCooldown) {
            Log.d(TAG, "语音冷却中，跳过播报: $text")
            return
        }
        
        try {
            val utteranceId = "utterance_${System.currentTimeMillis()}"
            val result = textToSpeech?.speak(text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
            
            if (result == TextToSpeech.SUCCESS) {
                lastSpeakTime = currentTime
                Log.d(TAG, "语音播报成功: $text")
            } else {
                Log.e(TAG, "语音播报失败，结果码: $result, 文本: $text")
                // 如果播报失败，使用Toast作为备用方案
                showToastMessage(text)
                // 尝试重新初始化
                initializeTTS()
            }
        } catch (e: Exception) {
            Log.e(TAG, "语音播报异常", e)
            // 如果发生异常，使用Toast作为备用方案
            showToastMessage(text)
            // 尝试重新初始化
            initializeTTS()
        }
    }
    
    private fun showToastMessage(text: String) {
        try {
            android.widget.Toast.makeText(context, text, android.widget.Toast.LENGTH_LONG).show()
            Log.d(TAG, "显示Toast消息: $text")
        } catch (e: Exception) {
            Log.e(TAG, "显示Toast失败", e)
        }
    }
    
    fun speakWithPriority(text: String, priority: VoicePriority = VoicePriority.NORMAL) {
        when (priority) {
            VoicePriority.HIGH -> {
                // 高优先级语音，忽略冷却时间
                textToSpeech?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "high_priority_${System.currentTimeMillis()}")
            }
            VoicePriority.NORMAL -> {
                speak(text)
            }
            VoicePriority.LOW -> {
                // 低优先级语音，使用队列
                textToSpeech?.speak(text, TextToSpeech.QUEUE_ADD, null, "low_priority_${System.currentTimeMillis()}")
            }
        }
    }
    
    fun stop() {
        textToSpeech?.stop()
    }
    
    fun setSpeechRate(rate: Float) {
        textToSpeech?.setSpeechRate(rate)
    }
    
    fun setPitch(pitch: Float) {
        textToSpeech?.setPitch(pitch)
    }
    
    fun isSpeaking(): Boolean {
        return textToSpeech?.isSpeaking == true
    }
    
    fun cleanup() {
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        isInitialized = false
    }
}

enum class VoicePriority {
    HIGH,   // 高优先级，如警告信息
    NORMAL, // 普通优先级，如检测结果
    LOW     // 低优先级，如状态信息
}

// 语音消息生成器
class VoiceMessageGenerator {
    
    fun generateDetectionMessage(detection: DetectionResult): String {
        return when (detection.riskLevel) {
            "高" -> "警告！前方${String.format("%.1f", detection.distance)}米处有${detection.label}，请立即避让"
            "中" -> "请注意，前方${String.format("%.1f", detection.distance)}米处有${detection.label}，请向${detection.direction}方向前进"
            else -> "前方${String.format("%.1f", detection.distance)}米处有${detection.label}"
        }
    }
    
    fun generateBlindPathMessage(blindPathResult: BlindPathResult): String {
        return if (blindPathResult.detected) {
            "盲道已识别，请沿盲道前进"
        } else {
            "未检测到盲道，请小心前进"
        }
    }
    
    fun generateCollisionWarning(detection: DetectionResult): String {
        return if (detection.collisionRisk > 0.7f) {
            "碰撞风险高！请立即停止前进"
        } else if (detection.collisionRisk > 0.4f) {
            "注意碰撞风险，请谨慎前进"
        } else {
            ""
        }
    }
    
    fun generateSystemMessage(message: String): String {
        return message
    }
}
