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
    private val speakCooldown = 1000L // 减少到1秒冷却时间
    
    // 语音队列管理
    private val voiceQueue = mutableListOf<VoiceTask>()
    private var isProcessingQueue = false
    private val queueLock = Any()
    
    data class VoiceTask(
        val text: String,
        val priority: VoicePriority,
        val timestamp: Long = System.currentTimeMillis()
    )
    
    init {
        initializeTTS()
    }
    
    private fun initializeTTS() {
        textToSpeech = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech?.setLanguage(Locale.CHINESE)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w(TAG, "中文语音包不可用，使用默认语言")
                    textToSpeech?.setLanguage(Locale.US)
                }
                
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
            } else {
                Log.e(TAG, "TTS初始化失败")
            }
        }
    }
    
    fun speak(text: String, priority: VoicePriority = VoicePriority.NORMAL) {
        if (!isInitialized || textToSpeech == null) {
            Log.w(TAG, "TTS未初始化，无法播放语音")
            return
        }
        
        // 添加到语音队列
        synchronized(queueLock) {
            val task = VoiceTask(text, priority)
            
            when (priority) {
                VoicePriority.HIGH -> {
                    // 高优先级，插入到队列前面
                    voiceQueue.add(0, task)
                }
                VoicePriority.NORMAL -> {
                    // 普通优先级，添加到队列末尾
                    voiceQueue.add(task)
                }
                VoicePriority.LOW -> {
                    // 低优先级，添加到队列末尾
                    voiceQueue.add(task)
                }
            }
            
            // 启动队列处理
            if (!isProcessingQueue) {
                processVoiceQueue()
            }
        }
    }
    
    fun speakWithPriority(text: String, priority: VoicePriority = VoicePriority.NORMAL) {
        speak(text, priority)
    }
    
    private fun processVoiceQueue() {
        synchronized(queueLock) {
            if (isProcessingQueue || voiceQueue.isEmpty()) {
                return
            }
            isProcessingQueue = true
        }
        
        Thread {
            while (true) {
                val task = synchronized(queueLock) {
                    if (voiceQueue.isEmpty()) {
                        isProcessingQueue = false
                        return@Thread
                    }
                    voiceQueue.removeAt(0)
                }
                
                // 检查冷却时间（根据优先级调整）
                val currentTime = System.currentTimeMillis()
                val cooldown = when (task.priority) {
                    VoicePriority.HIGH -> 500L
                    VoicePriority.NORMAL -> 1000L
                    VoicePriority.LOW -> 1500L
                }
                
                if (currentTime - lastSpeakTime < cooldown) {
                    // 如果还在冷却中，将任务重新加入队列
                    synchronized(queueLock) {
                        voiceQueue.add(0, task)
                    }
                    Thread.sleep(100)
                    continue
                }
                
                // 执行语音播报
                executeSpeech(task.text, task.priority)
                lastSpeakTime = System.currentTimeMillis()
            }
        }.start()
    }
    
    private fun executeSpeech(text: String, priority: VoicePriority) {
        try {
            val utteranceId = "utterance_${System.currentTimeMillis()}"
            val queueMode = when (priority) {
                VoicePriority.HIGH -> TextToSpeech.QUEUE_FLUSH
                VoicePriority.NORMAL -> TextToSpeech.QUEUE_FLUSH
                VoicePriority.LOW -> TextToSpeech.QUEUE_ADD
            }
            
            val result = textToSpeech?.speak(text, queueMode, null, utteranceId)
            
            if (result == TextToSpeech.SUCCESS) {
                Log.d(TAG, "语音播报成功 (优先级${priority}): $text")
            } else {
                Log.e(TAG, "语音播报失败: $text")
            }
        } catch (e: Exception) {
            Log.e(TAG, "语音播报异常", e)
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
    
    fun clearQueue() {
        synchronized(queueLock) {
            voiceQueue.clear()
        }
    }
    
    fun getQueueSize(): Int {
        synchronized(queueLock) {
            return voiceQueue.size
        }
    }
    
    fun cleanup() {
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        isInitialized = false
        clearQueue()
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