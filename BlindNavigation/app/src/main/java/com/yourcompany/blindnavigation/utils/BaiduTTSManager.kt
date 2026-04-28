package com.yourcompany.blindnavigation.utils

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.util.Log
import kotlinx.coroutines.*
import okhttp3.*
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ConcurrentLinkedQueue

object BaiduTTSManager {
    private const val TAG = "BaiduTTSManager"
    private const val API_KEY = "w978fA2S7PJmUy4IEvlGqxfx"
    private const val SECRET_KEY = "ZeTBNN1UYQRL1kaDEEImHm07Y09jgaRc"
    private const val TOKEN_URL = "https://openapi.baidu.com/oauth/2.0/token"
    private const val TTS_URL = "https://tsn.baidu.com/text2audio"

    private var accessToken: String? = null
    private val client = OkHttpClient()
    private var mediaPlayer: MediaPlayer? = null
    private var isTokenFetching = false

    // ✅ 播放队列
    private val playQueue = ConcurrentLinkedQueue<PlayItem>()
    private var isPlaying = false
    private var currentPlayItem: PlayItem? = null

    data class PlayItem(
        val text: String,
        val context: Context
    )

    /**
     * 获取访问令牌
     */
    suspend fun fetchToken(): String? = withContext(Dispatchers.IO) {
        if (!accessToken.isNullOrEmpty()) {
            Log.d(TAG, "✅ 使用缓存的 Token")
            return@withContext accessToken
        }

        if (isTokenFetching) {
            Log.d(TAG, "⏳ Token 正在获取中，等待...")
            var retryCount = 0
            while (isTokenFetching && retryCount < 30) {
                delay(100)
                retryCount++
            }
            return@withContext accessToken
        }

        isTokenFetching = true
        val url = "$TOKEN_URL?grant_type=client_credentials&client_id=$API_KEY&client_secret=$SECRET_KEY"
        val request = Request.Builder().url(url).get().build()

        try {
            Log.d(TAG, "🔄 正在获取百度 TTS Token...")
            val response = client.newCall(request).execute()
            val respStr = response.body?.string()
            if (respStr != null) {
                val json = JSONObject(respStr)
                accessToken = json.optString("access_token")
                if (!accessToken.isNullOrEmpty()) {
                    Log.d(TAG, "✅ Token 获取成功: ${accessToken?.take(20)}...")
                } else {
                    Log.e(TAG, "❌ Token 获取失败: ${json.optString("error_description")}")
                }
                return@withContext accessToken
            } else {
                Log.e(TAG, "❌ Token 响应为空")
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Token 获取异常: ${e.message}", e)
        } finally {
            isTokenFetching = false
        }
        return@withContext null
    }

    /**
     * 语音合成并播放（使用队列）
     */
    fun speak(context: Context, text: String) {
        Log.e(TAG, "🔊 speak() 被调用, text: $text")

        if (text.isBlank()) {
            Log.w(TAG, "⚠️ 文本为空，跳过播报")
            return
        }

        // 添加到队列
        playQueue.offer(PlayItem(text, context.applicationContext))
        Log.e(TAG, "📝 添加到队列，当前队列长度: ${playQueue.size}")

        // 开始处理队列
        processQueue()
    }

    private fun processQueue() {
        // 如果正在播放，等待播放完成
        if (isPlaying) {
            Log.d(TAG, "⏸️ 正在播放中，等待...")
            return
        }

        // 从队列中取出下一个
        val item = playQueue.poll()
        if (item == null) {
            Log.d(TAG, "📭 队列为空")
            return
        }

        currentPlayItem = item
        isPlaying = true
        Log.e(TAG, "🎙️ 开始处理: ${item.text}")

        // 异步合成并播放
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val token = fetchToken()
                if (token.isNullOrEmpty()) {
                    Log.e(TAG, "❌ Token 获取失败，跳过当前播报")
                    onPlayComplete()
                    return@launch
                }
                synthesizeAndPlaySpeech(item.context, item.text, token)
            } catch (e: Exception) {
                Log.e(TAG, "❌ 处理异常: ${e.message}", e)
                onPlayComplete()
            }
        }
    }

    private fun onPlayComplete() {
        isPlaying = false
        currentPlayItem = null
        Log.e(TAG, "✅ 当前播报完成，继续处理队列")
        // 处理下一个
        processQueue()
    }

    /**
     * 合成语音并播放
     */
    private suspend fun synthesizeAndPlaySpeech(context: Context, text: String, token: String) {
        Log.e(TAG, "🎙️ 开始合成语音: $text")

        val formBody = FormBody.Builder()
            .add("tex", text)
            .add("tok", token)
            .add("cuid", "blind_navigation_app")
            .add("ctp", "1")
            .add("lan", "zh")
            .add("spd", "5")  // 语速正常
            .add("vol", "9")  // 音量最大
            .add("pit", "5")  // 音调正常
            .add("aue", "3")  // mp3格式
            .build()

        val request = Request.Builder().url(TTS_URL).post(formBody).build()

        try {
            val response = client.newCall(request).execute()
            val contentType = response.header("Content-Type", "")

            Log.d(TAG, "📥 TTS 响应状态码: ${response.code}")
            Log.d(TAG, "📥 Content-Type: $contentType")

            if (response.isSuccessful && contentType?.contains("audio") == true) {
                response.body?.bytes()?.let { audioData ->
                    Log.e(TAG, "✅ 收到音频数据, 大小: ${audioData.size} bytes")
                    val audioFile = File(context.cacheDir, "tts_${System.currentTimeMillis()}.mp3")
                    FileOutputStream(audioFile).use { it.write(audioData) }
                    Log.e(TAG, "💾 音频已保存: ${audioFile.absolutePath}")
                    playAudio(context, audioFile.absolutePath, text)
                } ?: run {
                    Log.e(TAG, "❌ 音频数据为空")
                    onPlayComplete()
                }
            } else {
                val errorBody = response.body?.string() ?: "Unknown error"
                Log.e(TAG, "❌ TTS API 错误: $errorBody")
                onPlayComplete()
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ TTS 请求失败: ${e.message}", e)
            onPlayComplete()
        }
    }

    /**
     * 播放音频
     */
    private fun playAudio(context: Context, audioPath: String, originalText: String) {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                // 释放之前的MediaPlayer
                mediaPlayer?.release()

                mediaPlayer = MediaPlayer().apply {
                    setAudioAttributes(
                        AudioAttributes.Builder()
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                            .build()
                    )
                    setDataSource(audioPath)

                    setOnPreparedListener { player ->
                        Log.e(TAG, "🎵 MediaPlayer 准备完成，开始播放: $originalText")
                        player.start()
                    }

                    setOnCompletionListener { player ->
                        Log.e(TAG, "✅ 播放完成: $originalText")
                        player.release()
                        mediaPlayer = null
                        // 删除临时文件
                        try {
                            File(audioPath).delete()
                        } catch (e: Exception) {
                            Log.w(TAG, "删除临时文件失败: ${e.message}")
                        }
                        // 播放完成，继续处理队列
                        onPlayComplete()
                    }

                    setOnErrorListener { player, what, extra ->
                        Log.e(TAG, "❌ MediaPlayer 错误: what=$what, extra=$extra, text=$originalText")
                        player.release()
                        mediaPlayer = null
                        onPlayComplete()
                        true
                    }

                    prepareAsync()
                }
            } catch (e: Exception) {
                Log.e(TAG, "❌ playAudio 错误: ${e.message}", e)
                mediaPlayer?.release()
                mediaPlayer = null
                onPlayComplete()
            }
        }
    }

    /**
     * 停止播放并清空队列
     */
    fun stopSpeaking() {
        // 清空队列
        playQueue.clear()

        // 停止当前播放
        mediaPlayer?.let { player ->
            if (player.isPlaying) {
                player.stop()
                Log.d(TAG, "⏹️ 停止播放")
            }
            player.release()
            mediaPlayer = null
        }

        isPlaying = false
        currentPlayItem = null
        Log.d(TAG, "⏹️ 已停止播放并清空队列")
    }

    /**
     * 释放资源
     */
    fun release() {
        stopSpeaking()
        Log.d(TAG, "TTS 资源已释放")
    }
}