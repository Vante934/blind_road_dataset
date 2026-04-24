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
import java.io.IOException

object BaiduTTSManager {
    private const val TAG = "BaiduTTSManager"
    private const val API_KEY = "w978fA2S7PJmUy4IEvlGqxfx"
    private const val SECRET_KEY = "ZeTBNN1UYQRL1kaDEEImHm07Y09jgaRc"
    private const val TOKEN_URL = "https://openapi.baidu.com/oauth/2.0/token"
    private const val TTS_URL = "https://tsn.baidu.com/text2audio"

    private var accessToken: String? = null
    private val client = OkHttpClient()
    private var mediaPlayer: MediaPlayer? = null // 改为类级别变量

    /**
     * 获取访问令牌
     */
    suspend fun fetchToken(): String? = withContext(Dispatchers.IO) {
        val url = "$TOKEN_URL?grant_type=client_credentials&client_id=$API_KEY&client_secret=$SECRET_KEY"
        val request = Request.Builder().url(url).get().build()

        try {
            val response = client.newCall(request).execute()
            val respStr = response.body?.string()
            if (respStr != null) {
                val json = JSONObject(respStr)
                accessToken = json.optString("access_token")
                Log.d(TAG, "Token fetched successfully")
                return@withContext accessToken
            }
        } catch (e: Exception) {
            Log.e(TAG, "Token error: ${e.message}")
        }
        return@withContext null
    }

    /**
     * 语音合成并播放
     */
    fun speak(context: Context, text: String) {
        // 先停止之前的播放
        stopSpeaking()

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val token = accessToken ?: fetchToken()
                if (token.isNullOrEmpty()) {
                    Log.e(TAG, "No access token available!")
                    return@launch
                }

                synthesizeAndPlaySpeech(context, text, token)
            } catch (e: Exception) {
                Log.e(TAG, "Speak error: ${e.message}")
            }
        }
    }

    /**
     * 合成语音并播放
     */
    private suspend fun synthesizeAndPlaySpeech(context: Context, text: String, token: String) {
        val formBody = FormBody.Builder()
            .add("tex", text)
            .add("tok", token)
            .add("cuid", "blind_navigation_app")  // 修复拼写
            .add("ctp", "1")
            .add("lan", "zh")
            .add("spd", "5")
            .add("vol", "9")
            .add("pit", "5")
            .add("aue", "3") // mp3格式
            .build()

        val request = Request.Builder().url(TTS_URL).post(formBody).build()

        try {
            val response = client.newCall(request).execute()
            val contentType = response.header("Content-Type", "")

            if (contentType?.contains("audio") == true) {
                // 处理音频响应
                response.body?.bytes()?.let { audioData ->
                    val audioFile = File(context.cacheDir, "tts_${System.currentTimeMillis()}.mp3")
                    FileOutputStream(audioFile).use { it.write(audioData) }
                    playAudio(context, audioFile.absolutePath)
                }
            } else {
                val errorBody = response.body?.string() ?: "Unknown error"
                Log.e(TAG, "TTS API error: $errorBody")
            }
        } catch (e: Exception) {
            Log.e(TAG, "TTS request failed: ${e.message}")
        }
    }

    /**
     * 播放音频
     */
    private fun playAudio(context: Context, audioPath: String) {
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
                        player.start()
                        Log.d(TAG, "Audio playback started")
                    }
                    setOnCompletionListener { player ->
                        player.release()
                        mediaPlayer = null
                        Log.d(TAG, "Audio playback completed")
                    }
                    setOnErrorListener { player, what, extra ->
                        Log.e(TAG, "MediaPlayer error: what=$what, extra=$extra")
                        player.release()
                        mediaPlayer = null
                        true
                    }
                    prepareAsync()
                }
            } catch (e: Exception) {
                Log.e(TAG, "playAudio error: ${e.message}")
                mediaPlayer?.release()
                mediaPlayer = null
            }
        }
    }

    /**
     * 停止播放
     */
    fun stopSpeaking() {
        mediaPlayer?.let { player ->
            if (player.isPlaying) {
                player.stop()
            }
            player.release()
            mediaPlayer = null
        }
        Log.d(TAG, "Speech stopped")
    }

    /**
     * 释放资源
     */
    fun release() {
        stopSpeaking()
        // 可以在这里添加其他清理逻辑
    }
}