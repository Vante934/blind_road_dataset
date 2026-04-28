package com.yourcompany.blindnavigation.utils

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream

object AudioRecorder {
    private const val TAG = "AudioRecorder"

    // 音频配置
    private const val SAMPLE_RATE = 16000
    private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordJob: Job? = null

    // 录音时长（秒）
    private const val RECORD_DURATION_SECONDS = 5

    /**
     * 开始录音（异步）
     * @param onAudioReady 录音完成后的回调，返回 Base64 编码的音频数据
     */
    fun startRecording(onAudioReady: (String?) -> Unit) {
        if (isRecording) {
            Log.w(TAG, "正在录音中，请稍后再试")
            onAudioReady(null)
            return
        }

        recordJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                val audioData = recordAudio()
                if (audioData.isNotEmpty()) {
                    val base64Audio = Base64.encodeToString(audioData, Base64.NO_WRAP)
                    Log.e(TAG, "✅ 录音完成，音频大小: ${audioData.size} bytes, Base64长度: ${base64Audio.length}")
                    withContext(Dispatchers.Main) {
                        onAudioReady(base64Audio)
                    }
                } else {
                    Log.e(TAG, "❌ 录音数据为空")
                    withContext(Dispatchers.Main) {
                        onAudioReady(null)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "❌ 录音失败: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    onAudioReady(null)
                }
            }
        }
    }

    /**
     * 录制音频
     * @return PCM 音频字节数组
     */
    private fun recordAudio(): ByteArray {
        // 计算缓冲区大小（5秒的音频数据）
        val bufferSizeInBytes = SAMPLE_RATE * 2 * RECORD_DURATION_SECONDS  // 16bit = 2字节
        val buffer = ByteArray(bufferSizeInBytes)

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            BUFFER_SIZE
        ).apply {
            startRecording()
            isRecording = true
            Log.e(TAG, "🎤 开始录音，持续 ${RECORD_DURATION_SECONDS} 秒...")
        }

        var totalRead = 0
        var read: Int

        while (isRecording && totalRead < bufferSizeInBytes) {
            read = audioRecord?.read(buffer, totalRead, bufferSizeInBytes - totalRead) ?: 0
            if (read > 0) {
                totalRead += read
            } else if (read == AudioRecord.ERROR_INVALID_OPERATION || read == AudioRecord.ERROR_BAD_VALUE) {
                Log.e(TAG, "录音错误: read = $read")
                break
            }
        }

        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        isRecording = false

        Log.e(TAG, "🎤 录音结束，共录制 ${totalRead} 字节")

        // 返回实际录制的数据
        return buffer.copyOfRange(0, totalRead)
    }

    /**
     * 停止录音
     */
    fun stopRecording() {
        isRecording = false
        recordJob?.cancel()
        audioRecord?.let {
            try {
                it.stop()
                it.release()
            } catch (e: Exception) {
                Log.e(TAG, "停止录音异常: ${e.message}")
            }
        }
        audioRecord = null
        Log.d(TAG, "录音已停止")
    }
}