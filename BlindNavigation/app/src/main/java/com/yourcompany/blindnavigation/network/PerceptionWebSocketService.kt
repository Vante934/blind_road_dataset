// 文件路径：app/src/main/java/com/yourcompany/blindnavigation/network/PerceptionWebSocketService.kt

package com.yourcompany.blindnavigation.network

import android.util.Log
import com.google.gson.Gson
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import okhttp3.*
import java.util.concurrent.TimeUnit

/**
 * 感知服务WebSocket客户端（连接A成员）
 * 负责：发送摄像头画面、音频、TOF数据，接收障碍物预警
 */
class PerceptionWebSocketService(private val deviceId: String) {

    private val TAG = "PerceptionWebSocket"
    private val gson = Gson()

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(30, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private var reconnectJob: Job? = null
    private var heartbeatJob: Job? = null

    private val _connectionState = MutableStateFlow(ConnectionState.DISCONNECTED)
    val connectionState: StateFlow<ConnectionState> = _connectionState

    private val _warningMessages = MutableSharedFlow<WarningMessage>(replay = 0)
    val warningMessages: SharedFlow<WarningMessage> = _warningMessages

    enum class ConnectionState {
        CONNECTING, CONNECTED, DISCONNECTED, ERROR
    }

    fun connect() {
        Log.e(TAG, "========== connect() 被调用 ==========")
        Log.e(TAG, "设备ID: $deviceId")
        Log.e(TAG, "WS_URL: ${ApiConfig.PerceptionService.WS_URL}")
        Log.e(TAG, "WS_NAVIGATION: ${ApiConfig.PerceptionService.WS_NAVIGATION}")

        if (_connectionState.value == ConnectionState.CONNECTED ||
            _connectionState.value == ConnectionState.CONNECTING) {
            Log.w(TAG, "已经连接或正在连接中")
            return
        }

        // 连接A成员的感知服务
        val url = "${ApiConfig.PerceptionService.WS_URL}${ApiConfig.PerceptionService.WS_NAVIGATION}/$deviceId"
        Log.e(TAG, "完整URL: $url")
        val request = Request.Builder().url(url).build()

        Log.i(TAG, "正在连接感知服务: $url")
        _connectionState.value = ConnectionState.CONNECTING

        webSocket = client.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.e(TAG, "✅ 感知服务已连接，设备ID: $deviceId")
                _connectionState.value = ConnectionState.CONNECTED
                reconnectJob?.cancel()
                startHeartbeat()
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d(TAG, "📥 收到感知服务消息: $text")

                CoroutineScope(Dispatchers.IO).launch {
                    try {
                        val wsMessage = gson.fromJson(text, Map::class.java)
                        val type = wsMessage["type"] as? String ?: ""
                        val data = wsMessage["data"] as? Map<*, *>

                        when (type) {
                            "warning" -> {
                                Log.e(TAG, "⚠️ 收到预警消息")
                                handleWarningMessage(data)
                            }
                            "ping" -> {
                                sendPong()
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "消息解析失败", e)
                    }
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "❌ 感知服务错误", t)
                _connectionState.value = ConnectionState.ERROR
                stopHeartbeat()
                scheduleReconnect()
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "感知服务已关闭: $code - $reason")
                _connectionState.value = ConnectionState.DISCONNECTED
                stopHeartbeat()
            }
        })
    }

    private fun handleWarningMessage(data: Map<*, *>?) {
        if (data == null) return

        // ✅ 正确解析字段
        val warningLevel = when (val level = data["warning_level"]) {
            is Int -> level
            is Double -> level.toInt()
            is Number -> level.toInt()
            else -> 0
        }

        val warningLevelName = data["warning_level_name"] as? String ?: ""
        val ttsText = data["tts_text"] as? String ?: ""

        // ✅ 正确解析振动强度
        val vibrationIntensity = when (val vib = data["vibration_intensity"]) {
            is Int -> vib
            is Double -> vib.toInt()
            is Number -> vib.toInt()
            else -> 0
        }

        val vibrationPattern = data["vibration_pattern"] as? String ?: ""
        val primaryThreat = data["primary_threat"] as? String ?: ""
        val threatBreakdown = data["threat_breakdown"] as? Map<*, *>

        Log.e(TAG, "⚠️ 预警内容: level=$warningLevel, text=$ttsText, vibration=$vibrationIntensity")

        val warningMsg = WarningMessage(
            type = "warning",
            warningLevel = warningLevel,
            warningLevelName = warningLevelName,
            ttsText = ttsText,
            vibrationIntensity = vibrationIntensity,
            vibrationPattern = vibrationPattern,
            primaryThreat = primaryThreat,
            threatBreakdown = threatBreakdown,
            obstaclesInfo = null
        )

        CoroutineScope(Dispatchers.IO).launch {
            _warningMessages.emit(warningMsg)
        }
    }

    fun sendSensorData(
        tofDistance: Double,
        tofDirection: String,
        videoFrameBase64: String? = null,
        audioBase64: String? = null
    ) {
        val audioData = if (audioBase64 != null) {
            mapOf(
                "audio_base64" to audioBase64,
                "audio_format" to "pcm",
                "sample_rate" to 16000,
                "channel" to 1
            )
        } else null

        val message = mapOf(
            "type" to "sensor_data",
            "data" to mapOf(
                "device_id" to deviceId,
                "timestamp" to System.currentTimeMillis(),
                "tof_distance" to tofDistance,
                "tof_direction" to tofDirection,
                "video_frame" to videoFrameBase64,
                "audio_data" to audioData
            )
        )

        val json = gson.toJson(message)
        webSocket?.send(json)
        Log.d(TAG, "📤 发送传感器数据: distance=$tofDistance, direction=$tofDirection")
    }

    private fun startHeartbeat() {
        stopHeartbeat()
        heartbeatJob = CoroutineScope(Dispatchers.IO).launch {
            while (_connectionState.value == ConnectionState.CONNECTED) {
                delay(30000)
                sendHeartbeat()
            }
        }
    }

    private fun stopHeartbeat() {
        heartbeatJob?.cancel()
        heartbeatJob = null
    }

    private fun sendHeartbeat() {
        val message = mapOf("type" to "heartbeat", "data" to emptyMap<String, Any>())
        webSocket?.send(gson.toJson(message))
        Log.d(TAG, "💓 发送心跳")
    }

    private fun sendPong() {
        val message = mapOf("type" to "pong", "timestamp" to System.currentTimeMillis() / 1000.0)
        webSocket?.send(gson.toJson(message))
    }

    private fun scheduleReconnect() {
        reconnectJob?.cancel()
        reconnectJob = CoroutineScope(Dispatchers.IO).launch {
            delay(ApiConfig.WS_RECONNECT_DELAY)
            if (_connectionState.value != ConnectionState.CONNECTED) {
                Log.i(TAG, "🔄 尝试重连感知服务...")
                connect()
            }
        }
    }

    fun disconnect() {
        stopHeartbeat()
        reconnectJob?.cancel()
        webSocket?.close(1000, "用户断开")
        _connectionState.value = ConnectionState.DISCONNECTED
        Log.i(TAG, "感知服务已主动断开")
    }
}

data class WarningMessage(
    val type: String,
    val warningLevel: Int,
    val warningLevelName: String,
    val ttsText: String,
    val vibrationIntensity: Int,
    val vibrationPattern: String,
    val primaryThreat: String,
    val threatBreakdown: Map<*, *>?,
    val obstaclesInfo: List<*>?
)