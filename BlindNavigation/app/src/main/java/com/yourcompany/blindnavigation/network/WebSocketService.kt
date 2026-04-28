// 文件路径：app/src/main/java/com/yourcompany/blindnavigation/network/WebSocketService.kt

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
 * 导航服务WebSocket客户端（连接B成员）
 * 负责：发送GPS坐标，接收导航指令（转弯提示）
 */
class WebSocketService(private val userId: String) {

    private val TAG = "WebSocketService"
    private val gson = Gson()

    // 普通 OkHttpClient，不需要 SSL（因为用 ws://）
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(30, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private var reconnectJob: Job? = null

    private val _connectionState = MutableStateFlow(ConnectionState.DISCONNECTED)
    val connectionState: StateFlow<ConnectionState> = _connectionState

    private val _messages = MutableSharedFlow<NavigationMessage>(replay = 0)
    val messages: SharedFlow<NavigationMessage> = _messages

    enum class ConnectionState {
        CONNECTING, CONNECTED, DISCONNECTED, ERROR
    }

    fun connect() {
        Log.e(TAG, "========== connect() 被调用 ==========")
        Log.e(TAG, "用户ID: $userId")

        if (_connectionState.value == ConnectionState.CONNECTED ||
            _connectionState.value == ConnectionState.CONNECTING) {
            Log.w(TAG, "已经连接或正在连接中")
            return
        }

        // 正确的 URL：ws://474451a.r6.cpolar.cn/ws/nav/yssss
        val url = "${ApiConfig.NavigationService.WS_URL}${ApiConfig.NavigationService.WS_NAVIGATION}/$userId"
        Log.e(TAG, "完整URL: $url")
        val request = Request.Builder().url(url).build()

        Log.i(TAG, "正在连接导航服务: $url")
        _connectionState.value = ConnectionState.CONNECTING

        webSocket = client.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.e(TAG, "✅ 导航服务已连接，用户ID: $userId")
                _connectionState.value = ConnectionState.CONNECTED
                reconnectJob?.cancel()
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d(TAG, "📥 收到导航服务消息: $text")

                CoroutineScope(Dispatchers.IO).launch {
                    try {
                        val wsMessage = gson.fromJson(text, Map::class.java)
                        val type = wsMessage["type"] as? String ?: ""
                        val data = wsMessage["data"] as? Map<*, *>

                        when (type) {
                            "position_update" -> {
                                handlePositionUpdate(data)
                            }
                            "route_plan" -> {
                                handleRoutePlan(data)
                            }
                            "nav_started" -> {
                                Log.e(TAG, "🚀 收到导航开始消息")
                                _messages.emit(NavigationMessage(
                                    type = "nav_started",
                                    smoothedLat = null, smoothedLng = null,
                                    predictedLat = null, predictedLng = null,
                                    instruction = "导航已开始",
                                    currentStep = null, totalSteps = null,
                                    distanceToNext = null
                                ))
                            }
                            "nav_stopped" -> {
                                Log.e(TAG, "🛑 收到导航结束消息")
                                _messages.emit(NavigationMessage(
                                    type = "nav_stopped",
                                    smoothedLat = null, smoothedLng = null,
                                    predictedLat = null, predictedLng = null,
                                    instruction = "导航已结束",
                                    currentStep = null, totalSteps = null,
                                    distanceToNext = null
                                ))
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
                Log.e(TAG, "❌ 导航服务错误", t)
                _connectionState.value = ConnectionState.ERROR
                scheduleReconnect()
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "导航服务已关闭: $code - $reason")
                _connectionState.value = ConnectionState.DISCONNECTED
            }
        })
    }

    private suspend fun handlePositionUpdate(data: Map<*, *>?) {
        if (data == null) return

        val smoothed = data["smoothed"] as? Map<*, *>
        val instruction = data["instruction"] as? Map<*, *>

        val instructionText = instruction?.get("message") as? String
        Log.e(TAG, "📍 位置更新指令: $instructionText")

        _messages.emit(NavigationMessage(
            type = "position_update",
            smoothedLat = (smoothed?.get("smoothed_lat") as? Double),
            smoothedLng = (smoothed?.get("smoothed_lng") as? Double),
            predictedLat = (smoothed?.get("predicted_lat") as? Double),
            predictedLng = (smoothed?.get("predicted_lng") as? Double),
            instruction = instructionText,
            currentStep = (instruction?.get("current_step") as? Double)?.toInt(),
            totalSteps = (instruction?.get("total_steps") as? Double)?.toInt(),
            distanceToNext = (instruction?.get("distance_to_next") as? Double)
        ))
    }

    private suspend fun handleRoutePlan(data: Map<*, *>?) {
        if (data == null) return
        Log.e(TAG, "🗺️ 收到路径规划结果: $data")
    }

    private fun scheduleReconnect() {
        reconnectJob?.cancel()
        reconnectJob = CoroutineScope(Dispatchers.IO).launch {
            delay(3000)  // 重连延迟 3 秒，不要疯狂重连
            if (_connectionState.value != ConnectionState.CONNECTED) {
                Log.i(TAG, "🔄 尝试重连导航服务...")
                connect()
            }
        }
    }

    fun disconnect() {
        reconnectJob?.cancel()
        webSocket?.close(1000, "用户断开")
        _connectionState.value = ConnectionState.DISCONNECTED
        Log.i(TAG, "导航服务已主动断开")
    }

    // 发送GPS数据时，检查连接状态
    fun sendGPSUpdate(gpsData: GPSData) {
        // 👇 只在连接成功时才发送数据
        if (_connectionState.value != ConnectionState.CONNECTED) {
            Log.d(TAG, "⏸️ 导航服务未连接，跳过GPS发送")
            return
        }

        val message = mapOf(
            "type" to "gps_update",
            "data" to mapOf(
                "latitude" to gpsData.latitude,
                "longitude" to gpsData.longitude,
                "accuracy" to gpsData.accuracy,
                "timestamp" to gpsData.timestamp,
                "speed" to gpsData.speed,
                "bearing" to gpsData.bearing
            )
        )

        val json = gson.toJson(message)
        webSocket?.send(json)
        Log.d(TAG, "📤 发送GPS数据: (${gpsData.latitude}, ${gpsData.longitude})")
    }

    fun startNavigation(steps: List<Map<String, Any>>) {
        if (_connectionState.value != ConnectionState.CONNECTED) {
            Log.w(TAG, "导航服务未连接，无法发送开始导航指令")
            return
        }

        val message = mapOf(
            "type" to "start_navigation",
            "data" to mapOf("steps" to steps),
            "timestamp" to System.currentTimeMillis() / 1000.0
        )

        val json = gson.toJson(message)
        webSocket?.send(json)
        Log.i(TAG, "📤 发送开始导航指令")
    }

    fun stopNavigation() {
        if (_connectionState.value != ConnectionState.CONNECTED) {
            Log.w(TAG, "导航服务未连接，无法发送停止导航指令")
            return
        }

        val message = mapOf(
            "type" to "stop_navigation",
            "data" to emptyMap<String, Any>(),
            "timestamp" to System.currentTimeMillis() / 1000.0
        )

        val json = gson.toJson(message)
        webSocket?.send(json)
        Log.i(TAG, "📤 发送停止导航指令")
    }

    private fun sendPong() {
        if (_connectionState.value != ConnectionState.CONNECTED) return

        val message = mapOf(
            "type" to "pong",
            "timestamp" to System.currentTimeMillis() / 1000.0
        )

        val json = gson.toJson(message)
        webSocket?.send(json)
        Log.d(TAG, "💓 发送心跳响应")
    }
}