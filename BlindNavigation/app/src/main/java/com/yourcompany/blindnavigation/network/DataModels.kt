// 文件路径：app/src/main/java/com/yourcompany/blindnavigation/network/DataModels.kt

package com.yourcompany.blindnavigation.network

import com.google.gson.annotations.SerializedName

/**
 * 网络请求/响应数据模型
 */

// =====================================
// 认证相关
// =====================================

data class LoginRequest(
    val username: String,
    val password: String
)

data class LoginResponse(
    val success: Boolean,
    val token: String?,
    val message: String,
    @SerializedName("user_id")
    val userId: Int?,
    val phone: String? = null,           // 👈 添加手机号字段
    @SerializedName("emergency_contact")
    val emergencyContact: String? = null  // 👈 添加紧急联系人字段
)

// 修改 RegisterRequest 以匹配后端要求
data class RegisterRequest(
    val username: String,           // 必填
    val password: String,           // 必填
    val phone: String,              // 必填
    @SerializedName("emergency_contact")
    val emergencyContact: String? = null  // 选填
)

// =====================================
// 导航相关
// =====================================

data class RouteRequest(
    @SerializedName("origin_lng")
    val originLng: Double,
    @SerializedName("origin_lat")
    val originLat: Double,
    @SerializedName("dest_lng")
    val destLng: Double,
    @SerializedName("dest_lat")
    val destLat: Double
)

data class RouteStep(
    val instruction: String,
    val distance: Double,
    val polyline: String,
    val orientation: String,
    val road: String? = null,
    val duration: Int? = null
)

data class RouteResponse(
    val success: Boolean,
    @SerializedName("total_distance")
    val totalDistance: Double,
    @SerializedName("total_duration")
    val totalDuration: Int,
    val steps: List<RouteStep>,
    val message: String
)

// =====================================
// 检测相关
// =====================================

data class DetectionResult(
    @SerializedName("class_name")
    val className: String,
    val confidence: Double,
    val bbox: List<Double>,
    @SerializedName("distance_estimate")
    val distanceEstimate: Double?,
    val direction: String,
    @SerializedName("danger_level")
    val dangerLevel: String
)

data class DetectionResponse(
    val success: Boolean,
    val obstacles: List<DetectionResult>,
    @SerializedName("blind_road_detected")
    val blindRoadDetected: Boolean,
    @SerializedName("blind_road_status")
    val blindRoadStatus: String?,
    @SerializedName("voice_alert")
    val voiceAlert: String?,
    @SerializedName("processing_time_ms")
    val processingTimeMs: Double
)

// =====================================
// WebSocket消息
// =====================================

data class GPSData(
    val latitude: Double,
    val longitude: Double,
    val accuracy: Float,
    val timestamp: Double,
    val speed: Float? = null,
    val bearing: Float? = null
)

data class NavigationMessage(
    val type: String,
    val smoothedLat: Double?,
    val smoothedLng: Double?,
    val predictedLat: Double?,
    val predictedLng: Double?,
    val instruction: String?,
    val currentStep: Int?,
    val totalSteps: Int?,
    val distanceToNext: Double?
)

// =====================================
// 健康检查
// =====================================

data class HealthResponse(
    val status: String,
    val timestamp: Double
)