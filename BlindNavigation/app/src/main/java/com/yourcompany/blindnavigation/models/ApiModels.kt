package com.yourcompany.blindnavigation.models
import com.google.gson.annotations.SerializedName

/**
 * 障碍物检测相关模型
 */
data class ObstacleDetectionRequest(
    @SerializedName("image_data") val imageData: String? = null,
    @SerializedName("sensor_data") val sensorData: List<SensorData>? = null,
    @SerializedName("location") val location: LocationData? = null
)

data class ObstacleDetectionResponse(
    @SerializedName("obstacles") val obstacles: List<Obstacle>? = emptyList(),
    @SerializedName("confidence") val confidence: Double? = null,
    @SerializedName("warning_level") val warningLevel: String? = null // "low", "medium", "high"
)

data class Obstacle(
    @SerializedName("type") val type: String? = null,
    @SerializedName("distance") val distance: Double? = null,
    @SerializedName("direction") val direction: String? = null,
    @SerializedName("severity") val severity: String? = null
)

/**
 * 环境信息相关模型
 */
data class EnvironmentInfoResponse(
    @SerializedName("weather") val weather: WeatherInfo? = null,
    @SerializedName("traffic") val traffic: TrafficInfo? = null,
    @SerializedName("hazards") val hazards: List<Hazard>? = emptyList(),
    @SerializedName("accessibility") val accessibility: AccessibilityInfo? = null
)

data class WeatherInfo(
    @SerializedName("condition") val condition: String? = null,
    @SerializedName("temperature") val temperature: Double? = null,
    @SerializedName("humidity") val humidity: Double? = null
)

data class TrafficInfo(
    @SerializedName("level") val level: String? = null,
    @SerializedName("description") val description: String? = null
)

data class Hazard(
    @SerializedName("type") val type: String? = null,
    @SerializedName("location") val location: LocationData? = null,
    @SerializedName("description") val description: String? = null
)

/**
 * 语音指令相关模型
 */
data class VoiceCommandRequest(
    @SerializedName("audio_data") val audioData: String? = null,
    @SerializedName("text_command") val textCommand: String? = null
)

data class VoiceCommandResponse(
    @SerializedName("intent") val intent: String? = null,
    @SerializedName("action") val action: String? = null,
    @SerializedName("parameters") val parameters: Map<String, String>? = null,
    @SerializedName("response_text") val responseText: String? = null
)

/**
 * 位置相关模型
 */
data class LocationResponse(
    @SerializedName("location") val location: LocationData? = null,
    @SerializedName("accuracy") val accuracy: Double? = null,
    @SerializedName("timestamp") val timestamp: Long? = null
)

data class POIResponse(
    @SerializedName("pois") val pois: List<PointOfInterest>? = emptyList(),
    @SerializedName("count") val count: Int? = null
)

/**
 * 通用数据模型
 */
data class LocationData(
    @SerializedName("latitude") val latitude: Double,
    @SerializedName("longitude") val longitude: Double,
    @SerializedName("address") val address: String? = null
)

data class SensorData(
    @SerializedName("type") val type: String,
    @SerializedName("value") val value: Double,
    @SerializedName("timestamp") val timestamp: Long
)

data class RoutePreferences(
    @SerializedName("avoid_stairs") val avoidStairs: Boolean = true,
    @SerializedName("avoid_crowds") val avoidCrowds: Boolean = false,
    @SerializedName("prefer_accessible") val preferAccessible: Boolean = true
)

/**
 * 导航路径相关模型
 *
 * 说明：
 * - 请求字段使用 origin / destination（与后端期望一致）
 * - 响应包含两种常见形式：新的 total_distance/steps 等以及 legacy 的 routes 字段。
 */
data class RouteRequest(
    @SerializedName("origin") val origin: LocationData,
    @SerializedName("destination") val destination: LocationData,
    @SerializedName("mode") val mode: String = "walking", // walking, public_transit
    @SerializedName("preferences") val preferences: RoutePreferences? = null
)

/**
 * 与后端实际返回结构对齐的 RouteResponse（以后端返回为准）
 * - 示例字段：total_distance, total_time_min, total_steps, calories, steps (list)
 */
data class RouteResponse(
    @SerializedName("success") val success: Boolean = false,
    @SerializedName("message") val message: String? = null,
    @SerializedName("total_distance") val totalDistance: Double? = null,
    @SerializedName("total_duration") val totalDuration: Int? = null,
    @SerializedName("steps") val steps: List<StepItem>? = emptyList()
)

data class StepItem(
    @SerializedName("instruction") val instruction: String? = null,
    @SerializedName("distance") val distance: Double? = null,
    @SerializedName("duration") val duration: Int? = null,
    @SerializedName("orientation") val orientation: String? = null,
    @SerializedName("polyline") val polyline: String? = null,
    @SerializedName("road") val road: Any? = null
)

data class ObstacleItem(
    @SerializedName("class_name") val className: String? = null,
    @SerializedName("confidence") val confidence: Double? = null,
    @SerializedName("distance_estimate") val distanceEstimate: Double? = null,
    @SerializedName("box") val box: List<Int>? = null
)

/**
 * 为兼容原来代码保留的类型（如果其他模块仍在使用）
 */
data class Route(
    @SerializedName("steps") val steps: List<RouteStep>? = null,
    @SerializedName("polyline") val polyline: String? = null
)

data class RouteStep(
    @SerializedName("instruction") val instruction: String? = null,
    @SerializedName("distance") val distance: Double? = null,
    @SerializedName("duration") val duration: Int? = null,
    @SerializedName("type") val type: String? = null,
    // 修复：补充缺失的经纬度字段，解决Unresolved reference报错
    @SerializedName("end_lat") val endLat: Any? = null,
    @SerializedName("end_lon") val endLon: Any? = null
)

/**
 * 点位模型等
 */
data class PointOfInterest(
    @SerializedName("name") val name: String? = null,
    @SerializedName("type") val type: String? = null,
    @SerializedName("location") val location: LocationData? = null,
    @SerializedName("distance") val distance: Double? = null
)

data class AccessibilityInfo(
    @SerializedName("sidewalk_quality") val sidewalkQuality: String? = null,
    @SerializedName("curb_ramps") val curbRamps: Boolean? = null,
    @SerializedName("tactile_paving") val tactilePaving: Boolean? = null
)

/**
 * 图像分析请求
 */
data class ImageAnalysisRequest(
    @SerializedName("image_data")
    val imageData: String, // base64 编码的图像数据
    @SerializedName("location")
    val location: LocationData? = null
)

/**
 * 图像分析响应
 */
data class ImageAnalysisResponse(
    @SerializedName("obstacles")
    val obstacles: List<Obstacle>? = emptyList(),
    @SerializedName("confidence")
    val confidence: Double? = null,
    @SerializedName("warning_level")
    val warningLevel: String? = null
)

/**
 * 路径规划请求
 */
data class RoutePlanRequest(
    @SerializedName("origin")
    val origin: LocationData,
    @SerializedName("destination")
    val destination: LocationData,
    @SerializedName("preferences")
    val preferences: RoutePreferences? = null
)

/**
 * 路径规划响应
 */
data class RoutePlanResponse(
    @SerializedName("route")
    val route: Route? = null,
    @SerializedName("total_distance")
    val totalDistance: Int? = null,
    @SerializedName("estimated_time")
    val estimatedTime: Int? = null
)

/**
 * 位置更新请求
 */
data class LocationUpdateRequest(
    @SerializedName("location")
    val location: LocationData,
    @SerializedName("timestamp")
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * 位置更新响应
 */
data class LocationUpdateResponse(
    @SerializedName("success")
    val success: Boolean,
    @SerializedName("message")
    val message: String? = null
)

/**
 * 系统状态响应
 */
data class SystemStatusResponse(
    @SerializedName("status")
    val status: String,
    @SerializedName("version")
    val version: String? = null,
    @SerializedName("uptime")
    val uptime: Long? = null
)

/**
 * 导航状态响应
 */
data class NavigationStatusResponse(
    @SerializedName("current_location")
    val currentLocation: LocationData? = null,
    @SerializedName("destination")
    val destination: LocationData? = null,
    @SerializedName("progress")
    val progress: Double? = null,
    @SerializedName("next_instruction")
    val nextInstruction: String? = null
)

/**
 * 语音合成请求
 */
data class SpeechSynthesisRequest(
    @SerializedName("text")
    val text: String,
    @SerializedName("language")
    val language: String = "zh-CN"
)

/**
 * 语音合成响应
 */
data class SpeechSynthesisResponse(
    @SerializedName("audio_data")
    val audioData: String? = null, // base64 编码的音频数据
    @SerializedName("duration")
    val duration: Int? = null
)

/**
 * 数据上传请求
 */
data class DataUploadRequest(
    @SerializedName("sensor_data")
    val sensorData: List<SensorData>? = null,
    @SerializedName("usage_data")
    val usageData: Map<String, Any>? = null
)

/**
 * 数据上传响应
 */
data class DataUploadResponse(
    @SerializedName("success")
    val success: Boolean,
    @SerializedName("records_processed")
    val recordsProcessed: Int? = null
)

// 在 ApiModels.kt 文件末尾添加以下内容

/**
 * 登录请求
 */
data class LoginRequest(
    @SerializedName("username") val username: String,
    @SerializedName("password") val password: String
)

/**
 * 登录响应
 */
data class LoginResponse(
    @SerializedName("success") val success: Boolean,
    @SerializedName("message") val message: String? = null,
    @SerializedName("token") val token: String? = null,
    @SerializedName("user_id") val userId: Int? = null,
    @SerializedName("username") val username: String? = null,
    @SerializedName("phone") val phone: String? = null,           // 👈 添加手机号字段
    @SerializedName("emergency_contact") val emergencyContact: String? = null  // 👈 添加紧急联系人字段
)

/**
 * 注册请求
 */
data class RegisterRequest(
    @SerializedName("username") val username: String,
    @SerializedName("password") val password: String,
    @SerializedName("phone") val phone: String,
    @SerializedName("emergency_contact") val emergencyContact: String? = null
)

/**
 * 健康检查响应
 */
data class HealthResponse(
    @SerializedName("status") val status: String,
    @SerializedName("version") val version: String? = null,
    @SerializedName("timestamp") val timestamp: Long? = null
)

/**
 * 检测响应
 */
data class DetectionResponse(
    @SerializedName("success") val success: Boolean,
    @SerializedName("message") val message: String? = null,
    @SerializedName("obstacles") val obstacles: List<ObstacleItem> = emptyList()
)
// 修复：补充缺失的User模型，解决登录/注册时的Unresolved reference报错
// 在 ApiModels.kt 文件末尾添加

/**
 * 用户信息更新请求
 */
data class UpdateUserRequest(
    @SerializedName("nickname") val nickname: String? = null,
    @SerializedName("phone") val phone: String? = null,
    @SerializedName("email") val email: String? = null,
    @SerializedName("emergency_contact") val emergencyContact: String? = null,
    @SerializedName("avatar") val avatar: String? = null
)

/**
 * 用户信息响应
 */
data class UserInfoResponse(
    @SerializedName("success") val success: Boolean,
    @SerializedName("message") val message: String? = null,
    @SerializedName("user") val user: UserData? = null
)

/**
 * 用户数据
 */
data class UserData(
    @SerializedName("id") val id: Int,
    @SerializedName("username") val username: String,
    @SerializedName("nickname") val nickname: String? = null,
    @SerializedName("phone") val phone: String? = null,
    @SerializedName("email") val email: String? = null,
    @SerializedName("emergency_contact") val emergencyContact: String? = null,
    @SerializedName("avatar") val avatar: String? = null,
    @SerializedName("register_time") val registerTime: Long,
    @SerializedName("last_login_time") val lastLoginTime: Long
)

/**
 * 存储数据统计响应
 */
data class StorageStatsResponse(
    @SerializedName("success") val success: Boolean,
    @SerializedName("data") val data: StorageStats? = null
)

data class StorageStats(
    @SerializedName("cache_size") val cacheSize: Long,
    @SerializedName("user_data_size") val userDataSize: Long,
    @SerializedName("total_size") val totalSize: Long
)

/**
 * 清除缓存请求
 */
data class ClearCacheRequest(
    @SerializedName("user_id") val userId: Int
)

/**
 * 清除数据响应
 */
data class ClearDataResponse(
    @SerializedName("success") val success: Boolean,
    @SerializedName("message") val message: String? = null,
    @SerializedName("cleared_size") val clearedSize: Long? = null
)