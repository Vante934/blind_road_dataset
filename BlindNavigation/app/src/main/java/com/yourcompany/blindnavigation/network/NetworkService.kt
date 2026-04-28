// 文件路径：app/src/main/java/com/yourcompany/blindnavigation/network/NetworkService.kt

package com.yourcompany.blindnavigation.network

import android.util.Log
import com.google.gson.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.logging.HttpLoggingInterceptor
import com.yourcompany.blindnavigation.models.*
import java.util.concurrent.TimeUnit

/**
 * HTTP网络请求服务
 * 负责所有REST API调用
 * 所有请求都连接到 B 成员（导航服务）
 */
class NetworkService {

    private val TAG = "NetworkService"
    private val gson = Gson()
    private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()

    // OkHttp客户端 - 连接 B 成员
    private val client = OkHttpClient.Builder()
        .connectTimeout(ApiConfig.CONNECT_TIMEOUT, TimeUnit.SECONDS)
        .readTimeout(ApiConfig.READ_TIMEOUT, TimeUnit.SECONDS)
        .writeTimeout(ApiConfig.WRITE_TIMEOUT, TimeUnit.SECONDS)
        .addInterceptor(HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        })
        .build()

    // =====================================
    // 健康检查 - 连接 B 成员
    // =====================================
    suspend fun healthCheck(): Result<HealthResponse> = withContext(Dispatchers.IO) {
        try {
            val url = "${ApiConfig.NavigationService.BASE_URL}${ApiConfig.NavigationService.HEALTH}"
            Log.d(TAG, "健康检查 URL: $url")

            val request = Request.Builder()
                .url(url)
                .get()
                .build()

            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""

            Log.d(TAG, "健康检查响应码: ${response.code}")

            if (response.isSuccessful) {
                val data = gson.fromJson(body, HealthResponse::class.java)
                Result.success(data)
            } else {
                Result.failure(Exception("HTTP ${response.code}: $body"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "健康检查失败", e)
            Result.failure(e)
        }
    }

    // =====================================
    // 用户认证 - 连接 B 成员
    // =====================================
    suspend fun login(username: String, password: String): Result<LoginResponse> =
        withContext(Dispatchers.IO) {
            try {
                val requestBody = gson.toJson(LoginRequest(username, password))
                    .toRequestBody(JSON_MEDIA_TYPE)

                Log.d(TAG, "登录请求: ${gson.toJson(LoginRequest(username, password))}")

                val request = Request.Builder()
                    .url("${ApiConfig.NavigationService.BASE_URL}${ApiConfig.NavigationService.LOGIN}")
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()
                val body = response.body?.string() ?: ""

                Log.d(TAG, "登录响应码: ${response.code}")
                Log.d(TAG, "登录响应体: $body")

                if (response.isSuccessful) {
                    val data = gson.fromJson(body, LoginResponse::class.java)
                    Result.success(data)
                } else {
                    Result.failure(Exception("登录失败: HTTP ${response.code}, 响应: $body"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "登录异常", e)
                Result.failure(e)
            }
        }

    suspend fun register(
        username: String,
        password: String,
        phone: String,
        emergencyContact: String? = null
    ): Result<LoginResponse> = withContext(Dispatchers.IO) {
        try {
            val registerRequest = RegisterRequest(
                username = username,
                password = password,
                phone = phone,
                emergencyContact = emergencyContact
            )

            val requestBody = gson.toJson(registerRequest)
                .toRequestBody(JSON_MEDIA_TYPE)

            Log.d(TAG, "注册请求: ${gson.toJson(registerRequest)}")

            val request = Request.Builder()
                .url("${ApiConfig.NavigationService.BASE_URL}${ApiConfig.NavigationService.REGISTER}")
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""

            Log.d(TAG, "注册响应码: ${response.code}")
            Log.d(TAG, "注册响应体: $body")

            if (response.isSuccessful) {
                val data = gson.fromJson(body, LoginResponse::class.java)
                Result.success(data)
            } else {
                Result.failure(Exception("注册失败: HTTP ${response.code}, 响应: $body"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "注册异常", e)
            Result.failure(e)
        }
    }

    // =====================================
    // 路径规划 - 连接 B 成员
    // =====================================
    suspend fun planRoute(
        originLng: Double,
        originLat: Double,
        destLng: Double,
        destLat: Double
    ): Result<RouteResponse> = withContext(Dispatchers.IO) {
        try {
            val requestBody = gson.toJson(
                mapOf(
                    "origin_lng" to originLng,
                    "origin_lat" to originLat,
                    "dest_lng" to destLng,
                    "dest_lat" to destLat
                )
            ).toRequestBody(JSON_MEDIA_TYPE)

            Log.d(TAG, "路径规划请求: $requestBody")

            val request = Request.Builder()
                .url("${ApiConfig.NavigationService.BASE_URL}${ApiConfig.NavigationService.PLAN_ROUTE}")
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""

            Log.d(TAG, "响应码: ${response.code}")
            Log.d(TAG, "响应体: $body")

            if (response.isSuccessful) {
                try {
                    val data = gson.fromJson(body, RouteResponse::class.java)
                    Log.d(TAG, "✅ 解析成功: success=${data.success}")
                    Result.success(data)
                } catch (e: Exception) {
                    Log.e(TAG, "❌ JSON解析失败", e)
                    Result.failure(Exception("解析响应失败: ${e.message}"))
                }
            } else {
                Result.failure(Exception("路径规划失败: HTTP ${response.code}"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "路径规划异常", e)
            Result.failure(e)
        }
    }

    // =====================================
    // 障碍物检测 - 连接 B 成员
    // =====================================
    suspend fun detectObstaclesFromBytes(
        imageBytes: ByteArray,
        latitude: Double = 0.0,
        longitude: Double = 0.0
    ): Result<DetectionResponse> = withContext(Dispatchers.IO) {
        try {
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "image",
                    "frame.jpg",
                    imageBytes.toRequestBody("image/jpeg".toMediaType())
                )
                .addFormDataPart("latitude", latitude.toString())
                .addFormDataPart("longitude", longitude.toString())
                .build()

            val request = Request.Builder()
                .url("${ApiConfig.NavigationService.BASE_URL}${ApiConfig.NavigationService.DETECT_OBSTACLE}")
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""

            Log.d(TAG, "检测响应码: ${response.code}")
            Log.d(TAG, "检测响应体: $body")

            if (response.isSuccessful) {
                val data = gson.fromJson(body, DetectionResponse::class.java)
                Result.success(data)
            } else {
                Result.failure(Exception("检测失败: HTTP ${response.code}, 响应: $body"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "检测异常", e)
            Result.failure(e)
        }
    }

    // =====================================
    // 用户信息相关 - 连接 B 成员
    // =====================================

    /**
     * 获取用户信息
     */
    suspend fun getUserInfo(userId: Int): Result<UserInfoResponse> = withContext(Dispatchers.IO) {
        try {
            val url = "${ApiConfig.NavigationService.BASE_URL}${ApiConfig.NavigationService.USER_INFO}/$userId"
            Log.d(TAG, "获取用户信息 URL: $url")

            val request = Request.Builder()
                .url(url)
                .get()
                .build()

            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""

            Log.d(TAG, "获取用户信息响应码: ${response.code}")

            if (response.isSuccessful) {
                val data = gson.fromJson(body, UserInfoResponse::class.java)
                Result.success(data)
            } else {
                Result.failure(Exception("获取用户信息失败: HTTP ${response.code}"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "获取用户信息异常", e)
            Result.failure(e)
        }
    }

    /**
     * 更新用户信息
     */
    suspend fun updateUserInfo(
        userId: Int,
        request: UpdateUserRequest
    ): Result<UserInfoResponse> = withContext(Dispatchers.IO) {
        try {
            val requestBody = gson.toJson(request).toRequestBody(JSON_MEDIA_TYPE)

            val url = "${ApiConfig.NavigationService.BASE_URL}${ApiConfig.NavigationService.UPDATE_USER_INFO}/$userId"
            Log.d(TAG, "更新用户信息 URL: $url")

            val request = Request.Builder()
                .url(url)
                .put(requestBody)
                .build()

            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""

            Log.d(TAG, "更新用户信息响应码: ${response.code}")

            if (response.isSuccessful) {
                val data = gson.fromJson(body, UserInfoResponse::class.java)
                Result.success(data)
            } else {
                Result.failure(Exception("更新用户信息失败: HTTP ${response.code}"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "更新用户信息异常", e)
            Result.failure(e)
        }
    }

    /**
     * 获取存储数据统计
     */
    suspend fun getStorageStats(userId: Int): Result<StorageStatsResponse> = withContext(Dispatchers.IO) {
        try {
            val url = "${ApiConfig.NavigationService.BASE_URL}${ApiConfig.NavigationService.STORAGE_STATS}/$userId"
            Log.d(TAG, "获取存储统计 URL: $url")

            val request = Request.Builder()
                .url(url)
                .get()
                .build()

            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""

            Log.d(TAG, "获取存储统计响应码: ${response.code}")

            if (response.isSuccessful) {
                val data = gson.fromJson(body, StorageStatsResponse::class.java)
                Result.success(data)
            } else {
                Result.failure(Exception("获取存储统计失败: HTTP ${response.code}"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "获取存储统计异常", e)
            Result.failure(e)
        }
    }

    /**
     * 清除缓存
     */
    suspend fun clearCache(userId: Int): Result<ClearDataResponse> = withContext(Dispatchers.IO) {
        try {
            val requestBody = gson.toJson(ClearCacheRequest(userId)).toRequestBody(JSON_MEDIA_TYPE)

            val url = "${ApiConfig.NavigationService.BASE_URL}${ApiConfig.NavigationService.CLEAR_CACHE}"
            Log.d(TAG, "清除缓存 URL: $url")

            val request = Request.Builder()
                .url(url)
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""

            Log.d(TAG, "清除缓存响应码: ${response.code}")

            if (response.isSuccessful) {
                val data = gson.fromJson(body, ClearDataResponse::class.java)
                Result.success(data)
            } else {
                Result.failure(Exception("清除缓存失败: HTTP ${response.code}"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "清除缓存异常", e)
            Result.failure(e)
        }
    }

    /**
     * 清除所有用户数据
     */
    suspend fun clearAllUserData(userId: Int): Result<ClearDataResponse> = withContext(Dispatchers.IO) {
        try {
            val url = "${ApiConfig.NavigationService.BASE_URL}${ApiConfig.NavigationService.CLEAR_ALL_DATA}/$userId"
            Log.d(TAG, "清除所有数据 URL: $url")

            val request = Request.Builder()
                .url(url)
                .delete()
                .build()

            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""

            Log.d(TAG, "清除所有数据响应码: ${response.code}")

            if (response.isSuccessful) {
                val data = gson.fromJson(body, ClearDataResponse::class.java)
                Result.success(data)
            } else {
                Result.failure(Exception("清除所有数据失败: HTTP ${response.code}"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "清除所有数据异常", e)
            Result.failure(e)
        }
    }
}