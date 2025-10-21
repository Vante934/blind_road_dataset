package com.blindroad.detector

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class ServerCommunicator(private val context: Context) {
    
    companion object {
        private const val TAG = "ServerCommunicator"
    }
    
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    
    private var serverUrl = "http://10.82.209.144:8080"
    
    fun setServerUrl(url: String) {
        serverUrl = url.trimEnd('/')
    }
    
    suspend fun testConnection(): Boolean = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$serverUrl/status")
                .build()
            
            val response = httpClient.newCall(request).execute()
            val isSuccess = response.isSuccessful
            
            response.close()
            isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "Connection test failed", e)
            false
        }
    }
    
    suspend fun sendDetectionData(detectionData: DetectionData): Boolean = withContext(Dispatchers.IO) {
        try {
            val jsonData = JSONObject().apply {
                put("device_id", detectionData.deviceId)
                put("timestamp", detectionData.timestamp)
                put("detections", detectionData.detections)
                put("model_version", detectionData.modelVersion)
            }
            
            val requestBody = jsonData.toString()
                .toRequestBody("application/json".toMediaType())
            
            val request = Request.Builder()
                .url("$serverUrl/detection_data")
                .post(requestBody)
                .build()
            
            val response = httpClient.newCall(request).execute()
            val isSuccess = response.isSuccessful
            
            if (isSuccess) {
                val responseBody = response.body?.string()
                Log.d(TAG, "Detection data sent successfully: $responseBody")
            } else {
                Log.e(TAG, "Failed to send detection data: ${response.code}")
            }
            
            response.close()
            isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "Send detection data failed", e)
            false
        }
    }
    
    suspend fun sendTrainingData(trainingData: TrainingData): Boolean = withContext(Dispatchers.IO) {
        try {
            val jsonData = JSONObject().apply {
                put("device_id", trainingData.deviceId)
                put("data", trainingData.data)
                put("model_version", trainingData.modelVersion)
            }
            
            val requestBody = jsonData.toString()
                .toRequestBody("application/json".toMediaType())
            
            val request = Request.Builder()
                .url("$serverUrl/training_data")
                .post(requestBody)
                .build()
            
            val response = httpClient.newCall(request).execute()
            val isSuccess = response.isSuccessful
            
            if (isSuccess) {
                Log.d(TAG, "Training data sent successfully")
            } else {
                Log.e(TAG, "Failed to send training data: ${response.code}")
            }
            
            response.close()
            isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "Send training data failed", e)
            false
        }
    }
    
    suspend fun checkModelUpdate(deviceId: String): ModelUpdateInfo? = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$serverUrl/model_status?device_id=$deviceId")
                .build()
            
            val response = httpClient.newCall(request).execute()
            
            if (response.isSuccessful) {
                val responseBody = response.body?.string()
                val jsonResponse = JSONObject(responseBody)
                
                val updateAvailable = jsonResponse.optBoolean("update_available", false)
                if (updateAvailable) {
                    ModelUpdateInfo(
                        version = jsonResponse.optString("version", ""),
                        downloadUrl = jsonResponse.optString("download_url", "")
                    )
                } else {
                    null
                }
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Check model update failed", e)
            null
        }
    }
    
    suspend fun downloadModel(url: String): ByteArray? = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url(url)
                .build()
            
            val response = httpClient.newCall(request).execute()
            
            if (response.isSuccessful) {
                response.body?.bytes()
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Download model failed", e)
            null
        }
    }
    
    fun cleanup() {
        httpClient.dispatcher.executorService.shutdown()
    }
}

// 数据类
data class DetectionData(
    val deviceId: String,
    val timestamp: Long,
    val detections: List<Detection>,
    val modelVersion: String
)

data class Detection(
    val bbox: List<Int>,
    val className: String,
    val confidence: Float
)

data class TrainingData(
    val deviceId: String,
    val data: List<TrainingItem>,
    val modelVersion: String
)

data class TrainingItem(
    val timestamp: Long,
    val frameId: Int,
    val detections: List<Detection>,
    val imagePath: String
)

data class ModelUpdateInfo(
    val version: String,
    val downloadUrl: String
)