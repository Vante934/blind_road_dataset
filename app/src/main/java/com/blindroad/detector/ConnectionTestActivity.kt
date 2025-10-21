package com.blindroad.detector

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class ConnectionTestActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "ConnectionTest"
        private const val REQUEST_CODE_PERMISSIONS = 100
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.INTERNET,
            Manifest.permission.ACCESS_NETWORK_STATE,
            Manifest.permission.CAMERA
        )
    }
    
    // UI组件
    private lateinit var serverUrlInput: EditText
    private lateinit var connectButton: Button
    private lateinit var testDetectionButton: Button
    private lateinit var statusText: TextView
    private lateinit var logText: TextView
    private lateinit var progressBar: ProgressBar
    
    // 网络客户端
    private lateinit var httpClient: OkHttpClient
    
    // 服务器配置
    private var serverUrl = "http://10.82.209.144:8080"
    private var isConnected = false
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_connection_test)
        
        initViews()
        initHttpClient()
        setupListeners()
        
        // 检查权限
        if (allPermissionsGranted()) {
            setupConnection()
        } else {
            requestPermissions()
        }
    }
    
    private fun initViews() {
        serverUrlInput = findViewById(R.id.server_url_input)
        connectButton = findViewById(R.id.connect_button)
        testDetectionButton = findViewById(R.id.test_detection_button)
        statusText = findViewById(R.id.status_text)
        logText = findViewById(R.id.log_text)
        progressBar = findViewById(R.id.progress_bar)
        
        // 设置默认服务器地址
        serverUrlInput.setText(serverUrl)
        
        // 初始状态
        testDetectionButton.isEnabled = false
        progressBar.visibility = View.GONE
    }
    
    private fun initHttpClient() {
        httpClient = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()
    }
    
    private fun setupListeners() {
        connectButton.setOnClickListener {
            connectToServer()
        }
        
        testDetectionButton.setOnClickListener {
            testDetection()
        }
    }
    
    private fun setupConnection() {
        updateStatus("准备连接服务器...")
        addLog("应用已启动，准备连接服务器")
    }
    
    private fun connectToServer() {
        serverUrl = serverUrlInput.text.toString().trim()
        if (serverUrl.isEmpty()) {
            showToast("请输入服务器地址")
            return
        }
        
        if (!serverUrl.startsWith("http://") && !serverUrl.startsWith("https://")) {
            serverUrl = "http://$serverUrl"
        }
        
        updateStatus("正在连接服务器...")
        progressBar.visibility = View.VISIBLE
        connectButton.isEnabled = false
        
        lifecycleScope.launch {
            try {
                val success = withContext(Dispatchers.IO) {
                    testServerConnection()
                }
                
                withContext(Dispatchers.Main) {
                    if (success) {
                        isConnected = true
                        updateStatus("✅ 服务器连接成功")
                        addLog("服务器连接成功: $serverUrl")
                        testDetectionButton.isEnabled = true
                        connectButton.text = "重新连接"
                    } else {
                        updateStatus("❌ 服务器连接失败")
                        addLog("服务器连接失败: $serverUrl")
                        testDetectionButton.isEnabled = false
                    }
                    progressBar.visibility = View.GONE
                    connectButton.isEnabled = true
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    updateStatus("❌ 连接异常: ${e.message}")
                    addLog("连接异常: ${e.message}")
                    progressBar.visibility = View.GONE
                    connectButton.isEnabled = true
                }
            }
        }
    }
    
    private fun testServerConnection(): Boolean {
        return try {
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
    
    private fun testDetection() {
        if (!isConnected) {
            showToast("请先连接服务器")
            return
        }
        
        updateStatus("正在测试检测功能...")
        progressBar.visibility = View.VISIBLE
        testDetectionButton.isEnabled = false
        
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    sendTestDetectionData()
                }
                
                withContext(Dispatchers.Main) {
                    if (result) {
                        updateStatus("✅ 检测测试成功")
                        addLog("检测数据发送成功")
                    } else {
                        updateStatus("❌ 检测测试失败")
                        addLog("检测数据发送失败")
                    }
                    progressBar.visibility = View.GONE
                    testDetectionButton.isEnabled = true
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    updateStatus("❌ 检测测试异常: ${e.message}")
                    addLog("检测测试异常: ${e.message}")
                    progressBar.visibility = View.GONE
                    testDetectionButton.isEnabled = true
                }
            }
        }
    }
    
    private fun sendTestDetectionData(): Boolean {
        return try {
            val testData = JSONObject().apply {
                put("device_id", "android_test_device")
                put("timestamp", System.currentTimeMillis())
                put("detections", JSONObject().apply {
                    put("test_detection", true)
                    put("confidence", 0.95)
                    put("class_name", "test_obstacle")
                })
                put("model_version", "1.0.0")
            }
            
            val requestBody = testData.toString()
                .toRequestBody("application/json".toMediaType())
            
            val request = Request.Builder()
                .url("$serverUrl/detection_data")
                .post(requestBody)
                .build()
            
            val response = httpClient.newCall(request).execute()
            val isSuccess = response.isSuccessful
            
            if (isSuccess) {
                val responseBody = response.body?.string()
                addLog("服务器响应: $responseBody")
            }
            
            response.close()
            isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "Test detection failed", e)
            false
        }
    }
    
    private fun updateStatus(message: String) {
        statusText.text = message
        Log.d(TAG, message)
    }
    
    private fun addLog(message: String) {
        val timestamp = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault())
            .format(java.util.Date())
        val logMessage = "[$timestamp] $message\n"
        
        runOnUiThread {
            logText.append(logMessage)
            // 自动滚动到底部
            val scrollView = findViewById<ScrollView>(R.id.log_scroll_view)
            scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
        }
    }
    
    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
    
    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
        }
    }
    
    private fun requestPermissions() {
        ActivityCompat.requestPermissions(
            this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
        )
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                setupConnection()
            } else {
                showToast("需要网络和相机权限才能使用此应用")
                finish()
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        httpClient.dispatcher.executorService.shutdown()
    }
}

