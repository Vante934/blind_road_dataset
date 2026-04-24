package com.blindroad.detector

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.common.util.concurrent.ListenableFuture
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "BlindRoadDetector"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_EXTERNAL_STORAGE
        )
    }
    
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    
    // UI 组件
    private lateinit var previewView: PreviewView
    private lateinit var statusTextView: TextView
    private lateinit var detectionTextView: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var settingsButton: Button
    
    // 检测相关
    private var isDetecting = false
    private lateinit var detectionManager: DetectionManager
    private lateinit var voiceManager: VoiceManager
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // 初始化UI
        initUI()
        
        // 初始化检测管理器
        detectionManager = DetectionManager(this)
        voiceManager = VoiceManager(this)
        
        // 请求权限
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
        
        cameraExecutor = Executors.newSingleThreadExecutor()
    }
    
    private fun initUI() {
        previewView = findViewById(R.id.preview_view)
        statusTextView = findViewById(R.id.status_text)
        detectionTextView = findViewById(R.id.detection_text)
        startButton = findViewById(R.id.start_button)
        stopButton = findViewById(R.id.stop_button)
        settingsButton = findViewById(R.id.settings_button)
        
        startButton.setOnClickListener {
            startDetection()
        }
        
        stopButton.setOnClickListener {
            stopDetection()
        }
        
        settingsButton.setOnClickListener {
            openSettings()
        }
        
        // 初始状态
        stopButton.isEnabled = false
        statusTextView.text = "准备就绪，点击开始检测"
    }
    
    private fun startDetection() {
        if (!isDetecting) {
            isDetecting = true
            startButton.isEnabled = false
            stopButton.isEnabled = true
            statusTextView.text = "检测中..."
            
            // 启动检测
            detectionManager.startDetection()
            voiceManager.speak("开始检测障碍物")
            
            Toast.makeText(this, "检测已开始", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun stopDetection() {
        if (isDetecting) {
            isDetecting = false
            startButton.isEnabled = true
            stopButton.isEnabled = false
            statusTextView.text = "检测已停止"
            
            // 停止检测
            detectionManager.stopDetection()
            voiceManager.speak("检测已停止")
            
            Toast.makeText(this, "检测已停止", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun openSettings() {
        // 打开设置界面
        val intent = android.content.Intent(this, SettingsActivity::class.java)
        startActivity(intent)
    }
    
    private fun startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, DetectionAnalyzer())
                }
            
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e(TAG, "相机绑定失败", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "需要相机权限才能运行", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        detectionManager.cleanup()
        voiceManager.cleanup()
    }
    
    // 图像分析器
    inner class DetectionAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(image: ImageProxy) {
            if (isDetecting) {
                // 执行检测
                val detections = detectionManager.detectObjects(image)
                
                // 更新UI
                runOnUiThread {
                    updateDetectionUI(detections)
                }
                
                // 语音播报
                if (detections.isNotEmpty()) {
                    val nearestDetection = detections.minByOrNull { it.distance }
                    nearestDetection?.let { detection ->
                        voiceManager.speak(detection.getVoiceMessage())
                    }
                }
            }
            
            image.close()
        }
        
        private fun updateDetectionUI(detections: List<DetectionResult>) {
            if (detections.isEmpty()) {
                detectionTextView.text = "未检测到障碍物"
                return
            }
            
            val nearest = detections.minByOrNull { it.distance }
            nearest?.let { detection ->
                val text = "检测到: ${detection.label}\n" +
                          "距离: ${String.format("%.1f", detection.distance)}米\n" +
                          "方向: ${detection.direction}\n" +
                          "置信度: ${String.format("%.1f", detection.confidence)}"
                detectionTextView.text = text
            }
        }
    }
} 