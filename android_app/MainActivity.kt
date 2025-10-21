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
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.INTERNET
        )
    }
    
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    
    // UI组件
    private lateinit var previewView: androidx.camera.view.PreviewView
    private lateinit var statusText: TextView
    private lateinit var detectionText: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var collectButton: Button
    private lateinit var trainButton: Button
    
    // 功能模块
    private lateinit var detectionManager: DetectionManager
    private lateinit var dataCollector: DataCollector
    private lateinit var modelTrainer: ModelTrainer
    private lateinit var serverCommunicator: ServerCommunicator
    
    private var isDetecting = false
    private var isCollecting = false
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // 初始化UI组件
        initViews()
        
        // 初始化功能模块
        initModules()
        
        // 检查权限
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        
        // 设置按钮点击事件
        setupButtonListeners()
        
        cameraExecutor = Executors.newSingleThreadExecutor()
    }
    
    private fun initViews() {
        previewView = findViewById(R.id.preview_view)
        statusText = findViewById(R.id.status_text)
        detectionText = findViewById(R.id.detection_text)
        startButton = findViewById(R.id.start_button)
        stopButton = findViewById(R.id.stop_button)
        collectButton = findViewById(R.id.collect_button)
        trainButton = findViewById(R.id.train_button)
    }
    
    private fun initModules() {
        detectionManager = DetectionManager(this)
        dataCollector = DataCollector(this)
        modelTrainer = ModelTrainer(this)
        serverCommunicator = ServerCommunicator(this)
        
        // 设置回调
        detectionManager.setDetectionCallback { detections ->
            runOnUiThread {
                updateDetectionDisplay(detections)
            }
        }
        
        dataCollector.setCollectionCallback { data ->
            serverCommunicator.sendTrainingData(data)
        }
    }
    
    private fun setupButtonListeners() {
        startButton.setOnClickListener {
            startDetection()
        }
        
        stopButton.setOnClickListener {
            stopDetection()
        }
        
        collectButton.setOnClickListener {
            toggleDataCollection()
        }
        
        trainButton.setOnClickListener {
            startTraining()
        }
    }
    
    private fun startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
            
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalyzer())
                }
            
            try {
                cameraProvider.unbindAll()
                
                cameraProvider.bindToLifecycle(
                    this as LifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
                
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun startDetection() {
        if (!isDetecting) {
            isDetecting = true
            detectionManager.startDetection()
            updateStatus("检测已启动")
            startButton.isEnabled = false
            stopButton.isEnabled = true
        }
    }
    
    private fun stopDetection() {
        if (isDetecting) {
            isDetecting = false
            detectionManager.stopDetection()
            updateStatus("检测已停止")
            startButton.isEnabled = true
            stopButton.isEnabled = false
        }
    }
    
    private fun toggleDataCollection() {
        if (!isCollecting) {
            isCollecting = true
            dataCollector.startCollection()
            collectButton.text = "停止收集"
            updateStatus("数据收集已启动")
        } else {
            isCollecting = false
            dataCollector.stopCollection()
            collectButton.text = "开始收集"
            updateStatus("数据收集已停止")
        }
    }
    
    private fun startTraining() {
        trainButton.isEnabled = false
        updateStatus("开始训练...")
        
        modelTrainer.startTraining { success ->
            runOnUiThread {
                trainButton.isEnabled = true
                if (success) {
                    updateStatus("训练完成")
                    Toast.makeText(this, "模型训练完成", Toast.LENGTH_SHORT).show()
                } else {
                    updateStatus("训练失败")
                    Toast.makeText(this, "模型训练失败", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
    
    private fun updateStatus(message: String) {
        statusText.text = message
        Log.d(TAG, message)
    }
    
    private fun updateDetectionDisplay(detections: List<Detection>) {
        val detectionInfo = detections.joinToString("\n") { detection ->
            "${detection.className}: ${String.format("%.2f", detection.confidence)}"
        }
        detectionText.text = if (detectionInfo.isNotEmpty()) detectionInfo else "未检测到物体"
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
                Toast.makeText(this, "需要相机和存储权限", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        detectionManager.stopDetection()
        dataCollector.stopCollection()
    }
    
    // 图像分析器
    private inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        @androidx.camera.core.ExperimentalGetImage
        override fun analyze(image: ImageProxy) {
            if (isDetecting) {
                detectionManager.processImage(image)
            }
            
            if (isCollecting) {
                dataCollector.collectImage(image)
            }
            
            image.close()
        }
    }
} 