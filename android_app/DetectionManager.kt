package com.blindroad.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class DetectionManager(private val context: Context) {
    
    companion object {
        private const val TAG = "DetectionManager"
        private const val MODEL_PATH = "models/best.pt"
        private const val CONFIDENCE_THRESHOLD = 0.5f
    }
    
    private var detectionCallback: ((List<Detection>) -> Unit)? = null
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private var isDetecting = false
    
    // YOLO模型相关
    private var yoloModel: Any? = null // 实际应该是YOLO模型实例
    
    // 轨迹预测相关
    private val trajectoryPredictor = TrajectoryPredictor()
    private val trackedObjects = mutableMapOf<Int, TrackedObject>()
    
    fun setDetectionCallback(callback: (List<Detection>) -> Unit) {
        detectionCallback = callback
    }
    
    fun startDetection() {
        isDetecting = true
        loadModel()
        Log.d(TAG, "检测已启动")
    }
    
    fun stopDetection() {
        isDetecting = false
        Log.d(TAG, "检测已停止")
    }
    
    private fun loadModel() {
        executor.execute {
            try {
                // 这里应该加载YOLO模型
                // 暂时使用模拟模型
                yoloModel = "simulated_model"
                Log.d(TAG, "模型加载成功")
            } catch (e: Exception) {
                Log.e(TAG, "模型加载失败", e)
            }
        }
    }
    
    fun processImage(image: ImageProxy) {
        if (!isDetecting) return
        
        executor.execute {
            try {
                val bitmap = imageProxyToBitmap(image)
                val detections = detectObjects(bitmap)
                val predictions = predictTrajectories(detections)
                
                // 合并检测和预测结果
                val results = mergeResults(detections, predictions)
                
                // 回调结果
                detectionCallback?.invoke(results)
                
            } catch (e: Exception) {
                Log.e(TAG, "图像处理失败", e)
            }
        }
    }
    
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer: ByteBuffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
    
    private fun detectObjects(bitmap: Bitmap): List<Detection> {
        val detections = mutableListOf<Detection>()
        
        try {
            // 这里应该调用YOLO模型进行检测
            // 暂时使用模拟检测结果
            if (Math.random() > 0.7) {
                detections.add(
                    Detection(
                        id = System.currentTimeMillis().toInt(),
                        className = "obstacle",
                        confidence = 0.85f,
                        bbox = Rect(100, 100, 200, 200),
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            
            // 如果有真实模型，应该这样调用：
            // val results = yoloModel.detect(bitmap)
            // 然后解析结果
            
        } catch (e: Exception) {
            Log.e(TAG, "物体检测失败", e)
        }
        
        return detections
    }
    
    private fun predictTrajectories(detections: List<Detection>): List<TrajectoryPrediction> {
        val predictions = mutableListOf<TrajectoryPrediction>()
        
        for (detection in detections) {
            // 更新跟踪对象
            val trackedObject = trackedObjects.getOrPut(detection.id) {
                TrackedObject(detection.id, detection.className)
            }
            
            // 更新位置
            trackedObject.updatePosition(detection.bbox)
            
            // 预测轨迹
            val prediction = trajectoryPredictor.predictTrajectory(trackedObject)
            if (prediction != null) {
                predictions.add(prediction)
            }
        }
        
        // 清理过期的跟踪对象
        val currentTime = System.currentTimeMillis()
        trackedObjects.entries.removeIf { (_, obj) ->
            currentTime - obj.lastUpdateTime > 5000 // 5秒未更新则删除
        }
        
        return predictions
    }
    
    private fun mergeResults(detections: List<Detection>, predictions: List<TrajectoryPrediction>): List<Detection> {
        val results = detections.toMutableList()
        
        // 将预测结果转换为检测结果
        for (prediction in predictions) {
            results.add(
                Detection(
                    id = prediction.objectId,
                    className = "${prediction.className}_predicted",
                    confidence = prediction.confidence,
                    bbox = prediction.predictedBbox,
                    timestamp = prediction.timestamp,
                    isPrediction = true
                )
            )
        }
        
        return results
    }
    
    fun getModelInfo(): ModelInfo {
        return ModelInfo(
            modelPath = MODEL_PATH,
            isLoaded = yoloModel != null,
            confidenceThreshold = CONFIDENCE_THRESHOLD,
            trackedObjectsCount = trackedObjects.size
        )
    }
    
    fun updateModel(newModelPath: String): Boolean {
        return try {
            // 这里应该重新加载模型
            Log.d(TAG, "模型更新: $newModelPath")
            true
        } catch (e: Exception) {
            Log.e(TAG, "模型更新失败", e)
            false
        }
    }
}

// 数据类
data class Detection(
    val id: Int,
    val className: String,
    val confidence: Float,
    val bbox: Rect,
    val timestamp: Long,
    val isPrediction: Boolean = false
)

data class TrajectoryPrediction(
    val objectId: Int,
    val className: String,
    val predictedBbox: Rect,
    val confidence: Float,
    val timestamp: Long,
    val trajectory: List<Rect>
)

data class TrackedObject(
    val id: Int,
    val className: String,
    var currentBbox: Rect = Rect(),
    var lastUpdateTime: Long = System.currentTimeMillis(),
    var velocity: Pair<Float, Float> = Pair(0f, 0f)
) {
    fun updatePosition(newBbox: Rect) {
        // 计算速度
        if (currentBbox.width() > 0) {
            val deltaX = (newBbox.centerX() - currentBbox.centerX()).toFloat()
            val deltaY = (newBbox.centerY() - currentBbox.centerY()).toFloat()
            velocity = Pair(deltaX, deltaY)
        }
        
        currentBbox = newBbox
        lastUpdateTime = System.currentTimeMillis()
    }
}

data class ModelInfo(
    val modelPath: String,
    val isLoaded: Boolean,
    val confidenceThreshold: Float,
    val trackedObjectsCount: Int
)

// 轨迹预测器
class TrajectoryPredictor {
    
    fun predictTrajectory(trackedObject: TrackedObject): TrajectoryPrediction? {
        try {
            // 简单的线性预测
            val (vx, vy) = trackedObject.velocity
            val centerX = trackedObject.currentBbox.centerX()
            val centerY = trackedObject.currentBbox.centerY()
            
            // 预测未来5帧的位置
            val predictionSteps = 5
            val trajectory = mutableListOf<Rect>()
            
            for (i in 1..predictionSteps) {
                val predictedX = centerX + (vx * i).toInt()
                val predictedY = centerY + (vy * i).toInt()
                
                val predictedBbox = Rect(
                    predictedX - trackedObject.currentBbox.width() / 2,
                    predictedY - trackedObject.currentBbox.height() / 2,
                    predictedX + trackedObject.currentBbox.width() / 2,
                    predictedY + trackedObject.currentBbox.height() / 2
                )
                
                trajectory.add(predictedBbox)
            }
            
            return TrajectoryPrediction(
                objectId = trackedObject.id,
                className = trackedObject.className,
                predictedBbox = trajectory.last(),
                confidence = 0.7f, // 预测置信度
                timestamp = System.currentTimeMillis(),
                trajectory = trajectory
            )
            
        } catch (e: Exception) {
            Log.e("TrajectoryPredictor", "轨迹预测失败", e)
            return null
        }
    }
} 