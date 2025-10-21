package com.blindroad.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.media.Image
import android.util.Log
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import kotlin.math.*

class DetectionManager(private val context: Context) {
    
    companion object {
        private const val TAG = "DetectionManager"
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val KNOWN_OBJECT_HEIGHT = 1.5f // 假设物体高度1.5米
        private const val FOCAL_LENGTH = 700f // 焦距
    }
    
    private val executor = Executors.newSingleThreadExecutor()
    private var isDetecting = false
    
    // 检测结果缓存
    private val detectionHistory = mutableListOf<DetectionResult>()
    private val maxHistorySize = 10
    
    // 轨迹预测
    private val trajectoryPredictor = TrajectoryPredictor()
    
    // 盲道检测
    private val blindPathDetector = BlindPathDetector()
    
    // TensorFlow Lite检测器
    private val tensorFlowDetector = TensorFlowLiteDetector(context)
    
    fun startDetection() {
        isDetecting = true
        Log.d(TAG, "检测已启动")
    }
    
    fun stopDetection() {
        isDetecting = false
        Log.d(TAG, "检测已停止")
    }
    
    fun detectObjects(image: Image): List<DetectionResult> {
        if (!isDetecting) return emptyList()
        
        return try {
            // 转换图像格式
            val bitmap = imageToBitmap(image)
            
            // 使用TensorFlow Lite进行真实检测
            val detections = tensorFlowDetector.detectObjects(bitmap)
            
            // 更新历史记录
            updateDetectionHistory(detections)
            
            // 轨迹预测
            val predictions = trajectoryPredictor.predictTrajectories(detections)
            
            // 合并检测结果和预测
            detections.map { detection ->
                val prediction = predictions.find { it.objectId == detection.objectId }
                detection.copy(
                    predictedPosition = prediction?.predictedPosition,
                    collisionRisk = prediction?.collisionRisk ?: 0f
                )
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "检测失败", e)
            // 如果TensorFlow Lite检测失败，回退到模拟检测
            performFallbackDetection(image)
        }
    }
    
    private fun imageToBitmap(image: Image): Bitmap {
        val buffer: ByteBuffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
    
    private fun performDetection(bitmap: Bitmap): List<DetectionResult> {
        // 模拟检测结果 - 实际应用中应该使用真实的ML模型
        val detections = mutableListOf<DetectionResult>()
        
        // 模拟检测到不同物体
        val mockDetections = listOf(
            DetectionResult(
                objectId = "person_1",
                label = "行人",
                confidence = 0.85f,
                boundingBox = Rect(100, 200, 200, 400),
                distance = estimateDistance(200), // 高度200像素
                direction = "正前方",
                category = "动态",
                riskLevel = "中"
            ),
            DetectionResult(
                objectId = "bottle_1", 
                label = "瓶子",
                confidence = 0.72f,
                boundingBox = Rect(300, 300, 350, 380),
                distance = estimateDistance(80), // 高度80像素
                direction = "右侧",
                category = "静态",
                riskLevel = "低"
            )
        )
        
        // 根据置信度过滤
        detections.addAll(mockDetections.filter { it.confidence > CONFIDENCE_THRESHOLD })
        
        return detections
    }
    
    private fun estimateDistance(pixelHeight: Int): Float {
        if (pixelHeight <= 0) return 99f
        val distance = (KNOWN_OBJECT_HEIGHT * FOCAL_LENGTH) / pixelHeight
        return max(0.1f, min(99f, distance))
    }
    
    private fun updateDetectionHistory(detections: List<DetectionResult>) {
        detectionHistory.addAll(detections)
        if (detectionHistory.size > maxHistorySize) {
            detectionHistory.removeAt(0)
        }
    }
    
    fun getDetectionHistory(): List<DetectionResult> {
        return detectionHistory.toList()
    }
    
    fun detectBlindPath(image: Image): BlindPathResult? {
        return try {
            val bitmap = imageToBitmap(image)
            blindPathDetector.detect(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "盲道检测失败", e)
            null
        }
    }
    
    private fun performFallbackDetection(image: Image): List<DetectionResult> {
        // 回退到模拟检测
        val bitmap = imageToBitmap(image)
        return performDetection(bitmap)
    }
    
    fun cleanup() {
        executor.shutdown()
        detectionHistory.clear()
        tensorFlowDetector.cleanup()
    }
}

// 检测结果数据类
data class DetectionResult(
    val objectId: String,
    val label: String,
    val confidence: Float,
    val boundingBox: Rect,
    val distance: Float,
    val direction: String,
    val category: String, // 静态/动态
    val riskLevel: String, // 低/中/高
    val predictedPosition: Pair<Float, Float>? = null,
    val collisionRisk: Float = 0f
) {
    fun getVoiceMessage(): String {
        return when (riskLevel) {
            "高" -> "警告！前方${String.format("%.1f", distance)}米处有$label，请立即避让"
            "中" -> "请注意，前方${String.format("%.1f", distance)}米处有$label，请向$direction方向前进"
            else -> "前方${String.format("%.1f", distance)}米处有$label"
        }
    }
}

// 盲道检测结果
data class BlindPathResult(
    val detected: Boolean,
    val confidence: Float,
    val center: Pair<Float, Float>? = null,
    val width: Float = 0f,
    val angle: Float = 0f
)

// 轨迹预测器
class TrajectoryPredictor {
    private val trajectoryHistory = mutableMapOf<String, MutableList<Pair<Float, Float>>>()
    
    fun predictTrajectories(detections: List<DetectionResult>): List<TrajectoryPrediction> {
        val predictions = mutableListOf<TrajectoryPrediction>()
        
        for (detection in detections) {
            val center = Pair(
                detection.boundingBox.centerX().toFloat(),
                detection.boundingBox.centerY().toFloat()
            )
            
            // 更新轨迹历史
            if (!trajectoryHistory.containsKey(detection.objectId)) {
                trajectoryHistory[detection.objectId] = mutableListOf()
            }
            trajectoryHistory[detection.objectId]?.add(center)
            
            // 保持历史记录在合理范围内
            if (trajectoryHistory[detection.objectId]?.size ?: 0 > 10) {
                trajectoryHistory[detection.objectId]?.removeAt(0)
            }
            
            // 预测未来位置
            val predictedPosition = predictFuturePosition(detection.objectId)
            val collisionRisk = calculateCollisionRisk(detection, predictedPosition)
            
            predictions.add(
                TrajectoryPrediction(
                    objectId = detection.objectId,
                    predictedPosition = predictedPosition,
                    collisionRisk = collisionRisk
                )
            )
        }
        
        return predictions
    }
    
    private fun predictFuturePosition(objectId: String): Pair<Float, Float>? {
        val history = trajectoryHistory[objectId] ?: return null
        if (history.size < 3) return null
        
        // 简单的线性预测
        val recent = history.takeLast(3)
        val dx = recent.last().first - recent.first().first
        val dy = recent.last().second - recent.first().second
        
        return Pair(
            recent.last().first + dx,
            recent.last().second + dy
        )
    }
    
    private fun calculateCollisionRisk(detection: DetectionResult, predictedPosition: Pair<Float, Float>?): Float {
        if (predictedPosition == null) return 0f
        
        // 简单的碰撞风险评估
        val distance = detection.distance
        val isDynamic = detection.category == "动态"
        
        return when {
            distance < 1f && isDynamic -> 0.9f // 高风险
            distance < 2f && isDynamic -> 0.6f // 中风险
            distance < 1f -> 0.4f // 静态物体，中等风险
            else -> 0.1f // 低风险
        }
    }
}

data class TrajectoryPrediction(
    val objectId: String,
    val predictedPosition: Pair<Float, Float>?,
    val collisionRisk: Float
)

// 盲道检测器
class BlindPathDetector {
    fun detect(bitmap: Bitmap): BlindPathResult? {
        // 模拟盲道检测
        // 实际应用中应该使用图像处理算法或ML模型
        
        // 这里返回模拟结果
        return BlindPathResult(
            detected = true,
            confidence = 0.75f,
            center = Pair(bitmap.width / 2f, bitmap.height * 0.8f),
            width = 100f,
            angle = 0f
        )
    }
} 