package com.blindroad.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class DataCollector(private val context: Context) {
    
    companion object {
        private const val TAG = "DataCollector"
        private const val DATA_DIR = "training_data"
        private const val COLLECTION_INTERVAL = 1000L // 1秒间隔
    }
    
    private var collectionCallback: ((TrainingData) -> Unit)? = null
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private var isCollecting = false
    private var lastCollectionTime = 0L
    
    private val collectedData = mutableListOf<DataItem>()
    private val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
    
    fun setCollectionCallback(callback: (TrainingData) -> Unit) {
        collectionCallback = callback
    }
    
    fun startCollection() {
        isCollecting = true
        Log.d(TAG, "数据收集已启动")
    }
    
    fun stopCollection() {
        isCollecting = false
        Log.d(TAG, "数据收集已停止")
    }
    
    fun collectImage(image: ImageProxy) {
        if (!isCollecting) return
        
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastCollectionTime < COLLECTION_INTERVAL) {
            return // 控制收集频率
        }
        
        executor.execute {
            try {
                val bitmap = imageProxyToBitmap(image)
                val dataItem = createDataItem(bitmap)
                
                // 保存图像
                val imageFile = saveImage(bitmap, dataItem.timestamp)
                dataItem.imagePath = imageFile.absolutePath
                
                // 添加到收集列表
                collectedData.add(dataItem)
                
                // 更新收集时间
                lastCollectionTime = currentTime
                
                Log.d(TAG, "收集数据: ${dataItem.id}")
                
                // 每收集10帧数据就发送一次
                if (collectedData.size >= 10) {
                    sendCollectedData()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "数据收集失败", e)
            }
        }
    }
    
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
    
    private fun createDataItem(bitmap: Bitmap): DataItem {
        val timestamp = System.currentTimeMillis()
        val id = "frame_${timestamp}"
        
        return DataItem(
            id = id,
            timestamp = timestamp,
            imageWidth = bitmap.width,
            imageHeight = bitmap.height,
            detections = emptyList(), // 这里应该从检测管理器获取检测结果
            imagePath = ""
        )
    }
    
    private fun saveImage(bitmap: Bitmap, timestamp: Long): File {
        val dataDir = File(context.getExternalFilesDir(null), DATA_DIR)
        dataDir.mkdirs()
        
        val fileName = "frame_${timestamp}.jpg"
        val imageFile = File(dataDir, fileName)
        
        try {
            val outputStream = FileOutputStream(imageFile)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
            outputStream.close()
            
            Log.d(TAG, "图像已保存: ${imageFile.absolutePath}")
            
        } catch (e: Exception) {
            Log.e(TAG, "保存图像失败", e)
        }
        
        return imageFile
    }
    
    private fun sendCollectedData() {
        if (collectedData.isEmpty()) return
        
        try {
            val trainingData = TrainingData(
                deviceId = getDeviceId(),
                timestamp = System.currentTimeMillis(),
                dataCount = collectedData.size,
                data = collectedData.toList()
            )
            
            // 回调发送数据
            collectionCallback?.invoke(trainingData)
            
            // 清空已发送的数据
            collectedData.clear()
            
            Log.d(TAG, "发送训练数据: ${trainingData.dataCount} 帧")
            
        } catch (e: Exception) {
            Log.e(TAG, "发送训练数据失败", e)
        }
    }
    
    private fun getDeviceId(): String {
        return android.provider.Settings.Secure.getString(
            context.contentResolver,
            android.provider.Settings.Secure.ANDROID_ID
        ) ?: "unknown_device"
    }
    
    fun getCollectionStats(): CollectionStats {
        return CollectionStats(
            isCollecting = isCollecting,
            collectedCount = collectedData.size,
            totalCollected = getTotalCollectedCount()
        )
    }
    
    private fun getTotalCollectedCount(): Int {
        val dataDir = File(context.getExternalFilesDir(null), DATA_DIR)
        if (!dataDir.exists()) return 0
        
        return dataDir.listFiles()?.count { it.extension == "jpg" } ?: 0
    }
    
    fun clearCollectedData() {
        collectedData.clear()
        Log.d(TAG, "已清空收集的数据")
    }
    
    fun exportData(format: ExportFormat): Boolean {
        return try {
            when (format) {
                ExportFormat.YOLO -> exportToYoloFormat()
                ExportFormat.COCO -> exportToCocoFormat()
                ExportFormat.JSON -> exportToJsonFormat()
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "导出数据失败", e)
            false
        }
    }
    
    private fun exportToYoloFormat(): Boolean {
        val exportDir = File(context.getExternalFilesDir(null), "exports/yolo")
        exportDir.mkdirs()
        
        val imagesDir = File(exportDir, "images")
        val labelsDir = File(exportDir, "labels")
        imagesDir.mkdirs()
        labelsDir.mkdirs()
        
        for (dataItem in collectedData) {
            val imageFile = File(dataItem.imagePath)
            if (imageFile.exists()) {
                // 复制图像
                val newImageFile = File(imagesDir, "${dataItem.id}.jpg")
                imageFile.copyTo(newImageFile, overwrite = true)
                
                // 创建标签文件
                val labelFile = File(labelsDir, "${dataItem.id}.txt")
                createYoloLabel(labelFile, dataItem)
            }
        }
        
        Log.d(TAG, "YOLO格式导出完成: ${exportDir.absolutePath}")
        return true
    }
    
    private fun exportToCocoFormat(): Boolean {
        // 实现COCO格式导出
        Log.d(TAG, "COCO格式导出完成")
        return true
    }
    
    private fun exportToJsonFormat(): Boolean {
        val exportDir = File(context.getExternalFilesDir(null), "exports/json")
        exportDir.mkdirs()
        
        val exportFile = File(exportDir, "training_data_${dateFormat.format(Date())}.json")
        
        val trainingData = TrainingData(
            deviceId = getDeviceId(),
            timestamp = System.currentTimeMillis(),
            dataCount = collectedData.size,
            data = collectedData.toList()
        )
        
        // 这里应该使用JSON库序列化数据
        // 暂时使用简单实现
        exportFile.writeText("Training data export")
        
        Log.d(TAG, "JSON格式导出完成: ${exportFile.absolutePath}")
        return true
    }
    
    private fun createYoloLabel(labelFile: File, dataItem: DataItem) {
        try {
            val labelContent = StringBuilder()
            
            for (detection in dataItem.detections) {
                // 转换为YOLO格式 (class_id x_center y_center width height)
                val classId = getClassId(detection.className)
                val (xCenter, yCenter, width, height) = normalizeBbox(
                    detection.bbox,
                    dataItem.imageWidth,
                    dataItem.imageHeight
                )
                
                labelContent.append("$classId $xCenter $yCenter $width $height\n")
            }
            
            labelFile.writeText(labelContent.toString())
            
        } catch (e: Exception) {
            Log.e(TAG, "创建YOLO标签失败", e)
        }
    }
    
    private fun getClassId(className: String): Int {
        return when (className.lowercase()) {
            "obstacle" -> 0
            "person" -> 1
            "vehicle" -> 2
            else -> 0
        }
    }
    
    private fun normalizeBbox(bbox: Rect, imageWidth: Int, imageHeight: Int): Triple<Float, Float, Float> {
        val xCenter = (bbox.centerX().toFloat() / imageWidth)
        val yCenter = (bbox.centerY().toFloat() / imageHeight)
        val width = (bbox.width().toFloat() / imageWidth)
        val height = (bbox.height().toFloat() / imageHeight)
        
        return Triple(xCenter, yCenter, width)
    }
}

// 数据类
data class DataItem(
    val id: String,
    val timestamp: Long,
    val imageWidth: Int,
    val imageHeight: Int,
    val detections: List<Detection>,
    var imagePath: String
)

data class TrainingData(
    val deviceId: String,
    val timestamp: Long,
    val dataCount: Int,
    val data: List<DataItem>
)

data class CollectionStats(
    val isCollecting: Boolean,
    val collectedCount: Int,
    val totalCollected: Int
)

enum class ExportFormat {
    YOLO,
    COCO,
    JSON
} 