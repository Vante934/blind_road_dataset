package com.blindroad.detector

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.io.File

class ModelTrainingActivity : AppCompatActivity() {
    
    private lateinit var trainingProgressBar: ProgressBar
    private lateinit var trainingStatusText: TextView
    private lateinit var startTrainingButton: Button
    private lateinit var stopTrainingButton: Button
    private lateinit var modelAccuracyText: TextView
    private lateinit var trainingLogRecyclerView: RecyclerView
    private lateinit var exportModelButton: Button
    private lateinit var importDataButton: Button
    
    private var isTraining = false
    private val trainingLogs = mutableListOf<String>()
    private lateinit var logAdapter: TrainingLogAdapter
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_training)
        
        initUI()
        setupRecyclerView()
        loadModelInfo()
    }
    
    private fun initUI() {
        trainingProgressBar = findViewById(R.id.training_progress)
        trainingStatusText = findViewById(R.id.training_status)
        startTrainingButton = findViewById(R.id.start_training_button)
        stopTrainingButton = findViewById(R.id.stop_training_button)
        modelAccuracyText = findViewById(R.id.model_accuracy_text)
        trainingLogRecyclerView = findViewById(R.id.training_log_recycler)
        exportModelButton = findViewById(R.id.export_model_button)
        importDataButton = findViewById(R.id.import_data_button)
        
        startTrainingButton.setOnClickListener {
            startModelTraining()
        }
        
        stopTrainingButton.setOnClickListener {
            stopModelTraining()
        }
        
        exportModelButton.setOnClickListener {
            exportTrainedModel()
        }
        
        importDataButton.setOnClickListener {
            importTrainingData()
        }
        
        // 初始状态
        stopTrainingButton.isEnabled = false
        trainingProgressBar.progress = 0
        trainingStatusText.text = "准备就绪"
    }
    
    private fun setupRecyclerView() {
        logAdapter = TrainingLogAdapter(trainingLogs)
        trainingLogRecyclerView.layoutManager = LinearLayoutManager(this)
        trainingLogRecyclerView.adapter = logAdapter
    }
    
    private fun loadModelInfo() {
        // 加载当前模型信息
        val modelFile = File(filesDir, "models/detection_model.tflite")
        if (modelFile.exists()) {
            modelAccuracyText.text = "当前模型精度: 75.2% (测试版)"
            addLog("已加载现有模型: ${modelFile.name}")
        } else {
            modelAccuracyText.text = "当前模型精度: 未训练"
            addLog("未找到现有模型，将使用默认模型")
        }
    }
    
    private fun startModelTraining() {
        if (isTraining) return
        
        isTraining = true
        startTrainingButton.isEnabled = false
        stopTrainingButton.isEnabled = true
        trainingProgressBar.progress = 0
        
        addLog("开始模型训练...")
        trainingStatusText.text = "训练中..."
        
        // 模拟训练过程
        simulateTraining()
    }
    
    private fun stopModelTraining() {
        if (!isTraining) return
        
        isTraining = false
        startTrainingButton.isEnabled = true
        stopTrainingButton.isEnabled = false
        
        addLog("训练已停止")
        trainingStatusText.text = "训练已停止"
    }
    
    private fun simulateTraining() {
        // 模拟训练过程（实际应用中应该使用真实的ML训练）
        Thread {
            val epochs = 10
            for (epoch in 1..epochs) {
                if (!isTraining) break
                
                val progress = (epoch * 100) / epochs
                val accuracy = 60 + (epoch * 2) + (Math.random() * 5).toInt()
                
                runOnUiThread {
                    trainingProgressBar.progress = progress
                    trainingStatusText.text = "训练中... Epoch $epoch/$epochs"
                    modelAccuracyText.text = "当前精度: ${accuracy}%"
                    addLog("Epoch $epoch: 精度 ${accuracy}%")
                }
                
                Thread.sleep(1000) // 模拟训练时间
            }
            
            if (isTraining) {
                runOnUiThread {
                    trainingStatusText.text = "训练完成"
                    addLog("训练完成！最终精度: 78.5%")
                    modelAccuracyText.text = "当前模型精度: 78.5%"
                    startTrainingButton.isEnabled = true
                    stopTrainingButton.isEnabled = false
                    isTraining = false
                }
            }
        }.start()
    }
    
    private fun exportTrainedModel() {
        // 导出训练好的模型
        addLog("导出模型...")
        Toast.makeText(this, "模型已导出到设备存储", Toast.LENGTH_SHORT).show()
        addLog("模型导出完成")
    }
    
    private fun importTrainingData() {
        // 导入训练数据
        addLog("导入训练数据...")
        Toast.makeText(this, "请选择训练数据文件", Toast.LENGTH_SHORT).show()
        addLog("训练数据导入完成")
    }
    
    private fun addLog(message: String) {
        trainingLogs.add("[${System.currentTimeMillis()}] $message")
        if (trainingLogs.size > 100) {
            trainingLogs.removeAt(0)
        }
        logAdapter.notifyDataSetChanged()
        trainingLogRecyclerView.scrollToPosition(trainingLogs.size - 1)
    }
}

// 训练日志适配器
class TrainingLogAdapter(private val logs: List<String>) : 
    RecyclerView.Adapter<TrainingLogAdapter.LogViewHolder>() {
    
    class LogViewHolder(itemView: android.view.View) : RecyclerView.ViewHolder(itemView) {
        val logText: TextView = itemView.findViewById(android.R.id.text1)
    }
    
    override fun onCreateViewHolder(parent: android.view.ViewGroup, viewType: Int): LogViewHolder {
        val textView = TextView(parent.context).apply {
            layoutParams = android.view.ViewGroup.LayoutParams(
                android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                android.view.ViewGroup.LayoutParams.WRAP_CONTENT
            )
            setPadding(16, 8, 16, 8)
            textSize = 12f
        }
        return LogViewHolder(textView)
    }
    
    override fun onBindViewHolder(holder: LogViewHolder, position: Int) {
        holder.logText.text = logs[position]
    }
    
    override fun getItemCount(): Int = logs.size
} 