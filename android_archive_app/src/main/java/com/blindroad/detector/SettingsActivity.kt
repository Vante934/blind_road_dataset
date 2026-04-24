package com.blindroad.detector

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat

class SettingsActivity : AppCompatActivity() {
    
    private lateinit var confidenceSeekBar: SeekBar
    private lateinit var confidenceText: TextView
    private lateinit var voiceSwitch: SwitchCompat
    private lateinit var trajectorySwitch: SwitchCompat
    private lateinit var blindPathSwitch: SwitchCompat
    private lateinit var speechRateSeekBar: SeekBar
    private lateinit var speechRateText: TextView
    private lateinit var modelTrainingButton: Button
    private lateinit var resetButton: Button
    private lateinit var saveButton: Button
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        
        initUI()
        loadSettings()
        setupListeners()
    }
    
    private fun initUI() {
        confidenceSeekBar = findViewById(R.id.confidence_seekbar)
        confidenceText = findViewById(R.id.confidence_text)
        voiceSwitch = findViewById(R.id.voice_switch)
        trajectorySwitch = findViewById(R.id.trajectory_switch)
        blindPathSwitch = findViewById(R.id.blind_path_switch)
        speechRateSeekBar = findViewById(R.id.speech_rate_seekbar)
        speechRateText = findViewById(R.id.speech_rate_text)
        modelTrainingButton = findViewById(R.id.model_training_button)
        resetButton = findViewById(R.id.reset_button)
        saveButton = findViewById(R.id.save_button)
    }
    
    private fun loadSettings() {
        val prefs = getSharedPreferences("BlindRoadDetector", MODE_PRIVATE)
        
        // 加载置信度阈值
        val confidence = prefs.getFloat("confidence_threshold", 0.5f)
        confidenceSeekBar.progress = (confidence * 100).toInt()
        confidenceText.text = "置信度阈值: ${String.format("%.2f", confidence)}"
        
        // 加载开关状态
        voiceSwitch.isChecked = prefs.getBoolean("voice_enabled", true)
        trajectorySwitch.isChecked = prefs.getBoolean("trajectory_enabled", true)
        blindPathSwitch.isChecked = prefs.getBoolean("blind_path_enabled", true)
        
        // 加载语音速率
        val speechRate = prefs.getFloat("speech_rate", 1.0f)
        speechRateSeekBar.progress = ((speechRate - 0.5f) * 100).toInt()
        speechRateText.text = "语音速率: ${String.format("%.1f", speechRate)}x"
    }
    
    private fun setupListeners() {
        confidenceSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val confidence = progress / 100f
                confidenceText.text = "置信度阈值: ${String.format("%.2f", confidence)}"
            }
            
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        
        speechRateSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val rate = 0.5f + (progress / 100f)
                speechRateText.text = "语音速率: ${String.format("%.1f", rate)}x"
            }
            
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        
        modelTrainingButton.setOnClickListener {
            openModelTraining()
        }
        
        resetButton.setOnClickListener {
            resetToDefaults()
        }
        
        saveButton.setOnClickListener {
            saveSettings()
            finish()
        }
    }
    
    private fun openModelTraining() {
        val intent = android.content.Intent(this, ModelTrainingActivity::class.java)
        startActivity(intent)
    }
    
    private fun resetToDefaults() {
        confidenceSeekBar.progress = 50 // 0.5
        confidenceText.text = "置信度阈值: 0.50"
        
        voiceSwitch.isChecked = true
        trajectorySwitch.isChecked = true
        blindPathSwitch.isChecked = true
        
        speechRateSeekBar.progress = 50 // 1.0
        speechRateText.text = "语音速率: 1.0x"
        
        Toast.makeText(this, "已重置为默认设置", Toast.LENGTH_SHORT).show()
    }
    
    private fun saveSettings() {
        val prefs = getSharedPreferences("BlindRoadDetector", MODE_PRIVATE)
        val editor = prefs.edit()
        
        // 保存置信度阈值
        val confidence = confidenceSeekBar.progress / 100f
        editor.putFloat("confidence_threshold", confidence)
        
        // 保存开关状态
        editor.putBoolean("voice_enabled", voiceSwitch.isChecked)
        editor.putBoolean("trajectory_enabled", trajectorySwitch.isChecked)
        editor.putBoolean("blind_path_enabled", blindPathSwitch.isChecked)
        
        // 保存语音速率
        val speechRate = 0.5f + (speechRateSeekBar.progress / 100f)
        editor.putFloat("speech_rate", speechRate)
        
        editor.apply()
        
        Toast.makeText(this, "设置已保存", Toast.LENGTH_SHORT).show()
    }
} 