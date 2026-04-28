package com.yourcompany.blindnavigation

import kotlinx.coroutines.Job
import com.yourcompany.blindnavigation.utils.AudioRecorder
import com.yourcompany.blindnavigation.network.WarningMessage
import com.yourcompany.blindnavigation.network.PerceptionWebSocketService
import com.yourcompany.blindnavigation.ui.components.UserInfoDialog
import com.yourcompany.blindnavigation.ui.components.EditUserInfoDialog
import com.yourcompany.blindnavigation.ui.components.StorageDataDialog
import com.yourcompany.blindnavigation.utils.StorageManager
import com.yourcompany.blindnavigation.utils.BaiduTTSManager
import kotlinx.coroutines.withContext
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Geocoder
import android.location.Location
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.focusable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.MyLocation
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material.icons.outlined.EmojiObjects
import androidx.compose.material.icons.outlined.Favorite
import androidx.compose.material.icons.outlined.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.net.toUri
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.yourcompany.blindnavigation.models.*
import com.yourcompany.blindnavigation.network.GPSData
import com.yourcompany.blindnavigation.network.NetworkService
import com.yourcompany.blindnavigation.network.RouteResponse
import com.yourcompany.blindnavigation.network.WebSocketService
import com.yourcompany.blindnavigation.ui.theme.*
import com.yourcompany.blindnavigation.utils.LocalAuthManager
import com.yourcompany.blindnavigation.utils.VibrationType
import com.yourcompany.blindnavigation.utils.VoiceFeedbackManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.util.concurrent.Executors


// 应用状态数据类
data class AppState(
    val isLoggedIn: Boolean = false,
    val currentPage: String = "main",
    val currentUser: User? = null,
    val destination: TextFieldValue = TextFieldValue(""),
    val isCameraOn: Boolean = true,
    val isVoiceNavigationOn: Boolean = true,
    val currentMode: String = "障碍物检测",
    val voicePromptMode: String = "详细",
    val detectionResult: String? = null
)

class MainActivity : ComponentActivity() {
    private lateinit var vibrator: Vibrator
    var appState by mutableStateOf(AppState())  // 改为 var，移除 private
        private set  // 外部只能读取

    // 位置服务
    private lateinit var fusedLocationClient: FusedLocationProviderClient

    // WebSocket服务
    private var webSocketService: WebSocketService? = null

    // GPS 发送标志
    private var isSendingGps = false

    // 导航启动标志
    private var isNavigationStarted = false

    // 语音防重复机制
    private var lastSpeakTime = 0L
    private var lastSpeakText = ""
    private val speakDebounceMs = 1000L

    private var lastNavigationInstruction: String = ""
    private var lastInstructionTime: Long = 0L
    private val instructionDebounceMs = 3000L  // 3秒内相同指令不重复播报



    // 服务器连接状态
    var serverReachable by mutableStateOf(false)
        private set

    // 网络服务 - 移除 private，让 SettingsPage 可以访问
    val networkService = NetworkService()

    // 感知服务WebSocket（A成员）- 接收预警
    var perceptionWebSocket: PerceptionWebSocketService? = null

    // 导航服务WebSocket（B成员）- 接收导航指令
    var navigationWebSocket: WebSocketService? = null

    private var isAudioRecordingEnabled = true  // 音频采集开关
    private var audioRecordJob: Job? = null
    private var isWarningPlaying = false  // 是否正在播报警告
    private var pendingNavigationInstruction: String? = null  // 被中断的导航指令

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val deniedPermissions = permissions.filter { !it.value }.keys
        if (deniedPermissions.isNotEmpty()) {
            speak("权限未授予，部分功能受限")
            Toast.makeText(this, "请授予必要权限", Toast.LENGTH_LONG).show()
        }
    }



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 初始化本地认证管理器
        LocalAuthManager.init(this)

        // 初始化振动器
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = getSystemService(VIBRATOR_MANAGER_SERVICE) as android.os.VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(VIBRATOR_SERVICE) as Vibrator
        }

        // 初始化位置服务
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        // ❌ 移除这行，不使用系统 TTS
        // BaiduTTSManager.initSystemTts(this)

        // 检查登录状态
        checkLoginStatus()

        // 请求必要权限
        requestPermissions()

        // 测试服务器连接
        testServerConnection()

        // ✅ 预获取百度 TTS Token
        CoroutineScope(Dispatchers.IO).launch {
            val token = BaiduTTSManager.fetchToken()
            withContext(Dispatchers.Main) {
                if (token != null) {
                    Log.e("TTS", "✅ 百度 TTS Token 预获取成功")
                } else {
                    Log.e("TTS", "❌ 百度 TTS Token 获取失败")
                }
            }
        }

        setContent {
            val colorScheme = darkColorScheme(
                primary = PrimaryBlue,
                onPrimary = Color.White,
                secondary = LightBlue,
                onSecondary = TextBlack,
                background = LightBlue,
                onBackground = TextBlack,
                surface = White,
                onSurface = TextBlack,
                error = AccentRed,
                onError = Color.White
            )

            MaterialTheme(
                colorScheme = colorScheme,
                typography = Typography
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    AppContent()
                }
            }
        }
    }

    @Composable
    private fun AppContent() {
        if (appState.isLoggedIn) {
            MainAppContent()
        } else {
            LoginRegisterContent(
                onLoginSuccess = { user ->
                    appState = appState.copy(
                        isLoggedIn = true,
                        currentUser = user
                    )
                    speak("登录成功，欢迎使用视界导航")
                    vibrateShort()
                },
                onRegisterSuccess = { user ->
                    appState = appState.copy(
                        isLoggedIn = true,
                        currentUser = user
                    )
                    speak("注册成功，欢迎使用视界导航")
                    vibrateShort()
                }
            )
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun MainAppContent() {
        Scaffold(
            topBar = {
                if (appState.currentPage != "camera") {
                    TopAppBar(
                        title = {
                            Text(
                                text = "视界 - 盲人无障碍导航",
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.semantics { contentDescription = "视界盲人无障碍导航应用标题" }
                            )
                        },
                        colors = TopAppBarDefaults.topAppBarColors(
                            containerColor = PrimaryBlue,
                            titleContentColor = Color.White
                        )
                    )
                }
            }
        ) { innerPadding ->
            Box(
                modifier = Modifier
                    .padding(innerPadding)
                    .fillMaxSize()
                    .background(LightBlue)
                    .clickable(enabled = appState.currentPage != "main" && appState.currentPage != "camera") {
                        if (appState.currentPage != "main") {
                            appState = appState.copy(currentPage = "main")
                            speak("切换到导航页面")
                            vibrateShort()
                        }
                    }
            ) {
                when (appState.currentPage) {
                    "main" -> NavigationPage(
                        destination = appState.destination,
                        onDestinationChange = { appState = appState.copy(destination = it) },
                        isVoiceNavigationOn = appState.isVoiceNavigationOn,
                        onVoiceNavigationToggle = { isOn ->
                            appState = appState.copy(isVoiceNavigationOn = isOn)
                            speak(if (isOn) "语音导航开启" else "语音导航关闭")
                            vibrateShort()
                        },
                        onNavigate = {
                            if (appState.destination.text.isBlank()) {
                                speak("请输入目的地")
                                vibrateError()
                            } else {
                                planRoute(appState.destination.text)
                            }
                        },
                        onPageChange = { page ->
                            appState = appState.copy(currentPage = page)
                            speak("切换到${getPageName(page)}页面")
                            vibrateShort()
                        },
                        currentMode = appState.currentMode,
                        onModeChange = { mode ->
                            appState = appState.copy(currentMode = mode)
                            speak("切换到${mode}模式")
                            vibrateShort()
                        },
                        serverReachable = serverReachable
                    )
                    "emergency" -> EmergencyPage(
                        onEmergencyCall = {
                            makeEmergencyCall()
                        },
                        onSendLocation = {
                            speak("位置信息已发送给紧急联系人")
                            vibrateShort()
                            sendEmergencyLocation()
                            Toast.makeText(this@MainActivity, "位置已发送", Toast.LENGTH_SHORT).show()
                        }
                    )
                    "settings" -> SettingsPage(
                        onPageChange = { page ->
                            appState = appState.copy(currentPage = page)
                            speak("切换到${getPageName(page)}页面")
                            vibrateShort()
                        },
                        onLogout = {
                            if (LocalAuthManager.logout()) {
                                stopNavigation()
                                webSocketService?.disconnect()
                                webSocketService = null
                                appState = AppState()
                                speak("已退出登录")
                                Toast.makeText(this@MainActivity, "已退出登录", Toast.LENGTH_SHORT).show()
                            }
                        },
                        voicePromptMode = appState.voicePromptMode,
                        onVoicePromptModeChange = { mode ->
                            appState = appState.copy(voicePromptMode = mode)
                            speak("语音提示模式已设置为${mode}")
                            vibrateShort()
                        },
                        serverReachable = serverReachable,
                        onTestConnection = { testServerConnection() }
                    )
                    "camera" -> CameraPage(
                        onPageChange = { page ->
                            appState = appState.copy(currentPage = page)
                            speak("切换到${getPageName(page)}页面")
                            vibrateShort()
                        },
                        isCameraOn = appState.isCameraOn,
                        onCameraToggle = { isOn ->
                            appState = appState.copy(isCameraOn = isOn)
                            speak(if (isOn) "摄像头已开启，开始检测" else "摄像头已关闭")
                            vibrateShort()
                        },
                        currentMode = appState.currentMode,
                        onModeChange = { mode ->
                            appState = appState.copy(currentMode = mode)
                            speak("切换到${mode}模式")
                            vibrateShort()
                        },
                        onAnalyzeImage = { imageBytes -> analyzeImage(imageBytes) },
                        serverReachable = serverReachable,
                        detectionResult = appState.detectionResult
                    )
                }
            }
        }
    }

    @Composable
    private fun LoginRegisterContent(
        onLoginSuccess: (User) -> Unit,
        onRegisterSuccess: (User) -> Unit
    ) {
        var showLogin by remember { mutableStateOf(true) }
        var username by remember { mutableStateOf("") }
        var password by remember { mutableStateOf("") }
        var phone by remember { mutableStateOf("") }
        var emergencyContact by remember { mutableStateOf("") }
        var errorMessage by remember { mutableStateOf<String?>(null) }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.White)
                .verticalScroll(rememberScrollState())
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "视界导航",
                fontSize = 36.sp,
                fontWeight = FontWeight.Bold,
                color = PrimaryBlue,
                modifier = Modifier.padding(bottom = 32.dp)
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 24.dp),
                horizontalArrangement = Arrangement.Center
            ) {
                Button(
                    onClick = {
                        showLogin = true
                        speak("登录")
                        vibrateShort()
                        errorMessage = null
                    },
                    modifier = Modifier
                        .weight(1f)
                        .padding(end = 4.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (showLogin) PrimaryBlue else LightGray
                    ),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        "登录",
                        color = if (showLogin) Color.White else TextBlack,
                        fontWeight = FontWeight.Medium
                    )
                }

                Button(
                    onClick = {
                        showLogin = false
                        speak("注册")
                        vibrateShort()
                        errorMessage = null
                    },
                    modifier = Modifier
                        .weight(1f)
                        .padding(start = 4.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (!showLogin) PrimaryBlue else LightGray
                    ),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        "注册",
                        color = if (!showLogin) Color.White else TextBlack,
                        fontWeight = FontWeight.Medium
                    )
                }
            }

            OutlinedTextField(
                value = username,
                onValueChange = { username = it },
                label = { Text("用户名") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                shape = RoundedCornerShape(12.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = PrimaryBlue,
                    focusedLabelColor = PrimaryBlue
                )
            )

            OutlinedTextField(
                value = password,
                onValueChange = { password = it },
                label = { Text("密码") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                shape = RoundedCornerShape(12.dp),
                visualTransformation = PasswordVisualTransformation(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = PrimaryBlue,
                    focusedLabelColor = PrimaryBlue
                )
            )

            if (!showLogin) {
                OutlinedTextField(
                    value = phone,
                    onValueChange = { phone = it },
                    label = { Text("手机号 *") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp),
                    shape = RoundedCornerShape(12.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = PrimaryBlue,
                        focusedLabelColor = PrimaryBlue
                    )
                )

                OutlinedTextField(
                    value = emergencyContact,
                    onValueChange = { emergencyContact = it },
                    label = { Text("紧急联系人（选填）") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp),
                    shape = RoundedCornerShape(12.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = PrimaryBlue,
                        focusedLabelColor = PrimaryBlue
                    )
                )
            }

            errorMessage?.let {
                Text(
                    text = it,
                    color = AccentRed,
                    modifier = Modifier.padding(bottom = 12.dp),
                    fontSize = 14.sp
                )
            }

            Button(
                onClick = {
                    vibrateShort()
                    if (showLogin) {
                        handleLogin(username, password, onLoginSuccess) { error ->
                            errorMessage = error
                        }
                    } else {
                        handleRegister(
                            username = username,
                            password = password,
                            phone = phone,
                            emergencyContact = emergencyContact.ifBlank { null },
                            onSuccess = onRegisterSuccess,
                            onError = { error -> errorMessage = error }
                        )
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .padding(bottom = 16.dp),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = PrimaryBlue
                ),
                enabled = if (showLogin)
                    username.isNotBlank() && password.isNotBlank()
                else
                    username.isNotBlank() && password.isNotBlank() && phone.isNotBlank()
            ) {
                Text(
                    text = if (showLogin) "登录" else "注册",
                    fontSize = 18.sp,
                    color = Color.White,
                    fontWeight = FontWeight.Bold
                )
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 8.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(12.dp)
                        .background(
                            color = if (serverReachable) Color.Green else Color.Red,
                            shape = CircleShape
                        )
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (serverReachable) "服务器已连接" else "服务器未连接",
                    fontSize = 14.sp,
                    color = if (serverReachable) Color.Green else Color.Red
                )
            }

            Button(
                onClick = {
                    vibrateShort()
                    testServerConnection()
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(44.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF2196F3)
                )
            ) {
                Text("测试服务器连接", fontSize = 14.sp, color = Color.White)
            }

            Spacer(modifier = Modifier.height(20.dp))
        }

        LaunchedEffect(showLogin) {
            delay(500)
            if (showLogin) {
                speak("登录页面，请输入用户名和密码")
            } else {
                speak("注册页面，请输入用户名、密码、手机号，紧急联系人可选")
            }
        }
    }

    private fun checkLoginStatus() {
        val isLoggedIn = LocalAuthManager.isLoggedIn()
        val currentUser = LocalAuthManager.getCurrentUser()

        appState = appState.copy(
            isLoggedIn = isLoggedIn,
            currentUser = currentUser
        )

        if (isLoggedIn && currentUser != null) {
            Log.d("MainActivity", "用户已登录: ${currentUser.username}")
            initializeWebSockets(currentUser.username)  // 👈 改为调用新方法
            android.os.Handler(mainLooper).postDelayed({
                speak("欢迎回来，${currentUser.username}")
            }, 1000)
        }
    }

    private fun initializeWebSockets(userId: String) {
        Log.e("MainActivity", "========== initializeWebSockets 被调用 ==========")
        Log.e("MainActivity", "用户ID: $userId")

        // 1. 感知服务（A成员）- 避免重复初始化
        if (perceptionWebSocket == null) {
            perceptionWebSocket = PerceptionWebSocketService(userId)
            perceptionWebSocket?.connect()
        } else {
            Log.d("MainActivity", "感知服务已存在，跳过初始化")
        }

        // 2. 导航服务（B成员）- 避免重复初始化
        if (navigationWebSocket == null) {
            navigationWebSocket = WebSocketService(userId)
            navigationWebSocket?.connect()
        } else {
            Log.d("MainActivity", "导航服务已存在，跳过初始化")
        }

        // 监听感知服务的预警消息
        CoroutineScope(Dispatchers.Main).launch {
            perceptionWebSocket?.warningMessages?.collect { warning ->
                Log.e("WARNING", "⚠️ 收到预警: level=${warning.warningLevel}, text=${warning.ttsText}")

                if (warning.ttsText.isNotBlank()) {
                    speak(warning.ttsText)
                }

                when (warning.vibrationIntensity) {
                    1 -> vibrateShort()
                    2 -> vibrateMedium()
                    3 -> vibrateLong()
                }

                when (warning.warningLevel) {
                    1 -> appState = appState.copy(detectionResult = "⚠️ ${warning.ttsText}")
                    2 -> appState = appState.copy(detectionResult = "⚠️⚠️ ${warning.ttsText}")
                    3 -> appState = appState.copy(detectionResult = "⚠️⚠️⚠️ ${warning.ttsText}")
                }
            }
        }

        // ✅ 添加缓存变量（在方法内部也可以）
        var lastInstruction: String = ""

        // 监听导航服务的消息（B成员）
        CoroutineScope(Dispatchers.Main).launch {
            navigationWebSocket?.messages?.collect { message ->
                Log.e("MAIN_ACTIVITY", "========== 收到导航消息 ==========")
                Log.e("MAIN_ACTIVITY", "type: ${message.type}")
                Log.e("MAIN_ACTIVITY", "instruction: ${message.instruction}")

                when (message.type) {
                    "nav_started" -> {
                        Log.e("MAIN_ACTIVITY", "🚀 导航开始")
                        // ✅ 导航开始时重置缓存
                        lastInstruction = ""
                        lastInstructionTime = 0L

                        // ✅ 导航开始时先播报一个测试语音，确保 TTS 初始化
                        VoiceFeedbackManager.provideFeedback(this@MainActivity, "导航已开始", VibrationType.SETTINGS)
                        isNavigationStarted = true
                    }
                    "nav_stopped" -> {
                        Log.e("MAIN_ACTIVITY", "🛑 导航结束")
                        VoiceFeedbackManager.provideFeedback(this@MainActivity, "导航已结束", VibrationType.SETTINGS)
                        isNavigationStarted = false
                        lastInstruction = ""
                        lastInstructionTime = 0L
                    }
                    "position_update" -> {
                        message.instruction?.let { instruction ->
                            val currentTime = System.currentTimeMillis()

                            // ✅ 打印详细信息
                            Log.e("MAIN_ACTIVITY", "当前指令: $instruction")
                            Log.e("MAIN_ACTIVITY", "上次指令: $lastInstruction")
                            Log.e("MAIN_ACTIVITY", "时间差: ${currentTime - lastInstructionTime}ms")

                            // ✅ 如果指令不同，或者距离上次播报超过30秒，就播报
                            if (instruction != lastInstruction ||
                                (currentTime - lastInstructionTime > 30000 && lastInstruction.isNotEmpty())) {

                                lastInstruction = instruction
                                lastInstructionTime = currentTime

                                Log.e("MAIN_ACTIVITY", "📢 播报新指令: $instruction")
                                VoiceFeedbackManager.provideFeedback(this@MainActivity, instruction, VibrationType.SETTINGS)
                            } else {
                                Log.e("MAIN_ACTIVITY", "⏭️ 跳过重复播报: $instruction")
                            }
                        }
                    }
                }
            }
        }
    }

    private fun handleLogin(
        username: String,
        password: String,
        onSuccess: (User) -> Unit,
        onError: (String) -> Unit
    ) {
        if (username.isBlank() || password.isBlank()) {
            onError("用户名和密码不能为空")
            speak("用户名和密码不能为空")
            return
        }

        Log.d("MainActivity", "尝试登录: $username")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                if (serverReachable) {
                    val result = networkService.login(username, password)

                    withContext(Dispatchers.Main) {
                        if (result.isSuccess) {
                            val response = result.getOrNull()
                            if (response != null && response.success) {
                                val user = User(
                                    username = username,
                                    password = password,
                                    email = null,
                                    phone = response.phone ?: "",
                                    emergencyContact = response.emergencyContact
                                )
                                LocalAuthManager.register(username, password, null)
                                onSuccess(user)
                                speak("登录成功")
                                Log.d("MainActivity", "✅ 登录成功: $username")
                                initializeWebSockets(username)
                            } else {
                                onError(response?.message ?: "登录失败")
                                speak("登录失败")
                                Log.e("MainActivity", "❌ 登录失败: ${response?.message}")
                            }
                        } else {
                            val error = result.exceptionOrNull()
                            onError("网络错误: ${error?.message}")
                            speak("网络错误")
                            Log.e("MainActivity", "❌ 登录网络错误", error)
                        }
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        if (LocalAuthManager.login(username, password)) {
                            val user = LocalAuthManager.getCurrentUser()
                            if (user != null) {
                                onSuccess(user)
                                speak("离线模式登录成功")
                            } else {
                                onError("登录失败")
                            }
                        } else {
                            onError("用户名或密码错误")
                            speak("用户名或密码错误")
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "登录异常", e)
                withContext(Dispatchers.Main) {
                    onError("网络错误，请稍后重试")
                    speak("网络错误，请稍后重试")
                }
            }
        }
    }

    private fun handleRegister(
        username: String,
        password: String,
        phone: String,
        emergencyContact: String?,
        onSuccess: (User) -> Unit,
        onError: (String) -> Unit
    ) {
        if (username.isBlank() || password.isBlank()) {
            onError("用户名和密码不能为空")
            speak("用户名和密码不能为空")
            return
        }

        if (phone.isBlank()) {
            onError("手机号不能为空")
            speak("手机号不能为空")
            return
        }

        if (password.length < 6) {
            onError("密码至少需要6位")
            speak("密码至少需要6位")
            return
        }

        Log.d("MainActivity", "尝试注册: $username, 手机号: $phone")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                if (serverReachable) {
                    val result = networkService.register(
                        username = username,
                        password = password,
                        phone = phone,
                        emergencyContact = emergencyContact
                    )

                    withContext(Dispatchers.Main) {
                        if (result.isSuccess) {
                            val response = result.getOrNull()
                            if (response?.success == true) {
                                val user = User(
                                    username = username,
                                    password = password,
                                    email = null,
                                    phone = phone,
                                    emergencyContact = emergencyContact
                                )
                                if (LocalAuthManager.register(username, password, null)) {
                                    onSuccess(user)
                                    speak("注册成功")
                                    Log.d("MainActivity", "✅ 注册成功: $username")
                                    initializeWebSockets(username)
                                } else {
                                    onError("注册失败")
                                }
                            } else {
                                onError(response?.message ?: "注册失败")
                                speak("注册失败: ${response?.message}")
                                Log.e("MainActivity", "❌ 注册失败: ${response?.message}")
                            }
                        } else {
                            val error = result.exceptionOrNull()
                            onError("网络错误: ${error?.message}")
                            speak("网络错误，请稍后重试")
                            Log.e("MainActivity", "❌ 注册网络错误", error)
                        }
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        if (LocalAuthManager.register(username, password, null)) {
                            val user = User(
                                username = username,
                                password = password,
                                email = null,
                                phone = phone,
                                emergencyContact = emergencyContact
                            )
                            onSuccess(user)
                            speak("离线模式注册成功")
                        } else {
                            onError("用户名已存在")
                            speak("用户名已存在")
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "注册异常", e)
                withContext(Dispatchers.Main) {
                    onError("网络错误，请稍后重试")
                    speak("网络错误，请稍后重试")
                }
            }
        }
    }

    private fun testServerConnection() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val result = networkService.healthCheck()
                withContext(Dispatchers.Main) {
                    serverReachable = result.isSuccess
                    if (result.isSuccess) {
                        val healthData = result.getOrNull()
                        speak("服务器连接正常")
                        Log.d("MainActivity", "服务器状态: ${healthData?.status}")
                    } else {
                        speak("无法连接服务器")
                        Log.e("MainActivity", "服务器连接失败")
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "服务器连接测试失败", e)
                withContext(Dispatchers.Main) {
                    serverReachable = false
                    speak("服务器连接失败")
                }
            }
        }
    }

    private fun getCurrentLocation(onLocationResult: (Location?) -> Unit) {
        try {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {
                onLocationResult(null)
                return
            }

            fusedLocationClient.lastLocation.addOnSuccessListener { location: Location? ->
                if (location != null) {
                    onLocationResult(location)
                } else {
                    fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
                        .addOnSuccessListener { newLocation: Location? ->
                            onLocationResult(newLocation)
                        }.addOnFailureListener {
                            onLocationResult(null)
                        }
                }
            }.addOnFailureListener {
                onLocationResult(null)
            }
        } catch (e: SecurityException) {
            Log.e("MainActivity", "位置权限未授予", e)
            onLocationResult(null)
        }
    }

    @Suppress("DEPRECATION")
    private fun getLocationFromAddress(address: String): Pair<Double, Double>? {
        return try {
            val geocoder = Geocoder(this)
            val addresses = geocoder.getFromLocationName(address, 1)
            if (!addresses.isNullOrEmpty()) {
                val location = addresses[0]
                Pair(location.longitude, location.latitude)
            } else {
                null
            }
        } catch (e: IOException) {
            Log.e("MainActivity", "地理编码失败", e)
            null
        }
    }

    private fun planRoute(destinationText: String) {
        getCurrentLocation { location ->
            if (location == null) {
                speak("无法获取当前位置，请检查GPS")
                return@getCurrentLocation
            }

            val destCoordinates = getLocationFromAddress(destinationText)
            if (destCoordinates == null) {
                speak("无法找到目的地坐标，请使用更具体的地名")
                return@getCurrentLocation
            }

            val (destLng, destLat) = destCoordinates
            executeRoutePlanning(destinationText, location.longitude, location.latitude, destLng, destLat)
        }
    }

    private fun executeRoutePlanning(
        destinationText: String,
        originLng: Double,
        originLat: Double,
        destLng: Double,
        destLat: Double
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.e("MAIN_DEBUG", "========== 开始路径规划 ==========")
                Log.e("MAIN_DEBUG", "目的地: $destinationText")
                Log.e("MAIN_DEBUG", "起点: ($originLng, $originLat)")
                Log.e("MAIN_DEBUG", "终点: ($destLng, $destLat)")
                Log.e("MAIN_DEBUG", "服务器状态: $serverReachable")

                if (!serverReachable) {
                    withContext(Dispatchers.Main) {
                        Log.e("SPEAK_DEBUG", "🔊 服务器未连接")
                        VoiceFeedbackManager.provideFeedback(this@MainActivity, "服务器未连接，无法规划路线", VibrationType.SETTINGS)
                    }
                    return@launch
                }

                Log.e("MAIN_DEBUG", "开始调用 networkService.planRoute...")
                val result = networkService.planRoute(
                    originLng = originLng,
                    originLat = originLat,
                    destLng = destLng,
                    destLat = destLat
                )
                Log.e("MAIN_DEBUG", "planRoute 调用完成，result.isSuccess = ${result.isSuccess}")

                withContext(Dispatchers.Main) {
                    if (result.isSuccess) {
                        val response = result.getOrNull()
                        Log.e("MAIN_DEBUG", "response = $response")
                        Log.e("MAIN_DEBUG", "response?.success = ${response?.success}")
                        Log.e("MAIN_DEBUG", "response?.message = ${response?.message}")

                        if (response != null && response.success) {
                            val distance = response.totalDistance ?: 0.0
                            val time = response.totalDuration ?: 0

                            Log.e("MAIN_DEBUG", "✅ 路径规划成功: distance=$distance, time=$time")

                            val distanceText = if (distance >= 1000) {
                                String.format(java.util.Locale.CHINA, "%.1f公里", distance / 1000)
                            } else {
                                "${distance.toInt()}米"
                            }

                            val successMessage = "路径规划成功，总距离${distanceText}，预计${time}分钟"
                            Log.e("SPEAK_DEBUG", "🔊🔊🔊 强制播报: $successMessage")

                            try {
                                VoiceFeedbackManager.provideFeedback(this@MainActivity, successMessage, VibrationType.SETTINGS)
                                Log.e("SPEAK_DEBUG", "✅ VoiceFeedbackManager 调用成功")
                            } catch (e: Exception) {
                                Log.e("SPEAK_DEBUG", "❌ VoiceFeedbackManager 调用失败", e)
                            }

                            appState = appState.copy(
                                destination = TextFieldValue("前往$destinationText，距离${distanceText}")
                            )

                            startNavigationWithWebSocket(response)
                        } else {
                            val errorMsg = response?.message ?: "未知错误"
                            Log.e("MAIN_DEBUG", "❌ 路径规划失败: $errorMsg")
                            Log.e("SPEAK_DEBUG", "🔊 播报失败: $errorMsg")
                            VoiceFeedbackManager.provideFeedback(this@MainActivity, "路径规划失败: $errorMsg", VibrationType.SETTINGS)
                        }
                    } else {
                        val error = result.exceptionOrNull()
                        Log.e("MAIN_DEBUG", "❌ 网络请求失败", error)
                        Log.e("MAIN_DEBUG", "错误信息: ${error?.message}")
                        Log.e("SPEAK_DEBUG", "🔊 播报网络错误")
                        VoiceFeedbackManager.provideFeedback(this@MainActivity, "路径规划失败: ${error?.message ?: "网络错误"}", VibrationType.SETTINGS)
                    }
                }
            } catch (e: Exception) {
                Log.e("MAIN_DEBUG", "❌ 路径规划异常", e)
                Log.e("MAIN_DEBUG", "异常信息: ${e.message}")
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    Log.e("SPEAK_DEBUG", "🔊 播报异常")
                    VoiceFeedbackManager.provideFeedback(this@MainActivity, "路径规划失败", VibrationType.SETTINGS)
                }
            }
        }
    }

    private fun startNavigationWithWebSocket(routeResponse: RouteResponse) {
        if (isNavigationStarted) {
            Log.w("MAIN_DEBUG", "导航已启动，跳过重复调用")
            return
        }

        val currentUser = appState.currentUser
        if (currentUser != null && serverReachable) {
            // 使用 navigationWebSocket 而不是重新创建
            if (navigationWebSocket == null) {
                navigationWebSocket = WebSocketService(currentUser.username)
                navigationWebSocket?.connect()
            }

            // 确保 navigationWebSocket 已经连接
            CoroutineScope(Dispatchers.IO).launch {
                // 等待 WebSocket 连接
                var retryCount = 0
                while (navigationWebSocket?.connectionState?.value != WebSocketService.ConnectionState.CONNECTED
                    && retryCount < 10) {
                    delay(500)
                    retryCount++
                }

                if (navigationWebSocket?.connectionState?.value != WebSocketService.ConnectionState.CONNECTED) {
                    withContext(Dispatchers.Main) {
                        VoiceFeedbackManager.provideFeedback(this@MainActivity, "导航服务连接失败", VibrationType.SETTINGS)
                    }
                    return@launch
                }

                val steps = mutableListOf<Map<String, Any>>()

                routeResponse.steps?.let { stepList ->
                    stepList.forEachIndexed { index, step ->
                        var endLat = 0.0
                        var endLng = 0.0

                        try {
                            step.polyline?.let { polyline ->
                                val points = polyline.split(";")
                                if (points.isNotEmpty()) {
                                    val lastPoint = points.last()
                                    val coords = lastPoint.split(",")
                                    if (coords.size == 2) {
                                        endLng = coords[0].toDouble()
                                        endLat = coords[1].toDouble()
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("MainActivity", "解析polyline失败", e)
                        }

                        steps.add(
                            mapOf(
                                "index" to index,
                                "instruction" to (step.instruction ?: "继续前行"),
                                "distance" to (step.distance ?: 0.0),
                                "duration" to (step.duration ?: 0),
                                "polyline" to (step.polyline ?: ""),
                                "road" to (step.road ?: ""),
                                "end_location" to mapOf(
                                    "lat" to endLat,
                                    "lng" to endLng
                                )
                            )
                        )
                    }
                }

                withContext(Dispatchers.Main) {
                    if (steps.isNotEmpty()) {
                        navigationWebSocket?.startNavigation(steps)
                        startSendingGPSUpdates()  // 这个方法已经在使用 navigationWebSocket

                        isNavigationStarted = true

                        // ✅ 导航开始时启动音频采集
                        startPeriodicAudioRecording()

                        steps.firstOrNull()?.let { firstStep ->
                            val instruction = firstStep["instruction"] as? String ?: "开始导航"
                            val navMessage = "导航已开始，${instruction}"
                            Log.e("SPEAK_DEBUG", "🔊🔊🔊 强制播报导航: $navMessage")

                            try {
                                VoiceFeedbackManager.provideFeedback(this@MainActivity, navMessage, VibrationType.SETTINGS)
                                Log.e("SPEAK_DEBUG", "✅ 导航播报成功")
                            } catch (e: Exception) {
                                Log.e("SPEAK_DEBUG", "❌ 导航播报失败", e)
                            }
                        }
                    } else {
                        Log.e("SPEAK_DEBUG", "🔊 播报: 无详细步骤")
                        VoiceFeedbackManager.provideFeedback(this@MainActivity, "路径规划成功，但无详细步骤信息", VibrationType.SETTINGS)
                    }
                }
            }
        } else {
            Log.e("SPEAK_DEBUG", "❌ 无法启动导航: currentUser=$currentUser, serverReachable=$serverReachable")
        }
    }

    private fun startSendingGPSUpdates() {
        if (isSendingGps) return
        isSendingGps = true

        CoroutineScope(Dispatchers.IO).launch {
            try {
                var retryCount = 0
                while (isSendingGps && appState.currentPage == "main" && serverReachable) {
                    // 等待导航服务连接
                    if (navigationWebSocket?.connectionState?.value == WebSocketService.ConnectionState.CONNECTED) {
                        retryCount = 0  // 重置重试计数
                        getCurrentLocation { location ->
                            if (location != null) {
                                val gpsData = GPSData(
                                    latitude = location.latitude,
                                    longitude = location.longitude,
                                    accuracy = location.accuracy,
                                    timestamp = System.currentTimeMillis() / 1000.0,
                                    speed = location.speed,
                                    bearing = location.bearing
                                )
                                navigationWebSocket?.sendGPSUpdate(gpsData)
                                Log.d("GPS", "📤 GPS发送成功: (${location.latitude}, ${location.longitude})")
                            } else {
                                Log.w("GPS", "⚠️ 无法获取位置")
                            }
                        }
                    } else {
                        Log.d("GPS", "⏸️ 等待导航服务连接... (${retryCount++}/10)")
                        if (retryCount >= 10) {
                            Log.e("GPS", "❌ 导航服务连接超时")
                            break
                        }
                    }
                    delay(1000)  // 每秒一次
                }
            } finally {
                isSendingGps = false
            }
        }
    }

    private fun stopSendingGPSUpdates() {
        isSendingGps = false
    }

    private fun stopNavigation() {
        isNavigationStarted = false
        stopSendingGPSUpdates()
        navigationWebSocket?.stopNavigation()
        // 注意：如果之前有 stopBackgroundDetection()，先注释掉
        // stopBackgroundDetection()
        stopPeriodicAudioRecording()  // 这行需要先定义这个方法
    }

    // ✅ 启动定时录音（每5秒录一次）
    private fun startPeriodicAudioRecording() {
        if (audioRecordJob != null) return

        audioRecordJob = CoroutineScope(Dispatchers.IO).launch {
            while (isAudioRecordingEnabled && isNavigationStarted) {
                delay(5000)

                if (isWarningPlaying) {
                    Log.d("AudioRecorder", "⏸️ 正在播报警告，跳过录音")
                    continue
                }

                AudioRecorder.startRecording { audioBase64: String? ->
                    if (audioBase64 != null && perceptionWebSocket?.connectionState?.value == PerceptionWebSocketService.ConnectionState.CONNECTED) {
                        perceptionWebSocket?.sendSensorData(
                            tofDistance = 999.0,
                            tofDirection = "unknown",
                            videoFrameBase64 = null,
                            audioBase64 = audioBase64
                        )
                        Log.d("AudioRecorder", "📤 音频数据已发送")
                    }
                }
            }
        }
    }

    // ✅ 停止音频采集
    private fun stopPeriodicAudioRecording() {
        audioRecordJob?.cancel()
        audioRecordJob = null
        AudioRecorder.stopRecording()
    }

    // ✅ 预警消息处理，添加打断逻辑
    private fun handleWarningWithInterrupt(warning: WarningMessage) {
        if (warning.warningLevel >= 2 && warning.ttsText.isNotBlank()) {
            BaiduTTSManager.stopSpeaking()
            isWarningPlaying = true

            speak(warning.ttsText)

            when (warning.vibrationIntensity) {
                1 -> vibrateShort()
                2 -> vibrateMedium()
                3 -> vibrateLong()
            }

            CoroutineScope(Dispatchers.Main).launch {
                delay(3000)
                isWarningPlaying = false
                pendingNavigationInstruction?.let {
                    speak(it)
                    pendingNavigationInstruction = null
                }
            }
        } else if (warning.ttsText.isNotBlank()) {
            speak(warning.ttsText)
            when (warning.vibrationIntensity) {
                1 -> vibrateShort()
                2 -> vibrateMedium()
                3 -> vibrateLong()
            }
        }
    }

    private fun analyzeImage(imageBytes: ByteArray) {
        Log.e("PerceptionTest", "🔍🔍🔍🔍🔍 analyzeImage 被调用！图片大小: ${imageBytes.size}")

        if (imageBytes.isEmpty()) {
            Log.e("PerceptionTest", "❌ 图片数据为空")
            return
        }

        if (perceptionWebSocket?.connectionState?.value == PerceptionWebSocketService.ConnectionState.CONNECTED) {
            CoroutineScope(Dispatchers.IO).launch {
                val videoFrameBase64 = android.util.Base64.encodeToString(
                    imageBytes,
                    android.util.Base64.NO_WRAP
                )
                Log.e("PerceptionTest", "📸 Base64 编码完成，长度: ${videoFrameBase64.length}")

                perceptionWebSocket?.sendSensorData(
                    tofDistance = 999.0,
                    tofDirection = "unknown",
                    videoFrameBase64 = videoFrameBase64,
                    audioBase64 = null
                )
            }
        } else {
            Log.e("PerceptionTest", "⚠️ 感知服务未连接")
        }
    }

    // =====================================
// 紧急求助相关方法
// =====================================

    /**
     * 显示设置紧急联系人对话框（使用 AlertDialog）
     */
    private fun makeEmergencyCall() {
        val currentUser = appState.currentUser
        val emergencyContact = currentUser?.emergencyContact

        if (emergencyContact.isNullOrBlank()) {
            // 没有紧急联系人，显示设置对话框
            showSetEmergencyContactComposeDialog()
        } else {
            // 有紧急联系人，显示确认对话框
            showEmergencyConfirmComposeDialog(emergencyContact)
        }
    }

    /**
     * 显示设置紧急联系人对话框（使用 AlertDialog）
     */
    private fun showSetEmergencyContactComposeDialog() {
        // 由于需要在 Compose 中显示对话框，我们需要在 Activity 中创建一个 Compose 视图
        // 简单起见，使用传统的 AlertDialog
        val editText = android.widget.EditText(this).apply {
            hint = "请输入紧急联系人手机号"
            inputType = android.text.InputType.TYPE_CLASS_PHONE
        }

        android.app.AlertDialog.Builder(this)
            .setTitle("设置紧急联系人")
            .setMessage("未设置紧急联系人，请先设置")
            .setView(editText)
            .setPositiveButton("保存并呼叫") { _, _ ->
                val phoneNumber = editText.text.toString()
                if (phoneNumber.isNotBlank()) {
                    saveEmergencyContact(phoneNumber)
                    performEmergencyCall(phoneNumber)
                } else {
                    speak("请输入有效的手机号码")
                }
            }
            .setNegativeButton("取消") { _, _ ->
                speak("已取消紧急求助")
            }
            .show()
    }

    /**
     * 显示紧急求助确认对话框（使用 AlertDialog）
     */
    private fun showEmergencyConfirmComposeDialog(emergencyContact: String) {
        android.app.AlertDialog.Builder(this)
            .setTitle("紧急求助")
            .setMessage("是否立即呼叫紧急联系人 $emergencyContact？")
            .setPositiveButton("呼叫") { _, _ ->
                performEmergencyCall(emergencyContact)
            }
            .setNegativeButton("取消") { _, _ ->
                speak("已取消紧急求助")
            }
            .show()
    }

    /**
     * 执行紧急呼叫
     */
    private fun performEmergencyCall(phoneNumber: String) {
        speak("正在呼叫紧急联系人 $phoneNumber")
        vibrateLong()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CALL_PHONE)
            == PackageManager.PERMISSION_GRANTED
        ) {
            try {
                val intent = Intent(Intent.ACTION_CALL).apply {
                    data = "tel:$phoneNumber".toUri()
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                startActivity(intent)
                Log.d("MainActivity", "正在呼叫紧急联系人: $phoneNumber")
            } catch (e: Exception) {
                Log.e("MainActivity", "拨打电话失败", e)
                speak("拨打电话失败，请检查权限")
                Toast.makeText(this, "拨打电话失败: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        } else {
            speak("缺少拨打电话权限，请授予权限")
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.CALL_PHONE))
        }
    }

    /**
     * 保存紧急联系人
     */
    private fun saveEmergencyContact(phoneNumber: String) {
        val currentUser = appState.currentUser
        if (currentUser != null) {
            val updatedUser = currentUser.copy(emergencyContact = phoneNumber)
            appState = appState.copy(currentUser = updatedUser)
            LocalAuthManager.updateUser(updatedUser)
            speak("紧急联系人已设置为 $phoneNumber")

            if (serverReachable) {
                CoroutineScope(Dispatchers.IO).launch {
                    try {
                        val request = UpdateUserRequest(emergencyContact = phoneNumber)
                        networkService.updateUserInfo(currentUser.id, request)
                    } catch (e: Exception) {
                        Log.e("MainActivity", "同步紧急联系人失败", e)
                    }
                }
            }
        }
    }

    /**
     * 发送紧急位置
     */
    private fun sendEmergencyLocation() {
        getCurrentLocation { location ->
            if (location != null) {
                val locationText = "当前位置：经度 ${location.longitude}，纬度 ${location.latitude}"
                speak("已发送位置信息给紧急联系人")
                Log.d("MainActivity", "紧急位置: $locationText")

                val currentUser = appState.currentUser
                val emergencyContact = currentUser?.emergencyContact
                if (!emergencyContact.isNullOrBlank()) {
                    sendLocationSms(emergencyContact, location.latitude, location.longitude)
                }
            } else {
                speak("无法获取当前位置，请检查GPS")
            }
        }
    }

    /**
     * 发送位置短信
     */
    @Suppress("DEPRECATION")
    private fun sendLocationSms(phoneNumber: String, latitude: Double, longitude: Double) {
        try {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS)
                == PackageManager.PERMISSION_GRANTED
            ) {
                val locationText = "紧急求助！我当前位置：https://maps.google.com/?q=$latitude,$longitude"
                val smsManager = android.telephony.SmsManager.getDefault()
                smsManager.sendTextMessage(phoneNumber, null, locationText, null, null)

                Log.d("MainActivity", "位置短信已发送到: $phoneNumber")
                speak("位置短信已发送")
                Toast.makeText(this, "位置已发送", Toast.LENGTH_SHORT).show()
            } else {
                Log.e("MainActivity", "没有发送短信权限")
                speak("缺少发送短信权限")
                requestPermissionLauncher.launch(arrayOf(Manifest.permission.SEND_SMS))
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "发送位置短信失败", e)
            speak("发送位置短信失败")
        }
    }

    private fun requestPermissions() {
        val permissions = mutableListOf<String>()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.CAMERA)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.ACCESS_FINE_LOCATION)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.ACCESS_COARSE_LOCATION)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.RECORD_AUDIO)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CALL_PHONE)
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.CALL_PHONE)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS)
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.SEND_SMS)
        }

        if (permissions.isNotEmpty()) {
            requestPermissionLauncher.launch(permissions.toTypedArray())
        }
    }

    fun speak(text: String, vibrationType: VibrationType? = VibrationType.SETTINGS) {
        val currentTime = System.currentTimeMillis()

        if (text == lastSpeakText && currentTime - lastSpeakTime < speakDebounceMs) {
            Log.w("SPEAK_DEBUG", "⏭️ 跳过重复播报: $text")
            return
        }

        lastSpeakTime = currentTime
        lastSpeakText = text

        Log.e("SPEAK_DEBUG", "🔊 播报语音: $text")
        Log.d("MainActivity", "Speaking: $text")

        runOnUiThread {
            VoiceFeedbackManager.provideFeedback(this, text, vibrationType)
        }
    }

    fun vibrateShort() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(150, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(150)
        }
    }

    fun vibrateLong() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(400, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(400)
        }
    }

    fun vibrateError() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 150, 80, 150), -1))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(longArrayOf(0, 150, 80, 150), -1)
        }
    }

    fun vibrateMedium() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(250, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(250)
        }
    }

    fun getCurrentUser(): User? = appState.currentUser

    fun updateAppState(update: (AppState) -> AppState) {
        appState = update(appState)
    }

    private fun getPageName(page: String): String {
        return when (page) {
            "main" -> "导航主页面"
            "emergency" -> "紧急求助"
            "settings" -> "设置"
            "camera" -> "摄像头检测"
            else -> "导航主页面"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopNavigation()
        webSocketService?.disconnect()
        stopPeriodicAudioRecording()  // ✅ 确保停止录音
        BaiduTTSManager.release()
    }

    override fun onResume() {
        super.onResume()
        checkLoginStatus()
    }
}

// =====================================
// Composable 组件
// =====================================

/**
 * 长按3秒触发的SOS按钮组件
 */
@Composable
fun LongPressSOSButton(
    onLongPress: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    @Suppress("NewApi")
    val vibrator = remember {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = context.getSystemService(android.os.VibratorManager::class.java)
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(android.os.Vibrator::class.java)
        }
    }

    var isPressing by remember { mutableStateOf(false) }
    var longPressProgress by remember { mutableFloatStateOf(0f) }

    LaunchedEffect(isPressing) {
        if (isPressing) {
            var elapsed = 0f
            while (isPressing && elapsed < 3000f) {
                delay(50)
                elapsed += 50f
                longPressProgress = elapsed / 3000f
            }
            if (elapsed >= 3000f) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    vibrator?.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE))
                } else {
                    @Suppress("DEPRECATION")
                    vibrator?.vibrate(500)
                }
                onLongPress()
                longPressProgress = 0f
            }
        } else {
            longPressProgress = 0f
        }
    }

    Box(
        modifier = modifier
            .size(120.dp)
            .background(
                color = Color.Red.copy(alpha = 0.8f + 0.2f * longPressProgress),
                shape = CircleShape
            )
            .pointerInput(Unit) {
                detectTapGestures(
                    onPress = {
                        isPressing = true
                        tryAwaitRelease()
                        isPressing = false
                    }
                )
            }
            .semantics {
                contentDescription = "紧急求助按钮，长按3秒触发紧急呼叫"
            },
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = Icons.Default.Call,
                contentDescription = null,
                modifier = Modifier.size(48.dp),
                tint = Color.White
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "长按3秒",
                fontSize = 14.sp,
                color = Color.White
            )
            Text(
                text = "SOS",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            if (longPressProgress > 0) {
                Spacer(modifier = Modifier.height(4.dp))
                LinearProgressIndicator(
                    progress = { longPressProgress },
                    modifier = Modifier
                        .width(80.dp)
                        .height(4.dp),
                    color = Color.White,
                    trackColor = Color.White.copy(alpha = 0.3f)
                )
            }
        }
    }
}

@Composable
fun CameraPage(
    onPageChange: (String) -> Unit,
    isCameraOn: Boolean,
    onCameraToggle: (Boolean) -> Unit,
    currentMode: String,
    @Suppress("UNUSED_PARAMETER") onModeChange: (String) -> Unit,
    onAnalyzeImage: (ByteArray) -> Unit,
    serverReachable: Boolean,
    detectionResult: String?
) {
    val activity = LocalContext.current as MainActivity
    val lifecycleOwner = LocalLifecycleOwner.current
    val context = LocalContext.current

    var isFrontCamera by remember { mutableStateOf(false) }
    var showRouteMemoryDialog by remember { mutableStateOf(false) }
    var isAnalyzing by remember { mutableStateOf(false) }

    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val imageCapture = remember { ImageCapture.Builder().build() }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    var cameraSelector by remember { mutableStateOf(CameraSelector.DEFAULT_BACK_CAMERA) }

    LaunchedEffect(Unit) {
        if (!isCameraOn) {
            onCameraToggle(true)
        }
    }

    LaunchedEffect(isCameraOn, currentMode, cameraSelector) {
        Log.e("CameraPage", "🔍 LaunchedEffect 执行，isCameraOn=$isCameraOn, currentMode=$currentMode")
        if (isCameraOn && currentMode == "障碍物检测") {
            Log.e("CameraPage", "✅ 条件满足，开始拍照循环")
            // 先检查感知服务连接状态
            if (activity.perceptionWebSocket?.connectionState?.value != PerceptionWebSocketService.ConnectionState.CONNECTED) {
                Log.e("CameraPage", "⚠️ 感知服务未连接，尝试连接...")
                activity.perceptionWebSocket?.connect()
                delay(2000) // 等待连接
            }

            while (true) {
                delay(3000)
                if (!isAnalyzing) {
                    // 每次拍照前检查连接状态
                    if (activity.perceptionWebSocket?.connectionState?.value != PerceptionWebSocketService.ConnectionState.CONNECTED) {
                        Log.e("CameraPage", "⚠️ 感知服务未连接，跳过拍照")
                        continue
                    }

                    isAnalyzing = true

                    val outputDir = context.cacheDir
                    val photoFile = File(outputDir, "${System.currentTimeMillis()}.jpg")

                    val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

                    imageCapture.takePicture(
                        outputOptions,
                        ContextCompat.getMainExecutor(context),
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                                Log.e("CameraPage", "📸 拍照成功，图片已保存")
                                try {
                                    val imageBytes = photoFile.readBytes()
                                    Log.e("CameraPage", "📸 图片大小: ${imageBytes.size} bytes")
                                    onAnalyzeImage(imageBytes)
                                    photoFile.delete()
                                } catch (e: Exception) {
                                    Log.e("CameraPage", "读取图片失败", e)
                                } finally {
                                    isAnalyzing = false
                                }
                            }

                            override fun onError(exc: ImageCaptureException) {
                                Log.e("CameraPage", "❌ 拍照失败: ${exc.message}", exc)
                                isAnalyzing = false
                            }
                        }
                    )
                }
            }
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER

                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(surfaceProvider)
                    }

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            cameraSelector,
                            preview,
                            imageCapture
                        )
                    } catch (e: Exception) {
                        Log.e("CameraPage", "摄像头绑定失败", e)
                        activity.speak("摄像头启动失败")
                    }
                }
            },
            modifier = Modifier
                .fillMaxSize()
                .padding(bottom = 100.dp)
                .semantics {
                    contentDescription = if (isCameraOn) "摄像头预览中，检测障碍物" else "摄像头未开启"
                }
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null
                ) {
                    if (isCameraOn) {
                        onCameraToggle(false)
                        activity.speak("停止检测")
                    }
                }
        )

        if (detectionResult != null && isCameraOn) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 120.dp)
                    .background(
                        color = Color.Black.copy(alpha = 0.7f),
                        shape = RoundedCornerShape(16.dp)
                    )
                    .padding(horizontal = 20.dp, vertical = 12.dp)
            ) {
                Text(
                    text = detectionResult,
                    color = Color.White,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(140.dp)
                .align(Alignment.BottomCenter)
                .pointerInput(Unit) {
                    detectTapGestures { }
                }
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        brush = Brush.verticalGradient(
                            colors = listOf(
                                Color.Transparent,
                                Color.Black.copy(alpha = 0.8f),
                                Color.Black.copy(alpha = 0.95f)
                            )
                        )
                    )
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 32.dp)
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 20.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Bottom
            ) {
                CameraPageSmallButton(
                    icon = Icons.Default.Call,
                    text = "求助",
                    color = Color(0xFF1565C0),
                    onClick = {
                        activity.speak("切换到紧急求助页面")
                        activity.vibrateShort()
                        onPageChange("emergency")
                    },
                    contentDescription = "紧急求助按钮",
                    activity = activity
                )

                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier
                        .align(Alignment.Bottom)
                        .offset(y = (-20).dp)
                        .width(120.dp)
                        .semantics { contentDescription = "切换到导航页面按钮" }
                        .clickable {
                            activity.speak("切换到导航页面")
                            activity.vibrateShort()
                            onPageChange("main")
                        }
                ) {
                    Card(
                        modifier = Modifier
                            .size(100.dp)
                            .padding(bottom = 8.dp),
                        shape = CircleShape,
                        colors = CardDefaults.cardColors(
                            containerColor = Color(0xFF1565C0)
                        ),
                        elevation = CardDefaults.cardElevation(
                            defaultElevation = 6.dp
                        )
                    ) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.MyLocation,
                                contentDescription = null,
                                modifier = Modifier.size(44.dp),
                                tint = Color.White
                            )
                        }
                    }
                    Text(
                        text = "导航",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }

                CameraPageSmallButton(
                    icon = Icons.Default.Settings,
                    text = "设置",
                    color = Color(0xFF1565C0),
                    onClick = {
                        activity.speak("切换到设置页面")
                        activity.vibrateShort()
                        onPageChange("settings")
                    },
                    contentDescription = "设置按钮",
                    activity = activity
                )
            }
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 48.dp)
                .align(Alignment.TopCenter)
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "摄像头检测",
                    fontSize = 20.sp,
                    color = Color.Black,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                Text(
                    text = "当前模式：$currentMode",
                    fontSize = 14.sp,
                    color = Color.Black.copy(alpha = 0.7f)
                )
            }
        }


        Box(
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(end = 24.dp, bottom = 220.dp)
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier
                    .semantics { contentDescription = if (isFrontCamera) "切换到后置摄像头" else "切换到前置摄像头" }
                    .clickable {
                        isFrontCamera = !isFrontCamera
                        cameraSelector = if (isFrontCamera) {
                            CameraSelector.DEFAULT_FRONT_CAMERA
                        } else {
                            CameraSelector.DEFAULT_BACK_CAMERA
                        }

                        try {
                            val cameraProvider = cameraProviderFuture.get()
                            val preview = Preview.Builder().build()
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                cameraSelector,
                                preview,
                                imageCapture
                            )
                        } catch (e: Exception) {
                            Log.e("CameraPage", "切换摄像头失败", e)
                        }

                        activity.speak(if (isFrontCamera) "切换到前置摄像头" else "切换到后置摄像头")
                        activity.vibrateShort()
                    }
            ) {
                Card(
                    modifier = Modifier
                        .size(60.dp)
                        .padding(bottom = 6.dp),
                    shape = CircleShape,
                    colors = CardDefaults.cardColors(
                        containerColor = PrimaryBlue.copy(alpha = 0.9f)
                    ),
                    elevation = CardDefaults.cardElevation(
                        defaultElevation = 4.dp
                    )
                ) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = if (isFrontCamera) Icons.Default.Videocam else Icons.Default.Camera,
                            contentDescription = null,
                            modifier = Modifier.size(28.dp),
                            tint = Color.White
                        )
                    }
                }
                Text(
                    text = if (isFrontCamera) "前摄" else "后摄",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium,
                    color = Color.White
                )
            }
        }
    }

    if (showRouteMemoryDialog) {
        AlertDialog(
            onDismissRequest = {
                showRouteMemoryDialog = false
                activity.speak("取消保存路线")
            },
            title = {
                Text(text = "记忆路线", fontWeight = FontWeight.Bold)
            },
            text = {
                Column {
                    Text(text = "将当前检测路线保存为：")
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "$currentMode 检测路线",
                        color = PrimaryBlue,
                        fontWeight = FontWeight.Medium
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        showRouteMemoryDialog = false
                        activity.speak("检测路线已保存")
                        activity.vibrateShort()
                        Toast.makeText(activity, "检测路线已保存", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = PrimaryBlue
                    )
                ) {
                    Text("保存")
                }
            },
            dismissButton = {
                Button(
                    onClick = {
                        showRouteMemoryDialog = false
                        activity.speak("取消保存路线")
                    }
                ) {
                    Text("取消")
                }
            }
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }

    LaunchedEffect(Unit) {
        delay(500)
        if (isCameraOn) {
            // ✅ 根据感知服务连接状态播报，而不是 serverReachable
            if (activity.perceptionWebSocket?.connectionState?.value == PerceptionWebSocketService.ConnectionState.CONNECTED) {
                activity.speak("摄像头检测页面，检测中")
            } else {
                activity.speak("摄像头检测页面，感知服务未连接，正在重连")
                activity.perceptionWebSocket?.connect()
            }
        } else {
            activity.speak("摄像头检测页面，摄像头已关闭")
        }
    }
}

@Composable
fun EmergencyPage(
    onEmergencyCall: () -> Unit,
    onSendLocation: () -> Unit
) {
    val activity = LocalContext.current as MainActivity
    var showConfirmDialog by remember { mutableStateOf(false) }
    var longPressProgress by remember { mutableFloatStateOf(0f) }
    var isPressing by remember { mutableStateOf(false) }

    // 长按计时
    LaunchedEffect(isPressing) {
        if (isPressing) {
            var elapsed = 0f
            while (isPressing && elapsed < 3000f) {
                delay(50)
                elapsed += 50f
                longPressProgress = elapsed / 3000f
            }
            if (elapsed >= 3000f) {
                showConfirmDialog = true
                longPressProgress = 0f
            }
        } else {
            longPressProgress = 0f
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "紧急求助",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = PrimaryBlue,
            modifier = Modifier.padding(bottom = 40.dp)
        )

        // SOS 长按按钮
        Box(
            modifier = Modifier
                .size(140.dp)
                .background(
                    color = Color.Red.copy(alpha = 0.8f + 0.2f * longPressProgress),
                    shape = CircleShape
                )
                .pointerInput(Unit) {
                    detectTapGestures(
                        onPress = {
                            isPressing = true
                            tryAwaitRelease()
                            isPressing = false
                        }
                    )
                }
                .semantics {
                    contentDescription = "SOS紧急求助按钮，长按3秒触发"
                },
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    imageVector = Icons.Default.Call,
                    contentDescription = null,
                    modifier = Modifier.size(48.dp),
                    tint = Color.White
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "长按3秒",
                    fontSize = 14.sp,
                    color = Color.White
                )
                Text(
                    text = "SOS",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                if (longPressProgress > 0) {
                    Spacer(modifier = Modifier.height(4.dp))
                    LinearProgressIndicator(
                        progress = { longPressProgress },
                        modifier = Modifier
                            .width(80.dp)
                            .height(4.dp),
                        color = Color.White,
                        trackColor = Color.White.copy(alpha = 0.3f)
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "长按红色按钮3秒触发紧急求助",
            fontSize = 14.sp,
            color = Color.Gray,
            modifier = Modifier.padding(bottom = 32.dp)
        )

        // 发送位置按钮
        Button(
            onClick = onSendLocation,
            modifier = Modifier
                .fillMaxWidth(0.8f)
                .height(60.dp)
                .semantics { contentDescription = "发送当前位置给紧急联系人" },
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = PrimaryBlue
            )
        ) {
            Row(
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.LocationOn,
                    contentDescription = null,
                    modifier = Modifier.size(28.dp),
                    tint = Color.White
                )
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = "发送位置",
                    fontSize = 18.sp,
                    color = Color.White,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        Spacer(modifier = Modifier.height(60.dp))

        Text(
            text = "点击空白处返回",
            fontSize = 14.sp,
            color = Color.Gray,
            modifier = Modifier.padding(top = 40.dp)
        )
    }

    // 确认对话框
    if (showConfirmDialog) {
        AlertDialog(
            onDismissRequest = { showConfirmDialog = false },
            title = { Text("确认求助") },
            text = { Text("是否立即呼叫紧急联系人？") },
            confirmButton = {
                Button(
                    onClick = {
                        showConfirmDialog = false
                        onEmergencyCall()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = AccentRed)
                ) {
                    Text("呼叫")
                }
            },
            dismissButton = {
                Button(
                    onClick = { showConfirmDialog = false }
                ) {
                    Text("取消")
                }
            }
        )
    }

    LaunchedEffect(Unit) {
        delay(500)
        activity.speak("紧急求助页面，长按红色按钮3秒触发紧急呼叫")
    }
}

@Composable
fun NavigationPage(
    destination: TextFieldValue,
    onDestinationChange: (TextFieldValue) -> Unit,
    @Suppress("UNUSED_PARAMETER") isVoiceNavigationOn: Boolean,
    @Suppress("UNUSED_PARAMETER") onVoiceNavigationToggle: (Boolean) -> Unit,
    onNavigate: () -> Unit,
    onPageChange: (String) -> Unit,
    currentMode: String,
    onModeChange: (String) -> Unit,
    serverReachable: Boolean
) {
    val activity = LocalContext.current as MainActivity

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Top
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp, bottom = 8.dp, start = 16.dp, end = 16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(10.dp)
                            .background(
                                color = if (serverReachable) Color.Green else Color.Red,
                                shape = CircleShape
                            )
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = if (serverReachable) "在线" else "离线",
                        fontSize = 12.sp,
                        color = if (serverReachable) Color.Green else Color.Red
                    )
                }

//                Text(
//                    text = "12:46",
//                    fontSize = 16.sp,
//                    fontWeight = FontWeight.Bold,
//                    color = Color.Black
//                )

                Text(
                    text = "5G",
                    fontSize = 14.sp,
                    color = Color.Black,
                    fontWeight = FontWeight.Medium
                )
            }

            OutlinedTextField(
                value = destination,
                onValueChange = onDestinationChange,
                label = { Text("输入目的地") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp)
                    .semantics { contentDescription = "输入目的地" }
                    .focusable(true),
                placeholder = { Text("例如：人民公园、北京南站、天安门") },
                shape = RoundedCornerShape(12.dp)
            )

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(horizontal = 16.dp, vertical = 8.dp)
                    .semantics {
                        contentDescription = "地图预览区域，点击开始导航"
                    }
                    .clickable {
                        onNavigate()
                    },
                shape = RoundedCornerShape(16.dp),
                border = BorderStroke(2.dp, PrimaryBlue),
                colors = CardDefaults.cardColors(
                    containerColor = LightBlue
                ),
                elevation = CardDefaults.cardElevation(
                    defaultElevation = 8.dp
                )
            ) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .size(180.dp)
                                .background(Color.White, CircleShape)
                                .padding(24.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.LocationOn,
                                contentDescription = null,
                                modifier = Modifier.size(80.dp),
                                tint = PrimaryBlue
                            )
                        }
                        Spacer(modifier = Modifier.height(20.dp))

                        if (destination.text.isNotBlank()) {
                            Text(
                                text = "前往 ${destination.text}",
                                fontSize = 22.sp,
                                fontWeight = FontWeight.Bold,
                                color = PrimaryBlue,
                                textAlign = TextAlign.Center
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "点击地图开始导航",
                                fontSize = 16.sp,
                                color = Color.Gray,
                                textAlign = TextAlign.Center
                            )
                        } else {
                            Text(
                                text = "请输入目的地",
                                fontSize = 26.sp,
                                fontWeight = FontWeight.Bold,
                                color = PrimaryBlue,
                                textAlign = TextAlign.Center
                            )

                            Spacer(modifier = Modifier.height(12.dp))

                            Text(
                                text = "输入目的地后点击开始导航",
                                fontSize = 16.sp,
                                color = TextBlack,
                                textAlign = TextAlign.Center,
                                modifier = Modifier.padding(horizontal = 24.dp)
                            )

                            Spacer(modifier = Modifier.height(20.dp))

                            Text(
                                text = "点击地图开始导航",
                                fontSize = 16.sp,
                                color = Color.Gray,
                                textAlign = TextAlign.Center
                            )
                        }
                    }

                    Column(
                        modifier = Modifier
                            .align(Alignment.CenterEnd)
                            .padding(end = 16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(18.dp)
                    ) {
                        Card(
                            modifier = Modifier
                                .size(50.dp)
                                .semantics { contentDescription = "切换到障碍物检测模式" }
                                .clickable {
                                    activity.speak("切换到摄像头检测页面，障碍物检测模式")
                                    activity.vibrateShort()
                                    onModeChange("障碍物检测")
                                    onPageChange("camera")
                                },
                            shape = CircleShape,
                            colors = CardDefaults.cardColors(
                                containerColor = if (currentMode == "障碍物检测") PrimaryBlue else Color.LightGray
                            ),
                            border = if (currentMode == "障碍物检测") BorderStroke(2.dp, PrimaryBlue) else BorderStroke(0.dp, Color.Transparent),
                            elevation = CardDefaults.cardElevation(
                                defaultElevation = if (currentMode == "障碍物检测") 4.dp else 2.dp
                            )
                        ) {
                            Box(
                                modifier = Modifier.fillMaxSize(),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.Warning,
                                    contentDescription = null,
                                    modifier = Modifier.size(26.dp),
                                    tint = if (currentMode == "障碍物检测") Color.White else Color.Gray
                                )
                            }
                        }

                        Card(
                            modifier = Modifier
                                .size(50.dp)
                                .semantics { contentDescription = "切换到环境检测模式" }
                                .clickable {
                                    activity.speak("切换到摄像头检测页面，环境检测模式")
                                    activity.vibrateShort()
                                    onModeChange("环境检测")
                                    onPageChange("camera")
                                },
                            shape = CircleShape,
                            colors = CardDefaults.cardColors(
                                containerColor = if (currentMode == "环境检测") PrimaryBlue else Color.LightGray
                            ),
                            border = if (currentMode == "环境检测") BorderStroke(2.dp, PrimaryBlue) else BorderStroke(0.dp, Color.Transparent),
                            elevation = CardDefaults.cardElevation(
                                defaultElevation = if (currentMode == "环境检测") 4.dp else 2.dp
                            )
                        ) {
                            Box(
                                modifier = Modifier.fillMaxSize(),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.EmojiObjects,
                                    contentDescription = null,
                                    modifier = Modifier.size(26.dp),
                                    tint = if (currentMode == "环境检测") Color.White else Color.Gray
                                )
                            }
                        }

                        Card(
                            modifier = Modifier
                                .size(50.dp)
                                .semantics { contentDescription = "切换到关怀模式" }
                                .clickable {
                                    activity.speak("切换到摄像头检测页面，关怀模式")
                                    activity.vibrateShort()
                                    onModeChange("关怀模式")
                                    onPageChange("camera")
                                },
                            shape = CircleShape,
                            colors = CardDefaults.cardColors(
                                containerColor = if (currentMode == "关怀模式") PrimaryBlue else Color.LightGray
                            ),
                            border = if (currentMode == "关怀模式") BorderStroke(2.dp, PrimaryBlue) else BorderStroke(0.dp, Color.Transparent),
                            elevation = CardDefaults.cardElevation(
                                defaultElevation = if (currentMode == "关怀模式") 4.dp else 2.dp
                            )
                        ) {
                            Box(
                                modifier = Modifier.fillMaxSize(),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.Favorite,
                                    contentDescription = null,
                                    modifier = Modifier.size(26.dp),
                                    tint = if (currentMode == "关怀模式") Color.White else Color.Gray
                                )
                            }
                        }
                    }
                }
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(140.dp)
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier
                            .align(Alignment.CenterStart)
                            .padding(top = 20.dp)
                            .width(100.dp)
                            .semantics { contentDescription = "紧急求助按钮" }
                            .clickable {
                                activity.speak("切换到紧急求助页面")
                                activity.vibrateShort()
                                onPageChange("emergency")
                            }
                    ) {
                        Card(
                            modifier = Modifier
                                .size(70.dp)
                                .padding(bottom = 8.dp),
                            shape = CircleShape,
                            colors = CardDefaults.cardColors(
                                containerColor = Color(0xFF1565C0)
                            ),
                            elevation = CardDefaults.cardElevation(
                                defaultElevation = 4.dp
                            )
                        ) {
                            Box(
                                modifier = Modifier.fillMaxSize(),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Call,
                                    contentDescription = null,
                                    modifier = Modifier.size(32.dp),
                                    tint = Color.White
                                )
                            }
                        }
                        Text(
                            text = "紧急求助",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Medium,
                            color = PrimaryBlue
                        )
                    }

                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier
                            .align(Alignment.Center)
                            .padding(bottom = 20.dp)
                            .width(120.dp)
                            .semantics { contentDescription = "摄像头检测按钮" }
                            .clickable {
                                activity.speak("切换到摄像头检测页面")
                                activity.vibrateShort()
                                onPageChange("camera")
                            }
                    ) {
                        Card(
                            modifier = Modifier
                                .size(100.dp)
                                .padding(bottom = 8.dp),
                            shape = CircleShape,
                            colors = CardDefaults.cardColors(
                                containerColor = Color(0xFF1565C0)
                            ),
                            elevation = CardDefaults.cardElevation(
                                defaultElevation = 6.dp
                            )
                        ) {
                            Box(
                                modifier = Modifier.fillMaxSize(),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Videocam,
                                    contentDescription = null,
                                    modifier = Modifier.size(44.dp),
                                    tint = Color.White
                                )
                            }
                        }
                        Text(
                            text = "摄像头",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            color = PrimaryBlue
                        )
                    }

                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier
                            .align(Alignment.CenterEnd)
                            .padding(top = 20.dp)
                            .width(100.dp)
                            .semantics { contentDescription = "设置按钮" }
                            .clickable {
                                activity.speak("切换到设置页面")
                                activity.vibrateShort()
                                onPageChange("settings")
                            }
                    ) {
                        Card(
                            modifier = Modifier
                                .size(70.dp)
                                .padding(bottom = 8.dp),
                            shape = CircleShape,
                            colors = CardDefaults.cardColors(
                                containerColor = Color(0xFF1565C0)
                            ),
                            elevation = CardDefaults.cardElevation(
                                defaultElevation = 4.dp
                            )
                        ) {
                            Box(
                                modifier = Modifier.fillMaxSize(),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Settings,
                                    contentDescription = null,
                                    modifier = Modifier.size(32.dp),
                                    tint = Color.White
                                )
                            }
                        }
                        Text(
                            text = "设置",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Medium,
                            color = PrimaryBlue
                        )
                    }
                }
            }
        }

        LaunchedEffect(destination) {
            delay(500)
            if (destination.text.isNotBlank()) {
                activity.speak("导航页面，前往${destination.text}")
            } else {
                activity.speak("导航页面，请输入目的地")
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsPage(
    onPageChange: (String) -> Unit,
    onLogout: () -> Unit,
    voicePromptMode: String,
    onVoicePromptModeChange: (String) -> Unit,
    serverReachable: Boolean,
    onTestConnection: () -> Unit
) {
    val activity = LocalContext.current as MainActivity
    var showLogoutDialog by remember { mutableStateOf(false) }
    var showUserInfoDialog by remember { mutableStateOf(false) }
    var showEditUserDialog by remember { mutableStateOf(false) }
    var showStorageDialog by remember { mutableStateOf(false) }
    var storageSize by remember { mutableStateOf("计算中...") }
    var isUpdating by remember { mutableStateOf(false) }
    var userData by remember { mutableStateOf<UserData?>(null) }

    // 获取当前用户 - 使用 activity.appState.currentUser
    val currentUser = activity.appState.currentUser

    // 加载用户信息和存储统计
    LaunchedEffect(Unit) {
        if (serverReachable && currentUser != null) {
            // 加载用户信息
            val userInfoResult = activity.networkService.getUserInfo(currentUser.id)
            if (userInfoResult.isSuccess) {
                val response = userInfoResult.getOrNull()
                if (response?.success == true) {
                    userData = response.user
                }
            }

            // 加载存储统计
            val storageResult = activity.networkService.getStorageStats(currentUser.id)
            if (storageResult.isSuccess) {
                val response = storageResult.getOrNull()
                if (response?.success == true) {
                    val stats = response.data
                    if (stats != null) {
                        storageSize = formatSize(stats.totalSize)
                    }
                }
            }
        } else if (currentUser != null) {
            // 离线模式使用本地计算
            val size = StorageManager.getTotalStorageSize(activity)
            storageSize = StorageManager.formatSize(size)
            // 使用本地用户数据
            userData = UserData(
                id = currentUser.id,
                username = currentUser.username,
                nickname = currentUser.nickname,
                phone = currentUser.phone,
                email = currentUser.email,
                emergencyContact = currentUser.emergencyContact,
                registerTime = currentUser.registerTime,
                lastLoginTime = currentUser.lastLoginTime,
                avatar = currentUser.avatar
            )
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White),
        horizontalAlignment = Alignment.Start,
        verticalArrangement = Arrangement.Top
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp)
                .background(PrimaryBlue)
                .padding(start = 16.dp),
            contentAlignment = Alignment.CenterStart
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(12.dp)
                        .background(
                            color = if (serverReachable) Color.Green else Color.Red,
                            shape = CircleShape
                        )
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (serverReachable) "服务器已连接" else "服务器未连接",
                    fontSize = 14.sp,
                    color = Color.White
                )
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp)
                .padding(start = 16.dp)
                .clickable {
                    activity.speak("返回导航页面")
                    activity.vibrateShort()
                    onPageChange("main")
                },
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Start
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "返回",
                modifier = Modifier
                    .size(24.dp)
                    .padding(end = 12.dp),
                tint = Color.Black
            )
            Text(
                text = "返回",
                fontSize = 18.sp,
                fontWeight = FontWeight.Medium,
                color = Color.Black,
                modifier = Modifier.semantics { contentDescription = "返回按钮，返回导航页面" }
            )
        }

        Button(
            onClick = {
                activity.vibrateShort()
                onTestConnection()
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp)
                .height(48.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (serverReachable) Color.Green else Color.Red
            )
        ) {
            Text(
                text = if (serverReachable) "服务器已连接 - 点击测试" else "服务器未连接 - 点击测试",
                color = Color.White
            )
        }

        Text(
            text = "语音提示设置",
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = Color.Black,
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 16.dp, top = 8.dp, bottom = 8.dp)
        )

        SettingItem(
            title = "语音提示模式",
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 16.dp, end = 16.dp)
                .clickable {
                    activity.speak("语音提示模式")
                    activity.vibrateShort()
                }
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 32.dp, top = 8.dp, bottom = 16.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Box(modifier = Modifier.weight(1f)) {
                        VoiceModeButton(
                            title = "详细",
                            isSelected = voicePromptMode == "详细",
                            onClick = { onVoicePromptModeChange("详细") }
                        )
                    }
                    Box(modifier = Modifier.weight(1f)) {
                        VoiceModeButton(
                            title = "简洁",
                            isSelected = voicePromptMode == "简洁",
                            onClick = { onVoicePromptModeChange("简洁") }
                        )
                    }
                    Box(modifier = Modifier.weight(1f)) {
                        VoiceModeButton(
                            title = "静默",
                            isSelected = voicePromptMode == "静默",
                            onClick = { onVoicePromptModeChange("静默") }
                        )
                    }
                }
            }
        }

        Text(
            text = "账户信息",
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = Color.Black,
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 16.dp, top = 24.dp, bottom = 8.dp)
        )

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 16.dp, end = 16.dp),
            verticalArrangement = Arrangement.spacedBy(1.dp)
        ) {
            // 用户信息按钮
            SettingItem(
                title = "用户信息",
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        activity.speak("查看用户信息")
                        activity.vibrateShort()
                        showUserInfoDialog = true
                    }
            ) {
                Text(
                    text = currentUser?.username ?: "未登录",
                    fontSize = 12.sp,
                    color = Color.Gray,
                    modifier = Modifier.padding(start = 32.dp, top = 2.dp, bottom = 8.dp)
                )
            }

            // 编辑资料按钮
            SettingItem(
                title = "编辑资料",
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        activity.speak("编辑个人资料")
                        activity.vibrateShort()
                        showEditUserDialog = true
                    }
            )

            // 存储数据按钮
            SettingItem(
                title = "存储数据",
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        activity.speak("存储数据管理")
                        activity.vibrateShort()
                        showStorageDialog = true
                    }
            ) {
                Text(
                    text = "已用空间：$storageSize",
                    fontSize = 12.sp,
                    color = Color.Gray,
                    modifier = Modifier.padding(start = 32.dp, top = 2.dp, bottom = 8.dp)
                )
            }

            // 退出登录按钮
            SettingItem(
                title = "退出登录",
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        activity.speak("确认退出登录")
                        activity.vibrateLong()
                        showLogoutDialog = true
                    }
            )
        }

        LaunchedEffect(Unit) {
            delay(500)
            activity.speak("设置页面，可设置语音提示模式和查看用户信息")
        }
    }

    // 用户信息查看对话框
    if (showUserInfoDialog && userData != null) {
        UserInfoDialog(
            userData = userData!!,
            onEdit = {
                showUserInfoDialog = false
                showEditUserDialog = true
            },
            onDismiss = { showUserInfoDialog = false }
        )
    }

    // 编辑用户信息对话框
    if (showEditUserDialog && currentUser != null) {
        EditUserInfoDialog(
            user = currentUser,
            isSaving = isUpdating,
            onSave = { request ->
                CoroutineScope(Dispatchers.Main).launch {
                    isUpdating = true
                    activity.speak("正在保存")

                    val result = activity.networkService.updateUserInfo(currentUser.id, request)

                    withContext(Dispatchers.Main) {
                        isUpdating = false
                        if (result.isSuccess) {
                            val response = result.getOrNull()
                            if (response?.success == true) {
                                // 更新本地用户信息
                                response.user?.let { userDataFromResponse ->
                                    val updatedUser = currentUser.copy(
                                        nickname = userDataFromResponse.nickname,
                                        phone = userDataFromResponse.phone,
                                        email = userDataFromResponse.email,
                                        emergencyContact = userDataFromResponse.emergencyContact
                                    )
                                    LocalAuthManager.updateUser(updatedUser)
                                    // 使用公开方法更新 appState
                                    activity.updateAppState { state ->
                                        state.copy(currentUser = updatedUser)
                                    }
                                    activity.speak("保存成功")
                                    // 重新加载用户信息
                                    val newUserInfo = activity.networkService.getUserInfo(currentUser.id)
                                    if (newUserInfo.isSuccess) {
                                        val newResponse = newUserInfo.getOrNull()
                                        userData = newResponse?.user
                                    }
                                }
                                showEditUserDialog = false
                                showUserInfoDialog = true
                            } else {
                                activity.speak(response?.message ?: "保存失败")
                            }
                        } else {
                            activity.speak("网络错误，保存失败")
                        }
                    }
                }
            },
            onDismiss = { showEditUserDialog = false }
        )
    }

    // 存储数据对话框 - 修复未使用的变量
    if (showStorageDialog && currentUser != null) {
        StorageDataDialog(
            storageSize = storageSize,
            onClearCache = {
                CoroutineScope(Dispatchers.Main).launch {
                    activity.speak("正在清理缓存")
                    if (serverReachable) {
                        val result = activity.networkService.clearCache(currentUser.id)
                        withContext(Dispatchers.Main) {
                            if (result.isSuccess) {
                                // 不需要 response 变量
                                activity.speak("缓存清理成功")
                                // 重新加载存储统计
                                val newStats = activity.networkService.getStorageStats(currentUser.id)
                                if (newStats.isSuccess) {
                                    val statsResponse = newStats.getOrNull()
                                    if (statsResponse?.success == true) {
                                        val stats = statsResponse.data
                                        if (stats != null) {
                                            storageSize = formatSize(stats.totalSize)
                                        }
                                    }
                                }
                            } else {
                                activity.speak("缓存清理失败")
                            }
                        }
                    } else {
                        // 离线模式使用本地清理
                        if (StorageManager.clearCache(activity)) {
                            activity.speak("缓存清理成功")
                            val newSize = StorageManager.getTotalStorageSize(activity)
                            storageSize = StorageManager.formatSize(newSize)
                        } else {
                            activity.speak("缓存清理失败")
                        }
                    }
                }
            },
            onClearAllData = {
                CoroutineScope(Dispatchers.Main).launch {
                    activity.speak("正在清除所有数据")
                    if (serverReachable) {
                        val result = activity.networkService.clearAllUserData(currentUser.id)
                        withContext(Dispatchers.Main) {
                            if (result.isSuccess) {
                                activity.speak("数据清除成功，即将退出登录")
                                onLogout()
                            } else {
                                activity.speak("数据清除失败")
                            }
                        }
                    } else {
                        // 离线模式使用本地清理
                        if (StorageManager.clearAllData(activity)) {
                            activity.speak("数据清除成功，即将退出登录")
                            onLogout()
                        } else {
                            activity.speak("数据清除失败")
                        }
                    }
                }
            },
            onDismiss = { showStorageDialog = false }
        )
    }

    if (showLogoutDialog) {
        AlertDialog(
            onDismissRequest = {
                showLogoutDialog = false
                activity.speak("取消退出登录")
            },
            title = { Text("退出登录") },
            text = { Text("确定要退出登录吗？") },
            confirmButton = {
                Button(onClick = {
                    showLogoutDialog = false
                    activity.speak("正在退出登录")
                    onLogout()
                }) {
                    Text("确定")
                }
            },
            dismissButton = {
                Button(onClick = {
                    showLogoutDialog = false
                    activity.speak("取消退出登录")
                }) {
                    Text("取消")
                }
            }
        )
    }
}

// 辅助函数：格式化文件大小
private fun formatSize(size: Long): String {
    return when {
        size < 1024 -> "$size B"
        size < 1024 * 1024 -> String.format("%.2f KB", size / 1024.0)
        size < 1024 * 1024 * 1024 -> String.format("%.2f MB", size / (1024.0 * 1024))
        else -> String.format("%.2f GB", size / (1024.0 * 1024 * 1024))
    }
}

@Suppress("unused")
@Composable
fun SettingSubItem(
    title: String,
    isSelected: Boolean = false,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(40.dp)
            .clickable(onClick = onClick)
            .background(
                if (isSelected) PrimaryBlue.copy(alpha = 0.1f)
                else Color.Transparent
            )
            .padding(start = 16.dp, end = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Start
    ) {
        Text(
            text = title,
            fontSize = 14.sp,
            color = if (isSelected) PrimaryBlue else Color.Gray,
            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
            modifier = Modifier.semantics { contentDescription = title }
        )
    }
}

@Composable
fun VoiceModeButton(
    title: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val activity = LocalContext.current as MainActivity

    Button(
        onClick = {
            onClick()
            activity.vibrateShort()
        },
        modifier = Modifier
            .fillMaxWidth()
            .height(40.dp)
            .semantics { contentDescription = "${title}语音提示模式" },
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isSelected) PrimaryBlue else Color.LightGray
        ),
        shape = RoundedCornerShape(8.dp)
    ) {
        Text(
            text = title,
            fontSize = 14.sp,
            color = if (isSelected) Color.White else Color.Black
        )
    }
}

@Composable
fun SettingItem(
    title: String,
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null,
    content: @Composable (() -> Unit)? = null
) {
    Column(modifier = modifier) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp)
                .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Start
        ) {
            Text(
                text = title,
                fontSize = 16.sp,
                color = Color.Black,
                modifier = Modifier.semantics { contentDescription = title }
            )
        }
        content?.invoke()
    }
}

@Composable
fun CameraPageSmallButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    text: String,
    color: Color,
    onClick: () -> Unit,
    contentDescription: String,
    activity: MainActivity
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .width(64.dp)
            .semantics { this.contentDescription = contentDescription }
            .clickable {
                onClick()
                activity.vibrateShort()
            }
    ) {
        Card(
            modifier = Modifier
                .size(56.dp)
                .padding(bottom = 6.dp),
            shape = CircleShape,
            colors = CardDefaults.cardColors(
                containerColor = color
            ),
            elevation = CardDefaults.cardElevation(
                defaultElevation = 4.dp
            )
        ) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    modifier = Modifier.size(26.dp),
                    tint = Color.White
                )
            }
        }
        Text(
            text = text,
            fontSize = 13.sp,
            color = Color.White,
            fontWeight = FontWeight.Medium
        )
    }
}