package com.yourcompany.blindnavigation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.clickable
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.layout.Arrangement
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.yourcompany.blindnavigation.utils.*

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                val navController = rememberNavController()
                AppNavigation(navController = navController)
            }
        }
    }
}

// 定义黄黑配色方案
object YellowBlackTheme {
    val PrimaryYellow = Color(0xFFFFD700)  // 主黄色
    val DarkYellow = Color(0xFFFFC400)     // 深黄色（按下状态）
    val BackgroundBlack = Color(0xFF121212) // 背景黑色
    val SurfaceBlack = Color(0xFF1E1E1E)   // 表面黑色
    val TextWhite = Color(0xFFFFFFFF)      // 文字白色
    val TextYellow = Color(0xFFFFD700)     // 文字黄色
}

@Composable
fun AppNavigation(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = "main"
    ) {
        composable("main") {
            MainScreen(navController = navController)
        }
        composable("navigation") {
            FeatureScreen("智能导航", navController)
        }
        composable("obstacle") {
            FeatureScreen("障碍检测", navController)
        }
        composable("location") {
            FeatureScreen("当前位置", navController)
        }
        composable("transport") {
            FeatureScreen("公共交通", navController)
        }
        composable("assistance") {
            FeatureScreen("紧急求助", navController)
        }
        composable("settings") {
            FeatureScreen("系统设置", navController)
        }
    }
}

@Composable
fun MainScreen(navController: NavHostController) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(YellowBlackTheme.BackgroundBlack)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // 应用标题 - 黄色文字
            Text(
                text = "盲人导航助手",
                color = YellowBlackTheme.PrimaryYellow,
                fontSize = 36.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 40.dp)
            )

            val context = LocalContext.current

            // 第一行按钮
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                FeatureButton(
                    title = "智能导航",
                    navController = navController,
                    destination = "navigation",
                    context = context,
                    vibrationType = VibrationType.NAVIGATION
                )
                Spacer(modifier = Modifier.padding(12.dp))
                FeatureButton(
                    title = "障碍检测",
                    navController = navController,
                    destination = "obstacle",
                    context = context,
                    vibrationType = VibrationType.OBSTACLE
                )
            }

            Spacer(modifier = Modifier.height(20.dp))

            // 第二行按钮
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                FeatureButton(
                    title = "当前位置",
                    navController = navController,
                    destination = "location",
                    context = context,
                    vibrationType = VibrationType.LOCATION
                )
                Spacer(modifier = Modifier.padding(12.dp))
                FeatureButton(
                    title = "公共交通",
                    navController = navController,
                    destination = "transport",
                    context = context,
                    vibrationType = VibrationType.TRANSPORT
                )
            }

            Spacer(modifier = Modifier.height(20.dp))

            // 第三行按钮
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                FeatureButton(
                    title = "紧急求助",
                    navController = navController,
                    destination = "assistance",
                    context = context,
                    vibrationType = VibrationType.ASSISTANCE
                )
                Spacer(modifier = Modifier.padding(12.dp))
                FeatureButton(
                    title = "系统设置",
                    navController = navController,
                    destination = "settings",
                    context = context,
                    vibrationType = VibrationType.SETTINGS
                )
            }
        }
    }
}

@Composable
fun FeatureButton(
    title: String,
    navController: NavHostController,
    destination: String,
    context: android.content.Context,
    vibrationType: VibrationType
) {
    Surface(
        modifier = Modifier
            .height(140.dp)
            .fillMaxWidth(0.48f),
        color = YellowBlackTheme.SurfaceBlack, // 黑色按钮背景
        shape = MaterialTheme.shapes.medium,
        onClick = {
            VoiceFeedbackManager.provideFeedback(context, text = title, vibrationType = vibrationType)
            navController.navigate(destination)
        }
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .fillMaxSize()
                .background(
                    color = YellowBlackTheme.SurfaceBlack,
                    shape = MaterialTheme.shapes.medium
                )
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                // 按钮文字 - 黄色文字
                Text(
                    text = title,
                    color = YellowBlackTheme.PrimaryYellow, // 黄色文字
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                    lineHeight = 24.sp
                )
                // 添加黄色装饰线
                Spacer(modifier = Modifier.height(12.dp))
                Box(
                    modifier = Modifier
                        .height(3.dp)
                        .width(50.dp)
                        .background(YellowBlackTheme.PrimaryYellow) // 黄色装饰线
                )
            }
        }
    }
}

@Composable
fun FeatureScreen(
    title: String,
    navController: NavHostController
) {
    val context = LocalContext.current
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(YellowBlackTheme.BackgroundBlack) // 黑色背景
            .clickable {
                VoiceFeedbackManager.provideFeedback(context, text = "返回主界面", vibrationType = VibrationType.NAVIGATION)
                navController.popBackStack()
            },
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // 主标题 - 黄色文字
            Text(
                text = title,
                color = YellowBlackTheme.PrimaryYellow,
                fontSize = 38.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )

            // 黄色装饰性分隔线
            Spacer(modifier = Modifier.height(20.dp))
            Box(
                modifier = Modifier
                    .height(4.dp)
                    .width(140.dp)
                    .background(YellowBlackTheme.PrimaryYellow) // 黄色分隔线
            )

            // 返回提示 - 白色文字
            Spacer(modifier = Modifier.height(28.dp))
            Text(
                text = "点击屏幕返回主界面",
                color = YellowBlackTheme.TextWhite.copy(alpha = 0.8f),
                fontSize = 18.sp,
                fontWeight = FontWeight.Medium
            )
        }
    }
}