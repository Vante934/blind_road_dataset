// 文件路径：app/src/main/java/com/yourcompany/blindnavigation/ui/components/StorageDataDialog.kt
package com.yourcompany.blindnavigation.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.yourcompany.blindnavigation.ui.theme.AccentRed
import com.yourcompany.blindnavigation.ui.theme.PrimaryBlue
import kotlinx.coroutines.launch

@Composable
fun StorageDataDialog(
    storageSize: String,
    onClearCache: () -> Unit,
    onClearAllData: () -> Unit,
    onDismiss: () -> Unit
) {
    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            shape = MaterialTheme.shapes.medium
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    text = "存储数据管理",
                    fontSize = 20.sp,
                    style = MaterialTheme.typography.titleLarge
                )

                Text("已用空间：$storageSize")

                Divider()

                Text(
                    text = "缓存数据包括：",
                    fontSize = 14.sp,
                    style = MaterialTheme.typography.bodyMedium
                )
                Text("• 摄像头拍摄的临时图片", fontSize = 12.sp, color = Color.Gray)
                Text("• 语音合成缓存", fontSize = 12.sp, color = Color.Gray)
                Text("• 应用临时文件", fontSize = 12.sp, color = Color.Gray)

                Divider()

                Text(
                    text = "⚠️ 清除所有数据将删除您的账号信息和所有设置，请谨慎操作！",
                    fontSize = 12.sp,
                    color = AccentRed
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Button(
                        onClick = onClearCache,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = PrimaryBlue
                        )
                    ) {
                        Text("清除缓存")
                    }

                    Button(
                        onClick = onClearAllData,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = AccentRed
                        )
                    ) {
                        Text("清除所有数据")
                    }

                    Button(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("取消")
                    }
                }
            }
        }
    }
}