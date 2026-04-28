// 文件路径：app/src/main/java/com/yourcompany/blindnavigation/ui/components/EditUserInfoDialog.kt
package com.yourcompany.blindnavigation.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.yourcompany.blindnavigation.models.UpdateUserRequest
import com.yourcompany.blindnavigation.models.User
import kotlinx.coroutines.launch

@Composable
fun EditUserInfoDialog(
    user: User,
    onSave: (UpdateUserRequest) -> Unit,
    onDismiss: () -> Unit,
    isSaving: Boolean = false
) {
    var nickname by remember { mutableStateOf(user.nickname ?: "") }
    var phone by remember { mutableStateOf(user.phone ?: "") }
    var email by remember { mutableStateOf(user.email ?: "") }
    var emergencyContact by remember { mutableStateOf(user.emergencyContact ?: "") }

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
                    .padding(24.dp)
                    .verticalScroll(rememberScrollState()),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "编辑个人信息",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(bottom = 20.dp)
                )

                OutlinedTextField(
                    value = nickname,
                    onValueChange = { nickname = it },
                    label = { Text("昵称") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(12.dp))

                OutlinedTextField(
                    value = phone,
                    onValueChange = { phone = it },
                    label = { Text("手机号") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(12.dp))

                OutlinedTextField(
                    value = email,
                    onValueChange = { email = it },
                    label = { Text("邮箱") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(12.dp))

                OutlinedTextField(
                    value = emergencyContact,
                    onValueChange = { emergencyContact = it },
                    label = { Text("紧急联系人") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(24.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Button(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f),
                        enabled = !isSaving
                    ) {
                        Text("取消")
                    }

                    Button(
                        onClick = {
                            val request = UpdateUserRequest(
                                nickname = nickname.ifBlank { null },
                                phone = phone.ifBlank { null },
                                email = email.ifBlank { null },
                                emergencyContact = emergencyContact.ifBlank { null }
                            )
                            onSave(request)
                        },
                        modifier = Modifier.weight(1f),
                        enabled = !isSaving,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary
                        )
                    ) {
                        Text(if (isSaving) "保存中..." else "保存")
                    }
                }
            }
        }
    }
}