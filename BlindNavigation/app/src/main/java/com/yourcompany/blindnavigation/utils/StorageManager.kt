// 文件路径：app/src/main/java/com/yourcompany/blindnavigation/utils/StorageManager.kt
package com.yourcompany.blindnavigation.utils

import android.content.Context
import android.util.Log
import java.io.File

object StorageManager {

    private const val TAG = "StorageManager"

    fun getTotalStorageSize(context: Context): Long {
        return try {
            val appFile = context.filesDir
            getFolderSize(appFile) + getFolderSize(context.cacheDir)
        } catch (e: Exception) {
            Log.e(TAG, "获取存储大小失败", e)
            0L
        }
    }

    private fun getFolderSize(file: File): Long {
        var size = 0L
        try {
            val files = file.listFiles()
            if (files != null) {
                for (f in files) {
                    size += if (f.isDirectory) {
                        getFolderSize(f)
                    } else {
                        f.length()
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "获取文件夹大小失败", e)
        }
        return size
    }

    fun clearCache(context: Context): Boolean {
        return try {
            clearFolder(context.cacheDir)
            true
        } catch (e: Exception) {
            Log.e(TAG, "清除缓存失败", e)
            false
        }
    }

    private fun clearFolder(file: File) {
        if (file.isDirectory) {
            val files = file.listFiles()
            if (files != null) {
                for (f in files) {
                    if (f.isDirectory) {
                        clearFolder(f)
                        f.delete()
                    } else {
                        f.delete()
                    }
                }
            }
        } else {
            file.delete()
        }
    }

    fun clearAllData(context: Context): Boolean {
        return try {
            clearFolder(context.cacheDir)
            clearFolder(context.filesDir)
            context.getSharedPreferences("auth_prefs", Context.MODE_PRIVATE)
                .edit()
                .clear()
                .apply()
            true
        } catch (e: Exception) {
            Log.e(TAG, "清除所有数据失败", e)
            false
        }
    }

    fun formatSize(size: Long): String {
        return when {
            size < 1024 -> "$size B"
            size < 1024 * 1024 -> String.format("%.2f KB", size / 1024.0)
            size < 1024 * 1024 * 1024 -> String.format("%.2f MB", size / (1024.0 * 1024))
            else -> String.format("%.2f GB", size / (1024.0 * 1024 * 1024))
        }
    }
}