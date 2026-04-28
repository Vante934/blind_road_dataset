package com.yourcompany.blindnavigation.network

object ApiConfig {

    // =====================================
    // 感知服务（A成员）- 障碍物检测、预警
    // 公网地址：18c03663.r22.cpolar.top
    // =====================================
    object PerceptionService {
        // ✅ 使用最新的 cpolar 公网地址
        const val HOST = "516d7faf.r33.cpolar.top"
        val BASE_URL get() = "http://$HOST"
        val WS_URL get() = "ws://$HOST"
        const val WS_NAVIGATION = "/ws"
    }

    // =====================================
    // 导航服务（B成员）- 路径规划、GPS平滑、用户系统、SOS
    // 公网地址：11a70393.r15.vip.cpolar.cn
    // =====================================
    object NavigationService {
        const val HOST = "6711bfa5.r21.cpolar.top"
        val BASE_URL get() = "http://$HOST"
        val WS_URL get() = "ws://$HOST"
        const val WS_NAVIGATION = "/ws/nav"

        // HTTP API 端点
        const val HEALTH = "/api/health"
        const val LOGIN = "/api/auth/login"
        const val REGISTER = "/api/auth/register"
        const val PLAN_ROUTE = "/api/navigation/route"
        const val DETECT_OBSTACLE = "/api/detection/analyze"

        // 用户信息相关
        const val USER_INFO = "/api/user/info"
        const val UPDATE_USER_INFO = "/api/user/update"
        const val STORAGE_STATS = "/api/user/storage"
        const val CLEAR_CACHE = "/api/user/clear-cache"
        const val CLEAR_ALL_DATA = "/api/user/clear-all"
    }

    // =====================================
    // 统一超时配置
    // =====================================
    const val CONNECT_TIMEOUT = 15L
    const val READ_TIMEOUT = 15L
    const val WRITE_TIMEOUT = 15L
    const val WS_RECONNECT_DELAY = 3000L
}