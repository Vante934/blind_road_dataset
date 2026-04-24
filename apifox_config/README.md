# Apifox配置文件夹

这个文件夹包含用于Apifox前后端联调的所有配置文件。

## 📁 文件说明

- `blind_road_api.json` - OpenAPI规范文件（自动生成）
- `apifox_config.json` - Apifox项目配置
- `README.md` - 本说明文档

## 🚀 快速开始

### 步骤1：导出OpenAPI规范

运行导出脚本：
```bash
python apifox_config/export_openapi.py
```

这将自动：
- 连接本地服务器
- 下载OpenAPI规范
- 生成配置文件

### 步骤2：导入到Apifox

1. 打开Apifox应用
2. 创建新项目："盲道检测系统"
3. 点击【导入】→【OpenAPI】
4. 选择 `blind_road_api.json` 文件
5. 等待导入完成

### 步骤3：配置环境

在Apifox中创建以下环境：

#### 开发环境
```
base_url: http://localhost:8082
api_version: v1
device_id: test_device_001
```

#### 测试环境
```
base_url: http://192.168.1.100:8082
api_version: v1
device_id: test_device_002
```

#### 生产环境
```
base_url: https://api.yourdomain.com
api_version: v1
```

### 步骤4：开始测试

#### 测试系统状态
```
GET /avatars/status
GET /api/v1/system/status
```

#### 测试图像检测
```
POST /api/v1/detection/analyze
```

在Apifox中：
1. 上传测试图片
2. 自动转换为base64
3. 发送请求
4. 查看响应

## 📋 核心测试场景

### 场景1：完整检测流程

1. **设备注册**
   - POST `/api/v1/android/register`
   
2. **图像检测**
   - POST `/api/v1/detection/analyze`
   
3. **语音播报**
   - POST `/api/v1/voice/synthesize`
   
4. **数据上传**
   - POST `/api/v1/android/data/upload`

### 场景2：异常处理测试

1. **无效图片**
   - 上传损坏的图片
   - 验证错误响应
   
2. **网络超时**
   - 模拟网络中断
   - 测试重试机制

## 🔍 使用技巧

### 1. 自动生成测试数据
在Apifox中可以为请求自动生成测试数据：
- 随机字符串
- 随机数字
- 时间戳等

### 2. 前置脚本
使用前置脚本自动化测试：
```javascript
// 生成随机设备ID
pm.environment.set("device_id", "device_" + Math.random().toString(36).substr(2, 9));

// 设置时间戳
pm.environment.set("timestamp", new Date().toISOString());
```

### 3. 后置脚本
验证响应数据：
```javascript
// 验证响应格式
pm.test("Response is successful", function () {
    pm.response.to.have.status(200);
});

pm.test("Response has required fields", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('success');
    pm.expect(jsonData).to.have.property('code');
    pm.expect(jsonData).to.have.property('message');
});
```

## 📊 导出测试报告

在Apifox中可以导出测试报告：
1. 点击【测试报告】
2. 选择要导出的测试
3. 导出为HTML或PDF格式

## 🔗 相关文档

- [完整联调方案](../APIFOX_联调方案.md)
- [API文档](http://localhost:8082/docs)
- [接口设计规范](../docs/接口设计规范.md)

## ❓ 常见问题

### Q: 无法连接到服务器？
A: 确保服务器正在运行：
```bash
python start_complete_server.py
```

### Q: OpenAPI导入失败？
A: 检查JSON格式是否正确，确保服务器返回有效的OpenAPI规范

### Q: 如何更新API文档？
A: 重新运行导出脚本：
```bash
python apifox_config/export_openapi.py
```

然后在Apifox中重新导入更新后的文件

### Q: 如何共享Apifox项目？
A: 在Apifox中导出项目文件（.json），分享给团队成员导入

---

**最后更新**: 2024年



