# GitHub协作工作流

## 📋 概述

本文档描述了盲道检测系统在GitHub上的协作流程，从Apifox联调到最终部署的完整工作流。

## 🌳 分支策略

### 主分支
- `main` - 生产环境分支，只接受来自release分支的合并
- `develop` - 开发主分支，集成所有功能开发

### 功能分支
- `feature/apifox-integration` - Apifox联调功能
- `feature/api-enhancement` - API功能增强
- `feature/android-optimization` - Android优化
- `feature/voice-enhancement` - 语音功能增强

### 修复分支
- `bugfix/connection-issue` - 连接问题修复
- `bugfix/api-response-error` - API响应错误修复

### 发布分支
- `release/v2.0` - 版本2.0发布准备

## 🔄 完整工作流程

### 阶段1：Apifox联调（1周）

#### Day 1-2: 环境准备
```bash
# 1. 从develop创建feature分支
git checkout develop
git pull origin develop
git checkout -b feature/apifox-integration

# 2. 工作内容
# - 导出OpenAPI规范
# - 创建Apifox配置文件
# - 统一端口配置

# 3. 提交代码
git add .
git commit -m "feat: 添加Apifox配置文件和使用指南"
git push origin feature/apifox-integration

# 4. 创建Pull Request
# 在GitHub网页上创建PR，选择base: develop
```

#### Day 3-4: API测试
```bash
# 继续在feature分支上工作

# 1. 测试所有API接口
# 2. 记录问题和改进建议
# 3. 提交测试结果

git add .
git commit -m "test: 完成核心API接口测试"
git push origin feature/apifox-integration
```

#### Day 5: 联调验证
```bash
# 1. 创建自动化测试场景
# 2. 验证错误处理
# 3. 优化API响应

argst add .
git commit -m "feat: 添加自动化测试场景和错误处理"
git push origin feature/apifox-integration

# 4. 在GitHub上请求Review
# - 添加Reviewers
# - 添加Labels: feature, testing
```

### 阶段2：代码审查与合并（2天）

#### Code Review清单
- [ ] 代码风格一致
- [ ] API文档更新
- [ ] 测试用例通过
- [ ] 无硬编码配置
- [ ] 错误处理完善

#### 合并到develop
```bash
# Reviewer批准后
git checkout develop
git pull origin develop
git merge feature/apifox-integration

# 删除功能分支
git branch -d feature/apifox-integration肉
git push origin --delete feature/apifox-integration
```

### 阶段3：持续集成测试（持续）

#### GitHub Actions配置

创建 `.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [ develop, main ]
  pull_request:
    branches: [ develop, main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v --cov=.
```

#### 自动化测试结果
- 每次push自动运行测试
- PR页面上显示测试结果
- 只有测试通过才能合并

### 阶段4：Android APK打包（3天）

#### Day 1: 配置Android项目
```bash
# 1. 创建功能分支
git checkout -b feature/android-apk-build

# 2. 修改build.gradle
# - 添加签名配置
# - 配置ProGuard
# - 优化构建参数

# 3. 提交
git add .
git commit -m "feat: 配置Android APK打包"
git push origin feature/android-apk-build
```

#### Day 2: 实现自动打包

创建 `.github/workflows/android-build.yml`:
```yaml
name: Android Build

on:
  push:
    branches: [ main ]
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up JDK
      uses: actions/setup-java@v3
      with:
        java-version: '17'
        distribution: 'temurin'
    
    - name: Build APK
      working-directory: android_app
      run: |
        chmod +x gradlew
        ./gradlew assembleRelease
    
    - name: Upload APK
      uses: actions/upload-artifact@v3
      with:
        name: app-release
        path: android_app/app/build/outputs/apk/release/app-release.apk
```

#### Day 3: 发布到GitHub Releases
```bash
# 1. 创建release分支
git checkout -b release/v2.0

# 2. 版本号管理
# 修改 versionCode 和 versionName

# 3. 合并到main
git checkout main
git merge release/v2.0
git tag v2.0.0
git push origin main --tags

# GitHub Actions自动创建Release并上传APK
```

### 阶段5：测试与优化（持续）

#### Issue管理
```bash
# 1. 在GitHub上创建Issue
# - Bug报告
# - 功能建议
# - 性能优化

# 2. Posted + Label分类
# Labels:
# - bug
# - enhancement
# - performance
# - question
```

#### 问题修复流程
```bash
# 1. 从Issue创建分支
от checkout develop
git checkout -b bugfix/[issue-number]-[description]

# 2. 修复问题
# ... 修复代码 ...

# 3. 提交并关联Issue
git add .
git commit -m "fix: mounting [description] (#issue-number)"
git push origin bugfix/[issue-number]-[description]

# 4. 创建PR，关联Issue
# 在PR描述中添加: Closes #issue-number
```

## 📊 版本管理

### 语义化版本
- Major (x.0.0): 重大更改，不向后兼容
- Minor (0.x.0): 新功能，向后兼容
- Patch (0.0.x): Bug修复，向后兼容

### 版本发布流程
```bash
# 1. 创建release分支
git checkout -b release/v2.0.0

# 2. 更新版本号
# - CHANGELOG.md
# - version.py
# - android_app/build.gradle

# 3. 测试
# - 运行所有测试
# - 集成测试
# - 性能测试

# 4. 合并到main
git checkout main
git merge release/v2.0.0
git tag v2.0.0

# 5. 合并到develop
git checkout develop
git merge release/v2.0.0

# 6. 推送
git push origin main --tags
git push origin develop
```

### 变更日志

创建 `CHANGELOG.md`:
```markdown
# Changelog

## [2.0.0] - 2024-XX-XX

### Added
- Apifox联调功能
- 自动化CI/CD流程
- Android APK自动打包

### Changed
- 统一端口配置为8082
- 优化API响应格式

### Fixed
- 修复前后端连接问题
- 修复Android端硬编码IP问题
```

## 🔍 Code Review指南

### 提交PR时包含
- [ ] 清晰的PR描述
- [ ] 关联的Issue编号
- [ ] 测试截图/视频
- [ ] 更新API文档
- [ ] 添加测试用例

### Review检查点
- [ ] 代码逻辑正确
- [ ] 遵循编码规范
- [ ] 异常处理完善
- [ ] 性能考虑
- [ ] 安全性检查

## 📱 手机端部署

### 内部测试
```bash
# 1. 使用GitHub Releases的APK
# 2. 通过二维码分发
# 3. 收集反馈

# 创建Issue跟踪问题
```

### 生产环境
```bash
# 1. 服务器部署
# - 使用云服务器
# - 配置HTTPS
# - 设置域名

# 2. 更新APK中的服务器地址
# 3. 发布新版本
```

## 🎯 成功指标

### 代码质量
- ✅ 所有测试通过率 > 95%
- ✅ Code Coverage > 80%
- ✅ 无Critical级别Bug

### 协作效率
- ✅ PR平均Review时间 < 24小时
- ✅ Issue平均响应时间 < 48小时
- ✅ 自动化构建成功率 > 98%

### 版本发布
- ✅ 每个版本都有完整测试
- ✅ 发布说明清晰
- ✅ 支持快速回滚

## 🔗 相关资源

- [GitHub Actions文档](https://docs.github.com/en/actions)
- [Git工作流最佳实践](https://www.atlassian.com/git/tutorials/comparing-workflows)
- [语义化版本控制](https://semver.org/)
- [项目完整联调方案](../APIFOX_联调方案.md)

---

**最后更新**: 2024年
**维护者**: 开发团队



