# Montreal Forced Aligner (MFA) 安装和使用说明

## 安装完成

✅ **安装成功**：Montreal Forced Aligner 3.3.4 已成功安装在 Python 3.10 环境中

## 环境信息
- **环境名称**: aligner
- **Python版本**: 3.10
- **MFA版本**: 3.3.4
- **Whisper版本**: 20250625
- **安装路径**: `C:\Users\User\anaconda3\envs\aligner`

## 使用方法

### 激活环境
```bash
conda activate aligner
```

### 运行 MFA 命令
```bash
# 直接使用（在激活环境后）
mfa version

# 使用完整路径（无需激活环境）
C:\Users\User\anaconda3\envs\aligner\Scripts\mfa.exe version
```

### 基本对齐命令
```bash
mfa align [选项] 语料目录 词典路径 声学模型路径 输出目录
```

## 常用命令
- `mfa version` - 查看版本
- `mfa align` - 执行强制对齐
- `mfa train` - 训练声学模型
- `mfa validate` - 验证语料库
- `mfa download` - 下载预训练模型

## 安装方法（备用）
如果需要重新安装：
```bash
conda create -n aligner -c conda-forge montreal-forced-aligner python=3.10 -y
```

## 官方文档
- 官方网站: https://montreal-forced-aligner.readthedocs.io/
- 安装指南: https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html

## 安装额外工具
### Whisper 语音识别
```bash
# 在 aligner 环境中安装
pip install openai-whisper
```

## 注意事项
1. 确保使用 conda 环境来运行 MFA
2. 首次使用可能需要下载预训练模型：`mfa download`
3. 支持多种语言和声学模型
4. 需要准备正确的语料库结构和发音词典
5. Whisper 可用于语音转录，与 MFA 配合使用