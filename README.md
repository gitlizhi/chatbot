# 🤖 老年陪伴机器人 - AI chatbot
一个基于人工智能的老年陪伴机器人系统，具备实时语音交互、长期记忆、声纹识别和个性化陪伴等功能。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![AI Powered](https://img.shields.io/badge/AI-Powered-orange)](https://your-project-url.com)


# 🌟 项目特色
## 核心功能

🎤 实时语音交互 - 24/7持续监听，自然对话体验

🧠 长期记忆系统 - 记住用户偏好和个人信息

👤 声纹识别 - 区分不同家庭成员，个性化交互

💭 智能上下文 - 基于记忆的连贯对话

🔊 高质量语音 - 自然流畅的语音合成

## 技术亮点

- 基于大语言模型的智能对话

- 向量数据库支持的长期记忆

- 实时语音活动检测(VAD)

- 多用户声纹识别与管理（待后续开发）

- 模块化设计，易于扩展

# 🚀 快速开始

环境要求
Python 3.10+

麦克风设备

扬声器设备

# 安装步骤

## 克隆项目

```python
git clone https://github.com/gitlizhi/chatbot.git
cd chatbot
```

## 安装依赖

```python
# 建议先创建一个虚拟环境再做安装
pip install -r requirements.txt
```

## 配置API密钥

```python
# 设置阿里云百炼API密钥
export DASHSCOPE_API_KEY="your-api-key-here"
```

## 运行程序

```python
# 启动完整版（推荐）
python main.py
```

## 或直接运行核心模块（项目结构）

chatbot/

├── main.py                      # 主程序入口

├── tts_realtime_client.py       # 语音tts核心类

├── commit.py                    # 语音合成

├── memory.py                    # 长期记忆管理系统

├── vad_tool.py                  # 语音活动检测工具

├── requirements.txt             # 项目依赖

├── memory_db/                   # 记忆数据库目录

├── temp_audio/                  # 临时音频文件

└── README.md                    # 项目说明


## 🛠 核心模块

1. 实时语音监听 (vad_tool.py)
   
- 智能语音活动检测

- 自动开始/结束录音

- 多线程音频处理

2. 长期记忆系统 (memory.py)
- 基于向量数据库的记忆存储

- 语义相似度检索

- 多用户记忆隔离

3. 声纹识别 (voiceprint.py)（待完善）
- 用户声纹注册与管理

- 实时说话人识别

- 相似度阈值控制

4. 智能对话 (companion.py)
- 大语言模型集成

- 上下文感知回复

- 个性化交互策略


## 🔧 高级功能

自定义记忆规则
在 memory.py 中修改记忆触发规则：

```python
memory_triggers = {
    "personal_info": ["我叫", "我今年", "我住在"],
    "health": ["血压", "血糖", "不舒服"],
    "preferences": ["喜欢", "讨厌", "爱看"],
    # 添加自定义规则...
}
```

扩展对话场景
修改 main.py 中的提示词模板：

```python
prompt_template = """
你是一个专门为老年人设计的陪伴机器人{name}。
{memory_context}
当前对话：{user_input}

请用温暖、关切的语气回复：
```

### 🐛 常见问题

- Q: 录音设备无法识别

- A: 检查麦克风权限，或在代码中指定设备索引：

```python
stream = p.open(input_device_index=1)  # 尝试不同的设备索引
```

- Q: 声纹识别准确率低（该功能待完善）

- A:

    - 确保在安静环境中注册声纹

    - 延长注册录音时长（8-10秒）

    - 调整相似度阈值

- Q: 记忆检索不准确

- A:

    -检查向量数据库连接

    -调整检索参数（n_results）

    -验证嵌入模型是否正常加载

- Q: API调用失败

- A:

    -检查网络连接

    -验证API密钥配置

    -查看服务配额和限制

## 📈 性能优化

- 降低延迟
    - 使用更小的chunk大小（如480）

    - 启用流式语音识别

    - 优化向量检索参数

- 提高准确率
    - 增加声纹注册样本数量

    - 调整VAD阈值适应环境噪音

    - 优化记忆提取规则

- 资源管理
    - 定期清理临时音频文件

    - 限制记忆存储数量

    - 使用量化模型减少内存占用

## 🤝 参与贡献

我们欢迎各种形式的贡献！

报告问题 - 在Issues中提交bug报告或功能建议

代码贡献 - 提交Pull Request改进代码

文档完善 - 帮助改进文档和示例

测试反馈 - 在实际环境中测试并提供反馈

## 开发指南

Fork本项目

- 创建特性分支 (git checkout -b feature/AmazingFeature)

- 提交更改 (git commit -m 'Add some AmazingFeature')

- 推送到分支 (git push origin feature/AmazingFeature)

- 开启Pull Request

## 📄 许可证
本项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情。

## 🙏 致谢

- 阿里云百炼 - 提供强大的大语言模型服务

- ChromaDB - 向量数据库支持

- Resemblyzer - 声纹识别库

- PyAudio - 音频处理库

## 📞 联系我们

如有问题或建议，请通过以下方式联系：

📧 Email: lizhi944254211@163.com

💬 Issues: [GitHub Issues](https://github.com/gitlizhi/chatbot/issues)


<div align="center">
让技术温暖每一个需要陪伴的心灵 ❤️
</div>

注意：本项目为开源研究项目，请遵守相关法律法规，尊重用户隐私。在生产环境中使用时，请确保符合当地的数据保护和隐私法规。
