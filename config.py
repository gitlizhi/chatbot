# 配置文件
import os
from dotenv import load_dotenv
load_dotenv()


class Config:
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    # QwenTTS 服务配置
    TTS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime"

    # qwen配置 阿里云百炼
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    CHAT_MODEL = "qwen-max"
    # embedding_model
    EM_MODEL = "text-embedding-v1"
