import os
import asyncio
import logging
import wave
from tts_realtime_client import TTSRealtimeClient, SessionMode
import re
import pyaudio
from dotenv import load_dotenv
load_dotenv()

# QwenTTS 服务配置
URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime"
API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    raise ValueError("Please set DASHSCOPE_API_KEY environment variable")

# 收集音频数据
_audio_chunks = []
_AUDIO_SAMPLE_RATE = 24000
_audio_pyaudio = pyaudio.PyAudio()


def _audio_callback(audio_bytes: bytes):
    """TTSRealtimeClient 音频回调: 只收集音频数据，不播放"""
    _audio_chunks.append(audio_bytes)
    logging.info(f"Received audio chunk: {len(audio_bytes)} bytes")


def _save_audio_to_file(filename: str = "output.wav", sample_rate: int = 24000) -> bool:
    """将收集到的音频数据保存为 WAV 文件"""
    if not _audio_chunks:
        logging.warning("No audio data to save")
        return False

    try:
        audio_data = b"".join(_audio_chunks)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        logging.info(f"Audio saved to: {filename}")
        return True
    except Exception as exc:
        logging.error(f"Failed to save audio: {exc}")
        return False


async def _user_input_loop(client: TTSRealtimeClient, text: str):
    """持续获取用户输入并发送文本，当用户输入空文本时发送commit事件并结束本次会话"""
    print("请输入文本（直接按Enter发送commit事件并结束本次会话，按Ctrl+C或Ctrl+D结束整个程序）：")
    text_fragments = re.split(r'[,.;:，。！；]', text)
    for user_text in text_fragments:
        try:
            logging.info(f"发送文本: {user_text}")
            await client.append_text(user_text)
        except EOFError:  # 用户按下Ctrl+D
            break
        except KeyboardInterrupt:  # 用户按下Ctrl+C
            break
    # 空输入视为一次对话的结束: 提交缓冲区 -> 结束会话 -> 跳出循环
    logging.info("空输入，发送 commit 事件并结束本次会话")
    await client.commit_text_buffer()
    # 适当等待服务器处理 commit，防止过早结束会话导致丢失音频
    await asyncio.sleep(0.3)
    await client.finish_session()
    # 结束会话
    logging.info("结束会话...")


async def _run_demo(text, audio_file):
    """运行完整 Demo"""
    client = TTSRealtimeClient(
        base_url=URL,
        api_key=API_KEY,
        voice="Cherry",
        language_type="Chinese",  # 建议与文本语种一致，以获得正确的发音和自然的语调。
        mode=SessionMode.COMMIT,  # 修改为COMMIT模式
        audio_callback=_audio_callback
    )

    # 建立连接
    await client.connect()

    # 并行执行消息处理与用户输入
    consumer_task = asyncio.create_task(client.handle_messages())
    producer_task = asyncio.create_task(_user_input_loop(client, text))

    await producer_task  # 等待用户输入完成

    # 额外等待，确保所有音频数据收取完毕
    await asyncio.sleep(5)

    # 关闭连接并取消消费者任务
    await client.close()
    consumer_task.cancel()

    # 保存音频数据
    os.makedirs("outputs", exist_ok=True)
    save_file = os.path.join("outputs", audio_file)
    _save_audio_to_file(save_file)
    _audio_chunks.clear()       # 清空缓存区的数据
    return save_file