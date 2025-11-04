import pyaudio
import wave
from http import HTTPStatus
import asyncio
import os
from openai import OpenAI
from dashscope.audio.asr import Recognition
import dashscope
import time
from commit import _run_demo
from memory import MemoryManager
from vad_tool import WebRTCVADRecorder, RealTimeVoiceMonitor
from config import Config

class ElderlyCompanionDemo:
    def __init__(self):
        # 初始化客户端
        self.setup_clients()

        # 音频参数
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 5  # 每次录音时长

    def setup_clients(self):
        """初始化阿里云各服务客户端"""
        # 调用百炼API
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=Config.API_KEY,
            base_url=Config.BASE_URL,
        )

    def record_audio(self, filename="user_audio.wav"):
        """录制音频 - 智能停顿"""
        filename = WebRTCVADRecorder().record_until_silence()
        return filename

    def record_audio1(self, filename="user_audio.wav"):
        """录制音频 - 固定时长为 self.record_seconds 秒"""
        try:
            p = pyaudio.PyAudio()

            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=1
            )

            print(f"开始录音...（{self.record_seconds}秒）")
            frames = []

            total_frames = int(self.rate * self.record_seconds / self.chunk)
            # print(f"预计录制帧数: {total_frames}")

            for i in range(total_frames):
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
                    # 显示录音进度
                    # if i % 10 == 0:  # 每10帧打印一次进度
                    #     print(f"录制进度: {i + 1}/{total_frames}")
                except Exception as e:
                    print(f"读取音频数据时出错: {e}")
                    break

            print(f"录音结束，共录制 {len(frames)} 帧数据")

            stream.stop_stream()
            stream.close()
            p.terminate()

            # 检查数据
            if not frames:
                print("警告：没有录制到任何音频数据")
                return False

            # 保存为wav文件
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            file_size = os.path.getsize(filename)
            print(f"音频文件已保存: {filename}, 大小: {file_size} 字节")

            # 验证文件是否可以正常读取
            try:
                test_wf = wave.open(filename, 'rb')
                test_frames = test_wf.getnframes()
                test_rate = test_wf.getframerate()
                test_wf.close()
                print(f"文件验证: {test_frames} 帧, 采样率: {test_rate} Hz")
            except Exception as e:
                print(f"文件验证失败: {e}")
                return False

            return filename

        except Exception as e:
            print(f"录音过程中出错: {e}")
            return None

    def speech_to_text(self, audio_file="user_audio.wav"):
        """语音转文本 - 使用文件转写服务"""
        try:
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            recognition = Recognition(model='fun-asr-realtime',
                                      format='wav',
                                      sample_rate=16000,
                                      callback=None)
            result = recognition.call(audio_file)
            if result.status_code != HTTPStatus.OK:
                print('Error: ', result.message)
                return ""
            # print('\n语音转文本识别结果：')
            # print(result.get_sentence())
            # print(
            #     '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
            #     .format(
            #         recognition.get_last_request_id(),
            #         recognition.get_first_package_delay(),
            #         recognition.get_last_package_delay(),
            #     ))
            resp = result.get_sentence()
            if len(resp) > 0:
                return resp[0]['text']
            else:
                return ""
        except Exception as e:
            print(f"语音识别错误: {e}")
            return ""

    def text_to_speech(self, text, audio_name="response_audio.wav"):
        """文本转语音"""
        try:
            save_file = asyncio.run(_run_demo(text, audio_name))
            return save_file

        except Exception as e:
            print(f"语音合成错误: {e}")
            return None

    def play_audio(self, audio_file):
        """播放音频"""
        try:
            wf = wave.open(audio_file, 'rb')
            p = pyaudio.PyAudio()

            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

            data = wf.readframes(self.chunk)
            while data:
                stream.write(data)
                data = wf.readframes(self.chunk)

            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()

        except Exception as e:
            print(f"播放音频错误: {e}")

    def call_bailian_api(self, memory_context, user_input):
        """调用百炼大模型API"""
        try:
            completion = self.client.chat.completions.create(
                # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                # model="qwen-plus",
                model="qwen-max",
                messages=[
                    {"role": "system", "content": f"你是一个专门为老年人设计的陪伴机器人，名字叫小伴。你说话要温柔、耐心、简洁，语速要慢一点, {memory_context}, "
                                                  f"绝对禁止在回复中使用任何形式的表情符号、颜文字（例如：😀, 😊, :) , :( 等）。"
                                                  f"回复内容应保持正式、书面化的语言风格，确保输出为纯净的中文文本内容。"
                                                  f"你所有的回复都将被用于语音合成，任何非文本字符都会导致合成失败。"},
                    {"role": "user", "content": f"{user_input}"},
                ],
            )
            resp_text = completion.choices[0].message.content
            # print(f'大模型回复：{resp_text}')
            return resp_text

        except Exception as e:
            print(f"调用大模型API错误: {e}")
            return "抱歉，我刚才没听清楚，能再说一次吗？"

    def run_conversation_cycle(self, user_text=None):
        """运行一次完整的对话循环"""
        try:
            # 1. 录音
            audio_file = self.record_audio()

            # 2. 语音转文本
            if audio_file:
                # print("正在识别语音...")
                user_text = self.speech_to_text(audio_file)
            if not user_text:
                user_text = "你好,我是李爷爷"  # 默认问候

            # print(f"识别结果: {user_text}")

            # # 3. 调用大模型
            # print("正在生成回复...")
            response_text = self.call_bailian_api(memory_context="", user_input=user_text)
            # print(f"AI回复: {response_text}")

            # 4. 文本转语音
            # print("正在合成AI语音...")
            response_audio = self.text_to_speech(response_text)

            # 5. 播放回复
            if response_audio:
                print("播放回复...")
                self.play_audio(response_audio)

            return True

        except Exception as e:
            print(f"对话循环错误: {e}")
            return False

    def test_audio_system(self):
        """测试整个音频系统"""
        import pyaudio
        import wave

        p = pyaudio.PyAudio()

        print("=== 音频系统测试 ===")

        # 测试输入设备
        print("输入设备:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  设备 {i}: {info['name']}")

        # 测试输出设备
        print("输出设备:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                print(f"  设备 {i}: {info['name']}")

        p.terminate()

        # 测试 wave 模块
        print("wave 模块测试...")
        try:
            with wave.open('test.wav', 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b'\x00' * 32000)  # 2秒静音
            print("wave 写入测试通过")

            with wave.open('test.wav', 'r') as wf:
                frames = wf.getnframes()
                print(f"读取测试: {frames} 帧")
            print("wave 读取测试通过")

            import os
            os.remove('test.wav')

        except Exception as e:
            print(f"wave 测试失败: {e}")


class ElderlyCompanionWithMemory(ElderlyCompanionDemo):
    def __init__(self):
        super().__init__()
        # 记忆管理器
        self.memory_manager = MemoryManager()
        # 实时监听器
        self.realtime_monitor = RealTimeVoiceMonitor(self)
    def call_bailian_api_with_memory(self, user_input):
        """带长期记忆的大模型调用"""
        try:
            # 1. 检索相关记忆
            related_memories = self.memory_manager.retrieve_related_memories(user_input)

            # 2. 构建记忆上下文
            memory_context = ""
            if related_memories:
                memory_context = "相关记忆：\n"
                for memory in related_memories:
                    speaker = "老人" if memory["speaker"] == "user" else "小伴"
                    memory_context += f"- {speaker}曾说过：{memory['content']}\n"
                memory_context += "\n"
            print(f'检索到相关记忆：{memory_context}')

            # 3. 调用大模型（使用API调用方式）
            response = self.call_bailian_api(memory_context, user_input)  # 复用您现有的方法

            # 4. 存储当前对话到长期记忆
            self.memory_manager.store_memory(user_input, "user")
            self.memory_manager.store_memory(response, "assistant")

            return response

        except Exception as e:
            print(f"带记忆的对话错误: {e}")
            return self.call_bailian_api(user_input)  # 降级到普通对话

    def run_conversation_cycle_with_memory(self):
        """带长期记忆的对话循环"""
        try:
            # 1. 录音
            audio_file = self.record_audio()

            # 2. 语音转文本
            print("正在识别语音...")
            a = time.time()
            user_text = self.speech_to_text(audio_file)
            if not user_text:
                user_text = "你好"
            print(f"正在识别语音耗时{time.time()-a}")
            print(f"识别结果: {user_text}")

            # 3. 带记忆的大模型调用
            print("正在生成回复（带记忆）...")
            a = time.time()
            response_text = self.call_bailian_api_with_memory(user_text)
            print(f"正在生成回复（带记忆）...耗时{time.time() - a}")
            print(f"AI回复: {response_text}")

            # 4. 文本转语音
            print("正在合成语音...")
            a = time.time()
            response_audio = self.text_to_speech(response_text)
            print(f"正在合成语音...耗时{time.time() - a}")
            # 5. 播放回复
            if response_audio:
                print("播放回复...")
                self.play_audio(response_audio)

            return True

        except Exception as e:
            print(f"对话循环错误: {e}")
            return False

    def show_memory_stats(self):
        """显示记忆统计"""
        try:
            # 获取所有记忆
            all_memories = self.memory_manager.collection.get()
            count = len(all_memories['ids']) if all_memories['ids'] else 0

            print(f"\n=== 记忆系统统计 ===")
            print(f"总记忆数量: {count}")

            # 按分类统计
            if count > 0:
                categories = {}
                for metadata in all_memories['metadatas']:
                    cat = metadata.get('category', 'unknown')
                    categories[cat] = categories.get(cat, 0) + 1

                for cat, num in categories.items():
                    chinese_name = self.memory_manager.memory_categories.get(cat, cat)
                    print(f"{chinese_name}: {num}条")

            return count

        except Exception as e:
            print(f"统计记忆错误: {e}")
            return 0

    def search_memories(self, query):
        """搜索特定记忆"""
        memories = self.memory_manager.retrieve_related_memories(query, n_results=5)

        print(f"\n=== 搜索 '{query}' 相关记忆 ===")
        for i, memory in enumerate(memories, 1):
            speaker = "老人" if memory["speaker"] == "user" else "小伴"
            category = self.memory_manager.memory_categories.get(memory["category"], memory["category"])
            print(f"{i}. [{category}] {speaker}: {memory['content']}")

        return memories

    def start_realtime_companion(self):
        """启动实时陪伴模式"""
        print("=" * 60)
        print("老年陪伴机器人 - 实时监听模式")
        print("=" * 60)
        print("特性:")
        print("  • 24/7持续监听环境")
        print("  • 智能语音活动检测")
        print("  • 自动开始/结束录音")
        print("  • 实时处理与响应")
        print("  • 10秒、30秒、甚至几分钟后说话都能响应")
        print("=" * 60)

        # 显示记忆统计
        self.show_memory_stats()

        # 启动实时监听
        self.realtime_monitor.start_realtime_listening()

    def start_demo(self, memory=True, realtime=False):
        """启动Demo - 增加实时模式选项"""
        if realtime:
            self.start_realtime_companion()
        else:
            # 原有的循环模式
            self.start_demo_old(memory)

    def start_demo_old(self, memory=True):
        """启动Demo"""
        print("=" * 50)
        print("老年陪伴机器人 Demo 启动")
        print("按下 Ctrl+C 退出")
        print("=" * 50)

        # 主循环
        while True:
            try:
                print("\n请说话...")
                if memory:
                    success = self.run_conversation_cycle_with_memory()
                else:
                    success = self.run_conversation_cycle()
                print("\n正在等待下一次循环...")
                time.sleep(1000)
                if not success:
                    print("对话失败，重新开始...")
                    time.sleep(2)

            except KeyboardInterrupt:
                print("\n\n感谢使用，再见！")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                time.sleep(1)


# 运行Demo
if __name__ == "__main__":
    companion = ElderlyCompanionWithMemory()
    companion.start_demo(memory=True, realtime=True)           # 带记忆版本 支持 实时监听语音
    # companion.start_demo_old(memory=True)


