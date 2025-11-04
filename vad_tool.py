import webrtcvad
import collections
import pyaudio
import os
import wave
import threading
import queue
import time
import numpy as np
from collections import deque


class WebRTCVADRecorder:

    def __init__(self, rate=16000, aggressiveness=2):
        self.rate = rate
        self.vad = webrtcvad.Vad(aggressiveness)  # 0-3，3最激进

        # 参数配置
        self.frame_duration = 30  # 毫秒，webrtcvad要求10,20,30ms
        self.chunk = int(rate * self.frame_duration / 1000)  # 每帧采样数
        self.silence_timeout = 2.0  # 静音超时（秒）
        self.min_recording_duration = 1.0  # 最小录音时长（秒）

        self.format = pyaudio.paInt16
        self.channels = 1

    def record_until_silence(self, filename="user_audio.wav"):
        """使用WebRTC VAD进行智能录音"""
        p = pyaudio.PyAudio()

        try:
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=1
            )

            print("开始智能录音（WebRTC VAD）...")
            frames = []
            voiced_frames = []
            ring_buffer = collections.deque(maxlen=10)  # 用于平滑检测

            silence_frames_threshold = int(self.silence_timeout * 1000 / self.frame_duration)
            min_frames = int(self.min_recording_duration * 1000 / self.frame_duration)

            silence_frames = 0
            total_frames = 0
            is_recording = False

            while True:
                data = stream.read(self.chunk, exception_on_overflow=False)

                # 使用VAD检测语音活动
                try:
                    is_speech = self.vad.is_speech(data, self.rate)
                except:
                    is_speech = False

                ring_buffer.append(1 if is_speech else 0)

                # 使用滑动窗口判断（减少误判）
                speech_ratio = sum(ring_buffer) / len(ring_buffer)
                is_voiced = speech_ratio > 0.5

                if is_voiced:
                    if not is_recording:
                        print("检测到语音，开始录音...")
                        is_recording = True
                    silence_frames = 0
                    voiced_frames.append(data)
                else:
                    silence_frames += 1

                # 如果已经开始录音，保存所有帧
                if is_recording:
                    frames.append(data)
                    total_frames += 1

                    # 检查停止条件
                    if (silence_frames >= silence_frames_threshold and
                            len(voiced_frames) >= min_frames):
                        duration = total_frames * self.frame_duration / 1000
                        print(f"检测到持续静音，停止录音。录音时长: {duration:.2f}秒")
                        break

                # 安全限制
                if total_frames > (30 * 1000 / self.frame_duration):  # 30秒
                    print("达到最大录音时长，自动停止")
                    break

            stream.stop_stream()
            stream.close()

            # 保存音频文件
            if frames:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
                wf.close()

                file_size = os.path.getsize(filename)
                print(f"智能录音完成: {filename}, 大小: {file_size} 字节")
                return filename
            else:
                print("没有录制到音频")
                return None

        except Exception as e:
            print(f"智能录音过程中出错: {e}")
            return None
        finally:
            p.terminate()


class RealTimeVoiceMonitor:
    def __init__(self, companion_instance, rate=16000, chunk=480,
                 silence_threshold=200, min_silence_duration=1.5,
                 max_single_utterance=10.0):
        """
        实时语音监控器

        Args:
            companion_instance: 陪伴机器人实例
            rate: 采样率
            chunk: 每次读取的音频块大小（推荐480，对应30ms，适合VAD）
            silence_threshold: 静音阈值
            min_silence_duration: 最小静音持续时间（秒）
            max_single_utterance: 单次说话最大时长（秒）
        """
        self.companion = companion_instance
        self.rate = rate
        self.chunk = chunk
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.max_single_utterance = max_single_utterance

        self.format = pyaudio.paInt16
        self.channels = 1

        # 状态控制
        self.is_listening = False
        self.is_processing = False
        self.current_state = "idle"  # idle, detecting, recording, processing

        # 音频缓冲区
        self.audio_buffer = deque(maxlen=int(rate * 2 / chunk))  # 保存2秒音频用于预触发
        self.current_recording = []

        # 线程和队列
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()

        # 统计信息
        self.stats = {
            "total_detections": 0,
            "processed_utterances": 0,
            "last_activity_time": time.time()
        }

    def calculate_energy(self, audio_data):
        """计算音频能量 - 稳定版本"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0

            # 使用绝对值的平均值，避免平方根问题
            # 这种方法对静音帧更稳定，计算开销也更小
            energy = np.mean(np.abs(audio_array))
            return energy

        except Exception as e:
            print(f"计算能量时出错: {e}")
            return 0

    def auto_calibrate_threshold(self, calibration_seconds=3):
        """自动校准静音阈值"""
        print("正在进行环境噪音校准，请保持安静...")

        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=1
            )

            energies = []
            total_frames = int(self.rate * calibration_seconds / self.chunk)

            for i in range(total_frames):
                data = stream.read(self.chunk, exception_on_overflow=False)
                energy = self.calculate_energy(data)
                energies.append(energy)

            avg_energy = np.mean(energies)
            std_energy = np.std(energies)

            # 设置阈值为平均值加上2倍标准差
            new_threshold = avg_energy + 2 * std_energy
            # 确保阈值至少为400
            new_threshold = max(400, new_threshold)

            print(f"环境噪音水平: {avg_energy:.2f} ± {std_energy:.2f}")
            print(f"自动设置静音阈值: {new_threshold:.2f}")

            self.silence_threshold = new_threshold

            stream.stop_stream()
            stream.close()
            return new_threshold

        except Exception as e:
            print(f"校准过程中出错: {e}")
            return None
        finally:
            p.terminate()

    def audio_capture_thread(self):
        """音频捕获线程 - 持续从麦克风读取数据"""
        p = pyaudio.PyAudio()

        try:
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=1
            )

            print("音频捕获线程启动 - 开始持续监听...")

            while self.is_listening:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    self.audio_queue.put(data)
                except Exception as e:
                    print(f"音频捕获错误: {e}")
                    time.sleep(0.1)

            stream.stop_stream()
            stream.close()

        except Exception as e:
            print(f"音频捕获线程错误: {e}")
        finally:
            p.terminate()

    def voice_detection_thread(self):
        """语音检测线程 - 实时检测语音活动"""
        print("语音检测线程启动 - 等待语音活动...")

        silence_frames_threshold = int(self.min_silence_duration * self.rate / self.chunk)
        max_recording_frames = int(self.max_single_utterance * self.rate / self.chunk)

        silence_frames = 0
        is_recording = False
        recording_start_time = 0
        recording_frames = 0

        # 预触发缓冲区（保存语音开始前0.5秒的音频）
        pre_trigger_buffer = deque(maxlen=int(0.5 * self.rate / self.chunk))

        while self.is_listening:
            try:
                # 从队列获取音频数据（阻塞，最多等待100ms）
                data = self.audio_queue.get(timeout=0.1)

                # 更新预触发缓冲区
                pre_trigger_buffer.append(data)

                # 计算能量
                energy = self.calculate_energy(data)

                if is_recording:
                    # 录音状态
                    self.current_recording.append(data)
                    recording_frames += 1

                    # 检查能量水平
                    if energy <= self.silence_threshold:
                        silence_frames += 1
                    else:
                        silence_frames = 0

                    # 检查停止条件
                    current_time = time.time()
                    recording_duration = current_time - recording_start_time

                    # 条件1: 持续静音超过阈值
                    condition1 = silence_frames >= silence_frames_threshold
                    # 条件2: 达到最大录音时长
                    condition2 = recording_duration >= self.max_single_utterance
                    # 条件3: 录音帧数过多
                    condition3 = recording_frames >= max_recording_frames

                    if condition1 or condition2 or condition3:
                        # 停止录音，准备处理
                        print(f"语音结束，录音时长: {recording_duration:.2f}秒")

                        # 将完整的录音数据放入处理队列
                        full_recording = list(pre_trigger_buffer) + self.current_recording
                        self.processing_queue.put(full_recording)

                        # 重置状态
                        is_recording = False
                        self.current_recording = []
                        silence_frames = 0
                        recording_frames = 0
                        self.current_state = "processing"

                    # 实时显示录音状态
                    if recording_frames % 10 == 0:  # 每10帧显示一次
                        print(f"录音中... {recording_duration:.1f}秒, 能量: {energy:.1f}")

                else:
                    # 检测状态 - 等待语音开始
                    if energy > self.silence_threshold:
                        # 检测到语音活动，开始录音
                        print("检测到语音活动，开始录音...")
                        is_recording = True
                        recording_start_time = time.time()
                        self.current_state = "recording"
                        self.stats["total_detections"] += 1

                        # 保存当前预触发缓冲区
                        self.current_recording = list(pre_trigger_buffer)
                        recording_frames = len(pre_trigger_buffer)

            except queue.Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                print(f"语音检测错误: {e}")
                time.sleep(0.1)

    def processing_thread(self):
        """处理线程 - 处理检测到的语音"""
        print("处理线程启动 - 等待处理任务...")

        temp_audio_dir = "temp_audio"
        os.makedirs(temp_audio_dir, exist_ok=True)

        while self.is_listening:
            try:
                # 从处理队列获取录音数据（阻塞）
                audio_frames = self.processing_queue.get(timeout=1)

                if audio_frames is None:  # 退出信号
                    break

                print("开始处理检测到的语音...")
                self.is_processing = True

                # 保存临时音频文件
                timestamp = int(time.time())
                audio_filename = os.path.join(temp_audio_dir, f"utterance_{timestamp}.wav")

                # 保存音频文件
                p = pyaudio.PyAudio()
                wf = wave.open(audio_filename, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(audio_frames))
                wf.close()
                p.terminate()

                # 语音转文本
                print("正在进行语音识别...")
                user_text = self.companion.speech_to_text(audio_filename)

                if user_text and isinstance(user_text, str) and len(user_text.strip()) > 0:
                    print(f"识别结果: {user_text}")

                    # 调用大模型生成回复
                    print("生成回复...")
                    if hasattr(self.companion, 'call_bailian_api_with_memory'):
                        response_text = self.companion.call_bailian_api_with_memory(user_text)
                    else:
                        response_text = self.companion.call_bailian_api(user_text)

                    print(f"AI回复: {response_text}")

                    # 文本转语音并播放
                    # print("合成语音...")
                    response_audio = self.companion.text_to_speech(response_text)

                    if response_audio:
                        print("播放回复...")
                        self.companion.play_audio(response_audio)

                    self.stats["processed_utterances"] += 1
                else:
                    print("未识别到有效语音")

                # 清理临时文件
                try:
                    os.remove(audio_filename)
                except:
                    pass

                self.is_processing = False
                self.current_state = "idle"
                print("处理完成，返回监听状态...")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理线程错误: {e}")
                self.is_processing = False
                self.current_state = "idle"

    def start_realtime_listening(self):
        """启动实时监听"""
        print("启动实时语音监听系统...")
        print("机器人现在处于持续监听状态，可以随时说话")
        print("按下 Ctrl+C 停止监听")
        # 自动校准环境噪音
        self.auto_calibrate_threshold()

        self.is_listening = True
        self.current_state = "idle"

        # 启动各个线程
        threads = []

        # 音频捕获线程
        capture_thread = threading.Thread(target=self.audio_capture_thread)
        capture_thread.daemon = True
        capture_thread.start()
        threads.append(capture_thread)

        # 语音检测线程
        detection_thread = threading.Thread(target=self.voice_detection_thread)
        detection_thread.daemon = True
        detection_thread.start()
        threads.append(detection_thread)

        # 处理线程
        processing_thread = threading.Thread(target=self.processing_thread)
        processing_thread.daemon = True
        processing_thread.start()
        threads.append(processing_thread)

        # 状态显示线程
        status_thread = threading.Thread(target=self.status_monitor_thread)
        status_thread.daemon = True
        status_thread.start()
        threads.append(status_thread)

        try:
            # 主线程保持运行
            while self.is_listening:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n收到停止信号，正在关闭实时监听...")
            self.stop_realtime_listening()

        # 等待线程结束
        for thread in threads:
            thread.join(timeout=2)

        print("实时监听系统已关闭")

    def status_monitor_thread(self):
        """状态监控线程 - 定期显示系统状态"""
        tmp_print = ""
        while self.is_listening:
            status_info = {
                "idle": "🟢 监听中 - 等待语音",
                "detecting": "🟡 检测中 - 分析音频",
                "recording": "🔴 录音中 - 请继续说话",
                "processing": "🟣 处理中 - 生成回复"
            }

            status_emoji = status_info.get(self.current_state, "⚪ 未知状态")
            stats_text = f"检测: {self.stats['total_detections']}次, 处理: {self.stats['processed_utterances']}次"
            now_print = f"\r{status_emoji} | {stats_text} | 按Ctrl+C退出"
            if now_print != tmp_print:
                tmp_print = now_print
                print(tmp_print, end="\n", flush=True)
            time.sleep(0.5)

    def stop_realtime_listening(self):
        """停止实时监听"""
        self.is_listening = False
        # 发送退出信号到处理队列
        self.processing_queue.put(None)