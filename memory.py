import chromadb
# from sentence_transformers import SentenceTransformer
import os
from langchain_openai import OpenAIEmbeddings
import uuid
from datetime import datetime
from chromadb import Documents, EmbeddingFunction, Embeddings
from config import Config


# 创建一个符合ChromaDB接口的嵌入函数类
class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, texts: Documents) -> Embeddings:
        # 将文本列表转换为向量
        embeddings = self.embedding_model.embed_documents(texts)
        return embeddings


class MemoryManager:
    def __init__(self, persist_directory="./memory_db"):
        # 初始化向量数据库
        self.client = chromadb.PersistentClient(path=persist_directory)
        # self.collection = self.client.get_or_create_collection(
        #     name="elderly_memory",
        #     metadata={"description": "长期记忆存储"}
        # )

        # 加载嵌入模型
        # self.embedding_model = SentenceTransformer('BAAI/bge-small-zh')

        self.embedding_model = OpenAIEmbeddings(
            base_url=Config.BASE_URL,
            api_key=Config.API_KEY,
            model=Config.EM_MODEL,
            deployment=Config.EM_MODEL,
            check_embedding_ctx_length=False
        )

        embedding_function = OpenAIEmbeddingFunction(self.embedding_model)
        # 使用自定义嵌入函数创建集合
        self.collection = self.client.get_or_create_collection(
            name="elderly_memory",
            embedding_function=embedding_function,
            metadata={"description": "长期记忆存储"}
        )
        # 记忆分类
        self.memory_categories = {
            "personal_info": "个人信息",
            "family": "家庭情况",
            "health": "健康状况",
            "preferences": "个人喜好",
            "daily_life": "日常生活",
            "emotions": "情绪状态"
        }

    def extract_memory_content(self, text, speaker="user"):
        """从对话中提取值得记忆的内容"""
        memory_candidates = []

        # 基于规则的关键词触发
        memory_triggers = {
            "personal_info": ["我叫", "我今年", "我住在", "我的电话"],
            "family": ["我儿子", "我女儿", "我孙子", "我老伴", "我家人"],
            "health": ["血压", "血糖", "头疼", "不舒服", "吃药", "医院"],
            "preferences": ["喜欢", "讨厌", "爱看", "爱吃", "爱听"],
            "daily_life": ["今天去了", "昨天", "上周", "经常"],
            "emotions": ["开心", "难过", "孤单", "担心", "想念"]
        }

        for category, triggers in memory_triggers.items():
            for trigger in triggers:
                if trigger in text:
                    memory_candidates.append({
                        "content": text,
                        "category": category,
                        "speaker": speaker,
                        "timestamp": datetime.now().isoformat()
                    })
                    break

        return memory_candidates

    def store_memory(self, text, speaker="user"):
        """存储记忆到向量数据库"""
        # 确保text是纯文本
        clean_text = self.extract_text_from_asr_result(text)
        print(f'存储记忆：clean_text={clean_text}')
        memories = self.extract_memory_content(clean_text, speaker)

        for memory in memories:
            memory_id = str(uuid.uuid4())

            # 存储到向量数据库
            self.collection.add(
                documents=[memory["content"]],
                metadatas=[{
                    "category": memory["category"],
                    "speaker": memory["speaker"],
                    "timestamp": memory["timestamp"],
                    "type": "conversation_memory"
                }],
                ids=[memory_id]
            )

            print(f"存储记忆: {memory['content'][:50]}...")

        return len(memories)

    def extract_text_from_asr_result(self, asr_result):
        """从ASR结果中提取纯文本"""
        try:
            # 如果已经是字符串，直接返回
            if isinstance(asr_result, str):
                return asr_result

            # 如果是字典，尝试提取文本字段
            elif isinstance(asr_result, dict):
                # 尝试常见的文本字段名
                for field in ['text', 'Text', 'result', 'Result', 'transcript', 'Transcript']:
                    if field in asr_result:
                        text = asr_result[field]
                        if isinstance(text, str):
                            return text

            # 如果是列表，尝试处理每个元素
            elif isinstance(asr_result, list) and len(asr_result) > 0:
                # 检查第一个元素是否有文本
                first_item = asr_result[0]
                if isinstance(first_item, dict):
                    for field in ['text', 'Text', 'result', 'Result']:
                        if field in first_item:
                            text = first_item[field]
                            if isinstance(text, str):
                                return text

                # 如果是您提供的错误信息中的结构
                if 'text' in first_item:
                    return first_item['text']

            # 如果以上都不行，尝试转换为字符串
            return str(asr_result)

        except Exception as e:
            print(f"提取ASR文本错误: {e}")
            return str(asr_result)  # 最后的手段

    def extract_text_from_your_asr_format(self, asr_result):
        """专门针对您的ASR结果格式提取文本"""
        try:
            # 如果是您提供的错误信息中的格式
            if (isinstance(asr_result, list) and
                    len(asr_result) > 0 and
                    isinstance(asr_result[0], dict) and
                    'text' in asr_result[0]):
                return asr_result[0]['text']

            # 如果是其他格式，回退到通用方法
            return self.extract_text_from_asr_result(asr_result)

        except Exception as e:
            print(f"提取ASR文本错误: {e}")
            return str(asr_result)

    def retrieve_related_memories(self, query, n_results=3):
        """检索相关记忆"""
        try:
            # 从ASR结果中提取纯文本
            if hasattr(self, 'extract_text_from_your_asr_format'):
                query_text = self.extract_text_from_your_asr_format(query)
            else:
                query_text = self.extract_text_from_asr_result(query)

            # 如果没有提取到有效文本，直接返回空结果
            if not query_text or query_text.strip() == "":
                print("无法从查询中提取有效文本")
                return []

            print(f"检索记忆，查询文本: {query_text}")

            results = self.collection.query(
                query_texts=[query_text],  # 使用提取的纯文本
                n_results=n_results
            )

            if results['documents']:
                memories = []
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    memories.append({
                        "content": doc,
                        "category": metadata["category"],
                        "speaker": metadata["speaker"],
                        "timestamp": metadata["timestamp"]
                    })
                print(f"找到 {len(memories)} 条相关记忆")
                return memories
            else:
                print("未找到相关记忆")
                return []

        except Exception as e:
            print(f"记忆检索错误: {e}")
            return []

    def get_user_profile(self):
        """获取用户画像摘要"""
        try:
            # 检索所有个人信息类记忆
            personal_memories = self.collection.get(
                where={"category": "personal_info"}
            )

            profile = {}
            if personal_memories['documents']:
                # 这里可以添加更复杂的画像构建逻辑
                profile["summary"] = " | ".join(personal_memories['documents'][:5])

            return profile

        except Exception as e:
            print(f"获取用户画像错误: {e}")
            return {}


