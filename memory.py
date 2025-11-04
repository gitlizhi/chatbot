import chromadb
# from sentence_transformers import SentenceTransformer
import uuid
import dashscope
from langchain_openai import OpenAIEmbeddings
from datetime import datetime
from chromadb import Documents, EmbeddingFunction, Embeddings
from config import Config
from typing import List, Dict
import numpy as np


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

        # 重排序模型配置
        self.reranker_config = {
            "enable_reranking": True,
            "initial_retrieval_count": 10,   # 初步检索数量
            "final_retrieval_count": 5,      # 最终返回数量
            "reranker_type": "gte-rerank-v2"  # "bge_reranker" 会下载模型到本地，或 "gte-rerank-v2"(调用云端API)
        }

        # 初始化重排序模型
        self._init_reranker()

    def _init_reranker(self):
        """初始化重排序模型"""
        try:
            if self.reranker_config["reranker_type"] == "bge_reranker":
                # 使用BGE重排序模型（免费且效果好）
                from FlagEmbedding import FlagReranker
                self.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=False)
                print("BGE重排序模型加载成功")

            elif self.reranker_config["reranker_type"] == "gte-rerank-v2":
                # 使用阿里云百炼的重排序模型qwen3-rerank
                self.reranker = None  # 使用API调用方式
                print("将使用阿里云百炼的重排序模型qwen3-rerank API进行重排序")

            else:
                print("未配置重排序模型，将使用基础检索")
                self.reranker_config["enable_reranking"] = False

        except ImportError as e:
            print(f"重排序模型加载失败: {e}")
            print("请安装: pip install FlagEmbedding")
            self.reranker_config["enable_reranking"] = False
        except Exception as e:
            print(f"重排序模型初始化错误: {e}")
            self.reranker_config["enable_reranking"] = False

    def rerank_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        """使用重排序模型对记忆进行重新排序"""
        if not self.reranker_config["enable_reranking"] or len(memories) <= 1:
            return memories[:self.reranker_config["final_retrieval_count"]]

        try:
            if hasattr(self, 'reranker') and self.reranker is not None:
                # 使用BGE重排序模型
                pairs = [(query, memory["content"]) for memory in memories]
                scores = self.reranker.compute_score(pairs)

                # 将分数添加到记忆中
                for i, memory in enumerate(memories):
                    memory["relevance_score"] = float(scores[i])

                # 按相关性分数降序排序
                memories.sort(key=lambda x: x["relevance_score"], reverse=True)

                print(f"重排序完成，最高分: {memories[0]['relevance_score']:.4f}")
            elif self.reranker_config["reranker_type"] == "gte-rerank-v2":
                memories = self.dashscope_rerank(query, memories, self.reranker_config["final_retrieval_count"])
            else:
                # 备用方案：使用嵌入相似度进行重排序
                memories = self._fallback_rerank(query, memories)

            return memories[:self.reranker_config["final_retrieval_count"]]

        except Exception as e:
            print(f"重排序失败: {e}, 使用原始顺序")
            return memories[:self.reranker_config["final_retrieval_count"]]

    def _fallback_rerank(self, query: str, memories: List[Dict]) -> List[Dict]:
        """备用重排序方案：使用嵌入相似度"""
        try:
            # 获取查询的嵌入向量
            query_embedding = self.embedding_model.embed_query(query)

            # 计算每个记忆与查询的余弦相似度
            for memory in memories:
                memory_embedding = self.embedding_model.embed_query(memory["content"])
                similarity = self._cosine_similarity(query_embedding, memory_embedding)
                memory["relevance_score"] = similarity

            # 按相似度降序排序
            memories.sort(key=lambda x: x["relevance_score"], reverse=True)
            return memories

        except Exception as e:
            print(f"备用重排序失败: {e}")
            return memories

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def dashscope_rerank(self, query: str, memories: List[Dict], top_n: int):
        """调用阿里云百炼重排序模型sdk"""
        documents = [memory["content"] for memory in memories]
        resp = dashscope.TextReRank.call(
            model="gte-rerank-v2",
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=True
        )
        if resp['status_code'] == 200:
            memory = []
            for dic in resp['output']['results']:
                for tmp_memory in memories:
                    if dic['document']['text'] == tmp_memory['content']:
                        tmp_memory['relevance_score'] = dic['relevance_score']
                        memory.append(tmp_memory)
                        break
        else:
            return []
        print(f'阿里云百炼API memory={memory}')
        return memory

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

                # 如果是提供的错误信息中的结构
                if 'text' in first_item:
                    return first_item['text']

            # 如果以上都不行，尝试转换为字符串
            return str(asr_result)

        except Exception as e:
            print(f"提取ASR文本错误: {e}")
            return str(asr_result)  # 最后的手段

    def extract_text_from_your_asr_format(self, asr_result):
        """专门针对ASR结果格式提取文本"""
        try:
            # 如果是提供的错误信息中的格式
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
        """检索相关记忆 - 增强版，包含重排序"""
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

            # 第一步：初步检索更多记忆
            initial_results = self.collection.query(
                query_texts=[query_text],
                n_results=self.reranker_config["initial_retrieval_count"]
            )

            if not initial_results['documents']:
                print("未找到相关记忆")
                return []

            # 构建记忆列表
            memories = []
            for i, doc in enumerate(initial_results['documents'][0]):
                metadata = initial_results['metadatas'][0][i]
                memories.append({
                    "content": doc,
                    "category": metadata["category"],
                    "speaker": metadata["speaker"],
                    "timestamp": metadata["timestamp"],
                    "initial_rank": i + 1  # 记录初始排名
                })

            print(f"初步检索到 {len(memories)} 条记忆")
            print(memories)
            # 第二步：使用重排序模型重新排序
            if self.reranker_config["enable_reranking"] and len(memories) > 1:
                memories = self.rerank_memories(query_text, memories)
                print(f"重排序后选择前 {len(memories)} 条记忆")
            else:
                # 如果没有启用重排序，直接取前n_results个
                memories = memories[:n_results]

            # 显示排序结果
            for i, memory in enumerate(memories):
                score_info = f" (分数: {memory.get('relevance_score', 0):.4f})" if 'relevance_score' in memory else ""
                print(f"记忆 {i + 1}: {memory['content'][:60]}...{score_info}")

            return memories

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


