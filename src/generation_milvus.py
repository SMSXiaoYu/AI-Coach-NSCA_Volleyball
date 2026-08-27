import os
from pymilvus import connections, Collection
from openai import OpenAI
from chromadb.utils import embedding_functions
import chromadb

# ==================== 配置 ====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b"

# ==================== 连接 Milvus ====================
connections.connect(host="localhost", port="19530")
milvus_collection = Collection("ai_coach_milvus")
milvus_collection.load()

# ==================== 初始化 Embedding 函数 ====================
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# ==================== 初始化 LLM ====================
llm_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"
)

# ==================== 获取知识库信息（从 Chroma 元数据） ====================
def get_kb_info():
    try:
        chroma_client = chromadb.PersistentClient(path="chroma_db")
        chroma_collection = chroma_client.get_collection("ai_coach")
        result = chroma_collection.get()
        metadatas = result.get('metadatas', [])
        if not metadatas:
            return "空知识库", 0
        sources = list(set([m.get('source', '未知文件') for m in metadatas if m and m.get('source')]))
        total_chunks = len(metadatas)
        if len(sources) <= 3:
            books_display = ", ".join(sources)
        else:
            books_display = f"{len(sources)} 本书（{', '.join(sources[:3])} 等）"
        return books_display, total_chunks
    except Exception as e:
        return f"读取失败: {e}", 0

# ==================== 检索函数（Milvus） ====================
def retrieve(query, n_results=3):
    """使用 Milvus 检索相关文档"""
    try:
        # 将问题转为向量
        query_embedding = embedding_fn([query])
        
        # Milvus 搜索
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = milvus_collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=n_results,
            output_fields=["text", "source"]
        )
        
        docs = []
        sources = []
        for hits in results:
            for hit in hits:
                text = hit.entity.get('text', '')
                source = hit.entity.get('source', '未知')
                if text:
                    docs.append(text)
                    sources.append(source)
        
        return docs, sources
    except Exception as e:
        print(f"❌ 检索失败: {e}")
        return [], []

# ==================== 生成函数（LLM） ====================
def generate_response(query, context_docs, sources):
    """基于检索结果调用 LLM 生成回答"""
    context = "\n\n".join(context_docs)
    
    system_prompt = """你是一名专业的排球体能教练。请根据以下书籍内容，用中文回答用户的问题。

要求：
1. 只基于提供的书籍内容回答，不要编造
2. 如果内容中没有相关信息，请明确说"书中未提及"
3. 回答要专业、清晰、有条理
4. 如果涉及训练建议，请给出具体的动作、组数、次数等细节"""

    user_prompt = f"""
书籍参考内容：
{context}

用户问题：{query}

请用中文回答："""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = llm_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ LLM 调用失败: {e}"

# ==================== 主程序 ====================
if __name__ == "__main__":
    books_display, total_chunks = get_kb_info()
    
    print("=" * 50)
    print("🏀 AI 教练 - Milvus 版本（Phase 2）")
    print(f"📚 当前知识库: {books_display}")
    print(f"📊 向量总数: {total_chunks} 个")
    print(f"🗄️  向量数据库: Milvus（分布式索引）")
    print(f"🤖 大模型: {OLLAMA_MODEL} (本地 Ollama)")
    print("=" * 50)
    print("💡 输入问题，按回车查询")
    print("💡 输入 'exit' 或 'quit' 退出\n")
    
    while True:
        query = input("❓ 你的问题: ").strip()
        
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 再见！")
            break
        
        if not query:
            continue
        
        print("\n🔍 正在从 Milvus 检索...")
        docs, sources = retrieve(query, n_results=3)
        
        if not docs:
            print("❌ 未找到相关内容\n")
            continue
        
        print("\n📖 参考来源:")
        for i, (doc, source) in enumerate(zip(docs, sources), 1):
            print(f"   {i}. {source[:60]}...")
        
        print("\n🤖 正在生成回答...")
        print("-" * 50)
        answer = generate_response(query, docs, sources)
        print(answer)
        print("-" * 50)
        print()