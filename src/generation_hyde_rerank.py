import os
from pymilvus import connections, Collection
from openai import OpenAI
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import chromadb

# ==================== 配置 ====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b"

# ==================== 连接 Milvus ====================
connections.connect(host="localhost", port="19530")
milvus_collection = Collection("ai_coach_milvus")
milvus_collection.load()

# ==================== 初始化 ====================
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

print("📂 正在加载 Rerank 模型 (Cross-Encoder)...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Rerank 模型加载完成")

llm_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# ==================== 获取知识库信息 ====================
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
            books_display = f"{len(sources)} 本书"
        return books_display, total_chunks
    except Exception as e:
        return f"读取失败: {e}", 0

# ==================== HyDE 检索 ====================
def hyde_retrieve(query, n_results=10):
    print(f"   🔍 HyDE: 生成假设答案...")
    
    hyde_prompt = f"""请用一段话（150字以内）回答以下问题。注意：不需要真实引用，只需要假设性的回答，目的是让这段话能代表这个问题的核心主题。

问题：{query}

假设性回答："""
    
    try:
        hyde_response = llm_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": hyde_prompt}],
            temperature=0.5,
            max_tokens=200
        )
        hyde_answer = hyde_response.choices[0].message.content
        print(f"   📝 假设答案: {hyde_answer[:80]}...")
    except Exception as e:
        print(f"   ⚠️ HyDE 失败，使用原始问题: {e}")
        hyde_answer = query
    
    query_embedding = embedding_fn([hyde_answer])
    results = milvus_collection.search(
        data=query_embedding,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=n_results,
        output_fields=["text", "source"]
    )
    
    return results, hyde_answer

# ==================== Rerank 重排 ====================
def rerank_results(query, results, top_k=3):
    if not results or not results[0]:
        return [], [], []
    
    pairs = []
    docs = []
    sources = []
    for hit in results[0]:
        text = hit.entity.get('text', '')
        source = hit.entity.get('source', '未知')
        if text:
            pairs.append([query, text])
            docs.append(text)
            sources.append(source)
    
    if not pairs:
        return [], [], []
    
    print(f"   📊 Rerank: 对 {len(pairs)} 个结果重新打分...")
    scores = reranker.predict(pairs)
    
    scored = list(zip(docs, sources, scores))
    scored.sort(key=lambda x: x[2], reverse=True)
    
    top_docs = [item[0] for item in scored[:top_k]]
    top_sources = [item[1] for item in scored[:top_k]]
    top_scores = [item[2] for item in scored[:top_k]]
    
    return top_docs, top_sources, top_scores

# ==================== 完整检索 ====================
def retrieve_hyde_rerank(query, top_k=3):
    results, hyde_answer = hyde_retrieve(query, n_results=10)
    docs, sources, scores = rerank_results(query, results, top_k=top_k)
    return docs, sources, scores, hyde_answer

# ==================== 生成回答 ====================
def generate_response(query, context_docs, sources):
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
    
    print("=" * 60)
    print("🏐 Phase 3: AI 教练 (HyDE + Rerank)")
    print(f"📚 知识库: {books_display}")
    print(f"📊 向量数: {total_chunks}")
    print(f"🤖 模型: {OLLAMA_MODEL}")
    print("=" * 60)
    print("💡 输入 'exit' 退出\n")
    
    while True:
        query = input("❓ 你的问题: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 再见！")
            break
        if not query:
            continue
        
        print("\n🔍 检索中...")
        docs, sources, scores, hyde_answer = retrieve_hyde_rerank(query, top_k=3)
        
        if not docs:
            print("❌ 未找到相关内容\n")
            continue
        
        print("\n📖 参考来源:")
        for i, (doc, source, score) in enumerate(zip(docs, sources, scores), 1):
            print(f"   {i}. [{source[:40]}] (相关度: {score:.4f})")
        
        print("\n🤖 生成回答中...")
        print("-" * 60)
        answer = generate_response(query, docs, sources)
        print(answer)
        print("-" * 60)
        print()