import os
import time
from pymilvus import connections, Collection
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from openai import OpenAI

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

# 加载 Rerank 模型
print("📂 正在加载 Rerank 模型 (Cross-Encoder)...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Rerank 模型加载完成")

# 初始化 LLM（用于 HyDE 生成假设答案）
llm_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# ==================== HyDE 检索 ====================
def hyde_retrieve(query, n_results=10):
    """
    1. 用 LLM 生成假设答案（HyDE）
    2. 用假设答案检索 Milvus
    3. 返回 Top-K 结果
    """
    print(f"   🔍 HyDE: 生成假设答案...")
    
    # 生成假设答案
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
    
    # 用假设答案检索
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
    """用 Cross-Encoder 对检索结果重新排序"""
    if not results or not results[0]:
        return [], []
    
    # 准备打分数据
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
        return [], []
    
    print(f"   📊 Rerank: 对 {len(pairs)} 个结果重新打分...")
    
    # Cross-Encoder 打分
    scores = reranker.predict(pairs)
    
    # 按分数排序
    scored = list(zip(docs, sources, scores))
    scored.sort(key=lambda x: x[2], reverse=True)
    
    # 取 Top-K
    top_docs = [item[0] for item in scored[:top_k]]
    top_sources = [item[1] for item in scored[:top_k]]
    top_scores = [item[2] for item in scored[:top_k]]
    
    return top_docs, top_sources, top_scores

# ==================== 检索函数（完整流程） ====================
def retrieve_hyde_rerank(query, top_k=3):
    """HyDE + Rerank 完整检索流程"""
    # Step 1: HyDE 检索
    results, hyde_answer = hyde_retrieve(query, n_results=10)
    
    # Step 2: Rerank 重排
    docs, sources, scores = rerank_results(query, results, top_k=top_k)
    
    return docs, sources, scores, hyde_answer

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🏐 Phase 3: Retrieval (HyDE + Rerank)")
    print("=" * 60)
    print("📌 输入问题，查看检索结果对比")
    print("   - Baseline: 直接用问题检索")
    print("   - HyDE: 先用 LLM 生成假设答案再检索")
    print("   - Rerank: 用 Cross-Encoder 重新排序")
    print("=" * 60)
    print()
    
    test_queries = [
        "排球运动员的力量训练",
        "如何提高弹跳力",
        "营养补充建议",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"📝 问题: {query}")
        print(f"{'='*60}")
        
        # Phase 2 Baseline: 直接检索
        print("\n📌 Phase 2 (Baseline) - 直接检索:")
        query_embedding = embedding_fn([query])
        baseline_results = milvus_collection.search(
            data=query_embedding,
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=3,
            output_fields=["text", "source"]
        )
        for i, hit in enumerate(baseline_results[0]):
            source = hit.entity.get('source', '未知')
            text = hit.entity.get('text', '')[:120]
            print(f"   {i+1}. [{source[:30]}] {text}...")
        
        # Phase 3: HyDE + Rerank
        print("\n🚀 Phase 3 (HyDE + Rerank):")
        docs, sources, scores, hyde_answer = retrieve_hyde_rerank(query, top_k=3)
        
        print(f"   📝 HyDE 假设答案: {hyde_answer[:100]}...")
        print(f"\n   📊 Rerank 结果:")
        for i, (doc, source, score) in enumerate(zip(docs, sources, scores)):
            print(f"   {i+1}. [{source[:30]}] (分数: {score:.4f})")
            print(f"      {doc[:120]}...")
        
        print("\n" + "-" * 60)